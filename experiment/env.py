import sys
from collections import namedtuple

import numpy as np
from vehicle import Vehicle
from random import random, randint
from mec import MEC

Experience = namedtuple('Transition',
                        field_names=['state', 'action', 'reward', 'next_state'])  # Define a transition tuple

y = [2, 6, 10, 14]  # 车子y的坐标集 # 共四条车道
direction = [1, 1, -1, -1]  # 车子的方向
# velocity = 5  # 车子速度
MEC_loc = [[-300, 0], [0, 16], [300, 0]]  # mec的位置

Fv = 1  # 车的计算能力
N = 20  # 车的数量
K = 3  # mec的数量
MAX_NEIGHBOR = 5  # 最大邻居数

sigma = -114  # 噪声dbm
POWER = 1  # 功率w dbm
BrandWidth_Vehicle = 2  # 带宽MHz
BrandWidth_Mec = 10  # MHz
alpha = 1.75  # 信道增益
gama = 12.5  # 能量系数
a = 0.6  # 奖励中时间占比
b = 0.4  # 奖励中能量占比

eps_threshold = 0.9
min_eps = 0.1

np.random.seed(0)


class Env:
    def __init__(self, num_Vehicles=N, num_MECs=K):
        # 环境内所有车辆
        self.vehicles = []
        # 环境内所有mec
        self.MECs = []
        # 车辆数以及mec数
        self.num_Vehicles = num_Vehicles
        self.num_MECs = num_MECs
        # 所有需要传输的任务
        self.need_trans_task = []
        # 当前平均奖励奖励数
        self.Reward = 0
        # 记录每辆车的奖励
        self.reward = []
        # 当前时间
        self.cur_frame = 0
        # 所有车的动作
        self.actions = [0] * num_Vehicles
        # 所有车要传输的对象 vehicle or mec
        self.aims = []
        # 当前全局的状态信息  维度:
        self.state = []

    # 添加车辆
    def add_new_vehicles(self, id, loc_x, loc_y, direction, velocity):
        vehicle = Vehicle(id=id, loc_x=loc_x, loc_y=loc_y, direction=direction, velocity=velocity)
        vehicle.creat_work()  # 初始化任务
        self.vehicles.append(vehicle)

    # 初始化网络
    # def init_network(self):
    #     # 初始化全局critic网络  获得当前价值函数
    #     self.crt_net = model.ModelCritic(STATES_CRITIC)
    #     self.tgt_ctr_net = ptan.agent.TargetNet(self.crt_net)
    #     # 双q网络获得状态动作函数
    #     self.twing_net = model.ModelSACTwinQ(STATES_CRITIC, ACTIONS)
    #     # 初始化每辆车的actor网络
    #     for vehicle in self.vehicles:
    #         vehicle.init_network(STATES_ACTOR, ACTIONS)

    # 初始化/重置环境
    def reset(self):
        self.Reward = 0
        self.state = []
        self.cur_frame = 0
        self.actions = [0] * self.num_Vehicles

        for i in range(self.num_Vehicles):
            self.reward.append([])

        for i in range(0, self.num_MECs):  # 初始化mec
            cur_mec = MEC(id=i, loc_x=MEC_loc[i][0], loc_y=MEC_loc[i][1])
            self.MECs.append(cur_mec)

        i = 0
        while i < self.num_Vehicles:  # 初始化车子
            n = np.random.randint(0, 4)  # 左闭右开
            self.add_new_vehicles(id=i, loc_x=randint(-500, 500), loc_y=y[n], direction=direction[n],
                                  velocity=randint(5, 10))
            i += 1
        # 初始化邻居信息
        self.renew_neighbor()
        self.renew_neighbor_mec()

        # 初始化状态信息
        for vehicle in self.vehicles:
            self.state.extend(vehicle.get_state())
        for mec in self.MECs:
            self.state.extend(mec.state)

    # 更新每辆车邻居表
    def renew_neighbor(self):
        for i in range(len(self.vehicles)):
            self.vehicles[i].neighbor = []
        z = np.array([[complex(c.get_x, c.get_y) for c in self.vehicles]])
        Distance = abs(z.T - z)

        for i in range(len(self.vehicles)):
            sort_idx = np.argsort(Distance[:, i])
            for j in range(MAX_NEIGHBOR):
                self.vehicles[i].neighbor.append(self.vehicles[sort_idx[j + 1]])

    # 更新每辆车最近的mec
    def renew_neighbor_mec(self):
        for vehicle in self.vehicles:
            distance = []
            for mec in self.MECs:
                distance.append(self.compute_distance(vehicle, mec))
            # print("vehicle{}".format(vehicle.id), distance)
            vehicle.mec_lest = self.MECs[distance.index(min(distance))]

    # 获得传输对象
    def get_aim(self, vehicle: Vehicle, action):
        if action == 0:
            return vehicle
        elif action == 1:
            return vehicle.mec_lest
        else:
            return vehicle.neighbor[action - 2]

    # 计算距离(车到车或者车到MEC)  aim：接受任务的目标
    def compute_distance(self, taskVehicle: Vehicle, aim):
        return round(np.sqrt(np.abs(taskVehicle.get_x - aim.get_x) ** 2 + np.abs(taskVehicle.get_y - aim.get_y) ** 2),
                     2)

    # 计算实时传输速率（在一个时隙内假设不变）
    def compute_rate(self, vehicle: Vehicle, aim):
        print("vehicle:{} aim:{} ".format(vehicle.id, aim.id))
        if aim == vehicle:  # 本地计算
            return 0
        if type(aim) == MEC:
            BrandWidth = BrandWidth_Mec
        else:
            BrandWidth = BrandWidth_Vehicle

        distance = self.compute_distance(vehicle, aim)
        # power_w = np.power(10, POWER / 10) / 1000  # w
        # sigma_w = np.power(10, sigma / 10)  # w
        sign = (POWER * np.power(distance / 1000, -alpha)) / aim.len_action
        SNR = (BrandWidth / aim.len_action) * np.log2(1 + sign)
        # print("nrR", SNR)
        return SNR / 8

    # 计算一个任务传输时间
    def compute_transmit(self, vehicle: Vehicle, aim):
        SNR = self.compute_rate(vehicle, aim)
        if SNR == 0:
            return 0
        return vehicle.task.size / SNR  # s

    # 初始化任务信息
    def task_info(self):
        self.compute_len_action()
        # 将任务放置待传队列中
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.task is not None:
                self.vehicles[i].task.id = vehicle.id
                self.vehicles[i].task.aim = self.get_aim(vehicle, self.actions[i])
                self.need_trans_task.append(self.vehicles[i].task)

        # 为每一个待传任务计算实时速率
        for task in self.need_trans_task:
            rate = self.compute_rate(self.vehicles[task.id], task.aim)
            task.rate = rate
            print("第{}辆车task rate：{}".format(task.id, rate))

    # 为每一个载体计算此时有多少个任务选择我
    def compute_len_action(self):
        for i, action in enumerate(self.actions):
            if self.vehicles[i].task is not None:
                if action == 0:
                    continue
                elif action == 1:
                    self.vehicles[i].mec_lest.len_action += 1
                else:
                    # print(len(self.vehicles[i].neighbor))
                    self.vehicles[i].neighbor[action - 2].len_action += 1

    # 计算本辆车这个任务所需的处理时间(实时，真实情况不是这样，只是计算当前)
    def compute_precessed(self, vehicle: Vehicle, action):
        aim = self.get_aim(vehicle, action)
        total_task = len(aim.accept_task) + aim.len_action  # 目标车辆或mec接收到的总任务
        # print(aim.len_action)
        if total_task == 0:  # 卸载到本地时会出现的现象
            total_task = 1
        f = aim.resources * np.power(10, 6) / total_task  # 平均分配    # Hz
        # print("f:", f, "GHZ")
        time = round(vehicle.task.need_precess_cycle / f, 2) / 1000  # s
        return time, f / np.power(10, 6)  # s

    # 计算两物体持续时间
    def compute_persist(self, vehicle: Vehicle, action):
        aim = self.get_aim(vehicle, action)
        distance = self.compute_distance(vehicle, aim)
        # print("aim:{} vehicle:{}".format(aim.id, vehicle.id))
        if distance > aim.range:
            return 0

        if type(aim) is Vehicle:  # 如果对象是车
            if vehicle.velocity == aim.velocity and vehicle.direction == aim.direction:
                return sys.maxsize  # return np.abs(vehicle.direction * 500 - np.max(np.abs(vehicle.get_x),
                # np.abs(aim.get_x))) / vehicle.velocity
            else:
                return (np.sqrt(vehicle.range ** 2 - (aim.get_y - vehicle.get_y) ** 2) + aim.get_x - vehicle.get_x) / \
                       np.abs(vehicle.velocity * vehicle.direction - aim.velocity * aim.direction)

        else:  # 对象是MEC
            return (np.sqrt(aim.range ** 2 - (aim.get_y - vehicle.get_y) ** 2) + aim.get_x - vehicle.get_x) / np.abs(
                vehicle.velocity * vehicle.direction)

    # 计算这个任务所需要消耗的能量
    def compute_energy(self, vehicle, trans_time, precess_time, f):
        task = vehicle.task
        if task.size is not None:
            if type(task.aim) != MEC:
                return gama * f * precess_time + POWER * trans_time
            else:
                return POWER * trans_time
        else:
            return 0

    # 根据动作将任务添加至对应的列表当中  分配任务
    def distribute_task(self, cur_frame):
        need_task = []
        time = cur_frame - self.cur_frame
        for task in self.need_trans_task:
            aim = task.aim
            # 卸载至本地
            if aim.id == task.id:
                task.arrive = cur_frame  # 可直接计算
                self.vehicles[task.id].accept_task.append(task)
                continue
            # 远程卸载
            # 能够传输完
            if task.need_trans_size <= time * task.rate:
                task.arrive = cur_frame + 1  # 下一个时隙才开始计算
                # mec
                if type(aim) == MEC:
                    self.vehicles[task.id].mec_lest.accept_task.append(task)
                    self.vehicles[task.id].mec_lest.len_action -= 1
                # neighbor vehicle
                else:
                    # vehicle_id = self.vehicles[task.id].neighbor[action - 2].id
                    # self.vehicles[vehicle_id].accept_task.append(task)
                    # self.vehicles[vehicle_id].len_action -= 1
                    aim.accept_task.append(task)
                    aim.len_action -= 1
            else:
                task.need_trans_size -= time * task.rate
                need_task.append(task)
        print("forward", self.need_trans_task)
        self.need_trans_task = need_task
        print("after", self.need_trans_task)

    # 获得平均奖励
    def get_averageReward(self):  # 获得奖励
        self.Reward = 0
        sum = 0
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.task is None:
                self.reward[i].append(0)
                continue
            else:
                sum += 1
                aim = self.get_aim(vehicle, self.actions[i])
                time_trans = self.compute_transmit(vehicle=vehicle, aim=aim)
                time_precess, f = self.compute_precessed(vehicle=vehicle, action=self.actions[i])
                print("第{}辆车的任务的处理时间:{}s".format(vehicle.id, time_precess))
                sum_time = time_trans + time_precess
                print("第{}辆车的任务总时间时间:{}s".format(vehicle.id, sum_time))
                time_persist = self.compute_persist(vehicle, self.actions[i])
                print("第{}辆车的任务持续时间：{}s".format(vehicle.id, time_persist))
                if sum_time > time_persist or sum_time > vehicle.task.max_time:
                    # vehicle.reward.append(-1)
                    self.reward[i].append(-1)
                    self.Reward += -1
                    print("第{}辆车的任务奖励{}".format(vehicle.id, -1))
                else:
                    energy = self.compute_energy(vehicle, time_trans, time_precess, f)
                    print("第{}辆车完成任务消耗的能量:{}".format(vehicle.id, energy))
                    reward = round(2 / 5 * (sum_time * a + b * energy), 2)
                    self.reward[i].append(reward)
                    print("第{}辆车的任务奖励{}".format(vehicle.id, reward))
                    self.Reward += reward
        return self.Reward / sum

    # 更新资源信息
    def renew_resources(self, cur_frame):
        time = cur_frame - self.cur_frame
        # 更新车的资源信息
        for i, vehicle in enumerate(self.vehicles):
            total_task = vehicle.accept_task
            # 去掉这个时隙内刚到的任务，这个时隙不对他进行运算
            for task in reversed(total_task):
                if task.arrive > cur_frame:
                    total_task.remove(task)

            if len(total_task) > 0 and vehicle.resources > 0:  # 此时有任务并且有剩余资源
                f = vehicle.resources / len(total_task)
                after_task = []
                for task in total_task:  # 遍历此车的所有任务列表
                    precessed_time = task.need_precess_cycle / (f * 1000)
                    if precessed_time > time:  # 不能处理完
                        task.need_precess_cycle -= f * 1000 * time * task.size
                        after_task.append(task)
                vehicle.accept_task = after_task  # 更新任务列表
                vehicle.sum_needDeal_task = len(after_task)
                # 更新能提供的资源（可以省略）
                # vehicle.resources = round((1 - np.random.uniform(0, 0.7)) * Fv, 2)  # GHz

        # 更新mec的资源信息
        for i, mec in enumerate(self.MECs):
            total_task = mec.accept_task
            for task in reversed(total_task):
                if task.arrive > cur_frame:
                    total_task.remove(task)

            if len(total_task) > 0 and mec.resources > 0:
                f = mec.resources / len(total_task)
                # print("mec {} give {} rescource".format(i, f))
                mec.resources = 0
                after_task = []
                for task in mec.accept_task:
                    precessed_time = task.need_precess_cycle / f
                    if precessed_time > time:
                        task.need_precess_cycle -= f * time * task.size
                        after_task.append(task)
                mec.accept_task = after_task
                mec.sum_needDeal_task = len(after_task)

    # 更新每辆车的位置
    def renew_locs(self, cur_frame):
        time = cur_frame - self.cur_frame
        for vehicle in self.vehicles:
            loc_x = vehicle.get_x + vehicle.direction * vehicle.velocity * time
            if loc_x > 500:
                vehicle.set_location(-500, vehicle.get_y)
            elif loc_x < -500:
                vehicle.set_location(500, vehicle.get_y)
            else:
                vehicle.set_location(loc_x, vehicle.get_y)

    # 更新状态
    def renew_state(self, cur_frame):
        self.state = []

        # 更新车位置信息
        self.renew_locs(cur_frame)
        # 更新车和mec的资源及任务列表信息
        self.renew_resources(cur_frame)
        # 更新邻居表
        self.renew_neighbor()
        # 更新最近mec
        self.renew_neighbor_mec()
        for vehicle in self.vehicles:
            # 产生任务
            vehicle.creat_work()
            # 更新资源已经接受的任务信息
            self.state.extend(vehicle.get_state())
        for mec in self.MECs:
            self.state.extend(mec.get_state())

        # 更新时间
        self.cur_frame = cur_frame

    # 执行动作
    def step(self, actions):
        cur_frame = self.cur_frame + 1  # s
        state = self.state
        self.actions = actions
        print(self.actions)
        self.task_info()  # 初始化任务信息
        self.Reward = self.get_averageReward()  # 获得当前奖励
        print("时间" + str(cur_frame) + "获得的奖励:" + str(self.Reward))
        self.distribute_task(cur_frame)  # 分配任务
        self.renew_state(cur_frame)  # 更新状态
        self.cur_frame = cur_frame
        return state, self.actions, self.Reward, self.state
