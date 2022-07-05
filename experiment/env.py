import sys
from collections import namedtuple

import numpy as np

from vehicle import Vehicle
from random import randint
from mec import MEC

Experience = namedtuple('Transition',
                        field_names=['state', 'action', 'reward', 'next_state'])  # Define a transition tuple

MAX_TASK = 5  # 只能选前五个任务
y = [2, 6, 10, 14]  # 车子y的坐标集 # 共四条车道
direction = [1, 1, -1, -1]  # 车子的方向
MEC_loc = [[-800, 0], [-400, 16], [0, 0], [400, 16], [800, 0]]  # mec的位置

Fv = 1  # 车的计算能力
N = 20  # 车的数量
K = 5  # mec的数量
MAX_NEIGHBOR = 5  # 最大邻居数
CAPACITY = 20000  # 缓冲池大小

sigma = -114  # 噪声dbm
POWER = 23  # 功率w dbm
BrandWidth_Mec = 100  # MHz

gama = 1.25 * (10 ** -6)  # 能量系数
a = 0.6  # 奖励中时间占比
b = 0.4  # 奖励中能量占比

Ki = -4  # 非法动惩罚项
Kq = 0.0025  # 任务队列长度系数
Ks = 0.5  # 奖励占比

np.random.seed(2)


class Env:
    def __init__(self, num_Vehicles=N, num_MECs=K):
        # 基站天线高度
        self.bsHeight = 25
        # 车辆天线高度
        self.vehHeight = 1.5
        self.stdV2I = 8
        self.stdV2V = 3
        self.freq = 2
        self.vehAntGain = 3
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehNoiseFigure = 9
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
        # 记录每辆车的历史奖励
        self.vehicleReward = []
        # 当前奖励
        self.reward = [0] * self.num_Vehicles
        # 当前时间
        self.cur_frame = 0
        # 所有车的卸载动作
        self.offloadingActions = [0] * num_Vehicles
        # 所有车的选择任务动作
        self.taskActions = [0] * num_Vehicles
        # 所有车要传输的对象 vehicle or mec
        self.aims = []
        # 当前全局的状态信息  维度:
        self.otherState = []
        # 全局任务队列信息
        self.taskState = []
        # 描述每辆车的状态
        self.vehicles_state = []
        # 系统的缓冲池
        self.buffer = []

    # 添加车辆
    def add_new_vehicles(self, id, loc_x, loc_y, direction, velocity):
        vehicle = Vehicle(id=id, loc_x=loc_x, loc_y=loc_y, direction=direction, velocity=velocity)
        vehicle.create_work()  # 初始化任务
        self.vehicles.append(vehicle)

    # 初始化/重置环境
    def reset(self):
        self.Reward = 0
        self.otherState = []
        self.cur_frame = 0
        self.offloadingActions = [0] * self.num_Vehicles
        self.taskActions = [0] * self.num_Vehicles
        self.reward = [0] * self.num_Vehicles

        for i in range(self.num_Vehicles):
            self.vehicleReward.append([])

        for i in range(0, self.num_MECs):  # 初始化mec
            cur_mec = MEC(id=i, loc_x=MEC_loc[i][0], loc_y=MEC_loc[i][1])
            self.MECs.append(cur_mec)

        i = 0
        while i < self.num_Vehicles:  # 初始化车子
            n = np.random.randint(0, 4)  # 左闭右开
            self.add_new_vehicles(id=i, loc_x=randint(-1000, 1000), loc_y=y[n], direction=direction[n],
                                  velocity=np.random.randint(17, 35))
            i += 1
        # 初始化邻居信息
        self.renew_neighbor()
        self.renew_neighbor_mec()

        # 初始化状态信息
        for vehicle in self.vehicles:
            # 全局状态
            self.otherState.extend(vehicle.get_state())
            # 描述车的状态
            self.vehicles_state.append(vehicle.otherState)
            # 全局任务状态
            self.taskState.append(vehicle.taskState)
        for mec in self.MECs:
            self.otherState.extend(mec.state)

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
            vehicle.mec_lest = self.MECs[distance.index(min(distance))]

    # 获得传输对象
    def get_aim(self, vehicle: Vehicle, action):
        if action == 0:
            return vehicle
        elif action == 1:
            return vehicle.mec_lest
        else:
            return vehicle.neighbor[action - 2]

    def process_taskActions(self):
        for i, vehicle in enumerate(self.vehicles):
            # 没有任务，不能执行任何动作,也不给惩罚
            if vehicle.len_task <= 0:
                vehicle.cur_task = None
                continue

            action = self.taskActions[i]
            # 获得要传输的任务
            # 大于当前任务长度
            if action >= vehicle.len_task:
                # 非法动作 给予惩罚项
                self.reward[vehicle.id] += Ki
                continue
            # 大于可选范围（默认选择第一个）
            elif action > MAX_TASK:
                task = vehicle.total_task[0]
            else:
                task = vehicle.total_task[action]

            # 初始化任务信息
            aim = self.get_aim(vehicle, self.offloadingActions[i])
            task.aim = aim

            # 卸载给本地 直接放到任务队列中
            if vehicle == aim:
                task.option = self.cur_frame
                vehicle.accept_task.append(task)
                vehicle.total_task.remove(task)
                vehicle.cur_task = task
                vehicle.len_task -= 1
                continue

            # 需要传输 卸载给远程
            # 有任务在传输
            if vehicle.task is not None:
                vehicle.cur_task = None
                # 非法动作 给予惩罚项
                self.reward[vehicle.id] += Ki
            else:
                task.option = self.cur_frame
                task.aim = aim
                aim.len_action += 1
                vehicle.len_task -= 1
                vehicle.cur_task = task
                vehicle.task = task
                vehicle.total_task.remove(task)
                self.need_trans_task.append(task)

    # 计算距离(车到车或者车到MEC)  aim：接受任务的目标
    def compute_distance(self, taskVehicle: Vehicle, aim):
        return round(np.sqrt(np.abs(taskVehicle.get_x - aim.get_x) ** 2 + np.abs(taskVehicle.get_y - aim.get_y) ** 2),
                     2)

    def generate_fading_V2I(self, dist_veh2bs):
        dist2 = (self.vehHeight - self.bsHeight) ** 2 + dist_veh2bs ** 2
        pathloss = 128.1 + 37.6 * np.log10(np.sqrt(dist2) / 1000)  # 路损公式中距离使用km计算
        combinedPL = -(np.random.randn() * self.stdV2I + pathloss)
        return combinedPL + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure

    def generate_fading_V2V(self, dist_DuePair):
        pathloss = 32.4 + 20 * np.log10(dist_DuePair) + 20 * np.log10(self.freq)
        combinedPL = -(np.random.randn() * self.stdV2V + pathloss)
        return combinedPL + self.vehAntGain * 2 - self.vehNoiseFigure

    # 计算实时传输速率（在一个时隙内假设不变）
    def compute_rate(self, vehicle: Vehicle, aim):
        # print("vehicle:{} aim:{} ".format(vehicle.id, aim.id))
        if aim == vehicle:  # 本地计算
            return 0

        distance = self.compute_distance(vehicle, aim)
        if type(aim) == MEC:
            fade = self.generate_fading_V2I(distance)
        else:
            fade = self.generate_fading_V2V(distance)
        power = np.power(10, (POWER + fade) / 10)
        sigma_w = np.power(10, sigma / 10)
        sign = power / sigma_w
        SNR = (BrandWidth_Mec / self.num_Vehicles) * np.log2(1 + sign)
        print("第{}辆车速率:".format(vehicle.id), SNR / 8)
        return SNR / 8

    # 计算两物体持续时间
    def compute_persist(self, vehicle: Vehicle, aim):
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

    # 计算这个任务传输所需要消耗的能量
    def compute_energy(self, trans_time):
        return np.power(10, POWER / 10) * trans_time

    # 根据动作将任务添加至对应的列表当中  分配任务
    def distribute_task(self, cur_frame):
        need_task = []
        time = cur_frame - self.cur_frame
        # 远程卸载
        for task in self.need_trans_task:
            aim = task.aim
            vehicle = task.vehicle
            distance = self.compute_distance(vehicle, aim)
            # 超出传输范围
            if distance > aim.range:
                self.reward[vehicle.id] += Ki
                continue
            # 传输时间加上一时隙
            task.trans_time += time
            # 计算实时速率
            rate = self.compute_rate(vehicle, aim)
            # 能够传输完
            if task.need_trans_size <= time * rate:
                # 表示当前任务能够卸载完成 可以继续卸载
                self.vehicles[vehicle.id].task = None
                print("第{}车任务传输完成".format(vehicle.id))
                # mec
                if type(aim) == MEC:
                    self.vehicles[vehicle.id].mec_lest.accept_task.append(task)
                    aim.len_action -= 1
                # neighbor vehicle
                else:
                    aim.accept_task.append(task)
                    aim.len_action -= 1
            else:
                task.need_trans_size -= time * rate
                need_task.append(task)
        print("forward", self.need_trans_task)
        self.need_trans_task = need_task
        print("after", self.need_trans_task)
        # 更新车和mec的需要处理的任务数量
        for vehicle in self.vehicles:
            vehicle.sum_needDeal_task = len(vehicle.accept_task)
        for mec in self.MECs:
            mec.sum_needDeal_task = len(mec.accept_task)

    # 获得奖励
    def get_reward(self, task):  # 获得奖励
        reward = 0
        aim = task.aim
        vehicle = task.vehicle
        # print("第{}辆车的任务的处理时间:{}s".format(vehicle.id, time_precess))
        sum_time = self.cur_frame - task.create_time
        # 总时间大于最大忍受时间
        if sum_time > task.max_time:
            reward += Ki
        # print("第{}辆车的任务总时间时间:{}s".format(vehicle.id, sum_time))
        else:
            energy = self.compute_energy(task.trans_time) + task.energy
            # print("第{}辆车完成任务消耗的能量:{}".format(vehicle.id, energy))
            reward += -0.5 * (a * sum_time + b * energy+task.vehicle.overflow) - Kq * vehicle.len_task
            # print("第{}辆车的任务奖励{}".format(vehicle.id, reward))
        return reward

    # 更新资源信息并为处理完的任务计算奖励
    def renew_resources(self, cur_frame):
        time = cur_frame - self.cur_frame
        # 更新车的资源信息
        for i, vehicle in enumerate(self.vehicles):
            total_task = vehicle.accept_task
            size = len(total_task)

            if size > 0 and vehicle.resources > 0:  # 此时有任务并且有剩余资源
                f = vehicle.resources / size
                # 记录这个时隙能够处理完的任务
                removed_task = []
                for task in total_task:  # 遍历此车的所有任务列表
                    precessed_time = task.need_precess_cycle / f
                    # 处理时间+一时隙
                    task.precess_time += time
                    # 加上这时隙所消耗能量
                    task.energy = gama * np.power(f, 3) * time
                    if precessed_time > time:  # 不能处理完
                        task.need_precess_cycle -= f * time
                    else:
                        removed_task.append(task)
                # 删除已经处理完的任务并为其计算奖励
                for task in removed_task:
                    self.reward[task.vehicle.id] += self.get_reward(task)
                    total_task.remove(task)

        # 更新mec的资源信息
        for i, mec in enumerate(self.MECs):
            total_task = mec.accept_task
            size = len(total_task)
            if size > 0 and mec.resources > 0:
                f = mec.resources / size
                # print("mec {} give {} rescource".format(i, f))
                removed_task = []
                for task in mec.accept_task:
                    precessed_time = task.need_precess_cycle / f
                    task.precess_time += time
                    if precessed_time > time:
                        task.need_precess_cycle -= f * time
                    else:
                        removed_task.append(task)
                # 删除已经处理完的任务
                for task in removed_task:
                    # 计算给任务的奖励
                    self.reward[task.vehicle.id] += self.get_reward(task)
                    total_task.remove(task)

        # 分配任务信息
        self.distribute_task(cur_frame=cur_frame)

    # 更新每辆车的位置
    def renew_locs(self, cur_frame):
        time = cur_frame - self.cur_frame
        for vehicle in self.vehicles:
            loc_x = round(vehicle.get_x + vehicle.direction * vehicle.velocity * time, 2)
            if loc_x > 1000:
                vehicle.set_location(-1000, vehicle.get_y)
            elif loc_x < -1000:
                vehicle.set_location(1000, vehicle.get_y)
            else:
                vehicle.set_location(loc_x, vehicle.get_y)

    # 更新状态
    def renew_state(self, cur_frame):
        self.otherState = []
        self.taskState = []
        self.vehicles_state = []

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
            vehicle.create_work()
            # 更新资源已经接受的任务信息
            self.otherState.extend(vehicle.get_state())
            self.taskState.append(vehicle.taskState)
            self.vehicles_state.append(vehicle.otherState)
            # 更新时间
            vehicle.cur_frame = cur_frame
            # 更新任务信息
            vehicle.cur_task = None
        for mec in self.MECs:
            mec.cur_frame = cur_frame
            self.otherState.extend(mec.get_state())

    # 执行动作
    def step(self, actions):
        cur_frame = self.cur_frame + 0.1  # s
        # 分配动作
        lenAction = int(len(actions) / 2)
        self.offloadingActions = actions[0:lenAction]
        self.taskActions = actions[lenAction:]

        # 重置奖励
        self.reward = [0] * self.num_Vehicles

        # 处理选取任务动作
        self.process_taskActions()

        other_state = self.otherState
        task_state = self.taskState

        self.renew_resources(cur_frame)

        self.renew_state(cur_frame)  # 更新状态

        # 更新时间
        self.cur_frame = cur_frame
        print("当前有{}个任务没完成".format(len(self.need_trans_task)))

        return other_state, task_state, self.vehicles_state, self.otherState, self.taskState,self.Reward, self.reward
