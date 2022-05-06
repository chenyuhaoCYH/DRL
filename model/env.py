import random
import numpy as np
import torch
from vehicle import Vehicle
from random import random, randint
from mec import MEC

y = [2, 6, 10, 14]  # 车子y的坐标集 # 共四条车道
direction = [1, 1, -1, -1]  # 车子的方向

Fv = 20  # 车的计算能力
N = 40  # 车的数量
K = 5  # mec的数量
ACTIONS = N + K + 1  # 动作空间维度
STATES = 9 * N + 4 * K  # 状态空间维度

sigma = -114  # 噪声dbm
POWER = 23  # 功率dbm
BrandWidth = 100  # 带宽MHz
alpha = 0.25  # 信道增益

eps_threshold = 0.9
min_eps = 0.1


class Env:
    def __init__(self, num_Vehicles=N, num_MECs=K):
        # 环境内所有车辆
        self.vehicles = []
        # 环境内所有mec
        self.MECs = []
        # 车辆数以及mec数
        self.num_Vehicles = num_Vehicles
        self.num_MECs = num_MECs
        # 总奖励数
        self.totalReward = 0
        # 当前时间
        self.cur_frame = 0
        # 所有车的动作
        self.actions = [0] * (num_Vehicles + num_MECs + 1)
        # 所有车要传输的对象 vehicle or mec
        self.aim = []
        # 当前所有的状态信息  维度： 9*num_Vehicles+4*num_MECs
        self.state = []

    # 添加车辆
    def add_new_vehicles(self, id, loc_x, loc_y, direction, velocity):
        vehicle = Vehicle(id=id, loc_x=loc_x, loc_y=loc_y, direction=direction, velocity=velocity)
        vehicle.creat_work()  # 初始化任务
        vehicle.init_network(STATES, ACTIONS)  # 初始化网络
        self.vehicles.append(vehicle)
        self.state.extend(vehicle.get_state())

    # 初始化/重置环境
    def reset(self):
        self.totalReward = 0
        self.state = []
        self.cur_frame = 0
        self.actions = [0] * (self.num_Vehicles + self.num_MECs + 1)
        i = 0
        while i < self.num_Vehicles:  # 初始化车子
            n = randint(0, 3)
            self.add_new_vehicles(id=i, loc_x=randint(-500, 500), loc_y=y[n], direction=direction[n], velocity=5)
            i += 1

        for i in range(0, self.num_MECs):
            cur_mec = MEC(id=i, loc_x=randint(-10, 10), loc_y=randint(-10, 10))
            self.MECs.append(cur_mec)
            self.state.extend(cur_mec.get_state())

    # 获得所有的动作
    def get_action(self):
        global eps_threshold
        self.actions = []
        # 逐个获取每个vehicle的动作
        for i, vehicle in enumerate(self.vehicles):
            sample = random()
            if sample > eps_threshold:  # epsilon-greeedy policy
                # 不计算梯度 防止出现噪声 因为此时只利用
                with torch.no_grad():
                    state = torch.tensor([self.state])
                    Q_value = vehicle.cur_network(state)  # Get the Q_value from DNN
                    action = Q_value.max(1)[1].view(1)  # 获得最大值的那一个下标为要采取的动作   (二维取列最大的下标值(一维)-》二维)
            else:
                action = torch.tensor([randint(0, self.num_MECs + self.num_Vehicles)],
                                      dtype=torch.int)  # randrange(1 + N + K)]
                eps_threshold = max(min_eps, eps_threshold - 0.01)   # 减小eps
            action = action.item()  # torch类型
            self.actions.append(action)

    # 获得要传输的对象
    def get_aim(self):
        self.aim = []
        for vehicle, action in zip(self.vehicles, self.actions):  # 遍历所有车
            if action == 0:  # 选择自己
                self.aim.append(vehicle)
            elif action <= self.num_MECs:  # 选择mec
                self.aim.append(self.MECs[action - 1])
            else:  # 选择其他车辆
                self.aim.append(self.vehicles[action - self.num_MECs - 1])

    # 计算距离(车到车或者车到MEC)  aim：接受任务的目标
    def compute_distance(self, taskVehicle: Vehicle, aim):
        return round(np.sqrt(np.abs(taskVehicle.get_x - aim.get_x) ** 2 + np.abs(taskVehicle.get_y - aim.get_y) ** 2),
                     2)

    # 计算一个任务传输时间
    def compute_transmit(self, taskVehicle: Vehicle, aim):
        if aim == taskVehicle:
            return 0
        sign = 10 ** (POWER * (self.compute_distance(taskVehicle, aim) ** (-alpha)) / sigma / 10)  # dB转w
        SNR = BrandWidth / N * np.log2(1 + sign)  # Mbit/s
        # print("SNR:", SNR, "Mbit/s")
        return round(taskVehicle.task[0] / SNR, 2)  # 单位s   Mb/(Mbit/s)

    # 根据动作将任务添加至对应的列表当中  分配任务
    def distribute_task(self):
        for i, action in enumerate(self.actions):
            task = self.vehicles[i].task
            if task[0] == 0:
                continue  # 没有任务
            if action == 0:  # 卸载至本地
                self.vehicles[i].recevied_task.append(task)
            elif action <= self.num_MECs:  # 卸载给MEC
                self.MECs[action - 1].recevied_task.append(task)
            else:  # 卸载给其他车辆
                self.vehicles[action - self.num_MECs - 1].recevied_task.append(task)

    # 计算本辆车这个任务所需的处理时间
    def compute_precessed(self, vehicle: Vehicle, aim):
        total_task = aim.recevied_task  # 目标车辆或mec接收到的总任务
        f = aim.resources / len(total_task)  # 平均分配    # GHz
        # print("f:", f, "GHZ")
        time = round(vehicle.task[1] / f, 2)  # s
        return time

    # 计算两物体持续时间  （还未完成）
    def compute_persist(self, vehicle: Vehicle, aim):
        if type(aim) is Vehicle:  # 如果对象是车
            pass
        else:  # 对象是MEC
            pass

    # 获得平均奖励
    def get_averageReward(self):  # 获得平均奖励
        reward = 0
        sum = 0
        for i, vehicle in enumerate(self.vehicles):
            # 只考虑有任务的车辆
            if vehicle.task[0] == 0:
                continue
            sum += 1
            aim = self.aim[i]
            trans_time = self.compute_transmit(taskVehicle=vehicle, aim=aim)
            precessed_time = self.compute_precessed(vehicle, aim=aim)
            total_time = trans_time + precessed_time
            if total_time > vehicle.task[2]:  # 超过最大忍耐时间
                cur_reward = -1
            else:
                cur_reward = vehicle.task[2] - total_time
            reward += cur_reward
            # print("vehicle{} get {} reward".format(i, cur_reward))
        if sum > 0:
            reward /= sum
        return round(reward, 5)

    # 更新资源信息
    def renew_resources(self, cur_frame):
        # 更新车的资源信息
        for i, vehicle in enumerate(self.vehicles):
            time = cur_frame - self.cur_frame
            total_task = vehicle.recevied_task
            if len(total_task) > 0 and vehicle.resources > 0:  # 此时有任务并且有剩余资源
                f = vehicle.resources / len(total_task)
                # vehicle.resources = 0
                after_task = []
                for task in total_task:  # 遍历此车的所有任务列表
                    precessed_time = task[1] / f
                    if precessed_time > time:  # 不能处理完
                        task[1] -= f * time
                        after_task.append(task)
                    # else:  # 能够完成
                    #     vehicle.resources += f  # 更新资源
                vehicle.recevied_task = after_task  # 更新任务列表

        # 更新mec的资源信息
        for i, mec in enumerate(self.MECs):
            total_task = mec.recevied_task
            if len(total_task) > 0 and mec.resources > 0:
                f = mec.resources / len(total_task)
                mec.resources = 0
                after_task = []
                for task in mec.recevied_task:
                    precessed_time = task[1] / f
                    if precessed_time > time:
                        task[1] -= f * time
                        after_task.append(task)
                    else:
                        mec.resources += f
                mec.recevied_task = after_task

    # 更新状态
    def renew_state(self, cur_frame):
        self.state = []
        # 更新车位置信息
        self.renew_locs(cur_frame)
        # 更新车和mec的资源及任务列表信息
        self.renew_resources(cur_frame)
        for vehicle in self.vehicles:
            # 产生任务
            vehicle.creat_work()
            # 更新资源已经接受的任务信息
            self.state.extend(vehicle.get_state())
        for mec in self.MECs:
            self.state.extend(mec.get_state())
        # 更新时间
        self.cur_frame = cur_frame

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

    # 执行动作
    def step(self):
        cur_frame = self.cur_frame + 1
        state = self.state
        self.get_action()  # 获得动作
        self.get_aim()  # 获得目标
        self.distribute_task()  # 分配任务
        reward = self.get_averageReward()  # 获得当前奖励
        self.totalReward += reward
        print("{}times average reward:".format(cur_frame), reward)
        print("total reward:", self.totalReward)
        self.renew_state(cur_frame)  # 更新状态
        return state, self.actions, reward, self.state
