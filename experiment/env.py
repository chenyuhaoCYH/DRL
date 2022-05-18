import random
from collections import namedtuple

import numpy as np
import ptan.agent
import torch
from vehicle import Vehicle
from random import random, randint
from mec import MEC
import model

Experience = namedtuple('Transition',
                        field_names=['state', 'action', 'reward', 'next_state'])  # Define a transition tuple

y = [2, 6, 10, 14]  # 车子y的坐标集 # 共四条车道
direction = [1, 1, -1, -1]  # 车子的方向
velocity = 5  # 车子速度
MEC_loc = [[-300, 0], [0, 16], [300, 0]]  # mec的位置

Fv = 1  # 车的计算能力
N = 100  # 车的数量
K = 3  # mec的数量
MAX_NEIGHBOR = 20  # 最大邻居数

ACTIONS = 1 + MAX_NEIGHBOR + 1  # actor动作空间维度(本地+邻居车+最近的mec)
STATES_CRITIC = 14 * N + 6 * K  # 状态空间维度(critic网络)
STATES_ACTOR = 20 + 14 * MAX_NEIGHBOR  # （actor网络）

sigma = -114  # 噪声dbm
POWER = 23  # 功率dbm
BrandWidth = 100  # 带宽MHz
alpha = 0.25  # 信道增益

eps_threshold = 0.9
min_eps = 0.1

np.random.seed(2)


class Env:
    def __init__(self, num_Vehicles=N, num_MECs=K):
        # 环境内所有车辆
        self.vehicles = []
        # 环境内所有mec
        self.MECs = []
        # 车辆数以及mec数
        self.num_Vehicles = num_Vehicles
        self.num_MECs = num_MECs
        # 当前平均奖励奖励数
        self.Reward = 0
        # 每辆车获得的即时奖励
        self.cur_reward = [0] * num_Vehicles
        # 当前时间
        self.cur_frame = 0
        # 所有车的动作
        self.actions = [0] * num_Vehicles
        # 所有车要传输的对象 vehicle or mec
        self.aims = []
        # 当前全局的状态信息  维度:
        self.state = []
        # 网络
        self.crt_net = None
        self.twing_net = None
        self.tgt_ctr_net = None

    # 添加车辆
    def add_new_vehicles(self, id, loc_x, loc_y, direction, velocity):
        vehicle = Vehicle(id=id, loc_x=loc_x, loc_y=loc_y, direction=direction, velocity=velocity)
        vehicle.creat_work()  # 初始化任务
        # vehicle.init_network(STATES, ACTIONS)  # 初始化网络
        self.vehicles.append(vehicle)

    # 初始化网络
    def init_network(self):
        # 初始化全局critic网络  获得当前价值函数
        self.crt_net = model.ModelCritic(STATES_CRITIC)
        self.tgt_ctr_net = ptan.agent.TargetNet(self.crt_net)
        # 双q网络获得状态动作函数
        self.twing_net = model.ModelSACTwinQ(STATES_CRITIC, ACTIONS)
        # 初始化每辆车的actor网络
        for vehicle in self.vehicles:
            vehicle.init_network(STATES_ACTOR, ACTIONS)

    # 初始化/重置环境
    def reset(self):
        self.Reward = 0
        self.state = []
        self.cur_frame = 0
        self.actions = [0] * self.num_Vehicles

        for i in range(0, self.num_MECs):  # 初始化mec
            cur_mec = MEC(id=i, loc_x=MEC_loc[i][0], loc_y=MEC_loc[i][1])
            self.MECs.append(cur_mec)
            self.state.extend(cur_mec.get_state())

        i = 0
        while i < self.num_Vehicles:  # 初始化车子
            n = np.random.randint(0, 4)  # 左闭右开
            self.add_new_vehicles(id=i, loc_x=randint(-500, 500), loc_y=y[n], direction=direction[n],
                                  velocity=randint(5, 10))
            i += 1
        # 初始化邻居信息
        self.renew_neighbor()
        self.renew_neighbor_mec()
        # 初始化网络
        self.init_network()
        # 初始化状态信息
        for vehicle in self.vehicles:
            vehicle.get_state()
        for vehicle in self.vehicles:
            self.state.extend(vehicle.get_state())

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

    # 获得所有的动作
    def get_action(self):
        self.actions = []
        # # 逐个获取每个vehicle的动作
        # for i, vehicle in enumerate(self.vehicles):
        #     sample = np.random.random()
        #     if sample > eps_threshold:  # epsilon-greeedy policy
        #         # 不计算梯度 防止出现噪声 因为此时只利用
        #         with torch.no_grad():
        #             state = torch.tensor([self.state])
        #             Q_value = vehicle.cur_network(state)  # Get the Q_value from DNN
        #             action = Q_value.max(1)[1].view(1)  # 获得最大值的那一个下标为要采取的动作   (二维取列最大的下标值(一维)-》二维)
        #             # print("vehicle {} current q value:".format(i), Q_value)
        #     else:
        #         action = torch.tensor([randint(0, self.num_MECs + self.num_Vehicles)],
        #                               dtype=torch.int)  # randrange(1 + N + K)]
        #     action = int(action.item())  # torch类型==>int
        #     self.actions.append(action)
        for vehicle in self.vehicles:
            self.actions.extend(vehicle.action)
        return self.actions

    # 获得要传输的对象
    def get_aim(self):
        self.aims = []
        for vehicle in self.vehicles:
            aim = []
            for i, action in enumerate(vehicle.action):
                if action != 0:
                    if i == 0:
                        aim.append(vehicle)
                    elif i == 1:
                        aim.append(vehicle.mec_lest)
                    else:
                        aim.append(vehicle.neighbor[i - 2])
            self.aims.append(aim)

    # 计算距离(车到车或者车到MEC)  aim：接受任务的目标
    def compute_distance(self, taskVehicle: Vehicle, aim):
        return round(np.sqrt(np.abs(taskVehicle.get_x - aim.get_x) ** 2 + np.abs(taskVehicle.get_y - aim.get_y) ** 2),
                     2)

    # 计算一个任务传输时间
    def compute_transmit(self, taskVehicle: Vehicle, aims):
        sum_time = 0
        cur = taskVehicle

    # 根据动作将任务添加至对应的列表当中  分配任务
    def distribute_task(self):
        for i, action in enumerate(self.actions):
            task = self.vehicles[i].task
            if task[0] == 0:
                continue  # 没有任务
            elif action == 0:  # 卸载至本地
                self.vehicles[i].accept_task.append(task)
            elif action <= self.num_MECs:  # 卸载给MEC
                self.MECs[action - 1].accept_task.append(task)
            else:  # 卸载给其他车辆
                self.vehicles[action - self.num_MECs - 1].accept_task.append(task)

    # 计算本辆车这个任务所需的处理时间
    def compute_precessed(self, vehicle: Vehicle, aim):
        total_task = aim.accept_task  # 目标车辆或mec接收到的总任务
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
            if vehicle.task[0] > 0:
                sum += 1
                aim = self.aim[i]
                distance = self.compute_distance(vehicle, aim)
                if distance > aim.range:  # 距离大于通信范围
                    cur_reward = -1
                else:  # 假设一直处于通信范围内
                    trans_time = self.compute_transmit(taskVehicle=vehicle, aim=aim)
                    precessed_time = self.compute_precessed(vehicle, aim=aim)
                    total_time = trans_time + precessed_time
                    if total_time > vehicle.task[2]:  # 超过最大忍耐时间
                        cur_reward = -1
                    else:
                        cur_reward = vehicle.task[2] - total_time  # 剩余时间作为奖励
            else:
                cur_reward = 0
            reward += cur_reward
            self.cur_reward[i] = cur_reward
            vehicle.total_reward += cur_reward  # 记录本车的总奖励
            # print("vehicle{} get {} reward".format(i, cur_reward))
        if sum > 0:
            reward /= sum
        return round(reward, 5)

    # 更新资源信息
    def renew_resources(self, cur_frame):
        time = cur_frame - self.cur_frame
        # 更新车的资源信息
        for i, vehicle in enumerate(self.vehicles):
            total_task = vehicle.accept_task
            if len(total_task) > 0 and vehicle.resources > 0:  # 此时有任务并且有剩余资源
                f = vehicle.resources / len(total_task)
                # print("vehicle {} give {} rescource".format(i, f))
                # vehicle.resources = 0
                after_task = []
                for task in total_task:  # 遍历此车的所有任务列表
                    precessed_time = task[1] / f
                    if precessed_time > time:  # 不能处理完
                        task[1] -= f * time
                        after_task.append(task)
                    # else:  # 能够完成
                    #     vehicle.resources += f  # 更新资源
                vehicle.accept_task = after_task  # 更新任务列表
                vehicle.sum_needDeal_task = len(after_task)
                # 更新能提供的资源（可以省略）
                # vehicle.resources = round((1 - np.random.uniform(0, 0.7)) * Fv, 2)  # GHz

        # 更新mec的资源信息
        for i, mec in enumerate(self.MECs):
            total_task = mec.accept_task
            if len(total_task) > 0 and mec.resources > 0:
                f = mec.resources / len(total_task)
                # print("mec {} give {} rescource".format(i, f))
                mec.resources = 0
                after_task = []
                for task in mec.accept_task:
                    precessed_time = task[1] / f
                    if precessed_time > time:
                        task[1] -= f * time
                        after_task.append(task)
                    else:
                        mec.resources += f
                mec.accept_task = after_task
                mec.sum_needDeal_task = len(after_task)

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

    # 将状态信息放入各自的缓冲池中
    def push(self, state, actions, rewards, next_state):
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.task[0] == 0:  # 没有任务不算经验
                continue
            exp = Experience(state, actions[i], rewards[i], next_state)
            vehicle.buffer.append(exp)

    # 执行动作
    def step(self, actions):
        cur_frame = self.cur_frame + 1
        state = self.state
        # print("vehicle 0 state:", self.vehicles[0].state)
        # print("vehicle 1 state:", self.vehicles[1].state)
        # print("mec 0 state:", self.MECs[0].state)
        self.actions = actions
        # self.get_action()  # 获得动作
        print(self.actions)
        self.get_aim()  # 获得目标
        self.distribute_task()  # 分配任务
        # print("vehicle 0 recevied task:", self.vehicles[0].recevied_task)
        # print("vehicle 1 recevied task:", self.vehicles[1].recevied_task)
        # print("mec 0 recevied task:", self.MECs[0].recevied_task)
        # print("actions:",self.actions)
        # print("aim:",self.aim)
        self.Reward = self.get_averageReward()  # 获得当前奖励
        # if cur_frame % 100 == 0:    # 只计算最近一百秒的平均奖励
        #     self.totalReward = 0
        # self.totalReward += reward
        # print("{}times average reward:".format(cur_frame), self.Reward)
        # print("total reward:", self.totalReward)
        self.renew_state(cur_frame)  # 更新状态
        return state, self.actions, self.cur_reward, self.state
