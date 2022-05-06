# -*- coding: utf-8 -*-
import math
import random
from random import random, uniform, randrange
import numpy as np
import torch
from dqn import DQN
import torch.optim as optim


sigma = -114  # 噪声dbm
POWER = 23  # 功率dbm
BrandWidth = 100  # 带宽MHz
alpha = 0.25  # 信道增益
Fv = 3  # 车的计算能力
N = 40  # 车的数量
K = 3  # MEC的数量
Dv = 20  # 车的最大通信范围
Dk = 200  # MEC的最大通信范围
# 网络学习率
LEARNING_RATE = 0.01
momentum = 0.005


class Vehicle:
    # 位置：x，y 速度、方向：-1左，1右
    def __init__(self, id, loc_x, loc_y, direction, velocity=5):
        # 车的位置信息
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.loc = [loc_x, loc_y]
        self.velocity = velocity  # m/s
        self.direction = direction
        self.id = id
        # 功率和信道增益
        self.alpha = 0.25
        # 通信范围
        self.range = 20
        # 邻居表
        # self.neighbor = []
        # 当前时间
        self.cur_frame = 0
        # 接受的任务数量
        self.recevied_task = []
        # 当前可用资源
        self.resources = round((1 - uniform(0, 0.7)) * Fv, 2)  # GHz
        # 当前任务
        self.task = []
        # 当前任务需要处理的时间
        self.needProcessedTime = 0
        # 当前状态信息
        self.state = []
        # 网络
        self.cur_network = None
        self.target_network = None
        self.optimizer = None

    # 获得位置
    @property
    def get_location(self):
        return self.loc

    # 设置位置
    def set_location(self, loc_x, loc_y):
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.loc = [self.loc_x, self.loc_y]

    # 获得x
    @property
    def get_x(self):
        return self.loc_x

    @property
    def get_y(self):
        return self.loc_y

    # 产生任务 传入当前时间
    def creat_work(self):
        # 每次有0.5的概率产生任务
        if random() < 0.5:
            self.task = [2, 2, 1.2]  # 任务：大小Mbit、需要资源 Mcycle、最大容忍时间s
        else:
            self.task = [0, 0, 0]        # 测试为1，1，1

    # # 计算两车之间的距离
    # def compute_dis(self, vehicle):
    #     return np.sqrt(np.abs(self.get_x - vehicle.get_x) ** 2 + np.abs(self.get_y - vehicle.get_y) ** 2)
    #
    # # 计算处理传输给邻居车当前任务所需要的时间传输时间   带宽平均分成N(车的数量)分 因此不会存在干扰
    # def compute_trans_to_vehicle(self, service_vehicle):
    #     SNR = BrandWidth / N * np.log2(1 + POWER * np.power(self.compute_dis(service_vehicle), -self.alpha) / sigma)
    #     return self.task[0] / SNR
    #
    # # 计算传输给MEC所需要的时间
    # def compute_trans_to_MEC(self, mec):
    #     SNR = BrandWidth / N * np.log2(1 + POWER * np.power(self.compute_dis(mec), -self.alpha) / sigma)
    #     return self.task[0] / SNR  # 单位 ms

    # 计算处理时间
    # action:所有车的决定   action_i:本辆车的决定
    # def compute_dispose(self, action: list, action_i, vehicles: list, mecs):
    #     sum = 0
    #     for i in action:
    #         if i == action_i:
    #             sum += 1  # 找有多少个任务卸载给同一辆车
    #     if action_i == 0:  # 给本地
    #         f = self.resources / sum  # 获得给这个任务分配的资源 GHZ
    #     elif action_i <= K:  # 给MEC
    #         f = mecs[action_i - 1].resources / sum
    #     else:  # 给邻居车
    #         f = vehicles[action_i - K - 1].resources / sum
    #
    #     return self.task[1] / f  # 单位cycles/bit
    #
    # # 计算两车通信的持续时间
    # def compute_persist(self, service_vehicle):
    #     return Dv / np.abs(self.velocity - service_vehicle.velocity) - (self.get_x - service_vehicle.get_x) / (
    #             self.velocity - service_vehicle.velocity)

    # 初始化当前网络
    def init_network(self, state: int, action: int):
        self.cur_network = DQN(state_n=state, actions_n=action)
        self.target_network = DQN(state_n=state, actions_n=action)
        self.target_network.load_state_dict(self.cur_network.state_dict())
        self.target_network.eval()
        # 设置优化器
        self.optimizer = optim.RMSprop(params=self.cur_network.parameters(), lr=LEARNING_RATE,
                                       momentum=momentum)

    # 获得状态  维度：[id,loc_x,loc_y,velocity,direction,resources,I,C,T]  1*9
    def get_state(self):
        self.state = []
        # 位置信息
        self.state.append(self.id)
        self.state.extend(self.loc)
        self.state.append(self.velocity)
        self.state.append(self.direction)
        # 邻居表
        # self.state.extend(self.neighbor)
        self.state.append(self.resources)
        # 任务信息
        self.state.extend(self.task)

        return self.state

    # # 获得当前智能体的动作
    # def get_action(self, state, eps_threshold):
    #     sample = random()
    #     if sample < eps_threshold:  # epsilon-greeedy policy
    #         # 不计算梯度 防止出现噪声 因为此时只利用
    #         with torch.no_grad():
    #             state = torch.tensor([state])
    #             Q_value = self.cur_network(state)  # Get the Q_value from DNN
    #             action = Q_value.max(1)[1].view(1)  # 获得最大值的那一个下标为要采取的动作   (二维取列最大的下标值(一维)-》二维)
    #     else:
    #         action = torch.tensor([randrange(1 + N + K)], dtype=torch.long)
    #     return action.item()  # torch类型

    # # 更新自己需要处理的任务：只保留需要处理的时间信息
    # def renew_recevied_task(self, action: list, vehicles):
    #     for i, action_i in enumerate(action):
    #         if action_i == (self.id + 1 + K) and vehicles[i].task[0] > 0:
    #             self.recevied_task.append(vehicles[i].task[1])

    # # 更新当前可用资源
    # def renew_resources(self, cur_frame):
    #     if len(self.recevied_task) > 0:
    #         f = self.resources / len(self.recevied_task)
    #         self.resources = 0
    #         after_task = []
    #         for i, need_time in enumerate(self.recevied_task):
    #             if need_time / f > cur_frame - self.cur_frame:
    #                 after_task.append(need_time - (cur_frame - self.cur_frame) * f)
    #             else:
    #                 self.resources += f
    #         self.recevied_task = after_task



    # # 更新状态
    # def renew_state(self, cur_frame, action: list, vehicles):
    #     # 更新位置
    #     # self.renew_location(cur_frame=cur_frame)
    #     # 更新邻居信息
    #
    #     # 更新资源信息
    #     self.renew_recevied_task(action, vehicles)
    #     self.renew_resources(cur_frame=cur_frame)
    #     self.get_state()
