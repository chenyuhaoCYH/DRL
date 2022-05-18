# -*- coding: utf-8 -*-

import numpy as np
import torch
import ptan

from MyErion.model.memory import ExperienceBuffer
from model import ModelActor
import torch.optim as optim

BrandWidth = 100  # 带宽MHz
alpha = 0.25  # 信道增益
Fv = 3  # 车的计算能力
Dv = 20  # 车的最大通信范围

CAPACITY = 20000  # 缓冲池大小
TASK_DISTRIBUTE = 4  # 可分的任务段数
MAX_NEIGHBOR = 20  # 最大邻居数
# 网络学习率
LEARNING_RATE = 1e-4

np.random.seed(2)


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
        self.alpha = alpha
        # 通信范围
        self.range = Dv
        # 邻居表
        self.neighbor = []
        # 最近的mec
        self.mec_lest = None
        # 当前时间
        self.cur_frame = 0
        # 接受的任务的列表  因为任务三优先级高所以分开处理
        self.accept_task3 = []
        self.accept_task1and2 = []
        # 接受任务的数量
        self.sum_needDeal_task1and2 = 0
        self.sum_needDeal_task3 = 0
        # 当前可用资源
        self.resources = round((1 - np.random.uniform(0.1, 0.6)) * Fv, 2)  # GHz
        # 当前任务
        self.task = []
        # 任务类型 共三种one-hot编码 [0,0,0]
        self.type = []
        # 当前任务需要处理的时间
        self.needProcessedTime = 0
        # 当前状态信息
        self.state = []
        # 去除邻居的状态信息用于邻居车观察和全局critic的处理
        self.excludeNeighbor_state = []
        # 网络 三个网络对应三个任务
        # 任务一
        self.actor1 = None
        self.target_actor1 = None
        self.optimizer1 = None
        # 任务二
        self.actor2 = None
        self.target_actor2 = None
        self.optimizer2 = None
        # 任务三
        self.actor3 = None
        self.target_actor3 = None
        self.optimizer3 = None
        # 缓冲池
        self.buffer = ExperienceBuffer(capacity=CAPACITY)
        # 总奖励
        self.total_reward = 0.0
        # 当前动作
        self.action = []

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

    # 获得y
    @property
    def get_y(self):
        return self.loc_y

    # 产生任务 传入当前时间
    def creat_work(self):
        # 每次有0.7的概率产生任务
        if np.random.random() <= 0.7:
            sample = np.random.random()
            if sample < 0.3:
                self.task = [2, 2, 1]  # 任务：大小M bit、需要资源 G cycle、最大容忍时间s
                self.type = [1, 0, 0]  # 可按数据独立运行的任务  可分成4段每段0.5M
            elif sample < 0.6:
                self.task = [2, 2, 2]
                self.type = [0, 1, 0]  # 可按代码连续执行的任务  可分成4段
            else:
                self.task = [1, 1, 1]
                self.type = [0, 0, 1]  # 只可独立运行的任务，优先级最高
        else:
            self.task = [0, 0, 0]  # 当前不产生任务
            self.type = [0, 0, 0]

    # 初始化当前网络
    def init_network(self, state: int, action: int):
        self.actor1 = ModelActor(state, action)
        self.actor2 = ModelActor(state, action)
        self.actor3 = ModelActor(state, action)

        self.target_actor1 = ptan.agent.TargetNet(self.actor1)
        self.target_actor2 = ptan.agent.TargetNet(self.actor2)
        self.target_actor3 = ptan.agent.TargetNet(self.actor3)

        # 设置优化器
        self.optimizer1 = optim.Adam(params=self.actor1.parameters(), lr=LEARNING_RATE)
        self.optimizer1 = optim.Adam(params=self.actor2.parameters(), lr=LEARNING_RATE)
        self.optimizer1 = optim.Adam(params=self.actor3.parameters(), lr=LEARNING_RATE)

    # 获得状态
    # 维度：[id,loc_x,loc_y,velocity,direction,[neighbor.excludeNeighbor_state]8*20,resources,I,C,T,[type]*3,
    # length_task1and2,length_task3]
    def get_state(self):
        self.state = []
        self.excludeNeighbor_state = []
        # 位置信息  5
        self.state.append(self.id)
        self.state.extend(self.loc)
        self.state.append(self.velocity)
        self.state.append(self.direction)
        self.excludeNeighbor_state.append(self.id)
        self.excludeNeighbor_state.extend(self.loc)
        self.excludeNeighbor_state.append(self.velocity)
        self.excludeNeighbor_state.append(self.direction)
        # 当前可用资源 1
        self.state.append(self.resources)
        self.excludeNeighbor_state.append(self.resources)
        # 任务信息  3
        self.state.extend(self.task)
        self.excludeNeighbor_state.extend(self.task)
        # 任务类型   3
        self.state.extend(self.type)
        self.excludeNeighbor_state.extend(self.type)
        # 需要处理的任务量   2
        self.state.append(self.sum_needDeal_task1and2)
        self.state.append(self.sum_needDeal_task3)
        self.excludeNeighbor_state.append(self.sum_needDeal_task1and2)
        self.excludeNeighbor_state.append(self.sum_needDeal_task3)

        # 邻居表  14*数量
        for neighbor in self.neighbor:
            self.state.extend(neighbor.excludeNeighbor_state)

        # 最近mec的状态 6
        if self.mec_lest is not None:
            self.state.extend(self.mec_lest.get_state())

        return self.excludeNeighbor_state

    # 获得动作
    def get_action(self):
        self.action = [0] * (1 + MAX_NEIGHBOR + 1)
        if self.task[0] == 0:  # 当前没有任务
            return self.action
        elif self.type[0] == 1:  # 任务一
            with torch.no_grad():
                self.actor1(torch.tensor([self.state]))
        elif self.type[1] == 1:  # 任务二
            pass
        else:  # 任务三
            with torch.no_grad():
                policy = self.actor3([torch.tensor(self.state)])


if __name__ == '__main__':
    vehicle = Vehicle(1, 20, 4, 1, 8)
    vehicle.creat_work()
    vehicle.get_state()
    print(vehicle.state)
    vehicle.init_network(len(vehicle.state), 23)
    print(vehicle.actor1(torch.tensor([vehicle.state])))