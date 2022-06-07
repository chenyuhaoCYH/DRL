# -*- coding: utf-8 -*-

import numpy as np
import torch
import ptan

from MyErion.experiment.task import Task
from MyErion.model.memory import ExperienceBuffer
from model import ModelActor
import torch.optim as optim

BrandWidth = 100  # 带宽MHz
alpha = 0.25  # 信道增益
Fv = 2  # 车的计算能力
Dv = 50  # 车的最大通信范围

CAPACITY = 20000  # 缓冲池大小
TASK_DISTRIBUTE = 4  # 可分的任务段数
MAX_NEIGHBOR = 20  # 最大邻居数
# 网络学习率
LEARNING_RATE = 1e-4

np.random.seed(0)


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
        self.accept_task = []
        # 接受任务的数量
        self.sum_needDeal_task = 0
        # 此时刻有多少动作选则我
        self.len_action = 0
        # 当前可用资源
        self.resources = round((1 - np.random.randint(1, 5) / 10) * Fv, 2)  # GHz
        # 当前任务
        self.task = None
        # 当前任务需要处理的时间
        self.needProcessedTime = 0
        # 当前状态信息
        self.state = []
        # 去除邻居的状态信息用于邻居车观察和全局critic的处理
        self.excludeNeighbor_state = []
        # # 网络 三个网络对应三个任务
        # # 任务一
        # self.actor1 = None
        # self.target_actor1 = None
        # self.optimizer1 = None
        # # 任务二
        # self.actor2 = None
        # self.target_actor2 = None
        # self.optimizer2 = None
        # # 任务三
        # self.actor3 = None
        # self.target_actor3 = None
        # self.optimizer3 = None
        # 缓冲池
        self.buffer = []  # ExperienceBuffer(capacity=CAPACITY)
        # 总奖励
        self.reward = []

        self.get_state()

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
        if np.random.random() <= 0.5:
            self.task = Task(self.id)
            print("第{}辆车产生了任务".format(self.id))
        else:
            self.task = None

    # # 初始化当前网络
    # def init_network(self, state: int, action: int):
    #     self.actor1 = ModelActor(state, action)
    #     self.actor2 = ModelActor(state, action)
    #     self.actor3 = ModelActor(state, action)
    #
    #     self.target_actor1 = ptan.agent.TargetNet(self.actor1)
    #     self.target_actor2 = ptan.agent.TargetNet(self.actor2)
    #     self.target_actor3 = ptan.agent.TargetNet(self.actor3)
    #
    #     # 设置优化器
    #     self.optimizer1 = optim.Adam(params=self.actor1.parameters(), lr=LEARNING_RATE)
    #     self.optimizer1 = optim.Adam(params=self.actor2.parameters(), lr=LEARNING_RATE)
    #     self.optimizer1 = optim.Adam(params=self.actor3.parameters(), lr=LEARNING_RATE)

    # 获得状态
    # 维度：[id,loc_x,loc_y,velocity,direction,[neighbor.excludeNeighbor_state]8*20,resources,I,C,T,[type]*3,
    # length_task1and2,length_task3]
    def get_state(self):
        self.state = []
        self.excludeNeighbor_state = []
        # 位置信息  5
        self.state.extend(self.loc)
        self.state.append(self.velocity)
        self.state.append(self.direction)
        self.excludeNeighbor_state.extend(self.loc)
        self.excludeNeighbor_state.append(self.velocity)
        self.excludeNeighbor_state.append(self.direction)
        # 当前可用资源 1
        self.state.append(self.resources)
        self.excludeNeighbor_state.append(self.resources)
        # 任务信息  3
        if self.task is not None:
            self.state.append(self.task.size)
            self.excludeNeighbor_state.append(self.task.size)
            self.state.append(self.task.cycle)
            self.excludeNeighbor_state.append(self.task.cycle)
            self.state.append(self.task.max_time)
            self.excludeNeighbor_state.append(self.task.max_time)
        else:
            self.state.extend([0, 0, 0])
            self.excludeNeighbor_state.extend([0, 0, 0])

        # 需要处理的任务量   2
        self.state.append(self.sum_needDeal_task)
        self.state.append(self.len_action)
        self.excludeNeighbor_state.append(self.sum_needDeal_task)
        self.excludeNeighbor_state.append(self.len_action)

        # 邻居表  7*数量
        for neighbor in self.neighbor:
            self.state.extend(neighbor.loc)
            self.state.append(neighbor.velocity)
            self.state.append(neighbor.direction)
            self.state.append(neighbor.resources)
            self.state.append(neighbor.sum_needDeal_task)
            self.state.append(neighbor.len_action)

        # 最近mec的状态 6
        if self.mec_lest is not None:
            self.state.extend(self.mec_lest.get_state())

        return self.excludeNeighbor_state

    def __str__(self) -> str:
        return super().__str__()
