# -*- coding: utf-8 -*-
import random

import numpy as np

from task import Task

Dv = 50  # 车的最大通信范围
Fv = 4000  # 车最大计算能力  MHZ
alpha = 0.25
MAX_TASK = 10  # 任务队列最大长度

CAPACITY = 20000  # 缓冲池大小
TASK_DISTRIBUTE = 4  # 可分的任务段数
MAX_NEIGHBOR = 20  # 最大邻居数
TASK_SOLT = 10  # 任务产生时隙
# 网络学习率
LEARNING_RATE = 1e-4

np.random.seed(0)


class Vehicle:
    # 位置：x，y 速度、方向：-1左，1右
    def __init__(self, id, loc_x, loc_y, direction, velocity=20):
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
        # 接受的任务的列表
        self.accept_task = []
        # 接受任务的数量
        self.sum_needDeal_task = 0
        # 此时刻有多少动作选则我
        self.len_action = 0
        # 当前可用资源
        self.resources = round((1 - np.random.randint(1, 5) / 10) * Fv, 2)  # MHz
        # 当前正在传输的任务
        self.trans_task = None
        # 当前处理的任务
        self.cur_task = None
        # 任务队列
        self.total_task = []
        # 任务队列的长度
        self.len_task = len(self.total_task)
        # 当前状态信息
        self.otherState = []
        # 当前任务队列状态
        self.taskState = []
        # 去除邻居的状态信息用于邻居车观察和全局critic的处理
        self.excludeNeighbor_state = []
        # 缓冲池
        self.buffer = []  # ExperienceBuffer(capacity=CAPACITY)
        # 总奖励
        self.reward = []
        # 任务溢出的数量
        self.overflow = 0
        # 需等待时长
        self.hold_on = 0

        self.get_state()
        self.create_work()

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
    def create_work(self):
        # 每20ms进行一次任务产生
        if self.cur_frame % TASK_SOLT == 0:
            # 每次有0.5的概率产生任务
            if random.random() < 0.6:
                if self.len_task < MAX_TASK:  # 队列不满
                    task = Task(self, self.cur_frame)
                    self.total_task.append(task)
                    self.len_task += 1
                    print("第{}辆车产生了任务".format(self.id))
                    self.overflow = 0
                else:
                    print("第{}辆车任务队列已满".format(self.id))
                    self.overflow += 1

    # 获得状态
    def get_state(self):
        self.otherState = []
        self.excludeNeighbor_state = []
        self.taskState = []

        # 位置信息  4
        self.otherState.extend(self.loc)
        self.otherState.append(self.velocity)
        self.otherState.append(self.direction)
        self.excludeNeighbor_state.extend(self.loc)
        self.excludeNeighbor_state.append(self.velocity)
        self.excludeNeighbor_state.append(self.direction)

        # 资源信息（可用资源、正在处理的任务量、正在传输的任务量）
        self.otherState.append(self.resources)
        self.excludeNeighbor_state.append(self.resources)
        self.otherState.append(self.sum_needDeal_task)
        self.otherState.append(self.len_action)
        self.excludeNeighbor_state.append(self.sum_needDeal_task)
        self.excludeNeighbor_state.append(self.len_action)

        # 正在传输的任务信息
        if self.trans_task is not None:
            self.otherState.append(self.trans_task.need_trans_size)
            self.excludeNeighbor_state.append(self.trans_task.need_trans_size)
        else:
            self.otherState.append(0)
            self.excludeNeighbor_state.append(0)
        self.otherState.append(self.len_task)  # 当前队列长度
        self.excludeNeighbor_state.append(self.len_task)

        # 邻居表  7*数量
        for neighbor in self.neighbor:
            self.otherState.extend(neighbor.loc)  # 位置
            self.otherState.append(neighbor.velocity)  # 速度
            self.otherState.append(neighbor.direction)  # 方向
            self.otherState.append(neighbor.resources)  # 可用资源
            self.otherState.append(neighbor.sum_needDeal_task)  # 正在处理的任务量
            self.otherState.append(neighbor.len_action)  # 正在接受的任务量

        # 最近mec的状态 6
        if self.mec_lest is not None:
            self.otherState.extend(self.mec_lest.get_state())

        # 任务状态信息
        for i in range(MAX_TASK):
            if i < self.len_task:
                task = self.total_task[i]
                self.taskState.append([task.create_time, task.need_trans_size, task.need_precess_cycle, task.max_time])
            else:
                self.taskState.append([0, 0, 0, 0])

        return self.excludeNeighbor_state

    def __str__(self) -> str:
        return super().__str__()
