# -*- coding: utf-8 -*-

import numpy as np

from memory import ExperienceBuffer
from task import Task

Dv = 100  # 车的最大通信范围
Fv = 2000  # 车最大计算能力  MHZ
MAX_TASK = 10  # 任务队列最大长度

CAPACITY = 10000  # 缓冲池大小
TASK_SOLT = 10  # 任务产生时隙

# 等待队列最长长度
MAX_QUEUE = 10

np.random.seed(2)

direction_map = {"d": 1, "u": 2, "l": 3, "r": 4}


class Vehicle:
    # 位置：x，y 速度、方向：-1左，1右
    def __init__(self, id, position, direction, velocity=20, max_queue=MAX_QUEUE):
        self.id = id
        # 车的位置信息
        self.loc_x = position[0]
        self.loc_y = position[1]
        self.position = position
        self.velocity = velocity  # m/s
        self.direction = direction
        # 通信范围
        self.range = Dv
        # 邻居表
        self.neighbor = []
        # mec
        self.Mec = None
        # 当前时间
        self.cur_frame = 0
        # 接受的任务的列表(最多同时处理5个任务)
        self.accept_task = []
        # 最多处理任务量
        self.max_task = 5
        # 等待队列最长长度
        self.max_queue = max_queue
        # 等待计算的任务队列（理解为挂起状态）
        self.task_queue = []
        # 用于奖励计算的任务队列
        self.task_queue_for_reward = []
        # 接受任务的数量
        self.sum_needDeal_task = 0
        # 此时刻有多少动作选择我进行卸载对象
        self.len_action = 0
        # 当前可用资源
        self.resources = round((1 - np.random.randint(1, 3) / 10) * Fv, 2)  # MHz
        # 表示当前是否有任务正在传输给邻居车辆（0：没有，1：有）
        self.trans_task_for_vehicle = 0
        # 当前是否有任务正在传输给mec
        self.trans_task_for_mec = 0
        # 当前处理的任务（用于计算奖励，不用于状态信息）
        self.cur_task = None
        # 任务队列
        self.total_task = []
        # 任务队列的长度
        self.len_task = len(self.total_task)

        # 当前状态信息
        self.self_state = []
        # 当前任务队列状态
        self.task_state = []
        # 去除邻居的状态信息用于邻居车观察和全局critic的处理
        self.excludeNeighbor_state = []
        # 缓冲池
        self.buffer = ExperienceBuffer(capacity=CAPACITY)
        # 总奖励
        self.reward = []
        # 任务溢出的数量
        self.overflow = 0
        # 上一个任务产生的时间
        self.lastCreatWorkTime = 0

        # 产生任务
        self.create_work()

    # 获得位置
    @property
    def get_location(self):
        return self.position

    # 设置位置
    def set_location(self, loc_x, loc_y):
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.position = [self.loc_x, self.loc_y]

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
        if self.id % 3 == 0:
            return
            # 每隔一段时间进行一次任务产生
        if (self.cur_frame - self.lastCreatWorkTime) % TASK_SOLT == 0:
            # # 每次有0.6的概率产生任务
            if np.random.random() < 0.8:
                if self.len_task < MAX_TASK:  # 队列不满
                    task = Task(self, self.cur_frame)
                    self.lastCreatWorkTime = self.cur_frame
                    self.total_task.append(task)
                    self.len_task = len(self.total_task)
                    # print("第{}辆车产生了任务".format(self.id))
                    self.overflow = 0
                else:
                    # print("第{}辆车任务队列已满".format(self.id))
                    self.overflow = 1

    """
    获得状态
    """

    def get_state(self):
        self.self_state = []
        self.excludeNeighbor_state = []
        self.task_state = []

        # 位置信息  4
        self.self_state.extend(self.position)
        self.self_state.append(self.velocity)
        self.self_state.append(direction_map.get(self.direction))
        self.excludeNeighbor_state.extend(self.position)
        self.excludeNeighbor_state.append(self.velocity)
        self.excludeNeighbor_state.append(direction_map.get(self.direction))

        # 资源信息（可用资源）
        self.self_state.append(self.resources)
        self.excludeNeighbor_state.append(self.resources)

        # 当前处理的任务量
        self.self_state.append(self.sum_needDeal_task)
        self.excludeNeighbor_state.append(self.sum_needDeal_task)
        # 当前接受传输的任务量
        self.self_state.append(self.len_action)
        self.excludeNeighbor_state.append(self.sum_needDeal_task)

        # 当前是否有任务在传输
        self.excludeNeighbor_state.append(self.trans_task_for_vehicle)
        self.excludeNeighbor_state.append(self.trans_task_for_mec)
        self.self_state.append(self.trans_task_for_vehicle)
        self.self_state.append(self.trans_task_for_mec)

        # 正在传输的任务信息
        # if self.trans_task is not None:
        #     self.otherState.append(self.trans_task.need_trans_size)
        #     self.excludeNeighbor_state.append(self.trans_task.need_trans_size)
        # else:
        #     self.otherState.append(0)
        #     self.excludeNeighbor_state.append(0)

        # 当前队列长度
        self.self_state.append(self.len_task)
        self.excludeNeighbor_state.append(self.len_task)

        # 邻居表  7*数量
        for neighbor in self.neighbor:
            self.self_state.extend(neighbor.position)  # 位置
            self.self_state.append(neighbor.velocity)  # 速度
            self.self_state.append(direction_map.get(neighbor.direction))  # 方向
            self.self_state.append(neighbor.resources)  # 可用资源
            self.self_state.append(neighbor.sum_needDeal_task)  # 处理任务长度
            self.self_state.append(neighbor.len_action)  # 当前正在传输任务数量

        self.self_state.extend(self.Mec.state)

        # 任务状态信息
        for i in range(MAX_TASK):
            if i < self.len_task:
                task = self.total_task[i]
                self.task_state.append([task.create_time, task.need_trans_size, task.need_precess_cycle, task.max_time])
            else:
                self.task_state.append([0, 0, 0, 0])

        return self.excludeNeighbor_state
