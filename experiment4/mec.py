# -*- coding: utf-8 -*-

RANGE_MEC = 650  # MEC通信范围 /m
RESOURCE = 20000  # 可用资源  MHz
MAX_QUEUE = 10


# 边缘服务器
class MEC:
    def __init__(self, position, resources=RESOURCE, max_queue=MAX_QUEUE):
        self.loc_x = position[0]
        self.loc_y = position[1]
        self.loc = position
        # 当前可用资源 MHz
        self.resources = resources
        self.state = []
        # 通信范围 m
        self.range = RANGE_MEC
        # 当前接到需要处理的任务信息(最多同时处理10个任务)
        self.accept_task = []
        # 最多处理任务量
        self.max_task = 10
        # 接受任务的数量
        self.sum_needDeal_task = 0
        # 此时刻有多少动作选则我 多少任务选择传输给我
        self.len_action = 0
        # 等待计算的任务队列（理解为挂起状态）
        self.task_queue = []
        # 用于奖励计算的任务队列
        self.task_queue_for_reward = []
        # 队列最长长度
        self.max_queue = max_queue
        # 当前状态
        self.get_state()

    @property
    def get_x(self):
        return self.loc_x

    @property
    def get_y(self):
        return self.loc_y

    @property
    def get_location(self):
        return self.loc

    """
        获得状态
    """

    def get_state(self):
        """
        :return:state 维度：[loc_x,loc_y,sum_needDeal_task,resources]
        """
        self.state = []
        self.state.extend(self.loc)
        self.state.append(self.sum_needDeal_task)
        self.state.append(self.len_action)
        self.state.append(self.resources)
        return self.state
