import random

import numpy as np

np.random.seed(2)


class Task:
    """
    定义任务类型
    """

    def __init__(self, id):
        self.id = id  # 车辆的id
        self.size = 3  # MB
        self.cycle = 3000  # cycle/bit
        self.max_time = 1.2  # s
        self.need_trans_size = self.size  # s
        self.need_precess_cycle = self.cycle * self.size
        self.action = None
        self.arrive = 0  # 传输到达时间
        self.rate = 0  # 当前传输速率
