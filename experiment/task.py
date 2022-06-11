import random

import numpy as np

np.random.seed(2)

C = 1.1  # cpu轮数和任务大小关系


class Task:
    """
    定义任务类型
    """

    def __init__(self, Id, createTime):
        self.id = Id  # 车辆的id
        self.size = random.randint(1, 3)  # MB
        self.cycle = np.random.randint(100, 3200)  # cycle/byte
        self.max_time = 3  # s
        self.need_trans_size = self.size  # s 还剩余多少未传输完成
        self.need_precess_cycle = self.cycle * self.size  # MB * cycle/byte =MB cycle 还剩余多少轮次未完成
        self.arrive = 0  # 传输到达时间
        self.create_time = createTime
        self.rate = 0  # 当前传输速率
        self.aim = None  # 传送对象
