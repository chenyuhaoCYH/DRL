import random

import numpy as np

np.random.seed(2)


class Task:
    """
    定义任务类型
    """

    def __init__(self, id):
        self.id = id  # 车辆的id
        self.size = round(np.random.uniform(0.5, 3), 2)  # MB
        self.cycle = np.random.randint(100, 3200)  # cycle/byte
        self.max_time = 3  # s
        self.need_trans_size = self.size  # s
        self.need_precess_cycle = self.cycle * self.size * np.power(2, 20)  # byte * cycle/byte =cycle
        self.arrive = 0  # 传输到达时间
        self.rate = 0  # 当前传输速率
        self.aim = None  # 传送对象
