import random

import numpy as np

np.random.seed(2)

Fv = 400  # 车最大计算能力  MHZ
C = 1.1  # cpu轮数和任务大小关系


class Task:
    """
    定义任务类型
    """

    def __init__(self, vehicle, createTime):
        # 产生任务的车辆
        self.vehicle = vehicle
        # MB
        self.size = random.randint(1, 3)
        # cycle/byte
        self.cycle = np.random.randint(100, 3200)
        # s  最大容忍时间
        self.max_time = self.size * self.cycle / (5 * Fv)
        # s 还剩余多少未传输完成
        self.need_trans_size = self.size
        # MB * cycle/byte =MB cycle 还剩余多少轮次未完成
        self.need_precess_cycle = self.cycle * self.size
        # 完成该任务所消耗的cup资源
        self.energy = 0
        # 传输所需要的时间
        self.trans_time = 0
        # 任务产生时间
        self.create_time = createTime
        # 被选择的时刻（出队列时间）
        self.option = 0
        # 任务处理所需要的时间
        self.precess_time = 0
        # 传送对象
        self.aim = None
