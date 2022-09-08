import numpy as np

np.random.seed(2)

Fv = 400  # 车最大计算能力  MHZ


class Task:
    """
    定义任务类型
    """

    def __init__(self, vehicle, createTime):
        self.vehicle = vehicle  # 产生任务的车辆
        self.size = np.random.uniform(0.2, 1)  # Mb
        self.cycle = np.random.randint(20, 50)  # cycle/bit
        self.max_time = self.size * self.cycle / (5 * Fv)  # ms  最大容忍时间
        self.need_trans_size = self.size * np.power(2, 10)  # Kb 还剩余多少未传输完成
        self.need_precess_cycle = self.cycle * self.size * 1000  # Mb * cycle/byte =M cycle 还剩余多少轮次未完成（10^6)

        # 通信速率
        self.rate = 1
        self.compute_resource = 1
        # 完成该任务所消耗的cup资源
        self.energy = 0
        self.trans_time = 0  # 传输所需要的时间
        self.create_time = createTime  # 任务产生时间
        self.pick_time = 0  # 被选择的时间（出队列时间）
        self.precess_time = 0  # 任务处理所需要的时间
        # 完成该任务所消耗的资源
        self.aim = None  # 传送对象
