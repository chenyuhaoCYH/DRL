import numpy as np

np.random.seed(2)


class Task:
    """
    定义任务类型
    """

    def __init__(self, vehicle=None, createTime=0):
        # 产生任务的车辆
        self.vehicle = vehicle
        # 完成该任务所消耗的资源
        self.aim = None  # 传送对象

        flag = np.random.randint(1, 3)
        if flag <= 2:
            # 娱乐性任务
            self.max_time = np.random.randint(50, 80)  # ms  最大容忍时间
        else:
            # 安全性任务
            self.max_time = np.random.randint(20, 40)
        self.size = np.random.uniform(0.2, 1)  # Mb
        self.cycle = np.random.randint(20, 50)  # cycle/bit
        self.need_trans_size = self.size * np.power(2, 10)  # Kb 还剩余多少未传输完成
        self.need_precess_cycle = self.cycle * self.size * 1000  # Mb * cycle/byte =M cycle 还剩余多少轮次未完成（10^6)
        self.need_time = 0  # 需要计算时间
        self.hold_time = 0  # 任务在计算等待队列中得等待时间

        self.rate = 0  # 当前速率

        self.compute_resource = 0

        self.create_time = createTime  # 任务产生时间
        self.pick_time = 0  # 被选择的时间（出队列时间）

        # 完成该任务所消耗的cup资源
        self.energy = 0
        self.trans_time = 0  # 传输所需要的时间（实际）
        self.precess_time = 0  # 任务处理所需要的时间(实际)
