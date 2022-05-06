# 经验类型
from collections import namedtuple
from random import sample

Experience = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))  # Define a transition tuple


class ReplayMemory(object):  # Define a replay memory

    # 初始化缓冲池
    def __init__(self, capacity):
        # 最大容量
        self.capacity = capacity
        # 缓冲池经验
        self.memory = []
        # ？
        self.position = 0

    # 存入经验
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            # 存入经验
        self.memory[self.position] = Experience(*args)
        # 记录最新经验所在位置
        self.position = (self.position + 1) % self.capacity

    # 采样
    def sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
