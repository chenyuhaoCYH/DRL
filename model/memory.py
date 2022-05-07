# 经验类型
import collections
from collections import namedtuple
from random import sample
import numpy as np

Experience = namedtuple('Transition',
                        field_names=['state', 'action', 'reward', 'next_state'])  # Define a transition tuple


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


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_states)
