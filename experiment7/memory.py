# 经验类型
import collections
from collections import namedtuple
from random import sample
import numpy as np

Experience = namedtuple('Transition',
                        field_names=['cur_otherState', 'cur_TaskState', "cur_NeighborState",  # 状态
                                     'taskAction', 'aimAction',  # 动作
                                     'reward',  # 奖励
                                     'next_otherState', 'next_TaskState',
                                     'next_NeighborState'])  # Define a transition tuple


class PPOMemory:
    def __init__(self, batch_size):
        self.self_state = []
        self.neighbor_state = []
        self.task_state = []
        self.vehicles_state = []
        self.task_probs = []
        self.aim_probs = []
        self.vals = []
        self.action = []
        self.rewards = []
        self.batch_size = batch_size

    def sample(self):
        batch_step = np.arange(0, len(self.self_state), self.batch_size)
        indices = np.arange(len(self.self_state), dtype=np.int64)
        # np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_step]
        return np.array(self.self_state), \
               np.array(self.neighbor_state), \
               np.array(self.task_state), \
               np.array(self.vehicles_state), \
               np.array(self.task_probs), \
               np.array(self.aim_probs), \
               np.array(self.vals), \
               np.array(self.action), \
               np.array(self.rewards), \
               batches

    def push(self, self_state, neighbor_state, task_state, vehicles_state,
             task_action, aim_action,
             task_probs, aim_probs,
             vals, reward):
        self.self_state.append(self_state)
        self.neighbor_state.append(neighbor_state)
        self.task_state.append(task_state)
        self.vehicles_state.append(vehicles_state)
        self.action.append([task_action, aim_action])
        self.task_probs.append(task_probs)
        self.aim_probs.append(aim_probs)
        self.vals.append(vals)
        self.rewards.append(reward)

    def clear(self):
        self.self_state = []
        self.neighbor_state = []
        self.task_state = []
        self.vehicles_state = []
        self.task_probs = []
        self.aim_probs = []
        self.vals = []
        self.action = []
        self.rewards = []


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
        self.maxLen = capacity
        self.buffer = collections.deque(maxlen=capacity)  # 队列，先进先出

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        cur_otherState, cur_TaskState, cur_NeighborState, taskAction, aimAction, rewards, next_otherState, next_TaskState, next_NeighborState = zip(
            *[self.buffer[idx] for idx in indices])
        # 转换成numpy
        return np.array(cur_otherState), np.array(cur_TaskState), np.array(cur_NeighborState), \
               np.array(taskAction), np.array(aimAction), \
               np.array(rewards, dtype=np.float32), \
               np.array(next_otherState), np.array(next_TaskState), np.array(next_NeighborState)

    # 清空
    def clear(self):
        self.buffer = collections.deque(maxlen=self.maxLen)
