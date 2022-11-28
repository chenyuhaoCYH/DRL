# 经验类型
import collections
from collections import namedtuple
import numpy as np

Experience = namedtuple('Transition',
                        field_names=['vehicle_state', 'neighbor_state', 'task_state', 'all_vehicle_state',
                                     'task_action', 'aim_action', 'reward',
                                     'next_vehicle_state', 'next_neighbor_state', 'next_task_state',
                                     'next_all_vehicle_state'])  # Define a transition tuple


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
        vehicle_state, neighbor_state, task_state, all_vehicle_state, \
        task_action, aim_action, reward, \
        next_vehicle_state, next_neighbor_state, next_task_state, next_all_vehicle_state = zip(
            *[self.buffer[idx] for idx in indices])
        # 转换成numpy
        return np.array(vehicle_state), np.array(neighbor_state), \
               np.array(task_state), np.array(all_vehicle_state), \
               np.array(task_action), np.array(aim_action), \
               np.array(reward, dtype=np.float32), \
               np.array(next_vehicle_state), np.array(next_neighbor_state), \
               np.array(next_task_state), np.array(next_all_vehicle_state)


def clear(self):
    """
    清空
    """
    self.buffer = collections.deque(maxlen=self.maxLen)
