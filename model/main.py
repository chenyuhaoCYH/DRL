# -*- coding: utf-8 -*-
from collections import namedtuple

import torch

from MyErion.model.env import Env
from MyErion.model.memory import ReplayMemory

capacity = 100  # 缓存池容量
Experience = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))  # Define a transition tuple


def run_episodes(env: Env, memory: ReplayMemory):
    state, actions, reward, next_state = env.step()
    memory.push(state, actions, reward, next_state)


if __name__ == '__main__':
    memory = ReplayMemory(capacity)
    env = Env()
    env.reset()
    for i in range(10):
        run_episodes(env, memory)
    transitions = memory.sample(3)
    batch = Experience(*zip(*transitions))
    print("第二次")
    print(torch.cat(batch.state))
