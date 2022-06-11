# -*- coding: utf-8 -*-
from collections import namedtuple

import torch
from torch.nn import functional as F
from MyErion.model.env import Env
from MyErion.model.memory import ReplayMemory
from MyErion.model.vehicle import Vehicle

capacity = 100  # 缓存池容量
gamma = 0.7  # 奖励折扣
Experience = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))  # Define a transition tuple


# 将list装换成tensor存入缓冲池中
def save_experience(state, action, reward, next_state, memory: ReplayMemory):
    reward = torch.tensor([reward])
    action = torch.tensor([action])
    state = torch.tensor(state)
    state = state.unsqueeze(0)
    next_state = torch.tensor(next_state)
    next_state = next_state.unsqueeze(0)
    memory.push(state, action, reward, next_state)


# 优化模型
def optimize_model(batch, vehicle: Vehicle):
    state = torch.cat(batch.otherState)
    reward = torch.cat(batch.reward)
    action = torch.cat(batch.action)
    next_state = torch.cat(batch.next_state)

    state_action_value = vehicle.cur_network(state).gather(1, action.unsqueeze(-1)).squeeze(1)
    print(state_action_value)

    next_state_action = torch.unsqueeze(vehicle.cur_network(state).max(1)[1], 1)
    next_state_values = vehicle.target_network(next_state).gather(1, next_state_action)
    expected_state_value = (next_state_values * gamma) + reward.unsqueeze(1)
    print(expected_state_value)

    loss = F.mse_loss(state_action_value, expected_state_value)
    vehicle.optimizer.zero_grad()
    loss.backward()
    vehicle.optimizer.step()


def run_episodes(env: Env, memory: ReplayMemory):
    state, actions, reward, next_state = env.step(env.get_action())
    save_experience(state, actions, reward, next_state, memory)


if __name__ == '__main__':
    memory = ReplayMemory(capacity)
    env = Env()
    env.reset()
    for i in range(10):
        run_episodes(env, memory)
        print(env.vehicles[0].otherState)

    # transitions = memory.sample(3)
    # batch = Experience(*zip(*transitions))
    # print(torch.cat(batch.state))
    # print(torch.cat(batch.action))
    # print(torch.cat(batch.next_state))
    # print(torch.cat(batch.reward))
    # optimize_model(batch, env.vehicles[0])
