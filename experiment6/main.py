# -*- coding: utf-8 -*-
import argparse
import os
import time
from collections import namedtuple

import ptan
import torch
import torch.nn.functional as F
import model
from env import Env
from mec import MEC
from vehicle import Vehicle
from memory import ReplayMemory
from tensorboardX import SummaryWriter
from torch.distributions.categorical import Categorical

ENV_ID = "computing offloading"
GAMMA = 0.99
GAE_LAMBDA = 0.95

TRAJECTORY_SIZE = 65
LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-4

PPO_EPS = 0.2
PPO_EPOCHES = 10
PPO_BATCH_SIZE = 64

TEST_ITERS = 10000
Experience = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))  # Define a transition tuple


# 将list装换成tensor存入缓冲池中
def save_experience(state, action, reward, next_state, memory: ReplayMemory):
    reward = torch.tensor([reward])
    action = torch.tensor([action])
    state = torch.tensor(state)
    state = state.unsqueeze(0)
    next_state = torch.tensor(next_state)
    next_state = next_state.unsqueeze(0)
    memory.push(state, action, reward, next_state)


def calc_adv_ref(trajectory, net_crt, states_v, device="cpu"):
    """
    By trajectory calculate advantage and 1-step ref value
    :param trajectory: trajectory list
    :param net_crt: critic network
    :param states_v: states tensor
    :return: tuple with advantage numpy array and reference values
    """
    values_v = net_crt(torch.tensor(states_v))
    values = values_v.squeeze().data.cpu().numpy()
    # generalized advantage estimator: smoothed version of the advantage
    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, exp in zip(reversed(values[:-1]),
                                  reversed(values[1:]),
                                  reversed(trajectory[:-1])):
        delta = exp.vehicleReward + GAMMA * next_val - val
        last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(list(reversed(result_adv)))
    ref_v = torch.FloatTensor(list(reversed(result_ref)))
    return adv_v.to(device), ref_v.to(device)


# 将状态信息放入各自的缓冲池中
def push(env, state, actions, next_state):
    for i, vehicle in enumerate(env.vehicles):
        if vehicle.task is not None:  # 没有任务不算经验
            continue
        exp = Experience(state, actions[i], env.vehicleReward[i][-1], next_state)
        vehicle.buffer.append(exp)


if __name__ == '__main__':
    task = MEC([10, 20])
    vehicle = Vehicle(1, [10, 20], 'd')
    print(type(task) == MEC)
    print(type(task) == Vehicle)
    print(type(vehicle) == Vehicle)
    print(type(vehicle))
    print(vehicle)
