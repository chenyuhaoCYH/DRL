# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from memory import ExperienceBuffer

Experience = namedtuple('Transition',
                        field_names=['state', 'action', 'reward', 'next_state'])  # Define a transition tuple
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01


@torch.no_grad()
def play_step(memory: ExperienceBuffer, epsilon, env, device="cpu"):
    actions = env.get_action(eps_threshold=epsilon)
    state, actions, reward, next_state = env.step(actions)
    exp = Experience(state, actions, reward, next_state)
    memory.append(exp)


# 计算一个智能体的损失
def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, next_states = batch
    states_v = torch.tensor(np.array(states, copy=False)).to(device)
    actions_v = torch.tensor(np.array(actions)).to(device)
    rewards_v = torch.tensor(np.array(rewards)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
