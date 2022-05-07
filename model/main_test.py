# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from MyErion.model.env import Env
from MyErion.model.dqn import DQN
from memory import ExperienceBuffer

Experience = namedtuple('Transition',
                        field_names=['state', 'action', 'reward', 'next_state'])  # Define a transition tuple
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 1000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01


@torch.no_grad()
def play_step(env, epsilon, device="cpu"):
    action = env.get_action(eps_threshold=epsilon)
    states, actions, rewards, next_states = env.step(action)
    env.push(states, actions, rewards, next_states)
    return env.totalReward  # 返回最近一百秒的总的平均奖励


# 计算一个智能体的损失
def calc_loss(batch, net: DQN, tgt_net: DQN, device="cpu"):
    states, actions, rewards, next_states = batch
    states_v = torch.tensor(np.array(states, copy=False)).to(device)
    actions_v = torch.tensor(np.array(actions)).to(device)
    rewards_v = torch.tensor(np.array(rewards)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)

    state_action_values = net(states_v).gether(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_states_values = tgt_net(next_states_v).max(1)[0]

    next_states_values = next_states_values.detach()
    expected_state_values = next_states_values * GAMMA + rewards_v

    return nn.MSELoss()(state_action_values, expected_state_values)


if __name__ == '__main__':
    env = Env()
    env.reset()
    env.init_network()

    frame_idx = 0
    # writer = SummaryWriter(comment="-" + env.__doc__)
    agents = env.vehicles
    print(agents[0].cur_network)

    total_reward = []

    epsilon = EPSILON_START
    while True:
        frame_idx += 1
        print("the {} steps".format(frame_idx))
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        reward = play_step(env, epsilon)
        total_reward.append(reward)
        print("current 100 times total rewards:", np.mean(total_reward[-100:]))
        if np.mean(total_reward[-100:]) > 20:
            break

        for agent in agents:
            if len(agent.buffer) < REPLAY_SIZE:  # 缓冲池要足够大
                continue
            if frame_idx % SYNC_TARGET_FRAMES == 0:  # 更新目标网络
                agent.target_network.load_state_dict(agent.cur_network.state_dict())
            agent.optimizer.zero_grad()
            batch = agent.buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, agent.cur_network, agent.target_network)
            loss_t.backward()
            agent.optimizer.step()
