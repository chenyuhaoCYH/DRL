# -*- coding: utf-8 -*-
import time
from collections import namedtuple
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from MyErion.model.env import Env
from MyErion.model.dqn import DQN
from memory import ExperienceBuffer
import matplotlib.pyplot as plt

Experience = namedtuple('Transition',
                        field_names=['state', 'action', 'reward', 'next_state'])  # Define a transition tuple
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 0.6
EPSILON_FINAL = 0.01


@torch.no_grad()
def play_step(env, epsilon, device="cpu"):
    action = env.get_action(eps_threshold=epsilon)
    states, actions, rewards, next_states = env.step(action)
    env.push(states, actions, rewards, next_states)  # 存入各自缓冲池
    return env.Reward  # 返回总的平均奖励


# 计算一个智能体的损失
def calc_loss(batch, net: DQN, tgt_net: DQN, device="cpu"):
    states, actions, rewards, next_states = batch
    states_v = torch.tensor(np.array(states, copy=False), dtype=torch.float32).to(device)
    # print("states_v:", states_v)  # batch状态
    actions_v = torch.tensor(np.array(actions), dtype=torch.int64).to(device)
    # print("actions_v", actions_v)  # batch动作
    rewards_v = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
    # print("rewards_v", rewards_v)  # batch奖励
    next_states_v = torch.tensor(np.array(next_states, copy=False), dtype=torch.float32).to(device)
    # print("next_states_v", next_states_v)  # batch下一个状态

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    # print("state_action_values", state_action_values)  # batch q值
    next_states_values = tgt_net(next_states_v).max(1)[0]  # 得到最大的q值

    # 防止梯度流入用于计算下一状态q近似值得NN
    next_states_values = next_states_values.detach()
    # print("next_states_values", next_states_values)
    expected_state_values = next_states_values * GAMMA + rewards_v
    # print(" expected_state_values", expected_state_values)

    return nn.MSELoss()(state_action_values, expected_state_values)


if __name__ == '__main__':
    env = Env()
    env.reset()
    env.init_network()

    frame_idx = 0
    # writer = SummaryWriter(comment="-" + env.__doc__)
    agents = env.vehicles
    for agent in agents:
        print(agent.get_location, agent.velocity)
    mecs = env.MECs
    for mec in mecs:
        print(mec.get_location)
    print(agents[0].cur_network)

    total_reward = []
    recent_reward = []

    epsilon = EPSILON_START
    eliposde = 100000
    while eliposde > 0:
        frame_idx += 1
        print("the {} steps".format(frame_idx))
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        reward = play_step(env, epsilon)
        total_reward.append(reward)
        print("current reward:", reward)
        print("current 100 times total rewards:", np.mean(total_reward[-100:]))
        recent_reward.append(np.mean(total_reward[-100:]))
        if np.mean(total_reward[-100:]) > 5:
            break

        for agent in agents:
            # print("length of {} buffer".format(agent.id), len(agent.buffer))
            if len(agent.buffer) < REPLAY_SIZE:  # 缓冲池要足够大
                continue
            if frame_idx % SYNC_TARGET_FRAMES == 0:  # 更新目标网络
                agent.target_network.load_state_dict(agent.cur_network.state_dict())
            agent.optimizer.zero_grad()
            batch = agent.buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, agent.cur_network, agent.target_network)
            # print("loss:", loss_t)
            loss_t.backward()
            agent.optimizer.step()
        eliposde -= 1

    cur_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
    # 创建文件夹
    os.makedirs("D:/pycharm/Project/VML/MyErion/result/"+cur_time)
    for i, vehicle in enumerate(env.vehicles):
        # 保存每个网络模型
        torch.save(vehicle.target_network.state_dict(), "D:/pycharm/Project/VML/MyErion/result/"+cur_time+"/vehicle"+str(i)+".pkl")
    plt.plot(range(len(recent_reward)), recent_reward)
    plt.show()
