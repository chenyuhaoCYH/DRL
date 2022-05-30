# -*- coding: utf-8 -*-
import os
import time
from collections import namedtuple

import torch
from MyErion.experiment.env import Env
from MyErion.experiment.memory import ReplayMemory
import model
from tensorboardX import SummaryWriter
import ptan
import argparse
import numpy as np

ENV_ID = "computing offloading"
GAMMA = 0.99
GAE_LAMBDA = 0.95

TRAJECTORY_SIZE = 2049
LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-4

PPO_EPS = 0.2
PPO_EPOCHES = 10
PPO_BATCH_SIZE = 64

N = 100

TEST_ITERS = 100000
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


def calc_adv_ref(trajectory, net_crt, states_v, device="cpu"):
    """
    By trajectory calculate advantage and 1-step ref value
    :param trajectory: trajectory list
    :param net_crt: critic network
    :param states_v: states tensor
    :return: tuple with advantage numpy array and reference values
    """
    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    # generalized advantage estimator: smoothed version of the advantage
    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, (exp,) in zip(reversed(values[:-1]),
                                     reversed(values[1:]),
                                     reversed(trajectory[:-1])):
        if exp.done:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + GAMMA * next_val - val
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
        exp = Experience(state, actions[i], env.reward[i][-1], next_state)
        vehicle.buffer.append(exp)


if __name__ == '__main__':
    time = str(time.time())
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", default=" ", help="Name of the run")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment id, default=" + ENV_ID)
    parser.add_argument("--lrc", default=LEARNING_RATE_CRITIC, type=float, help="Critic learning rate")
    parser.add_argument("--lra", default=LEARNING_RATE_ACTOR, type=float, help="Actor learning rate")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "ppo-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = Env()
    env.reset()
    test_env = Env()
    test_env.reset()

    act_state = len(env.vehicles[0].state)
    print(env.vehicles[0].state)
    print(act_state)
    act_action = 1 + 1 + len(env.vehicles[0].neighbor)

    act_nets = []
    for i in env.vehicles:
        act_net = model.ModelActor(act_state, act_action)
        act_nets.append(act_net)

    crt_net = model.ModelCritic(len(env.state))
    for act_net in act_nets:
        print(act_net)
    print(crt_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    writer = SummaryWriter(comment="compute offloading")

    act_opts = []
    for act_net in act_nets:
        act_opt = torch.optim.Adam(act_net.parameters(), lr=args.lra)
    crt_opt = torch.optim.Adam(crt_net.parameters(), lr=args.lrc)

    frame_idx = 0
    trajectory = []
    best_reward = None

    while True:
        action = []
        frame_idx += 1
        for i, act_net in enumerate(act_nets):
            pro = act_net(torch.tensor(env.vehicles[i].state))
            act = np.random.choice(pro.shape[0], 1, p=pro.detach().numpy())
            action.append(act[0])
        state, cur_action, reward, next_state = env.step(action)

        push(env, state, action, next_state)
        if frame_idx % N == 0:
            pass
        # print("当前状态", state)
        # print("当前动作", action)
        # print("当前平均奖励", reward)
        # print("下一状态", next_state)
        # print(env.reward)
