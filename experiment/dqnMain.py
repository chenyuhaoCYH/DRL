# -*- coding: utf-8 -*-
import time
from collections import namedtuple
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from MyErion.experiment import Env
from MyErion.experiment.model import DQN
from memory import ExperienceBuffer
import matplotlib.pyplot as plt

Experience = namedtuple('Transition',
                        field_names=['cur_otherState', 'cur_TaskState', 'aimAction', 'TaskAction', 'reward',
                                     'next_otherState', 'next_TaskState'])  # Define a transition tuple
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 0.6
EPSILON_FINAL = 0.01

momentum = 0.005

MAX_NEIGHBOR = 5  # 最大邻居数


@torch.no_grad()
def play_step(env, epsilon, models, device="cpu"):
    vehicles = env.vehicles
    action = []
    old_otherState = []
    old_taskState = []

    actionTask = []
    actionAim = []
    # 贪心选择动作
    for i, model in enumerate(models):
        old_otherState.append(vehicles[i].otherState)
        old_taskState.append(vehicles[i].taskState)
        if np.random.random() < epsilon:
            actionTask.append(np.random.randint(0, 7))
            actionAim.append(np.random.randint(0, 7))
        else:
            state_v = torch.tensor([vehicles[i].otherState], dtype=torch.float32)
            taskState_v = torch.tensor([[vehicles[i].taskState]], dtype=torch.float32)
            cur_action = np.array(model(state_v, taskState_v), dtype=np.float32).reshape(-1)
            lenAction = int(len(cur_action) / 2)
            taskAction = cur_action[lenAction:]
            aimAction = cur_action[0:lenAction]

            actionAim.append(np.argmax(aimAction))
            actionTask.append(np.argmax(taskAction))
    action.extend(actionAim)
    action.extend(actionTask)
    # print("action:", action)
    _, taskState, otherState, _, Reward, reward = env.step(action)
    # print("reward:", reward)

    # 加入各自的缓存池【当前其他状态、当前任务状态、目标动作、任务动作，下一其他状态、下一任务状态】
    for i, vehicle in enumerate(vehicles):
        exp = Experience(old_otherState[i], [old_taskState[i]], actionAim[i], actionTask[i], reward[i], otherState[i],
                         [taskState[i]])
        vehicle.buffer.append(exp)
    return Reward  # 返回总的平均奖励


# 计算一个智能体的损失
def calc_loss(batch, net: DQN, tgt_net: DQN, device="cpu"):
    cur_otherState, cur_TaskState, aimAction, taskAction, rewards, next_otherState, next_TaskState = batch
    otherStates_v = torch.tensor(np.array(cur_otherState, copy=False), dtype=torch.float32).to(device)
    taskStates_v = torch.tensor(np.array(cur_TaskState, copy=False), dtype=torch.float32).to(device)
    # print("states_v:", states_v)  # batch状态
    aimActions_v = torch.tensor(np.array(aimAction), dtype=torch.int64).to(device)
    taskActions_v = torch.tensor(np.array(taskAction), dtype=torch.int64).to(device)
    # print("actions_v", actions_v)  # batch动作
    rewards_v = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
    # print("rewards_v", rewards_v)  # batch奖励
    next_otherStates_v = torch.tensor(np.array(next_otherState, copy=False), dtype=torch.float32).to(device)
    next_taskStates_v = torch.tensor(np.array(next_TaskState, copy=False), dtype=torch.float32).to(device)
    # print("next_states_v", next_states_v)  # batch下一个状态

    state_action_values = net(otherStates_v, taskStates_v).gather(1, aimActions_v.unsqueeze(-1)).squeeze(-1)
    # print("state_action_values", state_action_values)  # batch q值
    next_states_values = tgt_net(next_otherStates_v, next_taskStates_v).max(1)[0]  # 得到最大的q值

    # 防止梯度流入用于计算下一状态q近似值得NN
    next_states_values = next_states_values.detach()
    # print("next_states_values", next_states_values)
    expected_state_values = next_states_values * GAMMA + rewards_v
    # print(" expected_state_values", expected_state_values)

    return nn.MSELoss()(state_action_values, expected_state_values)


if __name__ == '__main__':
    env = Env()
    env.reset()

    frame_idx = 0
    # writer = SummaryWriter(comment="-" + env.__doc__)
    agents = env.vehicles
    models = []
    tgt_models = []
    optimizers = []
    for agent in agents:
        # print(agent.get_location, agent.velocity)
        task_shape = np.array([agent.taskState]).shape
        # print(task_shape)
        model = DQN(len(agent.otherState), (len(agent.neighbor) + 2) * 2, task_shape)
        models.append(model)
        optimer = optim.RMSprop(params=model.parameters(), lr=LEARNING_RATE, momentum=momentum)
        optimizers.append(optimer)
    for agent in agents:
        # print(agent.get_location, agent.velocity)
        task_shape = np.array([agent.taskState]).shape
        # print(task_shape)
        model = DQN(len(agent.otherState), (len(agent.neighbor) + 2) * 2, task_shape)
        model.load_state_dict(models[agent.id].state_dict())
        tgt_models.append(model)
    print(models)
    total_reward = []
    recent_reward = []
    loss_1 = []
    reward_1 = []

    epsilon = EPSILON_START
    eliposde = 100000
    while eliposde > 0:
        frame_idx += 1
        print("the {} steps".format(frame_idx))
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        reward = play_step(env, epsilon, models)
        total_reward.append(reward)
        print("current reward:", reward)
        print("current 100 times total rewards:", np.mean(total_reward[-100:]))
        recent_reward.append(np.mean(total_reward[-100:]))
        if np.mean(total_reward[-100:]) > 5:
            break

        for i, agent in enumerate(agents):
            # print("length of {} buffer".format(agent.id), len(agent.buffer))
            if len(agent.buffer) < REPLAY_SIZE:  # 缓冲池要足够大
                continue
            if frame_idx % SYNC_TARGET_FRAMES == 0:  # 更新目标网络
                tgt_models[i].load_state_dict(models[i].state_dict())
            optimizers[i].zero_grad()
            batch = agent.buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, models[i], tgt_models[i])
            # print("loss:", loss_t)
            loss_t.backward()
            optimizers[i].step()
            if agent.id == 0:
                print("cur_loss:", loss_t.item())
                loss_1.append(loss_t.item())
                reward_1.append(env.reward[0])
        eliposde -= 1

    cur_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
    # 创建文件夹
    os.makedirs("D:/pycharm/Project/VML/MyErion/result/" + cur_time)
    for i, vehicle in enumerate(env.vehicles):
        # 保存每个网络模型
        torch.save(models[i].state_dict(),
                   "D:/pycharm/Project/VML/MyErion/result/" + cur_time + "/vehicle" + str(i) + ".pkl")

    plt.plot(range(len(recent_reward)), recent_reward)
    plt.title("奖励曲线")
    plt.show()
