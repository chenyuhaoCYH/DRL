# -*- coding: utf-8 -*-
import time
from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from pylab import mpl
import netron
import model

from env import Env
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

np.random.seed(2)

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams['axes.unicode_minus'] = False

Experience = namedtuple('Transition',
                        field_names=['vehicle_state', 'neighbor_state', 'task_state', 'all_vehicle_state',
                                     'task_action', 'aim_action', 'reward',
                                     'next_vehicle_state', 'next_neighbor_state', 'next_task_state',
                                     'next_all_vehicle_state'])  # Define a transition tuple

REPLAY_SIZE = 100000
LEARNING_RATE_Actor = 1e-5
LEARNING_RATE_Critic = 1e-4
GAMMA = 0.9
BATCH_SIZE = 64
REPLAY_INITIAL = 10000
TARGET_STEPS = 10

EPSILON = 400000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 0.6
EPSILON_FINAL = 0.01


@torch.no_grad()
def play_step(env, models, epsilon):
    vehicles = env.vehicles
    old_self_state = []
    old_neighbor_state = []
    old_task_state = []
    old_all_vehicle_state = env.vehicles_state

    actionTask = []
    actionAim = []
    for i, model in enumerate(models):
        old_self_state.append(vehicles[i].self_state)
        old_neighbor_state.append(vehicles[i].neighbor_state)
        old_task_state.append(vehicles[i].task_state)
        if np.random.random() < epsilon:
            task_action = np.random.randint(0, 5)
            aim_action = np.random.randint(0, 7)
        else:
            self_state_v = torch.tensor([vehicles[i].self_state], dtype=torch.float32)
            neighbor_state_v = torch.tensor([[vehicles[i].neighbor_state]], dtype=torch.float32)
            task_state_v = torch.tensor([[vehicles[i].task_state]], dtype=torch.float32)
            task_action, aim_action = model(self_state_v, neighbor_state_v, task_state_v)

            task_action = np.argmax(np.array(task_action, dtype=np.float32).reshape(-1))
            aim_action = np.argmax(np.array(aim_action, dtype=np.float32).reshape(-1))
            # 采样动作
            # task_action = torch.squeeze(task_dist.sample()).item()
            # aim_action = torch.squeeze(aim_dist.sample()).item()
            # task_action, aim_action = get_action(task_dist, aim_dist, False)

        actionTask.append(task_action)
        actionAim.append(aim_action)
    Reward, reward = env.step(actionTask, actionAim)

    # 加入各自的缓存池【当前其他状态、当前任务状态、目标动作、任务动作，下一其他状态、下一任务状态】
    for i, vehicle in enumerate(vehicles):
        exp = Experience(old_self_state[i], [old_neighbor_state[i]], [old_task_state[i]], [old_all_vehicle_state],
                         [actionTask[i]], [actionAim[i]], Reward,
                         vehicle.self_state, [vehicle.neighbor_state], [vehicle.task_state], [env.vehicles_state])
        vehicle.buffer.append(exp)
    return round(Reward, 5)  # 返回总的平均奖励


# 将经验转换成torch
def unpack_batch_ddpg(batch, device='cpu'):
    vehicle_state, neighbor_state, task_state, all_vehicle_state, \
    task_action, aim_action, reward, \
    next_vehicle_state, next_neighbor_state, next_task_state, next_all_vehicle_state = batch
    vehicle_state_v = torch.tensor(vehicle_state, dtype=torch.float32).to(device)
    neighbor_state_v = torch.tensor(neighbor_state, dtype=torch.float32).to(device)
    task_state_v = torch.tensor(task_state, dtype=torch.float32).to(device)
    all_vehicle_state_v = torch.tensor(all_vehicle_state, dtype=torch.float32).to(device)
    task_action_v = torch.tensor(task_action, dtype=torch.float32).to(device)
    aim_action_v = torch.tensor(aim_action, dtype=torch.float32).to(device)
    reward_v = torch.tensor(reward, dtype=torch.float32).to(device)
    next_vehicle_state_v = torch.tensor(next_vehicle_state, dtype=torch.float32).to(device)
    next_neighbor_state_v = torch.tensor(next_neighbor_state, dtype=torch.float32).to(device)
    next_task_state_v = torch.tensor(next_task_state, dtype=torch.float32).to(device)
    next_all_vehicle_state_v = torch.tensor(next_all_vehicle_state, dtype=torch.float32).to(device)

    return vehicle_state_v, neighbor_state_v, task_state_v, all_vehicle_state_v, \
           task_action_v, aim_action_v, reward_v, \
           next_vehicle_state_v, next_neighbor_state_v, next_task_state_v, next_all_vehicle_state_v


if __name__ == '__main__':
    env = Env()
    env.reset()

    vehicles = env.vehicles
    actor_models = []
    actor_target_models = []
    actor_optimizers = []
    critic_models = []
    critic_target_models = []
    critic_optimizers = []

    # 初始化网络
    TASK_DIM = 5
    AIM_DIM = len(vehicles[0].neighbor) + 2
    vehicle_shape = len(vehicles[0].self_state)
    neighbor_shape = np.array([vehicles[0].neighbor_state]).shape
    task_shape = np.array([vehicles[0].task_state]).shape
    all_vehicles_shape = np.array([env.vehicles_state]).shape
    for vehicle in vehicles:
        actor_model = model.ModelActor(vehicle_shape, neighbor_shape, task_shape, TASK_DIM, AIM_DIM)
        target_actor_model = model.TargetNet(actor_model)
        actor_optimizer = optim.Adam(actor_model.parameters(), LEARNING_RATE_Actor)

        critic_model = model.ModelCritic(all_vehicles_shape, task_shape, 1, 1)
        target_critic_model = model.TargetNet(critic_model)
        critic_optimizer = optim.Adam(critic_model.parameters(), LEARNING_RATE_Critic)

        actor_models.append(actor_model)
        actor_target_models.append(target_actor_model)
        actor_optimizers.append(actor_optimizer)
        critic_models.append(critic_model)
        critic_target_models.append(target_critic_model)
        critic_optimizers.append(critic_optimizer)

    time_solt = 0
    epsilon = EPSILON

    total_reward = []
    recent_reward = []
    while epsilon > 0:
        time_solt += 1
        print("the {} step".format(time_solt))
        # 执行一步
        eps = max(EPSILON_FINAL, EPSILON_START - time_solt / EPSILON_DECAY_LAST_FRAME)
        reward = play_step(env, actor_models, eps)
        total_reward.append(reward)
        print("current reward:", reward)
        print("current 100 times total rewards:", np.mean(total_reward[-100:]))
        recent_reward.append(np.mean(total_reward[-100:]))
        if np.mean(total_reward[-100:]) > 0.7:
            break

        for i, vehicle in enumerate(vehicles):
            if len(vehicle.buffer) < REPLAY_INITIAL:
                continue
            # 从经验池中选取经验
            batch = vehicle.buffer.sample(BATCH_SIZE)
            vehicle_state_v, neighbor_state_v, task_state_v, all_vehicle_state_v, \
            task_action_v, aim_action_v, reward_v, next_vehicle_state_v, next_neighbor_state_v, \
            next_task_state_v, next_all_vehicle_state_v = unpack_batch_ddpg(batch=batch)

            # train critic
            critic_optimizers[i].zero_grad()
            # 计算q
            task_q_v, aim_q_v = critic_models[i](all_vehicle_state_v, task_state_v, task_action_v, aim_action_v)
            next_task_action, next_aim_action = actor_target_models[i].target_model(next_vehicle_state_v,
                                                                                    next_neighbor_state_v,
                                                                                    next_task_state_v)
            # next_task_action = next_task_a_dist.sample()
            # next_task_action = next_task_action.unsqueeze(1)
            # next_aim_action = next_aim_a_v_dist.sample()
            # next_aim_action = next_aim_action.unsqueeze(1)
            next_task_action = torch.argmax(next_task_action, dim=1).unsqueeze(1)
            next_aim_action = torch.argmax(next_aim_action, dim=1).unsqueeze(1)
            # 计算q‘
            next_task_q_v, next_aim_q_v = critic_target_models[i].target_model(next_all_vehicle_state_v,
                                                                               next_task_state_v, next_task_action,
                                                                               next_aim_action)
            task_q_ref_v = reward_v.unsqueeze(dim=-1) + next_task_q_v * GAMMA
            aim_q_ref_v = reward_v.unsqueeze(dim=-1) + next_aim_q_v * GAMMA
            task_loss_v = F.mse_loss(task_q_v, task_q_ref_v.detach())
            aim_loss_v = F.mse_loss(aim_q_v, aim_q_ref_v.detach())
            torch.autograd.backward([task_loss_v, aim_loss_v])
            critic_optimizers[i].step()

            # train actor
            actor_optimizers[i].zero_grad()
            cur_task_action_v, cur_aim_action_v = actor_models[i](vehicle_state_v, neighbor_state_v, task_state_v)
            task_action = torch.argmax(cur_task_action_v, dim=1).unsqueeze(1)
            aim_action = torch.argmax(cur_aim_action_v, dim=1).unsqueeze(1)
            actor_task_loss, actor_aim_loss = critic_models[i](all_vehicle_state_v, task_state_v, task_action,
                                                               aim_action)
            actor_task_loss = -actor_task_loss.mean()
            actor_aim_loss = -actor_aim_loss.mean()
            torch.autograd.backward([actor_task_loss, actor_aim_loss])
            actor_optimizers[i].step()

            # 目标网络软更新
            if time_solt % TARGET_STEPS == 0:
                critic_target_models[i].alpha_sync(alpha=1 - 1e-3)
                actor_target_models[i].alpha_sync(alpha=1 - 1e-3)
        epsilon -= 1

    cur_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
    # 创建文件夹
    os.makedirs("D:/pycharm/Project/VML/MyErion/experiment6/result/" + cur_time)
    for i, vehicle in enumerate(env.vehicles):
        # 保存每个网络模型
        torch.save(actor_target_models[i].target_model.state_dict(),
                   "D:/pycharm/Project/VML/MyErion/experiment6/result/" + cur_time + "/vehicle" + str(i) + ".pkl")

    plt.plot(range(len(recent_reward)), recent_reward)
    plt.title("当前最近100次奖励曲线")
    plt.show()

    plt.plot(range(len(total_reward)), total_reward)
    plt.title("奖励曲线")
    plt.show()
