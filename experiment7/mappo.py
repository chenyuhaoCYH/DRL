# -*- coding: utf-8 -*-
import time

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim

from env import Env
import model
from pylab import mpl
import netron

from memory import PPOMemory

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

GAMMA = 0.99
GAE_LAMBDA = 0.95

TRAJECTORY_SIZE = 100
LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-4

PPO_EPS = 0.2
PPO_EPOCHES = 10
PPO_BATCH_SIZE = 64
EPISODE = 50000

TEST_ITERS = 50000

TASK_DIM = 5
AIM_DIM = 7

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams['axes.unicode_minus'] = False


def calc_adv_ref(trajectory, net_crt, vehicles_states_v, task_states_v, action_v, device="cpu"):
    """
    By trajectory calculate advantage and 1-step ref value
    :param trajectory: trajectory list
    :param net_crt: critic network
    :param vehicles_states_v: states tensor
    :param task_states_v: task states tensor
    :param action_v: action tensor
    :param device: device
    :return: tuple with advantage numpy array and reference values
    """
    # 获得价值
    values_v = net_crt(vehicles_states_v, task_states_v, action_v)
    values = values_v.squeeze().data.cpu().numpy()
    # generalized advantage estimator: smoothed version of the advantage
    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, exp in zip(reversed(values[:-1]),
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

    adv_v = torch.tensor(list(reversed(result_adv)))
    ref_v = torch.tensor(list(reversed(result_ref)))
    return adv_v.to(device), ref_v.to(device)


# 获取动作和动作熵
def choose_action(actor_network: model.ModelActor, self_states, neighbor_states, task_states, device="cpu"):
    # 转成tensor并送入actor网络中
    self_states = torch.tensor([self_states], dtype=torch.float32).to(device)
    neighbor_states = torch.tensor([[neighbor_states]], dtype=torch.float32).to(device)
    task_states = torch.tensor([[task_states]], dtype=torch.float32).to(device)
    task_dist, aim_dist = actor_network(self_states, neighbor_states, task_states)
    # 采样动作
    task_action = task_dist.sample()
    aim_action = aim_dist.sample()
    # 获取概率熵
    task_probs = torch.squeeze(task_dist.log_prob(task_action)).item()
    aim_probs = torch.squeeze(aim_dist.log_prob(aim_action)).item()
    # 获得数值
    task_action = torch.squeeze(task_action).item()
    aim_action = torch.squeeze(aim_action).item()
    # 返回 任务动作，任务动作熵，目标动作，目标动作熵
    return task_action, task_probs, aim_action, aim_probs


# 获得状态动作价值
def val_action(critic_network, vehicles_states, task_states, action, device="cpu"):
    vehicles_states = torch.tensor([[vehicles_states]], dtype=torch.float32).to(device)
    task_states = torch.tensor([[task_states]], dtype=torch.float32).to(device)
    action = torch.tensor([action], dtype=torch.float32).to(device)
    state_action_value = critic_network(vehicles_states, task_states, action)
    return torch.squeeze(state_action_value).item()


# 一次更新更新
def update(memory: PPOMemory, actor_network, actor_opt, critic_network, critic_opt, device="cpu"):
    for _ in range(PPO_EPOCHES):
        self_state_arr, neighbor_state_arr, task_states_arr, vehicles_states_arr, old_task_probs_arr, old_aim_probs_arr, vals_arr, actions_arr, reward_arr, batches = memory.sample()
        values = vals_arr[:]
        advantage = np.zeros(len(reward_arr), dtype=np.float32)
        for t in range(len(reward_arr) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr) - 1):
                a_t += discount * (reward_arr[k] + GAMMA * (values[k + 1]) - values[k])
                discount *= GAMMA * GAE_LAMBDA
            advantage[t] = a_t
        advantage_v = torch.tensor(advantage).to(device)
        # SGD
        values_v = torch.tensor(values).to(device)
        for batch in batches:
            self_state_v = torch.tensor(self_state_arr[batch], dtype=torch.float32).to(device)
            neighbor_state_v = torch.tensor(neighbor_state_arr[batch], dtype=torch.float32).to(device)
            task_state_v = torch.tensor(task_states_arr[batch], dtype=torch.float32).to(device)
            vehicles_states_v = torch.tensor(vehicles_states_arr[batch], dtype=torch.float32).to(device)
            old_task_probs_v = torch.tensor(old_task_probs_arr[batch], dtype=torch.float32).to(device)
            old_aim_probs_v = torch.tensor(old_aim_probs_arr[batch], dtype=torch.float32).to(device)
            actions_v = torch.tensor(actions_arr[batch], dtype=torch.float32).to(device)

            # 获取值(新网络)
            task_dist, aim_dist = actor_network(self_state_v, neighbor_state_v, task_state_v)
            q_value = critic_network(vehicles_states_v, task_state_v, actions_v)
            q_value = torch.squeeze(q_value)

            new_task_probs_v = task_dist.log_prob(actions_v[:, 0])
            new_aim_probs_v = aim_dist.log_prob(actions_v[:, 1])

            task_prob_ratio = new_task_probs_v.exp() / old_task_probs_v.exp()
            task_weighted_probs = advantage_v[batch] * task_prob_ratio
            task_weighted_probs_clipped = torch.clamp(task_weighted_probs, 1 - PPO_EPS, 1 + PPO_EPS) * advantage_v[
                batch]
            aim_prob_ratio = new_aim_probs_v.exp() / old_aim_probs_v.exp()
            aim_weighted_probs = advantage_v[batch] * aim_prob_ratio
            aim_weighted_probs_clipped = torch.clamp(aim_weighted_probs, 1 - PPO_EPS, 1 + PPO_EPS) * advantage_v[batch]
            task_actor_loss = -torch.min(task_weighted_probs_clipped, task_weighted_probs).mean()
            aim_actor_loss = -torch.min(aim_weighted_probs_clipped, aim_weighted_probs).mean()
            actor_loss = 0.4 * task_actor_loss + 0.6 * aim_actor_loss
            # torch.autograd.backward([task_actor_loss, aim_actor_loss])
            # actor_opt.step()

            returns = advantage_v[batch] + values_v[batch]
            critic_loss = (returns - q_value) ** 2
            critic_loss = critic_loss.mean()
            total_loss = actor_loss + 0.5 * critic_loss
            # critic_loss.backward()
            actor_opt.zero_grad()
            critic_opt.zero_grad()
            total_loss.backward()
            actor_opt.step()
            critic_opt.step()
    memory.clear()


if __name__ == '__main__':
    env = Env()
    env.reset()
    num_agents = env.num_Vehicles
    vehicles = env.vehicles

    # obs_shape = len(vehicles[0].self_state)
    # neighbor_shape = np.array([vehicles[0].neighbor_state]).shape
    # task_shape = np.array([vehicles[0].task_state]).shape
    # actor_model = model.ModelActor(obs_shape,
    #                                neighbor_shape,
    #                                task_shape,
    #                                5,
    #                                2 + 5)
    # # 针对有网络模型，但还没有训练保存 .pth 文件的情况
    # # actor网络
    #
    # modelPath = "./netStruct/actor_model.onnx"  # 定义模型结构保存的路径
    # self_state = torch.tensor([vehicles[0].self_state], dtype=torch.float32)
    # neighbor_state = torch.tensor([[vehicles[0].neighbor_state]], dtype=torch.float32)
    # task_state = torch.tensor([[vehicles[0].task_state]], dtype=torch.float32)
    # v1, v2 = actor_model(self_state, neighbor_state, task_state)
    # print(v1.probs)
    # print(v2.probs)
    # v1, v2, v3, v4 = choose_action(actor_model, self_state, neighbor_state, task_state)

    # torch.onnx.export(actor_model, (self_state, neighbor_state, task_state), modelPath)  # 导出并保存
    # netron.start(modelPath)

    # critic网络
    # tasks_shape = torch.tensor([vehicles[0].task_state]).shape
    # critic_model = model.ModelCritic(np.array([env.vehicles_state]).shape, tasks_shape, 2)
    # vehicles_state = torch.tensor([[env.vehicles_state]], dtype=torch.float32)
    # act_state = torch.tensor([[1, 1]], dtype=torch.float32)
    # task_state = torch.tensor([[vehicles[0].task_state]], dtype=torch.float32)
    # value = val_action(critic_model, vehicles_state, task_state, act_state)
    # print(value)
    #
    # # modelPath = "./netStruct/critic_model.onnx"
    # # torch.onnx.export(critic_model, (vehicles_state, task_state, act_state), modelPath)  # 导出并保存
    # # netron.start(modelPath)
    # calc_adv_ref([123, 1231, 23], critic_model, vehicles_state, task_state, act_state)
    actor_models = []
    actor_optimizers = []
    critic_models = []
    critic_optimizers = []

    # 初始化网络
    vehicle_shape = len(vehicles[0].self_state)
    neighbor_shape = np.array([vehicles[0].neighbor_state]).shape
    task_shape = np.array([vehicles[0].task_state]).shape
    all_vehicles_shape = np.array([env.vehicles_state]).shape
    for i in range(num_agents):
        actor_model = model.ModelActor(vehicle_shape, neighbor_shape, task_shape, TASK_DIM, AIM_DIM)
        actor_optimizer = optim.Adam(actor_model.parameters(), LEARNING_RATE_ACTOR)
        critic_model = model.ModelCritic(all_vehicles_shape, task_shape, 2)
        critic_optimizer = optim.Adam(critic_model.parameters(), LEARNING_RATE_CRITIC)
        actor_models.append(actor_model)
        actor_optimizers.append(actor_optimizer)
        critic_models.append(critic_model)
        critic_optimizers.append(critic_optimizer)

    rewards = []
    vehicle1_reward = []
    # 开始训练
    step = 0
    while step < EPISODE:
        step += 1
        # if step % 100 == 0:
        #     print("重置环境")
        #     env = Env()
        #     env.reset()
        #     vehicles = env.vehicles
        # 设置状态缓存队列
        self_states = []
        neighbor_states = []
        task_states = []
        all_vehicles_states = []
        task_probs = []
        aim_probs = []
        # 获取动作
        task_actions = []
        aim_actions = []
        vals = []
        # 遍历每辆车
        for i in range(num_agents):
            self_state = vehicles[i].self_state
            neighbor_state = vehicles[i].neighbor_state
            task_state = vehicles[i].task_state
            all_vehicles_state = env.vehicles_state

            # 获得一辆车的动作
            task_action, task_prob, aim_action, aim_prob = choose_action(actor_models[i], self_state, neighbor_state,
                                                                         task_state)
            # 评测动作价值
            val = val_action(critic_models[i], all_vehicles_state, task_state, [task_action, aim_action])
            # 存入缓冲中用于记录
            self_states.append(self_state)
            neighbor_states.append(neighbor_state)
            task_states.append(task_state)
            all_vehicles_states.append(all_vehicles_state)
            task_probs.append(task_prob)
            aim_probs.append(aim_prob)
            task_actions.append(task_action)
            aim_actions.append(aim_action)
            vals.append(val)

        # 环境执行一步
        other_state, task_state, vehicle_state, vehicles_state, otherState, taskState, neighborState, avg_reward, vehicle_reward = env.step(
            task_actions, aim_actions)
        rewards.append(avg_reward)
        vehicle1_reward.append(vehicle_reward[1])
        print("the {} step reward: {}".format(step, avg_reward))
        print("current 100 times total rewards:", np.mean(rewards[-100:]))

        # 存入经验池
        for i in range(num_agents):
            vehicles[i].memory.push(self_states[i], [neighbor_states[i]], [task_states[i]], [all_vehicles_states[i]],
                                    task_actions[i], aim_actions[i], task_probs[i], aim_probs[i],
                                    vals[i], avg_reward)

        if len(vehicles[0].memory.self_state) < TRAJECTORY_SIZE:
            continue
        print("开始更新。。")
        for i in range(num_agents):
            update(vehicles[i].memory, actor_models[i], actor_optimizers[i], critic_models[i], critic_optimizers[i])
            vehicles[i].memory.clear()

        if step % 1000 == 0:
            env = Env()
            env.reset()
            vehicles = env.vehicles

        if step % 10000 == 0 and step != 0:
            print("保存模型....")
            # 保存网络
            cur_time = time.strftime("%Y-%m-%d-%H", time.localtime(time.time())) + "-" + str(step)
            # 创建文件夹
            os.makedirs("D:/pycharm/Project/VML/MyErion/experiment7/result/ppo/" + cur_time)
            for i, vehicle in enumerate(env.vehicles):
                # 保存每个网络模型
                torch.save(actor_models[i].state_dict(),
                           "D:/pycharm/Project/VML/MyErion/experiment7/result/ppo/" + cur_time + "/vehicle" + str(
                               i) + ".pkl")

    # 打印数据
    plt.plot(range(len(rewards)), rewards)
    plt.title("rewards")
    plt.xlabel("episode")
    plt.show()

    # plt.plot(range(len(vehicle1_reward)), vehicle1_reward)
    # plt.title("训练时车辆1奖励曲线")
    # plt.xlabel("episode")
    # plt.show()
