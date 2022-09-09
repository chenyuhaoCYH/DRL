# -*- coding: utf-8 -*-
import os
import time
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from experiment import test_net
from env import Env
from memory import ReplayMemory
import model
from tensorboardX import SummaryWriter
import ptan
import argparse
import numpy as np

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
Experience = namedtuple('Transition', ['state', 'task_state', 'action', 'reward', 'next_state',
                                       'next_task_state'])  # Define a transition tuple


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
# def push(env, state, actions, next_state):
#     for i, vehicle in enumerate(env.vehicles):
#         if vehicle.task is not None:  # 没有任务不算经验
#             continue
#         exp = Experience(state, actions[i], env.vehicleReward[i][-1], next_state)
#         vehicle.buffer.append(exp)


if __name__ == '__main__':
    # 参数配置
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

    # 训练环境
    env = Env()
    env.reset()
    # 测试环境
    test_env = Env()
    test_env.reset()

    # 去除任务的状态空间
    act_state = len(env.vehicles[0].otherState)
    # 任务空间
    task_dim = np.array([env.vehicles[0].taskState]).shape
    # 动作空间
    act_action = 1 + 1 + len(env.vehicles[0].neighbor)

    act_nets = []
    for i in env.vehicles:
        act_net = model.ModelActor(act_state, act_action, task_dim)
        act_nets.append(act_net)

    crt_net = model.ModelCritic(len(env.otherState))
    for act_net in act_nets:
        print(act_net)
    print(crt_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    writer = SummaryWriter(comment="compute offloading")

    act_opts = []
    for act_net in act_nets:
        act_opt = torch.optim.Adam(act_net.parameters(), lr=args.lra)
        act_opts.append(act_opt)
    crt_opt = torch.optim.Adam(crt_net.parameters(), lr=args.lrc)

    step_idx = 0
    best_reward = None

    while True:
        # 卸载动作
        action_act = []
        # 选择任务动作
        action_task = []
        step_idx += 1

        with torch.no_grad():
            for i, act_net in enumerate(act_nets):
                _, action_pro, _, task_pro = act_net(torch.tensor([env.vehicles[i].otherState], dtype=torch.float32),
                                                     torch.tensor([[env.vehicles[i].taskState]]))
                # print(pro)
                # act = np.random.choice(pro.shape[0], 1, p=pro.detach().numpy())
                # 按照概率采样
                act = action_pro.sample()
                task = task_pro.sample()
                # print(act)
                # 加入数组中
                action_act.append(act.item())
                action_task.append(task.item())
        # print(action_act)
        # print(action_task)
        # 执行动作
        action_act.extend(action_task)
        print(action_act)
        state, task_state, vehicles_state, new_vehicles_state, new_state, new_task_state, Reward, reward = env.step(
            action_act)

        # 测试
        if step_idx % TEST_ITERS == 0:
            ts = time.time()
            rewards, steps = test_net(act_nets, test_env)
            print("Test done in %.2f sec, reward %.3f, steps %d" % (
                time.time() - ts, rewards, steps))
            writer.add_scalar("test_reward", rewards, step_idx)
            writer.add_scalar("test_steps", steps, step_idx)
            if best_reward is None or best_reward < rewards:
                if best_reward is not None:
                    print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                    name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                    fname = os.path.join(save_path, name)
                    for i, act_net in enumerate(act_nets):
                        torch.save(act_net.state_dict(), fname + "{}".format(i))
                best_reward = rewards

        for i, vehicle in enumerate(env.vehicles):
            # if vehicle.task is not None:  # 没有任务不算经验
            #     continue
            # 存储车经验
            action = [env.offloadingActions[i], env.taskActions[i]]
            exp = Experience(vehicles_state[i], [task_state[i]], action, reward[i],
                             new_vehicles_state[i], [new_task_state[i]])
            vehicle.buffer.append(exp)
        # 存储系统经验
        env.buffer.append(Experience(state, [task_state], action_act, Reward, new_state, [new_task_state]))

        print("the {} reward:{}".format(step_idx, reward))
        if step_idx % TRAJECTORY_SIZE != 0:
            continue

        # print("存储池", env.vehicles[0].buffer[0])
        # print(len(env.vehicles[0].buffer))
        # 车的缓冲池
        traj_states = [[] for _ in range(env.num_Vehicles)]
        traj_taskStates = [[] for _ in range(env.num_Vehicles)]
        traj_actions = [[] for _ in range(env.num_Vehicles)]

        traj_states_v = [[] for _ in range(env.num_Vehicles)]
        traj_taskStates_v = [[] for _ in range(env.num_Vehicles)]
        traj_actions_v = [[] for _ in range(env.num_Vehicles)]
        old_logprob_v = [[] for _ in range(env.num_Vehicles)]

        # 环境的缓冲池
        # 其他状态
        traj_statesOfenv = [t.next_state for t in env.buffer]
        # 任务状态
        traj_taskStatesOfenv = [t.next_task_state for t in env.buffer]
        # 转成tensor
        traj_statesOfenv_v = torch.FloatTensor(traj_statesOfenv)
        traj_taskStatesOfenv_v = torch.FloatTensor(traj_taskStatesOfenv)

        # 收集每辆车的经验
        for i in range(env.num_Vehicles):
            traj_states[i] = [t.next_state for t in env.vehicles[i].buffer]
            traj_taskStates = [t.next_task_state for t in env.vehicles[i].buffer]
            traj_actions[i] = [t.action for t in env.vehicles[i].buffer]
            # 转成tensor
            traj_states_v[i] = torch.FloatTensor(traj_states[i])
            traj_states_v[i] = traj_states_v[i].to(device)

            traj_taskStates_v[i] = torch.FloatTensor([traj_taskStates[i]])
            traj_taskStates_v[i] = traj_taskStates_v[i].to(device)

            traj_actions_v[i] = torch.FloatTensor(traj_actions[i])
            traj_actions_v[i] = traj_actions_v[i].to(device)

            _, pro_act, _, pro_task = act_nets[i](traj_states_v[i], traj_taskStates_v[i])
            action_act = [pro_act.sample(), pro_task.sample()]
            ans = torch.sum(pro_act, dim=1)
            ans1 = torch.sum(pro_task, dim=1)
            # print(ans.data)
            old_logprob_v[i] = Categorical.log_prob(pro_act, action_act[0]) + Categorical.log_prob(pro_task,
                                                                                                   action_act[1])

        traj_adv_v, traj_ref_v = calc_adv_ref(env.buffer, crt_net, traj_statesOfenv)

        # normalize advantages
        traj_adv_v = traj_adv_v - torch.mean(traj_adv_v)
        traj_adv_v /= torch.std(traj_adv_v)

        # drop last entry from the trajectory, an adv and ref value calculated without it
        env.buffer = env.buffer[:-1]
        for i in range(env.num_Vehicles):
            env.vehicles[i].buffer[i] = env.vehicles[i].buffer[i][:-1]
            old_logprob_v[i] = old_logprob_v[i][:-1].detach()

        sum_loss_value = [0.0 for i in env.vehicles]
        sum_loss_policy = [0.0 for i in env.vehicles]
        count_steps = 0

        for epoch in range(PPO_EPOCHES):
            for batch_ofs in range(0, len(env.buffer),
                                   PPO_BATCH_SIZE):
                batch_l = batch_ofs + PPO_BATCH_SIZE

                states_e = traj_statesOfenv_v[batch_ofs:batch_l]
                batch_adv_v = traj_adv_v[batch_ofs:batch_l]
                batch_adv_v = batch_adv_v.unsqueeze(-1)
                batch_ref_v = traj_ref_v[batch_ofs:batch_l]

                actions_v = [[] for _ in range(env.num_Vehicles)]
                states_v = [[] for _ in range(env.num_Vehicles)]
                batch_old_logprob_v = [[] for _ in range(env.num_Vehicles)]

                # critic training
                crt_opt.zero_grad()
                value_v = crt_net(states_e)
                value_v = value_v.squeeze()
                loss_value_v = F.mse_loss(
                    value_v, batch_ref_v)
                loss_value_v.backward()
                crt_opt.step()

                # actor training
                for i in range(env.num_Vehicles):
                    batch_old_logprob_v[i] = old_logprob_v[i][batch_ofs:batch_l]
                    states_v[i] = traj_states_v[i][batch_ofs:batch_l]
                    actions_v[i] = traj_actions_v[i][batch_ofs:batch_l]

                    act_opts[i].zero_grad()
                    _, pro_act = act_nets[i](states_v[i])
                    logprob_pi_v = Categorical.log_prob(pro_act, actions_v[i])
                    ratio_v = torch.exp(
                        logprob_pi_v - batch_old_logprob_v[i])
                    surr_obj_v = batch_adv_v * ratio_v
                    c_ratio_v = torch.clamp(ratio_v,
                                            1.0 - PPO_EPS,
                                            1.0 + PPO_EPS)
                    clipped_surr_v = batch_adv_v * c_ratio_v
                    loss_policy_v = -torch.min(
                        surr_obj_v, clipped_surr_v).mean()
                    loss_policy_v.backward()
                    act_opts[i].step()

                    sum_loss_value[i] += loss_value_v.item()
                    sum_loss_policy[i] += loss_policy_v.item()

                    count_steps += 1
        for i in range(env.num_Vehicles):
            env.vehicles[i].buffer.clear()
        env.buffer.clear()

        writer.add_scalar("advantage", traj_adv_v.mean().item(), step_idx)
        writer.add_scalar("values", traj_ref_v.mean().item(), step_idx)
        for i in range(env.num_Vehicles):
            writer.add_scalar("loss_policy{}".format(i), sum_loss_policy[i] / count_steps, step_idx)
            writer.add_scalar("loss_value{}".format(i), sum_loss_value[i] / count_steps, step_idx)
