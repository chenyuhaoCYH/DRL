# -*- coding: utf-8 -*-
import argparse
import os
import time
from collections import namedtuple

import ptan
import torch
import torch.nn.functional as F
from experiment2 import  model
from experiment2.env import Env
from experiment2.memory import ReplayMemory
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

    act_state = len(env.vehicles[0].otherState)
    act_action = 1 + 1 + len(env.vehicles[0].neighbor)

    act_nets = []
    for i in env.vehicles:
        act_net = model.ModelActor(act_state, act_action)
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
        action = []
        step_idx += 1

        with torch.no_grad():
            for i, act_net in enumerate(act_nets):
                _, pro = act_net(torch.tensor(env.vehicles[i].otherState))
                # print(pro)
                # act = np.random.choice(pro.shape[0], 1, p=pro.detach().numpy())
                act = pro.sample()
                # print(act)
                action.append(act.item())

        state, cur_action, reward, next_state = env.step(action)

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
            exp = Experience(env.vehicles_state[i], env.offloadingActions[i], env.vehicleReward[i][-1],
                             env.vehicles[i].otherState)
            vehicle.buffer.append(exp)
        # 存储系统经验
        env.buffer.append(Experience(state, cur_action, reward, next_state))

        print("the {} reward:{}".format(step_idx, reward))
        if step_idx % TRAJECTORY_SIZE != 0:
            continue

        # print("存储池", env.vehicles[0].buffer[0])
        # print(len(env.vehicles[0].buffer))
        traj_states = [[] for _ in range(env.num_Vehicles)]
        traj_actions = [[] for _ in range(env.num_Vehicles)]
        traj_states_v = [[] for _ in range(env.num_Vehicles)]
        traj_actions_v = [[] for _ in range(env.num_Vehicles)]
        old_logprob_v = [[] for _ in range(env.num_Vehicles)]

        traj_statesOfenv = [t.otherState for t in env.buffer]
        tarj_statesOfenv_v = torch.FloatTensor(traj_statesOfenv)

        for i in range(env.num_Vehicles):
            traj_states[i] = [t.otherState for t in env.vehicles[i].buffer]
            traj_actions[i] = [t.action for t in env.vehicles[i].buffer]

            traj_states_v[i] = torch.FloatTensor(traj_states[i])
            traj_states_v[i] = traj_states_v[i].to(device)

            traj_actions_v[i] = torch.FloatTensor(traj_actions[i])
            traj_actions_v[i] = traj_actions_v[i].to(device)

            _, pro_v = act_nets[i](traj_states_v[i])
            action = pro_v.sample()
            # ans=torch.sum(pro_v,dim=1)
            # print(ans.data)
            old_logprob_v[i] = Categorical.log_prob(pro_v, action)

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

                states_e = tarj_statesOfenv_v[batch_ofs:batch_l]
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
                    _, pro_v = act_nets[i](states_v[i])
                    logprob_pi_v = Categorical.log_prob(pro_v, actions_v[i])
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
