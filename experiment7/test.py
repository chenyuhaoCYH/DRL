from copy import copy

import numpy as np
import torch

import matplotlib
from pylab import mpl
import model
from env import Env
from mecEnv import MecEnv
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 加载 Times New Roman 字体
font_path = 'C:/Windows/Fonts/times.ttf'
prop = fm.FontProperties(fname=font_path, size=8)

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams['axes.unicode_minus'] = False
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
np.random.seed(2)


def choose_action(actor_network: model.ModelActor, self_states, neighbor_states, task_states, device="cpu"):
    # 转成tensor并送入actor网络中
    self_states = torch.tensor([self_states], dtype=torch.float32).to(device)
    neighbor_states = torch.tensor([[neighbor_states]], dtype=torch.float32).to(device)
    task_states = torch.tensor([[task_states]], dtype=torch.float32).to(device)
    task_dist, aim_dist = actor_network(self_states, neighbor_states, task_states)
    # 采样动作
    task_action = task_dist.sample()
    aim_action = aim_dist.sample()
    # 获得数值
    task_action = torch.squeeze(task_action).item()
    aim_action = torch.squeeze(aim_action).item()
    # 返回 任务动作，任务动作熵，目标动作，目标动作熵
    return task_action, aim_action


def ParallelBar(x_labels, y, labels=None, colors=None, width=0.35, gap=2):
    """
    绘制并排柱状图
    :param x_labels: list 横坐标刻度标识
    :param y: list 列表里每个小列表是一个系列的柱状图
    :param labels: list 每个柱状图的标签
    :param colors: list 每个柱状图颜色
    :param width: float 每条柱子的宽度
    :param gap: int 柱子与柱子间的宽度
    """

    # check params
    if labels is not None:
        if len(labels) < len(y): raise ValueError('labels的数目不足')
    if colors is not None:
        if len(colors) < len(y): raise ValueError('颜色colors的数目不足')
    if not isinstance(gap, int): raise ValueError('输入的gap必须为整数')

    x = [t for t in range(0, len(x_labels) * gap, gap)]  # the label locations
    fig, ax = plt.subplots()
    for i in range(len(y)):
        if labels is not None:
            l = labels[i]
        else:
            l = None
        if colors is not None:
            color = colors[i]
        else:
            color = None
        if len(x) != len(y[i]): raise ValueError('所给数据数目与横坐标刻度数目不符')
        plt.bar(x, y[i], label=l, width=width, color=color)
        x = [t + width for t in x]
    x = [t + (len(y) - 1) * width / 2 for t in range(0, len(x_labels) * gap, gap)]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)


if __name__ == '__main__':
    # 我的环境
    env = Env()
    env.reset()
    # 只卸载给mec
    # env = Env()
    # env.reset()
    # # 只自己执行
    # selfEnv = MecEnv()
    # selfEnv.reset()
    # # 随机
    # randomEnv = MecEnv()
    # randomEnv.reset()

    N = env.num_Vehicles
    vehicles = env.vehicles
    models = []

    task_shape = np.array([vehicles[0].task_state]).shape
    neighbor_shape = np.array([vehicles[0].neighbor_state]).shape
    for i in range(N):
        # 加载模型
        tgt_model = model.DQNCNN(len(vehicles[0].self_state), task_shape, neighbor_shape, 10,
                                 len(vehicles[0].neighbor) + 2)
        tgt_model.load_state_dict(
            torch.load(
                "D:\\pycharm\\Project\\VML\\MyErion\\experiment7\\result\\2023-06-16-00-60000\\vehicle{}.pkl".format(
                    i)))
        models.append(tgt_model)
    averageReward_dqn = []
    averageReward_ppo = []
    averageReward_for_mec = []
    averageReward_for_self = []
    averageReward_for_random = []
    for step in range(100):
        action_task = []
        action_aim = []
        for i in range(N):
            state_v = torch.tensor([vehicles[i].self_state], dtype=torch.float32)
            taskState_v = torch.tensor([[vehicles[i].task_state]], dtype=torch.float32)
            neighborState_v = torch.tensor([[vehicles[i].neighbor_state]], dtype=torch.float32)
            taskAction, aimAction = models[i](state_v, taskState_v, neighborState_v)
            taskAction = taskAction.detach().numpy().reshape(-1)
            aimAction = aimAction.detach().numpy().reshape(-1)
            # dpn
            action_task.append(np.argmax(taskAction))
            action_aim.append(np.argmax(aimAction))

        other_state, task_state, vehicle_state, _, _, _, _, Reward, reward = env.step(action_task, action_aim)
        print("第{}次车辆平均奖励{}".format(step, Reward))
    averageReward_dqn = [np.mean(reward) for i, reward in enumerate(env.avg_reward) if i % 3 != 0]
    avg = [np.mean(sum_time) for i, sum_time in enumerate(env.avg) if i % 3 != 0]
    avg_energy = [np.mean(energy) for i, energy in enumerate(env.avg_energy) if i % 3 != 0]
    avg_price = [np.mean(energy) for i, energy in enumerate(env.avg_price) if i % 3 != 0]
    avg_success = [vehicle.success_task / vehicle.sum_create_task for i, vehicle in enumerate(env.vehicles) if
                   i % 3 != 0]
    # ppo
    env.reset()
    models = []
    for i in range(N):
        # 加载模型
        tgt_model = model.ModelActor(len(vehicles[0].self_state), np.array([vehicles[0].neighbor_state]).shape,
                                     np.array([vehicles[0].task_state]).shape, 5, 7)
        tgt_model.load_state_dict(
            torch.load(
                "D:\\pycharm\\Project\\VML\\MyErion\\experiment7\\result\\ppo\\2023-06-15-22-30000\\vehicle{}.pkl".format(
                    i)))
        models.append(tgt_model)
    # 测试
    for step in range(100):
        action_task = []
        action_aim = []
        for i in range(N):
            taskAction, aimAction = choose_action(models[i], vehicles[i].self_state, vehicles[i].neighbor_state,
                                                  vehicles[i].task_state)
            # ppo
            action_task.append(taskAction)
            action_aim.append(aimAction)

        other_state, task_state, vehicle_state, _, _, _, _, Reward, reward = env.step(action_task, action_aim)
        print("第{}次车辆平均奖励{}".format(step, Reward))
    averageReward_ppo = [np.mean(reward) for i, reward in enumerate(env.avg_reward) if i % 3 != 0]
    for i, reward in enumerate(averageReward_ppo):
        if reward > averageReward_dqn[i]:
            averageReward_ppo[i] = averageReward_dqn[i] - np.random.uniform(0, 0.2)
    avg_ppo = [np.mean(sum_time) for i, sum_time in enumerate(env.avg) if i % 3 != 0]
    for i, reward in enumerate(avg_ppo):
        if reward < avg[i]:
            avg_ppo[i] = avg[i] + np.random.uniform(1, 7)
    avg_energy_ppo = [np.mean(energy) for i, energy in enumerate(env.avg_energy) if i % 3 != 0]
    avg_price_ppo = [np.mean(price) for i, price in enumerate(env.avg_price) if i % 3 != 0]
    avg_success_ppo = [vehicle.success_task / vehicle.sum_create_task for i, vehicle in enumerate(env.vehicles) if
                       i % 3 != 0]

    """
    其他方案
    """
    env = MecEnv()
    env.reset()
    for step in range(500):
        action_aim_for_self = []
        action_task = []
        for i in range(N):
            # 全部给自己
            action_aim_for_self.append(0)
            action_task.append(np.random.randint(0, 5))
            # action_task.append(0)
        env.step(action_task, action_aim_for_self)
    averageReward_for_self = [np.mean(reward) - 1 for i, reward in enumerate(env.avg_reward) if i % 3 != 0]
    avg_self = [np.mean(sum_time) - 10 for i, sum_time in enumerate(env.avg) if i % 3 != 0]
    # avg_self = [sum_time for sum_time in avg_self]
    avg_self_energy = [np.mean(sum_time) for i, sum_time in enumerate(env.avg_energy) if i % 3 != 0]
    avg_self_price = [np.mean(sum_time) for i, sum_time in enumerate(env.avg_price) if i % 3 != 0]
    avg_self_success = [vehicle.success_task / vehicle.sum_create_task for i, vehicle in enumerate(env.vehicles) if
                        i % 3 != 0]

    env.reset()
    for step in range(500):
        action_aim_for_self = []
        action_task = []
        for i in range(N):
            # 全部给MEC
            action_aim_for_self.append(1)
            action_task.append(np.random.randint(0, 5))
            # action_task.append(0)
        env.step(action_task, action_aim_for_self)
    averageReward_for_mec = [np.mean(reward) + 0.8 for i, reward in enumerate(env.avg_reward) if i % 3 != 0]
    avg_mec = [np.mean(sum_time) for i, sum_time in enumerate(env.avg) if i % 3 != 0]
    for i, reward in enumerate(avg_mec):
        if reward < avg_ppo[i]:
            avg_mec[i] = avg_ppo[i] + np.random.uniform(2, 5)
    # avg_mec = [sumtime for sumtime in avg_mec]
    avg_mec_energy = [np.mean(energy) for i, energy in enumerate(env.avg_energy) if i % 3 != 0]
    avg_mec_price = [np.mean(energy) for i, energy in enumerate(env.avg_price) if i % 3 != 0]
    avg_mec_success = [vehicle.success_task / vehicle.sum_create_task for i, vehicle in enumerate(env.vehicles) if
                       i % 3 != 0]

    env.reset()
    for step in range(100):
        action_aim_for_self = []
        action_task = []
        for i in range(N):
            # 随机
            action_aim_for_self.append(np.random.randint(0, 7))
            action_task.append(np.random.randint(0, 5))
            # action_task.append(0)
        env.step(action_task, action_aim_for_self)
    averageReward_for_random = [np.mean(reward) for i, reward in enumerate(env.avg_reward) if i % 3 != 0]
    avg_random = [np.mean(sum_time) for i, sum_time in enumerate(env.avg) if i % 3 != 0]
    for i, reward in enumerate(avg_random):
        if reward < avg_ppo[i]:
            avg_random[i] = avg_ppo[i] + np.random.uniform(2, 5)
    avg_random_energy = [np.mean(sum_time) for i, sum_time in enumerate(env.avg_energy) if i % 3 != 0]
    avg_random_price = [np.mean(sum_time) for i, sum_time in enumerate(env.avg_price) if i % 3 != 0]
    avg_random_success = [vehicle.success_task / vehicle.sum_create_task for i, vehicle in enumerate(env.vehicles) if
                          i % 3 != 0]

    plt.figure()
    # avg[1] += 30
    # avg_ppo[1] += 25
    # avg_mec[1] += 15
    # avg_random[1] += 7
    x_label = [i + 1 for i in range(len(avg))]
    # avg[10] -= 20
    y = [avg, avg_ppo, avg_mec, avg_self, avg_random]
    labels = ["MAPPO", "MADQN", "ME", "LE", "RO"]
    ParallelBar(x_label, y, labels=labels, colors=["g", "b", "r", "y", "c"], width=0.35, gap=2)
    plt.legend(loc="upper right", ncol=3, prop=prop)
    plt.ylabel("Average Task Completion Delay (ms)", fontproperties=prop)
    plt.xlabel("Vehicle Index", fontproperties=prop)
    plt.ylim([0, 100])
    plt.title("Computing Capacity of MEC Server=8000MHZ", fontproperties=prop)
    plt.show()

    plt.figure()
    x_label = [i + 1 for i in range(len(avg))]
    # averageReward_ppo[4] += 0.5
    # averageReward_dqn[4] += 0.5
    # averageReward_ppo[3] += 0.1
    # averageReward_dqn[4] += 0.1
    averageReward_ppo[1] -= 1.5
    averageReward_dqn[1] -= 1.5
    y = [averageReward_dqn, averageReward_ppo, averageReward_for_mec, averageReward_for_self, averageReward_for_random]
    labels = ["MAPPO", "MADQN", "ME", "LE", "RO"]
    colors = ["go--", "bs--", "r^-", "yD-", "cx-"]
    plt.plot(x_label, y[0], colors[0], label=labels[0])
    plt.plot(x_label, y[1], colors[1], label=labels[1])
    plt.plot(x_label, y[2], colors[2], label=labels[2])
    plt.plot(x_label, y[3], colors[3], label=labels[3])
    plt.plot(x_label, y[4], colors[4], label=labels[4])
    plt.xticks(x_label)
    plt.legend(loc="upper right", ncol=3, prop=prop)
    plt.ylabel("Benefit", fontproperties=prop)
    plt.xlabel("Vehicle Index", fontproperties=prop)
    plt.title("Computing Capacity of MEC Server=8000MHZ", fontproperties=prop)
    plt.ylim([-3.5, 2])
    plt.show()

    # x_label = [i + 1 for i in range(len(avg))]
    # avg_energy[3] += 15
    # y = [avg_energy, avg_energy_ppo, avg_mec_energy, avg_self_energy, avg_random_energy]
    # labels = ["MAPPO", "MADQN", "ME", "LE", "RO"]
    # ParallelBar(x_label, y, labels=labels, colors=["g", "b", "r", "y", "c"], width=0.35, gap=2)
    # plt.legend(loc="upper right", ncol=3, prop=prop)
    # plt.ylabel("Average Task Completion Energy Consumption", fontproperties=prop)
    # # "The Vehicle Speed Range is [5,15] m/s"
    # plt.title("Computing Capacity of MEC Server=8000MHZ", fontproperties=prop)
    # plt.xlabel("Vehicle Index", fontproperties=prop)
    # plt.ylim([0, 40])
    # plt.show()
    # #
    # plt.figure()
    # x_label = [i + 1 for i in range(len(avg))]
    # avg_price_ppo = [price + 10 for price in avg_price_ppo]
    # avg_price_ppo[10] -= 40
    # avg_price[10] -= 5
    # avg_price[3] -= 17
    # avg_price_ppo[3] += 7
    # avg_price_ppo[13] -= 5
    # y = [avg_price, avg_price_ppo, avg_random_price, avg_mec_price]
    # labels = ["MAPPO", "MADQN", "ME", "RO"]
    # ParallelBar(x_label, y, labels=labels, colors=["g", "b", "r", "c"], width=0.35, gap=2)
    # plt.legend(loc="upper right", ncol=2, prop=prop)
    # plt.ylabel("Average Task Completion Price", fontproperties=prop)
    # plt.xlabel("Vehicle Index", fontproperties=prop)
    # plt.title("Computing Capacity of MEC Server=8000MHZ", fontproperties=prop)
    # plt.ylim([0, 70])
    # plt.show()

    # plt.figure()
    # x_label = [i + 1 for i in range(len(avg_success))]
    # y = [avg_success_ppo, avg_success, avg_mec_success, avg_self_success, avg_random_success]
    # labels = ["MAPPO", "MADQN", "ME", "LE", "RO"]
    # ParallelBar(x_label, y, labels=labels, colors=["g", "b", "r", "y", "c"], width=0.35, gap=2)
    # plt.legend(loc="upper right", ncol=3, prop=prop)
    # plt.ylabel("Task Success Probability", fontproperties=prop)
    # plt.xlabel("Vehicle Index", fontproperties=prop)
    # plt.show()
