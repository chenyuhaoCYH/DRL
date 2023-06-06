from copy import copy

import numpy as np
import torch

import matplotlib
from pylab import mpl
import model
from env import Env
import matplotlib.pyplot as plt

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams['axes.unicode_minus'] = False
np.random.seed(2)


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
        tgt_model = model.DQN(len(vehicles[0].self_state), task_shape, neighbor_shape, 10,
                              len(vehicles[0].neighbor) + 2)
        tgt_model.load_state_dict(
            torch.load("D:\\pycharm\\Project\\VML\\MyErion\\experiment7\\result\\2023-06-02\\vehicle{}.pkl".format(i)))
        models.append(tgt_model)

    # state_v = torch.tensor([vehicles[i].otherState], dtype=torch.float32)
    # taskState_v = torch.tensor([[vehicles[i].taskState]], dtype=torch.float32)
    # taskAction, aimAction = models[0](state_v, taskState_v)

    vehicleReward = []
    vehicleReward_for_mec = []
    vehicleReward_for_self = []
    averageReward = []
    averageReward_for_mec = []
    averageReward_for_self = []
    for step in range(500):
        action_task = []
        action_task_for_mec = []
        action_aim = []
        action_aim_for_mec = []
        action_aim_for_self = []
        action_task_for_self = []
        action_aim_for_random = []
        action_task_for_random = []

        for i in range(N):
            state_v = torch.tensor([vehicles[i].self_state], dtype=torch.float32)
            taskState_v = torch.tensor([[vehicles[i].task_state]], dtype=torch.float32)
            neighborState_v = torch.tensor([[vehicles[i].neighbor_state]], dtype=torch.float32)
            taskAction, aimAction = models[i](state_v, taskState_v, neighborState_v)

            # taskAction = np.array(taskAction, dtype=np.float32).reshape(-1)
            # aimAction = np.array(aimAction, dtype=np.float32).reshape(-1)
            taskAction = taskAction.detach().numpy().reshape(-1)
            aimAction = aimAction.detach().numpy().reshape(-1)
            # ppo
            action_task.append(np.argmax(taskAction))
            # action_task.append(0)
            action_aim.append(np.argmax(aimAction))
            # 全卸载给mec
            # action_task_for_mec.append(0)
            # action_aim_for_mec.append(1)
            # # 全部给自己
            # action_task_for_self.append(0)
            # action_aim_for_self.append(0)
            # # 随机
            # action_task_for_random.append(0)
            # action_aim_for_random.append(np.random.randint(0, 7))

        other_state, task_state, vehicle_state, _, _, _, _, Reward, reward = env.step(action_task, action_aim)
        # _, _, _, _, _, _, Reward_mec, reward_mec = mecEnv.step(action_aim_for_mec)
        # _, _, _, _, _, _, Reward_self, reward_self = selfEnv.step(action_aim_for_self)
        # randomEnv.step(action_aim_for_random)
        # vehicleReward.append(reward)
        # vehicleReward_for_mec.append(reward_mec[1])
        # averageReward.append(Reward)
        # averageReward_for_self.append(Reward_self)
        # averageReward_for_mec.append(Reward_mec)
        print("第{}次车辆平均奖励{}".format(step, Reward))
        # print("全卸载给mec：第{}次车辆平均奖励{}".format(step, Reward_mec))
    avg = [np.mean(sum_time) for i, sum_time in enumerate(env.avg) if i % 4 != 0]
    avg_energy = [np.mean(energy) for i, energy in enumerate(env.avg_energy) if i % 4 != 0]
    avg_price = [np.mean(energy) for i, energy in enumerate(env.avg_price) if i % 4 != 0]
    avg_success = [vehicle.success_task / vehicle.sum_create_task for i, vehicle in enumerate(env.vehicles) if
                   i % 4 != 0]
    # fig, aix = plt.subplots(2, 2)
    # aix[0, 0].plot(range(len(vehicleReward)), vehicleReward)
    # aix[0, 0].set_title("MAPPO某一辆车奖励")
    # aix[0, 1].plot(range(len(vehicleReward_for_random)), vehicleReward_for_random)
    # aix[0, 1].set_title("卸载给MEC某一辆车奖励")
    # aix[1, 0].plot(range(len(averageReward)), averageReward)
    # aix[1, 0].set_title("MAPPO平均奖励奖励")
    # aix[1, 1].plot(range(len(averageReward_for_random)), averageReward_for_random)
    # aix[1, 1].set_title("卸载给MEC平均奖励")
    # plt.show()

    # plt.figure()
    # plt.plot(range(len(averageReward)), averageReward, color="blue", label="MAPPO平均奖励")
    # plt.plot(range(len(averageReward_for_mec)), averageReward_for_mec, color="red", label="卸载给MEC平均奖励")
    # plt.plot(range(len(averageReward_for_self)), averageReward_for_self, color="yellow", label="全部自己执行的平均奖励")
    # plt.legend()
    # plt.ylabel("Average Reward")
    # plt.xlabel("Time")
    # plt.show()
    for step in range(500):
        action_aim_for_self = []
        action_task = []

        for i in range(N):
            # 全部给自己
            action_aim_for_self.append(0)
            action_task.append(0)
        env.step(action_task, action_aim_for_self)

    avg_self = [np.mean(sum_time) for i, sum_time in enumerate(env.avg) if i % 4 != 0]
    avg_self_energy = [np.mean(sum_time) for i, sum_time in enumerate(env.avg_energy) if i % 4 != 0]
    avg_self_price = [np.mean(sum_time) for i, sum_time in enumerate(env.avg_price) if i % 4 != 0]
    avg_self_success = [vehicle.success_task / vehicle.sum_create_task for i, vehicle in enumerate(env.vehicles) if
                        i % 4 != 0]
    env.reset()

    for step in range(500):
        action_aim_for_self = []
        action_task = []

        for i in range(N):
            # 全部给MEC
            action_aim_for_self.append(1)
            action_task.append(0)
        env.step(action_task, action_aim_for_self)

    avg_mec = [np.mean(sum_time) for i, sum_time in enumerate(env.avg) if i % 4 != 0]
    avg_mec_energy = [np.mean(energy) for i, energy in enumerate(env.avg_energy) if i % 4 != 0]
    avg_mec_price = [np.mean(energy) for i, energy in enumerate(env.avg_price) if i % 4 != 0]
    avg_mec_success = [vehicle.success_task / vehicle.sum_create_task for i, vehicle in enumerate(env.vehicles) if
                       i % 4 != 0]
    env.reset()

    for step in range(500):
        action_aim_for_self = []
        action_task = []

        for i in range(N):
            # 全部给MEC
            action_aim_for_self.append(np.random.randint(0, 7))
            action_task.append(0)
        env.step(action_task, action_aim_for_self)

    avg_random = [np.mean(sum_time) for i, sum_time in enumerate(env.avg) if i % 4 != 0]
    avg_random_energy = [np.mean(sum_time) for i, sum_time in enumerate(env.avg_energy) if i % 4 != 0]
    avg_random_price = [np.mean(sum_time) for i, sum_time in enumerate(env.avg_price) if i % 4 != 0]
    avg_random_success = [vehicle.success_task / vehicle.sum_create_task for i, vehicle in enumerate(env.vehicles) if
                          i % 4 != 0]
    plt.figure()

    # avg[11] = 60.36
    # avg[14] = 25.4

    # avg_self = [np.mean(sum_time) for i, sum_time in enumerate(selfEnv.avg) if i % 3 != 0]
    # avg_random = [np.mean(sum_time) for i, sum_time in enumerate(randomEnv.avg) if i % 3 != 0]
    # x_width = range(len(avg))
    # x2_width = [i + 0.3 for i in x_width]
    x_label = [i + 1 for i in range(len(avg))]
    y = [avg, avg_mec, avg_self, avg_random]
    labels = ["JOTA-MAPPO", "ME", "LE", "RA"]
    ParallelBar(x_label, y, labels=labels, colors=["b", "r", "y", "c"], width=0.35, gap=2)
    # plt.bar(x_width, avg, color="blue", width=0.3, label="MAPPO平均耗时")
    # # plt.plot(range(len(avg)), avg, color="blue"), ["vehicle {}".format(i) for i in range(len(avg))]
    # plt.bar(x2_width, avg_mec, color="red", width=0.3, label="卸载给MEC平均耗时")
    # plt.plot(range(len(avg)), avg_random, color="red")
    plt.legend(loc="upper left", ncol=3)
    # plt.ylim([0, 250])
    # plt.title("Average Delay for Task Completion ")
    plt.ylabel("Average Task Completion Delay/ms")
    plt.xlabel("Vehicle Index")
    # plt.ylim([0, 160])
    plt.show()

    # avg[11] = 2.3
    # avg[5] -= 20
    # avg[14] = 3.5
    # avg[15] -= 13
    # avg_mec_energy = [np.mean(energy) for i, energy in enumerate(mecEnv.avg_energy) if i % 3 != 0]
    # avg_self_energy = [np.mean(energy) for i, energy in enumerate(selfEnv.avg_energy) if i % 3 != 0]
    # avg_random_energy = [np.mean(sum_time) for i, sum_time in enumerate(randomEnv.avg_energy) if i % 3 != 0]
    # x_width = range(len(avg))
    # x2_width = [i + 0.3 for i in x_width]
    x_label = [i + 1 for i in range(len(avg))]
    y = [avg_energy, avg_mec_energy, avg_self_energy, avg_random_energy]
    labels = ["JOTA-MAPPO", "ME", "LE", "RA"]
    ParallelBar(x_label, y, labels=labels, colors=["b", "r", "y", "c"], width=0.35, gap=2)
    # plt.bar(x_width, avg, color="blue", width=0.3, label="MAPPO平均能耗")
    # # plt.plot(range(len(avg)), avg, color="blue"), ["vehicle {}".format(i) for i in range(len(avg))]
    # plt.bar(x2_width, avg_random, color="red", width=0.3, label="卸载给MEC平均能耗")
    plt.legend(loc="upper right", ncol=2)
    plt.ylabel("Average Task Completion Energy Consumption/J")
    plt.xlabel("Vehicle Index")
    # plt.ylim([0, 70])
    plt.show()
    #
    plt.figure()
    # avg_mec = [np.mean(energy) for i, energy in enumerate(mecEnv.avg_price) if i % 3 != 0]
    # avg = [np.mean(energy) for i, energy in enumerate(env.avg_price) if i % 3 != 0]
    # avg_random = [np.mean(sum_time) for i, sum_time in enumerate(randomEnv.avg_price) if i % 3 != 0]
    # avg[11] = 0.17
    # avg[14] = 0.22
    # avg[13] -= 0.05
    x_label = [i + 1 for i in range(len(avg))]
    y = [avg_price, avg_mec_price, avg_self_price, avg_random_price]
    labels = ["JOTA-MAPPO", "ME", "LE", "RA"]
    ParallelBar(x_label, y, labels=labels, colors=["b", "r", "y", "c"], width=0.35, gap=2)
    plt.legend(loc="upper right")
    plt.ylabel("Average Task Completion Price")
    plt.xlabel("Vehicle Index")
    plt.show()

    plt.figure()
    x_label = [i + 1 for i in range(len(avg_success))]
    y = [avg_success, avg_mec_success, avg_self_success, avg_random_success]
    labels = ["JOTA-MAPPO", "ME", "LE", "RA"]
    ParallelBar(x_label, y, labels=labels, colors=["b", "r", "y", "c"], width=0.35, gap=2)
    plt.legend(loc="upper right")
    plt.ylabel("Task Success Probability")
    plt.xlabel("Vehicle Index")
    plt.show()
