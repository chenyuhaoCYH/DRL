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
np.random.seed(0)

if __name__ == '__main__':
    # 我的环境
    env = Env()
    env.reset()
    # 只卸载给mec
    mecEnv = Env()
    mecEnv.reset()
    # 只自己执行
    selfEnv = Env()
    selfEnv.reset()

    N = env.num_Vehicles
    vehicles = env.vehicles
    models = []

    task_shape = np.array([vehicles[0].task_state]).shape
    for i in range(N):
        # 加载模型
        tgt_model = model.DQN(len(vehicles[0].self_state), task_shape, 10, len(vehicles[0].neighbor) + 2)
        tgt_model.load_state_dict(
            torch.load("D:\\pycharm\\Project\\VML\\MyErion\\experiment4\\result\\2023-04-13\\vehicle{}.pkl".format(i)))
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
    for step in range(100):
        action_task = []
        action_task_for_mec = []
        action_aim = []
        action_aim_for_mec = []
        action_aim_for_self = []
        action_task_for_self = []

        for i in range(N):
            state_v = torch.tensor([vehicles[i].self_state], dtype=torch.float32)
            taskState_v = torch.tensor([[vehicles[i].task_state]], dtype=torch.float32)
            taskAction, aimAction = models[i](state_v, taskState_v)

            # taskAction = np.array(taskAction, dtype=np.float32).reshape(-1)
            # aimAction = np.array(aimAction, dtype=np.float32).reshape(-1)
            taskAction = taskAction.detach().numpy().reshape(-1)
            aimAction = aimAction.detach().numpy().reshape(-1)
            # ppo
            action_task.append(np.argmax(taskAction))
            # action_task.append(0)
            action_aim.append(np.argmax(aimAction))
            # 全卸载给mec
            action_task_for_mec.append(0)
            action_aim_for_mec.append(1)
            # 全部给自己
            action_task_for_self.append(0)
            action_aim_for_self.append(0)

        print(action_task)
        print(action_aim)
        other_state, task_state, vehicle_state, _, _, _, Reward, reward = env.step(action_task, action_aim)
        _, _, _, _, _, _, Reward_mec, reward_mec = mecEnv.step(action_task_for_mec, action_aim_for_mec)
        _, _, _, _, _, _, Reward_self, reward_self = selfEnv.step(action_task_for_self, action_aim_for_self)
        vehicleReward.append(reward)
        vehicleReward_for_mec.append(reward_mec[1])
        averageReward.append(Reward)
        averageReward_for_self.append(Reward_self)
        averageReward_for_mec.append(Reward_mec)
        print("第{}次车辆平均奖励{}".format(step, Reward))
        print("全卸载给mec：第{}次车辆平均奖励{}".format(step, Reward_mec))

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
    # plt.title("平均奖励")
    # plt.ylabel("奖励")
    # plt.show()

    plt.figure()
    avg = [np.mean(sum_time) for i, sum_time in enumerate(env.avg) if i % 3 != 0]
    avg_random = [np.mean(sum_time) for i, sum_time in enumerate(mecEnv.avg) if i % 3 != 0]
    x_width = range(len(avg))
    x2_width = [i + 0.3 for i in x_width]
    plt.bar(x_width, avg, color="blue", width=0.3, label="MAPPO平均耗时")
    # plt.plot(range(len(avg)), avg, color="blue"), ["vehicle {}".format(i) for i in range(len(avg))]
    plt.bar(x2_width, avg_random, color="red", width=0.3, label="卸载给MEC平均耗时")
    # plt.plot(range(len(avg)), avg_random, color="red")
    plt.legend()
    plt.title("平均时延")
    plt.ylabel("时延/ms")
    plt.show()

    # plt.figure()
    # avg = [np.mean(energy) for i, energy in enumerate(env.avg_energy) if i % 3 != 0]
    # avg_random = [np.mean(energy) for i, energy in enumerate(mecEnv.avg_energy) if i % 3 != 0]
    # plt.scatter(range(len(avg)), avg, color="blue", label="MAPPO平均能耗")
    # plt.plot(range(len(avg)), avg, color="blue")
    # plt.scatter(range(len(avg)), avg_random, color="red", label="卸载给MEC平均能耗")
    # plt.plot(range(len(avg)), avg_random, color="red")
    # plt.legend(loc="upper right")
    # plt.title("平均能量消耗")
    # plt.ylabel("能量/J")
    # plt.show()
    #
    # plt.figure()
    # avg = [np.mean(energy) for i, energy in enumerate(env.avg_price) if i % 3 != 0]
    # avg_random = [np.mean(energy) for i, energy in enumerate(mecEnv.avg_price) if i % 3 != 0]
    # plt.scatter(range(len(avg)), avg, color="blue", label="MAPPO平均支付价格")
    # plt.plot(range(len(avg)), avg, color="blue")
    # plt.scatter(range(len(avg)), avg_random, color="red", label="卸载给MEC平均支付价格")
    # plt.plot(range(len(avg)), avg_random, color="red")
    # plt.legend(loc="upper right")
    # plt.title("平均支付价格")
    # plt.ylabel("价格")
    # plt.show()
