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

if __name__ == '__main__':
    env = Env()
    env.reset()

    N = env.num_Vehicles
    vehicles = env.vehicles
    models = []

    task_shape = np.array([vehicles[0].task_state]).shape
    for i in range(N):
        tgt_model = model.DQN(len(vehicles[0].self_state), task_shape, 10, len(vehicles[0].neighbor) + 2)
        tgt_model.load_state_dict(torch.load(
            "D:\pycharm\Project\VML\MyErion\experiment4\\result\\2022-11-26\\vehicle{}.pkl".format(i)))
        models.append(tgt_model)

    # state_v = torch.tensor([vehicles[i].otherState], dtype=torch.float32)
    # taskState_v = torch.tensor([[vehicles[i].taskState]], dtype=torch.float32)
    # taskAction, aimAction = models[0](state_v, taskState_v)

    vehicleReward = []
    averageReward = []
    for step in range(1000):
        action1 = []
        action2 = []

        for i in range(N):
            state_v = torch.tensor([vehicles[i].self_state], dtype=torch.float32)
            taskState_v = torch.tensor([[vehicles[i].task_state]], dtype=torch.float32)
            taskAction, aimAction = models[i](state_v, taskState_v)

            # taskAction = np.array(taskAction, dtype=np.float32).reshape(-1)
            # aimAction = np.array(aimAction, dtype=np.float32).reshape(-1)
            taskAction = taskAction.detach().numpy().reshape(-1)
            aimAction = aimAction.detach().numpy().reshape(-1)
            action1.append(np.argmax(taskAction))
            # action1.append(0)
            action2.append(np.argmax(aimAction))

        print(action1)
        print(action2)
        other_state, task_state, vehicle_state, _, _, _, Reward, reward = env.step(action1, action2)
        vehicleReward.append(reward[5])
        averageReward.append(Reward)
        print("第{}次车辆平均奖励{}".format(step, Reward))

    fig, aix = plt.subplots(2, 1)
    aix[0].plot(range(len(vehicleReward)), vehicleReward)
    aix[1].plot(range(len(averageReward)), averageReward)
    plt.show()

    avg = [np.mean(sum_time) for i, sum_time in enumerate(env.avg) if i % 3 != 0]
    plt.bar(range(len(avg)), avg)
    plt.title("平均时延")
    plt.ylabel("时延/ms")
    plt.show()

    avg = [np.mean(energy) for i, energy in enumerate(env.avg_energy) if i % 3 != 0]
    plt.bar(range(len(avg)), avg)
    plt.title("平均能量消耗")
    plt.ylabel("能量/J")
    plt.show()
