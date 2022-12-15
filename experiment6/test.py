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

    # 初始化网络
    TASK_DIM = 5
    AIM_DIM = len(vehicles[0].neighbor) + 2
    vehicle_shape = len(vehicles[0].self_state)
    neighbor_shape = np.array([vehicles[0].neighbor_state]).shape
    task_shape = np.array([vehicles[0].task_state]).shape
    for i in range(N):
        tgt_model = model.ModelActor(vehicle_shape, neighbor_shape, task_shape, TASK_DIM, AIM_DIM)
        tgt_model.load_state_dict(torch.load(
            "D:\pycharm\Project\VML\MyErion\experiment6\\result\\2022-12-09-05-11\\vehicle{}.pkl".format(i)))
        models.append(tgt_model)

    # state_v = torch.tensor([vehicles[i].otherState], dtype=torch.float32)
    # taskState_v = torch.tensor([[vehicles[i].taskState]], dtype=torch.float32)
    # taskAction, aimAction = models[0](state_v, taskState_v)

    vehicleReward = []
    averageReward = []
    for step in range(10000):
        action1 = []
        action2 = []

        for i in range(N):
            state_v = torch.tensor([vehicles[i].self_state], dtype=torch.float32)
            neighbor_state_v = torch.tensor([[vehicles[i].neighbor_state]], dtype=torch.float32)
            taskState_v = torch.tensor([[vehicles[i].task_state]], dtype=torch.float32)
            taskAction, aimAction = models[i](state_v, neighbor_state_v, taskState_v, False)

            # taskAction = np.array(taskAction, dtype=np.float32).reshape(-1)
            # aimAction = np.array(aimAction, dtype=np.float32).reshape(-1)
            taskAction = taskAction.detach().numpy().reshape(-1)
            aimAction = aimAction.detach().numpy().reshape(-1)
            action1.append(np.argmax(taskAction))
            # action1.append(0)
            action2.append(np.argmax(aimAction))

        print(action1)
        print(action2)
        Reward, reward = env.step(action1, action2)
        vehicleReward.append(reward[5])
        averageReward.append(Reward)
        print("第{}次车辆平均奖励{}".format(step, Reward))

    fig, aix = plt.subplots(2, 1)
    aix[0].plot(range(len(vehicleReward)), vehicleReward)
    aix[1].plot(range(len(averageReward)), averageReward)
    plt.show()
