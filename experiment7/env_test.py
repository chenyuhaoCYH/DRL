import torch

from experiment7 import model
from env import Env
import os
import numpy as np
from mecEnv import MecEnv
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
np.random.seed(2)
if __name__ == '__main__':
    print()
    # env = Env()
    env = MecEnv()
    env.reset()

    # 测试网络节点数
    # task = np.array(env.taskState)
    # print(task.shape)
    vehicles = env.vehicles

    # for vehicle in vehicles:
    #     print("第{}车状态：{}".format(vehicle.id, vehicle.self_state))
    #     print("该车邻居:")
    #     for i in vehicle.neighbor:
    #         print(i.id, end="  ")
    #     print()

    # 测试环境运行
    reward = []
    models = []

    # task_shape = np.array([vehicles[0].task_state]).shape
    # for i in range(env.num_Vehicles):
    #     # 加载模型
    #     tgt_model = model.DQN(len(vehicles[0].self_state), task_shape, 10, len(vehicles[0].neighbor) + 2)
    #     tgt_model.load_state_dict(
    #         torch.load("D:\\pycharm\\Project\\VML\\MyErion\\experiment7\\result\\2023-05-23\\vehicle{}.pkl".format(i)))
    #     models.append(tgt_model)
    for step in range(100):
        # for j in range(20):
        #     x[j].append(env.vehicles[j].position[0])
        #     y[j].append(env.vehicles[j].position[1])
        action_task = []
        action_aim = []
        for i in range(env.num_Vehicles):
            # state_v = torch.tensor([vehicles[i].self_state], dtype=torch.float32)
            # taskState_v = torch.tensor([[vehicles[i].task_state]], dtype=torch.float32)
            # taskAction, aimAction = models[i](state_v, taskState_v)
            #
            # taskAction = taskAction.detach().numpy().reshape(-1)
            # aimAction = aimAction.detach().numpy().reshape(-1)
            # # ppo
            # action_task.append(np.argmax(taskAction))
            # action_aim.append(np.argmax(aimAction))

            # action_task.append(np.random.randint(0, 10))
            action_task.append(0)
            # action_aim.append(np.random.randint(0, 7))
            action_aim.append(0)
            # action_aim.append(1)
            # other_state, task_state, vehicle_state, _, _, _, _,
        Reward, _ = env.step(action_task, action_aim)
        # reward.append(Reward)
        # print("第{}次平均奖励{}".format(step, Reward))
        # print("当前状态:", state)
        # print("下一状态:", next_state)
        # print("车状态:", vehicleState)
        # print("任务状态", taskState)
        # print("当前奖励:", reward)
        # print("每个奖励,", vehicleReward)
        # print("当前有{}任务没有传输完成".format(len(env.need_trans_task)))
        # print("average reward:", env.Reward)
    # plt.figure()
    # fix, ax = plt.subplots(5, 4)
    #
    # for i in range(5):
    #     for j in range(4):
    #         number = i * 4 + j
    #         ax[i, j].plot(x[number], y[number])
    #         ax[i, j].set_title('vehicle {}'.format(number))
    # plt.plot(range(len(reward)), reward)
    # plt.ylabel("Reward")
    # plt.show()

    plt.figure()
    avg = [np.mean(sum_time) for i, sum_time in enumerate(env.avg) if i % 4 != 0]
    plt.ylabel("sumTime")
    plt.bar(range(len(avg)), avg, color="blue")
    plt.show()

    plt.figure()
    avg = [np.mean(sum_time) for i, sum_time in enumerate(env.avg_reward) if i % 4 != 0]
    plt.ylabel("avg_reward")
    plt.plot(range(len(avg)), avg, color="blue")
    plt.show()

    plt.figure()
    avg = [np.mean(avg_energy) for i, avg_energy in enumerate(env.avg_energy) if i % 4 != 0]
    plt.ylabel("Energy")
    plt.bar(range(len(avg)), avg, color="blue")
    plt.show()
    #
    plt.figure()
    avg = [np.mean(sum_time) for i, sum_time in enumerate(env.avg_price) if i % 4 != 0]
    plt.ylabel("Price")
    plt.bar(range(len(avg)), avg, color="blue")
    plt.show()

    plt.figure()
    avg = [vehicle.success_task / vehicle.sum_create_task for i, vehicle in enumerate(env.vehicles) if i % 4 != 0]
    plt.ylabel("successRate")
    plt.bar(range(len(avg)), avg, color="blue")
    plt.show()

    # plt.figure()
    # plt.ylabel("transTime")
    # for i, time in enumerate(env.avg_trans_time):
    #     if i % 3 != 0:
    #         plt.plot(range(0, len(time)), time)
    # plt.show()
