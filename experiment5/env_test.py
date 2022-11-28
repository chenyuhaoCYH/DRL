from env import Env
import numpy as np
import matplotlib.pyplot as plt

# if __name__ == '__main__':
#     print()
#     env = Env()
#     env.reset()
#
#     # 测试网络节点数
#     task = np.array(env.taskState)
#     print(task.shape)
#     vehicles = env.vehicles
#
#     for vehicle in vehicles:
#         print("第{}车状态：{}".format(vehicle.id, vehicle.self_state))
#         print("该车邻居:")
#         for i in vehicle.neighbor:
#             print(i.id, end="  ")
#         print()
#
#     # 测试环境运行
#     reward = []
#     x = [[] for i in range(20)]
#     y = [[] for i in range(20)]
#     for i in range(1000):
#         # for j in range(20):
#         #     x[j].append(env.vehicles[j].position[0])
#         #     y[j].append(env.vehicles[j].position[1])
#         action1 = []
#         action2 = []
#         for j in range(40):
#             # action1.append(np.random.randint(0, 10))
#             action1.append(0)
#             # action2.append(np.random.randint(0, 7))
#             action2.append(1)
#         other_state, task_state, vehicle_state, _, _, _, Reward, _ = env.step(action1, action2)
#         reward.append(Reward)
#         print("第{}次平均奖励{}".format(i, Reward))
#         # print("当前状态:", state)
#         # print("下一状态:", next_state)
#         # print("车状态:", vehicleState)
#         # print("任务状态", taskState)
#         # print("当前奖励:", reward)
#         # print("每个奖励,", vehicleReward)
#         # print("当前有{}任务没有传输完成".format(len(env.need_trans_task)))
#         # print("average reward:", env.Reward)
#     # plt.figure(figsize=(100, 100))
#     # fix, ax = plt.subplots(5, 4)
#     #
#     # for i in range(5):
#     #     for j in range(4):
#     #         number = i * 4 + j
#     #         ax[i, j].plot(x[number], y[number])
#     #         ax[i, j].set_title('vehicle {}'.format(number))
#     plt.plot(range(len(reward)), reward)
#     print(reward)
#     plt.show()

if __name__ == '__main__':
    print()
    env = Env()
    env.reset()

    # 测试网络节点数
    vehicles = env.vehicles

    for vehicle in vehicles:
        print("第{}车状态：{}".format(vehicle.id, vehicle.self_state))
        print("该车邻居:")
        for i in vehicle.neighbor:
            print(i.id, end="  ")
        print()

    # 测试环境运行
    reward = []
    x = [[] for i in range(20)]
    y = [[] for i in range(20)]
    for i in range(1000):
        # for j in range(20):
        #     x[j].append(env.vehicles[j].position[0])
        #     y[j].append(env.vehicles[j].position[1])
        action1 = []
        action2 = []
        for j in range(20):
            # action1.append(np.random.randint(0, 10))
            action1.append(0)
            # action2.append(np.random.randint(0, 7))
            action2.append(1)
        Reward, _ = env.step(action1, action2)
        reward.append(Reward)
        print("第{}次平均奖励{}".format(i, Reward))
