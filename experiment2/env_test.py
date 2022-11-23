from env import Env
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print()
    env = Env()
    env.reset()
    # 测试找最邻近的mec
    # for vehicle in env.vehicles:
    #     print("vehicle{} location:".format(vehicle.id),vehicle.get_location)
    # for mec in env.MECs:
    #     print("mec{} location:".format(mec.id),mec.get_location)
    # for vehicle in env.vehicles:
    #     print(vehicle.mec_lest.get_location, end="  ")

    # 测试网络节点数
    task = np.array(env.taskState)
    print(task.shape)
    vehicles = env.vehicles
    # print(vehicles[0].actor1)
    # print(vehicles[0].target_actor1)
    # print(vehicles[0].state)
    # print(vehicles[0].get_state())
    # print(len(vehicles[0].state))
    # print(len(env.state))

    # 测试更新邻居表
    # for vehicle in vehicles:
    #     print(vehicle.get_location)
    #
    # print("-----------------------------------")
    # for vehicle in vehicles:
    #     for i in vehicle.neighbor:
    #         print(i.id, end=" ")
    #     print()
    # 测试更新total——task
    # list = [vehicles[0],vehicles[1],vehicles[2],vehicles[3],vehicles[4]]
    # print(list)
    # for i in reversed(list):
    #     if i.id >=2:
    #         list.remove(i)
    #     else:
    #         break
    # print(list)
    # list=[[]]*5
    # print(list)
    for vehicle in vehicles:
        print("第{}车状态：{}".format(vehicle.id, vehicle.self_state))
        print("该车邻居:")
        for i in vehicle.neighbor:
            print(i.id, end="  ")
        print()

    # 测试环境运行
    x = [[] for i in range(20)]
    y = [[] for i in range(20)]
    for i in range(10000):
        for j in range(20):
            x[j].append(env.vehicles[j].position[0])
            y[j].append(env.vehicles[j].position[1])
        action1 = []
        action2 = []
        action3 = []
        for j in range(20):
            action1.append(np.random.randint(0, 10))
            # action1.append(0)
            action2.append(np.random.randint(0, 7))
            # action2.append(0)
            # action3.append(round(np.random.random(), 2))
            action3.append(0.8)
        other_state, task_state, vehicle_state, _, _, _, Reward, _ = env.step(action1, action2)
        print("第{}次平均奖励{}".format(i, Reward))
        # print("当前状态:", state)
        # print("下一状态:", next_state)
        # print("车状态:", vehicleState)
        # print("任务状态", taskState)
        # print("当前奖励:", reward)
        # print("每个奖励,", vehicleReward)
        # print("当前有{}任务没有传输完成".format(len(env.need_trans_task)))
        # print("average reward:", env.Reward)
    plt.figure(figsize=(100, 100))
    fix, ax = plt.subplots(5, 4)

    for i in range(5):
        for j in range(4):
            number = i * 4 + j
            ax[i, j].plot(x[number], y[number])
            ax[i, j].set_title('vehicle {}'.format(number))
    plt.show()
