from env import Env
import numpy as np

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
        print("第{}车状态：{}".format(vehicle.id, vehicle.get_state()))
        print("该车邻居:")
        for i in vehicle.neighbor:
            print(i.id, end="  ")
        print()

    # 测试环境运行
    for i in range(5):
        action = []
        for j in range(20):
            action.append(np.random.randint(0, 7))
        print(len(action))
        env.step(action)
