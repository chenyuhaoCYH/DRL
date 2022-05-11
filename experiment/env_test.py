from env import Env

if __name__ == '__main__':
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
