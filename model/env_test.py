from env import Env

if __name__ == '__main__':
    env = Env()
    env.reset()
    # 测试状态，初始化过程
    # for vehicle in env.vehicles:
    #     print([vehicle.state])
    #     print(vehicle.cur_network)
    #     print(env.MECs[0].get_state())
    #     break
    # print(len(env.state))
    # print(len(env.actions))
    # env.get_action()
    # env.get_aim()
    # env.distribute_task()
    # print("all actions:", env.actions)
    # 测试动作选择及任务分配
    # print(len(env.actions))
    # for i, vehicle in enumerate(env.vehicles):
    #     aim = env.get_aim(id=i, action=env.actions[i])
    #     print(aim.id, end=" ")
    #     print(type(aim))
    # aim = env.get_aim(0, env.actions[0])
    # print("task:", env.vehicles[0].task)
    # print("distance:", env.compute_distance(env.vehicles[0], aim))
    # print(env.compute_transmit(env.vehicles[0], aim))
    # print(aim.id)
    # print(aim.state)
    # for i, action in enumerate(env.actions):
    #     aim = env.get_aim(i, action)
    #     print(env.compute_distance(env.vehicles[i], aim), "m")
    # env.distribute_task()
    # sum = 0
    # for mec in env.MECs:
    #     print(mec.recevied_task)
    #     sum += len(mec.recevied_task)
    # for vehicle in env.vehicles:
    #     print(vehicle.recevied_task)
    #     sum += len(vehicle.recevied_task)
    #
    # print("total task:", sum)

    # 测试距离计算及传输开销
    # vehicle = env.vehicles[0]
    # action = env.actions[0]
    # print("task:", vehicle.task)
    # print("self loc:", vehicle.get_location)
    # env.get_aim()
    # print("all aim:", len(env.aim))
    # aim = env.aim[0]
    # print("aim type and id:", type(aim), aim.id)
    # print("aim loc:", aim.get_location)
    # distance = env.compute_distance(vehicle, aim)
    # print("distance:", distance, "m")
    #
    # print("transport time:", env.compute_transmit(vehicle, aim), "s")
    #
    # env.distribute_task()
    # aim = env.aim[0]
    # print("total task:", aim.recevied_task)
    # # 测试处理时间
    # print("precessed time:", env.compute_precessed(vehicle, aim), "s")

    # 测试获得平均奖励
    # 测试更新状态
    # print("raw state:", env.state)
    # env.get_averageReward()
    # for vehicle in env.vehicles:
    #     print("{}vehicle recevied_task:".format(vehicle.id), vehicle.recevied_task)
    #     print("{}vehicle resources".format(vehicle.id), vehicle.resources)
    # for mec in env.MECs:
    #     print("{}mec recevied_task".format(mec.id), mec.recevied_task)
    #     print("{}mec resources".format(mec.id), mec.resources)
    # print(env.cur_frame)
    # env.step()
    # print("after state", env.state)
    # for vehicle in env.vehicles:
    #     print("{}recevied_task:".format(vehicle.id), vehicle.recevied_task)
    #     print("{}vehicle resources".format(vehicle.id), vehicle.resources)
    # for mec in env.MECs:
    #     print("{}mec recevied_task".format(mec.id), mec.recevied_task)
    #     print("{}mec resources".format(mec.id), mec.resources)
    # print(env.cur_frame)
    for i in range(1000):
        env.step()
