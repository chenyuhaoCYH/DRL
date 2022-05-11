# -*- coding: utf-8 -*-

RANGE_MEC = 200  # MEC通信范围
RESOURCE = 20   # 可用资源  GHz


# 边缘服务器
class MEC:
    def __init__(self, id, loc_x, loc_y, resources=RESOURCE):
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.loc = [self.loc_x, self.loc_y]
        self.id = id
        self.resources = resources  # 可用资源 GHz
        self.state = []
        self.range = RANGE_MEC  # 通信范围 m
        # 当前接到需要处理的任务信息
        self.accept_task3 = []
        self.accept_task1and2 = []
        # 接受任务的数量
        self.sum_needDeal_task1and2 = 0
        self.sum_needDeal_task3 = 0
        # 当前时间
        self.cur_frame = 0

    @property
    def get_x(self):
        return self.loc_x

    @property
    def get_y(self):
        return self.loc_y

    @property
    def get_location(self):
        return self.loc

    # 获得状态
    def get_state(self):
        """
        :return:state 维度：1+2+2 6维[id，loc_x,loc_y,resources,sum_needDeal_task1and2,sum_needDeal_task3]
        """
        self.state = []
        self.state.append(self.id)
        self.state.extend(self.loc)
        self.state.append(self.resources)
        self.state.append(self.sum_needDeal_task1and2)
        self.state.append(self.sum_needDeal_task3)
        return self.state

    # # 获得需要处理的任务
    # def get_task(self, action: list, vehicle: List[Vehicle]):
    #     """
    #     :param action: 所有车的动作
    #     :param vehicle: 所有车
    #     :return: 获得自己需要处理的任务信息
    #     """
    #     for i, action_i in enumerate(action):
    #         if action_i == self.id + 1 and vehicle[i].task[0] > 0:  # 这辆车的任务卸载给本MEC且不是空任务
    #             self.recevied_task.append(vehicle[i].task)  # task[1]：需要的cpu cycles/bit   task[0]：任务量的大小 kbit

    # def renew_resources(self, cur_frame):
    #     if len(self.recevied_task) > 0 and self.resources > 0:  # 如果自己有任务要处理且自己有剩余资源
    #         f = self.resources / len(self.recevied_task)  # 每次都将剩余资源拿出全部拿出(平均分配)
    #         print("f=", f)
    #         self.resources = 0  # 全部可用资源拿出去
    #         after_task = []  # 记录在此时刻还没完成的任务
    #         for i, task in enumerate(self.recevied_task):  # 遍历所有任务看是否有任务在此时刻完成
    #             if (task[1] / f) * task[0] / 1000 > cur_frame - self.cur_frame:  # 判断是否在当前能够完成此任务
    #                 task[0] -= f / task[1] * (cur_frame - self.cur_frame) * 1000
    #                 after_task.append(task)  # 剩余的任务量
    #             else:  # 能够完成  返还资源
    #                 self.resources += f
    #         self.recevied_task = after_task

    # def renew_state(self, cur_frame, action, vehicle):
    #     self.get_task(action, vehicle)
    #     self.renew_resources(cur_frame=cur_frame)
    #     self.cur_frame = cur_frame
    #     return self.get_state()


# 测试
if __name__ == '__main__':
    mec = MEC(10, 10, 1)
    # vehicles = []
    # for i in range(40):
    #     vehicle = Vehicle(i, random.randint(1, 5), random.randint(1, 5), random.randint(0, 4))
    #     vehicle.creat_work()
    #     vehicles.append(vehicle)
    # for i, vehicle in enumerate(vehicles):
    #     print("v{}.get_state():{}".format(i, vehicle.get_state()))
    # print("mec.get_state():", mec.get_state(), mec.cur_frame)
    # mec.get_task([2] * 40, vehicles)
    # print("mec.recevied_task:", mec.recevied_task)
    # print("resources:", mec.resources)
    # mec.renew_resources(1)
    # print("after recevied_task:", mec.recevied_task)
    # print("after resources:", mec.resources)
    # print("renew_state", mec.renew_state(1, [1, 2, 2], vehicles), mec.cur_frame)
    print(mec.get_location)
