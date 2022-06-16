# -*- coding: utf-8 -*-

RANGE_MEC = 200  # MEC通信范围
RESOURCE = 20000  # 可用资源  MHz


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
        self.accept_task = []
        # 接受任务的数量
        self.sum_needDeal_task = 0
        # 此时刻有多少动作选则我 多少任务正在传输给我
        self.len_action = 0
        # 当前时间
        self.cur_frame = 0
        # 当前状态
        self.get_state()

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
        self.state.extend(self.loc)
        self.state.append(self.resources)
        self.state.append(self.sum_needDeal_task)
        self.state.append(self.len_action)
        return self.state


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
