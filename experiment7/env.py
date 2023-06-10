"""
环境7（两个动作：选择任务和选择对象）
添加了计算等待
mec和车辆都设置了一个时刻最多计算的任务（防止被分配的计算资源过少）
为mec卸载和车辆卸载提供两种传输方式（即可同时像车辆和mec传输任务）
"""
import numpy as np
from mec import MEC
from vehicle import Vehicle

MAX_TASK = 5  # 只能选前五个任务

N = 20  # 车的数量
MAX_NEIGHBOR = 5  # 最大邻居数
CAPACITY = 20000  # 缓冲池大小

sigma = -114  # 噪声dbm
POWER = 23  # 功率 dbm
POWER_V2V = 46
BrandWidth_Mec = 500  # MHz
BrandWidth_V2V = 1000  # MHz

gama = 1.25  # * (10 ** -27)  # 能量系数 J/M cycle
a = 0.6  # 奖励中时间占比
b = 0.2  # 奖励中能量占比
c = 0.2  # 奖励中支付占比
T1 = -0.5
T2 = -0.7
T3 = 0.05

# 价格系数（MEC、VEC、local）
MEC_Price = 0.15 * 1000
VEC_Price = 0.1 * 1000
LOC_Price = 0.3

Ki = -10  # 非法动惩罚项(会导致任务直接失败，所以惩罚力度大)
Kq = 0.25  # 任务队列长度系数
ko = 0.5  # 溢出任务系数
Ks = 0.5  # 奖励占比

np.random.seed(2)


class Env:
    def __init__(self, num_Vehicles=N):
        self.avg_trans_time = [[] for _ in range(N)]
        self.avg = [[] for _ in range(N)]
        self.avg_energy = [[] for _ in range(N)]
        self.avg_price = [[] for _ in range(N)]
        self.avg_successRate = [[] for _ in range(N)]
        # ################## SETTINGS ######################
        # 参数初始化：这部分直接写在代码中，没有函数，大概包括：地图属性（路口坐标，整体地图尺寸）、#车、#邻居、#RB、#episode，一些算法参数
        # 对于地图参数 up_lanes / down_lanes / left_lanes / right_lanes 的含义，首先要了解本次所用的系统模型由3GPP TR 36.885的城市案例
        # 给出，（正反方向各两个车道） ，车道宽3.5m，模型网格（road grid）的尺寸以黄线之间的距离确定
        up_lanes = [498.75, 502.25]
        down_lanes = [491.75, 495.25]
        left_lanes = [498.75, 502.25]
        right_lanes = [491.75, 495.25]

        width = 1000
        height = 1000

        self.down_lanes = down_lanes
        self.up_lanes = up_lanes
        self.left_lanes = left_lanes
        self.right_lanes = right_lanes
        self.width = width
        self.height = height

        # 基站天线高度
        self.bsHeight = 25
        # 车辆天线高度
        self.vehHeight = 1.5
        self.stdV2I = 8
        self.stdV2V = 3
        self.freq = 2
        self.vehAntGain = 3
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehNoiseFigure = 9

        # 环境内所有车辆
        self.vehicles = []
        # 环境内的mec位于最中间
        self.MEC = MEC(position=[490, 504])
        # 车辆数以及mec数
        self.num_Vehicles = num_Vehicles

        # 所有需要传输的任务
        self.need_trans_task = []
        # 当前平均奖励奖励数
        self.Reward = 0
        # 记录每辆车的历史奖励
        self.vehicleReward = []
        # 当前奖励
        self.reward = [.0] * self.num_Vehicles
        # 当前时间
        self.cur_frame = 0
        # 所有车的卸载动作
        self.offloadingActions = [0] * num_Vehicles
        # 所有车的选择任务动作
        self.taskActions = [0] * num_Vehicles
        # 当前全局的状态信息
        self.otherState = []
        # 全局任务队列信息
        self.taskState = []
        # 邻居表信息
        self.neighborState = []
        # 描述每辆车的状态
        self.vehicles_state = []
        # 系统的缓冲池
        self.buffer = []

    # 添加车辆
    def add_new_vehicles(self, id, position, direction, velocity):
        vehicle = Vehicle(id=id, position=position, direction=direction, velocity=velocity)
        vehicle.Mec = self.MEC
        self.vehicles.append(vehicle)

    # 初始化/重置环境
    def reset(self):
        self.vehicles = []
        self.MEC = MEC(position=[490, 504])
        self.Reward = 0
        self.otherState = []
        self.cur_frame = 0
        self.offloadingActions = [0] * self.num_Vehicles
        self.taskActions = [0] * self.num_Vehicles
        self.reward = [0] * self.num_Vehicles

        for i in range(self.num_Vehicles):
            self.vehicleReward.append([])

        i = 0
        while i < self.num_Vehicles:  # 初始化车子
            ind = np.random.randint(0, len(self.down_lanes))
            start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
            start_direction = 'd'  # velocity: 10 ~ 15 m/s, random
            # self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))
            self.add_new_vehicles(i, start_position, start_direction, np.random.randint(10, 15))
            start_position = [self.up_lanes[ind], np.random.randint(0, self.height)]
            start_direction = 'u'
            i += 1
            # self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))
            self.add_new_vehicles(i, start_position, start_direction, np.random.randint(10, 15))
            start_position = [np.random.randint(0, self.width), self.left_lanes[ind]]
            start_direction = 'l'
            i += 1
            # self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))
            self.add_new_vehicles(i, start_position, start_direction, np.random.randint(10, 15))
            start_position = [np.random.randint(0, self.width), self.right_lanes[ind]]
            start_direction = 'r'
            i += 1
            # self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))
            self.add_new_vehicles(i, start_position, start_direction, np.random.randint(10, 15))
            i += 1
        # 初始化邻居信息
        self.renew_neighbor()

        # 初始化状态信息
        for vehicle in self.vehicles:
            # 全局状态
            self.otherState.extend(vehicle.get_state())
            # 描述车的状态
            self.vehicles_state.append(vehicle.self_state)
            # 全局任务状态
            self.taskState.append(vehicle.task_state)
            self.neighborState.append(vehicle.neighbor_state)

        self.otherState.extend(self.MEC.state)

    # 更新每辆车邻居表
    def renew_neighbor(self):
        for i in range(len(self.vehicles)):
            self.vehicles[i].neighbor = []
        z = np.array([[complex(vehicle.get_x, vehicle.get_y) for vehicle in self.vehicles]])
        Distance = abs(z.T - z)

        for i in range(len(self.vehicles)):
            sort_idx = np.argsort(Distance[:, i])
            for j in range(MAX_NEIGHBOR):
                self.vehicles[i].neighbor.append(self.vehicles[sort_idx[j + 1]])

    # 获得卸载对象
    def get_aim(self, vehicle: Vehicle, action):
        if action == 0:
            return vehicle
        elif action == 1:
            return self.MEC
        else:
            return vehicle.neighbor[action - 2]

    def process_taskActions(self):
        """
        处理选择任务和选择卸载目标动作(包含持有)
        """
        # 同步队列
        # self.MEC.task_queue_for_reward = [task for task in self.MEC.task_queue]

        for i, vehicle in enumerate(self.vehicles):

            # 同步队列(用于等待时间计算)
            vehicle.task_queue_for_reward = [task for task in vehicle.task_queue]

            # 没有任务，无需执行卸载,也不给惩罚
            if vehicle.len_task <= 0:
                vehicle.cur_task = None
                continue

            action = self.taskActions[i]
            offloadingAction = self.offloadingActions[i]

            # 大于可选范围（默认选择第一个）
            if action >= vehicle.len_task:
                task = vehicle.total_task[0]
            # if action >= vehicle.len_task:
            #     # 非法动作 给予惩罚项 任务队列不变
            #     # print("选择了非法动作")
            #     self.reward[i] = Ki - Kq * vehicle.len_task - vehicle.overflow
            #     vehicle.cur_task = None
            #     continue
            else:
                task = vehicle.total_task[action]

            vehicle.cur_task = task
            # 去除被选择的任务
            vehicle.total_task.remove(task)
            vehicle.len_task -= 1
            # 初始化任务信息
            # 获得卸载对象
            aim = self.get_aim(vehicle, offloadingAction)

            # 目标任务等待队列已满(本地卸载)
            if len(aim.task_queue) >= aim.max_queue:
                # print("等待队列已满")
                # self.reward[i] = Ki - Kq * vehicle.len_task - vehicle.overflow
                # continue
                aim = vehicle

            task.aim = aim
            # 如果达到最高计算任务  放置等待队列中用于计算奖励
            if len(aim.accept_task) >= aim.max_task:
                # print("需要等待")
                aim.task_queue_for_reward.append(task)

            # 计算实时速率，用作奖励函数计算
            task.pick_time = self.cur_frame % 1000
            task.rate = self.compute_rate(vehicle, aim)
            if task.rate == 0:
                task.trans_time = 0
            else:
                task.trans_time = task.need_trans_size / task.rate
            self.avg_trans_time[task.vehicle.id].append(task.trans_time)
            task.compute_resource = aim.resources // (min(aim.max_task, 1 + len(aim.accept_task)))
            task.need_time = task.need_precess_cycle / task.compute_resource  # 记录任务需要计算时间(ms)
            # 卸载给本地 直接放到任务队列中
            if vehicle == aim:
                # 出队列时间（1s为一个周期）
                # task.trans_time = 0
                if len(aim.accept_task) >= aim.max_task:
                    vehicle.task_queue.append(task)
                else:
                    vehicle.accept_task.append(task)

            elif type(aim) == Vehicle:
                vehicle.queue_for_trans_vehicle.push(task)
                # 有任务在传输 放到传输等待队列中
                if vehicle.trans_task_for_vehicle == 0:
                    vehicle.trans_task_for_vehicle = 1
                    task.wait_time = 0
                else:
                    last_task = vehicle.queue_for_trans_vehicle.getLast()
                    task.wait_time = last_task.wait_time + last_task.need_trans_size / last_task.rate
                # self.reward[i] = Ki - Kq * vehicle.len_task - vehicle.overflow
            elif type(aim) == MEC:
                vehicle.queue_for_trans_mec.push(task)
                # 有任务在传输 任务失败
                if vehicle.trans_task_for_mec == 0:
                    vehicle.trans_task_for_mec = 1
                    task.wait_time = 0
                else:
                    last_task = vehicle.queue_for_trans_mec.getLast()
                    task.wait_time = last_task.wait_time + last_task.need_trans_size / last_task.rate
                # self.reward[i] = Ki - Kq * vehicle.len_task - vehicle.overflow
            # if task.wait_time > 0:
            #     print("传输需等待：{}ms".format(task.wait_time))
            # 需要传输 卸载给远程
            # else:
            #     # task.pick_time = self.cur_frame % 1000
            #     # task.aim = aim
            #     # task.rate = self.compute_rate(task.vehicle, task.aim)
            #
            #     # aim.len_action += 1
            #     # self.need_trans_task.append(task)
            #     if type(aim) == MEC:
            #         vehicle.trans_task_for_mec = 1
            #     else:
            #         vehicle.trans_task_for_vehicle = 1

    # 计算距离(车到车或者车到MEC)  aim：接受任务的目标
    @staticmethod
    def compute_distance(taskVehicle: Vehicle, aim):
        return round(np.sqrt(np.abs(taskVehicle.get_x - aim.get_x) ** 2 + np.abs(taskVehicle.get_y - aim.get_y) ** 2),
                     2)

    def generate_fading_V2I(self, dist_veh2bs):
        dist2 = (self.vehHeight - self.bsHeight) ** 2 + dist_veh2bs ** 2
        pathLoss = 128.1 + 37.6 * np.log10(np.sqrt(dist2) / 1000)  # 路损公式中距离使用km计算
        combinedPL = -(np.random.randn() * self.stdV2I + pathLoss)
        return combinedPL + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure

    def generate_fading_V2V(self, dist_DuePair):
        pathLoss = 32.4 + 20 * np.log10(dist_DuePair) + 20 * np.log10(self.freq)
        combinedPL = -(np.random.randn() * self.stdV2V + pathLoss)
        return combinedPL + self.vehAntGain * 2 - self.vehNoiseFigure

    def compute_rate(self, vehicle: Vehicle, aim):
        """
        计算实时传输速率（在一个时隙内假设不变）
        """
        # print("vehicle:{} aim:{} ".format(vehicle.id, aim.id))
        if aim == vehicle:  # 本地计算
            return 0

        distance = self.compute_distance(vehicle, aim)
        if type(aim) == MEC:
            fade = self.generate_fading_V2I(distance)
            power = np.power(10, (POWER + fade) / 10)
            sigma_w = np.power(10, sigma / 10)
            sign = power / sigma_w
            SNR = round((BrandWidth_Mec / self.num_Vehicles) * np.log2(1 + sign), 2)
        else:
            fade = self.generate_fading_V2V(distance)
            power = np.power(10, (POWER_V2V + fade) / 10)
            sigma_w = np.power(10, sigma / 10)
            sign = power / sigma_w
            SNR = (1 - np.random.randint(1, 3) / 10) * round((BrandWidth_V2V / self.num_Vehicles) * np.log2(1 + sign),
                                                             2)

        # print("第{}辆车速率:{} kb/ms".format(vehicle.id, SNR))
        return SNR / 8  # kb/ms

    @staticmethod
    def compute_energy(trans_time, aim):
        if type(aim) == MEC:
            return 0.5 * trans_time
        else:
            return trans_time

    def compute_hold_time(self):
        """
        计算任务在等待队列的等待时间
        """
        for i, vehicle in enumerate(self.vehicles):
            if len(vehicle.task_queue_for_reward) == 0:
                continue
            else:
                # 记录当前正在处理任务得所需时间(且已经从小到大排序)
                cur_need_time = []
                for task in vehicle.accept_task:
                    cur_need_time.append(task.need_time)

                for j, task in enumerate(vehicle.task_queue_for_reward):
                    if len(cur_need_time) == 0:
                        task.hold_time = 0
                        continue
                    task.hold_time = cur_need_time[0]
                    cur_need_time.pop(0)
                    # 每个任务减去最小时间
                    cur_need_time = [item - task.hold_time for item in cur_need_time]
                    # 加入当前任务
                    cur_need_time.append(task.need_time)
                    # 重新排序
                    cur_need_time.sort()
        # 对mec
        if len(self.MEC.task_queue_for_reward) > 0:
            # 记录当前正在处理任务得所需时间(且已经从小到大排序)
            cur_need_time = []
            for task in self.MEC.accept_task:
                cur_need_time.append(task.need_time)

            for j, task in enumerate(self.MEC.task_queue_for_reward):
                if len(cur_need_time) == 0:
                    task.hold_time = 0
                    continue
                task.hold_time = cur_need_time[0]
                cur_need_time.pop(0)
                # 每个任务减去最小时间
                cur_need_time = [item - task.hold_time for item in cur_need_time]
                # 加入当前任务
                cur_need_time.append(task.need_time)
                # 重新排序
                cur_need_time.sort()

    def get_reward(self, task):
        """
        计算此时任务的奖励
        """
        vehicle = task.vehicle
        aim = task.aim
        # 传输时间
        # if self.compute_distance(vehicle, aim) > aim.range:
        #     # 超出传输范围
        #     # print("选择了非法动作")
        #     reward = Ki - Kq * vehicle.len_task - vehicle.overflow
        #     return round(reward, 2)
        # else:
        trans_time = task.trans_time

        # 计算时间
        compute_time = task.need_time
        # 总时间=出队列时间-创建时间+传输时间+队列持有时间+处理时间+ 50) % 50
        #
        sum_time = trans_time + compute_time + np.abs(
            task.pick_time - task.create_time + 1000) % 1000 + task.hold_time + task.wait_time
        self.avg[vehicle.id].append(sum_time)

        # print("pick_time", task.pick_time)
        # print("create_time", task.create_time)
        # if trans_time > 10:
        #     print("传输需要{}ms".format(trans_time))
        # if task.hold_time > 0:
        #     print("等待时长为{}ms".format(task.hold_time))
        # print("计算需要{}ms".format(compute_time))
        if task.aim != vehicle:
            # 考虑通信消耗的能量（非本地卸载）
            energy = self.compute_energy(trans_time, aim)
            # print("传输需要{}ms".format(trans_time))
            # print("传输消耗{} J".format(energy))
            # 支付价格
            if type(task.aim) == MEC:
                price = MEC_Price * task.size
            else:
                price = VEC_Price * task.size
                # print("{}卸载给了邻居车".format(task.vehicle.id))
        else:
            # 计算任务消耗的能量（本地卸载）
            price = 0
            energy = round(gama * task.need_time * task.compute_resource / 1000, 2)
            # print("本地计算消耗{} J".format(energy))
        # if energy > 100:
        #     energy /= 10
        self.avg_price[vehicle.id].append(price)
        self.avg_energy[vehicle.id].append(energy)
        reward = - (a * sum_time + b * energy + c * price) / 10

        if sum_time > task.max_time:
            reward += T2 * (sum_time - task.max_time) / 10
        else:
            vehicle.success_task += 1
        return round(reward, 2)

    def compute_rewards(self):
        """
        计算此时环境所有车辆的奖励
        """
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.cur_task is not None:
                self.reward[i] = self.get_reward(vehicle.cur_task)
                vehicle.cur_task = None

    def distribute_task(self, cur_frame):
        """
        根据动作将任务添加至对应的列表当中  分配任务
        """
        for i, vehicle in enumerate(self.vehicles):
            time = cur_frame - self.cur_frame
            while not vehicle.queue_for_trans_mec.is_empty() and time > 0:
                task = vehicle.queue_for_trans_mec.peek()
                aim = task.aim
                needTime = task.need_trans_size / task.rate
                if time >= needTime:
                    time -= needTime
                    vehicle.queue_for_trans_mec.pop()
                    if len(aim.accept_task) < aim.max_task:  # 能够直接处理
                        aim.accept_task.append(task)
                    elif len(aim.task_queue) < aim.max_queue:  # 能够进入等待队列
                        aim.task_queue.append(task)
                else:
                    task.need_trans_size -= time * task.rate
                    break
            if vehicle.queue_for_trans_mec.is_empty():
                self.vehicles[vehicle.id].trans_task_for_mec = 0

            time = cur_frame - self.cur_frame
            while not vehicle.queue_for_trans_vehicle.is_empty():
                task = vehicle.queue_for_trans_vehicle.peek()
                aim = task.aim
                needTime = task.need_trans_size / task.rate
                if time >= needTime:
                    time -= needTime
                    vehicle.queue_for_trans_vehicle.pop()
                    if len(aim.accept_task) < aim.max_task:  # 能够直接处理
                        aim.accept_task.append(task)
                    elif len(aim.task_queue) < aim.max_queue:  # 能够进入等待队列
                        aim.task_queue.append(task)
                else:
                    task.need_trans_size -= time * task.rate
                    break
            if vehicle.queue_for_trans_vehicle.is_empty():
                self.vehicles[vehicle.id].trans_task_for_vehicle = 0

        # need_task = []
        # time = cur_frame - self.cur_frame
        # # 远程卸载
        # for task in self.need_trans_task:
        #     aim = task.aim
        #     vehicle = task.vehicle
        #     distance = self.compute_distance(vehicle, aim)
        #     # 超出传输范围
        #     if distance > aim.range:
        #         continue
        #     # 真实传输时间加上一时隙
        #     # task.trans_time += time
        #     # 计算实时速率
        #     rate = task.rate
        #     # 能够传输完
        #     if task.need_trans_size <= time * rate:
        #         remainTime = time - task.need_trans_size / rate
        #         # 表示当前任务能够传输完成 可以继续传输其他任务
        #         if type(task.aim) == MEC:
        #             if vehicle.queue_for_trans_mec.is_empty():
        #                 self.vehicles[vehicle.id].trans_task_for_mec = 0
        #             else:
        #                 while not vehicle.queue_for_trans_mec.is_empty():
        #                     task = vehicle.queue_for_trans_mec.pop()
        #                     remainTime -= task.need_trans_size / task.rate
        #                     if remainTime <= 0:
        #                         break
        #                 if vehicle.queue_for_trans_mec.is_empty():
        #                     self.vehicles[vehicle.id].trans_task_for_mec = 0
        #                 else:
        #                     need_task.append(vehicle.queue_for_trans_mec.pop())
        #         else:
        #             if vehicle.queue_for_trans_vehicle.is_empty():
        #                 self.vehicles[vehicle.id].trans_task_for_vehicle = 0
        #             else:
        #                 while not vehicle.queue_for_trans_vehicle.is_empty():
        #                     task = vehicle.queue_for_trans_vehicle.pop()
        #                     remainTime -= task.need_trans_size / task.rate
        #                     if remainTime <= 0:
        #                         break
        #                 if vehicle.queue_for_trans_vehicle.is_empty():
        #                     self.vehicles[vehicle.id].trans_task_for_vehicle = 0
        #                 else:
        #                     need_task.append(vehicle.queue_for_trans_mec.pop())
        #         # print("第{}车任务传输完成，真实花费{}ms".format(vehicle.id, task.trans_time))
        #         if len(aim.accept_task) < aim.max_task:  # 能够直接处理
        #             aim.accept_task.append(task)
        #         elif len(aim.task_queue) < aim.max_queue:  # 能够进入等待队列
        #             aim.task_queue.append(task)
        #         # aim.len_action -= 1
        #     else:
        #         task.need_trans_size -= time * rate
        #         need_task.append(task)
        # self.need_trans_task = need_task

    # 更新资源信息并为处理完的任务计算奖励
    def renew_resources(self, cur_frame):
        """
        更新资源信息
        """
        time = cur_frame - self.cur_frame
        # 更新车的资源信息
        for i, vehicle in enumerate(self.vehicles):
            total_task = vehicle.accept_task

            if len(total_task) > 0:  # 此时有任务并且有剩余资源
                # 记录这个时隙能够处理完的任务
                retain_task = []
                for task in total_task:
                    f = task.compute_resource
                    # 遍历此车的所有任务列表
                    precessed_time = task.need_precess_cycle / f
                    if precessed_time > time:
                        # 不能处理完
                        task.need_precess_cycle -= f * time
                        retain_task.append(task)
                vehicle.accept_task = retain_task

        # 更新mec的资源信息
        total_task = self.MEC.accept_task
        if len(total_task) > 0:
            retain_task = []
            for task in total_task:
                f = task.compute_resource
                precessed_time = task.need_precess_cycle / f
                # task.precess_time += time
                if precessed_time > time:
                    task.need_precess_cycle -= f * time
                    retain_task.append(task)
            self.MEC.accept_task = retain_task

        # 更新车需要处理的任务数量
        for vehicle in self.vehicles:
            if vehicle.max_task > len(vehicle.accept_task) and len(vehicle.task_queue) > 0:
                # 等待队列不为空且未达到最高任务量
                delta_task = vehicle.max_task - len(vehicle.accept_task)
                for i in range(min(delta_task, len(vehicle.task_queue))):
                    cur_task = vehicle.task_queue[0]
                    vehicle.task_queue.remove(cur_task)
                    vehicle.accept_task.append(cur_task)
            # 按照计算时间排序(方便计算任务得等待时间)
            vehicle.accept_task.sort(key=lambda item: item.need_time)
            vehicle.sum_needDeal_task = len(vehicle.accept_task) + len(vehicle.task_queue)

        # 处理mec
        if len(self.MEC.accept_task) < self.MEC.max_task and len(self.MEC.task_queue) > 0:
            delta_task = self.MEC.max_task - len(self.MEC.accept_task)
            # 等待任务队列数量足够多
            for i in range(min(delta_task, len(self.MEC.task_queue))):
                cur_task = self.MEC.task_queue[0]
                self.MEC.task_queue.remove(cur_task)
                self.MEC.accept_task.append(cur_task)
        self.MEC.accept_task.sort(key=lambda item: item.need_time)
        self.MEC.sum_needDeal_task = len(self.MEC.accept_task) + len(self.MEC.task_queue)

        # 分配任务信息（在计算之后执行是因为将一个时隙看作为原子操作，因此这个时隙接受到的任务不能进行计算）
        self.distribute_task(cur_frame=cur_frame)

    # 更新车辆位置：renew_position(无)，遍历每辆车，根据其方向和速度更新位置，
    def renew_positions(self, cur_frame):
        time = cur_frame - self.cur_frame  # ms
        i = 0
        while i < len(self.vehicles):
            delta_distance = self.vehicles[i].velocity * time / 1000
            change_direction = False
            # 向上
            if self.vehicles[i].direction == 'u':
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.vehicles[i].position[1] <= self.left_lanes[j]) and (
                            (self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):  # came to a cross
                        if np.random.uniform(0, 1) < 0.4:
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (
                                    delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])),
                                                         self.left_lanes[j]]
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if not change_direction:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] <= self.right_lanes[j]) and (
                                (self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if np.random.uniform(0, 1) < 0.4:
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (
                                        delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])),
                                                             self.right_lanes[j]]
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if not change_direction:
                    self.vehicles[i].position[1] += delta_distance
            elif (self.vehicles[i].direction == 'd') and (not change_direction):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.vehicles[i].position[1] >= self.left_lanes[j]) and (
                            (self.vehicles[i].position[1] - delta_distance) <= self.left_lanes[j]):  # came to a cross
                        if np.random.uniform(0, 1) < 0.4:
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (
                                    delta_distance - (self.vehicles[i].position[1] - self.left_lanes[j])),
                                                         self.left_lanes[j]]
                            # print ('down with left', self.vehicles[i].position)
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if not change_direction:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] >= self.right_lanes[j]) and (
                                self.vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                            if np.random.uniform(0, 1) < 0.4:
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (
                                        delta_distance + (self.vehicles[i].position[1] - self.right_lanes[j])),
                                                             self.right_lanes[j]]
                                # print ('down with right', self.vehicles[i].position)
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if not change_direction:
                    self.vehicles[i].position[1] -= delta_distance
            elif (self.vehicles[i].direction == 'r') and (not change_direction):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].position[0] <= self.up_lanes[j]) and (
                            (self.vehicles[i].position[0] + delta_distance) >= self.up_lanes[j]):  # came to a cross
                        if np.random.uniform(0, 1) < 0.4:
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (
                                    delta_distance - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if not change_direction:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] <= self.down_lanes[j]) and (
                                (self.vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                            if np.random.uniform(0, 1) < 0.4:
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (
                                        delta_distance - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                if not change_direction:
                    self.vehicles[i].position[0] += delta_distance
            elif (self.vehicles[i].direction == 'l') and (not change_direction):
                for j in range(len(self.up_lanes)):

                    if (self.vehicles[i].position[0] >= self.up_lanes[j]) and (
                            (self.vehicles[i].position[0] - delta_distance) <= self.up_lanes[j]):  # came to a cross
                        if np.random.uniform(0, 1) < 0.4:
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (
                                    delta_distance - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if not change_direction:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] >= self.down_lanes[j]) and (
                                (self.vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                            if np.random.uniform(0, 1) < 0.4:
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (
                                        delta_distance - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                    if not change_direction:
                        self.vehicles[i].position[0] -= delta_distance

            # if it comes to an exit
            elif (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or (
                    self.vehicles[i].position[0] > self.width) or (self.vehicles[i].position[1] > self.height):
                # delete
                #    print ('delete ', self.position[i])
                if self.vehicles[i].direction == 'u':
                    self.vehicles[i].direction = 'r'
                    self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[-1]]
                else:
                    if self.vehicles[i].direction == 'd':
                        self.vehicles[i].direction = 'l'
                        self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[0]]
                    else:
                        if self.vehicles[i].direction == 'l':
                            self.vehicles[i].direction = 'u'
                            self.vehicles[i].position = [self.up_lanes[0], self.vehicles[i].position[1]]
                        else:
                            if self.vehicles[i].direction == 'r':
                                self.vehicles[i].direction = 'd'
                                self.vehicles[i].position = [self.down_lanes[-1], self.vehicles[i].position[1]]

            i += 1

    def renew_state(self, cur_frame):
        """
        更新状态
        """
        self.otherState = []
        self.taskState = []
        self.vehicles_state = []
        self.neighborState = []

        # 更新车位置信息
        self.renew_positions(cur_frame)
        # 更新邻居表
        self.renew_neighbor()
        for vehicle in self.vehicles:
            # 更新时间
            vehicle.cur_frame = cur_frame
            # 产生任务
            vehicle.create_work()
            # 更新资源已经接受的任务信息
            self.otherState.extend(vehicle.get_state())
            self.taskState.append(vehicle.task_state)
            self.neighborState.append(vehicle.neighbor_state)
            self.vehicles_state.append(vehicle.self_state)
            # 更新任务信息
            vehicle.cur_task = None

        self.otherState.extend(self.MEC.get_state())

    # 执行动作
    def step(self, taskActions, offloadingActions):
        cur_frame = self.cur_frame + 10  # ms
        # 分配动作
        # 卸载动作
        self.offloadingActions = offloadingActions
        # 任务选择动作
        self.taskActions = taskActions

        # 重置奖励
        self.reward = [0] * self.num_Vehicles

        # 处理选取任务动作
        self.process_taskActions()

        # 计算等待时间
        self.compute_hold_time()

        # 计算奖励
        self.compute_rewards()

        # 记录当前状态
        other_state = self.otherState
        task_state = self.taskState
        vehicle_state = self.vehicles_state
        # neighbor_state = self.neighborState

        # 更新资源信息以及车辆任务信息
        self.renew_resources(cur_frame)

        # 更新状态
        self.renew_state(cur_frame)

        # 更新时间
        self.cur_frame = cur_frame
        # print("当前有{}个任务没传输完成".format(len(self.need_trans_task)))

        # 平均奖励
        self.Reward = np.mean([reward for i, reward in enumerate(self.reward) if i % 4 != 0])

        return other_state, task_state, vehicle_state, self.vehicles_state, self.otherState, self.taskState, self.neighborState, self.Reward, self.reward
