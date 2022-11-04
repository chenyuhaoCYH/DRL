"""
环境1（三个动作：选择任务和选择对象，以及选择分配资源）
"""
import sys
from random import randint
import numpy as np
from mec import MEC
from experiment.vehicle import Vehicle

MAX_TASK = 5  # 只能选前五个任务
y = [2, 6, 10, 14]  # 车子y的坐标集 # 共四条车道
directions = [1, 1, -1, -1]  # 车子的方向
MEC_loc = [[-800, 0], [-400, 16], [0, 0], [400, 16], [800, 0]]  # mec的位置

HOLD_TIME = [5, 10, 20, 30]  # 维持时间

N = 20  # 车的数量
K = 5  # mec的数量
MAX_NEIGHBOR = 5  # 最大邻居数
CAPACITY = 20000  # 缓冲池大小

sigma = -114  # 噪声dbm
POWER = 23  # 功率 dbm
BrandWidth_Mec = 100  # MHz

gama = 1.25 * (10 ** -11)  # 能量系数 J/M cycle
a = 0.6  # 奖励中时间占比
b = 0.4  # 奖励中能量占比
T1 = -0.5
T2 = -0.2
T3 = 0.05

# 价格系数（MEC、VEC、local）
MEC_Price = 0.6
VEC_Price = 0.4
LOC_Price = 0.3

Ki = -4  # 非法动惩罚项(会导致任务直接失败，所以惩罚力度大)
Kq = 0.25  # 任务队列长度系数
Ks = 0.5  # 奖励占比

np.random.seed(2)


class Env:
    def __init__(self, num_Vehicles=N, num_MECs=K):
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
        # 环境内所有mec
        self.MECs = []
        # 车辆数以及mec数
        self.num_Vehicles = num_Vehicles
        self.num_MECs = num_MECs
        # 所有需要传输的任务
        self.need_trans_task = []
        # 当前平均奖励奖励数
        self.Reward = 0
        # 记录每辆车的历史奖励
        self.vehicleReward = []
        # 当前奖励
        self.reward = [0] * self.num_Vehicles
        # 当前时间
        self.cur_frame = 0
        # 所有车的卸载动作
        self.offloadingActions = [0] * num_Vehicles
        # 所有车的选择任务动作
        self.taskActions = [0] * num_Vehicles
        # 所有车是否等待动作
        self.holdActions = [0] * num_Vehicles
        # 所有车分配的资源比率
        self.computeRatioActions = [0] * num_Vehicles
        # 所有车要传输的对象 vehicle or mec
        self.aims = []
        # 当前全局的状态信息  维度:
        self.otherState = []
        # 全局任务队列信息
        self.taskState = []
        # 描述每辆车的状态
        self.vehicles_state = []
        # 系统的缓冲池
        self.buffer = []

    # 添加车辆
    def add_new_vehicles(self, id, loc_x, loc_y, direction, velocity):
        vehicle = Vehicle(id=id, loc_x=loc_x, loc_y=loc_y, direction=direction, velocity=velocity)
        self.vehicles.append(vehicle)

    # 初始化/重置环境
    def reset(self):
        self.vehicles = []
        self.MECs = []
        self.Reward = 0
        self.otherState = []
        self.cur_frame = 0
        self.offloadingActions = [0] * self.num_Vehicles
        self.taskActions = [0] * self.num_Vehicles
        self.reward = [0] * self.num_Vehicles

        for i in range(self.num_Vehicles):
            self.vehicleReward.append([])

        for i in range(0, self.num_MECs):  # 初始化mec
            cur_mec = MEC(id=i, loc_x=MEC_loc[i][0], loc_y=MEC_loc[i][1])
            self.MECs.append(cur_mec)

        i = 0
        while i < self.num_Vehicles:  # 初始化车子
            n = np.random.randint(0, 4)  # 左闭右开
            self.add_new_vehicles(id=i, loc_x=randint(-1000, 1000), loc_y=y[n], direction=directions[n],
                                  velocity=np.random.randint(17, 35))
            i += 1
        # 初始化邻居信息
        self.renew_neighbor()
        self.renew_neighbor_mec()

        # 初始化状态信息
        for vehicle in self.vehicles:
            # 全局状态
            self.otherState.extend(vehicle.get_state())
            # 描述车的状态
            self.vehicles_state.append(vehicle.otherState)
            # 全局任务状态
            self.taskState.append(vehicle.taskState)
        for mec in self.MECs:
            self.otherState.extend(mec.state)

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

    # 更新每辆车最近的mec
    def renew_neighbor_mec(self):
        for vehicle in self.vehicles:
            distance = []
            for mec in self.MECs:
                distance.append(self.compute_distance(vehicle, mec))
            vehicle.mec_lest = self.MECs[distance.index(min(distance))]

    # 获得卸载对象
    @staticmethod
    def get_aim(vehicle: Vehicle, action):
        if action == 0:
            return vehicle
        elif action == 1:
            return vehicle.mec_lest
        else:
            return vehicle.neighbor[action - 2]

    """
    处理选择任务和选择卸载目标动作(包含持有)
    """

    def process_taskActions(self):
        numOffloading = 1 + 1 + MAX_NEIGHBOR
        for i, vehicle in enumerate(self.vehicles):
            # 没有任务或者处于等待状态，不能执行卸载,也不给惩罚
            if vehicle.len_task <= 0 or vehicle.hold_on > 0:
                vehicle.cur_task = None
                continue

            action = self.taskActions[i]
            offloadingAction = self.offloadingActions[i]

            # 此时为持有动作
            # if offloadingAction >= numOffloading:
            #     vehicle.hold_on = HOLD_TIME[offloadingAction - numOffloading]
            #     # print("车辆{}将持有{}ms".format(i, vehicle.hold_on))
            #     vehicle.cur_task = None
            #     self.reward[i] += 0.05 * (vehicle.total_task[0].max_time - (self.cur_frame - vehicle.total_task[
            #         0].create_time + vehicle.hold_on)) - Kq * vehicle.len_task - vehicle.overflow
            #     print("持有奖励为{}".format(self.reward[i]))
            #     continue

            # 获得要传输的任务
            if action >= vehicle.len_task or action < 0:
                # 非法动作 给予惩罚项 任务队列不变
                self.reward[i] += Ki - Kq * vehicle.len_task - vehicle.overflow
                vehicle.cur_task = None
                continue
            # 大于可选范围（默认选择第一个）
            elif action >= MAX_TASK:
                task = vehicle.total_task[0]
            else:
                task = vehicle.total_task[action]

            # 初始化任务信息
            # 获得卸载对象
            aim = self.get_aim(vehicle, offloadingAction)
            task.aim = aim
            # 计算实时速率，用作奖励函数计算
            task.rate = self.compute_rate(vehicle, aim)

            # 卸载给本地 直接放到任务队列中
            if vehicle == aim:
                # 出队列时间
                task.pick_time = self.cur_frame
                vehicle.accept_task.append(task)
                vehicle.total_task.remove(task)
                vehicle.cur_task = task
                vehicle.len_task -= 1
                continue

            # 需要传输 卸载给远程
            if vehicle.trans_task == 1:
                # 有任务在传输 任务失败
                vehicle.cur_task = None
                # 非法动作 给予惩罚项
                self.reward[i] += Ki - Kq * vehicle.len_task - vehicle.overflow
            else:
                task.pick_time = self.cur_frame
                task.aim = aim
                task.rate = self.compute_rate(task.vehicle, task.aim)

                aim.len_action += 1
                vehicle.len_task -= 1
                vehicle.cur_task = task
                vehicle.trans_task = 1
                vehicle.total_task.remove(task)
                self.need_trans_task.append(task)

    """
    处理等待动作
    """

    def process_holdActions(self):
        # 减去一个时隙的等待时间
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.hold_on > 0:
                vehicle.hold_on -= 1
                for task in vehicle.total_task:
                    task.hold_on_time += 1

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

    """
    计算实时传输速率（在一个时隙内假设不变）
    """

    def compute_rate(self, vehicle: Vehicle, aim):
        # print("vehicle:{} aim:{} ".format(vehicle.id, aim.id))
        if aim == vehicle:  # 本地计算
            return 0

        distance = self.compute_distance(vehicle, aim)
        if type(aim) == MEC:
            fade = self.generate_fading_V2I(distance)
        else:
            fade = self.generate_fading_V2V(distance)
        power = np.power(10, (POWER + fade) / 10)
        sigma_w = np.power(10, sigma / 10)
        sign = power / sigma_w
        SNR = round((BrandWidth_Mec / self.num_Vehicles) * np.log2(1 + sign), 2)
        # print("第{}辆车速率:{} kb/ms".format(vehicle.id, SNR))
        return SNR  # kb/ms

    # 计算两物体持续时间
    def compute_persist(self, vehicle: Vehicle, aim):
        distance = self.compute_distance(vehicle, aim)
        # print("aim:{} vehicle:{}".format(aim.id, vehicle.id))
        if distance > aim.range:
            return 0

        if type(aim) is Vehicle:  # 如果对象是车
            if vehicle.velocity == aim.velocity and vehicle.direction == aim.direction:
                return sys.maxsize * 1000  # return np.abs(vehicle.direction * 500 - np.max(np.abs(vehicle.get_x),
                # np.abs(aim.get_x))) / vehicle.velocity
            else:
                return (np.sqrt(vehicle.range ** 2 - (aim.get_y - vehicle.get_y) ** 2) + aim.get_x - vehicle.get_x) / \
                       np.abs(vehicle.velocity * vehicle.direction - aim.velocity * aim.direction) * 1000

        else:  # 对象是MEC
            return (np.sqrt(aim.range ** 2 - (aim.get_y - vehicle.get_y) ** 2) + aim.get_x - vehicle.get_x) / np.abs(
                vehicle.velocity * vehicle.direction) * 1000

    # 计算这个任务传输所需要消耗的能量
    @staticmethod
    def compute_energy(trans_time):
        return np.power(10, POWER / 10) / 1000 * trans_time

    """
    给每个任务分配计算资源
    """

    def distribute_resource(self, ratio: list):
        # 记录分配比率
        sum_ratio_matrix = np.zeros((self.num_Vehicles, self.num_Vehicles + self.num_MECs), dtype=float)
        for i, vehicle in enumerate(self.vehicles):
            task = vehicle.cur_task
            if task is not None:
                if ratio[i] == 0:
                    # 分配比率为零(非法动作/任务失败)重新如任务队列
                    self.reward[i] += Ki - Kq * vehicle.len_task - vehicle.overflow
                    task.vehicle.cur_task = None
                    # 重新入队列
                    task.vehicle.total_task.append(task)
                    if task.aim == task.vehicle:
                        vehicle.accept_task.remove(task)
                    else:
                        self.need_trans_task.remove(task)
                resources = task.aim.resources
                # 分配资源
                task.compute_resource = ratio[i] * resources
                task.aim.resources -= task.compute_resource
                j = task.aim.id + self.num_Vehicles if type(task.aim) == MEC else task.aim.id
                sum_ratio_matrix[i][j] += ratio[i]

        # 对列求和(验证分配的合法性)
        sum_ratio = np.sum(sum_ratio_matrix, axis=0)
        for j, cur_ratio in enumerate(sum_ratio):
            aim = self.vehicles[j] if j < self.num_Vehicles else self.MECs[j - self.num_Vehicles]
            if cur_ratio > 1 or aim.resources <= 0:
                # print("第{}ms分配资源非法".format(self.cur_frame))
                # 分配非法
                for i in range(sum_ratio_matrix.shape[0]):
                    if sum_ratio_matrix[i][j] > 0:
                        task = self.vehicles[i].cur_task
                        task.vehicle.cur_task = None
                        self.reward[i] += Ki - Kq * self.vehicles[i].len_task - self.vehicles[i].overflow
                        # 移除任务（任务被判定为失败）
                        if i == j:
                            self.vehicles[i].accept_task.remove(task)
                        else:
                            self.need_trans_task.remove(task)
                        # 重新入队列
                        self.vehicles[i].total_task.append(task)
                        # 回收资源
                        aim.resources += task.compute_resource

    """
    计算此时任务的奖励
    """

    def get_reward(self, task):
        vehicle = task.vehicle
        aim = task.aim
        if vehicle == aim:
            trans_time = 0
        else:
            cur_rate = task.rate
            trans_time = task.need_trans_size / cur_rate
        cur_compute = task.compute_resource
        compute_time = task.need_precess_cycle / cur_compute
        communication_time = self.compute_persist(vehicle, aim)
        # 总时间=出队列时间-创建时间+传输时间+持有时间+处理时间
        sum_time = trans_time + compute_time + task.pick_time - task.create_time

        if sum_time > communication_time:
            # 最大通信时间小于总时间
            reward = Ki - Kq * vehicle.len_task - vehicle.overflow
        else:
            if task.aim != vehicle:
                # 考虑通信消耗的能量（非本地卸载）
                energy = self.compute_energy(trans_time)
                print("传输需要{}ms".format(trans_time))
                print("传输消耗{} J".format(energy))
                # 支付价格
                if type(task.aim) == MEC:
                    reward = -MEC_Price * task.size
                else:
                    reward = -VEC_Price * task.size
            else:
                # 计算任务消耗的能量（本地卸载）
                energy = round(gama * np.power(cur_compute, 3) * compute_time, 2)
                print("本地计算消耗{} J".format(energy))
                reward = -LOC_Price * task.size
            reward += 1 - T3 * (a * sum_time + b * energy) - Kq * vehicle.len_task

            if sum_time > task.max_time:
                deltaTime = sum_time - task.max_time
                if deltaTime > 20:
                    reward = Ki - Kq * vehicle.len_task - vehicle.overflow
                # 总时延大于任务阈值
                else:
                    reward += T2 * (sum_time - task.max_time)
                print("任务{}超时{}ms".format(vehicle.id, sum_time - task.max_time))
            print("车辆{}获得{}奖励".format(vehicle.id, reward))
        return round(reward, 2)

    """
    计算此时环境所有车辆的奖励
    """

    def compute_rewards(self):
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.cur_task is not None:
                # 保持状态
                if vehicle.hold_on <= 0:
                    self.reward[i] += self.get_reward(vehicle.cur_task)
                    vehicle.cur_task = None
        self.Reward = np.mean(self.reward)

    """
    根据动作将任务添加至对应的列表当中  分配任务
    """

    def distribute_task(self, cur_frame):
        need_task = []
        time = cur_frame - self.cur_frame
        # 远程卸载
        for task in self.need_trans_task:
            aim = task.aim
            vehicle = task.vehicle
            distance = self.compute_distance(vehicle, aim)
            # 超出传输范围
            if distance > aim.range:
                continue
            # 真实传输时间加上一时隙
            task.trans_time += time
            # 计算实时速率
            rate = self.compute_rate(vehicle, aim)
            # 能够传输完
            if task.need_trans_size <= time * rate:
                # 表示当前任务能够传输完成 可以继续传输其他任务
                self.vehicles[vehicle.id].trans_task = 0
                # print("第{}车任务传输完成，真实花费{}ms".format(vehicle.id, task.trans_time))
                aim.accept_task.append(task)
                aim.len_action -= 1
            else:
                task.need_trans_size -= time * rate
                need_task.append(task)
        # print("forward", self.need_trans_task)
        self.need_trans_task = need_task
        # print("after", self.need_trans_task)
        # 更新车和mec的需要处理的任务数量
        for vehicle in self.vehicles:
            vehicle.sum_needDeal_task = len(vehicle.accept_task)
        for mec in self.MECs:
            mec.sum_needDeal_task = len(mec.accept_task)

    # 更新资源信息并为处理完的任务计算奖励
    def renew_resources(self, cur_frame):
        time = cur_frame - self.cur_frame
        # 更新车的资源信息
        for i, vehicle in enumerate(self.vehicles):
            total_task = vehicle.accept_task
            size = len(total_task)

            if size > 0:  # 此时有任务并且有剩余资源
                # 记录这个时隙能够处理完的任务
                retain_task = []
                for task in total_task:
                    f = task.compute_resource
                    # 遍历此车的所有任务列表
                    precessed_time = task.need_precess_cycle / f
                    # 处理时间+一时隙
                    task.precess_time += time
                    # 加上这时隙所消耗能量
                    task.energy = gama * np.power(f / 1000, 3) * time
                    if precessed_time > time:
                        # 不能处理完
                        task.need_precess_cycle -= f * time
                        retain_task.append(task)
                    else:
                        # if task.aim == task.vehicle:
                        #     print("任务{}卸载给自己".format(task.vehicle.id))
                        # print("任务{}已完成，实际传输花费{}ms，实际计算花费{}ms".format(task.vehicle.id,
                        #                                                               task.trans_time,
                        #                                                               task.precess_time))
                        # 收回计算资源
                        task.aim.resources += task.compute_resource
                vehicle.accept_task = retain_task

        # 更新mec的资源信息
        for i, mec in enumerate(self.MECs):
            total_task = mec.accept_task
            size = len(total_task)
            if size > 0:
                retain_task = []
                for task in total_task:
                    f = task.compute_resource
                    precessed_time = task.need_precess_cycle / f
                    task.precess_time += time
                    if precessed_time > time:
                        task.need_precess_cycle -= f * time
                        retain_task.append(task)
                    else:
                        # print("任务{}已完成，传输花费{}ms，计算花费{}ms".format(task.vehicle.id, task.trans_time,
                        #                                                       task.precess_time))
                        # 收回计算资源
                        task.aim.resources += task.compute_resource
                mec.accept_task = retain_task

        # 分配任务信息（在计算之后执行是因为将一个时隙看作为原子操作，因此这个时隙接受到的任务不能进行计算）
        self.distribute_task(cur_frame=cur_frame)

    # 更新每辆车的位置
    def renew_locs(self, cur_frame):
        time = cur_frame - self.cur_frame
        for vehicle in self.vehicles:
            loc_x = round(vehicle.get_x + vehicle.direction * vehicle.velocity * time, 2)
            if loc_x > 1000:
                vehicle.set_location(-1000, vehicle.get_y)
            elif loc_x < -1000:
                vehicle.set_location(1000, vehicle.get_y)
            else:
                vehicle.set_location(loc_x, vehicle.get_y)

    """
    更新状态
    """

    def renew_state(self, cur_frame):
        self.otherState = []
        self.taskState = []
        self.vehicles_state = []

        # 更新车位置信息
        self.renew_locs(cur_frame)
        # 更新邻居表
        self.renew_neighbor()
        # 更新最近mec
        self.renew_neighbor_mec()
        for vehicle in self.vehicles:
            # 更新时间
            vehicle.cur_frame = cur_frame
            # 产生任务
            vehicle.create_work()
            # 更新资源已经接受的任务信息
            self.otherState.extend(vehicle.get_state())
            self.taskState.append(vehicle.taskState)
            self.vehicles_state.append(vehicle.otherState)
            # 更新任务信息
            vehicle.cur_task = None
        for mec in self.MECs:
            mec.cur_frame = cur_frame
            self.otherState.extend(mec.get_state())

    # 执行动作
    def step(self, taskActions, offloadingActions, computeRatioActions):
        cur_frame = self.cur_frame + 1  # ms
        # 分配动作
        # 卸载动作
        self.offloadingActions = offloadingActions
        # 任务选择动作
        self.taskActions = taskActions
        # 资源分配动作
        self.computeRatioActions = computeRatioActions

        # 重置奖励
        self.reward = [0] * self.num_Vehicles

        # 处理等待车辆
        # self.process_holdActions()

        # 处理选取任务动作
        self.process_taskActions()

        # 分配计算资源
        self.distribute_resource(self.computeRatioActions)
        # 计算奖励
        self.compute_rewards()

        # 记录当前状态
        other_state = self.otherState
        task_state = self.taskState
        vehicle_state = self.vehicles_state

        # 更新资源信息以及车辆任务信息
        self.renew_resources(cur_frame)

        # 更新状态
        self.renew_state(cur_frame)

        # 更新时间
        self.cur_frame = cur_frame
        # print("当前有{}个任务没传输完成".format(len(self.need_trans_task)))

        # 平均奖励
        self.Reward = np.mean(self.reward)

        return other_state, task_state, vehicle_state, self.vehicles_state, self.otherState, self.taskState, self.Reward, self.reward
