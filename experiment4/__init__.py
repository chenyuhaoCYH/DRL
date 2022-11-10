"""
环境4
（两个动作：选择任务和选择对象）
加入了mec和车在时隙内处理任务的上限（mec最多同时处理10个任务、车最多处理5个任务）
使用经典城市道路（使用不同数量车辆和邻居）
为mec卸载和车辆卸载提供两种传输方式（即可同时像车辆和mec传输任务）
"""
import ptan
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from env import Env


def test_net(nets, env: Env, count=10):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        env.reset()
        while steps < 1000:
            action = []
            with torch.no_grad():
                for vehicle in env.vehicles:
                    state = torch.tensor(vehicle.otherState)
                    _, pro = nets[vehicle.id](state)
                    act = Categorical.sample(pro)
                    action.append(act.item())
            _, _, reward, _ = env.step(action)
            rewards += reward
            steps += 1
    return rewards / count, steps / count

# def calc_logprob(pro_v, actions_v):
#     p1 = - ((mu_v - actions_v) ** 2) / (2 * torch.exp(logstd_v).clamp(min=1e-3))
#     p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
#     return p1 + p2
