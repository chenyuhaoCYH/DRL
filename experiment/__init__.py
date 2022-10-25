import ptan
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from experiment.env import Env


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
