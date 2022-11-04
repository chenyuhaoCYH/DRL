import numpy as np
import torch

import experiment2.model as model
from experiment2.env import Env

if __name__ == '__main__':
    env = Env()
    env.reset()

    vehicle = env.vehicles[0]
    task_shape = np.array([vehicle.taskState]).shape
    print(task_shape)
    print(vehicle.taskState)
    myModel = model.DQN(len(vehicle.otherState), 14, task_shape)
    print(myModel)

    ans = myModel(torch.tensor([vehicle.otherState]),
                  torch.tensor([[vehicle.taskState]]))
    print(ans)

    print(torch.max(ans, 1)[0])
    print(torch.max(ans, 1)[1])
