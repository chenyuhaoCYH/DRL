import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class DQN(nn.Module):
    def __init__(self, state_n, actions_n):
        """
        :param state_n:  状态空间维度
        :param actions_n: 动作空间维度
        """
        super(DQN, self).__init__()

        self.input_layer = nn.Linear(state_n, 128)
        self.hidden1 = nn.Linear(128, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, actions_n)

    def forward(self, x):
        """

        :param x: batch_size*state_n
        :return: batch_size*actions_n  输出每个动作对应的q值
        """
        x1 = F.relu(self.input_layer(x))
        x2 = F.relu(self.hidden1(x1))
        x3 = F.relu(self.hidden2(x2))
        out = self.output_layer(x3)

        return out


if __name__ == '__main__':
    # x = np.random.uniform(1,8,(2,3))
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
    print(x)
    with torch.no_grad():
        model = DQN(3, 3)
        # (0/1):0代表列，1代表行  [0、1]：0代表值，1代表下标
        result = model.forward(torch.from_numpy(x,))
        print(result)
        print(result.max(0)[0])
        print(result.max(0)[1])
        print(result.max(1)[0])
        print(result.max(1)[1])
