import copy

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

HID_SIZE = 64
HID_SIZE_MIN = 32


class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """

    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # 软更新
    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)


class ModelActor(nn.Module):
    def __init__(self, obs_dim, neighbor_dim, task_dim, task_aim_dim, act_aim_dim):
        super(ModelActor, self).__init__()

        self.cnn_task = CNNLayer(task_dim, HID_SIZE)
        self.cnn_neighbor = CNNLayer(neighbor_dim, HID_SIZE_MIN)
        self.same = nn.Sequential(
            nn.Linear(HID_SIZE + HID_SIZE_MIN + obs_dim, 2 * HID_SIZE),
            nn.ReLU(),
            nn.Linear(2 * HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 2 * HID_SIZE),
            nn.ReLU(),
        )
        self.task = nn.Sequential(
            nn.Linear(2 * HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE_MIN),
            nn.ReLU(),
            nn.Linear(HID_SIZE_MIN, task_aim_dim),
        )
        self.act = nn.Sequential(
            nn.Linear(2 * HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE_MIN),
            nn.ReLU(),
            nn.Linear(HID_SIZE_MIN, act_aim_dim),
        )

    def forward(self, obs, neighbor, task, train=True):
        task_out = self.cnn_task(task)
        neighbor_out = self.cnn_neighbor(neighbor)
        x = torch.cat((task_out, neighbor_out, obs), -1)
        same_out = self.same(x)
        act_out = self.act(same_out)
        task_out = self.task(same_out)

        # 训练完成之后无需添加噪音
        if train:
            # act_out += torch.tensor(np.random.normal(size=act_out.shape))
            # task_out += torch.tensor(np.random.normal(size=task_out.shape))
            act_out = F.gumbel_softmax(act_out, hard=True)
            task_out = F.gumbel_softmax(task_out, hard=True)
        # else:
        #     task_out = F.softmax(task_out, dim=-1)
        #     act_out = F.softmax(act_out, dim=-1)

        # act_pro = F.softmax(act_out, dim=-1)
        # task_pro = F.softmax(task_out, dim=-1)
        # print(act_pro)
        # print(torch.sum(act_pro))
        # print(task_pro)
        # return act_pro, task_pro  # 打印网络结构用
        # return Categorical(task_pro), Categorical(act_pro)  # 真实使用
        return task_out, act_out


class ModelCritic(nn.Module):
    def __init__(self, obs_size, task_size, task_action_size, aim_action_size):
        super(ModelCritic, self).__init__()

        self.cnn = CNNLayer(obs_size, HID_SIZE)

        self.task_cnn = CNNLayer(task_size, HID_SIZE)

        self.task_value = nn.Sequential(
            nn.Linear(HID_SIZE * 2 + task_action_size, HID_SIZE * 2),
            nn.ReLU(),
            nn.Linear(HID_SIZE * 2, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE_MIN),
            nn.ReLU(),
            nn.Linear(HID_SIZE_MIN, 1),
        )
        self.aim_value = nn.Sequential(
            nn.Linear(HID_SIZE * 2 + aim_action_size, HID_SIZE * 2),
            nn.ReLU(),
            nn.Linear(HID_SIZE * 2, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE_MIN),
            nn.ReLU(),
            nn.Linear(HID_SIZE_MIN, 1),
        )

    def forward(self, states_v, task_states_v, task_action_v, aim_action_v):
        cnn_out = self.cnn(states_v)
        task_out = self.task_cnn(task_states_v)

        v = torch.cat((cnn_out, task_out), -1)
        task_value = self.task_value(torch.cat((v, task_action_v), -1))
        aim_value = self.aim_value(torch.cat((v, aim_action_v), -1))
        return task_value, aim_value


class CNNLayer(nn.Module):
    def __init__(self, obs_shape, hidden_size, use_orthogonal=True, use_ReLU=True, kernel_size=3, stride=1):
        super(CNNLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):  # 权重使用正交初始化，激活函数使用relu
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        in_channels = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_size // 2,
                kernel_size=kernel_size,
                stride=stride)
            ),
            active_func,
            # nn.AvgPool2d(
            #     kernel_size=kernel_size,
            #     stride=stride),
            # active_func,
            # init_(nn.Conv2d(
            #     in_channels=3,
            #     out_channels=1,
            #     kernel_size=kernel_size,
            #     stride=stride)
            # ),
            # active_func,
            # nn.AvgPool2d(
            #     kernel_size=kernel_size,
            #     stride=stride),
            # active_func,
            nn.Flatten(),
            init_(nn.Linear(
                hidden_size // 2 * (input_width - kernel_size + stride) * (input_height - kernel_size + stride),
                hidden_size)
            ),
            active_func,
            init_(nn.Linear(hidden_size, hidden_size)), active_func)

    def forward(self, x):
        x = x / 255.0
        x = self.cnn(x)

        return x


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
