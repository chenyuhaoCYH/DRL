#import ptan
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

HID_SIZE = 64
HID_SIZE_MIN = 32


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
        self.logstd_task = nn.Parameter(torch.zeros(task_aim_dim))
        self.logstd_aim = nn.Parameter(torch.zeros(act_aim_dim))

    def forward(self, obs, neighbor, task, is_train=True):
        task_out = self.cnn_task(task)
        neighbor_out = self.cnn_neighbor(neighbor)
        x = torch.cat((task_out, neighbor_out, obs), -1)
        same_out = self.same(x)
        act_out = self.act(same_out)
        task_out = self.task(same_out)
        if is_train:
            rnd_task = torch.tensor(np.random.normal(size=task_out.shape))
            rnd_aim = torch.tensor(np.random.normal(size=act_out.shape))
            task_out = task_out + torch.exp(self.logstd_task) * rnd_task
            act_out = act_out + torch.exp(self.logstd_aim) * rnd_aim

        # act_out = F.gumbel_softmax(act_out)
        act_pro = F.softmax(act_out, dim=-1)
        task_pro = F.softmax(task_out, dim=-1)
        # print(act_pro)
        # print(torch.sum(act_pro))
        # print(task_pro)
        # return act_pro, task_pro  # 打印网络结构用
        return Categorical(task_pro), Categorical(act_pro)  # 真实使用


class ModelCritic(nn.Module):
    def __init__(self, obs_size, task_size, act_size):
        super(ModelCritic, self).__init__()

        self.cnn = CNNLayer(obs_size, HID_SIZE)

        self.task_cnn = CNNLayer(task_size, HID_SIZE)

        self.value = nn.Sequential(
            nn.Linear(HID_SIZE * 2 + act_size, HID_SIZE * 2),
            nn.ReLU(),
            nn.Linear(HID_SIZE * 2, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE_MIN),
            nn.ReLU(),
            nn.Linear(HID_SIZE_MIN, 1),
        )
        self.value1 = nn.Sequential(
            nn.Linear(HID_SIZE * 2 + act_size, HID_SIZE * 2),
            nn.ReLU(),
            nn.Linear(HID_SIZE * 2, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE_MIN),
            nn.ReLU(),
            nn.Linear(HID_SIZE_MIN, 1),
        )

    def forward(self, states_v, task_states_v, actions_v):
        cnn_out = self.cnn(states_v)
        task_out = self.task_cnn(task_states_v)

        v = torch.cat((actions_v, cnn_out, task_out), -1)
        task_value = self.value(v)
        # aim_value = self.value1(v)
        return task_value  # , aim_value


class ModelSACTwinQ(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelSACTwinQ, self).__init__()

        self.q1 = nn.Sequential(
            nn.Linear(obs_size + act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(obs_size + act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.q1(x), self.q2(x)

"""

class AgentDDPG(ptan.agent.BaseAgent):
    """"""
    Agent implementing Orstein-Uhlenbeck exploration process
    """"""

    def __init__(self, net, device="cpu", ou_enabled=True,
                 ou_mu=0.0, ou_teta=0.15, ou_sigma=0.2,
                 ou_epsilon=1.0):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon

    def initial_state(self):
        return None

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(
                        shape=action.shape, dtype=np.float32)
                a_state += self.ou_teta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(
                    size=action.shape)

                action += self.ou_epsilon * a_state
                new_a_states.append(a_state)
        else:
            new_a_states = agent_states

        actions = np.clip(actions, -1, 1)
        return actions, new_a_states

"""
class DQNCNN(nn.Module):
    def __init__(self, obs_dim, task_dim, neighbor_dim, taskAction_dim, aimAction_dim):
        super(DQNCNN, self).__init__()
        self.input_layer = nn.Linear(obs_dim + 32 + 32, 128)
        self.hidden1 = nn.Linear(128, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.hidden3 = nn.Linear(64, 128)
        self.cnn1 = CNNLayer(task_dim, 32)
        self.cnn2 = CNNLayer(neighbor_dim, 32)
        self.output_layer1 = self.common(64, taskAction_dim)
        self.output_layer2 = self.common(64, aimAction_dim)

    def common(self, input_dim, action_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            self.hidden1,
            nn.ReLU(),
            self.hidden2,
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x, task, neighbor):
        """

        :param x: batch_size*state_n
        :return: batch_size*actions_n  输出每个动作对应的q值
        """
        # 任务卷积层
        cnn_out1 = self.cnn1(task)
        cnn_out2 = self.cnn2(neighbor)
        x = torch.cat((x, cnn_out1, cnn_out2), -1)

        # 公共层
        x1 = F.relu(self.input_layer(x))
        x2 = F.relu(self.hidden1(x1))
        x3 = F.relu(self.hidden2(x2))

        taskActionValue = self.output_layer1(x3)
        aimActionValue = self.output_layer2(x3)

        return taskActionValue, aimActionValue


class DQN(nn.Module):
    def __init__(self, obs_dim, task_dim, taskAction_dim, aimAction_dim):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(obs_dim + 32, 128)
        self.hidden1 = nn.Linear(128, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.hidden3 = nn.Linear(64, 128)
        self.cnn = CNNLayer(task_dim, 32)
        self.output_layer1 = self.common(64, taskAction_dim)
        self.output_layer2 = self.common(64, aimAction_dim)

    def common(self, input_dim, action_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            self.hidden1,
            nn.ReLU(),
            self.hidden2,
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x, task):
        """

        :param x: batch_size*state_n
        :return: batch_size*actions_n  输出每个动作对应的q值
        """
        # 任务卷积层
        cnn_out = self.cnn(task)
        x = torch.cat((x, cnn_out), -1)

        # 公共层
        x1 = F.relu(self.input_layer(x))
        x2 = F.relu(self.hidden1(x1))
        x3 = F.relu(self.hidden2(x2))

        taskActionValue = self.output_layer1(x3)
        aimActionValue = self.output_layer2(x3)

        return taskActionValue, aimActionValue


class CNNLayer(nn.Module):
    def __init__(self, obs_shape, hidden_size, use_orthogonal=True, use_ReLU=True, kernel_size=3, stride=1):
        super(CNNLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):  # 权重使用正交初始化，激活函数使用relu
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel,
                            out_channels=hidden_size // 2,
                            kernel_size=kernel_size,
                            stride=stride)
                  ),
            active_func,
            nn.Flatten(),
            init_(nn.Linear(
                hidden_size // 2 * (input_width - kernel_size + stride) * (input_height - kernel_size + stride),
                hidden_size)
            ),
            active_func,
            init_(nn.Linear(hidden_size, hidden_size)), active_func)

    def forward(self, x):
        x = self.cnn(x)

        return x


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
