import ptan
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

HID_SIZE = 64


class ModelActor(nn.Module):
    def __init__(self, obs_dim, act_dim, task_dim):
        super(ModelActor, self).__init__()
        self.act_dim = act_dim

        self.mu = nn.Sequential(
            nn.Linear(obs_dim + HID_SIZE, HID_SIZE * 2),
            nn.ReLU(),
            nn.Linear(HID_SIZE * 2, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, act_dim * 2),
        )
        self.cnn = CNNLayer(task_dim, HID_SIZE)

    def forward(self, x, task):
        cnn_out = self.cnn(task)
        x = torch.cat((x, cnn_out), -1)
        out = self.mu(x)
        action_out = out[:, :self.act_dim]
        task_out = out[:, self.act_dim:]
        action_pro = F.softmax(action_out, dim=-1)
        task_pro = F.softmax(task_out, dim=1)
        print(action_pro)
        print(task_pro)
        return action_pro, Categorical(action_pro), task_pro, Categorical(task_pro)


class ModelCritic(nn.Module):
    def __init__(self, obs_size):
        super(ModelCritic, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, x):
        return self.value(x)


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


class AgentDDPG(ptan.agent.BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """

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


class DQN(nn.Module):
    def __init__(self, obs_dim, task_dim, taskAction_dim, aimAction_dim, resourceAction_dim):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(obs_dim + 32, 128)
        self.hidden1 = nn.Linear(128, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.hidden3 = nn.Linear(64, 128)
        self.cnn = CNNLayer(task_dim, 32)
        self.output_layer1 = self.common(64, taskAction_dim)
        self.output_layer2 = self.common(64, aimAction_dim)
        self.output_layer3 = self.common(64, resourceAction_dim)

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
        resourceActionValue = self.output_layer3(x3)

        return taskActionValue, aimActionValue, resourceActionValue


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
        x = x / 255.0
        x = self.cnn(x)

        return x


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
