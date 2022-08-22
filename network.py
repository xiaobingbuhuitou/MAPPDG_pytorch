"""
    定义A/ C网络结构
"""
import torch
import torch.nn as nn
from hyperparams import n_action_shape, n_state_shape, n_agents, batch_size


class Actor(nn.Module):
    """
        输入s 输出a  actor只观测自己的obs
    """

    def __init__(self, hid_dim, agent_id):
        super(Actor, self).__init__()
        self.act = nn.Sequential(
            nn.Linear(n_state_shape, hid_dim),  # 这边应是obs_shape[agent_id]
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, n_action_shape),
            nn.Tanh()  # 最后应该是(-1, 1) 但是要在low-high_action之间这
        )

    def forward(self, state):
        """
            使用局部(自己的)信息进行更新所以不用n_agents
            state.shape [B, n_state_shape]
            action.shape [B, n_state_shape]
        """
        action = self.act(state)
        return action


class Critic(nn.Module):
    """
        输入 s, a输出 Q  critic是全局的
    """

    def __init__(self, hid_dim):
        super(Critic, self).__init__()
        self.ctor = nn.Sequential(
            nn.Linear(n_agents * n_action_shape + n_agents * n_state_shape, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, s, a):
        """
           s.shape [n_agents, B, n_state_shape ]
           a.shape [n_agents, B, n_action_shape]
           在actor/ critic更新时Q的计算可能会出现inplace问题导致梯度无法计算---bug
        """
        s = s.transpose(0, 1).reshape(batch_size, -1)  # [B, n_agent * num] torch.Size([128, 48])
        a = a.transpose(0, 1).reshape(batch_size, -1)  # [B, n_agent * num]
        inputs = torch.cat([s, a], 1)  # inputs.shape torch.Size([128, 63])
        Q = self.ctor(inputs)  # Q.shape torch.Size([128, 1])
        return Q
