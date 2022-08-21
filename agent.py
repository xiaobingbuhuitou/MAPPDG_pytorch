"""
    智能体构建
"""
import torch
import numpy as np
from maddpg import MADDPG
from hyperparams import n_action_shape, n_state_shape, eps_greedy, action_high, action_low


class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.policy = MADDPG(self.agent_id)

    def select_action(self, obs):
        """
            p贪婪进行取值
        """
        p = np.random.uniform()
        if p < eps_greedy:
            # 随机进行采样
            action = np.random.uniform(action_low, action_high, n_action_shape)
            return action
        else:
            # 使用actor网络进行生成并添加噪声方便探索
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = self.policy.actor_network(obs).squeeze(
                0).detach().numpy()  # action [1, n_action_num] -> [n_action_num]
            noise = np.random.randn(n_action_shape)  # shape [n_action_num, ]
            action_noise = action + noise
            action_noise = np.clip(action_noise, action_low, action_high)
            return action_noise

    def update(self, agents_batch_s, agents_batch_a, agents_batch_r, agents_batch_done, agents_batch_obs_,
               other_agents):
        self.policy.update(agents_batch_s, agents_batch_a, agents_batch_r, agents_batch_done, agents_batch_obs_,
                           other_agents)
