"""
    构建maddpg框架 初始化了每个agent的 A/C, target_A/ target_C
    进行agent的网络更新和agent的模型保存
"""

import torch
import torch.nn as nn

import numpy as np
import os
from network import Actor, Critic
from hyperparams import n_agents, n_action_shape, n_state_shape, action_high, lr, tau, gamma, save_dir, e_name


class MADDPG(object):
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.update_step = 0

        # 定义网络
        self.actor_network = Actor(64, self.agent_id)
        self.critic_network = Critic(64)

        self.target_actor_network = Actor(64, self.agent_id)
        self.target_critic_network = Critic(64)

        # 两两对应的网络参数初始值相同
        self.target_actor_network.load_state_dict(self.actor_network.state_dict())
        self.target_critic_network.load_state_dict(self.critic_network.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(), lr=lr)

        self.critic_loss = nn.MSELoss()

        # 当我们的已经有训练好的模型时直接读取, 每个agent创建一个文件夹
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.model_path = save_dir + '/' + e_name  # ./model/env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % self.agent_id  # ./model/env_name/agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # 加载模型
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/critic_params.pkl'))

    def update(self, agents_batch_s, agents_batch_a, agents_batch_r, agents_batch_done, agents_batch_obs_,
               other_agents):
        """
            obs 当前观测, obs_下一个时刻观测,
            a当前action, r当前action后的奖励,  a_ 下一个时刻动作
            Q 原始网络Q_val; Q_next-> target 预测的值
           进行每一个agent的 actor和 critic网络的更新
           r 我们只使用当前agent自己的, actor也是自己的信息 局部
           critic是全局信息
        """
        with torch.autograd.set_detect_anomaly(True):  # 用于追溯inplace操作导致的梯度失效bug
            batch_r = agents_batch_r[self.agent_id]
            tensor_batch_r = torch.tensor(batch_r, dtype=torch.float32)
            tensor_agent_batch_obs = torch.tensor(agents_batch_s, dtype=torch.float32)
            tensor_agent_batch_obs_ = torch.tensor(agents_batch_obs_, dtype=torch.float32)
            tensor_agent_batch_a = torch.tensor(agents_batch_a, dtype=torch.float32)

            # target_actor计算出 a_next 放入target_critic中计算出 Q_next -> target
            actions_ = []
            with torch.no_grad():  # 前向计算出各个值
                index = 0
                for aid in range(n_agents):
                    # 若是当前的个体  区分当前的agent和其他agent
                    if aid == self.agent_id:
                        actions_.append(
                            self.target_actor_network(tensor_agent_batch_obs_[self.agent_id]).detach().numpy())
                    else:
                        # 这边数组中index 和会和aid对应的, 因为每次恰好去除了当前的agent, 同时id由range(n_agents)生成
                        actions_.append(
                            other_agents[index].policy.target_actor_network(
                                tensor_agent_batch_obs_[aid]).detach().numpy())
                        index += 1

                # 全局信息计算出Q_next
                actions_numpy = np.array(actions_)
                tensor_batch_a_ = torch.tensor(actions_numpy, dtype=torch.float32)  # a_next tensor
                # detach()可以不用 Q_next.shape [B, 1]
                Q_next = self.target_critic_network(tensor_agent_batch_obs_, tensor_batch_a_).detach()
                # 不用 gamma * (1 - done) * Q_next   target.shape [B, 1]
                target = (tensor_batch_r.unsqueeze(1) + gamma * Q_next).detach()

            # 原始网络的Q_val 全局信息
            Q_val = self.critic_network(tensor_agent_batch_obs, tensor_agent_batch_a)

            # critic_loss Q_val 拟合 target
            ctor_loss = self.critic_loss(Q_val, target)
            """
                计算出ctor_loss就直接进行更新critic 不然会出现inplace bug
                同时若将tensor_ab_a和actor_loss放在老三步前面的话也会出问题 
                或者这边先更新actor在进行critic更新
            """
            self.critic_optimizer.zero_grad()
            ctor_loss.backward()
            self.critic_optimizer.step()

            # 不清楚为什么这边还要做一次输出actor，感觉并没有改变actor网络???
            tensor_agent_batch_a[self.agent_id] = self.actor_network(tensor_agent_batch_obs[self.agent_id])
            # actor_loss就是我们的Q值 梯度上升
            actor_loss = -torch.mean(self.critic_network(tensor_agent_batch_obs, tensor_agent_batch_a))

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update_target_network()  # 理论应该是间隔一些轮才替换
            if self.update_step > 0 and self.update_step % 50 == 0:
                self.save_model()
            self.update_step += 1

    def _soft_update_target_network(self):
        # tau * theta + (1 - tau) * target_theta
        for target_param, param in zip(self.target_actor_network.parameters(), self.actor_network.parameters()):
            target_param = (tau * param + (1 - tau) * target_param)
        for target_param, param in zip(self.target_critic_network.parameters(), self.critic_network.parameters()):
            target_param = (tau * param + (1 - tau) * target_param)

    def save_model(self):
        dir_path = os.path.join(save_dir, e_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        model_path = os.path.join(dir_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # actor / critic 都要存 存储的路径名称为 ./model/env_name/agent_id/ actor_params.pkl & critic_params.pkl
        torch.save(self.actor_network.state_dict(), model_path + '/' + 'actor_params.pkl')
        torch.save(self.critic_network.state_dict(), model_path + '/' + 'critic_params.pkl')
