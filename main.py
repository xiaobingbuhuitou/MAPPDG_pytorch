"""
    train / test函数
    基本RL流程框架
"""

import torch
import numpy as np
import random
import os
from agent import Agent
from buffer import Buffer
from hyperparams import n_episode, n_step, create_env, batch_size, n_agents, evaluate_episodes, evaluate_episode_len, \
    e_name


class Runner:

    def __init__(self):
        self.replay_buffer = Buffer()
        self.agents = self.init_agent()
        self.model_path = os.path.join('./model', e_name)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def init_agent(self):
        agents = []
        for i in range(n_agents):
            agent = Agent(i)
            agents.append(agent)
        return agents

    def train(self):
        """
            n_obs 状态集合
            n_obs_下一个状态
            actions 动作集合
            n_r 奖励集合
            n_done 是否结束
        """
        env = create_env
        n_obs = env.reset()
        agent_r = [0.0 for _ in range(env.n)]  # 记录所有轮的每一个agent平均奖励和
        for i_eps in range(n_episode):
            eps_n_r = 0
            for i_step in range(n_step):
                print("_______________当前为{0}轮__________第{1}步骤".format(i_eps, i_step))
                actions = []
                # 获取每一个agent的动作----注意该环境中一个有4个角色，其中3个是agent
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(n_obs[agent_id])
                    actions.append(action)
                # 加入对手的动作---不控制随机生成
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                # 环境交互
                n_obs_, n_r, n_done, n_info = env.step(actions)
                # 如果全为done则该轮不用继续跑了
                done = all(n_done)
                if done:
                    n_obs = env.reset()
                    break
                eps_n_r += np.mean(n_r)
                # 给出每一个agent各自的reward
                for i, r in enumerate(n_r):
                    agent_r[i] += r

                # 存入buffer
                self.replay_buffer.store(n_obs[: n_agents], actions[: n_agents], n_r[:n_agents], n_done[: n_agents],
                                         n_obs_[:n_agents])

                # 状态更新
                n_obs = n_obs_

                # buffer采样进行agent网络更新训练
                if self.replay_buffer.over_batch_size:  # buffer数量大于batch_size时才进行采样
                    agents_batch_s, agents_batch_a, agents_batch_r, agents_batch_done, agents_batch_obs_ \
                        = self.replay_buffer.sample()
                    for agent in self.agents:
                        other_agents = self.agents.copy()
                        other_agents.remove(agent)
                        # 训练时r 是局部的(自己的), s, s_, a都是要全局
                        # 同样的actor是局部的, critic是全局的信息（指导actor）
                        agent.update(agents_batch_s, agents_batch_a, agents_batch_r, agents_batch_done,
                                     agents_batch_obs_,
                                     other_agents)
            print("_________当前为{0}轮________此时我们获得总体奖励和的平均为{1}".format(i_eps, eps_n_r / n_step))
            # agent_r为每一轮的平均
            for i, ar in enumerate(agent_r):
                agent_r[i] = ar / n_step
            if i_eps % 20 == 0:  # 每过20轮进行模型更新保存
                for agent in self.agents:
                    agent.policy.save_model()
        print("===================训练完毕================ 模型见./model")
        return agent_r

    def test(self):
        # 仅使用actor网络进行预测   test时我们agent的model是从model中读取的
        returns = []
        for episode in range(evaluate_episodes):
            # reset the environment
            s = create_env.reset()
            rewards = 0
            for time_step in range(evaluate_episode_len):
                create_env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id])
                        actions.append(action)
                # 单独加入对抗者动作
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = create_env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            print('Returns is', rewards)
        return sum(returns) / evaluate_episodes


if __name__ == '__main__':
    runner = Runner()
    agent_r = runner.train()
    print("agent_r: ", agent_r)
    avg = runner.test()
    print("avg: ", avg)
