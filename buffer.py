"""
    经验回放
"""
from hyperparams import n_agents, batch_size, n_action_shape, n_state_shape, max_size
import numpy as np
import random


class Buffer(object):
    def __init__(self):
        self.max_size = max_size
        self.position = 0
        self.over_batch_size = False  # 判断存入的记录是否大于batch_size
        self.buffer = [[] for _ in range(n_agents)]  # [[], [], []] 不固定大小因为随机采样会出现None

    def store(self, obs, actions, r, done, obs_):
        """
            先按照agent进行压缩然后一条条放入每个agent的buffer
            buffer应该是先获取  对应的agent_idx 然后是在取 o, a, r, done, o_
           每一条压缩为tuple, 取出的时候需要解压
           obs.shape: [n_agents, num], r.shape: [n_agents, ]

            print("obs长度", len(obs), len(obs[0]), " actions 长度", len(actions), len(actions[0]),
                                "r长度", len(r), "obs_长度", len(obs_), len(obs_[0]))
                obs长度 3 16  actions 长度 3 5 r长度 3 obs_长度 3 16

            print("存入buffer时每个len(self.buffer[0])", len(self.buffer[0]))
                存入buffer时每个len(self.buffer[0]) 299  总共300次
        """

        zip_gather = zip(obs, actions, r, done, obs_)

        if len(self.buffer[0]) < self.max_size:  # 每个agent_buffer长度是一样的判断一个即可
            for i in range(n_agents):
                self.buffer[i].append(None)  # 先进行扩容 然后用刚加入的记录进行覆盖

        index = 0  # 每次zip_gather个数恰好为agent的个数 对应的放入
        for record in zip_gather:
            self.buffer[index][self.position] = record
            index += 1

        self.position = (self.position + 1) % self.max_size
        if self.position >= batch_size:
            self.over_batch_size = True

    def sample(self):
        """
            对于每一个agent都要取batch_size个元组
            取出时将元组解压放入对应的描述中
             print("len(self.buffer)", len(self.buffer)) len(self.buffer) 3
             __len()__求得Buffer对象长度 和len(self.buffer）不同
        """
        batches_obs = np.empty((n_agents, batch_size, n_state_shape))
        batches_obs_ = np.empty((n_agents, batch_size, n_state_shape))
        batches_a = np.empty((n_agents, batch_size, n_action_shape))
        batches_r = np.empty((n_agents, batch_size))  # 是不是应该是  [n, B, 1]???
        batches_done = np.empty((n_agents, batch_size))

        for i in range(n_agents):
            B = random.sample(self.buffer[i], batch_size)
            batch_i_s, batch_i_a, batch_i_r, batch_i_done, batch_i_s_ = zip(*B)  # 随机采样的时候会出现还没满有的值为None
            # 每个agent取出batch个
            batches_obs[i] = batch_i_s
            batches_obs_[i] = batch_i_s_
            batches_done[i] = batch_i_done
            batches_r[i] = batch_i_r
            batches_a[i] = batch_i_a
        return batches_obs, batches_a, batches_r, batches_done, batches_obs_  # numpy

    def __len__(self):
        """
            返回buffer中一个agent的记录条数即可
        """
        return len(self.buffer[0])
