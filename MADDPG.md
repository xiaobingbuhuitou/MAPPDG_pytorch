# MADDPG

[用多智能体强化学习算法MADDPG解决"老鹰捉小鸡"问题 | 机器之心 (jiqizhixin.com)](https://www.jiqizhixin.com/articles/2020-08-13-11)

[强化学习(十六) 深度确定性策略梯度(DDPG) - 刘建平Pinard - 博客园 (cnblogs.com)](https://www.cnblogs.com/pinard/p/10345762.html)

[maddpg原理以及代码解读 | Jianeng (jianengli.github.io)](https://jianengli.github.io/2021/03/19/rl/maddpg/)

![image-20220819205634506](D:\Typora\文章\RL\MADDPG.assets\image-20220819205634506.png)

![image-20220819205657617](D:\Typora\文章\RL\MADDPG.assets\image-20220819205657617.png)

![image-20220819205908560](D:\Typora\文章\RL\MADDPG.assets\image-20220819205908560.png)

```python
        # 就是表示是不是最后一次交互了
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)
```

[从代码到论文理解并复现MADDPG算法(PARL) - 飞桨AI Studio (baidu.com)](https://aistudio.baidu.com/aistudio/projectdetail/637951?shared=1)

https://blog.csdn.net/sinat_39620217/article/details/115299073

[(45条消息) MADDPG翻译_qiusuoxiaozi的博客-CSDN博客_maddpg原文](https://blog.csdn.net/qiusuoxiaozi/article/details/79066612)

[(45条消息) 强化学习：DDPG到MADDPG_彩虹糖梦的博客-CSDN博客_ddpg和maddpg](https://blog.csdn.net/caozixuan98724/article/details/110854007)

[(45条消息) MADDPG翻译_qiusuoxiaozi的博客-CSDN博客_maddpg原文](https://blog.csdn.net/qiusuoxiaozi/article/details/79066612)

[两种经典的多智能体强化学习算法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/379948793)

[(45条消息) 强化学习笔记：连续控制与 MADDPG_UQI-LIUWJ的博客-CSDN博客_maddpg离散动作空间](https://blog.csdn.net/qq_40206371/article/details/125113278)

[(45条消息) MADDPG分析及其更新策略见解_Y. F. Zhang的博客-CSDN博客_maddpg](https://blog.csdn.net/weixin_43145941/article/details/112726116)

MADDPG算法是对DDPG算法为适应多Agent环境的改进，最核心的部分就是**每个Agent的Critic部分能够获取其余所有Agent的动作信息**，进行中心化训练和非中心化执行，即在训练的时候，**引入可以观察全局的Critic来指导Actor训练**，而**测试的时候只使用有局部观测的actor采取行动**。

![image-20220819211016995](D:\Typora\文章\RL\MADDPG.assets\image-20220819211016995.png)

最重要的即在对于Critic部分进行参数更新训练时，其中的输入值——action和observation，都是包含所有其他Agent的action和observation的。通过给予每个Agent其他Agent的信息，从而使每个单一的Agent受其他Agent改变policy的影响程度降低，从而认为在已知的信息条件下，**环境可以看作是稳定不变**。

![image-20220819205429135](D:\Typora\文章\RL\MADDPG.assets\image-20220819205429135.png)

- ### 网络参数里只有obs的就是Actor,因为Actor只需要根据环境的观察值输出动作;

- ### 既包含obs,又包含act的就是Critic了,Critic根据Actor输出的动作act以及环境的观察值obs对Actor进行打分,分数就是Q值。

![image-20220819211157644](D:\Typora\文章\RL\MADDPG.assets\image-20220819211157644.png)

## MADDPG网络

```python
"""

    定义A/ C网络结构
"""
import torch
import torch.nn as nn
import numpy as np


class Actor(nn.Module):  # 输入s 输出a  这边只观察到自己的obs
    def __init__(self, hid_dim, agent_id):
        super(Actor, self).__init__()
        self.act = nn.Sequential(
            nn.Linear(n_state_shape, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, n_action_shape),
            nn.Tanh()  # 最后应该是(-1, 1) 但是要在low-high_action之间这两个为相反数所以直接乘
            # 也可以不用tanh()直接用softmax, dim=-1上做平均
        )

    def forward(self, state):
        action = self.act(state)
        return action


class Critic(nn.Module):  # 输入 s, a输出 Q  注意这边的critic是全局的
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
        # 首先是进行s, a拼接  我们这边的输入s [n_agent, B, num], a [n_agent, B, num] -> [s(ag1, ag2,..) a(ag1, ag2,..)]
        s = s.transpose(0, 1).reshape(BS, -1)  # [B, n_agent * num]
        a = a.transpose(0, 1).reshape(BS, -1)  # [B, n_agent * num]
        inputs = torch.cat([s, a], 1)
        # 这样输入的每次就是看为一个整体后的每一条的信息
        Q = self.ctor(inputs)
        return Q  # [B, 1]
    
    
    
# 一个有意思的操作可以不用将s, a进行提前的cat 可以直接使用两者输出后的结果进行拼接
# Time: 2019-11-05
# Author: Zachary 
# Name: MADDPG_torch   git项目名
import torch
import torch.nn as nn
import torch.nn.functional as F

class abstract_agent(nn.Module):
    def __init__(self):
        super(abstract_agent, self).__init__()
    
    def act(self, input):
        policy, value = self.forward(input) # flow the input through the nn
        return policy, value

class actor_agent(abstract_agent):
    def __init__(self, num_inputs, action_size, args):
        super(actor_agent, self).__init__()
        self.linear_a1 = nn.Linear(num_inputs, args.num_units_1)
        self.linear_a2 = nn.Linear(args.num_units_1, args.num_units_2)
        self.linear_a = nn.Linear(args.num_units_2, action_size)
        self.reset_parameters()
        # Activation func init
        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh= nn.Tanh()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        self.linear_a1.weight.data.mul_(gain)
        self.linear_a2.weight.data.mul_(gain)
        self.linear_a.weight.data.mul_(gain_tanh)
    
    def forward(self, input):
        """
        The forward func defines how the data flows through the graph(layers)
        """
        x = self.LReLU(self.linear_a1(input))
        x = self.LReLU(self.linear_a2(x))
        policy = self.tanh(self.linear_a(x))
        return policy 

class critic_agent(abstract_agent):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(critic_agent, self).__init__()
        ##############################################################
        self.linear_o_c1 = nn.Linear(obs_shape_n, args.num_units_1)
        self.linear_a_c1 = nn.Linear(action_shape_n, args.num_units_1)
        self.linear_c2 = nn.Linear(args.num_units_1*2, args.num_units_2)
        self.linear_c = nn.Linear(args.num_units_2, 1)
        self.reset_parameters()

        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh= nn.Tanh()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        self.linear_o_c1.weight.data.mul_(gain)
        self.linear_a_c1.weight.data.mul_(gain)
        self.linear_c2.weight.data.mul_(gain)
        self.linear_c.weight.data.mul_(gain)

    def forward(self, obs_input, action_input):
        """
        input_g: input_global, input features of all agents
        """
        x_o = self.LReLU(self.linear_o_c1(obs_input))
        x_a = self.LReLU(self.linear_a_c1(action_input))
        x_cat = torch.cat([x_o, x_a], dim=1)
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value

class openai_critic(abstract_agent):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(openai_critic, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(action_shape_n+obs_shape_n, args.num_units_openai)
        self.linear_c2 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_c = nn.Linear(args.num_units_openai, 1)

        self.reset_parameters()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, action_input):
        """
        input_g: input_global, input features of all agents
        """
        x_cat = self.LReLU(self.linear_c1(torch.cat([obs_input, action_input], dim=1)))
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value

class openai_actor(abstract_agent):
    def __init__(self, num_inputs, action_size, args):
        super(openai_actor, self).__init__()
        self.tanh= nn.Tanh()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_a1 = nn.Linear(num_inputs, args.num_units_openai)
        self.linear_a2 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_a = nn.Linear(args.num_units_openai, action_size)

        self.reset_parameters()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.linear_a1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a.weight, gain=nn.init.calculate_gain('leaky_relu'))
    
    def forward(self, input, model_original_out=False):
        """
        The forward func defines how the data flows through the graph(layers)
        flag: 0 sigle input 1 batch input
        """
        x = self.LReLU(self.linear_a1(input))
        x = self.LReLU(self.linear_a2(x))
        model_out = self.linear_a(x)
        u = torch.rand_like(model_out)
        policy = F.softmax(model_out - torch.log(-torch.log(u)), dim=-1)
        if model_original_out == True:   return model_out, policy # for model_out criterion
        return policy

```

## 软更新参数

```python
    # 使用 名字-val对 进行更新
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        
        
    # 直接对应更新 感觉不用加data或者换detach()也行吧
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)
	
        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)
```

# 一个inplace问题

[Pytorch:copy.deepcopy对torch.tensor.contiguous()? - 我爱学习网 (5axxw.com)](https://www.5axxw.com/questions/content/du8dm1#:~:text=在pythontorch中， copy.deepcopy 方法似乎通常用于创建deep-copies的torch张量，而不是创建现有张量的视图。,同时，据我所知， torch.tensor.contiguous () 方法将non-contiguous张量转换为连续张量，或将视图转换为深度复制的张量。)

[(45条消息) Python numpy pytorch 中的数据复制 copy deepcopy clone detach_Think@的博客-CSDN博客_pytorch 复制变量](https://blog.csdn.net/qq_40728667/article/details/122161029)

[(45条消息) pytorch报错 RuntimeError: 一个被用作梯度计算的变量被inplace操作修改了_强殖装甲凯普的博客-CSDN博客](https://blog.csdn.net/qq_38163755/article/details/110957133)

[python - 梯度计算所需的变量之一已被原位操作修改： - 堆栈内存溢出 (stackoom.com)](https://stackoom.com/question/4JfTZ)

[pytorch中反向传播的loss.backward(retain_graph=True)报错 - 编程猎人 (programminghunter.com)](https://www.programminghunter.com/article/40322289874/)

https://blog.csdn.net/qq_33093927/article/details/124063916

[(45条消息) 【PyTorch】RuntimeError: one of the variables needed for gradient computation has been modified by an_ncc1995的博客-CSDN博客](https://blog.csdn.net/ncc1995/article/details/99542594)



```python
with torch.autograd.set_detect_anomaly(True):  # 用于追溯inplace操作导致的梯度失效bug
    .....
```

## 主要这边要是先进行更新critic在进行actor会出问题

```python
# 原始网络的Q_val 全局信息
 Q_val = self.critic_network(tensor_agent_batch_obs, tensor_agent_batch_a)
# critic_loss Q_val 拟合 target
 ctor_loss = self.critic_loss(Q_val, target)

# 不清楚为什么这边还要做一次输出actor，感觉并没有改变actor网络???
# tensor_agent_batch_a[self.agent_id] = self.actor_network(tensor_agent_batch_obs[self.agent_id])
# actor_loss就是我们的Q值 梯度上升 ???
# actor_loss = -torch.mean(self.critic_network(tensor_agent_batch_obs, tensor_agent_batch_a))
# actor_loss = - self.critic_network(tensor_agent_batch_obs, tensor_agent_batch_a).mean()
##############如果上述的段落在这边同时进行先critic更新在actor会导致inplace问题放在下面不会
##############当然要是放在这边也行那就是先进行更新actor在critic
############主要不了解问题先更新actor后更新critic就没问题这边先更新actor是因为actor这边没保存中间值所以出问题我们将ctor_loss中Q_val也直接放入再测试先critic后actor??

self.critic_optimizer.zero_grad()
# ctor_loss.backward(retain_graph=True)
ctor_loss.backward()
self.critic_optimizer.step()

 #不清楚为什么这边还要做一次输出actor，感觉并没有改变actor网络???
tensor_agent_batch_a[self.agent_id] = self.actor_network(tensor_agent_batch_obs[self.agent_id])
 # actor_loss就是我们的Q值 梯度上升 ??? ######这边单独写Q也不行还是梯度问题
actor_loss = -torch.mean(self.critic_network(tensor_agent_batch_obs, tensor_agent_batch_a))
# actor_loss = - self.critic_network(tensor_agent_batch_obs, tensor_agent_batch_a).mean()

self.actor_optimizer.zero_grad()
# self.actor_loss.backward(retain_graph=True)
actor_loss.backward()
self.actor_optimizer.step()
######### 涉及相同部分的梯度, retain_graph可能会导致内存泄露 第一个写第二个可以不写
```

## 提示的问题是定位到critic网络这边target_c不用更新；Q_val 与 -torch.mean(self.critic_network(tensor_agent_batch_obs, tensor_agent_batch_a))  出现了问题.........<font color="red">暂时不会解决</font>

```python
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
        # s, a拼接  s [n_agent, B, num], a [n_agent, B, num]
        # print("输入前的s 形状", s.shape, "输入前的a的形状", a.shape)
        # 输入前的s 形状 torch.Size([3, 128, 16]) 输入前的a的形状 torch.Size([3, 128, 5])
        s = s.transpose(0, 1).reshape(batch_size, -1)  # [B, n_agent * num]
        a = a.transpose(0, 1).reshape(batch_size, -1)  # [B, n_agent * num]
        inputs = torch.cat([s, a], 1)
        # print("reshape后的s形状", s.shape, " cat后inputs形状", inputs.shape)
        # reshape后的s形状 torch.Size([128, 48])  cat后inputs形状 torch.Size([128, 63])
        Q = self.ctor(inputs)
        # print("Q.shape", Q.shape) # Q.shape torch.Size([128, 1])
        out = Q
        return out  # [B, 1]
```

![image-20220822122341752](D:\Typora\文章\RL\MADDPG.assets\image-20220822122341752.png)

![image-20220822123509850](D:\Typora\文章\RL\MADDPG.assets\image-20220822123509850.png)

## 把放在actor_loss计算之前更新critic_loss就行

## list中有tensor转tensor效率

[(45条消息) ValueError:only one element tensors can be converted to Python scalars解决办法_甜度超标°的博客-CSDN博客](https://blog.csdn.net/qq_38703529/article/details/120216078)

```python
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
```



## 后话常看的几个 以及几种buffer

[(45条消息) 强化学习：DDPG到MADDPG_彩虹糖梦的博客-CSDN博客_ddpg和maddpg](https://blog.csdn.net/caozixuan98724/article/details/110854007)

[(45条消息) 强化学习笔记：连续控制与 MADDPG_UQI-LIUWJ的博客-CSDN博客_maddpg离散动作空间](https://blog.csdn.net/qq_40206371/article/details/125113278)

[(45条消息) MADDPG分析及其更新策略见解_Y. F. Zhang的博客-CSDN博客_maddpg](https://blog.csdn.net/weixin_43145941/article/details/112726116)

[MADDPG多智能体场景的Pytorch实现 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/92466991)

[(9 封私信 / 7 条消息) 有哪些常用的多智能体强化学习仿真环境？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/332942236)

[starry-sky6688/MADDPG: Pytorch implementation of the MARL algorithm, MADDPG, which correspondings to the paper "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments". (github.com)](https://github.com/starry-sky6688/MADDPG)

[caozixuan/RL_Learning (github.com)](https://github.com/caozixuan/RL_Learning)

[MADDPG(英文字幕)_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1uY4y1L7Yy?spm_id_from=333.337.search-card.all.click&vd_source=4f73591e958f521068cff7916b99f7cf)

## 关于实现buffer的几种方式

### 整体的数组来做

![image-20220816110529027](D:\Typora\文章\RL\MADDPG.assets\image-20220816110529027.png)

```python
import numpy as np
# 保存s，a，r，s_
    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        # 整理s，s_,方便直接输入网络计算
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)

        #把s, a, [r], s_横向堆叠
        transition = np.hstack((s, a, [r], s_))

        #pointer是记录了曾经有多少数据进来。
        #index是记录当前最新进来的数据位置。
        #所以是一个循环，当MEMORY_CAPACITY满了以后，index就重新在最底开始了
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        #把transition，也就是s, a, [r], s_存进去。
        self.memory[index, :] = transition
        self.pointer += 1
        
# 获取数值
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)    
    #随机BATCH_SIZE个随机数
        bt = self.memory[indices, :]                    
        #根据indices，选取数据bt，相当于随机
        bs = bt[:, :self.s_dim]                         #从bt获得数据s
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]  #从bt获得数据a
        br = bt[:, -self.s_dim - 1:-self.s_dim]         #从bt获得数据r
        bs_ = bt[:, -self.s_dim:]                       #从bt获得数据s'
```

### 直接使用普通的来做就是一行行的放入

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # 经验回放的容量
        self.buffer = []  # 缓冲区
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        '''
            缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  # 随机采出小批量转移
        state, action, reward, next_state, done = zip(*batch)  # 解压成状态，动作等
        # https://blog.csdn.net/ezio23/article/details/81414092
        return state, action, reward, next_state, done

    def __len__(self):
        '''

            返回当前存储的量
        '''
        return len(self.buffer)
```

### 最差的应该是这种全部的

```python
class ReplayMemory:
    def __init__(self, n_s, n_a):
        self.n_s = n_s
        self.n_a = n_a

        self.MEMORY_SIZE = 10000
        self.BATCH_SIZE = 64
        self.all_s = np.empty(shape=(self.MEMORY_SIZE, self.n_s), dtype=np.float64)
        self.all_a = np.random.randint(low=0, high=self.n_a, size=self.MEMORY_SIZE, dtype=np.uint8)
        self.all_r = np.empty(self.MEMORY_SIZE, dtype=np.float64)
        self.all_done = np.random.randint(low=0, high=2, size=self.MEMORY_SIZE, dtype=np.uint8)
        self.all_s_ = np.empty(shape=(self.MEMORY_SIZE, self.n_s), dtype=np.float64)
        self.count = 0
        self.t = 0

        # self.a1 = np.random.randint(low=0,high=)

    def add_memo(self, s, a, r, done, s_):
        self.all_s[self.t] = s
        self.all_a[self.t] = a
        self.all_r[self.t] = r
        self.all_done[self.t] = done
        self.all_s_[self.t] = s_
        self.count = max(self.count, self.t + 1)
        self.t = (self.t + 1) % self.MEMORY_SIZE

    def sample(self):
        if self.count < self.BATCH_SIZE:
            indexes = range(0, self.count)
        else:
            indexes = random.sample(range(0, self.count), self.BATCH_SIZE)

        batch_s = []
        batch_a = []
        batch_r = []
        batch_done = []
        batch_s_ = []
        for idx in indexes:
            batch_s.append(self.all_s[idx])
            batch_a.append(self.all_a[idx])
            batch_r.append(self.all_r[idx])
            batch_done.append(self.all_done[idx])
            batch_s_.append(self.all_s_[idx])

        batch_s_tensor = torch.as_tensor(np.asarray(batch_s), dtype=torch.float32)
        batch_a_tensor = torch.as_tensor(np.asarray(batch_a), dtype=torch.int64).unsqueeze(-1)
        batch_r_tensor = torch.as_tensor(np.asarray(batch_r), dtype=torch.float32).unsqueeze(-1)
        batch_done_tensor = torch.as_tensor(np.asarray(batch_done), dtype=torch.float32).unsqueeze(-1)
        batch_s__tensor = torch.as_tensor(np.asarray(batch_s_), dtype=torch.float32)

        return batch_s_tensor, batch_a_tensor, batch_r_tensor, batch_done_tensor, batch_s__tensor
```

### 这看不太懂

```python
class PolicyBuffer(list):

    def push(self, experience: Experience):
        self.append(experience)

    def sample(self, gamma: float):
        discount_rewards = [0]
        for e in reversed(self):  # ?????为什么要reverse??
            discount_rewards.append(discount_rewards[-1] * gamma + e.r)
        discount_rewards.reverse()
        discount_rewards.pop()

        advantages = np.array(discount_rewards, dtype=np.float32)
        # 这边为什么要进行标准化
        advantages = (advantages - advantages.mean()) / advantages.std()

        trajectories = [(at, e.s, e.a) for at, e in zip(advantages, self)]
        return zip(*trajectories)

```

## 安装make_env

```shell
(PyTorch1.6) D:\Coding\WorkForPC\RL_demo\multiagent-gail-master>pip install -e D:\Coding\WorkForPC\RL_demo\multiagent-gail
-master\multiagent-gail-master\multiagent-particle-envs # 就是进入根目录(看到setup)
# 用的是setup.py目录
# 然后将gym 
pip install gym==0.10.5   # 关键

```

## 一个四元组的压缩遍历

```python
>>> b = [[5, 6], [7, 8], [9, 10], [11, 12]] obs
>>> a = [[1, 2], [2, 3], [4, 5], [0, 1]] obs
>>> c = [1, 2, 3, 4] r
>>> d = [False, False, False, False] done
>>> for i in zip(a, b, c, d):
...     print(i)
...
([1, 2], [5, 6], 1, False)
([2, 3], [7, 8], 2, False)
([4, 5], [9, 10], 3, False)
([0, 1], [11, 12], 4, False)

>>> a = [[1, 2, 2], [2, 3, 3], [4, 5, 5], [0, 1, 5]]
>>> for i in zip(a, b, c, d):
...     print(i)
...
([1, 2, 2], [5, 6], 1, False)  <class 'tuple'>
([2, 3, 3], [7, 8], 2, False)
([4, 5, 5], [9, 10], 3, False)
([0, 1, 5], [11, 12], 4, False)

# 就是不用管每一个的里面的形状是啥样的，只要是对应的agent数量就行

# 更好的结合到buffer中

# 解压回来
>>> for i in zip(a, b, c, d):
...     print(*i)
...
[1, 2, 2] [5, 6] 1 False
[2, 3, 3] [7, 8] 2 False
[4, 5, 5] [9, 10] 3 False
[0, 1, 5] [11, 12] 4 False


>>> a
[[1, 2, 2], [2, 3, 3], [4, 5, 5], [0, 1, 5]]
>>> b
[[5, 6], [7, 8], [9, 10], [11, 12]]
>>> s = zip(a, b) # 压缩为了tuple
>>> i, j = zip(*s)  # 解压出来
>>> i
([1, 2, 2], [2, 3, 3], [4, 5, 5], [0, 1, 5])
>>> j
([5, 6], [7, 8], [9, 10], [11, 12])

```

## 将action和state进行合并

```python
>>> a = torch.tensor([[[2, 3, 4], [1, 1, 1], [2, 2, 2]], [[2, 3, 4], [1, 1, 1], [2, 2, 2]]])  # [2, 3, 3]
>>> a.shape
torch.Size([2, 3, 3])
>>> b = torch.tensor([[[2, 3, 4], [1, 1, 1], [2, 2, 2]], [[2, 3, 4], [1, 1, 1], [2, 2, 2]]])  # [2, 3, 3]
>>> torch.cat([a, b], -1)
tensor([[[2, 3, 4, 2, 3, 4],
         [1, 1, 1, 1, 1, 1],
         [2, 2, 2, 2, 2, 2]],

        [[2, 3, 4, 2, 3, 4],
         [1, 1, 1, 1, 1, 1],
         [2, 2, 2, 2, 2, 2]]])

```

## 一个list内部的合并

```python
>>> a = [[2, 3, 4], [1, 1, 1], [2, 2, 2]]
>>> b = [[2, 3, 4], [1, 1, 1], [2, 2, 2]]
>>> a1 = torch.tensor(a)
>>> b1 = torch.tensor(b)
>>> c = []
>>> c.append(a1)
>>> c.append(b1)
>>> c   [2, 3, 3]  -> [3, 6]  dim=1 
[tensor([[2, 3, 4],
        [1, 1, 1],
        [2, 2, 2]]), tensor([[2, 3, 4],
        [1, 1, 1],
        [2, 2, 2]])]
>>> torch.cat(c, 1)
tensor([[2, 3, 4, 2, 3, 4],  # 这边相当于把每一个agent给合并到一起了
        [1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2]])
>>> c1 = torch.cat(c, 1)
>>> c1.shape
torch.Size([3, 6])

```

## 类似tranformer的多头的操作 先reshape再拆分再换位置再合并

```python
>>> b  # shape[2, 3, 3]  2个agent B为3  每个B里面的东西为3单位的
tensor([[[2, 3, 4],
         [1, 1, 1],
         [2, 2, 2]],

        [[2, 3, 4],
         [1, 1, 1],
         [2, 2, 2]]])
>>> b1 = b.transpose(0,1)  # 先转为3个B 然后是2个agent 这边就是类似时序中的B First 同时扫描
>>> b1
tensor([[[2, 3, 4],
         [2, 3, 4]],

        [[1, 1, 1],
         [1, 1, 1]],

        [[2, 2, 2],
         [2, 2, 2]]])
>>> b1.shape
torch.Size([3, 2, 3])
>>> b2 = b1.reshape(3, -1)
>>> b2
tensor([[2, 3, 4, 2, 3, 4],
        [1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2]])


>>> b3 = b2
>>> b3
tensor([[2, 3, 4, 2, 3, 4],
        [1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2]])
>>> b4 = torch.cat([b2, b3], 1)
>>> b4
tensor([[2, 3, 4, 2, 3, 4,   2, 3, 4, 2, 3, 4], 
        # 相当于agent1/ agent2两个的 s a进行合并
        [1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2,   2, 2, 2, 2, 2, 2]])

```



