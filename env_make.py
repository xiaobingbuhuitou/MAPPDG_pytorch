"""

多智能体环境
"""


def make_env(num_adversaries=1):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    scenario = scenarios.load('simple_tag.py').Scenario()

    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    # env = MultiAgentEnv(world)
    n_players = env.n  # 包含敌人的所有玩家个数
    n_agents = env.n - num_adversaries
    # 需要操控的玩家个数，虽然敌人也可以控制，但是双方都学习的话需要不同的算法
    obs_shape = [env.observation_space[i].shape[0] for i in range(n_agents)]  # 每一维代表该agent的obs维度
    action_shape = []
    for content in env.action_space:
        action_shape.append(content.n)
    action_shape = action_shape[:n_agents]  # 每一维代表该agent的act维度
    # env.seed(1)
    return env, n_players, n_agents, obs_shape[0], action_shape[0]  # agent的状态和动作空间都一样就传一个
