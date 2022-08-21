from env_make import make_env

n_episode = 3
n_step = 100
e_name = 'simple_tag'
# num_adversaries = 1
create_env, n_players, n_agents, n_state_shape, n_action_shape = make_env()
# print("n_state_shape", n_state_shape, "n_action_shape", n_action_shape) # n_state_shape 16 n_action_shape 5
batch_size = 128
max_size = 1000

evaluate_episodes = 10
evaluate_episode_len = 100

eps_greedy = 0.3
action_high = 1
action_low = -1

lr = 0.001
tau = 0.03
gamma = 0.95

save_dir = './model'
