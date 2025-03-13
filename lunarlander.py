import gym
from training import DQNAgent

# Inisialisasi environment LunarLander-v2
env = gym.make("LunarLander-v2")

# Dapatkan ukuran state dan jumlah action
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Inisialisasi agen
agent = DQNAgent(state_size, action_size)

# Latih agen
agent.train(env, episodes=1000, batch_size=64, target_update_freq=10)

