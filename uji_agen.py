import gym
import numpy as np
from training import DQNAgent

# Inisialisasi environment
env = gym.make("LunarLander-v2", render_mode="human")

# Dapatkan ukuran state dan jumlah action
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Inisialisasi agen dan muat model yang telah dilatih
agent = DQNAgent(state_size, action_size)
agent.epsilon = 0.01  # Minimalkan eksplorasi saat testing

# Uji agen pada beberapa episode
for e in range(5):
    state = env.reset()[0]  # Adjust for Gym v26+
    state = np.reshape(state, [1, state_size])

    for time in range(1000):
        env.render()
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        state = np.reshape(next_state, [1, state_size])

        if done:
            print(f"Test Episode: {e+1}, Score: {time}")
            break

env.close()
