import gymnasium as gym
from stable_baselines3 import SAC

import sinergym

print("=" * 50)
print("Sinergym stack verification")
print("=" * 50)

env = gym.make("Eplus-5zone-hot-continuous-stochastic-v1")
obs, info = env.reset()

print(f"Sinergym version : {sinergym.__version__}")
print(f"Obs shape        : {obs.shape}")
print(f"Action space     : {env.action_space}")
print(f"Obs space        : {env.observation_space}")

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"Reward           : {round(reward, 4)}")
print(f"Info keys        : {list(info.keys())}")

env.close()
print("=" * 50)
print("All good — ready to train.")
print("=" * 50)
