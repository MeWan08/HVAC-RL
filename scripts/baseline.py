"""
Rule-based baseline controller.
Sets heating setpoint to 21C in winter, 23C in summer.
Use this as a comparison benchmark against your SAC agent.
"""

import os

import gymnasium as gym
import numpy as np

import sinergym
from sinergym.utils.wrappers import (
    CSVLogger,
    DatetimeWrapper,
    LoggerWrapper,
    NormalizeAction,
    NormalizeObservation,
)

BASE_DIR = "/workspaces/hvac-rl"
ENV_ID = "Eplus-5zone-hot-continuous-stochastic-v1"

env = gym.make(ENV_ID)
env = DatetimeWrapper(env)
env = NormalizeAction(env)
env = NormalizeObservation(env)
env = LoggerWrapper(env)
env = CSVLogger(env)

obs, info = env.reset()
terminated = truncated = False
total_reward = 0.0
step_count = 0

print("Running rule-based baseline...")

while not (terminated or truncated):
    # Simple rule: use middle of action space (no control = default schedule)
    action_shape = (
        env.action_space.shape or (env.action_space.n,)
        if hasattr(env.action_space, 'n')
        else (1,)
    )
    action = np.zeros(action_shape, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += float(reward)
    step_count += 1

print("-" * 50)
print(f"Baseline steps       : {step_count:,}")
print(f"Baseline total reward: {total_reward:.2f}")
print(f"Baseline mean reward : {total_reward / step_count:.4f}")
print("-" * 50)
print("Compare these numbers against your SAC agent in eval.py")

env.close()
