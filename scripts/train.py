import glob
import os

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv

import sinergym
from sinergym.utils.wrappers import (
    CSVLogger,
    DatetimeWrapper,
    LoggerWrapper,
    NormalizeAction,
    NormalizeObservation,
)

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = "/workspaces/hvac-rl"
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
TB_DIR = os.path.join(OUTPUTS_DIR, "tb")
EVAL_DIR = os.path.join(OUTPUTS_DIR, "eval")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────
ENV_ID = "Eplus-5zone-hot-continuous-stochastic-v1"
TOTAL_TIMESTEPS = 25_000
ENERGY_WEIGHT = 0.5


# ── Environment factory ──────────────────────────────────────────
def make_env(eval: bool = False):
    env = gym.make(
        ENV_ID,
        reward_kwargs={
            "energy_weight": ENERGY_WEIGHT,
            "lambda_energy": 1e-4,
            "lambda_temperature": 1.0,
            "temperature_variables": ["air_temperature"],
            "energy_variables": ["HVAC_electricity_demand_rate"],
            "range_comfort_winter": (20.0, 23.5),
            "range_comfort_summer": (23.0, 26.0),
        },
    )
    env = DatetimeWrapper(env)
    env = NormalizeAction(env)
    env = NormalizeObservation(env)
    if not eval:
        env = LoggerWrapper(env)
        env = CSVLogger(env)
    return env


# ── Envs ─────────────────────────────────────────────────────────
train_env = DummyVecEnv([make_env])
eval_env = DummyVecEnv([lambda: make_env(eval=True)])

# ── Callbacks ────────────────────────────────────────────────────
checkpoint_cb = CheckpointCallback(
    save_freq=2_500,
    save_path=MODELS_DIR,
    name_prefix="sac_hvac",
    verbose=1,
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(MODELS_DIR, "best"),
    log_path=EVAL_DIR,
    eval_freq=5_000,
    n_eval_episodes=1,
    deterministic=True,
    verbose=1,
)

# ── Resume from checkpoint if one exists ────────────────────────
checkpoint_files = sorted(
    glob.glob(os.path.join(MODELS_DIR, "sac_hvac_*.zip")),
    key=lambda x: int(x.split("_")[-2]),  # ← sort numerically not alphabetically
)

if checkpoint_files:
    latest = checkpoint_files[-1]
    steps_done = int(latest.split("_")[-2])
    REMAINING_STEPS = max(0, TOTAL_TIMESTEPS - steps_done)
    print(f"Resuming from checkpoint: {latest}")
    print(f"Steps already done : {steps_done:,}")
    print(f"Steps remaining    : {REMAINING_STEPS:,}")
    model = SAC.load(latest, env=train_env, tensorboard_log=TB_DIR)
else:
    print("No checkpoint found, starting fresh.")
    REMAINING_STEPS = TOTAL_TIMESTEPS
    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=25_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        tensorboard_log=TB_DIR,
    )

print(f"Training SAC on {ENV_ID}")
print(f"Total timesteps : {TOTAL_TIMESTEPS:,}")
print(f"Energy weight   : {ENERGY_WEIGHT}")
print(f"Models saved to : {MODELS_DIR}")
print(f"TensorBoard     : {TB_DIR}")
print("-" * 50)

# ── Train ────────────────────────────────────────────────────────
model.learn(
    total_timesteps=REMAINING_STEPS,
    callback=[checkpoint_cb, eval_cb],
    progress_bar=True,
    reset_num_timesteps=False,
)

model.save(os.path.join(MODELS_DIR, "sac_hvac_final"))
train_env.close()
eval_env.close()
print("Training complete. Model saved.")
