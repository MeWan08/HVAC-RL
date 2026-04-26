"""
eval.py
-------
Loads the best trained SAC model and runs one full year evaluation.
Prints a summary table and saves results to outputs/eval_results.csv

Run from inside the Dev Container:
    cd /workspaces/sinergym/scripts
    python eval.py
"""

import os

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
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
EVAL_OUT_DIR = os.path.join(OUTPUTS_DIR, "eval_run")
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────
ENV_ID = "Eplus-5zone-hot-continuous-stochastic-v1"
ENERGY_WEIGHT = 0.5

REWARD_KWARGS = {
    "energy_weight": ENERGY_WEIGHT,
    "lambda_energy": 1e-4,
    "lambda_temperature": 1.0,
    "temperature_variables": ["air_temperature"],
    "energy_variables": ["HVAC_electricity_demand_rate"],
    "range_comfort_winter": (20.0, 23.5),
    "range_comfort_summer": (23.0, 26.0),
}

# ── Load best model (falls back to final if best not found) ──────
best_path = os.path.join(MODELS_DIR, "best", "best_model.zip")
final_path = os.path.join(MODELS_DIR, "sac_hvac_final.zip")

if os.path.exists(best_path):
    model_path = best_path
    print(f"Loading best model : {best_path}")
elif os.path.exists(final_path):
    model_path = final_path
    print(f"Loading final model: {final_path}")
else:
    raise FileNotFoundError(
        "No trained model found. Run train.py first.\n"
        f"Looked in:\n  {best_path}\n  {final_path}"
    )


# ── Build eval environment ───────────────────────────────────────
def make_eval_env():
    env = gym.make(ENV_ID, reward_kwargs=REWARD_KWARGS)
    env = DatetimeWrapper(env)
    env = NormalizeAction(env)
    env = NormalizeObservation(env)
    env = LoggerWrapper(env)
    env = CSVLogger(env)
    return env


eval_env = make_eval_env()
model = SAC.load(model_path, env=eval_env)

# ── Run one full year episode ────────────────────────────────────
print("\nRunning full-year evaluation (35,040 steps)...")
print("This will take a while — same as one training episode.\n")

obs, info = eval_env.reset()
terminated = truncated = False

rewards = []
energy_vals = []
temp_violations = []
step = 0

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)

    rewards.append(float(reward))

    energy_vals.append(info.get("total_power_demand", np.nan))
    temp_violations.append(info.get("total_temperature_violation", np.nan))

    step += 1

eval_env.close()

# ── Print summary ────────────────────────────────────────────────
total_reward = sum(rewards)
mean_reward = np.mean(rewards)
mean_energy = np.mean(energy_vals) if energy_vals else float("nan")
mean_violation = np.mean(temp_violations) if temp_violations else float("nan")
pct_comfortable = (
    sum(1 for v in temp_violations if v == 0) / len(temp_violations) * 100
    if temp_violations
    else float("nan")
)

print("\n" + "=" * 55)
print("  SAC AGENT — FULL YEAR EVALUATION RESULTS")
print("=" * 55)
print(f"  Total steps          : {step:,}")
print(f"  Total reward         : {total_reward:,.2f}")
print(f"  Mean reward/step     : {mean_reward:.4f}")
print(f"  Mean HVAC power (W)  : {mean_energy:,.1f}")
print(f"  Mean temp violation  : {mean_violation:.4f} °C")
print(f"  Time in comfort zone : {pct_comfortable:.1f}%")
print("=" * 55)
print(f"\nResults saved to: {EVAL_OUT_DIR}")

# ── Save results CSV ─────────────────────────────────────────────
results_df = pd.DataFrame(
    {
        "step": range(1, step + 1),
        "reward": rewards,
        "energy_w": energy_vals if energy_vals else [None] * step,
        "temp_violation": temp_violations if temp_violations else [None] * step,
    }
)
results_df.to_csv(os.path.join(EVAL_OUT_DIR, "eval_results.csv"), index=False)
print("Saved eval_results.csv")
