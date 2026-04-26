"""
plot_results.py (FIXED + ROBUST)
--------------------------------
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = "/workspaces/hvac-rl"
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
EVAL_CSV = os.path.join(OUTPUTS_DIR, "eval_run", "eval_results.csv")
SINERGYM_DIR = "/workspaces/sinergym/scripts"

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.facecolor": "#0f1117",
        "axes.facecolor": "#1a1d27",
        "axes.edgecolor": "#3a3d4d",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#a0a0b0",
        "ytick.color": "#a0a0b0",
        "text.color": "#e0e0e0",
        "grid.color": "#2a2d3d",
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "font.family": "monospace",
        "legend.facecolor": "#1a1d27",
        "legend.edgecolor": "#3a3d4d",
    }
)

SAC_COLOR = "#00d4ff"
GOOD_COLOR = "#00ff9d"
WARN_COLOR = "#ff6b6b"

# ── Load training CSVs ───────────────────────────────────────────
pattern = os.path.join(SINERGYM_DIR, "*stochastic*", "progress.csv")
csv_files = sorted(glob.glob(pattern))
print(f"Found {len(csv_files)} SAC episode CSV(s)")

dfs = []
for f in csv_files:
    try:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  Loaded: {f} ({len(df)} rows)")
    except Exception as e:
        print(f"  Warning: {e}")

sac_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ── Load eval CSV ────────────────────────────────────────────────
if os.path.exists(EVAL_CSV):
    eval_df = pd.read_csv(EVAL_CSV)
    print(f"Loaded eval results: {len(eval_df)} steps")
else:
    eval_df = pd.DataFrame()
    print("Warning: eval_results.csv not found")

# ── Detect correct column names ──────────────────────────────────
energy_col = next(
    (c for c in ["energy_w", "total_power_demand"] if c in eval_df.columns),
    None,
)

violation_col = next(
    (
        c
        for c in ["temp_violation", "total_temperature_violation"]
        if c in eval_df.columns
    ),
    None,
)

# ── Plot 1: 2x2 Overview ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("SAC HVAC Agent — Results", fontsize=16, color="white")

# ── Training reward
ax = axes[0, 0]
ax.set_title("Training Reward")

if not sac_df.empty:
    reward_col = next(
        (c for c in ["reward", "mean_reward"] if c in sac_df.columns),
        None,
    )
    if reward_col:
        rewards = sac_df[reward_col].dropna().values
        steps = np.arange(len(rewards))
        rolling = pd.Series(rewards).rolling(max(1, len(rewards) // 20)).mean()

        ax.plot(steps, rewards, alpha=0.2)
        ax.plot(steps, rolling, linewidth=2)
        ax.grid(True)
else:
    ax.text(0.5, 0.5, "No training data", ha="center")

# ── Energy plot
ax = axes[0, 1]
ax.set_title("HVAC Energy")

if energy_col:
    energy = eval_df[energy_col].dropna().values
    if len(energy) > 0:
        ds = max(1, len(energy) // 500)
        x = np.arange(len(energy))[::ds]

        ax.plot(x, energy[::ds])
        ax.axhline(np.mean(energy), linestyle="--", label="Mean")
        ax.legend()
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "Empty energy data", ha="center")
else:
    ax.text(0.5, 0.5, "No energy column", ha="center")

# ── Violations plot
ax = axes[1, 0]
ax.set_title("Comfort Violations")

if violation_col:
    violations = eval_df[violation_col].dropna().values
    if len(violations) > 0:
        ds = max(1, len(violations) // 500)
        x = np.arange(len(violations))[::ds]

        pct = (violations == 0).mean() * 100

        ax.plot(x, violations[::ds])
        ax.axhline(0, linestyle="--", label=f"{pct:.1f}% comfortable")
        ax.legend()
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "Empty violation data", ha="center")
else:
    ax.text(0.5, 0.5, "No violation column", ha="center")

# ── Summary panel
ax = axes[1, 1]
ax.set_title("Summary")

summary = {}

if "reward" in eval_df.columns:
    summary["Mean Reward"] = eval_df["reward"].mean()

if energy_col:
    summary["Mean Energy"] = eval_df[energy_col].dropna().mean()

if violation_col:
    v = eval_df[violation_col].dropna()
    if len(v) > 0:
        summary["Comfort %"] = (v == 0).mean() * 100
        summary["Mean Violation"] = v.mean()

if summary:
    labels = list(summary.keys())
    values = list(summary.values())

    ax.barh(labels, values)
    for i, v in enumerate(values):
        ax.text(v, i, f"{v:.2f}", va="center")

    ax.grid(True)
else:
    ax.text(0.5, 0.5, "No summary data", ha="center")

plt.tight_layout()

out1 = os.path.join(PLOTS_DIR, "training_results.png")
plt.savefig(out1)
print(f"Saved: {out1}")
plt.close()

# ── Reward distribution ──────────────────────────────────────────
if "reward" in eval_df.columns:
    rewards = eval_df["reward"].dropna().values

    if len(rewards) > 0:
        plt.figure(figsize=(10, 5))
        plt.hist(rewards, bins=50)
        plt.axvline(np.mean(rewards), linestyle="--", label="Mean")
        plt.legend()

        out2 = os.path.join(PLOTS_DIR, "reward_distribution.png")
        plt.savefig(out2)
        print(f"Saved: {out2}")
        plt.close()

# ── Terminal summary ─────────────────────────────────────────────
print("\n===== RESULTS SUMMARY =====")

if "reward" in eval_df.columns:
    print(f"Total reward: {eval_df['reward'].sum():.2f}")

if energy_col:
    print(f"Mean HVAC power: {eval_df[energy_col].mean():.2f} W")

if violation_col:
    v = eval_df[violation_col].dropna()
    if len(v) > 0:
        print(f"Comfort %: {(v==0).mean()*100:.2f}%")

print("==========================")
