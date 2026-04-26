"""
server.py — Enhanced HVAC RL Dashboard Server
----------------------------------------------
Serves real training data from Sinergym CSV files and eval results.

Run from inside the Dev Container:
    cd /workspaces/sinergym/scripts
    pip install flask flask-cors
    python server.py

Open dashboard.html in browser at: http://localhost:5000
API debug endpoint: http://localhost:5000/api/debug
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = "/workspaces/hvac-rl"
SINERGYM_DIR = "/workspaces/sinergym/scripts"
EVAL_CSV = os.path.join(BASE_DIR, "outputs", "eval_run", "eval_results.csv")
EVAL_NPZ = os.path.join(BASE_DIR, "outputs", "eval", "evaluations.npz")
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ── Helpers ──────────────────────────────────────────────────────
def load_progress_csvs():
    """Load and concatenate all progress.csv files from all runs."""
    pattern = os.path.join(SINERGYM_DIR, "*stochastic*", "progress.csv")
    files = sorted(
        glob.glob(pattern),
        key=lambda f: int(
            "".join(filter(str.isdigit, os.path.basename(os.path.dirname(f)))) or 0
        ),
    )
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_float(val):
    try:
        v = float(val)
        return None if np.isnan(v) or np.isinf(v) else round(v, 4)
    except Exception:
        return None


def downsample(lst, max_pts=300):
    if len(lst) <= max_pts:
        return lst
    step = len(lst) // max_pts
    return lst[::step]


# ── /api/training ─────────────────────────────────────────────────
@app.route("/api/training")
def api_training():
    df = load_progress_csvs()
    if df.empty:
        return jsonify({"error": "No training CSV data found", "data": []})

    col = find_col(df, ["mean_reward", "reward", "Reward", "ep_rew_mean"])
    if not col:
        return jsonify(
            {"error": f"No reward column. Found: {list(df.columns)}", "data": []}
        )

    rewards = df[col].dropna().tolist()
    sampled = downsample(rewards)
    labels = list(range(0, len(rewards), max(1, len(rewards) // len(sampled))))

    w = max(1, len(sampled) // 20)
    rolling = pd.Series(sampled).rolling(w, min_periods=1).mean().round(4).tolist()

    return jsonify(
        {
            "labels": labels,
            "raw": [round(r, 4) for r in sampled],
            "rolling": rolling,
            "total_steps": len(rewards),
            "columns": list(df.columns),
        }
    )


# ── /api/training_detail ─────────────────────────────────────────
@app.route("/api/training_detail")
def api_training_detail():
    df = load_progress_csvs()
    if df.empty:
        return jsonify({"error": "No training data", "episodes": []})

    episodes = []
    for _, row in df.iterrows():
        ep = {
            "episode_num": int(row.get("episode_num", 0)),
            "mean_reward": safe_float(row.get("mean_reward")),
            "mean_energy_penalty": safe_float(row.get("mean_energy_penalty")),
            "mean_comfort_penalty": safe_float(row.get("mean_comfort_penalty")),
            "mean_power_demand": safe_float(row.get("mean_power_demand")),
            "mean_temperature_violation": safe_float(
                row.get("mean_temperature_violation")
            ),
            "comfort_violation_pct": safe_float(row.get("comfort_violation_time(%)")),
            "length_timesteps": safe_float(row.get("length(timesteps)")),
            "cumulative_power_demand": safe_float(row.get("cumulative_power_demand")),
        }
        episodes.append(ep)

    return jsonify({"episodes": episodes, "total_episodes": len(episodes)})


# ── /api/eval_checkpoints ────────────────────────────────────────
@app.route("/api/eval_checkpoints")
def api_eval_checkpoints():
    if not os.path.exists(EVAL_NPZ):
        return jsonify({"error": "evaluations.npz not found", "data": []})
    try:
        data = np.load(EVAL_NPZ)
        timesteps = data["timesteps"].tolist()
        results = data["results"].mean(axis=1).round(4).tolist()
        return jsonify(
            {
                "labels": [f"{int(t / 1000)}k" for t in timesteps],
                "rewards": results,
                "timesteps": timesteps,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e), "data": []})


# ── /api/eval_year ───────────────────────────────────────────────
@app.route("/api/eval_year")
def api_eval_year():
    if not os.path.exists(EVAL_CSV):
        return jsonify(
            {"error": "eval_results.csv not found — run eval.py first", "data": []}
        )
    try:
        df = pd.read_csv(EVAL_CSV)
        result = {"total_steps": len(df), "columns": list(df.columns)}

        if "reward" in df.columns:
            rewards = df["reward"].dropna()
            sampled = downsample(rewards.tolist())
            result["reward_labels"] = list(
                range(0, len(rewards), max(1, len(rewards) // len(sampled)))
            )
            result["reward_raw"] = [round(r, 4) for r in sampled]
            result["reward_mean"] = round(float(rewards.mean()), 4)
            result["reward_total"] = round(float(rewards.sum()), 2)

        if "energy_w" in df.columns:
            energy = df["energy_w"].dropna()
            if len(energy) > 0 and not energy.isna().all():
                sampled_e = downsample(energy.tolist())
                result["energy_labels"] = list(
                    range(0, len(energy), max(1, len(energy) // len(sampled_e)))
                )
                result["energy_values"] = [round(float(v), 1) for v in sampled_e]
                result["energy_mean"] = round(float(energy.mean()), 1)

        if "temp_violation" in df.columns:
            viol = df["temp_violation"].dropna()
            if len(viol) > 0 and not viol.isna().all():
                sampled_v = downsample(viol.tolist())
                result["viol_labels"] = list(
                    range(0, len(viol), max(1, len(viol) // len(sampled_v)))
                )
                result["viol_values"] = [round(float(v), 3) for v in sampled_v]
                result["viol_mean"] = round(float(viol.mean()), 4)
                result["comfort_pct"] = round(float((viol == 0).mean() * 100), 1)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "data": []})


# ── /api/sim_data ─────────────────────────────────────────────────
@app.route("/api/sim_data")
def api_sim_data():
    """
    Loads observation and reward data from the latest eval episode
    for the simulation playback. Returns downsampled data for performance.
    """
    # Find latest episode with monitor data
    pattern = os.path.join(
        SINERGYM_DIR, "*stochastic*", "episode-*", "monitor", "observations.csv"
    )
    files = sorted(glob.glob(pattern))

    if not files:
        return jsonify(
            {"error": "No observation data found. Run eval.py first.", "steps": 0}
        )

    # Prefer the eval run (res39 or latest)
    eval_obs_candidates = [
        f for f in files if "res39" in f or "res38" in f or "res37" in f
    ]
    obs_file = eval_obs_candidates[-1] if eval_obs_candidates else files[-1]
    ep_dir = os.path.dirname(obs_file)

    try:
        obs_df = pd.read_csv(obs_file)
        reward_file = os.path.join(ep_dir, "rewards.csv")
        reward_df = (
            pd.read_csv(reward_file) if os.path.exists(reward_file) else pd.DataFrame()
        )

        # Downsample to max 3000 points for browser performance
        step = max(1, len(obs_df) // 3000)
        obs_sampled = obs_df.iloc[::step].reset_index(drop=True)

        rewards_list = []
        if not reward_df.empty and "reward" in reward_df.columns:
            rewards_sampled = reward_df["reward"].iloc[::step].tolist()
            rewards_list = [round(float(r), 4) for r in rewards_sampled]
        else:
            rewards_list = [0.0] * len(obs_sampled)

        # Convert observations to list of dicts
        obs_records = []
        for _, row in obs_sampled.iterrows():
            rec = {}
            for col in obs_sampled.columns:
                val = row[col]
                if pd.notna(val):
                    try:
                        rec[col] = round(float(val), 3)
                    except Exception:
                        rec[col] = val
            obs_records.append(rec)

        # Ensure same length
        n = min(len(obs_records), len(rewards_list))
        obs_records = obs_records[:n]
        rewards_list = rewards_list[:n]

        return jsonify(
            {
                "steps": n,
                "step_size": step,
                "total_original_steps": len(obs_df),
                "observations": obs_records,
                "rewards": rewards_list,
                "columns": list(obs_df.columns),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e), "steps": 0})


# ── /api/model_info ───────────────────────────────────────────────
@app.route("/api/model_info")
def api_model_info():
    checkpoints = sorted(
        glob.glob(os.path.join(MODELS_DIR, "sac_hvac_*.zip")),
        key=lambda x: int(x.split("_")[-2]) if x.split("_")[-2].isdigit() else 0,
    )
    best_exists = os.path.exists(os.path.join(MODELS_DIR, "best", "best_model.zip"))
    final_exists = os.path.exists(os.path.join(MODELS_DIR, "sac_hvac_final.zip"))
    latest_step = 0
    if checkpoints:
        try:
            latest_step = int(checkpoints[-1].split("_")[-2])
        except Exception:
            pass

    return jsonify(
        {
            "checkpoints": len(checkpoints),
            "latest_step": latest_step,
            "best_exists": best_exists,
            "final_exists": final_exists,
            "checkpoint_list": [os.path.basename(c) for c in checkpoints],
        }
    )


# ── /api/debug ────────────────────────────────────────────────────
@app.route("/api/debug")
def api_debug():
    progress_csvs = glob.glob(
        os.path.join(SINERGYM_DIR, "*stochastic*", "progress.csv")
    )
    obs_csvs = glob.glob(
        os.path.join(
            SINERGYM_DIR, "*stochastic*", "episode-*", "monitor", "observations.csv"
        )
    )
    return jsonify(
        {
            "progress_csvs": progress_csvs,
            "obs_csvs_count": len(obs_csvs),
            "eval_csv": EVAL_CSV,
            "eval_csv_exists": os.path.exists(EVAL_CSV),
            "eval_npz_exists": os.path.exists(EVAL_NPZ),
            "models_dir": MODELS_DIR,
            "models_exist": os.path.exists(MODELS_DIR),
        }
    )


# ── Serve dashboard ───────────────────────────────────────────────
@app.route("/")
def serve_dashboard():
    return send_from_directory(
        os.path.dirname(os.path.abspath(__file__)), "dashboard.html"
    )


if __name__ == "__main__":
    print("=" * 55)
    print("  HVAC RL Dashboard Server — Enhanced")
    print("=" * 55)
    print(f"  Progress CSVs : {SINERGYM_DIR}/*stochastic*/progress.csv")
    print(f"  Eval CSV      : {EVAL_CSV}")
    print(f"  Models dir    : {MODELS_DIR}")
    print("=" * 55)
    print("  Dashboard     : http://localhost:5000")
    print("  API debug     : http://localhost:5000/api/debug")
    print("  Sim data      : http://localhost:5000/api/sim_data")
    print("  Training      : http://localhost:5000/api/training")
    print("  Detail        : http://localhost:5000/api/training_detail")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False)
