"""
train.py
========
End-to-end training pipeline:
  1. Collect exploration data from the environment
  2. Train linear and/or neural dynamics models
  3. Evaluate controllers and print a summary table
  4. Save all artifacts for use by the dashboard

Run:
    python train.py
    python train.py --model neural --transitions 8000
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from env.cooling_env import DataCenterEnv
from data.data_collector import DataCollector, DATASET_FILE
from models.dynamics_model import (
    get_model, default_model_path, LinearDynamicsModel, NeuralDynamicsModel,
    LINEAR_PATH, NEURAL_PATH,
)
from controllers.mpc_controller import get_controller


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_controller(
    ctrl_name:      str,
    dynamics_model=None,
    num_steps:      int = 200,
    seed:           int = 7,
) -> dict:
    """Run one episode with a controller and return metrics."""
    env   = DataCenterEnv(seed=seed)
    ctrl  = get_controller(ctrl_name, dynamics_model=dynamics_model, seed=seed)
    state = env.reset()
    ctrl.reset()

    temps, energies, rewards = [], [], []

    for _ in range(num_steps):
        action = ctrl.select_action(state)
        state, reward, done, info = env.step(action)
        temps.append(info['mean_temp'])
        energies.append(info['energy'])
        rewards.append(reward)
        if done:
            break

    return {
        "controller":   ctrl.name,
        "avg_temp":     float(np.mean(temps)),
        "max_temp":     float(np.max(temps)),
        "min_temp":     float(np.min(temps)),
        "temp_std":     float(np.std(temps)),
        "total_energy": float(np.sum(energies)),
        "avg_reward":   float(np.mean(rewards)),
        "steps":        len(temps),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    model_kind:   str = 'linear',
    num_transitions: int = 5_000,
    eval_steps:   int = 200,
    skip_collect: bool = False,
    verbose:      bool = True,
) -> None:
    t_start = time.time()

    print("\n" + "=" * 60)
    print("  🏭  Data Center RL — Training Pipeline")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Step 1: Data collection
    # ----------------------------------------------------------------
    if skip_collect and os.path.exists(DATASET_FILE):
        print(f"\n⏩  Skipping collection – loading existing dataset …")
        states, actions, next_states = DataCollector.load(DATASET_FILE)
    else:
        print(f"\n📡  Step 1/3 — Collecting {num_transitions} transitions …")
        collector = DataCollector(num_transitions=num_transitions, seed=42)
        states, actions, next_states = collector.collect(verbose=verbose)
        collector.save(states, actions, next_states)

    # ----------------------------------------------------------------
    # Step 2: Train model
    # ----------------------------------------------------------------
    print(f"\n🧠  Step 2/3 — Training {model_kind.upper()} dynamics model …")
    model = get_model(model_kind)
    metrics = model.fit(states, actions, next_states, verbose=verbose)
    save_path = default_model_path(model_kind)
    model.save(save_path)

    # ----------------------------------------------------------------
    # Step 3: Evaluate all controllers
    # ----------------------------------------------------------------
    print(f"\n🏁  Step 3/3 — Evaluating controllers ({eval_steps} steps each) …")
    results = []
    for ctrl_name in ['none', 'random', 'mpc']:
        r = evaluate_controller(
            ctrl_name,
            dynamics_model=model,
            num_steps=eval_steps,
        )
        results.append(r)

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    print("\n" + "─" * 70)
    print(f"  {'Controller':<16} {'Avg °C':>7} {'Max °C':>7} {'Std °C':>7} "
          f"{'Energy':>9} {'Avg Reward':>12}")
    print("─" * 70)
    for r in results:
        print(f"  {r['controller']:<16} "
              f"{r['avg_temp']:>7.2f} "
              f"{r['max_temp']:>7.2f} "
              f"{r['temp_std']:>7.3f} "
              f"{r['total_energy']:>9.1f} "
              f"{r['avg_reward']:>12.4f}")
    print("─" * 70)

    elapsed = time.time() - t_start
    print(f"\n✅  Pipeline complete in {elapsed:.1f}s")
    print(f"   Model saved  → {save_path}")
    print(f"   Dataset saved → {DATASET_FILE}\n")

    return results


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Center RL – training pipeline")
    parser.add_argument('--model',       default='linear', choices=['linear', 'neural'],
                        help="Dynamics model backend")
    parser.add_argument('--transitions', default=5_000, type=int,
                        help="Number of transitions to collect")
    parser.add_argument('--eval-steps',  default=200, type=int,
                        help="Steps per controller evaluation episode")
    parser.add_argument('--skip-collect', action='store_true',
                        help="Reuse existing dataset if present")
    args = parser.parse_args()

    run_pipeline(
        model_kind=args.model,
        num_transitions=args.transitions,
        eval_steps=args.eval_steps,
        skip_collect=args.skip_collect,
    )
