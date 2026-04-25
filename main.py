"""
main.py
=======
Entry point for the Data Center RL project.

Usage:
  python main.py              → runs full pipeline then launches dashboard
  python main.py --train-only → train only, no dashboard
  python main.py --dash-only  → launch dashboard (requires trained model)
"""

import argparse
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Data Center Cooling — Model-Based RL"
    )
    parser.add_argument('--train-only', action='store_true',
                        help="Run training pipeline only (no dashboard)")
    parser.add_argument('--dash-only',  action='store_true',
                        help="Launch dashboard only (model must exist)")
    parser.add_argument('--model',      default='linear', choices=['linear', 'neural'],
                        help="Dynamics model type")
    parser.add_argument('--transitions', default=5_000, type=int,
                        help="Transitions to collect during training")
    parser.add_argument('--skip-collect', action='store_true',
                        help="Skip data collection if dataset exists")
    args = parser.parse_args()

    # ── Training ────────────────────────────────────────────────────
    if not args.dash_only:
        from train import run_pipeline
        run_pipeline(
            model_kind=args.model,
            num_transitions=args.transitions,
            skip_collect=args.skip_collect,
        )

    # ── Dashboard ───────────────────────────────────────────────────
    if not args.train_only:
        dashboard_path = os.path.join(
            os.path.dirname(__file__),
            'visualization',
            'dashboard.py',
        )
        print(f"\n🚀  Launching Streamlit dashboard …")
        print(f"   Open your browser at http://localhost:8501\n")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            dashboard_path,
            "--server.headless", "true",
            "--server.port", "8501",
        ])


if __name__ == "__main__":
    main()
