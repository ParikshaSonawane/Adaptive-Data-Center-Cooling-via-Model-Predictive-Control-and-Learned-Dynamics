"""
data_collector.py
=================
Runs safe random exploration of the DataCenterEnv to generate a labelled
(state, action, next_state) dataset used to train the dynamics model.

Key design choices
------------------
* Smooth action changes  – actions are low-pass filtered so the controller
  never makes sudden jumps (avoids unrealistic cliffs in the dataset).
* Episode restarts       – environment is reset periodically so the dataset
  covers a wide range of temperature conditions.
* Configurable budget    – caller decides total transitions collected.
* Numpy .npz persistence – dataset saved / loaded from disk in one file.
"""

import numpy as np
import os
from typing import Optional, Tuple

# Local import
from env.cooling_env import DataCenterEnv


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------
DEFAULT_NUM_TRANSITIONS = 5_000
DEFAULT_EPISODE_LENGTH  = 100      # steps before env reset
DEFAULT_ACTION_SMOOTH   = 0.3      # EMA smoothing factor  (0=constant, 1=no smooth)
DATA_DIR                = os.path.join(os.path.dirname(__file__), 'collected')
DATASET_FILE            = os.path.join(DATA_DIR, 'dynamics_dataset.npz')


class DataCollector:
    """
    Collects (state, action, next_state) transitions via smooth random policy.

    Parameters
    ----------
    num_transitions : int   – total transitions to collect
    episode_length  : int   – steps per episode before env reset
    action_smooth   : float – EMA coefficient for smoothing actions
    seed            : int   – RNG seed for reproducibility
    """

    def __init__(
        self,
        num_transitions: int = DEFAULT_NUM_TRANSITIONS,
        episode_length:  int = DEFAULT_EPISODE_LENGTH,
        action_smooth:   float = DEFAULT_ACTION_SMOOTH,
        seed:            Optional[int] = 0,
    ):
        self.num_transitions = num_transitions
        self.episode_length  = episode_length
        self.action_smooth   = action_smooth
        self.seed            = seed

        self.env = DataCenterEnv(seed=seed)
        self.action_dim  = self.env.get_action_dim()
        self.state_dim   = self.env.get_state_dim()
        self.rng         = np.random.default_rng(seed)

        # Containers
        self.states:      list = []
        self.actions:     list = []
        self.next_states: list = []
        self.rewards:     list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def collect(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run exploration and return (states, actions, next_states).
        """
        self.states.clear()
        self.actions.clear()
        self.next_states.clear()
        self.rewards.clear()

        state      = self.env.reset()
        action     = self.rng.uniform(0, 1, self.action_dim)  # initial action
        step_in_ep = 0
        collected  = 0

        while collected < self.num_transitions:
            # --- Smooth action update ---
            raw_action  = self.rng.uniform(0, 1, self.action_dim)
            action      = (1 - self.action_smooth) * action + self.action_smooth * raw_action
            action_clip = np.clip(action, 0.0, 1.0)

            # --- Env step ---
            next_state, reward, done, info = self.env.step(action_clip)

            # --- Store transition ---
            self.states.append(state.copy())
            self.actions.append(action_clip.copy())
            self.next_states.append(next_state.copy())
            self.rewards.append(reward)

            state       = next_state
            step_in_ep += 1
            collected  += 1

            # --- Episode reset ---
            if step_in_ep >= self.episode_length or done:
                state      = self.env.reset()
                step_in_ep = 0
                # Randomise action on fresh episode
                action = self.rng.uniform(0, 1, self.action_dim)

            if verbose and collected % 500 == 0:
                print(f"  Collected {collected:5d}/{self.num_transitions}  "
                      f"mean_temp={info['mean_temp']:.2f}°C")

        states_arr      = np.array(self.states,      dtype=np.float32)
        actions_arr     = np.array(self.actions,     dtype=np.float32)
        next_states_arr = np.array(self.next_states, dtype=np.float32)
        rewards_arr     = np.array(self.rewards,     dtype=np.float32)

        if verbose:
            print(f"\n✅ Collection complete.")
            print(f"   states shape      : {states_arr.shape}")
            print(f"   actions shape     : {actions_arr.shape}")
            print(f"   next_states shape : {next_states_arr.shape}")

        return states_arr, actions_arr, next_states_arr

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(
        self,
        states:      np.ndarray,
        actions:     np.ndarray,
        next_states: np.ndarray,
        filepath:    str = DATASET_FILE,
    ) -> str:
        """Save dataset to .npz file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez_compressed(
            filepath,
            states=states,
            actions=actions,
            next_states=next_states,
            rewards=np.array(self.rewards, dtype=np.float32),
        )
        print(f"💾 Dataset saved → {filepath}")
        return filepath

    @staticmethod
    def load(filepath: str = DATASET_FILE) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load dataset from .npz file. Returns (states, actions, next_states)."""
        data = np.load(filepath)
        states      = data['states']
        actions     = data['actions']
        next_states = data['next_states']
        print(f"📂 Dataset loaded from {filepath}")
        print(f"   {len(states)} transitions  |  "
              f"state_dim={states.shape[1]}  |  action_dim={actions.shape[1]}")
        return states, actions, next_states

    # ------------------------------------------------------------------
    # Statistics helper
    # ------------------------------------------------------------------
    @staticmethod
    def dataset_stats(
        states:      np.ndarray,
        actions:     np.ndarray,
        next_states: np.ndarray,
    ) -> dict:
        """Return a dict of basic dataset statistics."""
        delta = next_states - states
        return {
            "n_transitions":      len(states),
            "state_dim":          states.shape[1],
            "action_dim":         actions.shape[1],
            "state_mean":         float(states.mean()),
            "state_std":          float(states.std()),
            "action_mean":        float(actions.mean()),
            "delta_mean":         float(delta.mean()),
            "delta_std":          float(delta.std()),
            "max_delta":          float(np.abs(delta).max()),
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("  Data Center RL — Data Collection")
    print("=" * 55)

    collector = DataCollector(num_transitions=2_000, seed=42)
    states, actions, next_states = collector.collect(verbose=True)
    collector.save(states, actions, next_states)

    stats = DataCollector.dataset_stats(states, actions, next_states)
    print("\n📊 Dataset Statistics:")
    for k, v in stats.items():
        print(f"   {k:<22}: {v}")
