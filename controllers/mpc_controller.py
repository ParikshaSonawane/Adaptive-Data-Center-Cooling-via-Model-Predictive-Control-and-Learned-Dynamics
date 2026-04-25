"""
mpc_controller.py
=================
Model Predictive Control (MPC) controller for data center cooling.

Algorithm  (receding horizon)
--------------------------
At each time step t:
  1. Sample N random candidate action sequences of length H (horizon).
  2. Roll out each sequence using the learned dynamics model.
  3. Score each sequence by summing:
       - Temperature deviation penalty  (mean squared distance from target)
       - Energy cost  (quadratic in cooling power)
  4. Apply the first action of the best-scoring sequence.
  5. Repeat next step with updated state (receding horizon).

This is a *random shooting* MPC variant – simple, parallelisable, and
effective for moderate-dimensional control tasks.

Other controllers included for comparison:
  NoController   – always returns zero action (baseline overheating)
  RandomController – smooth random exploration
"""

import numpy as np
from typing import Optional, Tuple

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from env.cooling_env import (
    COMFORT_BAND_HIGH, COMFORT_BAND_LOW, TARGET_TEMP, COOLING_ENERGY_FACTOR, NUM_ZONES
)
from models.dynamics_model import BaseDynamicsModel


# ---------------------------------------------------------------------------
# MPC hyper-parameters
# ---------------------------------------------------------------------------
HORIZON        = 5      # planning horizon  H  (steps into the future)
NUM_CANDIDATES = 200    # number of random action sequences to evaluate
OVERHEAT_WEIGHT = 5.0   # strong penalty when predicted mean temp exceeds target
OVERCOOL_WEIGHT = 1.0   # weaker penalty when predicted mean temp is below comfort
ENERGY_WEIGHT  = 0.5    # weight on energy consumption penalty
ACTION_SMOOTH  = 0.4    # EMA smoothing between consecutive actions


# ---------------------------------------------------------------------------
# Base Controller
# ---------------------------------------------------------------------------
class BaseController:
    """Minimal interface all controllers must satisfy."""

    def __init__(self, action_dim: int = NUM_ZONES):
        self.action_dim = action_dim

    def select_action(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self):
        """Called at the start of each episode."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# No-control baseline
# ---------------------------------------------------------------------------
class NoController(BaseController):
    """Never applies cooling – demonstrates overheating without control."""

    def select_action(self, state: np.ndarray) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float32)

    @property
    def name(self) -> str:
        return "No Control"


# ---------------------------------------------------------------------------
# Random controller
# ---------------------------------------------------------------------------
class RandomController(BaseController):
    """
    Applies smooth random cooling actions.
    Uses EMA smoothing to avoid abrupt changes.
    """

    def __init__(self, action_dim: int = NUM_ZONES, seed: Optional[int] = None):
        super().__init__(action_dim)
        self.rng    = np.random.default_rng(seed)
        self._action = self.rng.uniform(0, 1, action_dim)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        raw = self.rng.uniform(0, 1, self.action_dim)
        self._action = (1 - ACTION_SMOOTH) * self._action + ACTION_SMOOTH * raw
        return np.clip(self._action, 0.0, 1.0).astype(np.float32)

    def reset(self):
        self._action = self.rng.uniform(0, 1, self.action_dim)

    @property
    def name(self) -> str:
        return "Random Control"


# ---------------------------------------------------------------------------
# MPC Controller  (random shooting)
# ---------------------------------------------------------------------------
class MPCController(BaseController):
    """
    Random-shooting MPC controller backed by a learned dynamics model.

    Parameters
    ----------
    dynamics_model : BaseDynamicsModel
        Trained model: predict(states, actions) → next_states
    horizon        : int   – number of steps to look ahead
    num_candidates : int   – number of candidate action sequences
    target_temp    : float – desired grid mean temperature
    temp_weight    : float – penalty weight on temperature deviation
    energy_weight  : float – penalty weight on energy usage
    seed           : int   – for reproducible action sampling
    """

    def __init__(
        self,
        dynamics_model: BaseDynamicsModel,
        horizon:        int   = HORIZON,
        num_candidates: int   = NUM_CANDIDATES,
        target_temp:    float = TARGET_TEMP,
        overheat_weight: float = OVERHEAT_WEIGHT,
        overcool_weight: float = OVERCOOL_WEIGHT,
        energy_weight:  float = ENERGY_WEIGHT,
        seed:           Optional[int] = None,
    ):
        super().__init__()
        self.model         = dynamics_model
        self.horizon       = horizon
        self.num_candidates = num_candidates
        self.target_temp   = target_temp
        self.overheat_weight = overheat_weight
        self.overcool_weight = overcool_weight
        self.energy_weight = energy_weight
        self.rng           = np.random.default_rng(seed)

        # Previous action for action-smoothing
        self._prev_action = np.full(self.action_dim, 0.5, dtype=np.float32)

    # ------------------------------------------------------------------
    # Main action selection
    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Run random-shooting MPC and return the best first action.

        Parameters
        ----------
        state : np.ndarray  (state_dim,)  – current flat temperature grid

        Returns
        -------
        action : np.ndarray  (action_dim,)  in [0, 1]
        """
        state = np.atleast_1d(state).astype(np.float32)

        # Sample candidate action sequences
        # Shape: (num_candidates, horizon, action_dim)
        candidate_seqs = self.rng.uniform(
            0, 1,
            (self.num_candidates, self.horizon, self.action_dim)
        ).astype(np.float32)

        # Smooth candidates around previous action
        candidate_seqs[:, 0, :] = (
            (1 - ACTION_SMOOTH) * self._prev_action
            + ACTION_SMOOTH * candidate_seqs[:, 0, :]
        )
        candidate_seqs = np.clip(candidate_seqs, 0.0, 1.0)

        # Evaluate each candidate sequence
        total_costs = self._evaluate_sequences(state, candidate_seqs)

        # Pick best sequence
        best_idx    = int(np.argmin(total_costs))
        best_action = candidate_seqs[best_idx, 0, :]   # first action only

        # Smooth with previous action
        best_action = (
            (1 - ACTION_SMOOTH) * self._prev_action
            + ACTION_SMOOTH * best_action
        )
        best_action = np.clip(best_action, 0.0, 1.0).astype(np.float32)

        self._prev_action = best_action.copy()
        return best_action

    # ------------------------------------------------------------------
    # Batch rollout evaluation  (vectorised for speed)
    # ------------------------------------------------------------------
    def _evaluate_sequences(
        self,
        init_state:     np.ndarray,
        candidate_seqs: np.ndarray,
    ) -> np.ndarray:
        """
        Roll out all candidate sequences in parallel and return cost per sequence.

        Parameters
        ----------
        init_state     : (state_dim,)
        candidate_seqs : (num_candidates, horizon, action_dim)

        Returns
        -------
        costs : (num_candidates,)
        """
        N = self.num_candidates
        # Tile state to match batch size
        states = np.tile(init_state, (N, 1))   # (N, state_dim)
        total_costs = np.zeros(N, dtype=np.float32)

        for h in range(self.horizon):
            actions = candidate_seqs[:, h, :]   # (N, action_dim)

            # Predict next states  (N, state_dim)
            next_states = self.model.predict(states, actions)

            # -- Temperature penalty: mean squared deviation from target
            mean_temps = np.mean(next_states, axis=1)
            temp_cost = np.zeros(N, dtype=np.float32)

            in_band = (
                (mean_temps >= COMFORT_BAND_LOW)
                & (mean_temps <= COMFORT_BAND_HIGH)
            )
            overheating = mean_temps > self.target_temp
            overcooling = mean_temps < COMFORT_BAND_LOW

            temp_cost[overheating] = (
                (mean_temps[overheating] - self.target_temp)
                * self.overheat_weight
            )
            temp_cost[overcooling] = (
                (self.target_temp - mean_temps[overcooling])
                * self.overcool_weight
            )
            temp_cost[in_band] = 0.0

            # -- Energy penalty: sum of squared cooling power × zone size
            # Approximately: 0.5 * sum_z(power_z^2) * cells_per_zone
            cells_per_zone = next_states.shape[1] // self.action_dim
            energy_cost = self.energy_weight * (
                COOLING_ENERGY_FACTOR * cells_per_zone
                * np.sum(actions ** 2, axis=1)
            )   # (N,)

            total_costs += temp_cost + energy_cost
            states = next_states

        return total_costs

    def reset(self):
        self._prev_action = np.full(self.action_dim, 0.5, dtype=np.float32)

    @property
    def name(self) -> str:
        return "MPC Control"


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------
def get_controller(
    kind:           str,
    dynamics_model: Optional[BaseDynamicsModel] = None,
    seed:           Optional[int] = None,
) -> BaseController:
    """
    Factory function.

    kind: 'none' | 'random' | 'mpc'
    """
    if kind == 'none':
        return NoController()
    elif kind == 'random':
        return RandomController(seed=seed)
    elif kind == 'mpc':
        if dynamics_model is None:
            raise ValueError("MPCController requires a trained dynamics_model.")
        return MPCController(dynamics_model=dynamics_model, seed=seed)
    else:
        raise ValueError(f"Unknown controller kind: {kind}")


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from env.cooling_env import DataCenterEnv
    from models.dynamics_model import LinearDynamicsModel, LINEAR_PATH

    print("=" * 55)
    print("  Data Center RL — Controller Smoke Test")
    print("=" * 55)

    env   = DataCenterEnv(seed=0)
    model = LinearDynamicsModel.load(LINEAR_PATH)

    for ctrl_name in ['none', 'random', 'mpc']:
        ctrl  = get_controller(ctrl_name, dynamics_model=model, seed=42)
        state = env.reset()
        ctrl.reset()
        temps = []
        for _ in range(20):
            action           = ctrl.select_action(state)
            state, _, _, info = env.step(action)
            temps.append(info['mean_temp'])

        print(f"\n  [{ctrl.name:14s}]  "
              f"avg_temp={np.mean(temps):.2f}°C  "
              f"max_temp={max(temps):.2f}°C  "
              f"final_temp={temps[-1]:.2f}°C")
