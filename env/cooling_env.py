"""
cooling_env.py
==============
2D grid-based data center simulation environment.

Physics modelled:
  - Server heat generation (randomized load per rack)
  - Spatial heat diffusion between neighboring cells
  - Thermal inertia (sluggish temperature response)
  - Gaussian noise / disturbances
  - Zone-wise cooling actuation
"""

import numpy as np
from typing import Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID_ROWS = 10
GRID_COLS = 10
NUM_CELLS = GRID_ROWS * GRID_COLS

TARGET_TEMP = 25.0
AMBIENT_TEMP = 20.0
MAX_TEMP = 80.0
INIT_TEMP_MIN = 22.0
INIT_TEMP_MAX = 28.0
COMFORT_BAND_LOW = 24.0
COMFORT_BAND_HIGH = 26.0

# Heat generation parameters
BASE_HEAT_RATE = 0.8
LOAD_NOISE_STD = 0.15

# Thermal dynamics
DIFFUSION_COEFF = 0.08
INERTIA_COEFF = 0.85
NOISE_STD = 0.05

# Cooling parameters
NUM_ZONES = 4
MAX_COOLING_RATE = 3.0
COOLING_ENERGY_FACTOR = 0.5


class DataCenterEnv:
    """
    Simulates a 10x10 data center grid.

    Action space  : numpy array of shape (NUM_ZONES,) in [0, 1]
                    representing fractional cooling power per zone.
    State space   : flattened temperature grid  (NUM_CELLS,)
    Reward        : negative cost = -(temp_penalty + energy_penalty)
    """

    def __init__(
        self,
        rows: int = GRID_ROWS,
        cols: int = GRID_COLS,
        num_zones: int = NUM_ZONES,
        target_temp: float = TARGET_TEMP,
        seed: Optional[int] = None,
    ):
        self.rows = rows
        self.cols = cols
        self.num_zones = num_zones
        self.target_temp = target_temp
        self.rng = np.random.default_rng(seed)

        self._build_zone_masks()
        self._build_diffusion_neighbours()

        self.grid: np.ndarray = np.zeros((rows, cols))
        self.step_count: int = 0
        self.history: Dict = {"temps": [], "actions": [], "rewards": [], "energy": []}

        self.reset()

    def _build_zone_masks(self):
        """Divide grid into num_zones rectangular zones."""
        self.zone_masks = []
        half_r = self.rows // 2
        half_c = self.cols // 2
        slices = [
            (slice(0, half_r), slice(0, half_c)),
            (slice(0, half_r), slice(half_c, self.cols)),
            (slice(half_r, self.rows), slice(0, half_c)),
            (slice(half_r, self.rows), slice(half_c, self.cols)),
        ]
        for s in slices[:self.num_zones]:
            mask = np.zeros((self.rows, self.cols), dtype=bool)
            mask[s] = True
            self.zone_masks.append(mask)

    def _build_diffusion_neighbours(self):
        """Pre-build list of (i, j, neighbour_list) for diffusion step."""
        self._neighbours = {}
        for i in range(self.rows):
            for j in range(self.cols):
                nbrs = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.rows and 0 <= nj < self.cols:
                        nbrs.append((ni, nj))
                self._neighbours[(i, j)] = nbrs

    def reset(self) -> np.ndarray:
        """Reset environment to a warm random initial state."""
        self.grid = self.rng.uniform(INIT_TEMP_MIN, INIT_TEMP_MAX, (self.rows, self.cols))
        self.step_count = 0
        self.history = {"temps": [], "actions": [], "rewards": [], "energy": []}
        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Advance simulation by one time step.

        Parameters
        ----------
        action : np.ndarray of shape (num_zones,), values in [0, 1]

        Returns
        -------
        next_state : np.ndarray  (flattened temperature grid)
        reward     : float
        done       : bool
        info       : dict
        """
        action = np.clip(action, 0.0, 1.0)

        heat = self._generate_heat()
        diffused = self._diffuse(self.grid)
        new_grid = INERTIA_COEFF * self.grid + (1 - INERTIA_COEFF) * diffused + heat

        cooling_applied, energy = self._apply_cooling(new_grid, action)
        new_grid -= cooling_applied
        new_grid += self.rng.normal(0, NOISE_STD, new_grid.shape)
        new_grid = np.clip(new_grid, AMBIENT_TEMP, MAX_TEMP)

        self.grid = new_grid
        self.step_count += 1

        reward = self._compute_reward(energy)
        done = bool(np.mean(self.grid) > MAX_TEMP * 0.9)

        info = {
            "mean_temp": float(np.mean(self.grid)),
            "max_temp": float(np.max(self.grid)),
            "min_temp": float(np.min(self.grid)),
            "energy": float(energy),
            "step": self.step_count,
        }

        self.history["temps"].append(info["mean_temp"])
        self.history["actions"].append(action.copy())
        self.history["rewards"].append(reward)
        self.history["energy"].append(energy)

        return self._get_state(), reward, done, info

    def _generate_heat(self) -> np.ndarray:
        """Generate per-cell heat based on a noisy oscillating workload."""
        t = self.step_count
        base_load = 0.6 + 0.3 * np.sin(2 * np.pi * t / 50)
        load = base_load + self.rng.normal(0, LOAD_NOISE_STD, (self.rows, self.cols))
        load = np.clip(load, 0.1, 1.0)
        return BASE_HEAT_RATE * load

    def _diffuse(self, grid: np.ndarray) -> np.ndarray:
        """Apply Laplacian-like heat diffusion over the 4-neighbour graph."""
        diffused = grid.copy()
        for i in range(self.rows):
            for j in range(self.cols):
                nbrs = self._neighbours[(i, j)]
                if nbrs:
                    nbr_mean = np.mean([grid[ni, nj] for ni, nj in nbrs])
                    diffused[i, j] += DIFFUSION_COEFF * (nbr_mean - grid[i, j])
        return diffused

    def _apply_cooling(self, grid: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply zone-wise cooling. Returns (cooling_grid, total_energy)."""
        cooling = np.zeros_like(grid)
        total_energy = 0.0
        for mask, power in zip(self.zone_masks, action):
            drop = MAX_COOLING_RATE * power
            cooling[mask] += drop
            zone_cells = np.sum(mask)
            total_energy += COOLING_ENERGY_FACTOR * (power ** 2) * zone_cells
        return cooling, total_energy

    def _compute_reward(self, energy: float) -> float:
        """Reward = - (asymmetric temperature penalty + energy penalty)."""
        mean_temp = float(np.mean(self.grid))
        if COMFORT_BAND_LOW <= mean_temp <= COMFORT_BAND_HIGH:
            temp_penalty = 0.0
        elif mean_temp > self.target_temp:
            temp_penalty = (mean_temp - self.target_temp) * 5.0
        else:
            temp_penalty = (self.target_temp - mean_temp) * 1.0

        energy_penalty = 0.5 * energy
        return -(temp_penalty + energy_penalty)

    def _get_state(self) -> np.ndarray:
        """Return flattened temperature grid as state vector."""
        return self.grid.flatten().astype(np.float32)

    def get_grid(self) -> np.ndarray:
        """Return current 2D temperature grid."""
        return self.grid.copy()

    def get_state_dim(self) -> int:
        return NUM_CELLS

    def get_action_dim(self) -> int:
        return self.num_zones

    @property
    def observation_space_shape(self):
        return (NUM_CELLS,)

    @property
    def action_space_shape(self):
        return (self.num_zones,)


if __name__ == "__main__":
    env = DataCenterEnv(seed=42)
    print(f"State dim : {env.get_state_dim()}")
    print(f"Action dim: {env.get_action_dim()}")
    print(f"Init mean temp: {env.get_grid().mean():.2f} C")

    for _ in range(5):
        action = np.random.uniform(0, 1, env.get_action_dim())
        state, reward, done, info = env.step(action)
        print(
            f"  step={info['step']:3d}  mean={info['mean_temp']:.2f}  "
            f"max={info['max_temp']:.2f}  energy={info['energy']:.2f}  "
            f"reward={reward:.3f}"
        )
