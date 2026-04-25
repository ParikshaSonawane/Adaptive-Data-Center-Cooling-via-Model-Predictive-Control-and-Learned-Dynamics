"""
dynamics_model.py
=================
Learns the mapping  (state, action) → next_state  from collected data.

Two backends are provided:
  LinearDynamicsModel  – Ridge regression; fast, interpretable, good baseline
  NeuralDynamicsModel  – Small MLP via scikit-learn; better for non-linearities

Both share the same interface so the MPC controller is backend-agnostic:
  model.fit(states, actions, next_states)
  model.predict(states, actions) → next_states
  model.save(path) / model.load(path)
"""

import os
import pickle
import numpy as np
from typing import Optional, Tuple, Dict

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class BaseDynamicsModel:
    """Common interface for all dynamics models."""

    def fit(
        self,
        states:      np.ndarray,
        actions:     np.ndarray,
        next_states: np.ndarray,
        val_split:   float = 0.15,
        verbose:     bool  = True,
    ) -> Dict[str, float]:
        raise NotImplementedError

    def predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> 'BaseDynamicsModel':
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"📂 Model loaded from {path}")
        return model

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _build_input(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Concatenate state and action vectors into a single input matrix."""
        return np.concatenate([states, actions], axis=1).astype(np.float32)

    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        split:  str = "test"
    ) -> Dict[str, float]:
        mse  = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        r2   = float(r2_score(y_true, y_pred))
        # Per-dimension RMSE averaged
        per_dim_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0)).mean())
        print(f"   [{split}]  RMSE={rmse:.5f}  per-dim RMSE={per_dim_rmse:.5f}  R²={r2:.4f}")
        return {"rmse": rmse, "per_dim_rmse": per_dim_rmse, "r2": r2}


# ---------------------------------------------------------------------------
# 1. Linear Dynamics Model  (Ridge Regression)
# ---------------------------------------------------------------------------
class LinearDynamicsModel(BaseDynamicsModel):
    """
    Predicts  Δstate = next_state - state  using Ridge regression,
    then returns  next_state = state + Δstate.

    Predicting the *delta* rather than the absolute next state makes
    learning easier because the residual has much lower variance.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha    = alpha
        self.model    = Ridge(alpha=alpha, fit_intercept=True)
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.is_trained = False

    def fit(
        self,
        states:      np.ndarray,
        actions:     np.ndarray,
        next_states: np.ndarray,
        val_split:   float = 0.15,
        verbose:     bool  = True,
    ) -> Dict[str, float]:
        X  = self._build_input(states, actions)
        dY = (next_states - states).astype(np.float32)   # predict delta

        X_tr, X_val, dY_tr, dY_val, s_tr, s_val = train_test_split(
            X, dY, states, test_size=val_split, random_state=42
        )

        # Scale features
        X_tr_s  = self.x_scaler.fit_transform(X_tr)
        X_val_s = self.x_scaler.transform(X_val)
        dY_tr_s = self.y_scaler.fit_transform(dY_tr)

        if verbose:
            print(f"\n🔧 Training LinearDynamicsModel (Ridge α={self.alpha})")
            print(f"   Train: {len(X_tr)} | Val: {len(X_val)}")

        self.model.fit(X_tr_s, dY_tr_s)
        self.is_trained = True

        # Evaluate
        metrics = {}
        if verbose:
            pred_tr  = self.y_scaler.inverse_transform(
                self.model.predict(X_tr_s)) + s_tr
            pred_val = self.y_scaler.inverse_transform(
                self.model.predict(X_val_s)) + s_val
            metrics['train'] = self._compute_metrics(s_tr + dY_tr, pred_tr, "train")
            metrics['val']   = self._compute_metrics(s_val + dY_val, pred_val, "val")

        return metrics

    def predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        assert self.is_trained, "Model not trained yet."
        states  = np.atleast_2d(states).astype(np.float32)
        actions = np.atleast_2d(actions).astype(np.float32)
        X  = self._build_input(states, actions)
        Xs = self.x_scaler.transform(X)
        dY_s = self.model.predict(Xs)
        dY   = self.y_scaler.inverse_transform(dY_s)
        return (states + dY).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Neural Network Dynamics Model  (MLP via sklearn)
# ---------------------------------------------------------------------------
class NeuralDynamicsModel(BaseDynamicsModel):
    """
    Small MLP regressor predicting Δstate.
    Faster to train than PyTorch for moderate dataset sizes while
    still capturing nonlinear dynamics.

    Architecture: [input] → 256 → 256 → 128 → [output]
    """

    def __init__(
        self,
        hidden_layers: Tuple[int, ...] = (256, 256, 128),
        max_iter:      int   = 500,
        learning_rate: float = 1e-3,
    ):
        self.hidden_layers  = hidden_layers
        self.max_iter       = max_iter
        self.learning_rate  = learning_rate

        self.model    = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=False,
        )
        self.x_scaler   = StandardScaler()
        self.y_scaler   = StandardScaler()
        self.is_trained = False

    def fit(
        self,
        states:      np.ndarray,
        actions:     np.ndarray,
        next_states: np.ndarray,
        val_split:   float = 0.15,
        verbose:     bool  = True,
    ) -> Dict[str, float]:
        X  = self._build_input(states, actions)
        dY = (next_states - states).astype(np.float32)

        X_tr, X_val, dY_tr, dY_val, s_tr, s_val = train_test_split(
            X, dY, states, test_size=val_split, random_state=42
        )

        X_tr_s  = self.x_scaler.fit_transform(X_tr)
        X_val_s = self.x_scaler.transform(X_val)
        dY_tr_s = self.y_scaler.fit_transform(dY_tr)

        if verbose:
            print(f"\n🔧 Training NeuralDynamicsModel  layers={self.hidden_layers}")
            print(f"   Train: {len(X_tr)} | Val: {len(X_val)}")

        self.model.fit(X_tr_s, dY_tr_s)
        self.is_trained = True

        if verbose:
            print(f"   Converged in {self.model.n_iter_} iterations  "
                  f"best_val_loss={self.model.best_validation_score_:.5f}")
            pred_val = self.y_scaler.inverse_transform(
                self.model.predict(X_val_s)) + s_val
            metrics = self._compute_metrics(s_val + dY_val, pred_val, "val")
            return {'val': metrics}
        return {}

    def predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        assert self.is_trained, "Model not trained yet."
        states  = np.atleast_2d(states).astype(np.float32)
        actions = np.atleast_2d(actions).astype(np.float32)
        X  = self._build_input(states, actions)
        Xs = self.x_scaler.transform(X)
        dY_s = self.model.predict(Xs)
        dY   = self.y_scaler.inverse_transform(dY_s)
        return (states + dY).astype(np.float32)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------
MODELS_DIR  = os.path.join(os.path.dirname(__file__), 'saved')
LINEAR_PATH = os.path.join(MODELS_DIR, 'linear_dynamics.pkl')
NEURAL_PATH = os.path.join(MODELS_DIR, 'neural_dynamics.pkl')


def get_model(kind: str = 'linear') -> BaseDynamicsModel:
    """Factory: 'linear' or 'neural'."""
    if kind == 'linear':
        return LinearDynamicsModel(alpha=0.5)
    elif kind == 'neural':
        return NeuralDynamicsModel(hidden_layers=(256, 256, 128), max_iter=300)
    else:
        raise ValueError(f"Unknown model kind: {kind}")


def default_model_path(kind: str = 'linear') -> str:
    return LINEAR_PATH if kind == 'linear' else NEURAL_PATH


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data.data_collector import DataCollector, DATASET_FILE

    print("=" * 55)
    print("  Data Center RL — Dynamics Model Training")
    print("=" * 55)

    # Load dataset
    states, actions, next_states = DataCollector.load(DATASET_FILE)

    # --- Linear ---
    lin = get_model('linear')
    lin.fit(states, actions, next_states, verbose=True)
    lin.save(LINEAR_PATH)

    # Quick prediction test
    pred = lin.predict(states[:5], actions[:5])
    err  = np.abs(pred - next_states[:5]).mean()
    print(f"\n   Linear avg abs error (5 samples): {err:.4f} °C")

    # --- Neural ---
    nn = get_model('neural')
    nn.fit(states, actions, next_states, verbose=True)
    nn.save(NEURAL_PATH)

    pred_nn = nn.predict(states[:5], actions[:5])
    err_nn  = np.abs(pred_nn - next_states[:5]).mean()
    print(f"\n   Neural avg abs error (5 samples): {err_nn:.4f} °C")
