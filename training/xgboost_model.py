"""
training/xgboost_model.py
─────────────────────────
XGBoost thermal risk regressor — training logic only.

Responsibility
──────────────
Train, evaluate, save, and load the XGBoost model.
This file does NOT perform feature engineering or normalization;
it consumes pre-built (X, y) from data_processing.build_training_dataset().

Input contract
──────────────
  X : np.ndarray (n_samples, 16)
      15 telemetry features (from FeatureProcessor) + 1 gnn_embedding column.
      All values in [0, 1].

  y : np.ndarray (n_samples,)
      Risk score in [0, 1].

Output contract
───────────────
  Saved model: models/cooling_model.pkl  (joblib format)
  predict() returns float in [0, 1].

Risk fusion (inference time, per PRD §4.2):
  risk = 0.75 * xgb_prediction + 0.25 * gnn_embedding
  (Fusion is performed in src/inference.py, NOT here.)

Why XGBoost?
  • Best accuracy/latency trade-off for tabular sensor data at this scale.
  • Handles non-linear feature interactions natively.
  • Feature importance is interpretable — critical for patent documentation.
  • Inference < 1ms per row on CPU.
  • Degrades gracefully if GNN embedding is noisy during cold-start.

Dependencies: xgboost, scikit-learn, numpy, pandas, joblib
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_XGB_PARAMS: Dict = {
    # Tree structure
    "n_estimators":      300,
    "max_depth":         6,       # 4–8 for tabular sensor data; deeper → overfit
    "min_child_weight":  3,       # prevents splits on tiny subsets (noisy telemetry)

    # Learning & regularisation
    "learning_rate":     0.05,    # low LR + more trees = more robust
    "subsample":         0.8,     # stochastic boosting (row sampling per tree)
    "colsample_bytree":  0.8,     # feature sampling per tree
    "reg_alpha":         0.1,     # L1 — drives some feature weights to zero
    "reg_lambda":        1.0,     # L2 — smooths overall weights

    # Objective: regress risk score in [0,1]
    "objective":         "reg:squarederror",
    "eval_metric":       "rmse",

    # Reproducibility & performance
    "random_state":      42,
    "n_jobs":            -1,
    "tree_method":       "hist",  # fast histogram-based splits
}

# Feature name for the GNN column (must match data_processing.py)
GNN_FEATURE_NAME = "gnn_embedding"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CLASS
# ─────────────────────────────────────────────────────────────────────────────

class ThermalRiskXGB:
    """
    XGBoost regressor for thermal risk prediction.

    Input:  16-dim feature vector (15 telemetry + 1 gnn_embedding), all in [0,1].
    Output: risk score in [0,1].
    """

    def __init__(self, params: Optional[Dict] = None):
        self.params        = params or DEFAULT_XGB_PARAMS
        self.model         = xgb.XGBRegressor(**self.params)
        self.feature_names: List[str] = []
        self._is_trained   = False

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        X:              np.ndarray,
        y:              np.ndarray,
        feature_names:  Optional[List[str]] = None,
        val_fraction:   float = 0.15,
        early_stopping: int   = 30,
        verbose:        bool  = True,
    ) -> Dict:
        """
        Train with automatic train/val split and early stopping.

        Parameters
        ──────────
        X              : (n_samples, 16), all values in [0,1].
        y              : (n_samples,), risk scores in [0,1].
        feature_names  : column names for SHAP / feature importance output.
        val_fraction   : fraction held out for early-stopping monitor.
        early_stopping : stop if no improvement for this many rounds.
        verbose        : print training log.

        Returns
        ───────
        dict with val RMSE, MAE, R².
        """
        if feature_names:
            self.feature_names = feature_names

        # Validate input ranges
        if X.max() > 1.0 + 1e-6 or X.min() < -1e-6:
            raise ValueError(
                f"[ThermalRiskXGB] Feature matrix X contains values outside [0,1]. "
                f"Range: [{X.min():.4f}, {X.max():.4f}]. "
                f"Run data_processing.build_training_dataset() first."
            )

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=val_fraction, random_state=42
        )

        self.model.set_params(
            early_stopping_rounds=early_stopping,
            verbosity=1 if verbose else 0,
        )

        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=verbose,
        )

        self._is_trained = True
        metrics = self.evaluate(X_val, y_val, prefix="val")

        if verbose:
            self._print_metrics(metrics)

        return metrics

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict risk scores. Output is clipped to [0,1].

        Parameters
        ──────────
        X : np.ndarray (n_samples, 16) or (1, 16).

        Returns
        ───────
        scores : np.ndarray (n_samples,), in [0,1].
        """
        self._check_trained()
        raw = self.model.predict(X)
        return np.clip(raw, 0.0, 1.0)

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        X:      np.ndarray,
        y:      np.ndarray,
        prefix: str = "test",
    ) -> Dict:
        """Return RMSE, MAE, R² metrics for X against ground truth y."""
        self._check_trained()
        y_pred = self.predict(X)
        rmse   = float(np.sqrt(mean_squared_error(y, y_pred)))
        mae    = float(mean_absolute_error(y, y_pred))
        r2     = float(r2_score(y, y_pred))

        return {
            f"{prefix}_rmse": rmse,
            f"{prefix}_mae":  mae,
            f"{prefix}_r2":   r2,
        }

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importance_df(self, top_n: int = 20) -> pd.DataFrame:
        """
        Return sorted DataFrame of XGBoost feature importances.
        The gnn_embedding column appearing in the top positions validates
        that the analytic GNN adds predictive value.
        """
        self._check_trained()
        importances = self.model.feature_importances_
        names = self.feature_names if self.feature_names else [
            f"f{i}" for i in range(len(importances))
        ]
        return (
            pd.DataFrame({"feature": names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model + feature names to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump((self.model, self.feature_names), path)
        print(f"[ThermalRiskXGB] Model saved → {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ThermalRiskXGB] Model not found at '{path}'.")
        self.model, self.feature_names = joblib.load(path)
        self._is_trained = True
        print(f"[ThermalRiskXGB] Model loaded ← {path}")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _check_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError("[ThermalRiskXGB] Model not trained. Call .train() first.")

    @staticmethod
    def _print_metrics(metrics: Dict) -> None:
        print("\n── XGBoost Evaluation ──────────────────────")
        for k, v in metrics.items():
            print(f"  {k:<22s}: {v:.6f}")
        print("────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def train_from_csv(
    csv_path:       str,
    model_save_path: str = os.path.join('..', 'models', 'cooling_model.pkl'),
    state_save_path: str = os.path.join('..', 'models', 'preprocessor_state.pkl'),
    verbose:         bool = True,
) -> ThermalRiskXGB:
    """
    Full training pipeline from raw CSV to saved model.

    Parameters
    ──────────
    csv_path        : path to raw telemetry CSV (must have network_io column).
    model_save_path : where to save the trained XGBoost model.
    state_save_path : where to save the preprocessor state.

    Returns
    ───────
    Trained ThermalRiskXGB instance.
    """
    import sys
    _SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
    if _SRC_PATH not in sys.path:
        sys.path.insert(0, _SRC_PATH)

    from data_processing import build_training_dataset

    raw_df = pd.read_csv(csv_path)
    X, y, processor = build_training_dataset(raw_df, state_save_path=state_save_path)

    feature_names = processor.feature_names + [GNN_FEATURE_NAME]

    model = ThermalRiskXGB()
    model.train(X, y, feature_names=feature_names, verbose=verbose)
    model.save(model_save_path)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    _SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
    if _SRC_PATH not in sys.path:
        sys.path.insert(0, _SRC_PATH)

    from data_processing import generate_synthetic_telemetry, build_training_dataset

    print("\n[SMOKE TEST] xgboost_model.py")
    raw = generate_synthetic_telemetry(n_rows=300, seed=1)
    X, y, proc = build_training_dataset(raw, state_save_path="/tmp/preprocessor_state_xgb.pkl")

    feature_names = proc.feature_names + [GNN_FEATURE_NAME]
    model = ThermalRiskXGB()
    metrics = model.train(X, y, feature_names=feature_names, verbose=False)

    preds = model.predict(X[:4])
    print(f"\n  Metrics       : {metrics}")
    print(f"  Sample preds  : {preds.round(4)}   (expected: in [0,1])")
    assert all(0.0 <= p <= 1.0 for p in preds), "Predictions outside [0,1]!"

    fi = model.feature_importance_df(top_n=5)
    print(f"\n  Top-5 features:\n{fi.to_string(index=False)}")

    print("\ntraining/xgboost_model.py  ✓")
