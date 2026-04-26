"""
data_processing.py
------------------
Feature engineering, normalization, and label generation
for the sensor-free predictive cooling system.

Input schema: timestamp, rack_id, cpu, gpu, memory, disk_io, network
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List
import joblib
import os


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

TELEMETRY_COLS = ["cpu", "gpu", "memory", "disk_io", "network"]

# Overheating proxy thresholds (used when no real temp sensor exists)
THERMAL_THRESHOLDS = {
    "cpu":     85.0,   # % utilisation → maps to high thermal output
    "gpu":     90.0,
    "memory":  90.0,
    "disk_io": 80.0,
    "network": 75.0,
}

# Weights for composite thermal risk label (domain-derived)
THERMAL_WEIGHTS = {
    "cpu":     0.35,
    "gpu":     0.30,
    "memory":  0.15,
    "disk_io": 0.10,
    "network": 0.10,
}

ROLLING_WINDOWS = [3, 5, 10]   # time-steps (e.g. 3 s, 5 s, 10 s)


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given raw telemetry, produce an enriched feature matrix per rack.

    Features added
    ──────────────
    • Rolling mean / std   → captures sustained load vs. spikes
    • Rate-of-change (diff)→ captures acceleration of thermal build-up
    • Composite load score → single signal combining all resources
    • Cross-feature interaction: cpu×gpu (high both = thermal danger)
    """
    df = df.copy().sort_values(["rack_id", "timestamp"])

    for col in TELEMETRY_COLS:
        grp = df.groupby("rack_id")[col]

        # Rolling statistics (per rack)
        for w in ROLLING_WINDOWS:
            df[f"{col}_roll_mean_{w}"] = grp.transform(
                lambda s, w=w: s.rolling(w, min_periods=1).mean()
            )
            df[f"{col}_roll_std_{w}"] = grp.transform(
                lambda s, w=w: s.rolling(w, min_periods=1).std().fillna(0)
            )

        # Rate of change (first-order finite difference per rack)
        df[f"{col}_delta"] = grp.transform(lambda s: s.diff().fillna(0))

    # Composite thermal load score  (0–100 scale)
    df["composite_load"] = sum(
        THERMAL_WEIGHTS[c] * df[c] for c in TELEMETRY_COLS
    )

    # Cross-feature interaction: CPU × GPU (both hot = hotspot risk ↑)
    df["cpu_gpu_interaction"] = (df["cpu"] / 100.0) * (df["gpu"] / 100.0) * 100.0

    # Peak-pressure flag: any single metric above its threshold
    df["any_threshold_breach"] = (
        (df["cpu"]     >= THERMAL_THRESHOLDS["cpu"])     |
        (df["gpu"]     >= THERMAL_THRESHOLDS["gpu"])     |
        (df["memory"]  >= THERMAL_THRESHOLDS["memory"])  |
        (df["disk_io"] >= THERMAL_THRESHOLDS["disk_io"]) |
        (df["network"] >= THERMAL_THRESHOLDS["network"])
    ).astype(int)

    return df


# ─────────────────────────────────────────────
# LABEL GENERATION (THERMAL RISK PROXY)
# ─────────────────────────────────────────────

def generate_thermal_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a synthetic thermal risk score (0–100) from workload telemetry.

    Formula
    ───────
    base_risk   = weighted sum of utilisation values
    breach_bonus= +15 per column that is above its threshold
    risk_score  = clip(base_risk + breach_bonus, 0, 100)

    This acts as the ground truth label when stress-test data is available.
    In production, labels can be refined with actual thermal measurements
    from a one-time calibration run.
    """
    df = df.copy()

    base_risk = sum(THERMAL_WEIGHTS[c] * df[c] for c in TELEMETRY_COLS)

    breach_bonus = sum(
        15.0 * (df[c] >= THERMAL_THRESHOLDS[c]).astype(float)
        for c in TELEMETRY_COLS
    )

    df["thermal_risk_score"] = np.clip(base_risk + breach_bonus, 0, 100)

    # Binary hotspot label for classification evaluation
    df["hotspot"] = (df["thermal_risk_score"] >= 70).astype(int)

    return df


# ─────────────────────────────────────────────
# NORMALISATION
# ─────────────────────────────────────────────

class TelemetryScaler:
    """
    Wraps sklearn StandardScaler with save/load helpers.
    Fit only on training data; transform train + inference data.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._feature_cols: List[str] = []

    def fit_transform(self, df: pd.DataFrame, exclude_cols: List[str]) -> pd.DataFrame:
        self._feature_cols = [c for c in df.columns if c not in exclude_cols]
        df = df.copy()
        df[self._feature_cols] = self.scaler.fit_transform(df[self._feature_cols])
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self._feature_cols] = self.scaler.transform(df[self._feature_cols])
        return df

    def save(self, path: str):
        joblib.dump((self.scaler, self._feature_cols), path)

    def load(self, path: str):
        self.scaler, self._feature_cols = joblib.load(path)


# ─────────────────────────────────────────────
# TRAINING DATASET BUILDER
# ─────────────────────────────────────────────

def build_training_dataset(
    raw_df: pd.DataFrame,
    scaler_save_path: str = "scaler.pkl",
) -> Tuple[pd.DataFrame, pd.DataFrame, TelemetryScaler]:
    """
    End-to-end: raw telemetry → normalised feature matrix + labels.

    Returns
    ───────
    X_train : feature matrix (all engineered features, normalised)
    y_train : thermal_risk_score column
    scaler  : fitted TelemetryScaler for inference-time use
    """
    df = engineer_features(raw_df)
    df = generate_thermal_risk_label(df)

    EXCLUDE = ["timestamp", "rack_id", "thermal_risk_score", "hotspot"]

    scaler = TelemetryScaler()
    df_scaled = scaler.fit_transform(df, exclude_cols=EXCLUDE)
    scaler.save(scaler_save_path)

    feature_cols = [c for c in df_scaled.columns if c not in EXCLUDE]
    X = df_scaled[feature_cols]
    y = df_scaled["thermal_risk_score"]   # label kept in original scale

    # Re-attach unscaled label
    y = df["thermal_risk_score"]

    return X, y, scaler


# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR  (for testing)
# ─────────────────────────────────────────────

def generate_synthetic_telemetry(
    n_racks: int = 4,
    n_timesteps: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates realistic-ish telemetry for a small rack cluster.
    Useful for unit tests and demonstration without real hardware.
    """
    rng = np.random.default_rng(seed)
    records = []

    for rack_id in range(n_racks):
        # Each rack has a random baseline load profile
        base_cpu = rng.uniform(20, 60)
        base_gpu = rng.uniform(10, 50)

        for t in range(n_timesteps):
            # Simulate occasional stress spikes
            spike = 1.0 + (0.4 if rng.random() < 0.08 else 0.0)
            records.append({
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=t),
                "rack_id":   rack_id,
                "cpu":       float(np.clip(base_cpu * spike + rng.normal(0, 5), 0, 100)),
                "gpu":       float(np.clip(base_gpu * spike + rng.normal(0, 5), 0, 100)),
                "memory":    float(np.clip(rng.uniform(40, 85) + rng.normal(0, 3), 0, 100)),
                "disk_io":   float(np.clip(rng.uniform(10, 70) + rng.normal(0, 4), 0, 100)),
                "network":   float(np.clip(rng.uniform(5,  60) + rng.normal(0, 4), 0, 100)),
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# QUICK SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    raw = generate_synthetic_telemetry(n_racks=4, n_timesteps=50)
    X, y, scaler = build_training_dataset(raw, scaler_save_path="/tmp/scaler.pkl")
    print(f"Feature matrix : {X.shape}")
    print(f"Label range    : {y.min():.1f} – {y.max():.1f}")
    print(f"Feature columns:\n  {list(X.columns[:8])} …")
    print("data_processing.py  ✓")
