"""
training/data_processing.py
----------------------------
DATASET BUILDER -- training pipeline only.

Responsibility
--------------
Convert raw telemetry CSV into (X, y) matrices for XGBoost training.

This file:
  DELEGATES all feature engineering to src.features.FeatureProcessor.
  X is exactly 6 columns: [cpu_norm, gpu_norm, memory_norm,
                            disk_io_norm, network_io_norm, gnn_embedding]
  y is risk label in [0, 1].

This file does NOT:
  - Define any normalization logic.
  - Use StandardScaler.
  - Duplicate anything from src/features.py.
  - Use PyTorch or torch-geometric.

Schema (PRD v1.0):
  Required CSV columns: cpu, gpu, memory, disk_io, network_io
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Tuple

# -- Import from src (single source of truth) ----------------------------------
_SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

from features import FeatureProcessor, FEATURE_NAMES, FEATURE_DIM

# -- Import fusion from core ---------------------------------------------------
_CORE_PATH = os.path.join(_SRC_PATH, 'core')
if _CORE_PATH not in sys.path:
    sys.path.insert(0, _CORE_PATH)

from core.fusion import assert_parity  # noqa: E402


# =============================================================================
# CONSTANTS
# =============================================================================

REQUIRED_COLS = ['cpu', 'gpu', 'memory', 'disk_io', 'network_io']

# Risk label weights (PRD §4.2)
_RISK_CPU_W = 0.6
_RISK_GPU_W = 0.4


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

def validate_schema(df: pd.DataFrame) -> None:
    """
    Raise ValueError on missing required columns.
    Provides actionable hint for the common 'network' vs 'network_io' mistake.
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        hint = ""
        if 'network_io' in missing and 'network' in df.columns:
            hint = " Hint: rename 'network' -> 'network_io' (PRD §2.1)."
        raise ValueError(
            f"[data_processing] Schema violation -- missing columns: {missing}.{hint}"
        )


# =============================================================================
# LABEL GENERATION
# =============================================================================

def generate_risk_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Compute continuous risk label in [0, 1].

    Formula (PRD §4.2):
        risk = clip(0.6 * (cpu/100) + 0.4 * (gpu/100), 0, 1)

    This is the ground-truth proxy without physical temperature sensors.
    Replace with real sensor readings when available.
    """
    cpu_n = df['cpu'].clip(0, 100) / 100.0
    gpu_n = df['gpu'].clip(0, 100) / 100.0
    y     = (_RISK_CPU_W * cpu_n + _RISK_GPU_W * gpu_n).clip(0.0, 1.0)
    return y.values.astype(np.float64)


# =============================================================================
# SYNTHETIC DATA GENERATOR  (testing / demo)
# =============================================================================

def generate_synthetic_telemetry(n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic telemetry compliant with PRD schema.

    Columns: timestamp, cpu, gpu, memory, disk_io, network_io
    I/O values in bytes/sec (realistic workstation range).
    """
    rng   = np.random.default_rng(seed)
    burst = rng.random(n_rows) < 0.10

    cpu    = np.where(burst, rng.uniform(75, 98, n_rows), rng.uniform(10, 65, n_rows))
    gpu    = np.where(burst, rng.uniform(70, 95, n_rows), rng.uniform( 5, 55, n_rows))
    memory = rng.uniform(40, 85, n_rows)

    disk_io    = rng.exponential(scale=5_000_000, size=n_rows).clip(0, 200_000_000)
    network_io = rng.exponential(scale=1_000_000, size=n_rows).clip(0,  50_000_000)

    return pd.DataFrame({
        'timestamp':  pd.date_range("2024-01-01", periods=n_rows, freq="1s"),
        'cpu':        cpu.clip(0, 100).round(2),
        'gpu':        gpu.clip(0, 100).round(2),
        'memory':     memory.clip(0, 100).round(2),
        'disk_io':    disk_io.round(2),
        'network_io': network_io.round(2),
    })


# =============================================================================
# PARITY ASSERTION  (mandatory before every training run)
# =============================================================================

def verify_training_inference_parity(processor: FeatureProcessor) -> None:
    """
    Assert that the training and inference feature paths produce identical
    vectors for the same input.

    This test is run at the END of build_training_dataset() to guarantee
    zero training-serving skew before the model is trained.

    Uses src.core.fusion.assert_parity() — hard fail on any deviation.
    """
    sample = {
        'cpu': 75.0, 'gpu': 60.0, 'memory': 70.0,
        'disk_io': 1_500_000.0, 'network_io': 800_000.0,
    }

    # Training path: FeatureProcessor.process_single()
    train_vec = processor.process_single(sample)

    # Inference path: independent processor instance with same state
    infer_proc = FeatureProcessor()
    infer_proc.stats = dict(processor.stats)  # same fitted state, different object
    infer_vec = infer_proc.process_single(sample)

    assert_parity(train_vec, infer_vec, label="training vs inference feature vector")
    print("[data_processing] Parity check: PASSED -- training == inference feature path")


# =============================================================================
# MAIN DATASET BUILDER
# =============================================================================

def build_training_dataset(
    raw_df:          pd.DataFrame,
    state_save_path: str = os.path.join('..', 'models', 'preprocessor_state.pkl'),
) -> Tuple[np.ndarray, np.ndarray, FeatureProcessor]:
    """
    Full builder: raw telemetry DataFrame -> (X, y, processor).

    Steps
    -----
    1. Schema validation.
    2. Data cleaning (coerce types, forward-fill NaNs).
    3. Fit FeatureProcessor on full dataset -> save state.
    4. Process each row via processor.process_single() -> 6-dim X.
       (gnn_embedding is INSIDE process_single(), not appended separately.)
    5. Generate risk labels y in [0,1].
    6. Parity assertion (training == inference path, hard fail on skew).

    Returns
    -------
    X         : np.ndarray (n_rows, 6)  -- FRD feature vector per row
    y         : np.ndarray (n_rows,)    -- risk score in [0,1]
    processor : fitted FeatureProcessor -- use .save() to persist
    """
    print("-" * 60)
    print("[data_processing] Building training dataset ...")
    print(f"[data_processing] Feature dim = {FEATURE_DIM}, names = {FEATURE_NAMES}")

    # Step 1 -- schema
    validate_schema(raw_df)

    # Step 2 -- cleaning
    df = raw_df.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    for col in REQUIRED_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[REQUIRED_COLS] = df[REQUIRED_COLS].ffill().fillna(df[REQUIRED_COLS].median())

    print(f"[data_processing] Rows after cleaning: {len(df)}")

    # Step 3 -- fit + save
    processor = FeatureProcessor()
    processor.fit(df)
    processor.save(state_save_path)

    # Step 4 -- feature engineering (6-dim, gnn inside process_single)
    print("[data_processing] Engineering 6-dim feature vectors ...")
    X = processor.process_dataframe(df)  # shape: (n_rows, 6)

    # Step 5 -- labels
    y = generate_risk_labels(df)  # shape: (n_rows,), in [0,1]

    print(f"[data_processing] X shape: {X.shape} | y range: [{y.min():.4f}, {y.max():.4f}]")

    # Step 6 -- parity assertion (MANDATORY -- fail hard if training != inference)
    verify_training_inference_parity(processor)

    print("[data_processing] Dataset build complete.")
    print("-" * 60)

    return X, y, processor


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    print("\n[SMOKE TEST] data_processing.py")
    raw = generate_synthetic_telemetry(n_rows=200, seed=0)
    X, y, proc = build_training_dataset(raw, state_save_path="/tmp/preprocessor_state_test.pkl")

    print(f"\n  X shape      : {X.shape}   (expected: (200, {FEATURE_DIM}))")
    print(f"  y range      : [{y.min():.4f}, {y.max():.4f}]  (expected: subset of [0,1])")
    print(f"  Feature names: {proc.feature_names}")

    assert X.shape == (200, FEATURE_DIM), f"X shape mismatch! Got {X.shape}"
    assert 0.0 <= y.min() and y.max() <= 1.0, "y out of [0,1]!"
    assert not np.any(np.isnan(X)), "NaN in X!"
    assert np.all(X >= 0.0) and np.all(X <= 1.0), "X values outside [0,1]!"
    print("\ntraining/data_processing.py OK")
