"""
training/data_processing.py
────────────────────────────
DATASET BUILDER — training pipeline only.

Responsibility
──────────────
Convert raw telemetry CSV into (X, y) matrices ready for XGBoost training.

This file:
  ✅ Delegates ALL feature engineering to src.features.FeatureProcessor
  ✅ Uses src.features.AnalyticGNN for the GNN embedding column
  ✅ Generates risk labels using the PRD formula (0.6*cpu + 0.4*gpu → [0,1])
  ✅ Persists preprocessor state to models/preprocessor_state.pkl
  ✅ Produces X (15 features + gnn_embedding = 16 cols) and y (risk score)

  ❌ Does NOT define any normalization logic
  ❌ Does NOT use StandardScaler
  ❌ Does NOT duplicate feature engineering from src/features.py
  ❌ Does NOT use PyTorch/torch-geometric

Schema compliance (PRD v1.0):
  Raw CSV columns: timestamp, cpu, gpu, memory, disk_io, network_io
  (Any additional columns like gpu_temp, cpu_temp are passed through for labels only)
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Tuple

# ── Import from src (single source of truth) ──────────────────────────────────
# Resolve src/ relative to this file's location
_SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

from features import FeatureProcessor, AnalyticGNN


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (must match PRD schema)
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLS = ['cpu', 'gpu', 'memory', 'disk_io', 'network_io']

# Risk label formula (PRD §4.2): identical to inference-time risk derivation
_RISK_LABEL_CPU_W = 0.6
_RISK_LABEL_GPU_W = 0.4


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame) -> None:
    """
    Raise a ValueError if the DataFrame is missing required columns.
    Catches the most common mistake: 'network' instead of 'network_io'.
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        hint = ""
        if 'network_io' in missing and 'network' in df.columns:
            hint = " Hint: rename column 'network' → 'network_io' (PRD §2.1)."
        raise ValueError(
            f"[data_processing] Schema violation — missing columns: {missing}.{hint}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# LABEL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_risk_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Compute continuous risk label in [0, 1] from raw telemetry.

    Formula (PRD §4.2, identical to inference fusion target):
        risk = clip(0.6 * (cpu/100) + 0.4 * (gpu/100), 0, 1)

    This is the ground-truth proxy when no physical temperature sensors exist.
    In a calibrated deployment, replace with actual sensor readings.
    """
    cpu_n = df['cpu'].clip(0, 100) / 100.0
    gpu_n = df['gpu'].clip(0, 100) / 100.0
    y     = (_RISK_LABEL_CPU_W * cpu_n + _RISK_LABEL_GPU_W * gpu_n).clip(0, 1)
    return y.values.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# GNN EMBEDDING COLUMN (analytic, no PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def compute_gnn_embeddings(df: pd.DataFrame, processor: FeatureProcessor) -> np.ndarray:
    """
    Compute the analytic GNN embedding column for each training row.

    Uses AnalyticGNN from src/features.py with isolated-node mode
    (single-machine training data has no rack topology).

    Returns
    ───────
    gnn_col : np.ndarray of shape (n_rows,), values in [0, 1].
    """
    gnn = AnalyticGNN(adjacency=None, n_nodes=1)
    embeddings = []

    for _, row in df.iterrows():
        cpu_n  = float(row.get('cpu',  0.0)) / 100.0
        gpu_n  = float(row.get('gpu',  0.0)) / 100.0
        heat_n = float(np.clip(0.6 * cpu_n + 0.4 * gpu_n, 0.0, 1.0))
        emb    = gnn.compute_single(heat_norm=heat_n, neighbor_heats=None)
        embeddings.append(emb)

    return np.array(embeddings, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR  (testing / demo)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_telemetry(
    n_rows: int = 500,
    seed:   int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic telemetry compliant with PRD schema.

    Columns: timestamp, cpu, gpu, memory, disk_io, network_io
    I/O values are in bytes/sec (realistic range for a workstation).
    """
    rng = np.random.default_rng(seed)

    # Simulate bursts: 10% of rows have high CPU/GPU
    n = n_rows
    burst = rng.random(n) < 0.10

    cpu    = np.where(burst, rng.uniform(75, 98, n), rng.uniform(10, 65, n))
    gpu    = np.where(burst, rng.uniform(70, 95, n), rng.uniform( 5, 55, n))
    memory = rng.uniform(40, 85, n)

    # I/O in bytes/sec: baseline ~10 MB/s, spikes up to ~200 MB/s
    disk_io    = rng.exponential(scale=5_000_000, size=n).clip(0, 200_000_000)
    network_io = rng.exponential(scale=1_000_000, size=n).clip(0,  50_000_000)

    timestamps = pd.date_range("2024-01-01", periods=n, freq="1s")

    return pd.DataFrame({
        'timestamp':  timestamps,
        'cpu':        cpu.clip(0, 100).round(2),
        'gpu':        gpu.clip(0, 100).round(2),
        'memory':     memory.clip(0, 100).round(2),
        'disk_io':    disk_io.round(2),
        'network_io': network_io.round(2),
    })


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_training_dataset(
    raw_df:          pd.DataFrame,
    state_save_path: str = os.path.join('..', 'models', 'preprocessor_state.pkl'),
) -> Tuple[np.ndarray, np.ndarray, FeatureProcessor]:
    """
    End-to-end builder: raw telemetry DataFrame → (X, y, processor).

    Steps
    ─────
    1. Validate schema (crash-fast on wrong column names).
    2. Clean/coerce numeric types, forward-fill missing values.
    3. Fit FeatureProcessor on full dataset → save state.
    4. Process each row via processor.process_single() → 15 telemetry features.
    5. Compute analytic GNN embedding per row → 1 extra column.
    6. Concatenate → X of shape (n_rows, 16).
    7. Generate risk labels y in [0,1].

    Returns
    ───────
    X         : np.ndarray (n_rows, 16)  — 15 telemetry features + gnn_embedding
    y         : np.ndarray (n_rows,)     — risk score in [0,1]
    processor : fitted FeatureProcessor  — use to save/load state
    """
    print("─" * 60)
    print("[data_processing] Building training dataset …")

    # ── 1. Schema validation ──────────────────────────────────────────────────
    validate_schema(raw_df)

    # ── 2. Data cleaning ──────────────────────────────────────────────────────
    df = raw_df.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    for col in REQUIRED_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df[REQUIRED_COLS] = df[REQUIRED_COLS].ffill().fillna(df[REQUIRED_COLS].median())

    print(f"[data_processing] Rows after cleaning: {len(df)}")

    # ── 3. Fit FeatureProcessor ───────────────────────────────────────────────
    processor = FeatureProcessor()
    processor.fit(df)
    processor.save(state_save_path)

    # ── 4. Feature engineering (row-by-row, simulating inference loop) ────────
    print("[data_processing] Engineering 15 telemetry features …")
    tel_features = processor.process_dataframe(df)  # shape: (n_rows, 15)

    # ── 5. Analytic GNN embedding ─────────────────────────────────────────────
    print("[data_processing] Computing analytic GNN embeddings …")
    gnn_col = compute_gnn_embeddings(df, processor)  # shape: (n_rows,)

    # ── 6. Assemble X ─────────────────────────────────────────────────────────
    X = np.hstack([tel_features, gnn_col.reshape(-1, 1)])  # (n_rows, 16)

    # ── 7. Labels ─────────────────────────────────────────────────────────────
    y = generate_risk_labels(df)  # (n_rows,), in [0,1]

    print(f"[data_processing] X shape: {X.shape} | y range: [{y.min():.4f}, {y.max():.4f}]")
    print("[data_processing] Dataset build complete.")
    print("─" * 60)

    return X, y, processor


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[SMOKE TEST] data_processing.py")
    raw = generate_synthetic_telemetry(n_rows=200, seed=0)
    X, y, proc = build_training_dataset(raw, state_save_path="/tmp/preprocessor_state_test.pkl")
    print(f"\n  X shape        : {X.shape}   (expected: (200, 16))")
    print(f"  y range        : [{y.min():.4f}, {y.max():.4f}]   (expected: subset of [0,1])")
    print(f"  Feature names  : {proc.feature_names} + ['gnn_embedding']")
    assert X.shape == (200, 16), "X shape mismatch!"
    assert 0.0 <= y.min() and y.max() <= 1.0, "y out of [0,1]!"
    assert not np.any(np.isnan(X)), "NaN in X!"
    print("\ntraining/data_processing.py  ✓")
