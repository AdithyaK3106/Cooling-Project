"""
src/inference.py
────────────────
PRODUCTION INFERENCE ENGINE — runtime entry point.

Flow
────
collect_telemetry()
  → FeatureProcessor.process_single()     [src/features.py — shared with training]
  → AnalyticGNN.compute_single()          [src/features.py — shared with training]
  → XGBoost.predict()                     [models/cooling_model.pkl]
  → risk fusion: 0.75*xgb + 0.25*gnn     [PRD §4.2]
  → log to data/inference_logs.csv

Schema (PRD v1.0):
  Raw telemetry: cpu, gpu, memory, disk_io, network_io

Risk fusion (PRD §4.2):
  risk = clip(0.75 * xgb_prediction + 0.25 * gnn_embedding, 0, 1)

Output:
  risk_score  : float in [0,1]
  risk_level  : LOW | MED | HIGH | CRITICAL

Dependency policy:
  Runtime imports: numpy, pandas, psutil, xgboost, joblib, pickle  ONLY.
  NO torch. NO torch-geometric. NO sklearn StandardScaler.

Configuration:
  MODEL_PATH         = ../models/cooling_model.pkl
  PREPROCESSOR_STATE = ../models/preprocessor_state.pkl
  OUTPUT_LOG         = ../data/inference_logs.csv
  INTERVAL           = 1.0  (seconds)
"""

import os
import sys
import time
import pickle
import subprocess
import numpy as np
import pandas as pd
import psutil
from datetime import datetime

# ── Resolve src/ for local imports ────────────────────────────────────────────
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from features import FeatureProcessor, AnalyticGNN


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH         = os.path.join(os.path.dirname(__file__), '..', 'models', 'cooling_model.pkl')
PREPROCESSOR_STATE = os.path.join(os.path.dirname(__file__), '..', 'models', 'preprocessor_state.pkl')
OUTPUT_LOG         = os.path.join(os.path.dirname(__file__), '..', 'data', 'inference_logs.csv')
INTERVAL           = 1.0  # seconds between inference cycles

# Risk fusion weights (PRD §4.2)
_XGB_WEIGHT = 0.75
_GNN_WEIGHT = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# RISK SCORING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fuse_risk(xgb_score: float, gnn_embedding: float) -> float:
    """
    PRD §4.2 fusion formula.
    risk = clip(0.75 * xgb_prediction + 0.25 * gnn_embedding, 0, 1)
    """
    return float(np.clip(_XGB_WEIGHT * xgb_score + _GNN_WEIGHT * gnn_embedding, 0.0, 1.0))


def get_risk_level(score: float) -> str:
    """Map [0,1] risk score to human-readable level."""
    if score < 0.35: return "LOW"
    if score < 0.55: return "MED"
    if score < 0.75: return "HIGH"
    return "CRITICAL"


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Production inference engine for AI-Driven Predictive Cooling.

    Initialization
    ──────────────
    Loads the trained XGBoost model and the fitted FeatureProcessor state
    from disk. Fails loudly if either artifact is missing.

    Inference loop
    ──────────────
    Each cycle:
      1. Collect raw telemetry (cpu, gpu, memory, disk_io, network_io).
      2. FeatureProcessor.process_single() → 15-dim normalized vector.
      3. AnalyticGNN.compute_single() → scalar gnn_embedding ∈ [0,1].
      4. Concatenate → 16-dim X.
      5. XGBoost.predict(X) → xgb_score ∈ [0,1].
      6. fuse_risk(xgb_score, gnn_embedding) → risk_score ∈ [0,1].
      7. Log to CSV.
    """

    def __init__(
        self,
        model_path:  str = MODEL_PATH,
        state_path:  str = PREPROCESSOR_STATE,
        adjacency    = None,   # Optional rack adjacency for multi-rack deployments
    ):
        # 1. Load XGBoost model
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[InferenceEngine] Model not found at '{model_path}'. "
                f"Run the training pipeline first."
            )
        with open(model_path, 'rb') as f:
            loaded = pickle.load(f)

        # Handle both (model, feature_names) tuple and raw model
        if isinstance(loaded, tuple):
            self._xgb_model, self._feature_names = loaded
        else:
            self._xgb_model   = loaded
            self._feature_names = []

        # 2. Setup FeatureProcessor (shared with training pipeline)
        self.processor = FeatureProcessor()
        self.processor.load(state_path)

        # 3. Setup AnalyticGNN (no PyTorch — deterministic analytic formula)
        self.gnn = AnalyticGNN(adjacency=adjacency, n_nodes=1)

        print("[InferenceEngine] Ready.")
        print(f"  Model         : {os.path.basename(model_path)}")
        print(f"  Preprocessor  : {os.path.basename(state_path)}")
        print(f"  GNN mode      : {'multi-rack' if adjacency else 'isolated (single-rack)'}")

    # ── Telemetry Collection ─────────────────────────────────────────────────

    def collect_telemetry(
        self,
        disk_prev: dict,
        net_prev:  dict,
    ) -> tuple:
        """
        Collect one snapshot of system telemetry.

        Returns
        ───────
        (raw_data, new_disk_prev, new_net_prev)

        raw_data keys: timestamp, cpu, gpu, memory, disk_io, network_io
        """
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent

        # GPU utilization via nvidia-smi (graceful fallback to 0.0)
        gpu = 0.0
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1,
            )
            if res.returncode == 0:
                gpu = float(res.stdout.strip().split("\n")[0])
        except Exception:
            pass

        # I/O rates (bytes/sec delta from previous snapshot)
        now      = time.monotonic()
        dk       = psutil.disk_io_counters()
        disk_raw = (dk.read_bytes + dk.write_bytes) if dk else 0
        disk_rate = (
            (disk_raw - disk_prev['val']) / max(now - disk_prev['time'], 1e-6)
            if disk_prev else 0.0
        )

        nk       = psutil.net_io_counters()
        net_raw  = (nk.bytes_sent + nk.bytes_recv) if nk else 0
        net_rate = (
            (net_raw - net_prev['val']) / max(now - net_prev['time'], 1e-6)
            if net_prev else 0.0
        )

        raw_data = {
            "timestamp":  ts,
            "cpu":        cpu,
            "gpu":        gpu,
            "memory":     mem,
            "disk_io":    max(0.0, disk_rate),
            "network_io": max(0.0, net_rate),
        }

        return (
            raw_data,
            {"val": disk_raw, "time": now},
            {"val": net_raw,  "time": now},
        )

    # ── Single Inference Step ────────────────────────────────────────────────

    def predict(self, raw_data: dict) -> tuple:
        """
        Run one inference step on a single raw telemetry dict.

        Parameters
        ──────────
        raw_data : dict with keys cpu, gpu, memory, disk_io, network_io.

        Returns
        ───────
        (risk_score: float ∈ [0,1], risk_level: str, gnn_embedding: float)
        """
        # Step 1: Feature engineering (identical to training pipeline)
        features = self.processor.process_single(raw_data)  # (1, 15)

        # Step 2: Analytic GNN embedding (scalar)
        cpu_n  = float(raw_data.get('cpu', 0.0)) / 100.0
        gpu_n  = float(raw_data.get('gpu', 0.0)) / 100.0
        heat_n = float(np.clip(0.6 * cpu_n + 0.4 * gpu_n, 0.0, 1.0))
        gnn_emb = self.gnn.compute_single(heat_norm=heat_n)

        # Step 3: Assemble 16-dim input
        X = np.hstack([features.flatten(), [gnn_emb]]).reshape(1, -1)

        # Step 4: XGBoost prediction
        xgb_score = float(np.clip(self._xgb_model.predict(X)[0], 0.0, 1.0))

        # Step 5: Fuse (PRD §4.2)
        risk_score = fuse_risk(xgb_score, gnn_emb)
        risk_level = get_risk_level(risk_score)

        return risk_score, risk_level, gnn_emb

    # ── CSV Logging ──────────────────────────────────────────────────────────

    def log_result(
        self,
        raw_data:   dict,
        risk_score: float,
        risk_level: str,
        gnn_emb:    float,
    ) -> None:
        """Append one inference result row to the output CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_LOG)), exist_ok=True)
        file_exists = os.path.isfile(OUTPUT_LOG)

        log_row = {
            **raw_data,
            "gnn_embedding": round(float(gnn_emb),    4),
            "risk_score":    round(float(risk_score),  4),
            "risk_level":    risk_level,
        }
        pd.DataFrame([log_row]).to_csv(
            OUTPUT_LOG, mode='a', index=False, header=not file_exists
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("─" * 60)
    print("  AI-Driven Predictive Cooling — Inference Engine")
    print("─" * 60)

    try:
        engine = InferenceEngine()
    except Exception as exc:
        print(f"[FATAL] Initialization failed: {exc}")
        sys.exit(1)

    # Prime I/O baseline counters
    dk_init = psutil.disk_io_counters()
    nk_init = psutil.net_io_counters()
    dk_prev = {
        "val":  (dk_init.read_bytes + dk_init.write_bytes) if dk_init else 0,
        "time": time.monotonic(),
    }
    nk_prev = {
        "val":  (nk_init.bytes_sent + nk_init.bytes_recv) if nk_init else 0,
        "time": time.monotonic(),
    }

    print(f"\nRunning at {INTERVAL}s intervals. Press Ctrl-C to stop.\n")

    try:
        while True:
            tick_start = time.monotonic()

            # 1. Collect raw telemetry
            raw_data, dk_prev, nk_prev = engine.collect_telemetry(dk_prev, nk_prev)

            # 2. Predict
            risk_score, risk_level, gnn_emb = engine.predict(raw_data)

            # 3. Log
            engine.log_result(raw_data, risk_score, risk_level, gnn_emb)

            # 4. Console output
            print(
                f"[{raw_data['timestamp']}] "
                f"CPU={raw_data['cpu']:5.1f}% "
                f"GPU={raw_data['gpu']:5.1f}% "
                f"GNN={gnn_emb:.3f} "
                f"XGB→Fused={risk_score:.3f} "
                f"Level={risk_level:<8s}"
            )

            # Sleep remainder of interval
            elapsed = time.monotonic() - tick_start
            time.sleep(max(0.01, INTERVAL - elapsed))

    except KeyboardInterrupt:
        print("\n[InferenceEngine] Stopped by user.")


if __name__ == "__main__":
    main()
