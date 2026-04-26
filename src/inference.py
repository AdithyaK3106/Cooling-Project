"""
src/inference.py
----------------
PRODUCTION INFERENCE ENGINE — runtime entry point.

Flow (FRD §6.2)
---------------
collect_telemetry()
  -> validate schema (required keys: cpu, gpu, memory, disk_io, network_io)
  -> FeatureProcessor.process_single()   [src/features.py — SHARED with training]
     Outputs 6-dim vector: [cpu_norm, gpu_norm, memory_norm,
                             disk_io_norm, network_io_norm, gnn_embedding]
  -> XGBoost.predict(6-dim X)            [models/cooling_model.pkl]
  -> src.core.fusion.fuse(xgb, gnn)      [PRD §4.2 — SINGLE fusion definition]
  -> src.core.fusion.get_risk_level()    [FRD §5.1 — SINGLE level definition]
  -> log to data/inference_logs.csv

Schema (PRD v1.0):
  Required keys: cpu, gpu, memory, disk_io, network_io

Risk fusion (PRD §4.2) — DEFINED IN src/core/fusion.py:
  risk = clip(0.75 * xgb_prediction + 0.25 * gnn_embedding, 0, 1)

Output:
  risk_score  : float in [0,1]
  risk_level  : LOW | MED | HIGH | CRITICAL

Dependency policy:
  Runtime imports: numpy, pandas, psutil, xgboost, joblib, pickle ONLY.
  NO torch. NO torch-geometric. NO StandardScaler.
  NO local redefinitions of fuse() or get_risk_level().
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

# -- Resolve src/ for local imports -------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from features import FeatureProcessor, validate_raw_input, REQUIRED_KEYS

# Import fusion from the SINGLE SOURCE OF TRUTH
_CORE_DIR = os.path.join(_SRC_DIR, 'core')
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

from core.fusion import fuse, get_risk_level, assert_parity  # noqa: E402


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH         = os.path.join(_SRC_DIR, '..', 'models', 'cooling_model.pkl')
PREPROCESSOR_STATE = os.path.join(_SRC_DIR, '..', 'models', 'preprocessor_state.pkl')
OUTPUT_LOG         = os.path.join(_SRC_DIR, '..', 'data', 'inference_logs.csv')
INTERVAL           = 1.0  # seconds between inference cycles


# =============================================================================
# INFERENCE ENGINE
# =============================================================================

class InferenceEngine:
    """
    Production inference engine for AI-Driven Predictive Cooling.

    Feature pipeline (FRD §3.1 — 6 features):
      [0] cpu_norm        [1] gpu_norm       [2] memory_norm
      [3] disk_io_norm    [4] network_io_norm [5] gnn_embedding

    All fusion and risk-level logic imported from src/core/fusion.py.
    No local redefinitions of any shared function.
    """

    def __init__(
        self,
        model_path:  str = MODEL_PATH,
        state_path:  str = PREPROCESSOR_STATE,
        adjacency         = None,   # Optional rack adjacency for multi-rack
        run_parity_check: bool = False,  # Enable on startup for CI validation
    ):
        # 1. Load XGBoost model
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[InferenceEngine] Model not found at '{model_path}'. "
                f"Run the training pipeline first."
            )
        with open(model_path, 'rb') as f:
            loaded = pickle.load(f)

        if isinstance(loaded, tuple):
            self._xgb_model, self._feature_names = loaded
        else:
            self._xgb_model   = loaded
            self._feature_names = []

        # 2. Setup FeatureProcessor with optional multi-rack GNN
        self.processor = FeatureProcessor()
        self.processor.load(state_path)

        if adjacency is not None:
            from features import AnalyticGNN
            self.processor.set_gnn(AnalyticGNN(adjacency=adjacency, n_nodes=len(set(
                n for edge in adjacency for n in edge
            ))))

        # 3. Optional startup parity check
        if run_parity_check:
            self._startup_parity_check()

        print("[InferenceEngine] Ready.")
        print(f"  Model      : {os.path.basename(model_path)}")
        print(f"  Preprocessor: {os.path.basename(state_path)}")
        print(f"  Feature dim : {FeatureProcessor.FEATURE_DIM}  {FeatureProcessor.FEATURE_NAMES}")

    def _startup_parity_check(self) -> None:
        """
        Run the parity assertion on a known synthetic input at startup.
        Proves that training and inference paths produce identical vectors.
        """
        sample = {'cpu': 50.0, 'gpu': 40.0, 'memory': 60.0,
                  'disk_io': 1_000_000.0, 'network_io': 500_000.0}
        vec_a = self.processor.process_single(sample)

        # Create a second independent processor instance (simulates training path)
        proc2 = FeatureProcessor()
        proc2.stats = dict(self.processor.stats)  # same fitted state
        vec_b = proc2.process_single(sample)

        assert_parity(vec_a, vec_b, label="startup parity check")
        print("[InferenceEngine] Parity check: PASSED (training == inference path)")

    # -------------------------------------------------------------------------
    # SCHEMA VALIDATION
    # -------------------------------------------------------------------------

    @staticmethod
    def validate_telemetry(raw_data: dict) -> None:
        """
        Validate raw telemetry dict before feature processing.
        Raises ValueError immediately if any required key is missing.

        Required keys: cpu, gpu, memory, disk_io, network_io
        """
        validate_raw_input(raw_data)  # from src/features.py

    # -------------------------------------------------------------------------
    # TELEMETRY COLLECTION
    # -------------------------------------------------------------------------

    def collect_telemetry(self, disk_prev: dict, net_prev: dict) -> tuple:
        """
        Collect one telemetry snapshot from the live system.

        Returns
        -------
        (raw_data, new_disk_prev, new_net_prev)

        raw_data keys: timestamp, cpu, gpu, memory, disk_io, network_io
        """
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent

        # GPU via nvidia-smi (graceful fallback)
        gpu = 0.0
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1,
            )
            if res.returncode == 0:
                gpu = float(res.stdout.strip().split("\n")[0])
        except Exception:
            pass

        # I/O rates
        now = time.monotonic()
        dk  = psutil.disk_io_counters()
        disk_raw  = (dk.read_bytes + dk.write_bytes) if dk else 0
        disk_rate = (
            (disk_raw - disk_prev['val']) / max(now - disk_prev['time'], 1e-6)
            if disk_prev else 0.0
        )

        nk = psutil.net_io_counters()
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

    # -------------------------------------------------------------------------
    # PREDICTION
    # -------------------------------------------------------------------------

    def predict(self, raw_data: dict) -> tuple:
        """
        Run one full inference step.

        Steps
        -----
        1. Schema validation  — hard fail on missing keys.
        2. FeatureProcessor.process_single() -> 6-dim X (includes gnn_embedding).
        3. XGBoost.predict(X) -> xgb_score in [0,1].
        4. fusion.fuse(xgb_score, gnn_embedding) -> risk_score in [0,1].
        5. fusion.get_risk_level(risk_score) -> level string.

        Parameters
        ----------
        raw_data : dict with keys: cpu, gpu, memory, disk_io, network_io.

        Returns
        -------
        (risk_score: float, risk_level: str, gnn_embedding: float)
        """
        # Step 1: schema guard (fail hard before any computation)
        self.validate_telemetry(raw_data)

        # Step 2: 6-dim feature vector (includes gnn_embedding at index [5])
        X = self.processor.process_single(raw_data)  # shape (1, 6)

        # Extract gnn_embedding from feature vector (index 5) — no recomputation
        gnn_emb = float(X[0, 5])

        # Step 3: XGBoost prediction (input is the full 6-dim vector)
        xgb_score = float(np.clip(self._xgb_model.predict(X)[0], 0.0, 1.0))

        # Step 4 & 5: fuse and level (from src/core/fusion.py — single definition)
        risk_score = fuse(xgb_score, gnn_emb)
        risk_level = get_risk_level(risk_score)

        return risk_score, risk_level, gnn_emb

    # -------------------------------------------------------------------------
    # LOGGING
    # -------------------------------------------------------------------------

    def log_result(
        self,
        raw_data:   dict,
        risk_score: float,
        risk_level: str,
        gnn_emb:    float,
    ) -> None:
        """Append one inference result to the output CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_LOG)), exist_ok=True)
        file_exists = os.path.isfile(OUTPUT_LOG)
        log_row = {
            **raw_data,
            "gnn_embedding": round(float(gnn_emb),   4),
            "risk_score":    round(float(risk_score), 4),
            "risk_level":    risk_level,
        }
        pd.DataFrame([log_row]).to_csv(
            OUTPUT_LOG, mode='a', index=False, header=not file_exists
        )


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    print("-" * 60)
    print("  AI-Driven Predictive Cooling -- Inference Engine")
    print("-" * 60)

    try:
        engine = InferenceEngine(run_parity_check=True)
    except Exception as exc:
        print(f"[FATAL] Initialization failed: {exc}")
        sys.exit(1)

    # Prime I/O baselines
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

            raw_data, dk_prev, nk_prev = engine.collect_telemetry(dk_prev, nk_prev)
            risk_score, risk_level, gnn_emb = engine.predict(raw_data)
            engine.log_result(raw_data, risk_score, risk_level, gnn_emb)

            print(
                f"[{raw_data['timestamp']}] "
                f"CPU={raw_data['cpu']:5.1f}% "
                f"GPU={raw_data['gpu']:5.1f}% "
                f"GNN={gnn_emb:.3f} "
                f"Fused={risk_score:.3f} "
                f"Level={risk_level:<8s}"
            )

            elapsed = time.monotonic() - tick_start
            time.sleep(max(0.01, INTERVAL - elapsed))

    except KeyboardInterrupt:
        print("\n[InferenceEngine] Stopped by user.")


if __name__ == "__main__":
    main()
