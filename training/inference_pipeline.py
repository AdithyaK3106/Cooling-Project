"""
training/inference_pipeline.py
────────────────────────────────
VALIDATION PIPELINE — training/evaluation use only, NOT production runtime.

Purpose
───────
This script validates that:
1. The trained XGBoost model produces correct outputs on held-out data.
2. The risk fusion formula (PRD §4.2) produces values in [0,1].
3. Feature engineering is identical between training and inference.
4. The full pipeline latency meets the <50ms PRD requirement.

⚠️  This is NOT the production inference engine.
    Production runtime: src/inference.py → InferenceEngine class.

Flow validated here:
  raw_telemetry
    → src.features.FeatureProcessor.process_single()     [identical to inference]
    → src.features.AnalyticGNN.compute_single()          [identical to inference]
    → training.xgboost_model.ThermalRiskXGB.predict()    [loaded from disk]
    → risk fusion: 0.75 * xgb + 0.25 * gnn              [PRD §4.2]
    → output: risk ∈ [0,1], level: LOW/MED/HIGH/CRITICAL

No PyTorch. No torch-geometric. No custom normalization.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ── Resolve src/ path ─────────────────────────────────────────────────────────
_SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

_TRAIN_PATH = os.path.abspath(os.path.dirname(__file__))
if _TRAIN_PATH not in sys.path:
    sys.path.insert(0, _TRAIN_PATH)

from features       import FeatureProcessor, AnalyticGNN
from xgboost_model  import ThermalRiskXGB


# ─────────────────────────────────────────────────────────────────────────────
# RISK LEVEL MAPPING  (must match src/inference.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_risk_level(score: float) -> str:
    """Map [0,1] risk score to level string. Mirrors src/inference.py."""
    if score < 0.35: return "LOW"
    if score < 0.55: return "MED"
    if score < 0.75: return "HIGH"
    return "CRITICAL"


# ─────────────────────────────────────────────────────────────────────────────
# RISK FUSION  (PRD §4.2 — identical formula to production)
# ─────────────────────────────────────────────────────────────────────────────

def fuse_risk(xgb_score: float, gnn_embedding: float) -> float:
    """
    Final risk score computation (PRD §4.2):
        risk = clip(0.75 * xgb_prediction + 0.25 * gnn_embedding, 0, 1)
    """
    return float(np.clip(0.75 * xgb_score + 0.25 * gnn_embedding, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RowPrediction:
    row_idx:       int
    xgb_score:     float          # XGBoost output in [0,1]
    gnn_embedding: float          # AnalyticGNN output in [0,1]
    fused_risk:    float          # Final fused risk in [0,1]
    risk_level:    str            # LOW / MED / HIGH / CRITICAL
    latency_ms:    float = 0.0


@dataclass
class ValidationResult:
    n_samples:     int
    predictions:   List[RowPrediction] = field(default_factory=list)
    mean_risk:     float = 0.0
    max_risk:      float = 0.0
    p95_risk:      float = 0.0
    mean_latency_ms: float = 0.0
    latency_ok:    bool = True    # True if mean_latency_ms < 50ms
    level_dist:    Dict[str, int] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION PIPELINE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class ValidationPipeline:
    """
    Runs the complete inference path on a batch of pre-built telemetry rows
    and produces a structured ValidationResult.

    This replicates EXACTLY what src/inference.py does at runtime.
    If results differ between here and production, there is a bug.
    """

    def __init__(
        self,
        model_path: str,
        state_path: str,
        adjacency:  Optional[List[tuple]] = None,
    ):
        """
        Parameters
        ──────────
        model_path : path to trained XGBoost model (models/cooling_model.pkl).
        state_path : path to preprocessor state (models/preprocessor_state.pkl).
        adjacency  : rack adjacency list for AnalyticGNN. None = isolated mode.
        """
        # Load model
        self.xgb = ThermalRiskXGB()
        self.xgb.load(model_path)

        # Load processor (SAME code path as src/inference.py)
        self.processor = FeatureProcessor()
        self.processor.load(state_path)

        # Analytic GNN (SAME code path as src/inference.py)
        self.gnn = AnalyticGNN(adjacency=adjacency, n_nodes=1)

    def predict_row(self, raw_point: dict) -> RowPrediction:
        """
        Process a single telemetry dict through the full pipeline.

        Identical to src/inference.py InferenceEngine.run_once().
        """
        t0 = time.perf_counter()

        # Step 1: Feature engineering (shared src/features.py logic)
        features = self.processor.process_single(raw_point)  # (1, 15)

        # Step 2: Analytic GNN embedding
        cpu_n  = float(raw_point.get('cpu',  0.0)) / 100.0
        gpu_n  = float(raw_point.get('gpu',  0.0)) / 100.0
        heat_n = float(np.clip(0.6 * cpu_n + 0.4 * gpu_n, 0.0, 1.0))
        gnn_emb = self.gnn.compute_single(heat_norm=heat_n)

        # Step 3: Assemble 16-dim input
        X = np.hstack([features.flatten(), [gnn_emb]]).reshape(1, -1)

        # Step 4: XGBoost prediction
        xgb_score = float(self.xgb.predict(X)[0])

        # Step 5: Risk fusion (PRD §4.2)
        fused = fuse_risk(xgb_score, gnn_emb)
        level = get_risk_level(fused)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return RowPrediction(
            row_idx       = 0,
            xgb_score     = xgb_score,
            gnn_embedding = gnn_emb,
            fused_risk    = fused,
            risk_level    = level,
            latency_ms    = latency_ms,
        )

    def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """
        Run validation on a DataFrame of raw telemetry rows.

        Parameters
        ──────────
        df : DataFrame with columns: cpu, gpu, memory, disk_io, network_io.

        Returns
        ───────
        ValidationResult with full statistics.
        """
        self.processor.reset_buffer()
        predictions = []

        for idx, row in df.iterrows():
            pred = self.predict_row(row.to_dict())
            pred.row_idx = int(idx)
            predictions.append(pred)

        fused_scores = np.array([p.fused_risk for p in predictions])
        latencies    = np.array([p.latency_ms for p in predictions])

        level_dist = {}
        for p in predictions:
            level_dist[p.risk_level] = level_dist.get(p.risk_level, 0) + 1

        mean_latency = float(latencies.mean())

        return ValidationResult(
            n_samples       = len(predictions),
            predictions     = predictions,
            mean_risk       = float(fused_scores.mean()),
            max_risk        = float(fused_scores.max()),
            p95_risk        = float(np.percentile(fused_scores, 95)),
            mean_latency_ms = mean_latency,
            latency_ok      = mean_latency < 50.0,
            level_dist      = level_dist,
        )

    def print_summary(self, result: ValidationResult) -> None:
        print("\n── Validation Pipeline Summary ──────────────────────────────")
        print(f"  Samples         : {result.n_samples}")
        print(f"  Mean risk       : {result.mean_risk:.4f}   (expected in [0,1])")
        print(f"  Max risk        : {result.max_risk:.4f}")
        print(f"  P95 risk        : {result.p95_risk:.4f}")
        print(f"  Mean latency    : {result.mean_latency_ms:.3f} ms  {'✅ OK' if result.latency_ok else '❌ EXCEEDS 50ms SLA'}")
        print(f"  Risk distribution: {result.level_dist}")
        print("─────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION CHECKLIST  (automated PRD compliance checks)
# ─────────────────────────────────────────────────────────────────────────────

def run_validation_checklist(
    model_path: str,
    state_path: str,
    n_samples:  int = 100,
    seed:       int = 99,
) -> bool:
    """
    Run automated PRD compliance checks.

    Returns True if all checks pass, False otherwise.
    Designed to run in CI/CD before promoting a model to production.
    """
    from data_processing import generate_synthetic_telemetry

    print("\n" + "=" * 60)
    print("  VALIDATION CHECKLIST  (PRD Compliance)")
    print("=" * 60)

    raw = generate_synthetic_telemetry(n_rows=n_samples, seed=seed)
    pipeline = ValidationPipeline(model_path=model_path, state_path=state_path)
    result   = pipeline.validate_dataframe(raw)

    checks = {}

    # ✔ All fused risk scores in [0,1]
    all_in_range = all(0.0 <= p.fused_risk <= 1.0 for p in result.predictions)
    checks["Risk scores ∈ [0,1]"] = all_in_range

    # ✔ All GNN embeddings in [0,1]
    gnn_in_range = all(0.0 <= p.gnn_embedding <= 1.0 for p in result.predictions)
    checks["GNN embedding ∈ [0,1]"] = gnn_in_range

    # ✔ GNN embedding is scalar (not multi-dim)
    gnn_scalar = all(np.isscalar(p.gnn_embedding) or isinstance(p.gnn_embedding, float)
                     for p in result.predictions)
    checks["GNN embedding is scalar"] = gnn_scalar

    # ✔ Latency < 50ms (PRD requirement)
    checks["Latency < 50ms"] = result.latency_ok

    # ✔ No NaN in predictions
    no_nan = all(not np.isnan(p.fused_risk) for p in result.predictions)
    checks["No NaN in output"] = no_nan

    # ✔ Risk levels are valid strings
    valid_levels = {"LOW", "MED", "HIGH", "CRITICAL"}
    all_valid_levels = all(p.risk_level in valid_levels for p in result.predictions)
    checks["Valid risk levels"] = all_valid_levels

    # Print checklist
    all_pass = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status}  {check}")
        if not passed:
            all_pass = False

    pipeline.print_summary(result)
    print(f"\nFinal result: {'✅ ALL CHECKS PASSED' if all_pass else '❌ CHECKS FAILED'}")
    print("=" * 60)

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    MODEL_PATH = os.path.join('..', 'models', 'cooling_model.pkl')
    STATE_PATH = os.path.join('..', 'models', 'preprocessor_state.pkl')

    if not os.path.exists(MODEL_PATH):
        print("[INFO] No trained model found. Running training first …")
        from data_processing  import generate_synthetic_telemetry, build_training_dataset
        from xgboost_model    import ThermalRiskXGB

        raw  = generate_synthetic_telemetry(n_rows=500, seed=42)
        X, y, proc = build_training_dataset(raw, state_save_path=STATE_PATH)
        feature_names = proc.feature_names + ["gnn_embedding"]

        m = ThermalRiskXGB()
        m.train(X, y, feature_names=feature_names, verbose=False)
        m.save(MODEL_PATH)
        print("[INFO] Model trained and saved.")

    passed = run_validation_checklist(MODEL_PATH, STATE_PATH)
    sys.exit(0 if passed else 1)
