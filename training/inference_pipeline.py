"""
inference_pipeline.py
─────────────────────
End-to-end real-time inference pipeline for the sensor-free
predictive cooling system.

Flow
────
Raw telemetry → Feature engineering → GNN → XGBoost → Fusion → Decision

               ┌──────────────────────────────────────────────────┐
  Telemetry    │   Feature      ┌─────────┐  embedding            │
  per rack ───►│   Engineering ─► GNN     ├──────────────┐        │
               │                └─────────┘              ▼        │
               │                             ┌──────────────────┐ │
               │   Scaled tel. features ────►│   XGBoost        │ │
               │                             └──────┬───────────┘ │
               │                                    │ risk score  │
               │                         ┌──────────▼──────────┐  │
               │                         │  Weighted Fusion     │  │
               │                         │  (GNN + XGB)         │  │
               │                         └──────────┬───────────┘  │
               │                                    │              │
               │                         ┌──────────▼──────────┐  │
               │                         │  Cooling Decision    │  │
               │                         └─────────────────────┘  │
               └──────────────────────────────────────────────────┘

Model fusion formula
────────────────────
final_risk = α * xgb_risk  +  (1-α) * gnn_risk_from_embedding

where:
  α = 0.75  (XGBoost dominates — it sees both raw + embedding features)
  The 0.25 GNN contribution encodes cross-rack thermal propagation
  that XGBoost's tabular view alone cannot fully capture.

Cooling trigger logic
─────────────────────
  final_risk ≥ 90   → EMERGENCY  (immediate max cooling)
  final_risk ≥ 75   → HIGH       (step up cooling)
  final_risk ≥ 55   → ELEVATED   (pre-emptive cooling increase)
  final_risk <  55  → NORMAL

Dependencies
────────────
    pip install torch torch-geometric xgboost scikit-learn pandas numpy
"""

import time
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from data_processing  import engineer_features, TelemetryScaler, TELEMETRY_COLS
from gnn_model        import ThermalGNN, build_rack_graph, linear_rack_adjacency, get_thermal_embeddings
from xgboost_model    import ThermalRiskXGB, assemble_xgb_features, HOTSPOT_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# COOLING DECISION LEVELS
# ─────────────────────────────────────────────────────────────────────────────

COOLING_LEVELS = {
    "EMERGENCY": 90.0,
    "HIGH":      75.0,
    "ELEVATED":  55.0,
    "NORMAL":     0.0,
}


def risk_to_cooling_action(risk_score: float) -> str:
    if risk_score >= COOLING_LEVELS["EMERGENCY"]:
        return "EMERGENCY"
    elif risk_score >= COOLING_LEVELS["HIGH"]:
        return "HIGH"
    elif risk_score >= COOLING_LEVELS["ELEVATED"]:
        return "ELEVATED"
    return "NORMAL"


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RackPrediction:
    rack_id:          int
    final_risk_score: float        # 0–100  (fused)
    xgb_risk:         float        # raw XGBoost output
    gnn_risk:         float        # risk derived from GNN embedding magnitude
    hotspot_prob:     float        # 0–1
    cooling_action:   str          # NORMAL / ELEVATED / HIGH / EMERGENCY
    latency_ms:       float = 0.0


@dataclass
class BatchResult:
    timestamp:    str
    predictions:  List[RackPrediction] = field(default_factory=list)
    hotspot_racks: List[int]           = field(default_factory=list)
    max_risk:     float                = 0.0
    global_action: str                 = "NORMAL"


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class CoolingInferencePipeline:
    """
    Loads trained GNN + XGBoost models and runs real-time inference
    on incoming telemetry batches.

    Parameters
    ──────────
    gnn_model      : trained ThermalGNN
    xgb_model      : trained ThermalRiskXGB
    scaler         : fitted TelemetryScaler (from training)
    adjacency      : rack adjacency list  (matches training topology)
    gnn_alpha      : weight for XGBoost in fusion (default 0.75)
    embed_dim      : must match ThermalGNN.embed_dim
    hotspot_thresh : risk score above which a rack is flagged as hotspot
    """

    def __init__(
        self,
        gnn_model:       ThermalGNN,
        xgb_model:       ThermalRiskXGB,
        scaler:          TelemetryScaler,
        adjacency:       List[Tuple[int, int]],
        gnn_alpha:       float = 0.75,
        embed_dim:       int   = 32,
        hotspot_thresh:  float = HOTSPOT_THRESHOLD,
    ):
        self.gnn          = gnn_model
        self.xgb          = xgb_model
        self.scaler       = scaler
        self.adjacency    = adjacency
        self.alpha        = gnn_alpha       # XGBoost weight in fusion
        self.embed_dim    = embed_dim
        self.thresh       = hotspot_thresh

    # ── Core inference ────────────────────────────────────────────────────────

    def predict_batch(self, raw_df: pd.DataFrame) -> BatchResult:
        """
        Process one telemetry snapshot (one row per rack) through the full
        GNN → XGBoost → fusion pipeline.

        Parameters
        ──────────
        raw_df : DataFrame with columns [timestamp, rack_id, cpu, gpu,
                 memory, disk_io, network].  One row per rack, same timestamp.

        Returns
        ───────
        BatchResult with per-rack predictions and global cooling decision.
        """
        t0 = time.perf_counter()

        raw_df = raw_df.sort_values("rack_id").reset_index(drop=True)
        n_racks = len(raw_df)

        # ── 1. Feature engineering ────────────────────────────────────────
        feat_df = engineer_features(raw_df)

        EXCLUDE    = ["timestamp", "rack_id"]
        feat_cols  = [c for c in feat_df.columns if c not in EXCLUDE]
        feat_array = self.scaler.transform(feat_df)[feat_cols].values.astype(np.float32)

        # ── 2. GNN: thermal embedding per rack ────────────────────────────
        node_feats = torch.tensor(
            feat_df[TELEMETRY_COLS].values, dtype=torch.float
        )
        graph       = build_rack_graph(self.adjacency, node_feats)
        embeddings  = get_thermal_embeddings(self.gnn, graph)  # [n_racks, embed_dim]

        # ── 3. XGBoost: predict thermal risk ─────────────────────────────
        X, _   = assemble_xgb_features(feat_array, embeddings, embed_dim=self.embed_dim)
        xgb_risks   = self.xgb.predict_risk(X)
        hotspot_probs = self.xgb.predict_hotspot_prob(X)

        # ── 4. GNN-derived risk signal (L2 norm of embedding, 0–100) ─────
        emb_norms   = np.linalg.norm(embeddings, axis=1)
        # Normalise to 0–100 using 95th-percentile reference
        ref_norm    = max(np.percentile(emb_norms, 95), 1e-6)
        gnn_risks   = np.clip((emb_norms / ref_norm) * 100.0, 0, 100)

        # ── 5. Weighted fusion ────────────────────────────────────────────
        # final_risk = α·xgb_risk + (1-α)·gnn_risk
        final_risks = (self.alpha * xgb_risks + (1 - self.alpha) * gnn_risks)
        final_risks = np.clip(final_risks, 0, 100)

        t1          = time.perf_counter()
        latency_ms  = (t1 - t0) * 1000.0

        # ── 6. Assemble result ────────────────────────────────────────────
        rack_ids = raw_df["rack_id"].tolist()
        preds    = []

        for i, rid in enumerate(rack_ids):
            preds.append(RackPrediction(
                rack_id          = rid,
                final_risk_score = float(final_risks[i]),
                xgb_risk         = float(xgb_risks[i]),
                gnn_risk         = float(gnn_risks[i]),
                hotspot_prob     = float(hotspot_probs[i]),
                cooling_action   = risk_to_cooling_action(final_risks[i]),
                latency_ms       = latency_ms / n_racks,
            ))

        hotspot_racks  = [p.rack_id for p in preds if p.final_risk_score >= self.thresh]
        max_risk       = float(np.max(final_risks))
        global_action  = risk_to_cooling_action(max_risk)

        return BatchResult(
            timestamp     = str(raw_df["timestamp"].iloc[0]),
            predictions   = preds,
            hotspot_racks = hotspot_racks,
            max_risk      = max_risk,
            global_action = global_action,
        )

    # ── Streaming simulation ─────────────────────────────────────────────────

    def run_streaming(
        self,
        telemetry_stream: List[pd.DataFrame],
        interval_sec:     float = 1.0,
        on_result=None,
    ):
        """
        Simulates a streaming inference loop over a list of telemetry snapshots.

        Parameters
        ──────────
        telemetry_stream : list of DataFrames (one per time step)
        interval_sec     : simulated inter-arrival interval
        on_result        : optional callback(BatchResult) for custom handling

        In production this would be replaced by a Kafka / gRPC consumer.
        """
        print("\n── Streaming Inference Started ──────────────────────────")
        for step, snapshot_df in enumerate(telemetry_stream):
            result = self.predict_batch(snapshot_df)

            if on_result:
                on_result(result)
            else:
                self._default_print(step, result)

            time.sleep(interval_sec)

        print("── Streaming Inference Ended ────────────────────────────\n")

    # ── Pretty-print helper ──────────────────────────────────────────────────

    @staticmethod
    def _default_print(step: int, result: BatchResult):
        action_icon = {
            "NORMAL":    "🟢",
            "ELEVATED":  "🟡",
            "HIGH":      "🟠",
            "EMERGENCY": "🔴",
        }
        icon = action_icon.get(result.global_action, "⚪")
        print(f"\n[Step {step:>3d}]  {result.timestamp}")
        print(f"  Global action : {icon} {result.global_action}  "
              f"(max risk={result.max_risk:.1f})")
        if result.hotspot_racks:
            print(f"  Hotspot racks : {result.hotspot_racks}")
        for p in result.predictions:
            bar = "█" * int(p.final_risk_score / 5)
            print(f"    Rack {p.rack_id}: {p.final_risk_score:>5.1f}/100  "
                  f"|  {bar:<20s}  |  P(hot)={p.hotspot_prob:.2f}  "
                  f"|  {p.cooling_action}")


# ─────────────────────────────────────────────────────────────────────────────
# FULL INTEGRATION DEMO
# ─────────────────────────────────────────────────────────────────────────────

def run_full_demo():
    """
    Trains GNN + XGBoost on synthetic data, then runs the inference pipeline
    over 5 simulated time steps.  One rack is deliberately stressed to verify
    the hotspot detection path.
    """
    from data_processing import generate_synthetic_telemetry, build_training_dataset
    from gnn_model       import train_gnn

    print("="*60)
    print("  FULL PIPELINE DEMO  (GNN + XGBoost + Fusion)")
    print("="*60)

    N_RACKS     = 4
    N_TIMESTEPS = 120
    EMBED_DIM   = 8

    # ── 1. Generate synthetic training corpus ──────────────────────────────
    print("\n[1/5] Generating synthetic training data …")
    raw = generate_synthetic_telemetry(n_racks=N_RACKS, n_timesteps=N_TIMESTEPS)
    X_df, y_ser, scaler = build_training_dataset(raw, scaler_save_path="/tmp/scaler.pkl")
    feat_cols = list(X_df.columns)

    # ── 2. Pre-train GNN ──────────────────────────────────────────────────
    print("\n[2/5] Training GNN …")
    adj      = linear_rack_adjacency(N_RACKS)
    gnn      = ThermalGNN(in_features=len(TELEMETRY_COLS), hidden_dim=16, embed_dim=EMBED_DIM)

    # Build one graph snapshot (use mean telemetry per rack as representative)
    mean_tel  = raw.groupby("rack_id")[TELEMETRY_COLS].mean().sort_index()
    node_feats = torch.tensor(mean_tel.values, dtype=torch.float)
    graph      = build_rack_graph(adj, node_feats)

    # Labels: mean risk per rack from training set
    rack_labels = (
        raw.groupby("rack_id")[TELEMETRY_COLS]
        .mean()
        .apply(lambda row: sum(0.35 * row["cpu"] + 0.30 * row["gpu"]), axis=1)
    )
    labels = torch.tensor(rack_labels.values, dtype=torch.float)
    train_gnn(gnn, graph, labels, epochs=100, lr=5e-3, verbose=False)
    print("  GNN training complete.")

    # ── 3. Train XGBoost ──────────────────────────────────────────────────
    print("\n[3/5] Training XGBoost …")
    emb_all = get_thermal_embeddings(gnn, graph)   # [N_RACKS, EMBED_DIM]

    # Expand embeddings to match all training rows by rack_id
    rack_emb_map = {rid: emb_all[i] for i, rid in enumerate(sorted(raw["rack_id"].unique()))}
    emb_matrix   = np.vstack([rack_emb_map[rid] for rid in raw["rack_id"]])

    X_np = X_df.values.astype(np.float32)
    X_fused, feat_names = assemble_xgb_features(
        X_np, emb_matrix, feature_names=feat_cols, embed_dim=EMBED_DIM
    )

    xgb_model = ThermalRiskXGB()
    metrics   = xgb_model.train(X_fused, y_ser.values, feature_names=feat_names, verbose=False)
    print("  XGBoost training complete.")
    ThermalRiskXGB._print_metrics(metrics)

    # ── 4. Build inference pipeline ───────────────────────────────────────
    print("\n[4/5] Building inference pipeline …")
    pipeline = CoolingInferencePipeline(
        gnn_model   = gnn,
        xgb_model   = xgb_model,
        scaler      = scaler,
        adjacency   = adj,
        gnn_alpha   = 0.75,
        embed_dim   = EMBED_DIM,
    )

    # ── 5. Simulate 5 time-step stream with rack-1 stressed on step 3 ────
    print("\n[5/5] Running inference stream (5 steps) …\n")

    import datetime
    stream = []
    for step in range(5):
        rows = []
        for rack in range(N_RACKS):
            cpu = 90.0 if (step == 3 and rack == 1) else np.random.uniform(20, 55)
            gpu = 88.0 if (step == 3 and rack == 1) else np.random.uniform(10, 45)
            rows.append({
                "timestamp": datetime.datetime.now(),
                "rack_id":   rack,
                "cpu":       cpu,
                "gpu":       gpu,
                "memory":    np.random.uniform(40, 80),
                "disk_io":   np.random.uniform(10, 60),
                "network":   np.random.uniform(5,  50),
            })
        stream.append(pd.DataFrame(rows))

    pipeline.run_streaming(stream, interval_sec=0.1)

    print("inference_pipeline.py  ✓")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_full_demo()
