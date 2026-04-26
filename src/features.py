"""
src/features.py
───────────────
SINGLE SOURCE OF TRUTH for all feature engineering logic.

⚠️  CRITICAL RULE: Both training and inference MUST use this file.
    No normalization, scaling, or feature derivation logic may live anywhere else.

Schema (PRD v1.0):
    Raw input keys: cpu, gpu, memory, disk_io, network_io

Normalization contract (PRD §3.2):
    cpu_norm        = cpu / 100
    gpu_norm        = gpu / 100
    memory_norm     = memory / 100
    disk_io_norm    = log(1 + disk_io) / max_disk_log   → clipped [0,1]
    network_io_norm = log(1 + network_io) / max_net_log → clipped [0,1]

    NO StandardScaler. NO Z-score. CLIP all outputs to [0,1].

GNN embedding contract (PRD §4.1):
    heat_norm        = 0.6 * cpu_norm + 0.4 * gpu_norm
    neighbor_heat    = mean(heat_norm of neighbors)                  [default=heat_norm if isolated]
    gnn_embedding    = clip(0.7 * heat_norm + 0.3 * neighbor_heat, 0, 1)

    Output is a SCALAR in [0,1].  No PyTorch. No multi-dim vectors.

Risk fusion contract (PRD §4.2):
    risk = clip(0.75 * xgb_prediction + 0.25 * gnn_embedding, 0, 1)

Feature vector (15-dim, FIXED ORDER):
    [0]  cpu_norm
    [1]  gpu_norm
    [2]  mem_norm
    [3]  disk_io_norm
    [4]  net_io_norm
    [5]  heat_norm
    [6]  cpu_roll_5       (rolling mean over last 5 steps)
    [7]  gpu_roll_5
    [8]  heat_roll_5
    [9]  cpu_roll_10      (rolling mean over last 10 steps)
    [10] gpu_roll_10
    [11] heat_roll_10
    [12] cpu_delta        (current − previous step)
    [13] gpu_delta
    [14] heat_delta

Persistence:
    models/preprocessor_state.pkl  ← fit() saves max_disk_log, max_net_log
"""

import numpy as np
import pandas as pd
import pickle
import os
from collections import deque
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTIC GNN  (Production-grade, zero PyTorch dependency)
# ─────────────────────────────────────────────────────────────────────────────

class AnalyticGNN:
    """
    Lightweight, deterministic GNN approximation for production runtime.

    Implements the PRD §4.1 formula without any learned parameters:
        heat_norm     = 0.6 * cpu_norm + 0.4 * gpu_norm
        neighbor_heat = mean(heat_norm of neighbors)
        gnn_embedding = clip(0.7 * heat_norm + 0.3 * neighbor_heat, 0, 1)

    For single-rack (no adjacency info): gnn_embedding == heat_norm.

    Output:
        scalar in [0, 1] per rack.
    """

    def __init__(self, adjacency: Optional[List[tuple]] = None, n_nodes: int = 1):
        """
        Parameters
        ──────────
        adjacency : list of (src, dst) int tuples representing rack graph edges.
                    Pass None for single-rack (isolated) mode.
        n_nodes   : total number of racks/nodes.
        """
        self.n_nodes   = n_nodes
        self._neighbors: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}

        if adjacency:
            for src, dst in adjacency:
                if 0 <= src < n_nodes and 0 <= dst < n_nodes:
                    self._neighbors[src].append(dst)

    @staticmethod
    def _heat(cpu_norm: float, gpu_norm: float) -> float:
        return float(np.clip(0.6 * cpu_norm + 0.4 * gpu_norm, 0.0, 1.0))

    def compute(self, heat_norms: np.ndarray) -> np.ndarray:
        """
        Compute GNN embedding for all nodes.

        Parameters
        ──────────
        heat_norms : np.ndarray of shape [n_nodes], values in [0,1].

        Returns
        ───────
        gnn_embeddings : np.ndarray of shape [n_nodes], values in [0,1].
        """
        embeddings = np.zeros(self.n_nodes, dtype=np.float64)

        for i in range(self.n_nodes):
            self_heat    = float(heat_norms[i])
            neighbors    = self._neighbors[i]

            if neighbors:
                neighbor_heat = float(np.mean([heat_norms[j] for j in neighbors]))
            else:
                neighbor_heat = self_heat   # isolated node → no external pressure

            embeddings[i] = np.clip(0.7 * self_heat + 0.3 * neighbor_heat, 0.0, 1.0)

        return embeddings

    def compute_single(self, heat_norm: float, neighbor_heats: Optional[List[float]] = None) -> float:
        """
        Compute GNN embedding for a single rack.

        Parameters
        ──────────
        heat_norm       : float in [0,1], the rack's own heat proxy.
        neighbor_heats  : list of neighbor heat_norm values (if known).

        Returns
        ───────
        gnn_embedding : float in [0,1].
        """
        if neighbor_heats:
            neighbor_heat = float(np.mean(neighbor_heats))
        else:
            neighbor_heat = heat_norm

        return float(np.clip(0.7 * heat_norm + 0.3 * neighbor_heat, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE PROCESSOR  (Single source of truth for normalization + features)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureProcessor:
    """
    Unified Feature Engineering Engine.

    Contract
    ────────
    • Fit once on training data (captures max_disk_log, max_net_log).
    • Identical process_single() used by BOTH training pipeline and runtime.
    • No StandardScaler. No Z-score. All outputs clipped to [0,1].
    • Feature vector is exactly 15 dimensions in fixed order.
    • Schema: raw dict must contain keys: cpu, gpu, memory, disk_io, network_io
    """

    FEATURE_NAMES: List[str] = [
        'cpu_norm', 'gpu_norm', 'mem_norm',
        'disk_io_norm', 'net_io_norm',
        'heat_norm',
        'cpu_roll_5', 'gpu_roll_5', 'heat_roll_5',
        'cpu_roll_10', 'gpu_roll_10', 'heat_roll_10',
        'cpu_delta', 'gpu_delta', 'heat_delta',
    ]

    def __init__(self, history_size: int = 10):
        self.history_size  = history_size
        self.buffer: deque = deque(maxlen=history_size)

        # Scaling stats: set by fit(), persisted to disk
        self.stats: Dict[str, float] = {
            'max_disk_log': 1.0,
            'max_net_log':  1.0,
        }

    # ── Fitting (training-time only) ─────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> None:
        """
        Calculate log-scaling bounds from the training dataset.

        Reads columns: disk_io, network_io  (PRD schema).
        Prevents division-by-zero with fallback to 1.0.
        """
        if 'disk_io' not in df.columns:
            raise ValueError("Training DataFrame missing 'disk_io' column (PRD schema violation).")
        if 'network_io' not in df.columns:
            raise ValueError("Training DataFrame missing 'network_io' column (PRD schema — use 'network_io', not 'network').")

        max_disk = float(np.log1p(df['disk_io']).max())
        max_net  = float(np.log1p(df['network_io']).max())

        self.stats['max_disk_log'] = max(max_disk, 1e-6)
        self.stats['max_net_log']  = max(max_net,  1e-6)

        print(
            f"[FeatureProcessor] Fitted: "
            f"max_disk_log={self.stats['max_disk_log']:.4f}, "
            f"max_net_log={self.stats['max_net_log']:.4f}"
        )

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist fitted scaling bounds to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.stats, f)
        print(f"[FeatureProcessor] State saved → {path}")

    def load(self, path: str) -> None:
        """Load scaling bounds for inference."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[FeatureProcessor] Missing preprocessor state at '{path}'. "
                f"Run the training pipeline first to generate it."
            )
        with open(path, 'rb') as f:
            self.stats = pickle.load(f)
        print(f"[FeatureProcessor] State loaded ← {path}")

    def reset_buffer(self) -> None:
        """Reset the rolling history buffer (e.g. on restart)."""
        self.buffer.clear()

    # ── Core transformation ──────────────────────────────────────────────────

    def process_single(self, raw_point: dict) -> np.ndarray:
        """
        Transform a single telemetry dict into a validated 15-feature vector.

        Parameters
        ──────────
        raw_point : dict with keys: cpu, gpu, memory, disk_io, network_io
                    (All values are raw, unscaled measurements.)

        Returns
        ───────
        vector : np.ndarray of shape (1, 15), dtype float64, all values in [0,1].

        Raises
        ──────
        AssertionError : if output vector is not exactly 15 dimensions.
        """
        # ── 1. Base normalization (PRD §3.2) ─────────────────────────────────
        cpu_n  = float(raw_point.get('cpu',    0.0)) / 100.0
        gpu_n  = float(raw_point.get('gpu',    0.0)) / 100.0
        mem_n  = float(raw_point.get('memory', 0.0)) / 100.0

        # Log-scaling for I/O (handles byte-rate magnitudes correctly)
        disk_log  = np.log1p(float(raw_point.get('disk_io',    0.0)))
        net_log   = np.log1p(float(raw_point.get('network_io', 0.0)))

        disk_io_n = disk_log / self.stats['max_disk_log']
        net_io_n  = net_log  / self.stats['max_net_log']

        # ── 2. Heat proxy ─────────────────────────────────────────────────────
        heat_n = 0.6 * cpu_n + 0.4 * gpu_n   # PRD §4.1 formula

        # ── 3. Buffer update for temporal features ────────────────────────────
        step = {
            'cpu_norm':  cpu_n,
            'gpu_norm':  gpu_n,
            'heat_norm': heat_n,
        }
        self.buffer.append(step)
        hist_df = pd.DataFrame(list(self.buffer))

        # ── 4. Rolling means (graceful during warmup: tail(N) on short buffer) ─
        roll_5  = hist_df.tail(5).mean()
        roll_10 = hist_df.tail(10).mean()

        # ── 5. Deltas (zero on first step) ───────────────────────────────────
        if len(self.buffer) > 1:
            delta_cpu  = self.buffer[-1]['cpu_norm']  - self.buffer[-2]['cpu_norm']
            delta_gpu  = self.buffer[-1]['gpu_norm']  - self.buffer[-2]['gpu_norm']
            delta_heat = self.buffer[-1]['heat_norm'] - self.buffer[-2]['heat_norm']
        else:
            delta_cpu = delta_gpu = delta_heat = 0.0

        # ── 6. Assembly (FIXED ORDER — must match FEATURE_NAMES) ─────────────
        features = [
            cpu_n,   gpu_n,   mem_n,                                   # [0-2]
            disk_io_n, net_io_n,                                        # [3-4]
            heat_n,                                                      # [5]
            roll_5['cpu_norm'],  roll_5['gpu_norm'],  roll_5['heat_norm'],  # [6-8]
            roll_10['cpu_norm'], roll_10['gpu_norm'], roll_10['heat_norm'], # [9-11]
            delta_cpu, delta_gpu, delta_heat,                           # [12-14]
        ]

        # ── 7. Validation layer ───────────────────────────────────────────────
        vector = np.array(features, dtype=np.float64)

        # Replace any NaN introduced by degenerate inputs
        if np.any(np.isnan(vector)):
            vector = np.nan_to_num(vector, nan=0.0)

        # Clip ALL values to [0, 1]  (PRD §3.2 — mandatory)
        vector = np.clip(vector, 0.0, 1.0)

        assert len(vector) == 15, (
            f"[FeatureProcessor] Feature dimension error: expected 15, got {len(vector)}. "
            f"This is a bug — do not catch this assertion."
        )

        return vector.reshape(1, -1)

    def process_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Process an entire DataFrame row-by-row, simulating the real-time loop.
        Preserves temporal state across rows (intentional — matches inference behavior).

        Returns
        ───────
        X : np.ndarray of shape (n_rows, 15)
        """
        self.reset_buffer()
        rows = []
        for _, row in df.iterrows():
            vec = self.process_single(row.to_dict())
            rows.append(vec.flatten())
        return np.vstack(rows)

    @property
    def feature_names(self) -> List[str]:
        """Return the canonical ordered list of feature names."""
        return self.FEATURE_NAMES
