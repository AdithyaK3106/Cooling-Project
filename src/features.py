"""
src/features.py
---------------
SINGLE SOURCE OF TRUTH for ALL feature engineering logic.

CRITICAL RULE: Both training and inference MUST use this file.
No normalization, scaling, or feature derivation logic may live anywhere else.

============================================================
FRD FEATURE CONTRACT (6-dimensional, FIXED ORDER)
============================================================

  Index  Name            Formula
  -----  --------------- ------------------------------------------
  [0]    cpu_norm        cpu / 100                         in [0,1]
  [1]    gpu_norm        gpu / 100                         in [0,1]
  [2]    memory_norm     memory / 100                      in [0,1]
  [3]    disk_io_norm    log(1+disk_io) / max_disk_log     in [0,1]
  [4]    network_io_norm log(1+network_io) / max_net_log   in [0,1]
  [5]    gnn_embedding   AnalyticGNN scalar                in [0,1]

  Total: 6 features. No rolling windows. No deltas.

  Rationale for Option A (6 features):
    - Matches FRD specification exactly.
    - Inference latency: sub-millisecond (no stateful buffer required).
    - Temporal context is encoded via heat_norm inside gnn_embedding.
    - Simpler model = easier to audit, explain, and certify.

============================================================
NORMALIZATION CONTRACT (PRD §3.2)
============================================================

  cpu_norm        = cpu / 100
  gpu_norm        = gpu / 100
  memory_norm     = memory / 100
  disk_io_norm    = log(1 + disk_io) / max_disk_log    → clip [0,1]
  network_io_norm = log(1 + network_io) / max_net_log  → clip [0,1]

  NO StandardScaler. NO Z-score. CLIP all outputs to [0,1].

============================================================
GNN EMBEDDING CONTRACT (PRD §4.1)
============================================================

  heat_norm     = 0.6 * cpu_norm + 0.4 * gpu_norm          in [0,1]
  neighbor_heat = mean(heat_norm of neighbors)  [= heat_norm if isolated]
  gnn_embedding = clip(0.7 * heat_norm + 0.3 * neighbor_heat, 0, 1)

  Output: scalar in [0,1]. No PyTorch. No multi-dim vectors.
  Assertion: raises ValueError if result outside [0,1].

============================================================
SCHEMA (PRD v1.0)
============================================================

  Required raw input keys: cpu, gpu, memory, disk_io, network_io
  Raises ValueError if any key is missing.

============================================================
PERSISTENCE
============================================================

  models/preprocessor_state.pkl  <- fit() saves {max_disk_log, max_net_log}
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, List, Optional


# =============================================================================
# CONSTANTS
# =============================================================================

REQUIRED_KEYS: List[str] = ['cpu', 'gpu', 'memory', 'disk_io', 'network_io']

# Canonical 6-feature names in fixed order (FRD §3.1)
FEATURE_NAMES: List[str] = [
    'cpu_norm',
    'gpu_norm',
    'memory_norm',
    'disk_io_norm',
    'network_io_norm',
    'gnn_embedding',
]

FEATURE_DIM: int = len(FEATURE_NAMES)  # 6 — enforced by assertion in process_single()


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

def validate_raw_input(raw_point: dict) -> None:
    """
    Validate that a raw telemetry dict contains all required keys.

    Raises
    ------
    ValueError : listing every missing key, with a hint for the common
                 'network' vs 'network_io' mistake.
    """
    missing = [k for k in REQUIRED_KEYS if k not in raw_point]
    if missing:
        hint = ""
        if 'network_io' in missing and 'network' in raw_point:
            hint = " (found 'network' — rename to 'network_io' per PRD §2.1)"
        raise ValueError(
            f"[FeatureProcessor] Missing required keys: {missing}.{hint}\n"
            f"  Required: {REQUIRED_KEYS}\n"
            f"  Got:      {list(raw_point.keys())}"
        )


# =============================================================================
# ANALYTIC GNN  (production-grade, zero PyTorch dependency)
# =============================================================================

class AnalyticGNN:
    """
    Deterministic GNN approximation for production runtime.

    Implements PRD §4.1 formula without learned parameters:

        heat_norm     = 0.6 * cpu_norm + 0.4 * gpu_norm
        neighbor_heat = mean(heat_norm of neighbors)   [= heat_norm if isolated]
        gnn_embedding = clip(0.7 * heat_norm + 0.3 * neighbor_heat, 0, 1)

    Output: scalar in [0, 1] per rack.
    Raises ValueError if result outside [0, 1] (should never happen after clip,
    but guards against NaN propagation).
    """

    def __init__(
        self,
        adjacency: Optional[List[tuple]] = None,
        n_nodes: int = 1,
    ):
        """
        Parameters
        ----------
        adjacency : list of (src, dst) int pairs for the rack graph.
                    Pass None for isolated single-rack mode.
        n_nodes   : total number of rack nodes.
        """
        self.n_nodes = n_nodes
        self._neighbors: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}

        if adjacency:
            for src, dst in adjacency:
                if 0 <= src < n_nodes and 0 <= dst < n_nodes:
                    self._neighbors[src].append(dst)

    @staticmethod
    def heat_proxy(cpu_norm: float, gpu_norm: float) -> float:
        """
        Compute heat proxy from normalized CPU and GPU.

        Formula: heat_norm = 0.6 * cpu_norm + 0.4 * gpu_norm
        Both inputs must already be in [0, 1] (i.e., divided by 100).
        """
        return float(np.clip(0.6 * cpu_norm + 0.4 * gpu_norm, 0.0, 1.0))

    def compute_single(
        self,
        heat_norm: float,
        neighbor_heats: Optional[List[float]] = None,
    ) -> float:
        """
        Compute GNN embedding for a single rack.

        Parameters
        ----------
        heat_norm      : float in [0, 1] — own heat proxy (already normalized).
        neighbor_heats : list of float in [0, 1] for each adjacent rack.
                         If None or empty → isolated mode (neighbor = self).

        Returns
        -------
        gnn_embedding : float in [0, 1]

        Raises
        ------
        ValueError : if result is NaN or outside [0, 1].
        """
        if neighbor_heats:
            neighbor_heat = float(np.mean(neighbor_heats))
        else:
            neighbor_heat = float(heat_norm)  # isolated rack

        result = float(np.clip(0.7 * heat_norm + 0.3 * neighbor_heat, 0.0, 1.0))

        if np.isnan(result) or not (0.0 <= result <= 1.0):
            raise ValueError(
                f"[AnalyticGNN] Embedding out of bounds: {result:.6f}. "
                f"heat_norm={heat_norm}, neighbor_heat={neighbor_heat}."
            )
        return result

    def compute_batch(self, heat_norms: np.ndarray) -> np.ndarray:
        """
        Compute GNN embeddings for all nodes in the graph.

        Parameters
        ----------
        heat_norms : np.ndarray of shape (n_nodes,), values in [0, 1].

        Returns
        -------
        embeddings : np.ndarray of shape (n_nodes,), values in [0, 1].
        """
        embeddings = np.zeros(self.n_nodes, dtype=np.float64)
        for i in range(self.n_nodes):
            neighbors = self._neighbors[i]
            neighbor_heat = (
                float(np.mean([heat_norms[j] for j in neighbors]))
                if neighbors
                else float(heat_norms[i])
            )
            embeddings[i] = np.clip(0.7 * heat_norms[i] + 0.3 * neighbor_heat, 0.0, 1.0)
        return embeddings


# =============================================================================
# FEATURE PROCESSOR  (single source of truth)
# =============================================================================

class FeatureProcessor:
    """
    Unified 6-feature engineering engine (FRD §3.1).

    Contract
    --------
    - Fit ONCE on training data → captures max_disk_log, max_net_log.
    - process_single() is IDENTICAL whether called from training or inference.
    - Output is always shape (1, 6), dtype float64, all values in [0, 1].
    - Schema: input dict MUST contain: cpu, gpu, memory, disk_io, network_io.
    - No StandardScaler. No Z-score. No rolling windows. No stateful buffer.

    Why no rolling windows in the FRD feature set?
    - Rolling windows require stateful warm-up, which creates training/inference
      asymmetry (first N samples have different distributions).
    - The gnn_embedding already encodes thermal momentum via the heat proxy.
    - Rolling features can be added in Phase 2 as a named feature set extension,
      with full parity enforcement via assert_parity().
    """

    # Class-level constant — import this instead of hardcoding anywhere
    FEATURE_NAMES: List[str] = FEATURE_NAMES
    FEATURE_DIM:   int       = FEATURE_DIM

    def __init__(self):
        # Log-scaling bounds: set by fit(), loaded at inference time
        self.stats: Dict[str, float] = {
            'max_disk_log': 1.0,
            'max_net_log':  1.0,
        }
        # GNN for single-rack mode (default; override for multi-rack)
        self._gnn = AnalyticGNN(adjacency=None, n_nodes=1)

    def set_gnn(self, gnn: AnalyticGNN) -> None:
        """Override the GNN instance (e.g. for multi-rack topology)."""
        self._gnn = gnn

    # -------------------------------------------------------------------------
    # FITTING  (training-time only)
    # -------------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> None:
        """
        Calculate log-scaling bounds from the training dataset.

        Reads columns: disk_io, network_io  (PRD schema).
        Crash-fast if wrong column names are present.

        Parameters
        ----------
        df : DataFrame with at least columns 'disk_io' and 'network_io'.
        """
        if 'disk_io' not in df.columns:
            raise ValueError(
                "[FeatureProcessor.fit] Missing 'disk_io' column. PRD schema violation."
            )
        if 'network_io' not in df.columns:
            raise ValueError(
                "[FeatureProcessor.fit] Missing 'network_io' column. "
                "Use 'network_io', not 'network' (PRD §2.1)."
            )

        max_disk = float(np.log1p(df['disk_io']).max())
        max_net  = float(np.log1p(df['network_io']).max())

        self.stats['max_disk_log'] = max(max_disk, 1e-6)
        self.stats['max_net_log']  = max(max_net,  1e-6)

        print(
            f"[FeatureProcessor] Fitted: "
            f"max_disk_log={self.stats['max_disk_log']:.4f}, "
            f"max_net_log={self.stats['max_net_log']:.4f}"
        )

    # -------------------------------------------------------------------------
    # PERSISTENCE
    # -------------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist fitted scaling bounds to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.stats, f)
        print(f"[FeatureProcessor] State saved -> {path}")

    def load(self, path: str) -> None:
        """Load scaling bounds for inference."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[FeatureProcessor] Preprocessor state not found at '{path}'. "
                f"Run the training pipeline first."
            )
        with open(path, 'rb') as f:
            self.stats = pickle.load(f)
        print(f"[FeatureProcessor] State loaded <- {path}")

    # -------------------------------------------------------------------------
    # CORE TRANSFORMATION  (IDENTICAL in training and inference)
    # -------------------------------------------------------------------------

    def process_single(self, raw_point: dict) -> np.ndarray:
        """
        Transform a single raw telemetry dict into the canonical 6-feature vector.

        Steps
        -----
        1. Schema validation  — raises ValueError on missing keys.
        2. Base normalization  — cpu/100, gpu/100, memory/100.
        3. Log-scaling         — disk_io and network_io via log1p / max_log.
        4. Heat proxy          — 0.6*cpu_norm + 0.4*gpu_norm.
        5. GNN embedding       — AnalyticGNN.compute_single(heat_norm).
        6. Assembly            — 6-dim vector in fixed FRD order.
        7. NaN guard           — replace any NaN with 0.
        8. Clip                — all values to [0, 1].
        9. Dimension assert    — hard fail if not exactly 6 dims.

        Parameters
        ----------
        raw_point : dict with keys: cpu, gpu, memory, disk_io, network_io.
                    Values are raw, unscaled measurements (e.g. cpu in [0,100],
                    disk_io in bytes/sec).

        Returns
        -------
        vector : np.ndarray of shape (1, 6), dtype float64, all values in [0,1].

        Raises
        ------
        ValueError      : if required keys are missing.
        AssertionError  : if output is not exactly 6-dimensional (internal bug).
        """
        # Step 1 — schema validation (hard fail)
        validate_raw_input(raw_point)

        # Step 2 — base normalization
        cpu_n  = float(raw_point['cpu'])    / 100.0
        gpu_n  = float(raw_point['gpu'])    / 100.0
        mem_n  = float(raw_point['memory']) / 100.0

        # Step 3 — log-scaling for I/O (byte-rate magnitudes)
        disk_log = np.log1p(float(raw_point['disk_io']))
        net_log  = np.log1p(float(raw_point['network_io']))

        disk_io_n = disk_log / self.stats['max_disk_log']
        net_io_n  = net_log  / self.stats['max_net_log']

        # Step 4 — heat proxy (inputs already in [0,1])
        heat_n = AnalyticGNN.heat_proxy(cpu_n, gpu_n)

        # Step 5 — GNN embedding (scalar in [0,1])
        gnn_emb = self._gnn.compute_single(heat_norm=heat_n)

        # Step 6 — assemble in FRD-mandated order
        features = [cpu_n, gpu_n, mem_n, disk_io_n, net_io_n, gnn_emb]

        # Step 7 — NaN guard
        vector = np.array(features, dtype=np.float64)
        if np.any(np.isnan(vector)):
            vector = np.nan_to_num(vector, nan=0.0)

        # Step 8 — clip to [0, 1]
        vector = np.clip(vector, 0.0, 1.0)

        # Step 9 — dimension assertion (must equal FEATURE_DIM = 6)
        assert len(vector) == FEATURE_DIM, (
            f"[FeatureProcessor] Dimension error: expected {FEATURE_DIM}, "
            f"got {len(vector)}. This is an internal bug."
        )

        return vector.reshape(1, -1)

    def process_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Process an entire DataFrame row-by-row using process_single().

        This is the training-pipeline entry point. Each row is processed
        IDENTICALLY to how the inference engine processes live telemetry,
        guaranteeing training-serving parity.

        Parameters
        ----------
        df : DataFrame with required columns (validated per-row).

        Returns
        -------
        X : np.ndarray of shape (n_rows, 6)
        """
        rows = []
        for _, row in df.iterrows():
            vec = self.process_single(row.to_dict())
            rows.append(vec.flatten())
        return np.vstack(rows)

    @property
    def feature_names(self) -> List[str]:
        """Canonical ordered list of feature names (FRD §3.1)."""
        return self.FEATURE_NAMES
