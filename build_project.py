#!/usr/bin/env python3
"""build_project.py - self-contained builder for Phase 1 of
"Sub-Microsecond Intrusion Prevention: An eBPF-Driven Deep Reinforcement
Learning and TCN Framework for High-Speed Networks".

Standard library only (pathlib, zipfile, os, sys, argparse). Running it:

  1. writes every project file (embedded below as string literals) into --out,
  2. packages the tree into <out>/sub-microsecond-ebpf-drl-tcn-phase1.zip,
  3. re-opens the archive, lists the members and prints count + total size.

Usage:
  python build_project.py --out ./sub-microsecond-phase1 [--force]

User data is never bundled: data/ CSVs, artifacts/ runtime outputs (except
.gitkeep), __pycache__/, *.pyc, .git/, .venv/, venv/ and files > 50 MB are
excluded from the zip.
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path

ZIP_NAME = "sub-microsecond-ebpf-drl-tcn-phase1.zip"
MAX_FILE_BYTES = 50 * 1024 * 1024
EXCLUDED_DIRS = {"__pycache__", ".git", ".venv", "venv"}
EMPTY_FILES = ["data/cicids2017/.gitkeep", "data/cicids2018/.gitkeep", "artifacts/.gitkeep"]

FILES = {}

# ==========================================================================
# src/__init__.py
# ==========================================================================
FILES['src/__init__.py'] = r'''"""Phase-1 detection brain package (preprocessing, models, training, evaluation, inference API)."""
'''

# ==========================================================================
# src/config.py
# ==========================================================================
FILES['src/config.py'] = r'''"""Central configuration for Phase 1 of
"Sub-Microsecond Intrusion Prevention: An eBPF-Driven Deep Reinforcement
Learning and TCN Framework for High-Speed Networks".

Every magic number used anywhere in ``src/`` lives here. Paths are resolved
relative to the project root with :mod:`pathlib`, so the project can be moved
or unzipped anywhere.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

try:  # torch is optional for preprocessing; DEVICE falls back to CPU without it.
    import torch

    _CUDA_AVAILABLE: bool = bool(torch.cuda.is_available())
except Exception:  # pragma: no cover - torch missing or broken
    _CUDA_AVAILABLE = False

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_ROOT: Path = PROJECT_ROOT / "data"
DATA_DIRS: Dict[str, Path] = {
    "2017": DATA_ROOT / "cicids2017",
    "2018": DATA_ROOT / "cicids2018",
}
ARTIFACT_DIR: Path = PROJECT_ROOT / "artifacts"
EVAL_DIR: Path = ARTIFACT_DIR / "eval"
TB_DIR: Path = ARTIFACT_DIR / "tensorboard"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED: int = 42
DETERMINISTIC: bool = True  # cudnn.deterministic=True (spec §5)

# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
TRAIN_FRACTION: float = 0.80          # headline 80/20 split
INNER_VAL_FRACTION: float = 0.10      # 10 % of the 80 % -> early stopping + weights

# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
WINDOW_SIZE: int = 10
STRIDE: int = 5
WINDOW_WRITE_CHUNK: int = 100_000     # windows materialised per chunk (RAM bound)
LOAD_CHUNK_ROWS: int = 500_000        # CSV rows streamed per chunk during preprocessing
MIN_FREE_DISK_GB: int = 40            # warn below this (full 2017+2018 run, temp files included)

# --------------------------------------------------------------------------- #
# TCN
# --------------------------------------------------------------------------- #
TCN_CHANNELS: List[int] = [64, 128, 128]
TCN_KERNEL: int = 3
TCN_DILATIONS: List[int] = [1, 2, 4, 8]
TCN_EMBED_DIM: int = 128
DROPOUT: float = 0.3
HIDDEN_DIMS: List[int] = [128, 64, 32]
NUM_CLASSES: int = 2

# --------------------------------------------------------------------------- #
# Random Forest
# --------------------------------------------------------------------------- #
RF_N_ESTIMATORS: int = 200
RF_MAX_DEPTH: int = 20
RF_MIN_SAMPLES_LEAF: int = 5
RF_CALIBRATION_CV: int = 3
RF_CALIBRATION_METHOD: str = "isotonic"
# False (default): isotonic calibrator learned on 3-fold out-of-fold predictions,
#   then ONE RandomForest fit on all training flows is kept. Peak RAM = one
#   forest (fits 16 GB machines) and inference is 3x faster than ensemble=True.
# True: sklearn default - keeps 3 (RF, calibrator) pairs, each fit on 2/3 of the data.
RF_CALIBRATION_ENSEMBLE: bool = False
# Bootstrap size per tree. None = sklearn default (each tree draws N samples
# with replacement). A fraction such as 0.5 makes each tree smaller (less RAM,
# faster) while the forest as a whole still draws from every training flow.
# Only change this if the RF stage runs out of memory.
RF_MAX_SAMPLES: float | None = None
# "holdout": fit an extra calibrated RF on the inner-train flows only, so the
#            ensemble weight search on inner-val is not biased toward an RF
#            that has already seen inner-val flows (recommended, ~2x RF time).
# "final":   reuse the final RF (fit on the full 80 %) for weight search
#            (faster, but optimistically biased toward RF).
RF_SELECTION_MODE: str = "holdout"
TRANSFER_RF_N_ESTIMATORS: int = 100   # lighter RF for the cross-dataset matrix
RF_PREDICT_BATCH: int = 500_000

# --------------------------------------------------------------------------- #
# DRL
# --------------------------------------------------------------------------- #
DRL_ALGO: str = "double_dqn"          # {"double_dqn", "ppo"}
DRL_ALGOS: List[str] = ["double_dqn", "ppo"]
GAMMA: float = 0.99
LR: float = 1e-3
BATCH_SIZE: int = 256
REPLAY_BUFFER: int = 100_000          # minimum capacity; grows to hold the full 80 %
TARGET_UPDATE: int = 1000
EPS_START: float = 1.0
EPS_END: float = 0.05
EPS_DECAY_STEPS: int = 50_000
DRL_EPOCHS: int = 10
DRL_HIDDEN: List[int] = [128, 64]
HUBER_DELTA: float = 1.0
GRAD_CLIP_NORM: float = 10.0
# "epsilon":    each DRL epoch makes one full ε-greedy pass over all training
#               windows, fills the buffer with every transition, then updates.
# "exhaustive": the buffer holds every (state, action) pair for all 3 actions
#               (rewards are a known function of label & action).
DRL_BUFFER_MODE: str = "epsilon"
PPO_CLIP: float = 0.2
PPO_GAE_LAMBDA: float = 0.95
PPO_UPDATE_EPOCHS: int = 4
PPO_ENTROPY_COEF: float = 0.01
PPO_VALUE_COEF: float = 0.5

# Reward table: REWARDS[label][action]; label 0=benign, 1=attack
REWARDS: Dict[int, Dict[str, float]] = {
    0: {"pass": 1.0, "monitor": -0.5, "block": -2.0},
    1: {"pass": -2.0, "monitor": 1.0, "block": 2.0},
}

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
EPOCHS: int = 30
MIN_EPOCHS: int = 10                  # early stopping may not fire before this
PATIENCE: int = 5
WEIGHT_DECAY: float = 1e-4
PLATEAU_FACTOR: float = 0.5
PLATEAU_PATIENCE: int = 3
USE_AMP: bool = True
DEVICE: str = "cuda" if _CUDA_AVAILABLE else "cpu"
NUM_WORKERS: int = 4
PIN_MEMORY: bool = True
EVAL_BATCH_SIZE: int = 4096
# Arrays smaller than this are copied into RAM; larger ones stay memory-mapped.
MAX_IN_MEMORY_BYTES: int = 4 * 1024 ** 3   # sized for a 16 GB RAM machine

# --------------------------------------------------------------------------- #
# Ensemble / decision
# --------------------------------------------------------------------------- #
W_DNN_GRID: List[float] = [i / 20 for i in range(21)]
NON_DEGENERATE_MIN_W: float = 0.05
ENSEMBLE_WEIGHT_SELECTION: str = "unconstrained"  # or "non_degenerate"
DECISION_THRESHOLD: float = 0.5
TAU: float = 0.75
K_CONFIRM: int = 3
TAU_HARD: float = 0.95
USE_DRL_AT_INFERENCE: bool = False

# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
ACTIONS: List[str] = ["pass", "monitor", "block"]

# --------------------------------------------------------------------------- #
# Optional row cap (debugging only; None = use every row)
# --------------------------------------------------------------------------- #
MAX_ROWS_PER_FILE: Optional[int] = None

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
BOOTSTRAP_RESAMPLES: int = 300
BOOTSTRAP_CI: float = 0.95
TAU_SWEEP: List[float] = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
K_SWEEP: List[int] = [1, 2, 3, 4, 5, 7, 10]

# --------------------------------------------------------------------------- #
# Phase-2 compatibility block (consumed by the Step-3 eBPF/XDP userspace agent)
# --------------------------------------------------------------------------- #
FEATURE_ORDER_PATH: str = "artifacts/feature_columns.json"
SCALER_PATH: str = "artifacts/scaler.joblib"
BLOCK_TTL_SECONDS: int = 300
DECISION_OUTPUT_SCHEMA: str = "artifacts/decision_schema.json"

# --------------------------------------------------------------------------- #
# Artifact file names (all under ARTIFACT_DIR)
# --------------------------------------------------------------------------- #
ART = {
    "feature_columns": "feature_columns.json",
    "dropped_columns": "dropped_columns.json",
    "label_encoders": "label_encoders.joblib",
    "split_indices": "split_indices.npz",
    "scaler": "scaler.joblib",
    "class_weights": "class_weights.npy",
    "X_train": "X_train.npy",
    "y_train": "y_train.npy",
    "train_attack_type": "y_train_attack_type.npy",
    "train_inner_val": "train_inner_val_mask.npy",
    "train_group": "train_window_group.npy",
    "X_test": "X_test.npy",
    "y_test": "y_test.npy",
    "test_attack_type": "y_test_attack_type.npy",
    "test_group": "test_window_group.npy",
    "test_source": "test_window_source.npy",
    "X_flows_train": "X_flows_train.npy",
    "y_flows_train": "y_flows_train.npy",
    "flows_train_inner_val": "flows_train_inner_val_mask.npy",
    "flows_train_source": "flows_train_source.npy",
    "preprocess_summary": "preprocess_summary.json",
    "tcn": "tcn_best.pt",
    "rf": "rf_calibrated.joblib",
    "rf_selection": "rf_selection_holdout.joblib",
    "ensemble_weights": "ensemble_weights.json",
    "drl": "drl_agent.pt",
    "train_log": "train_log.csv",
    "drl_log": "drl_log.csv",
    "decision_schema": "decision_schema.json",
    "train_manifest": "train_manifest.json",
}


def artifact_path(key: str, artifact_dir: Optional[Path] = None) -> Path:
    """Return the absolute path of a named artifact.

    Args:
        key: Key into :data:`ART`.
        artifact_dir: Override for :data:`ARTIFACT_DIR`.

    Returns:
        Absolute :class:`Path` of the artifact.
    """
    base = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    return base / ART[key]
'''

# ==========================================================================
# src/schema.py
# ==========================================================================
FILES['src/schema.py'] = r'''"""Canonical CICIDS2017 / CICIDS2018 column lists and the 2018 -> 2017 mapping.

The lists below are copied verbatim from the project specification. The 2017
list contains ``"Fwd Header Length"`` twice; :data:`CICIDS2017_FEATURES` is the
de-duplicated version (first occurrence kept) and is the canonical feature
order. After constant-column removal the final order is persisted to
``artifacts/feature_columns.json`` - the exact array the Phase-2 XDP feature
extractor must emit.
"""
from __future__ import annotations

from typing import Dict, List

CICIDS2017_COLUMNS = [
    "Destination Port","Flow Duration","Total Fwd Packets","Total Backward Packets",
    "Total Length of Fwd Packets","Total Length of Bwd Packets",
    "Fwd Packet Length Max","Fwd Packet Length Min","Fwd Packet Length Mean","Fwd Packet Length Std",
    "Bwd Packet Length Max","Bwd Packet Length Min","Bwd Packet Length Mean","Bwd Packet Length Std",
    "Flow Bytes/s","Flow Packets/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min",
    "Fwd IAT Total","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min",
    "Bwd IAT Total","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min",
    "Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags",
    "Fwd Header Length","Bwd Header Length","Fwd Packets/s","Bwd Packets/s",
    "Min Packet Length","Max Packet Length","Packet Length Mean","Packet Length Std","Packet Length Variance",
    "FIN Flag Count","SYN Flag Count","RST Flag Count","PSH Flag Count","ACK Flag Count","URG Flag Count",
    "CWE Flag Count","ECE Flag Count","Down/Up Ratio","Average Packet Size",
    "Avg Fwd Segment Size","Avg Bwd Segment Size","Fwd Header Length",
    "Fwd Avg Bytes/Bulk","Fwd Avg Packets/Bulk","Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk","Bwd Avg Packets/Bulk","Bwd Avg Bulk Rate",
    "Subflow Fwd Packets","Subflow Fwd Bytes","Subflow Bwd Packets","Subflow Bwd Bytes",
    "Init_Win_bytes_forward","Init_Win_bytes_backward","act_data_pkt_fwd","min_seg_size_forward",
    "Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min",
]

CICIDS2018_COLUMNS = [
    "Dst Port","Protocol","Timestamp","Flow Duration","Tot Fwd Pkts","Tot Bwd Pkts",
    "TotLen Fwd Pkts","TotLen Bwd Pkts","Fwd Pkt Len Max","Fwd Pkt Len Min","Fwd Pkt Len Mean","Fwd Pkt Len Std",
    "Bwd Pkt Len Max","Bwd Pkt Len Min","Bwd Pkt Len Mean","Bwd Pkt Len Std",
    "Flow Byts/s","Flow Pkts/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min",
    "Fwd IAT Tot","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min",
    "Bwd IAT Tot","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min",
    "Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags",
    "Fwd Header Len","Bwd Header Len","Fwd Pkts/s","Bwd Pkts/s",
    "Pkt Len Min","Pkt Len Max","Pkt Len Mean","Pkt Len Std","Pkt Len Var",
    "FIN Flag Cnt","SYN Flag Cnt","RST Flag Cnt","PSH Flag Cnt","ACK Flag Cnt","URG Flag Cnt",
    "CWE Flag Count","ECE Flag Cnt","Down/Up Ratio","Pkt Size Avg",
    "Fwd Seg Size Avg","Bwd Seg Size Avg",
    "Fwd Byts/b Avg","Fwd Pkts/b Avg","Fwd Blk Rate Avg",
    "Bwd Byts/b Avg","Bwd Pkts/b Avg","Bwd Blk Rate Avg",
    "Subflow Fwd Pkts","Subflow Fwd Byts","Subflow Bwd Pkts","Subflow Bwd Byts",
    "Init Fwd Win Byts","Init Bwd Win Byts","Fwd Act Data Pkts","Fwd Seg Size Min",
    "Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min",
]

MAP_2018_TO_2017 = {
    "Dst Port":"Destination Port", "Tot Fwd Pkts":"Total Fwd Packets",
    "Tot Bwd Pkts":"Total Backward Packets", "TotLen Fwd Pkts":"Total Length of Fwd Packets",
    "TotLen Bwd Pkts":"Total Length of Bwd Packets",
    "Fwd Pkt Len Max":"Fwd Packet Length Max", "Fwd Pkt Len Min":"Fwd Packet Length Min",
    "Fwd Pkt Len Mean":"Fwd Packet Length Mean", "Fwd Pkt Len Std":"Fwd Packet Length Std",
    "Bwd Pkt Len Max":"Bwd Packet Length Max", "Bwd Pkt Len Min":"Bwd Packet Length Min",
    "Bwd Pkt Len Mean":"Bwd Packet Length Mean", "Bwd Pkt Len Std":"Bwd Packet Length Std",
    "Flow Byts/s":"Flow Bytes/s", "Flow Pkts/s":"Flow Packets/s",
    "Fwd IAT Tot":"Fwd IAT Total", "Bwd IAT Tot":"Bwd IAT Total",
    "Fwd Header Len":"Fwd Header Length", "Bwd Header Len":"Bwd Header Length",
    "Fwd Pkts/s":"Fwd Packets/s", "Bwd Pkts/s":"Bwd Packets/s",
    "Pkt Len Min":"Min Packet Length", "Pkt Len Max":"Max Packet Length",
    "Pkt Len Mean":"Packet Length Mean", "Pkt Len Std":"Packet Length Std",
    "Pkt Len Var":"Packet Length Variance",
    "FIN Flag Cnt":"FIN Flag Count", "SYN Flag Cnt":"SYN Flag Count",
    "RST Flag Cnt":"RST Flag Count", "PSH Flag Cnt":"PSH Flag Count",
    "ACK Flag Cnt":"ACK Flag Count", "URG Flag Cnt":"URG Flag Count",
    "ECE Flag Cnt":"ECE Flag Count", "Pkt Size Avg":"Average Packet Size",
    "Fwd Seg Size Avg":"Avg Fwd Segment Size", "Bwd Seg Size Avg":"Avg Bwd Segment Size",
    "Fwd Byts/b Avg":"Fwd Avg Bytes/Bulk", "Fwd Pkts/b Avg":"Fwd Avg Packets/Bulk",
    "Fwd Blk Rate Avg":"Fwd Avg Bulk Rate", "Bwd Byts/b Avg":"Bwd Avg Bytes/Bulk",
    "Bwd Pkts/b Avg":"Bwd Avg Packets/Bulk", "Bwd Blk Rate Avg":"Bwd Avg Bulk Rate",
    "Subflow Fwd Pkts":"Subflow Fwd Packets", "Subflow Fwd Byts":"Subflow Fwd Bytes",
    "Subflow Bwd Pkts":"Subflow Bwd Packets", "Subflow Bwd Byts":"Subflow Bwd Bytes",
    "Init Fwd Win Byts":"Init_Win_bytes_forward", "Init Bwd Win Byts":"Init_Win_bytes_backward",
    "Fwd Act Data Pkts":"act_data_pkt_fwd", "Fwd Seg Size Min":"min_seg_size_forward",
}

LABEL_COLUMN: str = "Label"
TIMESTAMP_COLUMN: str = "Timestamp"
PROTOCOL_COLUMN: str = "Protocol"
META_2018: List[str] = [TIMESTAMP_COLUMN, PROTOCOL_COLUMN]


def dedupe_keep_first(columns: List[str]) -> List[str]:
    """Return ``columns`` with duplicates removed, keeping first occurrences.

    Args:
        columns: Column names, possibly with repeats.

    Returns:
        Order-preserving de-duplicated list.
    """
    seen: set = set()
    out: List[str] = []
    for c in columns:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


#: Canonical (pre constant-drop) feature order - de-duplicated 2017 schema.
CICIDS2017_FEATURES: List[str] = dedupe_keep_first(CICIDS2017_COLUMNS)

#: 2018 feature columns (raw names), i.e. the 2018 schema minus metadata.
CICIDS2018_FEATURES_RAW: List[str] = [c for c in CICIDS2018_COLUMNS if c not in META_2018]


def rename_2018(name: str) -> str:
    """Map a 2018 column name to its 2017 canonical name (identity if shared).

    Args:
        name: A CICIDS2018 column name.

    Returns:
        The CICIDS2017 canonical name.
    """
    return MAP_2018_TO_2017.get(name, name)


def normalize_column_name(name: str) -> str:
    """Strip a UTF-8 BOM and surrounding whitespace from a CSV header.

    Args:
        name: Raw header string.

    Returns:
        Cleaned header string.
    """
    return str(name).replace("﻿", "").strip()


def validate_schema() -> Dict[str, int]:
    """Check that the renamed 2018 features equal the 2017 canonical features.

    Raises:
        ValueError: If the two schemas do not align one-to-one.

    Returns:
        Small dict of counts for logging.
    """
    renamed = [rename_2018(c) for c in CICIDS2018_FEATURES_RAW]
    a, b = set(renamed), set(CICIDS2017_FEATURES)
    if a != b or len(renamed) != len(set(renamed)):
        raise ValueError(
            "CICIDS2018->2017 mapping mismatch. Only in 2018: "
            f"{sorted(a - b)}; only in 2017: {sorted(b - a)}"
        )
    return {
        "n_2017_raw": len(CICIDS2017_COLUMNS),
        "n_2017_dedup": len(CICIDS2017_FEATURES),
        "n_2018_features": len(CICIDS2018_FEATURES_RAW),
    }


def canonical_attack_type(label: str) -> str:
    """Normalise a raw CICIDS label into a canonical attack-type string.

    Merges ``Web Attack - Brute Force / XSS / Sql Injection`` (2017, any dash
    encoding) and ``Brute Force -Web / -XSS / SQL Injection`` (2018) into
    ``WebAttack``; fixes ``Infilteration`` -> ``Infiltration``; ``BENIGN`` and
    ``Benign`` -> ``BENIGN``.

    Args:
        label: Raw label string.

    Returns:
        Canonical attack-type string.
    """
    s = str(label).strip()
    low = s.lower()
    if low == "benign":
        return "BENIGN"
    if low.startswith("web attack") or low in {
        "brute force -web", "brute force -xss", "sql injection"
    }:
        return "WebAttack"
    if low.startswith("infilt"):
        return "Infiltration"
    return s
'''

# ==========================================================================
# src/preprocess.py
# ==========================================================================
FILES['src/preprocess.py'] = r'''"""Step 1 - preprocessing for CICIDS2017 + CICIDS2018 (low-memory, disk-backed).

Pipeline (run with ``python -m src.preprocess --force``):

1. Recursively glob ``*.csv`` under ``data/cicids2017`` and ``data/cicids2018``.
2. Stream each file in chunks of ``LOAD_CHUNK_ROWS`` rows
   (``pd.read_csv(low_memory=False, chunksize=...)``), strip BOM/whitespace
   from headers, keep the canonical columns, rename 2018 -> 2017 names. 2018
   ``Timestamp`` / ``Protocol`` are kept *aside* as metadata, never as features.
3. Clean: +/-inf -> NaN, coerce to float32, drop NaN rows (and rows with a
   missing label / repeated header rows), drop exact duplicate rows
   (64-bit row hash of features + label + source, first occurrence kept), drop
   constant columns (names -> ``artifacts/dropped_columns.json``).
4. Labels: ``is_attack`` (binary) and canonical ``attack_type``.
5. Chronological 80/20 split per ``(source, day, attack_type)`` group. 2018 uses
   the parsed ``Timestamp``; 2017 has no timestamp column in the canonical
   schema, so **file order (row order inside each CSV) is the chronological
   proxy** and each CSV is treated as its own "day". Inside the 80 % train
   part the last 10 % of every group is the inner-validation segment.
6. ``StandardScaler`` fitted on the train split only; saved to ``scaler.joblib``.
7. Windowing (``WINDOW_SIZE`` flows, ``STRIDE``) inside each
   ``(source, day, Protocol | file index)`` group and inside each segment
   (inner-train / inner-val / test) so no window straddles a split boundary.
   Groups shorter than the window are left-padded with their first flow, and a
   final tail window is added when the stride does not land on the last flow,
   so every flow belongs to at least one window. Window label = majority
   ``is_attack`` (ties -> attack).
8. Inverse-frequency class weights -> ``class_weights.npy``.

Memory design (fits the full CICIDS2017 + CICIDS2018 in ~16 GB RAM): the
float32 flow matrix is never held in RAM. Cleaned rows are streamed to a
temporary on-disk matrix, metadata is kept as compact integer arrays, and all
later passes (scaling, windowing, flow-level export) read/write in chunks via
memory maps. Temporary files live in ``artifacts/_tmp`` and are deleted at the
end. Peak disk use is roughly 2.5x the final artifact size.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from sklearn.preprocessing import LabelEncoder, StandardScaler

from . import config as C
from . import schema as S
from .utils import get_logger, save_json, set_seed

log = get_logger()
_F32_MAX = float(np.finfo(np.float32).max)
SOURCES: List[str] = ["2017", "2018"]


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_files(source: str) -> List[Path]:
    """Recursively list CSV files of one dataset in deterministic order.

    Args:
        source: ``"2017"`` or ``"2018"``.

    Returns:
        Sorted list of CSV paths (may be empty).
    """
    root = C.DATA_DIRS[source]
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".csv")


def _wanted_columns(source: str) -> List[str]:
    if source == "2017":
        return S.CICIDS2017_FEATURES + [S.LABEL_COLUMN]
    return S.CICIDS2018_FEATURES_RAW + S.META_2018 + [S.LABEL_COLUMN]


def _to_float32_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    """Coerce columns to a float32 matrix; non-numeric / inf / overflow -> NaN."""
    out = np.empty((len(df), len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        bad = ~np.isfinite(v) | (np.abs(v) > _F32_MAX)
        v[bad] = np.nan
        out[:, j] = v.astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
# Streaming loader
# --------------------------------------------------------------------------- #
class _Vocab:
    """String -> int code dictionary (first-seen order)."""

    def __init__(self) -> None:
        self.index: Dict[str, int] = {}

    def encode(self, values: np.ndarray) -> np.ndarray:
        uniq, inv = np.unique(values.astype(str), return_inverse=True)
        codes = np.array([self.index.setdefault(u, len(self.index)) for u in uniq], dtype=np.int32)
        return codes[inv]

    @property
    def strings(self) -> List[str]:
        return list(self.index)


def iter_file_chunks(path: Path, source: str, max_rows: Optional[int]) -> Iterator[pd.DataFrame]:
    """Yield header-normalised, column-filtered DataFrame chunks of one CSV.

    Undecodable bytes (e.g. the cp1252 dash in CICIDS2017 web-attack labels)
    are replaced rather than aborting the read.

    Args:
        path: CSV path.
        source: ``"2017"`` or ``"2018"``.
        max_rows: Optional row cap.

    Yields:
        DataFrame chunks with canonical (2017) column names.

    Raises:
        ValueError: If a required column is missing.
    """
    wanted = set(_wanted_columns(source))
    reader = pd.read_csv(
        path, low_memory=False, nrows=max_rows, chunksize=C.LOAD_CHUNK_ROWS, encoding_errors="replace",
        usecols=lambda c: S.normalize_column_name(c) in wanted,
    )
    first = True
    for df in reader:
        df.columns = [S.normalize_column_name(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        if first:
            missing = sorted(wanted - set(df.columns))
            if missing:
                raise ValueError(
                    f"[{source}] {path} is missing required column(s): {missing}. Check that the file is an "
                    "original CICIDS CSV (MachineLearningCVE / CSE-CIC-IDS2018 Processed Traffic Data)."
                )
            first = False
        if source == "2018":
            df = df.rename(columns=S.MAP_2018_TO_2017)
        yield df


class LoadResult:
    """Container for the streamed, cleaned (pre-dedupe) dataset."""

    def __init__(self) -> None:
        self.raw_path: Path = Path()
        self.n: int = 0
        self.meta: Dict[str, np.ndarray] = {}
        self.label_strings: List[str] = []
        self.day_strings: List[str] = []
        self.files: List[Dict[str, object]] = []
        self.rows_raw: Dict[str, int] = {s: 0 for s in SOURCES}
        self.rows_dropped_nan: int = 0


def load_stream(max_rows: Optional[int], tmp_dir: Path) -> LoadResult:
    """Stream every CSV, clean rows, and write the float32 matrix to disk.

    Args:
        max_rows: Optional per-file row cap.
        tmp_dir: Folder for the temporary raw matrix.

    Returns:
        :class:`LoadResult` with compact per-row metadata arrays.

    Raises:
        SystemExit: If both data folders are empty.
    """
    S.validate_schema()
    files: List[Tuple[str, Path]] = []
    for src in SOURCES:
        found = discover_files(src)
        if not found:
            log.warning(f"No CSV files found under {C.DATA_DIRS[src]} - CICIDS{src} skipped.")
        files += [(src, p) for p in found]
    if not files:
        log.error("No CSV files found. Drop the raw CICIDS2017 CSVs into "
                  f"{C.DATA_DIRS['2017']} and/or the CICIDS2018 CSVs into {C.DATA_DIRS['2018']} "
                  "(sub-folders are fine), then rerun `python -m src.preprocess --force`.")
        raise SystemExit(2)

    res = LoadResult()
    res.raw_path = tmp_dir / "raw_flows.f32"
    labels, days = _Vocab(), _Vocab()
    parts: Dict[str, List[np.ndarray]] = {k: [] for k in
                                          ("source", "file_idx", "seq", "day", "ts_ns", "win_key", "label", "hash")}
    feats = S.CICIDS2017_FEATURES
    with open(res.raw_path, "wb") as fout:
        for file_idx, (src, path) in enumerate(files):
            seq0, last_ts, n_file = 0, None, 0
            for df in iter_file_chunks(path, src, max_rows):
                n = len(df)
                seq = np.arange(seq0, seq0 + n, dtype=np.int64)
                seq0 += n
                res.rows_raw[src] += n
                X = _to_float32_matrix(df, feats)
                lab_raw = df[S.LABEL_COLUMN]
                lab = lab_raw.astype(str).str.strip()
                valid = (~np.isnan(X).any(axis=1) & lab_raw.notna().to_numpy()
                         & (lab.str.len() > 0).to_numpy() & (lab.str.lower() != "label").to_numpy())
                if src == "2018":
                    ts = pd.to_datetime(df[S.TIMESTAMP_COLUMN], format="%d/%m/%Y %H:%M:%S", errors="coerce")
                    if ts.isna().mean() > 0.5:
                        ts = pd.to_datetime(df[S.TIMESTAMP_COLUMN], errors="coerce", dayfirst=True)
                    if last_ts is not None and pd.isna(ts.iloc[0]):
                        ts.iloc[0] = last_ts
                    ts = ts.ffill().bfill()
                    if ts.notna().any():
                        last_ts = ts.iloc[-1]
                        day_str = ts.dt.strftime("%Y-%m-%d").fillna(f"2018-f{file_idx:03d}").to_numpy()
                        ts_ns = np.where(ts.notna(), ts.astype("int64"), seq).astype(np.int64)
                    else:
                        day_str = np.full(n, f"2018-f{file_idx:03d}")
                        ts_ns = seq
                    win_key = pd.to_numeric(df[S.PROTOCOL_COLUMN], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
                else:
                    # 2017: no timestamp in the canonical schema -> file order is the chronological proxy.
                    day_str = np.full(n, f"2017-f{file_idx:03d}")
                    ts_ns = seq
                    win_key = np.full(n, file_idx, dtype=np.int64)
                res.rows_dropped_nan += int((~valid).sum())
                Xv = np.ascontiguousarray(X[valid])
                lab_codes = labels.encode(lab.to_numpy()[valid])
                src_code = SOURCES.index(src)
                hdf = pd.DataFrame(Xv, copy=False)
                hdf["_label"] = lab_codes
                hdf["_source"] = src_code
                h = pd.util.hash_pandas_object(hdf, index=False).to_numpy()
                del hdf
                fout.write(Xv.tobytes())
                nv = len(Xv)
                n_file += nv
                parts["source"].append(np.full(nv, src_code, dtype=np.int8))
                parts["file_idx"].append(np.full(nv, file_idx, dtype=np.int32))
                parts["seq"].append(seq[valid])
                parts["day"].append(days.encode(np.asarray(day_str)[valid]))
                parts["ts_ns"].append(np.asarray(ts_ns)[valid])
                parts["win_key"].append(np.asarray(win_key)[valid])
                parts["label"].append(lab_codes)
                parts["hash"].append(h)
            log.info(f"[{src}] streamed {path.name}: {seq0:,} rows ({n_file:,} valid)")
            res.files.append({"file_idx": file_idx, "source": src, "path": str(
                path.relative_to(C.PROJECT_ROOT) if path.is_relative_to(C.PROJECT_ROOT) else path)})
    res.meta = {k: np.concatenate(v) for k, v in parts.items()}
    res.n = len(res.meta["label"])
    res.label_strings = labels.strings
    # Re-code days so that integer order == lexicographic (chronological) order.
    ds = days.strings
    rank = np.argsort(np.argsort(np.array(ds, dtype=object)))
    res.meta["day"] = rank[res.meta["day"]].astype(np.int32)
    res.day_strings = sorted(ds)
    return res


def raw_matrix(res: LoadResult) -> np.ndarray:
    """Open the temporary raw float32 matrix as a read-only memmap."""
    return np.memmap(res.raw_path, dtype=np.float32, mode="r", shape=(res.n, len(S.CICIDS2017_FEATURES)))


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #
def constant_columns_chunked(X: np.ndarray, rows: np.ndarray, names: List[str]) -> List[str]:
    """Names of columns whose min equals max over ``rows`` (chunked scan).

    Args:
        X: Matrix / memmap.
        rows: Row ids to consider (sorted).
        names: Column names.

    Returns:
        Constant column names.
    """
    if len(rows) == 0:
        return []
    lo = np.full(X.shape[1], np.inf, dtype=np.float64)
    hi = np.full(X.shape[1], -np.inf, dtype=np.float64)
    for s in range(0, len(rows), C.LOAD_CHUNK_ROWS):
        blk = X[rows[s:s + C.LOAD_CHUNK_ROWS]]
        lo = np.minimum(lo, blk.min(axis=0))
        hi = np.maximum(hi, blk.max(axis=0))
    return [n for n, a, b in zip(names, lo, hi) if a == b]


def constant_columns(X: np.ndarray, names: List[str]) -> List[str]:
    """Names of constant columns of an in-memory matrix."""
    return constant_columns_chunked(X, np.arange(len(X)), names)


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def chronological_split(meta: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological 80/20 split per (source, day, attack_type) + inner-val tail.

    Args:
        meta: Metadata with ``source, day, ts_ns, file_idx, seq, attack_type``
            (any sortable dtypes; ``day`` must sort chronologically).

    Returns:
        ``(order, is_train, is_inner_val)`` - ``order`` is the chronological
        permutation of rows; the two masks are aligned with the *original* rows.
    """
    order = np.lexsort((
        meta["seq"].to_numpy(), meta["file_idx"].to_numpy(), meta["ts_ns"].to_numpy(),
        meta["day"].to_numpy(), meta["source"].to_numpy(),
    ))
    m = meta.iloc[order]
    keys = [m["source"], m["day"], m["attack_type"]]
    rank = m.groupby(keys, sort=False).cumcount().to_numpy()
    size = m.groupby(keys, sort=False)["seq"].transform("size").to_numpy()
    del m
    n_train = np.ceil(size * C.TRAIN_FRACTION).astype(np.int64)
    train_sorted = rank < n_train
    n_inner_train = np.ceil(n_train * (1.0 - C.INNER_VAL_FRACTION)).astype(np.int64)
    inner_val_sorted = train_sorted & (rank >= n_inner_train)
    is_train = np.zeros(len(meta), dtype=bool)
    is_inner_val = np.zeros(len(meta), dtype=bool)
    is_train[order] = train_sorted
    is_inner_val[order] = inner_val_sorted
    return order, is_train, is_inner_val


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def window_offsets(n: int, w: int = C.WINDOW_SIZE, s: int = C.STRIDE) -> np.ndarray:
    """Local row offsets of every window in a group of ``n`` chronologically ordered flows.

    Args:
        n: Group length (>0).
        w: Window size.
        s: Stride.

    Returns:
        ``(k, w)`` int64 array of offsets in ``[0, n)``.
    """
    if n >= w:
        starts = np.arange(0, n - w + 1, s, dtype=np.int64)
        if starts[-1] != n - w:
            starts = np.append(starts, n - w)  # tail window: last flow always covered
        return starts[:, None] + np.arange(w, dtype=np.int64)[None, :]
    pad = np.zeros(w - n, dtype=np.int64)  # left-pad with the first flow
    return np.concatenate([pad, np.arange(n, dtype=np.int64)])[None, :]


def build_window_index(rows_sorted: np.ndarray, meta: pd.DataFrame, group_offset: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build window row-index matrix for one segment.

    Args:
        rows_sorted: Row ids of the segment in chronological order.
        meta: Full metadata (``source, day, win_key``).
        group_offset: First group id to assign (keeps ids globally unique).

    Returns:
        ``(index, group_id)`` with shapes ``(N_w, W)`` and ``(N_w,)``.
    """
    if len(rows_sorted) == 0:
        return np.zeros((0, C.WINDOW_SIZE), dtype=np.int64), np.zeros(0, dtype=np.int64)
    sub = meta.iloc[rows_sorted]
    gkey = pd.MultiIndex.from_arrays([sub["source"], sub["day"], sub["win_key"]])
    codes, _ = pd.factorize(gkey, sort=False)
    del sub, gkey
    perm = np.lexsort((np.arange(len(codes)), codes))  # stable: keeps chronology inside group
    rows_g = rows_sorted[perm]
    codes_g = codes[perm]
    bounds = np.flatnonzero(np.diff(codes_g)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(codes_g)]])
    idx_parts, gid_parts = [], []
    for g, (a, b) in enumerate(zip(starts, ends)):
        off = window_offsets(int(b - a))
        idx_parts.append(rows_g[a + off])
        gid_parts.append(np.full(len(off), group_offset + g, dtype=np.int64))
    return np.concatenate(idx_parts), np.concatenate(gid_parts)


def window_labels(index: np.ndarray, y: np.ndarray, type_codes: np.ndarray, benign_code: int,
                  n_types: int) -> Tuple[np.ndarray, np.ndarray]:
    """Majority-vote window labels (ties -> attack) and majority attack type (chunked).

    Args:
        index: ``(N_w, W)`` row ids.
        y: Flow-level ``is_attack``.
        type_codes: Flow-level attack-type codes.
        benign_code: Code of ``BENIGN``.
        n_types: Number of attack-type codes.

    Returns:
        ``(y_window, type_window_codes)``.
    """
    label = np.empty(len(index), dtype=np.int64)
    tcode = np.empty(len(index), dtype=np.int64)
    for s in range(0, len(index), C.WINDOW_WRITE_CHUNK):
        ix = index[s:s + C.WINDOW_WRITE_CHUNK]
        yw = y[ix]
        lab = (2 * yw.sum(axis=1) >= ix.shape[1]).astype(np.int64)
        tw = type_codes[ix]
        counts = np.stack([((tw == k) & (yw == 1)).sum(axis=1) for k in range(n_types)], axis=1)
        counts[:, benign_code] = -1
        tc = counts.argmax(axis=1)
        tc[lab == 0] = benign_code
        label[s:s + len(ix)], tcode[s:s + len(ix)] = lab, tc
    return label, tcode


def write_windows(path: Path, X: np.ndarray, index: np.ndarray) -> None:
    """Materialise windows ``X[index]`` into an ``.npy`` file chunk by chunk.

    Args:
        path: Output ``.npy`` path.
        X: Scaled flow matrix ``(n, F)`` (array or memmap).
        index: ``(N_w, W)`` row ids.
    """
    mm = open_memmap(path, mode="w+", dtype=np.float32, shape=(len(index), index.shape[1], X.shape[1]))
    for s in range(0, len(index), C.WINDOW_WRITE_CHUNK):
        e = min(s + C.WINDOW_WRITE_CHUNK, len(index))
        mm[s:e] = X[index[s:e]]
    mm.flush()
    del mm


def write_rows(path: Path, X: np.ndarray, rows: np.ndarray) -> None:
    """Write ``X[rows]`` to an ``.npy`` file chunk by chunk.

    Args:
        path: Output path.
        X: Source matrix / memmap.
        rows: Row ids.
    """
    mm = open_memmap(path, mode="w+", dtype=np.float32, shape=(len(rows), X.shape[1]))
    for s in range(0, len(rows), C.WINDOW_WRITE_CHUNK):
        e = min(s + C.WINDOW_WRITE_CHUNK, len(rows))
        mm[s:e] = X[rows[s:e]]
    mm.flush()
    del mm


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _class_balance(y: np.ndarray) -> Dict[str, object]:
    n = int(len(y))
    a = int(np.asarray(y).sum())
    return {"n": n, "benign": n - a, "attack": a, "attack_pct": round(100.0 * a / n, 3) if n else 0.0}


def _check_disk(path: Path) -> None:
    free_gb = shutil.disk_usage(path).free / 1024 ** 3
    msg = f"Free disk under {path}: {free_gb:.1f} GB"
    if free_gb < C.MIN_FREE_DISK_GB:
        log.warning(msg + f" - the full CICIDS2017+2018 run needs about {C.MIN_FREE_DISK_GB} GB "
                    "(temporary files included). Free some space if preprocessing fails.")
    else:
        log.info(msg)


def run(force: bool = False, max_rows: Optional[int] = C.MAX_ROWS_PER_FILE) -> Dict[str, object]:
    """Execute the full Step-1 pipeline and write every artifact.

    Args:
        force: Overwrite existing artifacts.
        max_rows: Optional per-file row cap (debug only).

    Returns:
        The summary dict (also saved to ``preprocess_summary.json``).
    """
    set_seed(C.SEED)
    A = C.ARTIFACT_DIR
    A.mkdir(parents=True, exist_ok=True)
    if C.artifact_path("X_train").exists() and not force:
        log.error("Preprocessed artifacts already exist. Use `python -m src.preprocess --force` to rebuild.")
        raise SystemExit(1)
    for key in ("X_train", "X_test", "X_flows_train"):
        C.artifact_path(key).unlink(missing_ok=True)
    if C.EVAL_DIR.exists():
        shutil.rmtree(C.EVAL_DIR)
    tmp = A / "_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    _check_disk(A)

    try:
        summary = _run_inner(max_rows, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return summary


def _run_inner(max_rows: Optional[int], tmp: Path) -> Dict[str, object]:
    res = load_stream(max_rows, tmp)
    Xraw = raw_matrix(res)
    meta = res.meta

    # ---- exact-duplicate removal (64-bit row hash; first occurrence kept) ----
    dup = pd.Series(meta.pop("hash")).duplicated(keep="first").to_numpy()
    kept = np.flatnonzero(~dup)
    meta = {k: v[kept] for k, v in meta.items()}
    n = len(kept)
    stats = {"rows_raw": int(sum(res.rows_raw.values())), "rows_dropped_nan": res.rows_dropped_nan,
             "rows_dropped_duplicate": int(dup.sum()), "rows_clean": int(n)}
    del dup
    log.info(f"Cleaned: {n:,} rows kept ({stats['rows_dropped_duplicate']:,} duplicates, "
             f"{stats['rows_dropped_nan']:,} NaN/inf/invalid rows dropped)")

    # ---- constant columns ----
    feats_all = list(S.CICIDS2017_FEATURES)
    dropped = constant_columns_chunked(Xraw, kept, feats_all)
    keep_cols = np.array([i for i, c in enumerate(feats_all) if c not in dropped], dtype=np.int64)
    feats = [feats_all[i] for i in keep_cols]
    save_json(dropped, C.artifact_path("dropped_columns"))
    save_json(feats, C.artifact_path("feature_columns"))

    # ---- labels ----
    lab_str = np.array(res.label_strings, dtype=object)
    lab_is_attack = np.array([s.strip().lower() != "benign" for s in lab_str], dtype=np.int64)
    lab_type = np.array([S.canonical_attack_type(s) for s in lab_str], dtype=object)
    types_present = np.unique(np.append(lab_type[np.unique(meta["label"])].astype(str), "BENIGN"))
    le_type = LabelEncoder().fit(types_present)
    lab_type_code = le_type.transform(lab_type.astype(str)).astype(np.int64)
    y = lab_is_attack[meta["label"]]
    type_codes = lab_type_code[meta["label"]]
    benign_code = int(le_type.transform(["BENIGN"])[0])
    joblib.dump({"attack_type": le_type, "binary": {"BENIGN": 0, "ATTACK": 1}}, C.artifact_path("label_encoders"))

    # ---- chronological split ----
    mdf = pd.DataFrame({
        "source": meta["source"], "day": meta["day"], "ts_ns": meta["ts_ns"], "file_idx": meta["file_idx"],
        "seq": meta["seq"], "attack_type": type_codes.astype(np.int16), "win_key": meta["win_key"],
    })
    order, is_train, is_inner_val = chronological_split(mdf)
    is_test = ~is_train
    np.savez_compressed(
        C.artifact_path("split_indices"),
        train_idx=np.flatnonzero(is_train), test_idx=np.flatnonzero(is_test),
        inner_val_idx=np.flatnonzero(is_inner_val),
        row_file_idx=meta["file_idx"], row_seq_in_file=meta["seq"],
    )

    # ---- scaler (train rows only), then a scaled on-disk flow matrix ----
    scaler = StandardScaler()
    train_rows = np.flatnonzero(is_train)
    for s in range(0, len(train_rows), C.LOAD_CHUNK_ROWS):
        blk = Xraw[kept[train_rows[s:s + C.LOAD_CHUNK_ROWS]]][:, keep_cols]
        scaler.partial_fit(blk.astype(np.float64))
    scaler.feature_names_ordered_ = np.array(feats)
    joblib.dump(scaler, C.artifact_path("scaler"))
    Xs = open_memmap(tmp / "flows_scaled.npy", mode="w+", dtype=np.float32, shape=(n, len(feats)))
    for s in range(0, n, C.LOAD_CHUNK_ROWS):
        blk = Xraw[kept[s:s + C.LOAD_CHUNK_ROWS]][:, keep_cols]
        Xs[s:s + len(blk)] = scaler.transform(blk.astype(np.float64)).astype(np.float32)
    Xs.flush()
    del Xraw
    res.raw_path.unlink(missing_ok=True)  # free disk early
    log.info("Scaled flow matrix written; building windows ...")

    # ---- windows ----
    seg_inner_train = order[(is_train & ~is_inner_val)[order]]
    seg_inner_val = order[is_inner_val[order]]
    seg_test = order[is_test[order]]
    idx_it, g_it = build_window_index(seg_inner_train, mdf, 0)
    idx_iv, g_iv = build_window_index(seg_inner_val, mdf, int(g_it.max(initial=-1)) + 1)
    idx_te, g_te = build_window_index(seg_test, mdf, 0)
    del seg_inner_train, seg_inner_val, seg_test, mdf
    idx_tr = np.concatenate([idx_it, idx_iv])
    g_tr = np.concatenate([g_it, g_iv])
    inner_val_mask = np.concatenate([np.zeros(len(idx_it), bool), np.ones(len(idx_iv), bool)])
    del idx_it, idx_iv, g_it, g_iv

    n_types = len(le_type.classes_)
    y_tr, t_tr = window_labels(idx_tr, y, type_codes, benign_code, n_types)
    y_te, t_te = window_labels(idx_te, y, type_codes, benign_code, n_types)

    write_windows(C.artifact_path("X_train"), Xs, idx_tr)
    log.info(f"X_train.npy written: {len(idx_tr):,} windows")
    write_windows(C.artifact_path("X_test"), Xs, idx_te)
    log.info(f"X_test.npy written: {len(idx_te):,} windows")
    np.save(C.artifact_path("y_train"), y_tr)
    np.save(C.artifact_path("y_test"), y_te)
    np.save(C.artifact_path("train_attack_type"), le_type.classes_[t_tr].astype(str))
    np.save(C.artifact_path("test_attack_type"), le_type.classes_[t_te].astype(str))
    np.save(C.artifact_path("train_inner_val"), inner_val_mask)
    np.save(C.artifact_path("train_group"), g_tr)
    np.save(C.artifact_path("test_group"), g_te)
    src_str = np.array(SOURCES)
    np.save(C.artifact_path("test_source"), src_str[meta["source"][idx_te[:, -1]]])

    # ---- flow-level train data (RF is fit on the full 80 % at flow level) ----
    flows_tr = order[is_train[order]]
    write_rows(C.artifact_path("X_flows_train"), Xs, flows_tr)
    np.save(C.artifact_path("y_flows_train"), y[flows_tr])
    np.save(C.artifact_path("flows_train_inner_val"), is_inner_val[flows_tr])
    np.save(C.artifact_path("flows_train_source"), src_str[meta["source"][flows_tr]])
    del Xs

    # ---- class weights ----
    counts = np.bincount(y_tr, minlength=2).astype(np.float64)
    if (counts == 0).any():
        log.warning(f"A class is absent from the training windows (counts={counts.tolist()}); weight set to 1.0.")
    cw = np.where(counts > 0, counts.sum() / (2.0 * np.maximum(counts, 1)), 1.0).astype(np.float32)
    np.save(C.artifact_path("class_weights"), cw)

    clean_per_src = np.bincount(meta["source"].astype(np.int64), minlength=2)
    summary: Dict[str, object] = {
        "files": res.files,
        "rows_per_source_raw": {s: int(v) for s, v in res.rows_raw.items() if v},
        "rows_per_source_clean": {s: int(clean_per_src[i]) for i, s in enumerate(SOURCES) if clean_per_src[i]},
        **stats,
        "n_features": len(feats),
        "dropped_constant_columns": dropped,
        "flow_split": {
            "train": _class_balance(y[is_train]), "test": _class_balance(y[is_test]),
            "inner_val (subset of train)": _class_balance(y[is_inner_val]),
        },
        "window_counts": {
            "train": int(len(idx_tr)), "train_inner_val": int(inner_val_mask.sum()), "test": int(len(idx_te)),
        },
        "window_split": {"train": _class_balance(y_tr), "test": _class_balance(y_te)},
        "attack_types_test_windows": {str(k): int(v) for k, v in
                                      zip(*np.unique(le_type.classes_[t_te], return_counts=True))},
        "class_weights": cw.tolist(),
        "window_size": C.WINDOW_SIZE, "stride": C.STRIDE,
        "chronology_note": "2018: Timestamp; 2017: file/row order used as chronological proxy (one CSV = one day).",
        "max_rows_per_file": max_rows,
    }
    save_json(summary, C.artifact_path("preprocess_summary"))
    _print_summary(summary)
    return summary


def _print_summary(s: Dict[str, object]) -> None:
    log.info("=" * 72)
    log.info("PREPROCESS SUMMARY")
    log.info(f"Rows per source (raw):   {s['rows_per_source_raw']}")
    log.info(f"Rows per source (clean): {s['rows_per_source_clean']}")
    log.info(f"Dropped NaN/inf rows: {s['rows_dropped_nan']:,} | duplicate rows: {s['rows_dropped_duplicate']:,}")
    log.info(f"Feature count: {s['n_features']} | dropped constant columns: {s['dropped_constant_columns']}")
    for k, v in s["flow_split"].items():  # type: ignore[union-attr]
        log.info(f"Flow split {k:<26}: {v}")
    log.info(f"Window counts: {s['window_counts']}")
    for k, v in s["window_split"].items():  # type: ignore[union-attr]
        log.info(f"Window split {k:<6}: {v}")
    log.info(f"Class weights: {s['class_weights']}")
    log.info("=" * 72)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point: ``python -m src.preprocess [--force] [--max-rows-per-file N]``."""
    p = argparse.ArgumentParser(description="Step 1: preprocess CICIDS2017 + CICIDS2018.")
    p.add_argument("--force", action="store_true", help="Rebuild all preprocessing artifacts.")
    p.add_argument("--max-rows-per-file", type=int, default=C.MAX_ROWS_PER_FILE,
                   help="Debug-only row cap per CSV (default: None = all rows).")
    a = p.parse_args(argv)
    if a.max_rows_per_file is not None:
        log.warning(f"--max-rows-per-file={a.max_rows_per_file}: NOT the full dataset (debug mode).")
    run(force=a.force, max_rows=a.max_rows_per_file)


if __name__ == "__main__":
    main(sys.argv[1:])
'''

# ==========================================================================
# src/train.py
# ==========================================================================
FILES['src/train.py'] = r'''"""Step 2 - staged training on the 80 % split.

Stages (``python -m src.train --algo double_dqn --epochs 30 --device cuda``):

1. **TCN** on the inner-train windows (90 % of the 80 %), early stopping on the
   inner-val windows (last 10 % of every chronological group) ->
   ``tcn_best.pt``.
2. **Calibrated RF** on *every* training flow of the 80 % ->
   ``rf_calibrated.joblib``. With ``RF_SELECTION_MODE="holdout"`` an extra RF
   is fit on the inner-train flows only, so the weight search in stage 3 is not
   biased toward an RF that has already seen inner-val flows.
3. **Ensemble weight** grid search on inner-val F1 (tie-break ROC-AUC) ->
   ``ensemble_weights.json``.
4. **DRL policy** (Double-DQN or PPO) on TCN embeddings of *all* 80 % training
   windows -> ``drl_agent.pt``.
5. Logs -> ``train_log.csv``, ``drl_log.csv``, ``drl_steps.csv``, TensorBoard
   events under ``artifacts/tensorboard`` and ``train_manifest.json``.

Hard guarantees (spec §5): no silent subsampling (``--subsample-fraction`` is
opt-in and logged), ``drop_last=False``, an assertion that every batch of the
training set is iterated each epoch, early stopping can only fire after
``MIN_EPOCHS`` and is logged with the epoch and val loss, best checkpoint
restored. The 20 % test split is never opened here.
"""
from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .dataset import WindowDataset, expected_batches, make_loader, subsample_indices
from .drl_agent import PolicyEnv, make_agent, save_agent
from .ensemble import grid_search_weights
from .inference_api import write_decision_schema
from .rf_branch import fit_rf, last_flow, predict_attack_proba, save_rf
from .tcn import TCNClassifier, infer, load_checkpoint, save_checkpoint
from .utils import (OOM_HELP, autocast, binary_metrics, get_logger, grad_scaler, load_json,
                    log_device, resolve_device, save_json, set_seed)

log = get_logger()


class CSVLogger:
    """Append dict rows to a CSV file, extending the header as new keys appear."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.rows: List[Dict[str, Any]] = []

    def log(self, row: Dict[str, Any]) -> None:
        """Record one row and rewrite the file."""
        self.rows.append(row)
        write_csv(self.rows, self.path)


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write a list of dict rows to ``path`` in one go.

    Args:
        rows: Rows (union of keys becomes the header).
        path: Destination CSV.
    """
    keys: List[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _tb_writer(enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(C.TB_DIR))
    except Exception as e:  # tensorboard not installed
        log.warning(f"TensorBoard disabled ({e}). `pip install tensorboard` to enable event files.")
        return None


class EarlyStopping:
    """Early stopping on a monitored value (mode='min') with best-state restore.

    Args:
        patience: Epochs without improvement before stopping.
        min_epochs: Stopping is not allowed before this epoch.
        min_delta: Minimum improvement.
    """

    def __init__(self, patience: int = C.PATIENCE, min_epochs: int = C.MIN_EPOCHS, min_delta: float = 1e-5) -> None:
        self.patience, self.min_epochs, self.min_delta = patience, min_epochs, min_delta
        self.best = math.inf
        self.best_epoch = 0
        self.wait = 0
        self.best_state: Optional[Dict[str, torch.Tensor]] = None

    def step(self, value: float, epoch: int, model: nn.Module) -> bool:
        """Update with this epoch's value; return True when training should stop."""
        if value < self.best - self.min_delta:
            self.best, self.best_epoch, self.wait = value, epoch, 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.wait += 1
        return self.wait >= self.patience and epoch >= self.min_epochs

    def restore(self, model: nn.Module) -> None:
        """Load the best state back into ``model``."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


@torch.no_grad()
def _validate(model: TCNClassifier, loader, criterion: nn.Module, device: str, amp: bool) -> Dict[str, float]:
    model.eval()
    tot, n, probs, ys = 0.0, 0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        with autocast(device, amp):
            logits = model(xb)
        loss = criterion(logits.float(), yb)
        tot += float(loss) * len(yb)
        n += len(yb)
        probs.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
        ys.append(yb.cpu().numpy())
    p, y = np.concatenate(probs), np.concatenate(ys)
    m = binary_metrics(y, (p >= C.DECISION_THRESHOLD).astype(int), p)
    return {"val_loss": tot / max(n, 1), "val_f1": m["f1"], "val_auc": m["roc_auc"],
            "val_precision": m["precision"], "val_recall": m["recall"]}


def train_tcn(args: argparse.Namespace, device: str, y: np.ndarray, it_idx: np.ndarray, iv_idx: np.ndarray,
              cw: np.ndarray, csvlog: CSVLogger, tb) -> TCNClassifier:
    """Stage 1: train the TCN with AdamW + cosine + plateau + early stopping.

    Args:
        args: CLI args.
        device: Device string.
        y: Window labels for ``X_train.npy``.
        it_idx: Inner-train window indices.
        iv_idx: Inner-val window indices.
        cw: Class weights.
        csvlog: Train-log writer.
        tb: TensorBoard writer or ``None``.

    Returns:
        The model with the best (lowest val loss) weights restored.
    """
    xp = C.artifact_path("X_train")
    ds_tr = WindowDataset(xp, y, it_idx)
    ds_va = WindowDataset(xp, y, iv_idx)
    bs = args.batch_size
    dl_tr = make_loader(ds_tr, bs, True, device, args.num_workers)
    dl_va = make_loader(ds_va, C.EVAL_BATCH_SIZE, False, device, args.num_workers)
    exp = expected_batches(len(ds_tr), bs)
    if len(dl_tr) != exp["total"]:
        raise RuntimeError(f"DataLoader reports {len(dl_tr)} batches, expected {exp['total']} - refusing to train.")
    log.info(f"[TCN] inner-train windows={len(ds_tr):,} inner-val windows={len(ds_va):,} batch={bs} -> "
             f"{exp['full']} full batches (floor(N/bs)) + {exp['partial']} partial = {exp['total']} batches/epoch, "
             f"drop_last=False")

    model = TCNClassifier(ds_tr.n_features).to(device)
    log.info(f"[TCN] parameters: {sum(p.numel() for p in model.parameters()):,}")
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(cw, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=C.WEIGHT_DECAY)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=C.PLATEAU_FACTOR,
                                                         patience=C.PLATEAU_PATIENCE)
    amp = C.USE_AMP and not args.no_amp and device.startswith("cuda")
    scaler = grad_scaler(device, amp)
    es = EarlyStopping(patience=C.PATIENCE, min_epochs=min(C.MIN_EPOCHS, args.epochs))
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        n_batches, n_seen, tot_loss = 0, 0, 0.0
        for xb, yb in dl_tr:
            n_seen += len(yb)
            if len(yb) == 1:  # BatchNorm needs >1 sample; duplicate the lone sample
                xb, yb = xb.repeat(2, 1, 1), yb.repeat(2)
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with autocast(device, amp):
                logits = model(xb)
            loss = criterion(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM)
            scaler.step(opt)
            scaler.update()
            tot_loss += float(loss.detach()) * len(yb)
            n_batches += 1
            if epoch == 1 and n_batches == 1:
                log_device(device)
        if n_batches != exp["total"] or n_seen != len(ds_tr):
            raise RuntimeError(f"Epoch {epoch}: iterated {n_batches}/{exp['total']} batches and "
                               f"{n_seen}/{len(ds_tr)} samples - the full training set was NOT covered.")
        val = _validate(model, dl_va, criterion, device, amp)
        cosine.step()
        plateau.step(val["val_loss"])
        lr = opt.param_groups[0]["lr"]
        row = {"stage": "tcn", "epoch": epoch, "train_loss": tot_loss / max(n_seen, 1), **val, "lr": lr,
               "batches": n_batches, "samples": n_seen, "epoch_sec": round(time.time() - t0, 2)}
        csvlog.log(row)
        if tb is not None:
            for k in ("train_loss", "val_loss", "val_f1", "val_auc", "lr"):
                tb.add_scalar(f"tcn/{k}", row[k], epoch)
        log.info(f"[TCN] epoch {epoch}/{args.epochs} train_loss={row['train_loss']:.4f} "
                 f"val_loss={val['val_loss']:.4f} val_f1={val['val_f1']:.4f} val_auc={val['val_auc']:.4f} lr={lr:.2e}")
        if es.step(val["val_loss"], epoch, model):
            log.info(f"EarlyStopping triggered at epoch {epoch}/{args.epochs}, val_loss={val['val_loss']:.6f} "
                     f"(best epoch {es.best_epoch}, best val_loss={es.best:.6f})")
            stopped_early = True
            break
    if not stopped_early:
        log.info(f"[TCN] completed full schedule ({args.epochs} epochs); best epoch {es.best_epoch} "
                 f"val_loss={es.best:.6f}")
    es.restore(model)
    save_checkpoint(model, C.artifact_path("tcn"), {
        "best_epoch": es.best_epoch, "best_val_loss": es.best, "stopped_early": stopped_early,
        "epochs_max": args.epochs, "feature_columns": load_json(C.artifact_path("feature_columns")),
    })
    log.info(f"[TCN] saved best checkpoint -> {C.artifact_path('tcn')}")
    return model.eval()


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Step 2: train TCN + calibrated RF + ensemble + DRL on the 80 % split.")
    p.add_argument("--algo", choices=C.DRL_ALGOS, default=C.DRL_ALGO)
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--drl-epochs", type=int, default=C.DRL_EPOCHS)
    p.add_argument("--device", default=C.DEVICE)
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=C.LR)
    p.add_argument("--num-workers", type=int, default=C.NUM_WORKERS)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--rf-selection-mode", choices=["holdout", "final"], default=C.RF_SELECTION_MODE)
    p.add_argument("--rf-max-samples", type=float, default=C.RF_MAX_SAMPLES,
                   help="Per-tree bootstrap fraction (e.g. 0.5) - only if the RF stage runs out of RAM.")
    p.add_argument("--subsample-fraction", type=float, default=None,
                   help="OPT-IN debug subsample of the training set (default: disabled = 100 %%).")
    args = p.parse_args(argv)

    set_seed(C.SEED)
    if args.rf_max_samples is not None:
        C.RF_MAX_SAMPLES = args.rf_max_samples
        log.warning(f"RF per-tree bootstrap fraction = {args.rf_max_samples} (forest still sees all flows).")
    device = resolve_device(args.device)
    log_device(device)
    for key in ("X_train", "y_train", "train_inner_val", "train_group", "class_weights", "X_flows_train"):
        if not C.artifact_path(key).exists():
            log.error(f"Missing {C.artifact_path(key)}. Run `python -m src.preprocess --force` first.")
            raise SystemExit(2)

    y = np.load(C.artifact_path("y_train"))
    iv_mask = np.load(C.artifact_path("train_inner_val"))
    groups = np.load(C.artifact_path("train_group"))
    cw = np.load(C.artifact_path("class_weights"))
    it_idx, iv_idx = np.flatnonzero(~iv_mask), np.flatnonzero(iv_mask)
    if len(iv_idx) == 0 or len(it_idx) == 0:
        log.error("Inner-train or inner-val split is empty - dataset too small.")
        raise SystemExit(2)

    yf = np.load(C.artifact_path("y_flows_train"))
    fiv = np.load(C.artifact_path("flows_train_inner_val"))
    flow_idx = np.arange(len(yf))
    if args.subsample_fraction is not None and args.subsample_fraction < 1.0:
        log.warning(f"--subsample-fraction={args.subsample_fraction}: training on a SUBSAMPLE "
                    f"(not the full 80 %). Results are for debugging only.")
        it_idx = subsample_indices(it_idx, args.subsample_fraction)
        flow_idx = subsample_indices(flow_idx, args.subsample_fraction)
    else:
        log.info(f"Using 100 % of the 80 % training split: {len(y):,} windows "
                 f"({len(it_idx):,} inner-train + {len(iv_idx):,} inner-val), {len(yf):,} flows.")

    csvlog = CSVLogger(C.artifact_path("train_log"))
    tb = _tb_writer(not args.no_tensorboard)
    manifest: Dict[str, Any] = {"device": device, "args": vars(args), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        # ---------------- Stage 1: TCN ----------------
        model = train_tcn(args, device, y, it_idx, iv_idx, cw, csvlog, tb)
        model, _ = load_checkpoint(C.artifact_path("tcn"), device)

        # ---------------- Stage 2: RF -----------------
        # Memory-lean order: the holdout selection RF is fit, used and freed
        # before the final RF is fit, so only one forest is in RAM at a time.
        Xf = np.load(C.artifact_path("X_flows_train"), mmap_mode="r")
        Xw = np.load(C.artifact_path("X_train"), mmap_mode="r")
        X_iv_last = last_flow(Xw, iv_idx)
        full = len(flow_idx) == len(yf)
        if args.rf_selection_mode == "holdout":
            sel = flow_idx[~fiv[flow_idx]]
            log.info(f"[RF] holdout selection RF on {len(sel):,} inner-train flows (unbiased weight search)")
            rf_sel = fit_rf(Xf[sel], yf[sel])
            p_rf_iv = predict_attack_proba(rf_sel, X_iv_last)
            del rf_sel
            gc.collect()
        rf_final = fit_rf(Xf if full else Xf[flow_idx], yf[flow_idx])
        save_rf(rf_final, C.artifact_path("rf"))
        log.info(f"[RF] saved -> {C.artifact_path('rf')} (fit on {len(flow_idx):,} flows)")
        if args.rf_selection_mode == "final":
            log.warning("[RF] rf-selection-mode=final: weight search uses the final RF (biased toward RF).")
            p_rf_iv = predict_attack_proba(rf_final, X_iv_last)
        del rf_final, Xf
        gc.collect()
        csvlog.log({"stage": "rf", "n_flows": len(flow_idx), "selection_mode": args.rf_selection_mode})

        # ---------------- Stage 3: ensemble weights ----------------
        p_dnn_iv, _ = infer(model, Xw, device, iv_idx, use_amp=not args.no_amp)
        res = grid_search_weights(p_dnn_iv, p_rf_iv, y[iv_idx])
        save_json(res, C.artifact_path("ensemble_weights"))
        bu, bn = res["best_unconstrained"], res["best_non_degenerate"]
        log.info(f"[ENS] best unconstrained w_dnn={bu['w_dnn']:.2f} F1={bu['f1']:.4f} AUC={bu['roc_auc']:.4f}")
        log.info(f"[ENS] best non-degenerate (w_dnn>={C.NON_DEGENERATE_MIN_W}) w_dnn={bn['w_dnn']:.2f} "
                 f"F1={bn['f1']:.4f} AUC={bn['roc_auc']:.4f}")
        log.info(f"[ENS] selected ({res['selection_rule']}): w_dnn={res['w_dnn']:.2f}, w_rf={res['w_rf']:.2f}")
        csvlog.log({"stage": "ensemble", "w_dnn": res["w_dnn"], "val_f1": res["selected"]["f1"],
                    "val_auc": res["selected"]["roc_auc"]})

        # ---------------- Stage 4: DRL ----------------
        all_idx = np.arange(len(y))
        _, emb = infer(model, Xw, device, all_idx, return_embedding=True, use_amp=not args.no_amp)
        env = PolicyEnv(emb, y, groups, device)
        log.info(f"[DRL] algo={args.algo} | full pass over {env.n:,} training windows per epoch | "
                 f"buffer capacity=max(REPLAY_BUFFER={C.REPLAY_BUFFER:,}, N)")
        agent = make_agent(args.algo, env.state_dim, device)
        drl_log = CSVLogger(C.artifact_path("drl_log"))

        def _on_epoch(row: Dict[str, float]) -> None:
            drl_log.log(row)
            if tb is not None:
                for k in ("loss", "behaviour_reward", "greedy_reward", "epsilon"):
                    if row.get(k) == row.get(k):
                        tb.add_scalar(f"drl/{k}", row[k], int(row["epoch"]))

        logs = agent.train(env, epochs=args.drl_epochs, batch_size=args.batch_size, on_epoch=_on_epoch)
        write_csv(logs["steps"], C.ARTIFACT_DIR / "drl_steps.csv")
        save_agent(agent, C.artifact_path("drl"), {"epochs": args.drl_epochs, "n_windows": env.n})
        log.info(f"[DRL] saved -> {C.artifact_path('drl')}")
    except torch.cuda.OutOfMemoryError:
        log.error(OOM_HELP)
        raise SystemExit(3)
    finally:
        if tb is not None:
            tb.close()

    write_decision_schema(C.artifact_path("decision_schema"))
    manifest.update({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "n_train_windows": int(len(y)),
                     "n_inner_train_windows_used": int(len(it_idx)), "n_inner_val_windows": int(len(iv_idx)),
                     "n_train_flows_used": int(len(flow_idx)), "subsample_fraction": args.subsample_fraction,
                     "drl_algo": args.algo})
    save_json(manifest, C.artifact_path("train_manifest"))
    log.info("Training complete. Next: python -m src.evaluate --checkpoint artifacts/tcn_best.pt")


if __name__ == "__main__":
    main(sys.argv[1:])
'''

# ==========================================================================
# src/inference_api.py
# ==========================================================================
FILES['src/inference_api.py'] = r'''"""Phase-2 seam: one-call inference over a window of raw flow records.

Step 3 (Cilium eBPF / BCC + XDP) will run an XDP feature extractor that emits,
per flow, the feature vector in exactly the order of
``artifacts/feature_columns.json``. Its userspace consumer keeps the last
``WINDOW_SIZE`` records per flow key and calls::

    ids = IDSInference(Path("artifacts"), device="cuda")
    out = ids.predict_window(window_raw, flow_key="10.0.0.5")   # (WINDOW_SIZE, F)
    if out["decision"] == "BLOCK":
        ids.block_ip("10.0.0.5")          # Phase 2: also writes block_map

No retraining is needed: the scaler, feature order, models, ensemble weights
and decision thresholds are all loaded from ``artifacts/``. The per-flow
confirmation counter ``c_f`` and the block TTL table live here in user space
and mirror what Phase 2 re-implements as ``block_map`` + a BlockManager thread.

Self-test (no dataset needed): ``python -m src.inference_api --self-test``.
If trained artifacts are absent it runs in *synthetic mode* with randomly
initialised components purely to exercise the interface.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import config as C
from . import schema as S
from .ensemble import ConfirmationState
from .utils import get_logger, load_json, save_json

log = get_logger()

DECISION_FIELDS: Dict[str, Dict[str, str]] = {
    "flow_key": {"type": "string", "description": "Flow key whose confirmation counter was updated."},
    "p_dnn": {"type": "number", "description": "TCN P(attack) for the window, in [0,1]."},
    "p_rf": {"type": "number", "description": "Calibrated RF P(attack) for the window's last flow, in [0,1]."},
    "s_ensemble": {"type": "number", "description": "S = w_dnn*p_dnn + w_rf*p_rf."},
    "y_hat": {"type": "integer", "description": "1 if S >= 0.5 else 0."},
    "decision": {"type": "string", "enum": "PASS|MONITOR|BLOCK", "description": "Final action."},
    "confidence": {"type": "number", "description": "S if y_hat==1 else 1-S."},
    "consecutive_attacks": {"type": "integer", "description": "Confirmation counter c_f in [0, K_CONFIRM]."},
    "drl_action": {"type": "string|null", "description": "DRL greedy action (pass|monitor|block) or null."},
    "decision_source": {"type": "string", "description": "'ensemble' or 'drl' (USE_DRL_AT_INFERENCE)."},
    "latency_ms": {"type": "number", "description": "Userspace model latency for this call (not the XDP path)."},
}


def write_decision_schema(path: Path, n_features: Optional[int] = None) -> Dict[str, Any]:
    """Write ``decision_schema.json`` describing the ``predict_window`` contract.

    Args:
        path: Destination path.
        n_features: Feature count (read from ``feature_columns.json`` if omitted).

    Returns:
        The schema dict.
    """
    path = Path(path)
    fc = path.parent / Path(C.FEATURE_ORDER_PATH).name
    if n_features is None and fc.exists():
        n_features = len(load_json(fc))
    schema = {
        "title": "IDSInference.predict_window decision",
        "version": 1,
        "input": {
            "features_2d": {"shape": [C.WINDOW_SIZE, n_features if n_features else "F"], "dtype": "float32",
                            "units": "raw (unscaled) CICFlowMeter-compatible flow features",
                            "row_order": "oldest -> newest flow record of the flow key",
                            "column_order": C.FEATURE_ORDER_PATH},
            "flow_key": {"type": "string", "description": "e.g. source IP or 5-tuple hash used for c_f"},
        },
        "output": DECISION_FIELDS,
        "actions": C.ACTIONS,
        "decision_rule": {"tau": C.TAU, "k_confirm": C.K_CONFIRM, "tau_hard": C.TAU_HARD,
                          "threshold": C.DECISION_THRESHOLD},
        "block_ttl_seconds": C.BLOCK_TTL_SECONDS,
        "scaler_path": C.SCALER_PATH,
    }
    save_json(schema, path)
    return schema


def _single_thread_rf(model: Any) -> None:
    """Use n_jobs=1 for per-window RF scoring (thread start-up dominates on 1 row)."""
    for cc in getattr(model, "calibrated_classifiers_", []):
        est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if hasattr(est, "n_jobs"):
            est.n_jobs = 1
    if hasattr(model, "n_jobs") and not hasattr(model, "calibrated_classifiers_"):
        model.n_jobs = 1


class IDSInference:
    """Load Phase-1 artifacts and score windows of raw flow records.

    Args:
        artifact_dir: Folder containing the Phase-1 artifacts.
        device: ``"cuda"`` or ``"cpu"`` (auto-falls back to CPU).
        use_drl: Override ``USE_DRL_AT_INFERENCE``.
        components: Internal - pre-built components (synthetic self-test mode).
    """

    def __init__(self, artifact_dir: Path = C.ARTIFACT_DIR, device: str = "cuda", use_drl: Optional[bool] = None,
                 components: Optional[Dict[str, Any]] = None) -> None:
        import torch

        from .utils import resolve_device

        self.torch = torch
        self.artifact_dir = Path(artifact_dir)
        self.device = resolve_device(device)
        self.use_drl = C.USE_DRL_AT_INFERENCE if use_drl is None else bool(use_drl)
        comp = components if components is not None else self._load_components()
        self.feature_columns: List[str] = comp["feature_columns"]
        self.n_features = len(self.feature_columns)
        self.mean = np.asarray(comp["scaler"].mean_, dtype=np.float32)
        self.scale = np.asarray(comp["scaler"].scale_, dtype=np.float32)
        if len(self.mean) != self.n_features:
            raise ValueError(f"Scaler has {len(self.mean)} features but feature_columns.json has {self.n_features}.")
        self.tcn = comp["tcn"].to(self.device).eval()
        self.rf = comp["rf"]
        _single_thread_rf(self.rf)
        ew = comp["ensemble_weights"]
        self.w_dnn = float(ew["w_dnn"])
        self.w_rf = 1.0 - self.w_dnn
        self.state = ConfirmationState(ew.get("tau", C.TAU), ew.get("k_confirm", C.K_CONFIRM),
                                       ew.get("tau_hard", C.TAU_HARD))
        self.drl = comp.get("drl")
        if self.drl is not None:
            self.drl = self.drl.to(self.device)
        if self.use_drl and self.drl is None:
            raise FileNotFoundError("USE_DRL_AT_INFERENCE=True but drl_agent.pt was not found.")
        self.block_table: Dict[str, float] = {}
        self.synthetic = bool(comp.get("synthetic", False))
        schema_path = self.artifact_dir / Path(C.DECISION_OUTPUT_SCHEMA).name
        if not schema_path.exists():
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            write_decision_schema(schema_path, self.n_features)

    # ------------------------------------------------------------------ #
    def _load_components(self) -> Dict[str, Any]:
        import joblib

        from .drl_agent import load_policy
        from .rf_branch import load_rf
        from .tcn import load_checkpoint

        a = self.artifact_dir
        need = {"feature_columns": C.ART["feature_columns"], "scaler": C.ART["scaler"], "tcn": C.ART["tcn"],
                "rf": C.ART["rf"], "ensemble_weights": C.ART["ensemble_weights"]}
        missing = [n for n in need.values() if not (a / n).exists()]
        if missing:
            raise FileNotFoundError(f"Missing artifacts in {a}: {missing}. Run preprocess + train first.")
        tcn, payload = load_checkpoint(a / need["tcn"], "cpu")
        comp = {
            "feature_columns": load_json(a / need["feature_columns"]),
            "scaler": joblib.load(a / need["scaler"]),
            "tcn": tcn,
            "rf": load_rf(a / need["rf"]),
            "ensemble_weights": load_json(a / need["ensemble_weights"]),
            "drl": load_policy(a / C.ART["drl"], "cpu") if (a / C.ART["drl"]).exists() else None,
        }
        if payload["model_config"]["n_features"] != len(comp["feature_columns"]):
            raise ValueError("tcn_best.pt was trained with a different feature count than feature_columns.json.")
        return comp

    @classmethod
    def synthetic(cls, device: str = "cpu", artifact_dir: Optional[Path] = None) -> "IDSInference":
        """Build an instance from random components (interface smoke test only).

        Args:
            device: Device string.
            artifact_dir: Where ``decision_schema.json`` is written (temp-like ``artifacts/selftest``).

        Returns:
            A synthetic :class:`IDSInference` (predictions are meaningless).
        """
        import torch
        from sklearn.preprocessing import StandardScaler

        from .rf_branch import build_rf
        from .tcn import TCNClassifier

        torch.manual_seed(C.SEED)
        rng = np.random.default_rng(C.SEED)
        feats = list(S.CICIDS2017_FEATURES)
        Xr = rng.normal(size=(400, len(feats))).astype(np.float32)
        yr = (Xr[:, 0] > 0).astype(int)
        rf = build_rf(n_estimators=10, calibrated=True).fit(Xr, yr)
        comp = {"feature_columns": feats, "scaler": StandardScaler().fit(Xr), "tcn": TCNClassifier(len(feats)),
                "rf": rf, "ensemble_weights": {"w_dnn": 0.5}, "drl": None, "synthetic": True}
        return cls(artifact_dir or (C.ARTIFACT_DIR / "selftest"), device=device, use_drl=False, components=comp)

    # ------------------------------------------------------------------ #
    def _prepare(self, features_2d: np.ndarray) -> np.ndarray:
        x = np.asarray(features_2d, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"features_2d must have shape ({C.WINDOW_SIZE}, {self.n_features}); got {x.shape}")
        if x.shape[0] > C.WINDOW_SIZE:
            x = x[-C.WINDOW_SIZE:]
        elif x.shape[0] < C.WINDOW_SIZE:  # same left-padding rule as preprocessing
            x = np.concatenate([np.repeat(x[:1], C.WINDOW_SIZE - x.shape[0], axis=0), x], axis=0)
        x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
        return (x - self.mean) / self.scale

    def predict_window(self, features_2d: np.ndarray, flow_key: str = "default") -> Dict[str, Any]:
        """Score one window and update the flow key's confirmation counter.

        Args:
            features_2d: ``(WINDOW_SIZE, F)`` raw (unscaled) features, columns in
                ``feature_columns.json`` order, rows oldest -> newest. Fewer rows
                are left-padded with the first row; extra rows keep the newest.
            flow_key: Key for the per-flow ``c_f`` counter.

        Returns:
            Dict with ``p_dnn, p_rf, s_ensemble, decision, confidence,
            consecutive_attacks`` (+ ``flow_key, y_hat, drl_action,
            decision_source, latency_ms``); see ``decision_schema.json``.
        """
        torch = self.torch
        t0 = time.perf_counter()
        xs = self._prepare(features_2d)
        with torch.no_grad():
            xt = torch.from_numpy(xs[None]).to(self.device)
            logits, z = self.tcn.forward_with_embedding(xt)
            p_dnn = float(torch.softmax(logits.float(), 1)[0, 1])
        classes = list(getattr(self.rf, "classes_", [0, 1]))
        p_rf = float(self.rf.predict_proba(xs[-1:])[0, classes.index(1)])
        s = self.w_dnn * p_dnn + self.w_rf * p_rf
        decision, c_f = self.state.update(flow_key, s)
        drl_action = None
        source = "ensemble"
        if self.drl is not None:
            drl_action = C.ACTIONS[int(self.drl.act(z.float())[0])]
            if self.use_drl:
                decision, source = drl_action.upper(), "drl"
        y_hat = int(s >= C.DECISION_THRESHOLD)
        return {
            "flow_key": flow_key, "p_dnn": p_dnn, "p_rf": p_rf, "s_ensemble": float(s), "y_hat": y_hat,
            "decision": decision, "confidence": float(s if y_hat else 1.0 - s), "consecutive_attacks": int(c_f),
            "drl_action": drl_action, "decision_source": source,
            "latency_ms": (time.perf_counter() - t0) * 1e3,
        }

    def reset_flow_state(self, flow_key: str) -> None:
        """Forget the confirmation counter of ``flow_key`` (e.g. flow evicted from the LRU)."""
        self.state.reset(flow_key)

    def block_ip(self, ip: str, ttl: int = C.BLOCK_TTL_SECONDS) -> None:
        """Record a userspace block for ``ip`` that expires after ``ttl`` seconds.

        Phase 2 mirrors this into the XDP ``block_map``; Phase 1 only keeps the table.

        Args:
            ip: Address (string).
            ttl: Seconds until expiry.
        """
        self.block_table[ip] = time.monotonic() + float(ttl)

    def is_blocked(self, ip: str) -> bool:
        """True if ``ip`` has an unexpired block entry."""
        exp = self.block_table.get(ip)
        if exp is None:
            return False
        if exp <= time.monotonic():
            del self.block_table[ip]
            return False
        return True

    def purge_expired(self) -> List[str]:
        """Remove and return expired block entries (BlockManager tick)."""
        now = time.monotonic()
        gone = [ip for ip, e in self.block_table.items() if e <= now]
        for ip in gone:
            del self.block_table[ip]
        return gone


def self_test(device: str = "cpu") -> Dict[str, Any]:
    """Run one zero window through the API; synthetic mode if artifacts are missing.

    Args:
        device: Device string.

    Returns:
        The decision dict.
    """
    needed = [C.ART[k] for k in ("feature_columns", "scaler", "tcn", "rf", "ensemble_weights")]
    if all((C.ARTIFACT_DIR / n).exists() for n in needed):
        ids = IDSInference(C.ARTIFACT_DIR, device=device)
        log.info("Self-test using trained artifacts.")
    else:
        log.warning("Trained artifacts not found -> SYNTHETIC mode (random weights; interface check only).")
        ids = IDSInference.synthetic(device=device)
    window = np.zeros((C.WINDOW_SIZE, ids.n_features), dtype=np.float32)
    out = ids.predict_window(window, flow_key="selftest-flow")
    for _ in range(C.K_CONFIRM):
        out = ids.predict_window(window, flow_key="selftest-flow")
    ids.block_ip("192.0.2.1", ttl=1)
    assert ids.is_blocked("192.0.2.1")
    ids.reset_flow_state("selftest-flow")
    log.info(f"Feature count: {ids.n_features} | window: {C.WINDOW_SIZE} | w_dnn={ids.w_dnn:.2f} | "
             f"synthetic={ids.synthetic}")
    for k, v in out.items():
        log.info(f"  {k:<20} {v}")
    log.info(f"SELF-TEST OK - decision: {out['decision']}")
    return out


def main(argv: Optional[List[str]] = None) -> None:
    """CLI: ``python -m src.inference_api --self-test [--device cpu]``."""
    p = argparse.ArgumentParser(description="IDSInference (Phase-2 seam)")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--device", default=C.DEVICE)
    a = p.parse_args(argv)
    if a.self_test:
        self_test(a.device)
    else:
        p.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])
'''

# ==========================================================================
# src/dataset.py
# ==========================================================================
FILES['src/dataset.py'] = r'''"""Windowed PyTorch dataset over the ``(N, WINDOW_SIZE, F)`` arrays from Step 1.

The dataset stores the *path* of the ``.npy`` file and opens it lazily as a
read-only memory map in every worker process, so multi-GB window arrays are
never pickled into DataLoader workers. Batches are fetched with one sorted
fancy-index read via ``__getitems__`` (PyTorch >= 2.0), which is much faster
than per-sample indexing on a memory map.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import config as C


class WindowDataset(Dataset):
    """Map-style dataset of flow windows and binary window labels.

    Args:
        x_path: Path of the ``.npy`` window array ``(N, W, F)`` (float32).
        y: Window labels ``(N,)`` (int).
        indices: Optional subset of window indices (e.g. inner-train). ``None`` = all.
    """

    def __init__(self, x_path: Path, y: np.ndarray, indices: Optional[np.ndarray] = None) -> None:
        self.x_path = Path(x_path)
        self.y = np.asarray(y).astype(np.int64)
        self.indices = np.arange(len(self.y), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        self._X: Optional[np.ndarray] = None
        shape = np.load(self.x_path, mmap_mode="r").shape
        if shape[0] != len(self.y):
            raise ValueError(f"{self.x_path} has {shape[0]} windows but y has {len(self.y)}")
        self.window_size, self.n_features = int(shape[1]), int(shape[2])

    # Lazily opened memmap - never pickled into workers.
    @property
    def X(self) -> np.ndarray:
        """Read-only memory map of the window array."""
        if self._X is None:
            self._X = np.load(self.x_path, mmap_mode="r")
        return self._X

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_X"] = None
        return state

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        j = int(self.indices[i])
        return torch.from_numpy(np.array(self.X[j], dtype=np.float32)), torch.tensor(self.y[j])

    def __getitems__(self, idxs: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        rows = np.sort(self.indices[np.asarray(idxs, dtype=np.int64)])
        xb = np.ascontiguousarray(self.X[rows], dtype=np.float32)
        return torch.from_numpy(xb), torch.from_numpy(self.y[rows])


def batch_identity(batch: Any) -> Any:
    """Collate function for datasets that already return whole batches."""
    return batch


def _worker_init(worker_id: int) -> None:
    seed = (C.SEED + worker_id) % (2 ** 32)
    np.random.seed(seed)


def make_loader(ds: WindowDataset, batch_size: int, shuffle: bool, device: str,
                num_workers: int = C.NUM_WORKERS, seed: int = C.SEED) -> DataLoader:
    """Create a DataLoader that never drops samples (``drop_last=False``).

    Args:
        ds: Dataset.
        batch_size: Batch size.
        shuffle: Shuffle each epoch (seeded generator).
        device: Target device; enables ``pin_memory`` on CUDA.
        num_workers: Worker processes.
        seed: Generator seed.

    Returns:
        A :class:`torch.utils.data.DataLoader`.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=False,
        num_workers=num_workers, pin_memory=(C.PIN_MEMORY and device.startswith("cuda")),
        collate_fn=batch_identity, generator=g, worker_init_fn=_worker_init,
        persistent_workers=num_workers > 0,
    )


def expected_batches(n: int, batch_size: int) -> Dict[str, int]:
    """Batch accounting used for the "whole 80 % per epoch" assertion.

    Args:
        n: Number of samples.
        batch_size: Batch size.

    Returns:
        ``{"full": floor(n/bs), "partial": 0|1, "total": ceil(n/bs)}``.
    """
    full = n // batch_size
    return {"full": full, "partial": int(n % batch_size > 0), "total": math.ceil(n / batch_size)}


def subsample_indices(indices: np.ndarray, fraction: Optional[float], seed: int = C.SEED) -> np.ndarray:
    """Opt-in random subsample (sorted); identity when ``fraction`` is ``None`` or 1.

    Args:
        indices: Candidate indices.
        fraction: Fraction in ``(0, 1]`` or ``None``.
        seed: RNG seed.

    Returns:
        Sorted subset of ``indices``.
    """
    if fraction is None or fraction >= 1.0:
        return indices
    if not 0.0 < fraction < 1.0:
        raise ValueError("--subsample-fraction must be in (0, 1].")
    rng = np.random.default_rng(seed)
    k = max(1, int(round(len(indices) * fraction)))
    return np.sort(rng.choice(indices, size=k, replace=False))


def contiguous_runs(idx: np.ndarray) -> List[slice]:
    """Split sorted indices into contiguous slices (fast memmap reads).

    Args:
        idx: Sorted int indices.

    Returns:
        List of slices.
    """
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(idx)]])
    return [slice(int(idx[a]), int(idx[b - 1]) + 1) for a, b in zip(starts, ends)]
'''

# ==========================================================================
# src/tcn.py
# ==========================================================================
FILES['src/tcn.py'] = r'''"""Temporal Convolutional Network encoder + binary head.

Architecture (spec §6a):

* Causal dilated ``Conv1d`` residual blocks with weight normalisation, ReLU and
  Dropout(0.3). One block per entry of ``TCN_DILATIONS`` (1, 2, 4, 8). Block
  ``i`` uses ``TCN_CHANNELS[min(i, len-1)]`` output channels, i.e. 64, 128,
  128, 128 - the channel list is shorter than the dilation list, so the last
  width is repeated. Receptive field = 1 + 2*(k-1)*sum(d) = 61 >= WINDOW_SIZE.
* Global average pool over time -> Linear -> BatchNorm -> ReLU -> Dropout(0.3)
  -> embedding ``(B, TCN_EMBED_DIM)``. (ReLU added for a non-linear embedding.)
* Head: Linear(128->64) -> BN -> ReLU -> Dropout(0.3) -> Linear(64->32) -> ReLU
  -> Linear(32->2).

Input layout is ``(B, W, F)`` (window of flow records); it is transposed to
``(B, F, W)`` internally.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .utils import autocast


def _weight_norm(module: nn.Module) -> nn.Module:
    try:
        from torch.nn.utils.parametrizations import weight_norm

        return weight_norm(module)
    except ImportError:  # pragma: no cover - torch < 2.1
        return nn.utils.weight_norm(module)


class Chomp1d(nn.Module):
    """Remove right padding so the convolution is causal.

    Args:
        chomp: Number of trailing time steps to drop.
    """

    def __init__(self, chomp: int) -> None:
        super().__init__()
        self.chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return x[:, :, : -self.chomp].contiguous() if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    """Residual block of two causal dilated convolutions.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        kernel: Kernel size.
        dilation: Dilation.
        dropout: Dropout probability.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            _weight_norm(nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            _weight_norm(nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.relu(self.net(x) + self.downsample(x))


class TCNEncoder(nn.Module):
    """Stack of temporal blocks + pooling + projection to an embedding.

    Args:
        n_features: Number of per-flow features ``F``.
        channels: Channel widths (repeated to match ``dilations``).
        kernel: Kernel size.
        dilations: Dilation per block.
        embed_dim: Embedding size.
        dropout: Dropout probability.
    """

    def __init__(self, n_features: int, channels: List[int] = C.TCN_CHANNELS, kernel: int = C.TCN_KERNEL,
                 dilations: List[int] = C.TCN_DILATIONS, embed_dim: int = C.TCN_EMBED_DIM,
                 dropout: float = C.DROPOUT) -> None:
        super().__init__()
        blocks: List[nn.Module] = []
        in_ch = n_features
        for i, d in enumerate(dilations):
            out_ch = channels[min(i, len(channels) - 1)]
            blocks.append(TemporalBlock(in_ch, out_ch, kernel, d, dropout))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Sequential(nn.Linear(in_ch, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode windows.

        Args:
            x: ``(B, W, F)`` tensor.

        Returns:
            ``(B, embed_dim)`` embedding.
        """
        h = self.blocks(x.transpose(1, 2))  # (B, C, W)
        return self.proj(h.mean(dim=2))


class TCNClassifier(nn.Module):
    """TCN encoder followed by the binary MLP head.

    Args:
        n_features: Number of per-flow features.
        hidden_dims: ``[embed, h1, h2]`` (spec ``HIDDEN_DIMS``).
        **enc_kwargs: Forwarded to :class:`TCNEncoder`.
    """

    def __init__(self, n_features: int, hidden_dims: List[int] = C.HIDDEN_DIMS, dropout: float = C.DROPOUT,
                 num_classes: int = C.NUM_CLASSES, **enc_kwargs: Any) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.encoder = TCNEncoder(n_features, dropout=dropout, **enc_kwargs)
        e, h1, h2 = self.encoder.embed_dim, hidden_dims[1], hidden_dims[2]
        self.head = nn.Sequential(
            nn.Linear(e, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits ``(B, 2)`` for windows ``(B, W, F)``."""
        return self.head(self.encoder(x))

    def forward_with_embedding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, embedding)``."""
        z = self.encoder(x)
        return self.head(z), z


def model_config(n_features: int) -> Dict[str, Any]:
    """Serialisable hyper-parameters needed to rebuild the model.

    Args:
        n_features: Number of features.

    Returns:
        Config dict stored inside the checkpoint.
    """
    return {
        "n_features": n_features, "channels": list(C.TCN_CHANNELS), "kernel": C.TCN_KERNEL,
        "dilations": list(C.TCN_DILATIONS), "embed_dim": C.TCN_EMBED_DIM, "dropout": C.DROPOUT,
        "hidden_dims": list(C.HIDDEN_DIMS), "window_size": C.WINDOW_SIZE,
    }


def build_from_config(cfg: Dict[str, Any]) -> TCNClassifier:
    """Instantiate a :class:`TCNClassifier` from :func:`model_config` output."""
    return TCNClassifier(
        cfg["n_features"], hidden_dims=cfg["hidden_dims"], dropout=cfg["dropout"],
        channels=cfg["channels"], kernel=cfg["kernel"], dilations=cfg["dilations"], embed_dim=cfg["embed_dim"],
    )


def save_checkpoint(model: TCNClassifier, path: Path, extra: Optional[Dict[str, Any]] = None) -> None:
    """Save weights + config (+ metadata) to ``path``."""
    payload = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "model_config": model_config(model.n_features)}
    payload.update(extra or {})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: str = "cpu") -> Tuple[TCNClassifier, Dict[str, Any]]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Args:
        path: Checkpoint path.
        device: Map location.

    Returns:
        ``(model_in_eval_mode, payload)``.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_from_config(payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload


@torch.no_grad()
def infer(model: TCNClassifier, X: np.ndarray, device: str, indices: Optional[np.ndarray] = None,
          batch_size: int = C.EVAL_BATCH_SIZE, return_embedding: bool = False, use_amp: bool = C.USE_AMP
          ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Batched inference over a (memory-mapped) window array.

    Args:
        model: Trained classifier.
        X: ``(N, W, F)`` array or memmap (already scaled).
        device: Device string.
        indices: Optional sorted subset of rows.
        batch_size: Inference batch size.
        return_embedding: Also return embeddings (float16 to save RAM).
        use_amp: Use autocast on CUDA.

    Returns:
        ``(p_attack (n,), embeddings (n, E) | None)``.
    """
    model.eval()
    idx = np.arange(len(X)) if indices is None else np.asarray(indices)
    probs = np.empty(len(idx), dtype=np.float32)
    embs = np.empty((len(idx), model.encoder.embed_dim), dtype=np.float16) if return_embedding else None
    for s in range(0, len(idx), batch_size):
        sel = idx[s:s + batch_size]
        xb = np.array(X[sel[0]:sel[-1] + 1] if (sel[-1] - sel[0] + 1) == len(sel) else X[sel], dtype=np.float32)
        xt = torch.from_numpy(xb).to(device, non_blocking=True)
        with autocast(device, use_amp):
            logits, z = model.forward_with_embedding(xt)
        probs[s:s + len(sel)] = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        if embs is not None:
            embs[s:s + len(sel)] = z.float().cpu().numpy().astype(np.float16)
    return probs, embs
'''

# ==========================================================================
# src/rf_branch.py
# ==========================================================================
FILES['src/rf_branch.py'] = r'''"""Isotonic-calibrated RandomForest branch (spec §6b).

The RF is a *flow-level* model: it is fit on every training flow of the 80 %
split (not only on window endpoints). Inside the window pipeline it is applied
to the **last flow of each window**, so at inference time it scores the newest
flow record while the TCN scores the temporal context.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from . import config as C
from .utils import get_logger

log = get_logger()


def build_rf(n_estimators: int = C.RF_N_ESTIMATORS, calibrated: bool = True, seed: int = C.SEED):
    """Construct the (optionally calibrated) RandomForest.

    Args:
        n_estimators: Number of trees.
        calibrated: Wrap in ``CalibratedClassifierCV(method="isotonic", cv=3,
            ensemble=RF_CALIBRATION_ENSEMBLE)``.
        seed: ``random_state``.

    Returns:
        An unfitted scikit-learn estimator.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=C.RF_MAX_DEPTH, min_samples_leaf=C.RF_MIN_SAMPLES_LEAF,
        class_weight="balanced", n_jobs=-1, random_state=seed, max_samples=C.RF_MAX_SAMPLES,
    )
    if not calibrated:
        return rf
    try:
        return CalibratedClassifierCV(estimator=rf, method=C.RF_CALIBRATION_METHOD, cv=C.RF_CALIBRATION_CV,
                                      ensemble=C.RF_CALIBRATION_ENSEMBLE)
    except TypeError:  # scikit-learn < 1.2
        return CalibratedClassifierCV(base_estimator=rf, method=C.RF_CALIBRATION_METHOD, cv=C.RF_CALIBRATION_CV,
                                      ensemble=C.RF_CALIBRATION_ENSEMBLE)


def fit_rf(X: np.ndarray, y: np.ndarray, n_estimators: int = C.RF_N_ESTIMATORS, calibrated: bool = True):
    """Fit the RF branch on flow-level data.

    Args:
        X: ``(n, F)`` scaled flow features (copied into RAM if memory-mapped).
        y: ``(n,)`` binary labels.
        n_estimators: Number of trees.
        calibrated: Use isotonic calibration.

    Returns:
        Fitted estimator.

    Raises:
        ValueError: If ``y`` contains a single class.
    """
    y = np.asarray(y).astype(np.int64)
    if len(np.unique(y)) < 2:
        raise ValueError("RF training data contains a single class; cannot fit a binary classifier.")
    X = np.ascontiguousarray(X, dtype=np.float32)  # reads a memmap into RAM exactly once
    log.info(f"Fitting RF on {len(y):,} flows x {X.shape[1]} features "
             f"(trees={n_estimators}, calibrated={calibrated}, cv={C.RF_CALIBRATION_CV if calibrated else '-'})")
    model = build_rf(n_estimators=n_estimators, calibrated=calibrated)
    model.fit(X, y)
    return model


def predict_attack_proba(model, X: np.ndarray, batch: int = C.RF_PREDICT_BATCH) -> np.ndarray:
    """Return P(attack) for flow rows, in batches.

    Args:
        model: Fitted estimator.
        X: ``(n, F)`` features.
        batch: Rows per batch.

    Returns:
        ``(n,)`` float32 probabilities.
    """
    out = np.empty(len(X), dtype=np.float32)
    classes = list(getattr(model, "classes_", [0, 1]))
    col = classes.index(1) if 1 in classes else -1
    for s in range(0, len(X), batch):
        xb = np.ascontiguousarray(X[s:s + batch], dtype=np.float32)
        out[s:s + len(xb)] = model.predict_proba(xb)[:, col]
    return out


def last_flow(X_windows: np.ndarray, indices: Optional[np.ndarray] = None, batch: int = C.RF_PREDICT_BATCH) -> np.ndarray:
    """Extract the last flow of every window: ``X[:, -1, :]``.

    Args:
        X_windows: ``(N, W, F)`` array or memmap.
        indices: Optional sorted subset.
        batch: Read chunk size.

    Returns:
        ``(n, F)`` float32 array.
    """
    idx = np.arange(len(X_windows)) if indices is None else np.asarray(indices)
    out = np.empty((len(idx), X_windows.shape[2]), dtype=np.float32)
    for s in range(0, len(idx), batch):
        out[s:s + batch] = X_windows[idx[s:s + batch], -1, :]
    return out


def save_rf(model, path: Path) -> None:
    """Persist with joblib (compress=3)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)


def load_rf(path: Path):
    """Load a persisted RF branch."""
    return joblib.load(path)
'''

# ==========================================================================
# src/drl_agent.py
# ==========================================================================
FILES['src/drl_agent.py'] = r'''"""DRL response policy over TCN embeddings (spec §6c).

* :class:`PolicyEnv` - offline environment over the chronologically ordered
  training windows. State = frozen TCN embedding of the window, actions =
  ``{pass, monitor, block}``, reward from :data:`config.REWARDS`, next state =
  next window of the same flow group (terminal at the group's last window).
* :class:`DoubleDQNAgent` (default) - replay buffer, target-network sync every
  ``TARGET_UPDATE`` gradient steps, Huber loss, epsilon-greedy 1.0 -> 0.05 over
  ``EPS_DECAY_STEPS`` environment steps.
* :class:`PPOAgent` - clipped-objective actor-critic with GAE.

Full-pass guarantee: every DRL epoch first walks the **entire** 80 % training
set (all windows, inner-val included) to populate the buffer / rollout, then
runs one full pass of updates over it. The replay buffer stores transitions as
index arrays into a single embedding table, so its capacity grows to
``max(REPLAY_BUFFER, N)`` without duplicating embeddings in memory.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from .utils import autocast, get_logger, grad_scaler

log = get_logger()


def reward_matrix() -> np.ndarray:
    """Return the ``(2, 3)`` reward table ``R[label, action]``."""
    return np.array([[C.REWARDS[lab][a] for a in C.ACTIONS] for lab in (0, 1)], dtype=np.float32)


def epsilon_at(step: np.ndarray | int) -> np.ndarray | float:
    """Linear epsilon schedule ``EPS_START -> EPS_END`` over ``EPS_DECAY_STEPS``."""
    frac = np.minimum(np.asarray(step, dtype=np.float64) / max(C.EPS_DECAY_STEPS, 1), 1.0)
    return C.EPS_START + frac * (C.EPS_END - C.EPS_START)


def _mlp(inp: int, hidden: List[int], out: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(inp, h), nn.ReLU()]
        inp = h
    layers.append(nn.Linear(inp, out))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """MLP Q-function ``Q(s, .)`` over embeddings."""

    def __init__(self, state_dim: int, n_actions: int = len(C.ACTIONS), hidden: List[int] = C.DRL_HIDDEN) -> None:
        super().__init__()
        self.net = _mlp(state_dim, hidden, n_actions)

    def forward(self, s: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(s)


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic for PPO."""

    def __init__(self, state_dim: int, n_actions: int = len(C.ACTIONS), hidden: List[int] = C.DRL_HIDDEN) -> None:
        super().__init__()
        self.trunk = _mlp(state_dim, hidden[:-1], hidden[-1]) if len(hidden) > 1 else nn.Linear(state_dim, hidden[0])
        self.act = nn.ReLU()
        self.pi = nn.Linear(hidden[-1], n_actions)
        self.v = nn.Linear(hidden[-1], 1)

    def forward(self, s: torch.Tensor):  # noqa: D102
        h = self.act(self.trunk(s))
        return self.pi(h), self.v(h).squeeze(-1)


class PolicyEnv:
    """Offline environment over training-window embeddings.

    Args:
        embeddings: ``(N, E)`` TCN embeddings (float16/32), chronological per group.
        labels: ``(N,)`` window labels.
        groups: ``(N,)`` group id per window (contiguous runs).
        device: Device holding the embedding table.
    """

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, groups: np.ndarray, device: str) -> None:
        self.n = int(len(labels))
        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.states = torch.as_tensor(np.asarray(embeddings), dtype=self.dtype, device=device)
        self.labels_np = np.asarray(labels).astype(np.int64)
        g = np.asarray(groups)
        same_next = np.zeros(self.n, dtype=bool)
        same_next[:-1] = g[1:] == g[:-1]
        self.next_idx_np = np.where(same_next, np.arange(self.n) + 1, np.arange(self.n))
        self.done_np = (~same_next).astype(np.float32)
        self.R = torch.as_tensor(reward_matrix(), device=device)
        self.labels = torch.as_tensor(self.labels_np, device=device)
        self.next_idx = torch.as_tensor(self.next_idx_np, device=device)
        self.done = torch.as_tensor(self.done_np, device=device)
        self.state_dim = int(self.states.shape[1])

    def rewards(self, idx: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Reward for taking ``actions`` in windows ``idx``."""
        return self.R[self.labels[idx], actions]

    def state(self, idx: torch.Tensor) -> torch.Tensor:
        """Float32 states for ``idx``."""
        return self.states[idx].float()


class DoubleDQNAgent:
    """Double-DQN with Huber loss, target network and AMP on CUDA.

    Args:
        state_dim: Embedding size.
        device: Device string.
        lr: Learning rate.
    """

    algo = "double_dqn"

    def __init__(self, state_dim: int, device: str, lr: float = C.LR) -> None:
        self.device = device
        self.online = QNetwork(state_dim).to(device)
        self.target = QNetwork(state_dim).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = torch.optim.AdamW(self.online.parameters(), lr=lr, weight_decay=C.WEIGHT_DECAY)
        self.amp = C.USE_AMP and device.startswith("cuda")
        self.scaler = grad_scaler(device, self.amp)
        self.state_dim = state_dim
        self.env_steps = 0
        self.updates = 0

    @torch.no_grad()
    def greedy(self, states: torch.Tensor) -> torch.Tensor:
        """Greedy actions for float states ``(B, E)``."""
        self.online.eval()
        return self.online(states.float()).argmax(dim=1)

    @torch.no_grad()
    def _collect(self, env: PolicyEnv, batch: int, step_log: List[Dict[str, float]]) -> torch.Tensor:
        """One full epsilon-greedy pass over every training window."""
        actions = torch.empty(env.n, dtype=torch.long, device=self.device)
        gen = torch.Generator(device="cpu").manual_seed(C.SEED + self.env_steps)
        for s in range(0, env.n, batch):
            idx = torch.arange(s, min(s + batch, env.n), device=self.device)
            eps = torch.as_tensor(epsilon_at(self.env_steps + np.arange(len(idx))), dtype=torch.float32)
            greedy = self.greedy(env.state(idx)).cpu()
            rand_a = torch.randint(0, len(C.ACTIONS), (len(idx),), generator=gen)
            explore = torch.rand(len(idx), generator=gen) < eps
            a = torch.where(explore, rand_a, greedy).to(self.device)
            actions[idx] = a
            self.env_steps += len(idx)
            step_log.append({"env_step": self.env_steps, "epsilon": float(eps[-1]),
                             "mean_reward": float(env.rewards(idx, a).float().mean())})
        return actions

    def train(self, env: PolicyEnv, epochs: int = C.DRL_EPOCHS, batch_size: int = C.BATCH_SIZE,
              on_epoch: Optional[Callable[[Dict[str, float]], None]] = None) -> Dict[str, List[Dict[str, float]]]:
        """Train for ``epochs`` full passes.

        Args:
            env: Offline environment.
            epochs: Number of DRL epochs.
            batch_size: Minibatch size.
            on_epoch: Callback receiving the epoch log row.

        Returns:
            ``{"epochs": [...], "steps": [...]}`` logs.
        """
        epoch_log: List[Dict[str, float]] = []
        step_log: List[Dict[str, float]] = []
        n_act = len(C.ACTIONS)
        for ep in range(1, epochs + 1):
            # 1) populate the buffer with the full training set
            if C.DRL_BUFFER_MODE == "exhaustive":
                b_idx = torch.arange(env.n, device=self.device).repeat_interleave(n_act)
                b_act = torch.arange(n_act, device=self.device).repeat(env.n)
                self.env_steps += env.n
                step_log.append({"env_step": self.env_steps, "epsilon": float(epsilon_at(self.env_steps)),
                                 "mean_reward": float(env.rewards(b_idx, b_act).float().mean())})
            else:
                b_act = self._collect(env, C.EVAL_BATCH_SIZE, step_log)
                b_idx = torch.arange(env.n, device=self.device)
            capacity = max(C.REPLAY_BUFFER, len(b_idx))
            assert len(b_idx) <= capacity
            b_rew = env.rewards(b_idx, b_act).float()
            behaviour_reward = float(b_rew.mean())
            # 2) one full pass of Double-DQN updates over the buffer
            perm = torch.randperm(len(b_idx), device=self.device)
            losses = []
            self.online.train()
            n_batches = math.ceil(len(perm) / batch_size)
            for bi in range(n_batches):
                sel = perm[bi * batch_size:(bi + 1) * batch_size]
                i, a, r = b_idx[sel], b_act[sel], b_rew[sel]
                s, ns = env.state(i), env.state(env.next_idx[i])
                done = env.done[i]
                with torch.no_grad(), autocast(self.device, self.amp):
                    na = self.online(ns).argmax(dim=1, keepdim=True)
                    tq = self.target(ns).gather(1, na).squeeze(1).float()
                    y = r + C.GAMMA * (1.0 - done) * tq
                with autocast(self.device, self.amp):
                    q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(q.float(), y, beta=C.HUBER_DELTA)
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.online.parameters(), C.GRAD_CLIP_NORM)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.updates += 1
                if self.updates % C.TARGET_UPDATE == 0:
                    self.target.load_state_dict(self.online.state_dict())
                losses.append(float(loss.detach()))
            greedy_r = self.evaluate(env)
            row = {"epoch": ep, "updates": self.updates, "env_steps": self.env_steps, "buffer_size": len(b_idx),
                   "buffer_capacity": capacity, "epsilon": float(epsilon_at(self.env_steps)),
                   "loss": float(np.mean(losses)) if losses else float("nan"),
                   "behaviour_reward": behaviour_reward, **greedy_r}
            epoch_log.append(row)
            log.info(f"[DRL/double_dqn] epoch {ep}/{epochs} loss={row['loss']:.4f} eps={row['epsilon']:.3f} "
                     f"behaviour_R={behaviour_reward:.3f} greedy_R={greedy_r['greedy_reward']:.3f}")
            if on_epoch:
                on_epoch(row)
        return {"epochs": epoch_log, "steps": step_log}

    @torch.no_grad()
    def evaluate(self, env: PolicyEnv, batch: int = C.EVAL_BATCH_SIZE) -> Dict[str, float]:
        """Mean reward and action distribution of the greedy policy over all windows."""
        tot, counts = 0.0, np.zeros(len(C.ACTIONS))
        for s in range(0, env.n, batch):
            idx = torch.arange(s, min(s + batch, env.n), device=self.device)
            a = self.greedy(env.state(idx))
            tot += float(env.rewards(idx, a).float().sum())
            counts += np.bincount(a.cpu().numpy(), minlength=len(C.ACTIONS))
        out = {"greedy_reward": tot / max(env.n, 1)}
        out.update({f"frac_{n}": c / max(env.n, 1) for n, c in zip(C.ACTIONS, counts)})
        return out

    def policy_module(self) -> nn.Module:
        """The network used at inference (online Q-net)."""
        return self.online


class PPOAgent:
    """PPO actor-critic (clipped objective, GAE) over the offline env.

    Args:
        state_dim: Embedding size.
        device: Device string.
        lr: Learning rate.
    """

    algo = "ppo"

    def __init__(self, state_dim: int, device: str, lr: float = C.LR) -> None:
        self.device = device
        self.net = ActorCritic(state_dim).to(device)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=C.WEIGHT_DECAY)
        self.amp = C.USE_AMP and device.startswith("cuda")
        self.scaler = grad_scaler(device, self.amp)
        self.state_dim = state_dim
        self.env_steps = 0
        self.updates = 0

    @torch.no_grad()
    def greedy(self, states: torch.Tensor) -> torch.Tensor:
        """Most probable action for float states."""
        self.net.eval()
        logits, _ = self.net(states.float())
        return logits.argmax(dim=1)

    @torch.no_grad()
    def _rollout(self, env: PolicyEnv, batch: int):
        logp = torch.empty(env.n, device=self.device)
        vals = torch.empty(env.n, device=self.device)
        acts = torch.empty(env.n, dtype=torch.long, device=self.device)
        self.net.eval()
        for s in range(0, env.n, batch):
            idx = torch.arange(s, min(s + batch, env.n), device=self.device)
            logits, v = self.net(env.state(idx))
            dist = torch.distributions.Categorical(logits=logits.float())
            a = dist.sample()
            acts[idx], logp[idx], vals[idx] = a, dist.log_prob(a), v.float()
        self.env_steps += env.n
        rew = env.rewards(torch.arange(env.n, device=self.device), acts).float()
        # GAE over chronologically ordered windows, reset at group ends
        r, v, d = rew.cpu().numpy(), vals.cpu().numpy(), env.done_np
        nv = np.where(d > 0, 0.0, v[env.next_idx_np])
        delta = r + C.GAMMA * nv - v
        adv = np.zeros(env.n, dtype=np.float32)
        last = 0.0
        for t in range(env.n - 1, -1, -1):
            last = delta[t] + C.GAMMA * C.PPO_GAE_LAMBDA * (1.0 - d[t]) * last
            adv[t] = last
        ret = adv + v
        adv_t = torch.as_tensor(adv, device=self.device)
        return acts, logp, (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8), torch.as_tensor(ret, device=self.device), rew

    def train(self, env: PolicyEnv, epochs: int = C.DRL_EPOCHS, batch_size: int = C.BATCH_SIZE,
              on_epoch: Optional[Callable[[Dict[str, float]], None]] = None) -> Dict[str, List[Dict[str, float]]]:
        """Train for ``epochs`` full rollouts, each followed by ``PPO_UPDATE_EPOCHS`` passes."""
        epoch_log: List[Dict[str, float]] = []
        step_log: List[Dict[str, float]] = []
        for ep in range(1, epochs + 1):
            acts, old_logp, adv, ret, rew = self._rollout(env, C.EVAL_BATCH_SIZE)
            step_log.append({"env_step": self.env_steps, "epsilon": float("nan"), "mean_reward": float(rew.mean())})
            losses = []
            self.net.train()
            for _ in range(C.PPO_UPDATE_EPOCHS):
                perm = torch.randperm(env.n, device=self.device)
                for bi in range(math.ceil(env.n / batch_size)):
                    i = perm[bi * batch_size:(bi + 1) * batch_size]
                    with autocast(self.device, self.amp):
                        logits, v = self.net(env.state(i))
                    dist = torch.distributions.Categorical(logits=logits.float())
                    lp = dist.log_prob(acts[i])
                    ratio = torch.exp(lp - old_logp[i])
                    s1 = ratio * adv[i]
                    s2 = torch.clamp(ratio, 1 - C.PPO_CLIP, 1 + C.PPO_CLIP) * adv[i]
                    loss = (-torch.min(s1, s2).mean() + C.PPO_VALUE_COEF * F.mse_loss(v.float(), ret[i])
                            - C.PPO_ENTROPY_COEF * dist.entropy().mean())
                    self.opt.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(self.net.parameters(), C.GRAD_CLIP_NORM)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.updates += 1
                    losses.append(float(loss.detach()))
            greedy_r = DoubleDQNAgent.evaluate(self, env)  # same greedy evaluation routine
            row = {"epoch": ep, "updates": self.updates, "env_steps": self.env_steps, "buffer_size": env.n,
                   "buffer_capacity": env.n, "epsilon": float("nan"), "loss": float(np.mean(losses)),
                   "behaviour_reward": float(rew.mean()), **greedy_r}
            epoch_log.append(row)
            log.info(f"[DRL/ppo] epoch {ep}/{epochs} loss={row['loss']:.4f} "
                     f"behaviour_R={row['behaviour_reward']:.3f} greedy_R={greedy_r['greedy_reward']:.3f}")
            if on_epoch:
                on_epoch(row)
        return {"epochs": epoch_log, "steps": step_log}

    def policy_module(self) -> nn.Module:
        """The network used at inference."""
        return self.net


def make_agent(algo: str, state_dim: int, device: str):
    """Factory for ``"double_dqn"`` / ``"ppo"`` agents."""
    if algo == "double_dqn":
        return DoubleDQNAgent(state_dim, device)
    if algo == "ppo":
        return PPOAgent(state_dim, device)
    raise ValueError(f"Unknown DRL algo {algo!r}; choose from {C.DRL_ALGOS}")


def save_agent(agent, path: Path, extra: Optional[Dict[str, Any]] = None) -> None:
    """Persist the inference network + metadata."""
    payload = {"algo": agent.algo, "state_dim": agent.state_dim, "actions": list(C.ACTIONS),
               "hidden": list(C.DRL_HIDDEN),
               "state_dict": {k: v.detach().cpu() for k, v in agent.policy_module().state_dict().items()}}
    payload.update(extra or {})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


class DRLPolicy(nn.Module):
    """Inference-only wrapper returning greedy action indices.

    Args:
        payload: Dict written by :func:`save_agent`.
    """

    def __init__(self, payload: Dict[str, Any]) -> None:
        super().__init__()
        self.algo = payload["algo"]
        self.actions = list(payload["actions"])
        if self.algo == "double_dqn":
            self.net: nn.Module = QNetwork(payload["state_dim"], len(self.actions), payload["hidden"])
        else:
            self.net = ActorCritic(payload["state_dim"], len(self.actions), payload["hidden"])
        self.net.load_state_dict(payload["state_dict"])
        self.eval()

    @torch.no_grad()
    def act(self, states: torch.Tensor) -> torch.Tensor:
        """Greedy action indices for ``(B, E)`` float states."""
        out = self.net(states.float())
        logits = out[0] if isinstance(out, tuple) else out
        return logits.argmax(dim=1)


def load_policy(path: Path, device: str = "cpu") -> DRLPolicy:
    """Load ``drl_agent.pt`` as a :class:`DRLPolicy`."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return DRLPolicy(payload).to(device)
'''

# ==========================================================================
# src/ensemble.py
# ==========================================================================
FILES['src/ensemble.py'] = r'''"""Weighted soft voting, confidence gate and k-confirmation (spec §6d).

Per window::

    S = w_dnn * P_dnn + w_rf * P_rf            # w_dnn + w_rf = 1
    y_hat = 1 if S >= 0.5 else 0
    valid_attack = (y_hat == 1) and (S >= TAU)
    c_f = min(c_f + 1, K_CONFIRM) if valid_attack else max(0, c_f - 1)
    decision = "BLOCK" if c_f >= K_CONFIRM else ("MONITOR" if c_f > 0 else "PASS")
    if S >= TAU_HARD: decision = "BLOCK"

``c_f`` is a per-flow-key counter. Offline (evaluation) the key is the window
group (source, day, protocol / file); online (Phase 2) it is the 5-tuple /
flow key supplied by the XDP userspace consumer.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as C
from .utils import binary_metrics

DECISIONS: List[str] = ["PASS", "MONITOR", "BLOCK"]
PASS, MONITOR, BLOCK = 0, 1, 2


def ensemble_score(p_dnn: np.ndarray, p_rf: np.ndarray, w_dnn: float) -> np.ndarray:
    """Weighted soft vote ``S = w_dnn*P_dnn + (1-w_dnn)*P_rf``.

    Args:
        p_dnn: TCN attack probabilities.
        p_rf: RF attack probabilities.
        w_dnn: TCN weight in ``[0, 1]``.

    Returns:
        Ensemble scores.
    """
    return w_dnn * np.asarray(p_dnn, dtype=np.float64) + (1.0 - w_dnn) * np.asarray(p_rf, dtype=np.float64)


def decide_step(c_f: int, s: float, tau: float = C.TAU, k: int = C.K_CONFIRM,
                tau_hard: float = C.TAU_HARD) -> Tuple[int, int]:
    """Advance one flow key's confirmation counter.

    Args:
        c_f: Current counter.
        s: Ensemble score of the new window.
        tau: Confidence gate.
        k: Confirmations required.
        tau_hard: Immediate-block threshold.

    Returns:
        ``(new_c_f, decision_code)`` with codes ``PASS=0, MONITOR=1, BLOCK=2``.
    """
    y_hat = s >= C.DECISION_THRESHOLD
    valid = y_hat and s >= tau
    c_f = min(c_f + 1, k) if valid else max(0, c_f - 1)
    d = BLOCK if c_f >= k else (MONITOR if c_f > 0 else PASS)
    if s >= tau_hard:
        d = BLOCK
    return c_f, d


def decide_sequence(scores: np.ndarray, groups: np.ndarray, tau: float = C.TAU, k: int = C.K_CONFIRM,
                    tau_hard: float = C.TAU_HARD) -> Tuple[np.ndarray, np.ndarray]:
    """Run the gate + k-confirmation over chronologically ordered windows.

    The counter resets whenever the group id changes (new flow key).

    Args:
        scores: Ensemble scores in chronological order within each group.
        groups: Group id per window (contiguous runs).
        tau: Confidence gate.
        k: Confirmations.
        tau_hard: Hard block threshold.

    Returns:
        ``(decision_codes, counters)``.
    """
    s_list = np.asarray(scores, dtype=np.float64).tolist()
    g_list = np.asarray(groups).tolist()
    dec = np.empty(len(s_list), dtype=np.int8)
    cnt = np.empty(len(s_list), dtype=np.int16)
    c, prev = 0, None
    thr = C.DECISION_THRESHOLD
    for i, (s, g) in enumerate(zip(s_list, g_list)):
        if g != prev:
            c, prev = 0, g
        if s >= thr and s >= tau:
            c = c + 1 if c < k else k
        elif c > 0:
            c -= 1
        d = BLOCK if c >= k else (MONITOR if c > 0 else PASS)
        if s >= tau_hard:
            d = BLOCK
        dec[i] = d
        cnt[i] = c
    return dec, cnt


class ConfirmationState:
    """Per-flow-key counters for online use (mirrors the Phase-2 BlockManager)."""

    def __init__(self, tau: float = C.TAU, k: int = C.K_CONFIRM, tau_hard: float = C.TAU_HARD) -> None:
        self.tau, self.k, self.tau_hard = tau, k, tau_hard
        self.counters: Dict[str, int] = {}

    def update(self, key: str, s: float) -> Tuple[str, int]:
        """Update ``key`` with score ``s``.

        Args:
            key: Flow key.
            s: Ensemble score.

        Returns:
            ``(decision_name, c_f)``.
        """
        c, d = decide_step(self.counters.get(key, 0), float(s), self.tau, self.k, self.tau_hard)
        self.counters[key] = c
        return DECISIONS[d], c

    def reset(self, key: str) -> None:
        """Forget the counter of ``key``."""
        self.counters.pop(key, None)


def grid_search_weights(p_dnn: np.ndarray, p_rf: np.ndarray, y: np.ndarray,
                        grid: Optional[List[float]] = None) -> Dict[str, object]:
    """Grid-search ``w_dnn`` on inner-validation F1 (tie-break ROC-AUC).

    Args:
        p_dnn: TCN probabilities on inner-val windows.
        p_rf: RF probabilities on the same windows (last flow).
        y: Window labels.
        grid: Candidate ``w_dnn`` values (default ``W_DNN_GRID``).

    Returns:
        Dict with ``table``, ``best_unconstrained``, ``best_non_degenerate``
        and the ``selected`` entry (per ``ENSEMBLE_WEIGHT_SELECTION``).
    """
    grid = C.W_DNN_GRID if grid is None else grid
    rows = []
    for w in grid:
        s = ensemble_score(p_dnn, p_rf, w)
        m = binary_metrics(y, (s >= C.DECISION_THRESHOLD).astype(int), s)
        rows.append({"w_dnn": round(float(w), 4), "w_rf": round(1.0 - float(w), 4),
                     "f1": m["f1"], "roc_auc": m["roc_auc"], "precision": m["precision"],
                     "recall": m["recall"], "fpr": m["fpr"]})

    def key(r: Dict[str, float]) -> Tuple[float, float]:
        auc = r["roc_auc"]
        return (r["f1"], -1.0 if auc != auc else auc)  # NaN-safe

    best_u = max(rows, key=key)
    nd = [r for r in rows if r["w_dnn"] >= C.NON_DEGENERATE_MIN_W]
    best_nd = max(nd, key=key) if nd else best_u
    selected = best_nd if C.ENSEMBLE_WEIGHT_SELECTION == "non_degenerate" else best_u
    return {"table": rows, "best_unconstrained": best_u, "best_non_degenerate": best_nd,
            "selection_rule": C.ENSEMBLE_WEIGHT_SELECTION, "selected": selected,
            "w_dnn": selected["w_dnn"], "w_rf": selected["w_rf"],
            "tau": C.TAU, "k_confirm": C.K_CONFIRM, "tau_hard": C.TAU_HARD}
'''

# ==========================================================================
# src/evaluate.py
# ==========================================================================
FILES['src/evaluate.py'] = r'''"""Final evaluation on the held-out 20 % test split (read exactly once).

``python -m src.evaluate --checkpoint artifacts/tcn_best.pt``

Outputs (``artifacts/eval/``):

* ``metrics.json`` - binary metrics for every ablation variant.
* ``ablation.csv`` - TCN-only, RF-only, ensemble, ensemble+gating,
  ensemble+gating+confirmation (the full Phase-1 decision rule) and
  ensemble+DRL-inference. A variant's positive prediction is ``BLOCK``;
  ``MONITOR`` counts as not blocked.
* ``bootstrap_ci.csv`` - 95 % bootstrap CIs (300 resamples) for P/R/F1/FPR.
* ``per_attack_type.csv`` - mandatory per-attack-type FN breakdown + F1.
* ``tau_k_sweep.csv`` - gate/confirmation sensitivity.
* ``transfer_matrix.csv`` - cross-dataset transfer (only if both datasets
  are present); a lighter RF is trained per source dataset on its training
  flows and tested on each source's test windows (last flow).
* Plots: confusion matrices, per-class F1 bar, TCN training curves, DRL
  reward / epsilon curves, tau-k heatmaps, transfer heatmap.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import config as C
from .ensemble import BLOCK, decide_sequence, ensemble_score
from .utils import binary_metrics, get_logger, load_array, load_json, save_json, set_seed

log = get_logger()


# --------------------------------------------------------------------------- #
# Pure-numpy evaluation (no torch needed)
# --------------------------------------------------------------------------- #
def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def variant_predictions(p_dnn: np.ndarray, p_rf: np.ndarray, w_dnn: float, groups: np.ndarray,
                        drl_actions: Optional[np.ndarray]) -> Dict[str, Dict[str, Any]]:
    """Build predictions (and scores) for every ablation variant.

    Args:
        p_dnn: TCN P(attack).
        p_rf: RF P(attack) on the last flow.
        w_dnn: Selected ensemble weight.
        groups: Test window group ids (chronological, contiguous).
        drl_actions: Greedy DRL action indices or ``None``.

    Returns:
        ``{variant: {"pred": ..., "score": ... | None}}``.
    """
    thr = C.DECISION_THRESHOLD
    s = ensemble_score(p_dnn, p_rf, w_dnn)
    dec, _ = decide_sequence(s, groups, C.TAU, C.K_CONFIRM, C.TAU_HARD)
    out: Dict[str, Dict[str, Any]] = {
        "tcn_only": {"pred": (p_dnn >= thr).astype(int), "score": p_dnn},
        "rf_only": {"pred": (p_rf >= thr).astype(int), "score": p_rf},
        "ensemble": {"pred": (s >= thr).astype(int), "score": s},
        "ensemble+gating": {"pred": ((s >= thr) & (s >= C.TAU)).astype(int), "score": s},
        "ensemble+gating+confirmation": {"pred": (dec == BLOCK).astype(int), "score": s},
    }
    if drl_actions is not None:
        out["ensemble+drl_inference"] = {"pred": (drl_actions == C.ACTIONS.index("block")).astype(int), "score": None}
    return out


def bootstrap_ci(y: np.ndarray, preds: Dict[str, np.ndarray], n_boot: int = C.BOOTSTRAP_RESAMPLES,
                 ci: float = C.BOOTSTRAP_CI, seed: int = C.SEED) -> List[Dict[str, Any]]:
    """Percentile bootstrap CIs for precision, recall, F1 and FPR.

    Args:
        y: Labels.
        preds: ``{variant: predictions}``.
        n_boot: Resamples.
        ci: Confidence level.
        seed: RNG seed.

    Returns:
        Rows ``{variant, metric, point, lo, hi}``.
    """
    rng = np.random.default_rng(seed)
    names = list(preds)
    P = np.stack([preds[n] for n in names]).astype(bool)
    yb = y.astype(bool)
    stats = {m: np.empty((n_boot, len(names))) for m in ("precision", "recall", "f1", "fpr")}
    n = len(y)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yy, pp = yb[idx], P[:, idx]
        tp = (pp & yy).sum(1)
        fp = (pp & ~yy).sum(1)
        fn = (~pp & yy).sum(1)
        tn = (~pp & ~yy).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pr = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            rc = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
            f1 = np.where(pr + rc > 0, 2 * pr * rc / (pr + rc), 0.0)
            fpr = np.where(fp + tn > 0, fp / (fp + tn), 0.0)
        stats["precision"][b], stats["recall"][b], stats["f1"][b], stats["fpr"][b] = pr, rc, f1, fpr
    a = (1 - ci) / 2
    rows = []
    for j, name in enumerate(names):
        point = binary_metrics(y, preds[name])
        for m, arr in stats.items():
            rows.append({"variant": name, "metric": m, "point": point[m],
                         "lo": float(np.quantile(arr[:, j], a)), "hi": float(np.quantile(arr[:, j], 1 - a))})
    return rows


def per_attack_type(y: np.ndarray, types: np.ndarray, preds: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """Per-attack-type breakdown: windows, TP, FN, recall and F1 (type vs BENIGN).

    Args:
        y: Labels.
        types: Attack-type strings per window.
        preds: ``{variant: predictions}``.

    Returns:
        Rows per (variant, attack_type).
    """
    rows = []
    benign = types == "BENIGN"
    for name, p in preds.items():
        p = p.astype(bool)
        for t in sorted(np.unique(types)):
            m = types == t
            if t == "BENIGN":
                fp = int((p & m).sum())
                rows.append({"variant": name, "attack_type": t, "windows": int(m.sum()), "tp": 0, "fn": 0,
                             "fp_on_benign": fp, "recall": float("nan"),
                             "fpr": fp / max(int(m.sum()), 1), "f1_vs_benign": float("nan")})
                continue
            tp = int((p & m & (y == 1)).sum())
            fn = int((~p & m & (y == 1)).sum())
            sub = m | benign
            f1 = binary_metrics(y[sub], p[sub].astype(int))["f1"]
            rows.append({"variant": name, "attack_type": t, "windows": int(m.sum()), "tp": tp, "fn": fn,
                         "fp_on_benign": "", "recall": tp / max(tp + fn, 1), "fpr": "", "f1_vs_benign": f1})
    return rows


def tau_k_sweep(y: np.ndarray, s: np.ndarray, groups: np.ndarray) -> List[Dict[str, Any]]:
    """Sweep gate ``tau`` and confirmations ``k`` for the BLOCK decision.

    Args:
        y: Labels.
        s: Ensemble scores.
        groups: Window group ids.

    Returns:
        Rows ``{tau, k, precision, recall, f1, fpr, ...}``.
    """
    rows = []
    for tau in C.TAU_SWEEP:
        for k in C.K_SWEEP:
            dec, _ = decide_sequence(s, groups, tau, k, C.TAU_HARD)
            m = binary_metrics(y, (dec == BLOCK).astype(int))
            rows.append({"tau": tau, "k": k, **{x: m[x] for x in ("precision", "recall", "f1", "fpr", "accuracy")}})
    return rows


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_confusions(y: np.ndarray, preds: Dict[str, np.ndarray], out: Path) -> None:
    """Grid of 2x2 confusion matrices (one per variant)."""
    plt = _plt()
    n = len(preds)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows), squeeze=False)
    for ax, (name, p) in zip(axes.ravel(), preds.items()):
        m = binary_metrics(y, p)
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["benign", "attack"])
        ax.set_yticks([0, 1], ["benign", "attack"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{name}\nF1={m['f1']:.4f}", fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=140)
    plt.close(fig)


def plot_per_class_f1(rows: List[Dict[str, Any]], variant: str, out: Path) -> None:
    """Horizontal bar chart of per-attack-type F1 (type vs BENIGN) for one variant."""
    plt = _plt()
    r = [x for x in rows if x["variant"] == variant and x["attack_type"] != "BENIGN"]
    if not r:
        return
    r.sort(key=lambda x: x["f1_vs_benign"])
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(r) + 1.5))
    ax.barh([x["attack_type"] for x in r], [x["f1_vs_benign"] for x in r], color="#3b6ea5")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 (attack type vs BENIGN)")
    ax.set_title(f"Per-attack-type F1 - {variant}")
    fig.tight_layout()
    fig.savefig(out / "per_class_f1.png", dpi=140)
    plt.close(fig)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def plot_training_curves(out: Path) -> None:
    """TCN loss / F1 / LR curves from ``train_log.csv``."""
    rows = [r for r in _read_csv(C.artifact_path("train_log")) if r.get("stage") == "tcn"]
    if not rows:
        return
    plt = _plt()
    ep = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].plot(ep, [_f(r["train_loss"]) for r in rows], label="train")
    ax[0].plot(ep, [_f(r["val_loss"]) for r in rows], label="inner-val")
    ax[0].set_title("TCN loss")
    ax[0].legend()
    ax[1].plot(ep, [_f(r["val_f1"]) for r in rows], label="F1")
    ax[1].plot(ep, [_f(r["val_auc"]) for r in rows], label="ROC-AUC")
    ax[1].set_title("Inner-val metrics")
    ax[1].legend()
    ax[2].plot(ep, [_f(r["lr"]) for r in rows])
    ax[2].set_title("Learning rate")
    for a in ax:
        a.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=140)
    plt.close(fig)


def plot_drl_curves(out: Path) -> None:
    """DRL reward / epsilon curves from ``drl_log.csv`` and ``drl_steps.csv``."""
    ep_rows = _read_csv(C.artifact_path("drl_log"))
    st_rows = _read_csv(C.ARTIFACT_DIR / "drl_steps.csv")
    if not ep_rows and not st_rows:
        return
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    if ep_rows:
        e = [int(r["epoch"]) for r in ep_rows]
        ax[0].plot(e, [_f(r["behaviour_reward"]) for r in ep_rows], label="behaviour policy")
        ax[0].plot(e, [_f(r["greedy_reward"]) for r in ep_rows], label="greedy policy")
        ax[0].set_xlabel("DRL epoch")
        ax[0].set_ylabel("mean reward / window")
        ax[0].legend()
        ax[0].set_title("DRL reward")
    if st_rows:
        s = [_f(r["env_step"]) for r in st_rows]
        ax[1].plot(s, [_f(r["epsilon"]) for r in st_rows], label="epsilon")
        ax[1].plot(s, [_f(r["mean_reward"]) for r in st_rows], label="chunk mean reward", alpha=0.6)
        ax[1].set_xlabel("environment step")
        ax[1].legend()
        ax[1].set_title("Epsilon schedule & collection reward")
    fig.tight_layout()
    fig.savefig(out / "drl_reward_epsilon.png", dpi=140)
    plt.close(fig)


def plot_heatmap(mat: np.ndarray, rows: List[str], cols: List[str], title: str, path: Path, fmt: str = "{:.3f}") -> None:
    """Annotated heatmap helper."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 2.5, 0.45 * len(rows) + 1.8))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Cross-dataset transfer
# --------------------------------------------------------------------------- #
def transfer_matrix(test_last_flow: np.ndarray, y_test: np.ndarray, test_source: np.ndarray) -> List[Dict[str, Any]]:
    """Train-on-A / test-on-B matrix with a lighter (uncalibrated) RF per source.

    Args:
        test_last_flow: ``(N, F)`` last flow of every test window.
        y_test: Test window labels.
        test_source: Source dataset per test window.

    Returns:
        Rows ``{train_source, test_source, f1, recall, precision, fpr}``; empty if
        fewer than two sources are present.
    """
    from .rf_branch import fit_rf, predict_attack_proba

    fsrc = np.load(C.artifact_path("flows_train_source"))
    sources = sorted(set(np.unique(fsrc)) & set(np.unique(test_source)))
    if len(sources) < 2:
        log.info("Cross-dataset transfer matrix skipped (needs both CICIDS2017 and CICIDS2018).")
        return []
    Xf = np.load(C.artifact_path("X_flows_train"), mmap_mode="r")
    yf = np.load(C.artifact_path("y_flows_train"))
    rows = []
    for a in sources:
        idx = np.flatnonzero(fsrc == a)
        rf = fit_rf(Xf[idx], yf[idx], n_estimators=C.TRANSFER_RF_N_ESTIMATORS, calibrated=False)
        for b in sources:
            m = test_source == b
            p = predict_attack_proba(rf, test_last_flow[m])
            mt = binary_metrics(y_test[m], (p >= C.DECISION_THRESHOLD).astype(int), p)
            rows.append({"train_source": a, "test_source": b, **{k: mt[k] for k in
                         ("f1", "precision", "recall", "fpr", "roc_auc")}})
            log.info(f"[transfer] train {a} -> test {b}: F1={mt['f1']:.4f} recall={mt['recall']:.4f}")
        del rf
    return rows


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def report(y: np.ndarray, types: np.ndarray, groups: np.ndarray, sources: np.ndarray, p_dnn: np.ndarray,
           p_rf: np.ndarray, w_dnn: float, drl_actions: Optional[np.ndarray], last_flow_X: Optional[np.ndarray],
           out: Path, do_transfer: bool = True, plots: bool = True) -> Dict[str, Any]:
    """Compute every metric/table/plot from cached test predictions.

    Args:
        y: Test labels.
        types: Attack type per window.
        groups: Window groups.
        sources: Source per window.
        p_dnn: TCN probabilities.
        p_rf: RF probabilities.
        w_dnn: Ensemble weight.
        drl_actions: DRL actions or ``None``.
        last_flow_X: Last flow of every test window (for the transfer matrix).
        out: Output directory.
        do_transfer: Compute the cross-dataset matrix.
        plots: Render plots.

    Returns:
        ``metrics.json`` content.
    """
    out.mkdir(parents=True, exist_ok=True)
    variants = variant_predictions(p_dnn, p_rf, w_dnn, groups, drl_actions)
    metrics = {name: binary_metrics(y, v["pred"], v["score"]) for name, v in variants.items()}
    ablation = [{"variant": k, **{m: v.get(m, float("nan")) for m in
                 ("accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc", "tp", "fp", "tn", "fn")}}
                for k, v in metrics.items()]
    _write_csv(ablation, out / "ablation.csv")
    preds = {k: v["pred"] for k, v in variants.items()}
    ci_rows = bootstrap_ci(y, preds)
    _write_csv(ci_rows, out / "bootstrap_ci.csv")
    pat = per_attack_type(y, types, preds)
    _write_csv(pat, out / "per_attack_type.csv")
    s = ensemble_score(p_dnn, p_rf, w_dnn)
    sweep = tau_k_sweep(y, s, groups)
    _write_csv(sweep, out / "tau_k_sweep.csv")
    transfer: List[Dict[str, Any]] = []
    if do_transfer and last_flow_X is not None:
        transfer = transfer_matrix(last_flow_X, y, sources)
        _write_csv(transfer, out / "transfer_matrix.csv")
    per_source = {}
    for src in sorted(np.unique(sources)):
        m = sources == src
        per_source[str(src)] = binary_metrics(y[m], variants["ensemble+gating+confirmation"]["pred"][m], s[m])
    result = {"n_test_windows": int(len(y)), "w_dnn": w_dnn, "tau": C.TAU, "k_confirm": C.K_CONFIRM,
              "tau_hard": C.TAU_HARD, "variants": metrics, "per_source_full_system": per_source,
              "headline_variant": "ensemble+gating+confirmation"}
    save_json(result, out / "metrics.json")
    if plots:
        plot_confusions(y, preds, out)
        plot_per_class_f1(pat, "ensemble+gating+confirmation", out)
        plot_training_curves(out)
        plot_drl_curves(out)
        f1m = np.array([[r["f1"] for r in sweep if r["tau"] == t] for t in C.TAU_SWEEP])
        fprm = np.array([[r["fpr"] for r in sweep if r["tau"] == t] for t in C.TAU_SWEEP])
        rl, cl = [f"tau={t:.2f}" for t in C.TAU_SWEEP], [f"k={k}" for k in C.K_SWEEP]
        plot_heatmap(f1m, rl, cl, "BLOCK F1 vs (tau, k)", out / "tau_k_f1.png")
        plot_heatmap(fprm, rl, cl, "BLOCK FPR vs (tau, k)", out / "tau_k_fpr.png", fmt="{:.4f}")
        if transfer:
            srcs = sorted({r["train_source"] for r in transfer})
            tm = np.array([[next(r["f1"] for r in transfer if r["train_source"] == a and r["test_source"] == b)
                            for b in srcs] for a in srcs])
            plot_heatmap(tm, [f"train {a}" for a in srcs], [f"test {b}" for b in srcs],
                         "Cross-dataset transfer F1 (RF)", out / "transfer_matrix.png")
    _print_table(ablation, ci_rows)
    return result


def _print_table(ablation: List[Dict[str, Any]], ci_rows: List[Dict[str, Any]]) -> None:
    log.info("=" * 100)
    log.info(f"{'variant':<32}{'acc':>8}{'prec':>8}{'recall':>8}{'F1':>8}{'FPR':>9}{'ROC-AUC':>9}{'PR-AUC':>9}")
    for r in ablation:
        log.info(f"{r['variant']:<32}{r['accuracy']:>8.4f}{r['precision']:>8.4f}{r['recall']:>8.4f}{r['f1']:>8.4f}"
                 f"{r['fpr']:>9.5f}{r['roc_auc']:>9.4f}{r['pr_auc']:>9.4f}")
    log.info("95% bootstrap CI (full system = ensemble+gating+confirmation):")
    for r in ci_rows:
        if r["variant"] == "ensemble+gating+confirmation":
            log.info(f"  {r['metric']:<10} {r['point']:.4f}  [{r['lo']:.4f}, {r['hi']:.4f}]")
    log.info("=" * 100)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI: ``python -m src.evaluate --checkpoint artifacts/tcn_best.pt``."""
    import torch

    from .drl_agent import load_policy
    from .rf_branch import last_flow, load_rf, predict_attack_proba
    from .tcn import infer, load_checkpoint
    from .utils import resolve_device

    p = argparse.ArgumentParser(description="Evaluate Phase-1 models on the 20 % test split (read once).")
    p.add_argument("--checkpoint", default=str(C.artifact_path("tcn")))
    p.add_argument("--device", default=C.DEVICE)
    p.add_argument("--skip-transfer", action="store_true", help="Skip the cross-dataset transfer matrix.")
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args(argv)
    set_seed(C.SEED)
    device = resolve_device(a.device)

    for key in ("rf", "ensemble_weights", "X_test", "y_test", "test_attack_type", "test_group", "test_source"):
        if not C.artifact_path(key).exists():
            log.error(f"Missing {C.artifact_path(key)} - run preprocess and train first.")
            raise SystemExit(2)
    marker = C.EVAL_DIR / "test_access_log.json"
    if marker.exists():
        prev = load_json(marker)
        log.warning(f"The test split was already evaluated at {prev.get('last')}. Do not tune on test results.")

    model, _ = load_checkpoint(Path(a.checkpoint), device)
    rf = load_rf(C.artifact_path("rf"))
    ew = load_json(C.artifact_path("ensemble_weights"))
    drl = load_policy(C.artifact_path("drl"), device) if C.artifact_path("drl").exists() else None

    # ---- the single read of the 20 % test split ----
    X_test = load_array(C.artifact_path("X_test"))
    y = np.load(C.artifact_path("y_test"))
    types = np.load(C.artifact_path("test_attack_type"))
    groups = np.load(C.artifact_path("test_group"))
    sources = np.load(C.artifact_path("test_source"))
    log.info(f"Test split loaded once: {len(y):,} windows on device={device}")
    save_json({"last": time.strftime("%Y-%m-%d %H:%M:%S"), "n_windows": int(len(y))}, marker)

    p_dnn, emb = infer(model, X_test, device, return_embedding=drl is not None)
    lf = last_flow(X_test)
    p_rf = predict_attack_proba(rf, lf)
    drl_actions = None
    if drl is not None:
        acts = []
        with torch.no_grad():
            for s in range(0, len(emb), C.EVAL_BATCH_SIZE):
                z = torch.from_numpy(emb[s:s + C.EVAL_BATCH_SIZE].astype(np.float32)).to(device)
                acts.append(drl.act(z).cpu().numpy())
        drl_actions = np.concatenate(acts)
    report(y, types, groups, sources, p_dnn, p_rf, float(ew["w_dnn"]), drl_actions, lf, C.EVAL_DIR,
           do_transfer=not a.skip_transfer, plots=not a.no_plots)
    log.info(f"Evaluation artifacts written to {C.EVAL_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
'''

# ==========================================================================
# src/demo_replay.py
# ==========================================================================
FILES['src/demo_replay.py'] = r'''"""Attack-stopping replay demo (Phase 1, no packets touched).

Replays held-out test windows of one flow group in chronological order
through :class:`IDSInference` - exactly the call the Phase-2 XDP userspace
consumer will make - and shows, window by window, how the system goes from
PASS -> MONITOR -> BLOCK when an attack starts, when ``block_ip`` is issued,
and how long it took.

Usage::

    python -m src.demo_replay --list                       # attack types available
    python -m src.demo_replay --attack "DDoS attacks-LOIC-HTTP"
    python -m src.demo_replay --attack PortScan --before 30 --after 120
    python -m src.demo_replay --all                        # time-to-block summary for every attack type

Each test window holds the last ``WINDOW_SIZE`` flows of a flow group and
consecutive windows advance by ``STRIDE`` flows, i.e. one decision every
``STRIDE`` new flows - the same cadence Phase 2 will use. Windows are stored
standardised, so they are converted back to raw feature values with the saved
scaler before being passed to the API (the API re-scales them, exactly as it
will for live XDP features).

Output: a coloured terminal timeline, a summary, and a plot under
``artifacts/demo/``. The test split is only *replayed* for illustration here;
it is not used to tune anything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from . import config as C
from .utils import get_logger, save_json

log = get_logger()
DEMO_DIR = C.ARTIFACT_DIR / "demo"
_COL = {"PASS": "\033[32m", "MONITOR": "\033[33m", "BLOCK": "\033[31;1m", "END": "\033[0m"}


def load_test_meta() -> Dict[str, np.ndarray]:
    """Load the small test-split metadata arrays (labels, types, groups)."""
    for k in ("X_test", "y_test", "test_attack_type", "test_group"):
        if not C.artifact_path(k).exists():
            raise SystemExit(f"Missing {C.artifact_path(k)} - run preprocess/train first.")
    return {
        "y": np.load(C.artifact_path("y_test")),
        "types": np.load(C.artifact_path("test_attack_type")),
        "groups": np.load(C.artifact_path("test_group")),
    }


def attack_segments(meta: Dict[str, np.ndarray], attack: str) -> List[Dict[str, int]]:
    """Find contiguous runs of windows labelled ``attack`` inside one group.

    Args:
        meta: Output of :func:`load_test_meta`.
        attack: Attack-type string (as in ``y_test_attack_type.npy``).

    Returns:
        List of ``{"group", "start", "end"}`` (``end`` exclusive), longest first.
    """
    m = meta["types"] == attack
    if not m.any():
        return []
    idx = np.flatnonzero(m)
    breaks = np.flatnonzero((np.diff(idx) != 1) | (np.diff(meta["groups"][idx]) != 0)) + 1
    segs = []
    for a, b in zip(np.concatenate([[0], breaks]), np.concatenate([breaks, [len(idx)]])):
        segs.append({"group": int(meta["groups"][idx[a]]), "start": int(idx[a]), "end": int(idx[b - 1]) + 1})
    return sorted(segs, key=lambda s: s["end"] - s["start"], reverse=True)


def to_raw(windows_scaled: np.ndarray, scaler: Any) -> np.ndarray:
    """Undo standardisation so windows look like raw XDP feature records."""
    return windows_scaled * scaler.scale_.astype(np.float32) + scaler.mean_.astype(np.float32)


def replay(ids: Any, X: np.ndarray, meta: Dict[str, np.ndarray], scaler: Any, start: int, end: int,
           flow_key: str, verbose: bool = True) -> Dict[str, Any]:
    """Replay windows ``[start, end)`` of one group through ``ids.predict_window``.

    Args:
        ids: Object with ``predict_window(features_2d, flow_key)``, ``block_ip``,
            ``is_blocked`` and ``reset_flow_state`` (an :class:`IDSInference`).
        X: Scaled test windows (memmap).
        meta: Test metadata.
        scaler: Fitted scaler (to rebuild raw features).
        start: First window index.
        end: End window index (exclusive).
        flow_key: Key used for the confirmation counter and the block table.
        verbose: Print the coloured timeline.

    Returns:
        Dict with per-window records and summary statistics.
    """
    ids.reset_flow_state(flow_key)
    rows: List[Dict[str, Any]] = []
    first_attack: Optional[int] = None
    first_block: Optional[int] = None
    if verbose:
        log.info(f"{'step':>5} {'flows':>11} {'truth':<26} {'p_dnn':>6} {'p_rf':>6} {'S':>6} {'c_f':>3}  decision")
    for i in range(start, end):
        raw = to_raw(np.asarray(X[i], dtype=np.float32), scaler)
        out = ids.predict_window(raw, flow_key=flow_key)
        truth = str(meta["types"][i])
        is_att = int(meta["y"][i]) == 1
        step = i - start
        if is_att and first_attack is None:
            first_attack = step
        event = ""
        if out["decision"] == "BLOCK":
            if first_attack is not None and first_block is None:
                first_block = step
            if not ids.is_blocked(flow_key):
                ids.block_ip(flow_key)
                event = "  <-- block_ip() issued, TTL %ds" % C.BLOCK_TTL_SECONDS
        rows.append({"step": step, "window": i, "truth": truth, "is_attack": int(is_att), **{
            k: out[k] for k in ("p_dnn", "p_rf", "s_ensemble", "consecutive_attacks", "decision", "drl_action")}})
        if verbose:
            d = out["decision"]
            flows = f"{step * C.STRIDE + 1}-{step * C.STRIDE + C.WINDOW_SIZE}"
            log.info(f"{step:>5} {flows:>11} {truth[:26]:<26} {out['p_dnn']:>6.3f} {out['p_rf']:>6.3f} "
                     f"{out['s_ensemble']:>6.3f} {out['consecutive_attacks']:>3}  {_COL[d]}{d:<7}{_COL['END']}{event}")
    att = [r for r in rows if r["is_attack"]]
    ben = [r for r in rows if not r["is_attack"]]
    summary = {
        "flow_key": flow_key, "windows": len(rows), "attack_windows": len(att), "benign_windows": len(ben),
        "attack_blocked": sum(r["decision"] == "BLOCK" for r in att),
        "attack_monitored": sum(r["decision"] == "MONITOR" for r in att),
        "attack_passed": sum(r["decision"] == "PASS" for r in att),
        "benign_blocked": sum(r["decision"] == "BLOCK" for r in ben),
        "first_attack_step": first_attack, "first_block_step": first_block,
        "windows_to_block": None if first_block is None or first_attack is None else first_block - first_attack,
    }
    if summary["windows_to_block"] is not None:
        summary["flows_to_block"] = summary["windows_to_block"] * C.STRIDE + C.WINDOW_SIZE
    return {"rows": rows, "summary": summary}


def plot_replay(result: Dict[str, Any], title: str, path: Path) -> None:
    """Plot ensemble score, thresholds, decisions and the c_f counter."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result["rows"]
    x = np.array([r["step"] for r in rows])
    s = np.array([r["s_ensemble"] for r in rows])
    att = np.array([r["is_attack"] for r in rows], dtype=bool)
    dec = np.array([r["decision"] for r in rows])
    cf = np.array([r["consecutive_attacks"] for r in rows])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for a in (a1, a2):
        a.fill_between(x, 0, 1, where=att, transform=a.get_xaxis_transform(), color="#f4c7c3", alpha=0.6,
                       step="mid", label="true attack windows" if a is a1 else None)
    a1.plot(x, s, color="#333", lw=1.2, label="ensemble score S")
    a1.axhline(C.TAU, color="#d98c00", ls="--", lw=1, label=f"tau = {C.TAU}")
    a1.axhline(C.TAU_HARD, color="#b00020", ls=":", lw=1, label=f"tau_hard = {C.TAU_HARD}")
    colors = {"PASS": "#2e7d32", "MONITOR": "#f9a825", "BLOCK": "#c62828"}
    for d, c in colors.items():
        m = dec == d
        a1.scatter(x[m], s[m], s=14, color=c, zorder=3, label=d)
    a1.set_ylim(-0.02, 1.02)
    a1.set_ylabel("score")
    a1.set_title(title)
    a1.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    a2.step(x, cf, where="mid", color="#1565c0")
    a2.axhline(C.K_CONFIRM, color="#c62828", ls="--", lw=1)
    a2.set_ylabel("c_f")
    a2.set_xlabel(f"decision step (one step = {C.STRIDE} new flows)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _make_ids(device: str):
    from .inference_api import IDSInference

    return IDSInference(C.ARTIFACT_DIR, device=device)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Replay test traffic through IDSInference and watch attacks get blocked.")
    p.add_argument("--list", action="store_true", help="List attack types in the test split.")
    p.add_argument("--attack", help="Attack type to replay (see --list).")
    p.add_argument("--all", action="store_true", help="Replay the longest segment of every attack type (summary only).")
    p.add_argument("--before", type=int, default=20, help="Benign/context windows before the attack starts.")
    p.add_argument("--after", type=int, default=60, help="Attack windows to show after it starts.")
    p.add_argument("--device", default=C.DEVICE)
    a = p.parse_args(argv)

    meta = load_test_meta()
    if a.list or not (a.attack or a.all):
        types, counts = np.unique(meta["types"], return_counts=True)
        log.info("Attack types in the test split (windows):")
        for t, c in sorted(zip(types, counts), key=lambda z: -z[1]):
            if t != "BENIGN":
                log.info(f"  {c:>8,}  {t}")
        log.info('Replay one with: python -m src.demo_replay --attack "<name>"')
        return

    X = np.load(C.artifact_path("X_test"), mmap_mode="r")
    scaler = joblib.load(C.artifact_path("scaler"))
    ids = _make_ids(a.device)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    targets = [str(t) for t in np.unique(meta["types"]) if t != "BENIGN"] if a.all else [a.attack]
    summaries = []
    for attack in targets:
        segs = attack_segments(meta, attack)
        if not segs:
            log.error(f"No test windows labelled {attack!r}. Use --list to see valid names.")
            continue
        seg = segs[0]
        g = meta["groups"]
        lo = seg["start"]
        while lo > 0 and g[lo - 1] == seg["group"] and seg["start"] - lo < a.before:
            lo -= 1
        hi = min(seg["end"], seg["start"] + a.after)
        key = f"group-{seg['group']}"
        log.info("=" * 100)
        log.info(f"Replaying {attack!r}: flow group {seg['group']}, windows {lo}..{hi - 1} "
                 f"({seg['start'] - lo} before the attack, {hi - seg['start']} attack windows shown)")
        res = replay(ids, X, meta, scaler, lo, hi, key, verbose=not a.all)
        sm = {"attack": attack, **res["summary"]}
        summaries.append(sm)
        safe = "".join(ch if ch.isalnum() else "_" for ch in attack)
        plot_replay(res, f"{attack} - flow group {seg['group']}", DEMO_DIR / f"replay_{safe}.png")
        if sm["windows_to_block"] is None:
            log.info(f"RESULT: attack NOT blocked in the shown windows "
                     f"(blocked {sm['attack_blocked']}/{sm['attack_windows']}, monitored {sm['attack_monitored']})")
        else:
            log.info(f"RESULT: first BLOCK {sm['windows_to_block']} decision steps after the attack started "
                     f"(~{sm['flows_to_block']} flows). Attack windows blocked {sm['attack_blocked']}/"
                     f"{sm['attack_windows']}, benign windows blocked {sm['benign_blocked']}/{sm['benign_windows']}.")
    if summaries:
        save_json(summaries, DEMO_DIR / "replay_summary.json")
        if a.all:
            log.info("=" * 100)
            log.info(f"{'attack type':<28}{'blocked':>12}{'steps to block':>16}{'benign blocked':>16}")
            for s in summaries:
                tb = "never" if s["windows_to_block"] is None else str(s["windows_to_block"])
                log.info(f"{s['attack'][:27]:<28}{s['attack_blocked']:>6}/{s['attack_windows']:<5}{tb:>16}"
                         f"{s['benign_blocked']:>10}/{s['benign_windows']}")
        log.info(f"Plots and summary written to {DEMO_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
'''

# ==========================================================================
# src/utils.py
# ==========================================================================
FILES['src/utils.py'] = r'''"""Shared helpers: seeding, device resolution, logging, JSON IO, metrics."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import numpy as np

from . import config as C

_LOGGERS: Dict[str, logging.Logger] = {}


def get_logger(name: str = "ids") -> logging.Logger:
    """Return a configured stdout logger (idempotent).

    Args:
        name: Logger name.

    Returns:
        A :class:`logging.Logger` writing ``time | level | msg`` lines.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
    logger.propagate = False
    _LOGGERS[name] = logger
    return logger


def set_seed(seed: int = C.SEED, deterministic: bool = C.DETERMINISTIC) -> None:
    """Seed python, numpy and (if present) torch / CUDA; configure cuDNN.

    ``cudnn.benchmark=True`` and ``cudnn.deterministic=True`` are both set as
    the spec requires; with ``deterministic=True`` cuDNN restricts itself to
    deterministic kernels, so benchmarking only picks among those.

    Args:
        seed: Global seed.
        deterministic: Whether to request deterministic cuDNN kernels.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = bool(deterministic)
    except ImportError:
        pass


def resolve_device(requested: Optional[str] = None) -> str:
    """Resolve the compute device with automatic CPU fallback.

    Args:
        requested: ``"cuda"``, ``"cpu"``, ``"cuda:N"`` or ``None`` (config default).

    Returns:
        The device string actually usable.
    """
    import torch

    dev = (requested or C.DEVICE).lower()
    if dev.startswith("cuda") and not torch.cuda.is_available():
        get_logger().warning("CUDA requested but torch.cuda.is_available() is False -> falling back to CPU.")
        return "cpu"
    return dev


def log_device(device: str) -> None:
    """Log the device name and current / total VRAM.

    Args:
        device: Device string returned by :func:`resolve_device`.
    """
    import torch

    log = get_logger()
    if device.startswith("cuda"):
        idx = torch.device(device).index or 0
        props = torch.cuda.get_device_properties(idx)
        alloc = torch.cuda.memory_allocated(idx) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(idx) / 1024 ** 3
        log.info(
            f"Device: {device} ({props.name}) | VRAM total {props.total_memory / 1024 ** 3:.2f} GB | "
            f"allocated {alloc:.2f} GB | reserved {reserved:.2f} GB"
        )
    else:
        log.info("Device: cpu (AMP disabled, pin_memory disabled)")


OOM_HELP = (
    "CUDA out of memory. Remediation: rerun with a smaller batch, e.g. "
    "`python -m src.train --batch-size 128` (or 64), keep AMP enabled, and close "
    "other GPU processes (check `nvidia-smi`)."
)


def save_json(obj: Any, path: Path) -> None:
    """Write ``obj`` as pretty JSON, creating parent folders.

    Args:
        obj: JSON-serialisable object.
        path: Destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: Path) -> Any:
    """Read a JSON file.

    Args:
        path: Source path.

    Returns:
        Parsed object.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def load_array(path: Path, max_in_memory: int = C.MAX_IN_MEMORY_BYTES) -> np.ndarray:
    """Load a ``.npy`` file, memory-mapping it when it is larger than ``max_in_memory``.

    Args:
        path: ``.npy`` path.
        max_in_memory: Byte threshold for copying into RAM.

    Returns:
        ``np.ndarray`` or ``np.memmap``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact {path}. Run `python -m src.preprocess --force` first.")
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.nbytes <= max_in_memory:
        return np.asarray(arr)
    return arr


def batched(n: int, batch: int) -> Iterator[slice]:
    """Yield contiguous slices covering ``range(n)``.

    Args:
        n: Total length.
        batch: Slice size.

    Yields:
        ``slice`` objects.
    """
    for s in range(0, n, batch):
        yield slice(s, min(s + batch, n))


class Timer:
    """Context manager measuring wall time in seconds."""

    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self.t0


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute binary detection metrics (attack = positive class 1).

    Args:
        y_true: Ground-truth labels {0,1}.
        y_pred: Predicted labels {0,1}.
        scores: Optional continuous attack scores for ROC-AUC / PR-AUC.

    Returns:
        Dict with accuracy, precision, recall, f1, fpr, tp, fp, tn, fn and,
        when ``scores`` is given and both classes are present, roc_auc / pr_auc.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n = max(tp + fp + tn + fn, 1)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    out: Dict[str, float] = {
        "accuracy": (tp + tn) / n, "precision": prec, "recall": rec, "f1": f1, "fpr": fpr,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
    if scores is not None:
        both = len(np.unique(y_true)) == 2
        out["roc_auc"] = float(roc_auc_score(y_true, scores)) if both else float("nan")
        out["pr_auc"] = float(average_precision_score(y_true, scores)) if both else float("nan")
    return out


def iter_progress(it: Iterable, total: Optional[int] = None, desc: str = "") -> Iterable:
    """Wrap an iterable with tqdm if installed, else return it unchanged.

    Args:
        it: Iterable.
        total: Optional length.
        desc: Progress-bar label.

    Returns:
        Possibly-wrapped iterable.
    """
    try:
        from tqdm.auto import tqdm

        return tqdm(it, total=total, desc=desc, leave=False, mininterval=2.0)
    except ImportError:
        return it


def autocast(device: str, enabled: bool):
    """Return a ``torch.autocast`` context that is only active on CUDA.

    Args:
        device: Device string.
        enabled: Whether AMP is requested.

    Returns:
        ``torch.autocast`` context manager (no-op on CPU).
    """
    import torch

    is_cuda = device.startswith("cuda")
    return torch.autocast(device_type="cuda" if is_cuda else "cpu", enabled=bool(enabled and is_cuda))


def grad_scaler(device: str, enabled: bool):
    """Return a ``GradScaler`` enabled only for CUDA + AMP.

    Args:
        device: Device string.
        enabled: Whether AMP is requested.

    Returns:
        ``torch.amp.GradScaler``.
    """
    import torch

    on = bool(enabled and device.startswith("cuda"))
    try:
        return torch.amp.GradScaler("cuda", enabled=on)
    except (AttributeError, TypeError):  # pragma: no cover - torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=on)
'''

# ==========================================================================
# requirements.txt
# ==========================================================================
FILES['requirements.txt'] = r'''pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
torch>=2.2
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
tqdm>=4.65
pyyaml>=6.0
tensorboard>=2.14
'''

# ==========================================================================
# README.md
# ==========================================================================
FILES['README.md'] = r'''# Sub-Microsecond Intrusion Prevention — Phase 1 (Detection Brain)

**Sub-Microsecond Intrusion Prevention: An eBPF-Driven Deep Reinforcement Learning and TCN Framework for High-Speed Networks**

This repository is **Phase 1**: offline preprocessing and training of the detection brain on
CICIDS2017 + CICIDS2018. It covers a TCN encoder, an isotonic-calibrated RandomForest, a
weighted ensemble with a confidence gate and k-confirmation, and a DRL response policy
(Double-DQN by default, PPO optional). It also exposes a stable `IDSInference` API that the
Phase-2 eBPF/XDP data plane will call **without retraining**. See [PHASE.md](PHASE.md).

> Phase 1 makes **no latency claim**. The sub-microsecond figure is for the Phase-2 XDP fast
> path and will be measured there.

## 1. Install

```bash
pip install -r requirements.txt
```

You need Python 3.10 or newer. CUDA is used automatically when `torch.cuda.is_available()`;
otherwise every stage falls back to CPU, with AMP and `pin_memory` turned off.

## 2. Where to put the datasets

```
data/cicids2017/   <- the 8 MachineLearningCVE CSVs (e.g. Monday-WorkingHours.pcap_ISCX.csv ...)
data/cicids2018/   <- the 10 CSE-CIC-IDS2018 "Processed Traffic Data" CSVs (02-14-2018.csv ...)
```

Both folders are searched recursively for `*.csv`, so sub-folders are fine. Either dataset may be
missing (that one is then skipped with a warning). If both are empty, preprocessing stops with
an explanation. Nothing is downloaded.

## 3. Run

```bash
python -m src.preprocess --force
python -m src.train --algo double_dqn --epochs 30          # add --device cpu to force CPU
python -m src.evaluate --checkpoint artifacts/tcn_best.pt
python -m src.inference_api --self-test
```

`--self-test` needs no dataset. If trained artifacts are missing, it runs in *synthetic mode*
with random weights, which only checks the interface.

### Useful flags
| Command | Flag | Meaning |
|---|---|---|
| preprocess | `--max-rows-per-file N` | debug-only row cap (default: all rows) |
| train | `--batch-size 128` | the OOM remedy |
| train | `--algo ppo` | PPO instead of Double-DQN |
| train | `--rf-selection-mode final` | reuse the final RF for weight search (faster, biased) |
| train | `--subsample-fraction 0.1` | **opt-in** debug subsample, logged loudly |
| train | `--no-amp`, `--num-workers 0`, `--no-tensorboard` | troubleshooting |
| evaluate | `--skip-transfer`, `--no-plots` | skip the cross-dataset matrix / plots |

## 4. What each step does

**Step 1 — `src/preprocess.py`**
- The canonical columns come from `src/schema.py`. The duplicate 2017 `Fwd Header Length`
  column is dropped (first kept), and 2018 names are mapped to 2017 names. 2018 `Timestamp`
  and `Protocol` are kept as metadata only, never as features.
- Cleaning: ±inf → NaN, float32, drop NaN rows, drop exact-duplicate rows (64-bit row hash of
  features + label + source), drop constant columns (→ `artifacts/dropped_columns.json`).
  Non-UTF-8 2017 files are re-read as latin-1, and repeated header rows inside 2018 files are
  removed.
- Labels: `is_attack` (binary) and canonical `attack_type`. All web-attack variants become
  `WebAttack`, and `Infilteration` becomes `Infiltration`.
- **Chronological 80/20 split** per `(source, day, attack_type)`. 2018 uses `Timestamp`. The
  2017 canonical schema has no timestamp, so **file/row order is the chronological proxy**
  (one CSV = one day). The last 10 % of each group's train part is the inner-validation segment.
- `StandardScaler` is fitted on the train split only.
- Windows of 10 flows with stride 5 are built inside each `(source, day, Protocol | file)`
  group and inside each segment, so no window crosses a split. Short groups are left-padded
  with their first flow, and a tail window guarantees every flow is covered. The window label
  is the majority vote, with ties counted as attack.
- Outputs: `X_train.npy (N,10,F)`, `y_train.npy`, `X_test.npy`, `y_test.npy`,
  `y_test_attack_type.npy`, `X_flows_train.npy` (flow level, for the RF), `class_weights.npy`,
  `scaler.joblib`, `feature_columns.json`, `split_indices.npz`, `label_encoders.joblib`,
  `preprocess_summary.json`.

**Step 2 — `src/train.py`** (staged)
1. TCN → `tcn_best.pt`. Optimiser and schedulers: AdamW(1e-3, wd 1e-4), CosineAnnealingLR,
   ReduceLROnPlateau(0.5, 3) and EarlyStopping(patience 5, val_loss, restore best). AMP runs
   on CUDA.
2. Isotonic-calibrated RF (200 trees, depth 20, leaf 5, balanced, cv=3), fitted on **every
   training flow** → `rf_calibrated.joblib`. The calibrator is learned on 3-fold out-of-fold
   predictions and a single forest fit on all flows is kept.
3. `w_dnn` grid search over {0, 0.05, …, 1} on inner-val F1, with ROC-AUC as tie-break →
   `ensemble_weights.json`. Both "best unconstrained" and "best non-degenerate" are saved.
4. DRL (Double-DQN or PPO) on TCN embeddings of **all** training windows → `drl_agent.pt`.
5. Logs: `train_log.csv`, `drl_log.csv`, `drl_steps.csv`, `artifacts/tensorboard/`,
   `train_manifest.json`.

**Evaluation — `src/evaluate.py`** reads the 20 % test split **once** and writes to
`artifacts/eval/`:
- binary metrics (accuracy, P, R, F1, FPR, ROC-AUC, PR-AUC)
- 95 % bootstrap CIs (300 resamples)
- a mandatory per-attack-type FN breakdown
- a τ × k sensitivity sweep
- the ablation table (TCN-only, RF-only, ensemble, +gating, +gating+confirmation, +DRL)
- a cross-dataset transfer matrix, when both datasets are present
- confusion matrices, per-class F1, training curves and reward/ε curves

## 5. "Trained properly on the 80 %": guarantees

- **No silent subsampling.** Every training row is used. `--subsample-fraction` is opt-in and
  logs a warning. Training DataLoaders use `drop_last=False`.
- **Whole set every epoch.** Before training, the script checks that the loader yields
  `ceil(N/bs)` batches (`floor(N/bs)` full batches plus one partial). After each epoch it
  checks that every batch and every sample was seen, and **fails loudly** if not.
- **Full schedule.** `EPOCHS=30` is the maximum. Early stopping cannot fire before
  `MIN_EPOCHS=10`. When it fires, the log shows
  `EarlyStopping triggered at epoch N/30, val_loss=…` and the best checkpoint is restored.
- **Inner validation** is the chronological last 10 % of the 80 %. It is used only for early
  stopping and weight selection. The TCN takes its gradient steps on the other 90 %. The RF
  and DRL use the whole 80 %. With `RF_SELECTION_MODE="holdout"` (the default), one extra RF
  is fitted on inner-train flows only, so the weight search stays unbiased.
- **DRL full pass.** Each DRL epoch first walks all training windows to fill the replay
  buffer, then runs one full pass of updates. The buffer stores index arrays, so its capacity
  grows to `max(REPLAY_BUFFER, N)`. `DRL_BUFFER_MODE="exhaustive"` stores all three actions per
  state instead.
- **Determinism.** Seeds are set for random, numpy, torch and CUDA, with
  `cudnn.deterministic=True` and `cudnn.benchmark=True` (benchmarking picks only among
  deterministic kernels).

## 6. Resources (tested design target: 16 GB RAM, 6 GB GPU)

- **VRAM:** under 1 GB for the TCN and DRL at `BATCH=256` with AMP, so a 6 GB laptop GPU is
  plenty. The device and VRAM are logged at the first step. On OOM the script prints a remedy
  (`--batch-size 128`).
- **RAM:** preprocessing streams the CSVs in chunks and keeps the flow matrix on disk, so the
  full CICIDS2017 + CICIDS2018 needs roughly 3–5 GB of RAM. The RF stage holds the training
  flows (about 4 GB) plus one forest at a time; isotonic calibration uses 3-fold out-of-fold
  predictions (`RF_CALIBRATION_ENSEMBLE=False`), so only one 200-tree forest is kept. If the
  RF stage still runs out of memory, use `--rf-max-samples 0.5`.
- **Disk:** about 40 GB free under `artifacts/` during preprocessing (temporary files are
  deleted at the end; final artifacts are about 20 GB).
- **Slowest stage:** the calibrated RF on all flows (CPU, `n_jobs=-1`); expect a few hours on a
  laptop CPU. Holdout selection roughly doubles it.

## 7. Inference API (the Phase-2 seam)

```python
from pathlib import Path
from src.inference_api import IDSInference
ids = IDSInference(Path("artifacts"), device="cuda")
out = ids.predict_window(window_raw, flow_key="10.0.0.5")   # (10, F) raw, feature_columns.json order
# {"p_dnn","p_rf","s_ensemble","decision","confidence","consecutive_attacks", ...}
if out["decision"] == "BLOCK":
    ids.block_ip("10.0.0.5")                                  # TTL = BLOCK_TTL_SECONDS
```

- The decision rule is `S = w·p_dnn + (1−w)·p_rf`, then the τ gate (0.75), k-confirmation (3)
  per flow key, and an instant BLOCK when `S ≥ 0.95`. The contract is written to
  `artifacts/decision_schema.json`.
- `USE_DRL_AT_INFERENCE=False` (the default) means the ensemble decides, while the DRL action
  is still reported in `drl_action`. Set it to `True` in `src/config.py` (or pass
  `use_drl=True`) to let the DRL policy override the decision.

## 8. Watching attacks get stopped (replay demo)

```bash
python -m src.demo_replay --list
python -m src.demo_replay --attack "DDoS attacks-LOIC-HTTP"
python -m src.demo_replay --all
```

This replays held-out test windows of one flow group, in time order, through
`IDSInference.predict_window()`, which is the same call the Phase-2 XDP consumer makes. It
prints a PASS / MONITOR / BLOCK timeline, shows when `block_ip()` fires and how many flows it
took, and saves a plot to `artifacts/demo/`. No packets are dropped in Phase 1; Phase 2 turns
`block_ip()` into an XDP `block_map` entry.

## 9. Layout

```
src/config.py        every hyper-parameter / path
src/schema.py        canonical columns + 2018→2017 map
src/preprocess.py    Step 1
src/dataset.py       memmap-backed windowed Dataset
src/tcn.py           TCN encoder + head
src/rf_branch.py     calibrated RF
src/drl_agent.py     Double-DQN / PPO
src/ensemble.py      voting + gate + k-confirmation
src/inference_api.py IDSInference (Phase-2 seam)
src/demo_replay.py   attack-stopping replay demo
src/train.py         Step 2
src/evaluate.py      metrics, CIs, plots
src/utils.py         helpers
```
'''

# ==========================================================================
# PHASE.md
# ==========================================================================
FILES['PHASE.md'] = r'''# Phase plan

| Phase | Scope | Status | Deliverable |
|---|---|---|---|
| Phase 1 (this repo) | Offline detection: preprocessing, TCN, calibrated RF, DRL policy, 80/20 evaluation, stable IDSInference API | Complete | tcn_best.pt, rf_calibrated.joblib, drl_agent.pt, eval plots, decision_schema.json |
| Phase 2 | eBPF/XDP data plane: BCC-compiled xdp_feature_extractor.c emitting the same feature_columns.json order, 4 MB ring buffer, flow_table LRU hash, block_map, userspace consumer calling IDSInference.predict_window, sub-microsecond latency benchmark | Planned | xdp_ids.c, ids_userspace.py, Cilium eBPF integration, latency report |

The project title 'Sub-Microsecond Intrusion Prevention…' describes the completed two-phase system. This repository implements Phase 1 — the detection brain — designed so Phase 2 (Cilium eBPF / BCC + XDP) attaches without retraining. The sub-microsecond latency claim will be measured in Phase 2; Phase 1 does not claim it.

## Phase-1 → Phase-2 contract (frozen artifacts)

| Artifact | Consumer in Phase 2 | Purpose |
|---|---|---|
| `artifacts/feature_columns.json` | `xdp_feature_extractor.c` | Exact ordered feature array each flow record must contain |
| `artifacts/scaler.joblib` | `IDSInference` | Raw → standardised features (train-split statistics) |
| `artifacts/tcn_best.pt`, `rf_calibrated.joblib`, `ensemble_weights.json`, `drl_agent.pt` | `IDSInference` | Models + selected weights |
| `artifacts/decision_schema.json` | `ids_userspace.py` | Input/output contract of `predict_window` |
| `config.BLOCK_TTL_SECONDS` | BlockManager thread / `block_map` | Userspace block TTL mirrored into the kernel map |

`IDSInference` already holds the per-flow confirmation counter `c_f` and the block-TTL table in
user space. Phase 2 re-implements these in `block_map` plus a BlockManager thread and keeps
calling `predict_window(features_2d, flow_key)` with the last `WINDOW_SIZE` raw records per flow.

Out of scope for Phase 1: eBPF, XDP, BCC, Cilium, kernel maps, ring buffers, gateway-mode
NAT/iptables, live capture, Tcpreplay, the web dashboard and the latency measurement itself.
'''


def write_tree(out: Path, force: bool) -> int:
    """Write every embedded file (plus .gitkeep placeholders and this script) under ``out``."""
    if out.exists() and any(out.iterdir()) and not force:
        sys.exit(f"[build] {out} already exists and is not empty. Re-run with --force to overwrite "
                 f"the project files (data/ and artifacts/ contents are left untouched).")
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for rel, content in FILES.items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        n += 1
    for rel in EMPTY_FILES:
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        n += 1
    me = Path(os.path.abspath(__file__))
    target = out / "build_project.py"
    if me.resolve() != target.resolve():
        target.write_text(me.read_text(encoding="utf-8"), encoding="utf-8")
    n += 1
    return n


def include_in_zip(rel: Path, abs_path: Path) -> bool:
    """Zip filter: never bundle user data, runtime artifacts, caches or huge files."""
    parts = rel.parts
    if any(part in EXCLUDED_DIRS for part in parts):
        return False
    if rel.suffix == ".pyc" or rel.name == ZIP_NAME:
        return False
    if parts[0] in ("data", "artifacts") and rel.name != ".gitkeep":
        return False
    if abs_path.stat().st_size > MAX_FILE_BYTES:
        print(f"[build] skipping {rel} (> 50 MB)")
        return False
    return True


def make_zip(out: Path) -> Path:
    """Create the archive at ``out / ZIP_NAME`` (ZIP_DEFLATED)."""
    zpath = out / ZIP_NAME
    if zpath.exists():
        zpath.unlink()
    members = []
    for dirpath, dirnames, filenames in os.walk(out):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)
        for fn in sorted(filenames):
            ap = Path(dirpath) / fn
            rel = ap.relative_to(out)
            if include_in_zip(rel, ap):
                members.append((ap, rel.as_posix()))
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for ap, arc in members:
            zf.write(ap, arc)
    return zpath


def verify_zip(zpath: Path) -> None:
    """Re-open the archive, test CRCs, list members, print count and total size."""
    with zipfile.ZipFile(zpath) as zf:
        bad = zf.testzip()
        if bad is not None:
            sys.exit(f"[build] corrupt member in archive: {bad}")
        infos = zf.infolist()
        total = sum(i.file_size for i in infos)
        print(f"{'Length':>10}  Name")
        print(f"{'-' * 10}  {'-' * 40}")
        for i in infos:
            print(f"{i.file_size:>10}  {i.filename}")
        print(f"{'-' * 10}  {'-' * 40}")
        print(f"{total:>10}  {len(infos)} files")
    required = set(FILES) | set(EMPTY_FILES) | {"build_project.py"}
    names = {i.filename for i in infos}
    missing = sorted(required - names)
    if missing:
        sys.exit(f"[build] archive is missing: {missing}")
    print(f"[build] verified: {len(infos)} members, {total:,} bytes uncompressed, "
          f"{zpath.stat().st_size:,} bytes compressed")


def main() -> None:
    ap = argparse.ArgumentParser(description="Write the Phase-1 project and package it as a zip.")
    ap.add_argument("--out", default="./sub-microsecond-phase1", help="Target project directory.")
    ap.add_argument("--force", action="store_true", help="Overwrite project files if --out exists.")
    a = ap.parse_args()
    if sys.version_info < (3, 10):
        sys.exit("[build] Python 3.10+ required.")
    out = Path(a.out).expanduser().resolve()
    n = write_tree(out, a.force)
    print(f"[build] wrote {n} files under {out}")
    zpath = make_zip(out)
    verify_zip(zpath)
    print()
    print(f"Project created at: {out}")
    print(f"Archive created at: {zpath}")
    print("Next:")
    print(f"  unzip {ZIP_NAME} -d myproject && cd myproject")
    print("  pip install -r requirements.txt")
    print("  # drop CICIDS2017 CSVs into data/cicids2017/, CICIDS2018 CSVs into data/cicids2018/")
    print("  python -m src.preprocess --force")
    print("  python -m src.train --algo double_dqn --epochs 30")
    print("  python -m src.evaluate --checkpoint artifacts/tcn_best.pt")
    print("  python -m src.inference_api --self-test")


if __name__ == "__main__":
    main()
