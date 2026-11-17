"""Central configuration for Phase 1 of
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
