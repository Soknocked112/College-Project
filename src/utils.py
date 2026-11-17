"""Shared helpers: seeding, device resolution, logging, JSON IO, metrics."""
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
