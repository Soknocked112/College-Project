"""Isotonic-calibrated RandomForest branch (spec §6b).

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
