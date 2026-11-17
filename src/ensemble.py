"""Weighted soft voting, confidence gate and k-confirmation (spec §6d).

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
