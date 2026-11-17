"""Attack-detection test: feed known attack flows from the held-out test set
through the trained models and confirm they are detected and blocked.

This is a Phase-1 test. It drops no real packets (that is Phase 2); it checks
that the trained detection brain, reached through the same
``IDSInference.predict_window`` call the XDP data plane will use, detects and
blocks attack traffic taken from the 20% test split.

What it reports, per attack type and overall:

* detection rate   - share of attack windows whose ensemble score S >= 0.5
* block rate       - share of attack windows the decision rule would BLOCK,
                     feeding each attack as a continuous burst under one flow
                     key so the confidence gate and k-confirmation can act
* a benign control - the same test on benign windows, i.e. the false-block rate
* DRL action mix   - what the learned policy would do
* latency          - user-space model time per window (not the XDP fast path)

Usage::

    python -m src.test_attack                       # every attack type + benign control
    python -m src.test_attack --attack SSH-Bruteforce
    python -m src.test_attack --attack WebAttack --limit 200
    python -m src.test_attack --device cpu

Exit status is 0 if the overall attack detection rate meets --min-detection
(default 0.80 over detectable types), 1 otherwise, so it can be used as a smoke
test in a script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

from . import config as C
from .utils import get_logger

log = get_logger()


def _load_test():
    """Load the test windows (scaled), their labels and attack types."""
    need = ["X_test", "y_test", "test_attack_type"]
    for k in need:
        if not C.artifact_path(k).exists():
            raise SystemExit(f"Missing {C.artifact_path(k)}. Run preprocess, train and evaluate first.")
    X = np.load(C.artifact_path("X_test"), mmap_mode="r")
    y = np.load(C.artifact_path("y_test"))
    t = np.load(C.artifact_path("test_attack_type"))
    return X, y, t


def _raw_window(X: np.ndarray, i: int, scaler) -> np.ndarray:
    """Undo standardisation so the window looks like raw XDP features again."""
    return np.asarray(X[i], dtype=np.float32) * scaler.scale_.astype(np.float32) + scaler.mean_.astype(np.float32)


def test_group(ids, X, idx: np.ndarray, scaler, flow_key: str, as_burst: bool) -> Dict[str, object]:
    """Feed a set of windows through the model and tally the outcomes.

    Args:
        ids: An :class:`IDSInference` instance.
        X: Scaled test window array.
        idx: Indices of the windows to feed.
        scaler: Fitted scaler (to rebuild raw features).
        flow_key: Flow key for the confirmation counter.
        as_burst: If True, feed every window under one key so the counter can
            build up (the realistic case for a sustained attack). If False,
            reset the counter before each window (isolated-window detection).

    Returns:
        Dict of counts, rates, DRL action mix and mean latency.
    """
    n = len(idx)
    detected = blocked = monitored = 0
    drl = {a: 0 for a in C.ACTIONS}
    lat = 0.0
    smax = 0.0
    for j, i in enumerate(idx):
        if not as_burst:
            ids.reset_flow_state(flow_key)
        out = ids.predict_window(_raw_window(X, int(i), scaler), flow_key=flow_key)
        if out["s_ensemble"] >= C.DECISION_THRESHOLD:
            detected += 1
        if out["decision"] == "BLOCK":
            blocked += 1
        elif out["decision"] == "MONITOR":
            monitored += 1
        if out["drl_action"] is not None:
            drl[out["drl_action"]] += 1
        lat += out["latency_ms"]
        smax = max(smax, out["s_ensemble"])
    ids.reset_flow_state(flow_key)
    return {
        "n": n, "detected": detected, "blocked": blocked, "monitored": monitored,
        "detection_rate": detected / n if n else 0.0,
        "block_rate": blocked / n if n else 0.0,
        "drl": drl, "mean_latency_ms": lat / n if n else 0.0, "max_score": smax,
    }


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Feed known attack flows from the test set through the trained models.")
    p.add_argument("--attack", default=None, help="One attack type (default: every type). See the printed list.")
    p.add_argument("--limit", type=int, default=1000, help="Max windows per attack type to test (default 1000).")
    p.add_argument("--device", default=C.DEVICE)
    p.add_argument("--isolated", action="store_true",
                   help="Reset the counter before each window (test single-window detection, not a sustained burst).")
    p.add_argument("--min-detection", type=float, default=0.80,
                   help="Overall detection rate (detectable types) required for exit status 0.")
    a = p.parse_args(argv)

    from .inference_api import IDSInference  # imported here so --help works without torch

    X, y, types = _load_test()
    scaler = joblib.load(C.artifact_path("scaler"))
    ids = IDSInference(C.ARTIFACT_DIR, device=a.device)
    if ids.synthetic:
        log.warning("Running with SYNTHETIC (random) weights - train the models first for a real test.")
    rng = np.random.default_rng(C.SEED)

    attack_types = [a.attack] if a.attack else [t for t in np.unique(types) if t != "BENIGN"]
    log.info("=" * 92)
    log.info(f"{'attack type':<28}{'windows':>8}{'detected':>10}{'blocked':>9}{'max S':>8}  DRL(block/monitor/pass)")
    log.info("-" * 92)

    rows: List[Dict[str, object]] = []
    total_det = total_n = 0
    for at in attack_types:
        sel = np.flatnonzero(types == at)
        if len(sel) == 0:
            log.warning(f"No test windows labelled {at!r} - skipped.")
            continue
        if len(sel) > a.limit:
            sel = np.sort(rng.choice(sel, a.limit, replace=False))
        r = test_group(ids, X, sel, scaler, f"attack::{at}", as_burst=not a.isolated)
        rows.append({"attack": at, **r})
        d = r["drl"]
        log.info(f"{at[:27]:<28}{r['n']:>8}{r['detection_rate']*100:>9.1f}%{r['block_rate']*100:>8.1f}%"
                 f"{r['max_score']:>8.3f}  {d['block']}/{d['monitor']}/{d['pass']}")
        # Only count types that are statistically meaningful toward the pass/fail gate.
        if r["n"] >= 20:
            total_det += r["detected"]
            total_n += r["n"]

    # Benign control: false-block rate
    ben = np.flatnonzero(types == "BENIGN")
    if len(ben) > a.limit:
        ben = np.sort(rng.choice(ben, a.limit, replace=False))
    bctrl = test_group(ids, X, ben, scaler, "benign::control", as_burst=not a.isolated)
    log.info("-" * 92)
    log.info(f"{'BENIGN (control)':<28}{bctrl['n']:>8}{bctrl['detection_rate']*100:>9.1f}%"
             f"{bctrl['block_rate']*100:>8.1f}%{bctrl['max_score']:>8.3f}   (false positives)")
    log.info("=" * 92)

    overall = total_det / total_n if total_n else 0.0
    log.info(f"Overall attack detection rate (types with >=20 windows): {overall*100:.1f}%  "
             f"over {total_n:,} windows")
    log.info(f"Benign false-block rate: {bctrl['block_rate']*100:.2f}%  |  "
             f"mean model latency: {bctrl['mean_latency_ms']:.1f} ms/window")
    if a.isolated:
        log.info("Mode: isolated windows (counter reset each window).")
    else:
        log.info("Mode: sustained burst (one flow key per attack type; gate + k-confirmation active).")

    ok = overall >= a.min_detection
    log.info(("PASS" if ok else "BELOW THRESHOLD") +
             f": overall detection {overall*100:.1f}% vs required {a.min_detection*100:.0f}%.")
    log.info("Note: web/infiltration/low-and-slow attacks are known weak classes (see the report); "
             "a low rate there is expected, not a failure of the test.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
