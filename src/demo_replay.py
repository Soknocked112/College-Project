"""Attack-stopping replay demo (Phase 1, no packets touched).

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
