"""Final evaluation on the held-out 20 % test split (read exactly once).

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
