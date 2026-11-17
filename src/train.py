"""Step 2 - staged training on the 80 % split.

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
