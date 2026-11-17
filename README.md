# Sub-Microsecond Intrusion Prevention — Phase 1 (Detection Brain)

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
