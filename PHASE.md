# Phase plan

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
