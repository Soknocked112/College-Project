"""Phase-2 seam: one-call inference over a window of raw flow records.

Step 3 (Cilium eBPF / BCC + XDP) will run an XDP feature extractor that emits,
per flow, the feature vector in exactly the order of
``artifacts/feature_columns.json``. Its userspace consumer keeps the last
``WINDOW_SIZE`` records per flow key and calls::

    ids = IDSInference(Path("artifacts"), device="cuda")
    out = ids.predict_window(window_raw, flow_key="10.0.0.5")   # (WINDOW_SIZE, F)
    if out["decision"] == "BLOCK":
        ids.block_ip("10.0.0.5")          # Phase 2: also writes block_map

No retraining is needed: the scaler, feature order, models, ensemble weights
and decision thresholds are all loaded from ``artifacts/``. The per-flow
confirmation counter ``c_f`` and the block TTL table live here in user space
and mirror what Phase 2 re-implements as ``block_map`` + a BlockManager thread.

Self-test (no dataset needed): ``python -m src.inference_api --self-test``.
If trained artifacts are absent it runs in *synthetic mode* with randomly
initialised components purely to exercise the interface.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import config as C
from . import schema as S
from .ensemble import ConfirmationState
from .utils import get_logger, load_json, save_json

log = get_logger()

DECISION_FIELDS: Dict[str, Dict[str, str]] = {
    "flow_key": {"type": "string", "description": "Flow key whose confirmation counter was updated."},
    "p_dnn": {"type": "number", "description": "TCN P(attack) for the window, in [0,1]."},
    "p_rf": {"type": "number", "description": "Calibrated RF P(attack) for the window's last flow, in [0,1]."},
    "s_ensemble": {"type": "number", "description": "S = w_dnn*p_dnn + w_rf*p_rf."},
    "y_hat": {"type": "integer", "description": "1 if S >= 0.5 else 0."},
    "decision": {"type": "string", "enum": "PASS|MONITOR|BLOCK", "description": "Final action."},
    "confidence": {"type": "number", "description": "S if y_hat==1 else 1-S."},
    "consecutive_attacks": {"type": "integer", "description": "Confirmation counter c_f in [0, K_CONFIRM]."},
    "drl_action": {"type": "string|null", "description": "DRL greedy action (pass|monitor|block) or null."},
    "decision_source": {"type": "string", "description": "'ensemble' or 'drl' (USE_DRL_AT_INFERENCE)."},
    "latency_ms": {"type": "number", "description": "Userspace model latency for this call (not the XDP path)."},
}


def write_decision_schema(path: Path, n_features: Optional[int] = None) -> Dict[str, Any]:
    """Write ``decision_schema.json`` describing the ``predict_window`` contract.

    Args:
        path: Destination path.
        n_features: Feature count (read from ``feature_columns.json`` if omitted).

    Returns:
        The schema dict.
    """
    path = Path(path)
    fc = path.parent / Path(C.FEATURE_ORDER_PATH).name
    if n_features is None and fc.exists():
        n_features = len(load_json(fc))
    schema = {
        "title": "IDSInference.predict_window decision",
        "version": 1,
        "input": {
            "features_2d": {"shape": [C.WINDOW_SIZE, n_features if n_features else "F"], "dtype": "float32",
                            "units": "raw (unscaled) CICFlowMeter-compatible flow features",
                            "row_order": "oldest -> newest flow record of the flow key",
                            "column_order": C.FEATURE_ORDER_PATH},
            "flow_key": {"type": "string", "description": "e.g. source IP or 5-tuple hash used for c_f"},
        },
        "output": DECISION_FIELDS,
        "actions": C.ACTIONS,
        "decision_rule": {"tau": C.TAU, "k_confirm": C.K_CONFIRM, "tau_hard": C.TAU_HARD,
                          "threshold": C.DECISION_THRESHOLD},
        "block_ttl_seconds": C.BLOCK_TTL_SECONDS,
        "scaler_path": C.SCALER_PATH,
    }
    save_json(schema, path)
    return schema


def _single_thread_rf(model: Any) -> None:
    """Use n_jobs=1 for per-window RF scoring (thread start-up dominates on 1 row)."""
    for cc in getattr(model, "calibrated_classifiers_", []):
        est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if hasattr(est, "n_jobs"):
            est.n_jobs = 1
    if hasattr(model, "n_jobs") and not hasattr(model, "calibrated_classifiers_"):
        model.n_jobs = 1


class IDSInference:
    """Load Phase-1 artifacts and score windows of raw flow records.

    Args:
        artifact_dir: Folder containing the Phase-1 artifacts.
        device: ``"cuda"`` or ``"cpu"`` (auto-falls back to CPU).
        use_drl: Override ``USE_DRL_AT_INFERENCE``.
        components: Internal - pre-built components (synthetic self-test mode).
    """

    def __init__(self, artifact_dir: Path = C.ARTIFACT_DIR, device: str = "cuda", use_drl: Optional[bool] = None,
                 components: Optional[Dict[str, Any]] = None) -> None:
        import torch

        from .utils import resolve_device

        self.torch = torch
        self.artifact_dir = Path(artifact_dir)
        self.device = resolve_device(device)
        self.use_drl = C.USE_DRL_AT_INFERENCE if use_drl is None else bool(use_drl)
        comp = components if components is not None else self._load_components()
        self.feature_columns: List[str] = comp["feature_columns"]
        self.n_features = len(self.feature_columns)
        self.mean = np.asarray(comp["scaler"].mean_, dtype=np.float32)
        self.scale = np.asarray(comp["scaler"].scale_, dtype=np.float32)
        if len(self.mean) != self.n_features:
            raise ValueError(f"Scaler has {len(self.mean)} features but feature_columns.json has {self.n_features}.")
        self.tcn = comp["tcn"].to(self.device).eval()
        self.rf = comp["rf"]
        _single_thread_rf(self.rf)
        ew = comp["ensemble_weights"]
        self.w_dnn = float(ew["w_dnn"])
        self.w_rf = 1.0 - self.w_dnn
        self.state = ConfirmationState(ew.get("tau", C.TAU), ew.get("k_confirm", C.K_CONFIRM),
                                       ew.get("tau_hard", C.TAU_HARD))
        self.drl = comp.get("drl")
        if self.drl is not None:
            self.drl = self.drl.to(self.device)
        if self.use_drl and self.drl is None:
            raise FileNotFoundError("USE_DRL_AT_INFERENCE=True but drl_agent.pt was not found.")
        self.block_table: Dict[str, float] = {}
        self.synthetic = bool(comp.get("synthetic", False))
        schema_path = self.artifact_dir / Path(C.DECISION_OUTPUT_SCHEMA).name
        if not schema_path.exists():
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            write_decision_schema(schema_path, self.n_features)

    # ------------------------------------------------------------------ #
    def _load_components(self) -> Dict[str, Any]:
        import joblib

        from .drl_agent import load_policy
        from .rf_branch import load_rf
        from .tcn import load_checkpoint

        a = self.artifact_dir
        need = {"feature_columns": C.ART["feature_columns"], "scaler": C.ART["scaler"], "tcn": C.ART["tcn"],
                "rf": C.ART["rf"], "ensemble_weights": C.ART["ensemble_weights"]}
        missing = [n for n in need.values() if not (a / n).exists()]
        if missing:
            raise FileNotFoundError(f"Missing artifacts in {a}: {missing}. Run preprocess + train first.")
        tcn, payload = load_checkpoint(a / need["tcn"], "cpu")
        comp = {
            "feature_columns": load_json(a / need["feature_columns"]),
            "scaler": joblib.load(a / need["scaler"]),
            "tcn": tcn,
            "rf": load_rf(a / need["rf"]),
            "ensemble_weights": load_json(a / need["ensemble_weights"]),
            "drl": load_policy(a / C.ART["drl"], "cpu") if (a / C.ART["drl"]).exists() else None,
        }
        if payload["model_config"]["n_features"] != len(comp["feature_columns"]):
            raise ValueError("tcn_best.pt was trained with a different feature count than feature_columns.json.")
        return comp

    @classmethod
    def synthetic(cls, device: str = "cpu", artifact_dir: Optional[Path] = None) -> "IDSInference":
        """Build an instance from random components (interface smoke test only).

        Args:
            device: Device string.
            artifact_dir: Where ``decision_schema.json`` is written (temp-like ``artifacts/selftest``).

        Returns:
            A synthetic :class:`IDSInference` (predictions are meaningless).
        """
        import torch
        from sklearn.preprocessing import StandardScaler

        from .rf_branch import build_rf
        from .tcn import TCNClassifier

        torch.manual_seed(C.SEED)
        rng = np.random.default_rng(C.SEED)
        feats = list(S.CICIDS2017_FEATURES)
        Xr = rng.normal(size=(400, len(feats))).astype(np.float32)
        yr = (Xr[:, 0] > 0).astype(int)
        rf = build_rf(n_estimators=10, calibrated=True).fit(Xr, yr)
        comp = {"feature_columns": feats, "scaler": StandardScaler().fit(Xr), "tcn": TCNClassifier(len(feats)),
                "rf": rf, "ensemble_weights": {"w_dnn": 0.5}, "drl": None, "synthetic": True}
        return cls(artifact_dir or (C.ARTIFACT_DIR / "selftest"), device=device, use_drl=False, components=comp)

    # ------------------------------------------------------------------ #
    def _prepare(self, features_2d: np.ndarray) -> np.ndarray:
        x = np.asarray(features_2d, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"features_2d must have shape ({C.WINDOW_SIZE}, {self.n_features}); got {x.shape}")
        if x.shape[0] > C.WINDOW_SIZE:
            x = x[-C.WINDOW_SIZE:]
        elif x.shape[0] < C.WINDOW_SIZE:  # same left-padding rule as preprocessing
            x = np.concatenate([np.repeat(x[:1], C.WINDOW_SIZE - x.shape[0], axis=0), x], axis=0)
        x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
        return (x - self.mean) / self.scale

    def predict_window(self, features_2d: np.ndarray, flow_key: str = "default") -> Dict[str, Any]:
        """Score one window and update the flow key's confirmation counter.

        Args:
            features_2d: ``(WINDOW_SIZE, F)`` raw (unscaled) features, columns in
                ``feature_columns.json`` order, rows oldest -> newest. Fewer rows
                are left-padded with the first row; extra rows keep the newest.
            flow_key: Key for the per-flow ``c_f`` counter.

        Returns:
            Dict with ``p_dnn, p_rf, s_ensemble, decision, confidence,
            consecutive_attacks`` (+ ``flow_key, y_hat, drl_action,
            decision_source, latency_ms``); see ``decision_schema.json``.
        """
        torch = self.torch
        t0 = time.perf_counter()
        xs = self._prepare(features_2d)
        with torch.no_grad():
            xt = torch.from_numpy(xs[None]).to(self.device)
            logits, z = self.tcn.forward_with_embedding(xt)
            p_dnn = float(torch.softmax(logits.float(), 1)[0, 1])
        classes = list(getattr(self.rf, "classes_", [0, 1]))
        p_rf = float(self.rf.predict_proba(xs[-1:])[0, classes.index(1)])
        s = self.w_dnn * p_dnn + self.w_rf * p_rf
        decision, c_f = self.state.update(flow_key, s)
        drl_action = None
        source = "ensemble"
        if self.drl is not None:
            drl_action = C.ACTIONS[int(self.drl.act(z.float())[0])]
            if self.use_drl:
                decision, source = drl_action.upper(), "drl"
        y_hat = int(s >= C.DECISION_THRESHOLD)
        return {
            "flow_key": flow_key, "p_dnn": p_dnn, "p_rf": p_rf, "s_ensemble": float(s), "y_hat": y_hat,
            "decision": decision, "confidence": float(s if y_hat else 1.0 - s), "consecutive_attacks": int(c_f),
            "drl_action": drl_action, "decision_source": source,
            "latency_ms": (time.perf_counter() - t0) * 1e3,
        }

    def reset_flow_state(self, flow_key: str) -> None:
        """Forget the confirmation counter of ``flow_key`` (e.g. flow evicted from the LRU)."""
        self.state.reset(flow_key)

    def block_ip(self, ip: str, ttl: int = C.BLOCK_TTL_SECONDS) -> None:
        """Record a userspace block for ``ip`` that expires after ``ttl`` seconds.

        Phase 2 mirrors this into the XDP ``block_map``; Phase 1 only keeps the table.

        Args:
            ip: Address (string).
            ttl: Seconds until expiry.
        """
        self.block_table[ip] = time.monotonic() + float(ttl)

    def is_blocked(self, ip: str) -> bool:
        """True if ``ip`` has an unexpired block entry."""
        exp = self.block_table.get(ip)
        if exp is None:
            return False
        if exp <= time.monotonic():
            del self.block_table[ip]
            return False
        return True

    def purge_expired(self) -> List[str]:
        """Remove and return expired block entries (BlockManager tick)."""
        now = time.monotonic()
        gone = [ip for ip, e in self.block_table.items() if e <= now]
        for ip in gone:
            del self.block_table[ip]
        return gone


def self_test(device: str = "cpu") -> Dict[str, Any]:
    """Run one zero window through the API; synthetic mode if artifacts are missing.

    Args:
        device: Device string.

    Returns:
        The decision dict.
    """
    needed = [C.ART[k] for k in ("feature_columns", "scaler", "tcn", "rf", "ensemble_weights")]
    if all((C.ARTIFACT_DIR / n).exists() for n in needed):
        ids = IDSInference(C.ARTIFACT_DIR, device=device)
        log.info("Self-test using trained artifacts.")
    else:
        log.warning("Trained artifacts not found -> SYNTHETIC mode (random weights; interface check only).")
        ids = IDSInference.synthetic(device=device)
    window = np.zeros((C.WINDOW_SIZE, ids.n_features), dtype=np.float32)
    out = ids.predict_window(window, flow_key="selftest-flow")
    for _ in range(C.K_CONFIRM):
        out = ids.predict_window(window, flow_key="selftest-flow")
    ids.block_ip("192.0.2.1", ttl=1)
    assert ids.is_blocked("192.0.2.1")
    ids.reset_flow_state("selftest-flow")
    log.info(f"Feature count: {ids.n_features} | window: {C.WINDOW_SIZE} | w_dnn={ids.w_dnn:.2f} | "
             f"synthetic={ids.synthetic}")
    for k, v in out.items():
        log.info(f"  {k:<20} {v}")
    log.info(f"SELF-TEST OK - decision: {out['decision']}")
    return out


def main(argv: Optional[List[str]] = None) -> None:
    """CLI: ``python -m src.inference_api --self-test [--device cpu]``."""
    p = argparse.ArgumentParser(description="IDSInference (Phase-2 seam)")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--device", default=C.DEVICE)
    a = p.parse_args(argv)
    if a.self_test:
        self_test(a.device)
    else:
        p.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])
