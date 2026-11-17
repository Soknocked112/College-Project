"""Temporal Convolutional Network encoder + binary head.

Architecture (spec §6a):

* Causal dilated ``Conv1d`` residual blocks with weight normalisation, ReLU and
  Dropout(0.3). One block per entry of ``TCN_DILATIONS`` (1, 2, 4, 8). Block
  ``i`` uses ``TCN_CHANNELS[min(i, len-1)]`` output channels, i.e. 64, 128,
  128, 128 - the channel list is shorter than the dilation list, so the last
  width is repeated. Receptive field = 1 + 2*(k-1)*sum(d) = 61 >= WINDOW_SIZE.
* Global average pool over time -> Linear -> BatchNorm -> ReLU -> Dropout(0.3)
  -> embedding ``(B, TCN_EMBED_DIM)``. (ReLU added for a non-linear embedding.)
* Head: Linear(128->64) -> BN -> ReLU -> Dropout(0.3) -> Linear(64->32) -> ReLU
  -> Linear(32->2).

Input layout is ``(B, W, F)`` (window of flow records); it is transposed to
``(B, F, W)`` internally.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .utils import autocast


def _weight_norm(module: nn.Module) -> nn.Module:
    try:
        from torch.nn.utils.parametrizations import weight_norm

        return weight_norm(module)
    except ImportError:  # pragma: no cover - torch < 2.1
        return nn.utils.weight_norm(module)


class Chomp1d(nn.Module):
    """Remove right padding so the convolution is causal.

    Args:
        chomp: Number of trailing time steps to drop.
    """

    def __init__(self, chomp: int) -> None:
        super().__init__()
        self.chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return x[:, :, : -self.chomp].contiguous() if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    """Residual block of two causal dilated convolutions.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        kernel: Kernel size.
        dilation: Dilation.
        dropout: Dropout probability.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            _weight_norm(nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            _weight_norm(nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.relu(self.net(x) + self.downsample(x))


class TCNEncoder(nn.Module):
    """Stack of temporal blocks + pooling + projection to an embedding.

    Args:
        n_features: Number of per-flow features ``F``.
        channels: Channel widths (repeated to match ``dilations``).
        kernel: Kernel size.
        dilations: Dilation per block.
        embed_dim: Embedding size.
        dropout: Dropout probability.
    """

    def __init__(self, n_features: int, channels: List[int] = C.TCN_CHANNELS, kernel: int = C.TCN_KERNEL,
                 dilations: List[int] = C.TCN_DILATIONS, embed_dim: int = C.TCN_EMBED_DIM,
                 dropout: float = C.DROPOUT) -> None:
        super().__init__()
        blocks: List[nn.Module] = []
        in_ch = n_features
        for i, d in enumerate(dilations):
            out_ch = channels[min(i, len(channels) - 1)]
            blocks.append(TemporalBlock(in_ch, out_ch, kernel, d, dropout))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Sequential(nn.Linear(in_ch, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode windows.

        Args:
            x: ``(B, W, F)`` tensor.

        Returns:
            ``(B, embed_dim)`` embedding.
        """
        h = self.blocks(x.transpose(1, 2))  # (B, C, W)
        return self.proj(h.mean(dim=2))


class TCNClassifier(nn.Module):
    """TCN encoder followed by the binary MLP head.

    Args:
        n_features: Number of per-flow features.
        hidden_dims: ``[embed, h1, h2]`` (spec ``HIDDEN_DIMS``).
        **enc_kwargs: Forwarded to :class:`TCNEncoder`.
    """

    def __init__(self, n_features: int, hidden_dims: List[int] = C.HIDDEN_DIMS, dropout: float = C.DROPOUT,
                 num_classes: int = C.NUM_CLASSES, **enc_kwargs: Any) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.encoder = TCNEncoder(n_features, dropout=dropout, **enc_kwargs)
        e, h1, h2 = self.encoder.embed_dim, hidden_dims[1], hidden_dims[2]
        self.head = nn.Sequential(
            nn.Linear(e, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits ``(B, 2)`` for windows ``(B, W, F)``."""
        return self.head(self.encoder(x))

    def forward_with_embedding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, embedding)``."""
        z = self.encoder(x)
        return self.head(z), z


def model_config(n_features: int) -> Dict[str, Any]:
    """Serialisable hyper-parameters needed to rebuild the model.

    Args:
        n_features: Number of features.

    Returns:
        Config dict stored inside the checkpoint.
    """
    return {
        "n_features": n_features, "channels": list(C.TCN_CHANNELS), "kernel": C.TCN_KERNEL,
        "dilations": list(C.TCN_DILATIONS), "embed_dim": C.TCN_EMBED_DIM, "dropout": C.DROPOUT,
        "hidden_dims": list(C.HIDDEN_DIMS), "window_size": C.WINDOW_SIZE,
    }


def build_from_config(cfg: Dict[str, Any]) -> TCNClassifier:
    """Instantiate a :class:`TCNClassifier` from :func:`model_config` output."""
    return TCNClassifier(
        cfg["n_features"], hidden_dims=cfg["hidden_dims"], dropout=cfg["dropout"],
        channels=cfg["channels"], kernel=cfg["kernel"], dilations=cfg["dilations"], embed_dim=cfg["embed_dim"],
    )


def save_checkpoint(model: TCNClassifier, path: Path, extra: Optional[Dict[str, Any]] = None) -> None:
    """Save weights + config (+ metadata) to ``path``."""
    payload = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "model_config": model_config(model.n_features)}
    payload.update(extra or {})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: str = "cpu") -> Tuple[TCNClassifier, Dict[str, Any]]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Args:
        path: Checkpoint path.
        device: Map location.

    Returns:
        ``(model_in_eval_mode, payload)``.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_from_config(payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload


@torch.no_grad()
def infer(model: TCNClassifier, X: np.ndarray, device: str, indices: Optional[np.ndarray] = None,
          batch_size: int = C.EVAL_BATCH_SIZE, return_embedding: bool = False, use_amp: bool = C.USE_AMP
          ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Batched inference over a (memory-mapped) window array.

    Args:
        model: Trained classifier.
        X: ``(N, W, F)`` array or memmap (already scaled).
        device: Device string.
        indices: Optional sorted subset of rows.
        batch_size: Inference batch size.
        return_embedding: Also return embeddings (float16 to save RAM).
        use_amp: Use autocast on CUDA.

    Returns:
        ``(p_attack (n,), embeddings (n, E) | None)``.
    """
    model.eval()
    idx = np.arange(len(X)) if indices is None else np.asarray(indices)
    probs = np.empty(len(idx), dtype=np.float32)
    embs = np.empty((len(idx), model.encoder.embed_dim), dtype=np.float16) if return_embedding else None
    for s in range(0, len(idx), batch_size):
        sel = idx[s:s + batch_size]
        xb = np.array(X[sel[0]:sel[-1] + 1] if (sel[-1] - sel[0] + 1) == len(sel) else X[sel], dtype=np.float32)
        xt = torch.from_numpy(xb).to(device, non_blocking=True)
        with autocast(device, use_amp):
            logits, z = model.forward_with_embedding(xt)
        probs[s:s + len(sel)] = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        if embs is not None:
            embs[s:s + len(sel)] = z.float().cpu().numpy().astype(np.float16)
    return probs, embs
