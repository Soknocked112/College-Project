"""Windowed PyTorch dataset over the ``(N, WINDOW_SIZE, F)`` arrays from Step 1.

The dataset stores the *path* of the ``.npy`` file and opens it lazily as a
read-only memory map in every worker process, so multi-GB window arrays are
never pickled into DataLoader workers. Batches are fetched with one sorted
fancy-index read via ``__getitems__`` (PyTorch >= 2.0), which is much faster
than per-sample indexing on a memory map.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import config as C


class WindowDataset(Dataset):
    """Map-style dataset of flow windows and binary window labels.

    Args:
        x_path: Path of the ``.npy`` window array ``(N, W, F)`` (float32).
        y: Window labels ``(N,)`` (int).
        indices: Optional subset of window indices (e.g. inner-train). ``None`` = all.
    """

    def __init__(self, x_path: Path, y: np.ndarray, indices: Optional[np.ndarray] = None) -> None:
        self.x_path = Path(x_path)
        self.y = np.asarray(y).astype(np.int64)
        self.indices = np.arange(len(self.y), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        self._X: Optional[np.ndarray] = None
        shape = np.load(self.x_path, mmap_mode="r").shape
        if shape[0] != len(self.y):
            raise ValueError(f"{self.x_path} has {shape[0]} windows but y has {len(self.y)}")
        self.window_size, self.n_features = int(shape[1]), int(shape[2])

    # Lazily opened memmap - never pickled into workers.
    @property
    def X(self) -> np.ndarray:
        """Read-only memory map of the window array."""
        if self._X is None:
            self._X = np.load(self.x_path, mmap_mode="r")
        return self._X

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_X"] = None
        return state

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        j = int(self.indices[i])
        return torch.from_numpy(np.array(self.X[j], dtype=np.float32)), torch.tensor(self.y[j])

    def __getitems__(self, idxs: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        rows = np.sort(self.indices[np.asarray(idxs, dtype=np.int64)])
        xb = np.ascontiguousarray(self.X[rows], dtype=np.float32)
        return torch.from_numpy(xb), torch.from_numpy(self.y[rows])


def batch_identity(batch: Any) -> Any:
    """Collate function for datasets that already return whole batches."""
    return batch


def _worker_init(worker_id: int) -> None:
    seed = (C.SEED + worker_id) % (2 ** 32)
    np.random.seed(seed)


def make_loader(ds: WindowDataset, batch_size: int, shuffle: bool, device: str,
                num_workers: int = C.NUM_WORKERS, seed: int = C.SEED) -> DataLoader:
    """Create a DataLoader that never drops samples (``drop_last=False``).

    Args:
        ds: Dataset.
        batch_size: Batch size.
        shuffle: Shuffle each epoch (seeded generator).
        device: Target device; enables ``pin_memory`` on CUDA.
        num_workers: Worker processes.
        seed: Generator seed.

    Returns:
        A :class:`torch.utils.data.DataLoader`.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=False,
        num_workers=num_workers, pin_memory=(C.PIN_MEMORY and device.startswith("cuda")),
        collate_fn=batch_identity, generator=g, worker_init_fn=_worker_init,
        persistent_workers=num_workers > 0,
    )


def expected_batches(n: int, batch_size: int) -> Dict[str, int]:
    """Batch accounting used for the "whole 80 % per epoch" assertion.

    Args:
        n: Number of samples.
        batch_size: Batch size.

    Returns:
        ``{"full": floor(n/bs), "partial": 0|1, "total": ceil(n/bs)}``.
    """
    full = n // batch_size
    return {"full": full, "partial": int(n % batch_size > 0), "total": math.ceil(n / batch_size)}


def subsample_indices(indices: np.ndarray, fraction: Optional[float], seed: int = C.SEED) -> np.ndarray:
    """Opt-in random subsample (sorted); identity when ``fraction`` is ``None`` or 1.

    Args:
        indices: Candidate indices.
        fraction: Fraction in ``(0, 1]`` or ``None``.
        seed: RNG seed.

    Returns:
        Sorted subset of ``indices``.
    """
    if fraction is None or fraction >= 1.0:
        return indices
    if not 0.0 < fraction < 1.0:
        raise ValueError("--subsample-fraction must be in (0, 1].")
    rng = np.random.default_rng(seed)
    k = max(1, int(round(len(indices) * fraction)))
    return np.sort(rng.choice(indices, size=k, replace=False))


def contiguous_runs(idx: np.ndarray) -> List[slice]:
    """Split sorted indices into contiguous slices (fast memmap reads).

    Args:
        idx: Sorted int indices.

    Returns:
        List of slices.
    """
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(idx)]])
    return [slice(int(idx[a]), int(idx[b - 1]) + 1) for a, b in zip(starts, ends)]
