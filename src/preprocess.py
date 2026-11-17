"""Step 1 - preprocessing for CICIDS2017 + CICIDS2018 (low-memory, disk-backed).

Pipeline (run with ``python -m src.preprocess --force``):

1. Recursively glob ``*.csv`` under ``data/cicids2017`` and ``data/cicids2018``.
2. Stream each file in chunks of ``LOAD_CHUNK_ROWS`` rows
   (``pd.read_csv(low_memory=False, chunksize=...)``), strip BOM/whitespace
   from headers, keep the canonical columns, rename 2018 -> 2017 names. 2018
   ``Timestamp`` / ``Protocol`` are kept *aside* as metadata, never as features.
3. Clean: +/-inf -> NaN, coerce to float32, drop NaN rows (and rows with a
   missing label / repeated header rows), drop exact duplicate rows
   (64-bit row hash of features + label + source, first occurrence kept), drop
   constant columns (names -> ``artifacts/dropped_columns.json``).
4. Labels: ``is_attack`` (binary) and canonical ``attack_type``.
5. Chronological 80/20 split per ``(source, day, attack_type)`` group. 2018 uses
   the parsed ``Timestamp``; 2017 has no timestamp column in the canonical
   schema, so **file order (row order inside each CSV) is the chronological
   proxy** and each CSV is treated as its own "day". Inside the 80 % train
   part the last 10 % of every group is the inner-validation segment.
6. ``StandardScaler`` fitted on the train split only; saved to ``scaler.joblib``.
7. Windowing (``WINDOW_SIZE`` flows, ``STRIDE``) inside each
   ``(source, day, Protocol | file index)`` group and inside each segment
   (inner-train / inner-val / test) so no window straddles a split boundary.
   Groups shorter than the window are left-padded with their first flow, and a
   final tail window is added when the stride does not land on the last flow,
   so every flow belongs to at least one window. Window label = majority
   ``is_attack`` (ties -> attack).
8. Inverse-frequency class weights -> ``class_weights.npy``.

Memory design (fits the full CICIDS2017 + CICIDS2018 in ~16 GB RAM): the
float32 flow matrix is never held in RAM. Cleaned rows are streamed to a
temporary on-disk matrix, metadata is kept as compact integer arrays, and all
later passes (scaling, windowing, flow-level export) read/write in chunks via
memory maps. Temporary files live in ``artifacts/_tmp`` and are deleted at the
end. Peak disk use is roughly 2.5x the final artifact size.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from sklearn.preprocessing import LabelEncoder, StandardScaler

from . import config as C
from . import schema as S
from .utils import get_logger, save_json, set_seed

log = get_logger()
_F32_MAX = float(np.finfo(np.float32).max)
SOURCES: List[str] = ["2017", "2018"]


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_files(source: str) -> List[Path]:
    """Recursively list CSV files of one dataset in deterministic order.

    Args:
        source: ``"2017"`` or ``"2018"``.

    Returns:
        Sorted list of CSV paths (may be empty).
    """
    root = C.DATA_DIRS[source]
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".csv")


def _wanted_columns(source: str) -> List[str]:
    if source == "2017":
        return S.CICIDS2017_FEATURES + [S.LABEL_COLUMN]
    return S.CICIDS2018_FEATURES_RAW + S.META_2018 + [S.LABEL_COLUMN]


def _to_float32_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    """Coerce columns to a float32 matrix; non-numeric / inf / overflow -> NaN."""
    out = np.empty((len(df), len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        bad = ~np.isfinite(v) | (np.abs(v) > _F32_MAX)
        v[bad] = np.nan
        out[:, j] = v.astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
# Streaming loader
# --------------------------------------------------------------------------- #
class _Vocab:
    """String -> int code dictionary (first-seen order)."""

    def __init__(self) -> None:
        self.index: Dict[str, int] = {}

    def encode(self, values: np.ndarray) -> np.ndarray:
        uniq, inv = np.unique(values.astype(str), return_inverse=True)
        codes = np.array([self.index.setdefault(u, len(self.index)) for u in uniq], dtype=np.int32)
        return codes[inv]

    @property
    def strings(self) -> List[str]:
        return list(self.index)


def iter_file_chunks(path: Path, source: str, max_rows: Optional[int]) -> Iterator[pd.DataFrame]:
    """Yield header-normalised, column-filtered DataFrame chunks of one CSV.

    Undecodable bytes (e.g. the cp1252 dash in CICIDS2017 web-attack labels)
    are replaced rather than aborting the read.

    Args:
        path: CSV path.
        source: ``"2017"`` or ``"2018"``.
        max_rows: Optional row cap.

    Yields:
        DataFrame chunks with canonical (2017) column names.

    Raises:
        ValueError: If a required column is missing.
    """
    wanted = set(_wanted_columns(source))
    reader = pd.read_csv(
        path, low_memory=False, nrows=max_rows, chunksize=C.LOAD_CHUNK_ROWS, encoding_errors="replace",
        usecols=lambda c: S.normalize_column_name(c) in wanted,
    )
    first = True
    for df in reader:
        df.columns = [S.normalize_column_name(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        if first:
            missing = sorted(wanted - set(df.columns))
            if missing:
                raise ValueError(
                    f"[{source}] {path} is missing required column(s): {missing}. Check that the file is an "
                    "original CICIDS CSV (MachineLearningCVE / CSE-CIC-IDS2018 Processed Traffic Data)."
                )
            first = False
        if source == "2018":
            df = df.rename(columns=S.MAP_2018_TO_2017)
        yield df


class LoadResult:
    """Container for the streamed, cleaned (pre-dedupe) dataset."""

    def __init__(self) -> None:
        self.raw_path: Path = Path()
        self.n: int = 0
        self.meta: Dict[str, np.ndarray] = {}
        self.label_strings: List[str] = []
        self.day_strings: List[str] = []
        self.files: List[Dict[str, object]] = []
        self.rows_raw: Dict[str, int] = {s: 0 for s in SOURCES}
        self.rows_dropped_nan: int = 0


def load_stream(max_rows: Optional[int], tmp_dir: Path) -> LoadResult:
    """Stream every CSV, clean rows, and write the float32 matrix to disk.

    Args:
        max_rows: Optional per-file row cap.
        tmp_dir: Folder for the temporary raw matrix.

    Returns:
        :class:`LoadResult` with compact per-row metadata arrays.

    Raises:
        SystemExit: If both data folders are empty.
    """
    S.validate_schema()
    files: List[Tuple[str, Path]] = []
    for src in SOURCES:
        found = discover_files(src)
        if not found:
            log.warning(f"No CSV files found under {C.DATA_DIRS[src]} - CICIDS{src} skipped.")
        files += [(src, p) for p in found]
    if not files:
        log.error("No CSV files found. Drop the raw CICIDS2017 CSVs into "
                  f"{C.DATA_DIRS['2017']} and/or the CICIDS2018 CSVs into {C.DATA_DIRS['2018']} "
                  "(sub-folders are fine), then rerun `python -m src.preprocess --force`.")
        raise SystemExit(2)

    res = LoadResult()
    res.raw_path = tmp_dir / "raw_flows.f32"
    labels, days = _Vocab(), _Vocab()
    parts: Dict[str, List[np.ndarray]] = {k: [] for k in
                                          ("source", "file_idx", "seq", "day", "ts_ns", "win_key", "label", "hash")}
    feats = S.CICIDS2017_FEATURES
    with open(res.raw_path, "wb") as fout:
        for file_idx, (src, path) in enumerate(files):
            seq0, last_ts, n_file = 0, None, 0
            for df in iter_file_chunks(path, src, max_rows):
                n = len(df)
                seq = np.arange(seq0, seq0 + n, dtype=np.int64)
                seq0 += n
                res.rows_raw[src] += n
                X = _to_float32_matrix(df, feats)
                lab_raw = df[S.LABEL_COLUMN]
                lab = lab_raw.astype(str).str.strip()
                valid = (~np.isnan(X).any(axis=1) & lab_raw.notna().to_numpy()
                         & (lab.str.len() > 0).to_numpy() & (lab.str.lower() != "label").to_numpy())
                if src == "2018":
                    ts = pd.to_datetime(df[S.TIMESTAMP_COLUMN], format="%d/%m/%Y %H:%M:%S", errors="coerce")
                    if ts.isna().mean() > 0.5:
                        ts = pd.to_datetime(df[S.TIMESTAMP_COLUMN], errors="coerce", dayfirst=True)
                    if last_ts is not None and pd.isna(ts.iloc[0]):
                        ts.iloc[0] = last_ts
                    ts = ts.ffill().bfill()
                    if ts.notna().any():
                        last_ts = ts.iloc[-1]
                        day_str = ts.dt.strftime("%Y-%m-%d").fillna(f"2018-f{file_idx:03d}").to_numpy()
                        ts_ns = np.where(ts.notna(), ts.astype("int64"), seq).astype(np.int64)
                    else:
                        day_str = np.full(n, f"2018-f{file_idx:03d}")
                        ts_ns = seq
                    win_key = pd.to_numeric(df[S.PROTOCOL_COLUMN], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
                else:
                    # 2017: no timestamp in the canonical schema -> file order is the chronological proxy.
                    day_str = np.full(n, f"2017-f{file_idx:03d}")
                    ts_ns = seq
                    win_key = np.full(n, file_idx, dtype=np.int64)
                res.rows_dropped_nan += int((~valid).sum())
                Xv = np.ascontiguousarray(X[valid])
                lab_codes = labels.encode(lab.to_numpy()[valid])
                src_code = SOURCES.index(src)
                hdf = pd.DataFrame(Xv, copy=False)
                hdf["_label"] = lab_codes
                hdf["_source"] = src_code
                h = pd.util.hash_pandas_object(hdf, index=False).to_numpy()
                del hdf
                fout.write(Xv.tobytes())
                nv = len(Xv)
                n_file += nv
                parts["source"].append(np.full(nv, src_code, dtype=np.int8))
                parts["file_idx"].append(np.full(nv, file_idx, dtype=np.int32))
                parts["seq"].append(seq[valid])
                parts["day"].append(days.encode(np.asarray(day_str)[valid]))
                parts["ts_ns"].append(np.asarray(ts_ns)[valid])
                parts["win_key"].append(np.asarray(win_key)[valid])
                parts["label"].append(lab_codes)
                parts["hash"].append(h)
            log.info(f"[{src}] streamed {path.name}: {seq0:,} rows ({n_file:,} valid)")
            res.files.append({"file_idx": file_idx, "source": src, "path": str(
                path.relative_to(C.PROJECT_ROOT) if path.is_relative_to(C.PROJECT_ROOT) else path)})
    res.meta = {k: np.concatenate(v) for k, v in parts.items()}
    res.n = len(res.meta["label"])
    res.label_strings = labels.strings
    # Re-code days so that integer order == lexicographic (chronological) order.
    ds = days.strings
    rank = np.argsort(np.argsort(np.array(ds, dtype=object)))
    res.meta["day"] = rank[res.meta["day"]].astype(np.int32)
    res.day_strings = sorted(ds)
    return res


def raw_matrix(res: LoadResult) -> np.ndarray:
    """Open the temporary raw float32 matrix as a read-only memmap."""
    return np.memmap(res.raw_path, dtype=np.float32, mode="r", shape=(res.n, len(S.CICIDS2017_FEATURES)))


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #
def constant_columns_chunked(X: np.ndarray, rows: np.ndarray, names: List[str]) -> List[str]:
    """Names of columns whose min equals max over ``rows`` (chunked scan).

    Args:
        X: Matrix / memmap.
        rows: Row ids to consider (sorted).
        names: Column names.

    Returns:
        Constant column names.
    """
    if len(rows) == 0:
        return []
    lo = np.full(X.shape[1], np.inf, dtype=np.float64)
    hi = np.full(X.shape[1], -np.inf, dtype=np.float64)
    for s in range(0, len(rows), C.LOAD_CHUNK_ROWS):
        blk = X[rows[s:s + C.LOAD_CHUNK_ROWS]]
        lo = np.minimum(lo, blk.min(axis=0))
        hi = np.maximum(hi, blk.max(axis=0))
    return [n for n, a, b in zip(names, lo, hi) if a == b]


def constant_columns(X: np.ndarray, names: List[str]) -> List[str]:
    """Names of constant columns of an in-memory matrix."""
    return constant_columns_chunked(X, np.arange(len(X)), names)


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def chronological_split(meta: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological 80/20 split per (source, day, attack_type) + inner-val tail.

    Args:
        meta: Metadata with ``source, day, ts_ns, file_idx, seq, attack_type``
            (any sortable dtypes; ``day`` must sort chronologically).

    Returns:
        ``(order, is_train, is_inner_val)`` - ``order`` is the chronological
        permutation of rows; the two masks are aligned with the *original* rows.
    """
    order = np.lexsort((
        meta["seq"].to_numpy(), meta["file_idx"].to_numpy(), meta["ts_ns"].to_numpy(),
        meta["day"].to_numpy(), meta["source"].to_numpy(),
    ))
    m = meta.iloc[order]
    keys = [m["source"], m["day"], m["attack_type"]]
    rank = m.groupby(keys, sort=False).cumcount().to_numpy()
    size = m.groupby(keys, sort=False)["seq"].transform("size").to_numpy()
    del m
    n_train = np.ceil(size * C.TRAIN_FRACTION).astype(np.int64)
    train_sorted = rank < n_train
    n_inner_train = np.ceil(n_train * (1.0 - C.INNER_VAL_FRACTION)).astype(np.int64)
    inner_val_sorted = train_sorted & (rank >= n_inner_train)
    is_train = np.zeros(len(meta), dtype=bool)
    is_inner_val = np.zeros(len(meta), dtype=bool)
    is_train[order] = train_sorted
    is_inner_val[order] = inner_val_sorted
    return order, is_train, is_inner_val


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def window_offsets(n: int, w: int = C.WINDOW_SIZE, s: int = C.STRIDE) -> np.ndarray:
    """Local row offsets of every window in a group of ``n`` chronologically ordered flows.

    Args:
        n: Group length (>0).
        w: Window size.
        s: Stride.

    Returns:
        ``(k, w)`` int64 array of offsets in ``[0, n)``.
    """
    if n >= w:
        starts = np.arange(0, n - w + 1, s, dtype=np.int64)
        if starts[-1] != n - w:
            starts = np.append(starts, n - w)  # tail window: last flow always covered
        return starts[:, None] + np.arange(w, dtype=np.int64)[None, :]
    pad = np.zeros(w - n, dtype=np.int64)  # left-pad with the first flow
    return np.concatenate([pad, np.arange(n, dtype=np.int64)])[None, :]


def build_window_index(rows_sorted: np.ndarray, meta: pd.DataFrame, group_offset: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build window row-index matrix for one segment.

    Args:
        rows_sorted: Row ids of the segment in chronological order.
        meta: Full metadata (``source, day, win_key``).
        group_offset: First group id to assign (keeps ids globally unique).

    Returns:
        ``(index, group_id)`` with shapes ``(N_w, W)`` and ``(N_w,)``.
    """
    if len(rows_sorted) == 0:
        return np.zeros((0, C.WINDOW_SIZE), dtype=np.int64), np.zeros(0, dtype=np.int64)
    sub = meta.iloc[rows_sorted]
    gkey = pd.MultiIndex.from_arrays([sub["source"], sub["day"], sub["win_key"]])
    codes, _ = pd.factorize(gkey, sort=False)
    del sub, gkey
    perm = np.lexsort((np.arange(len(codes)), codes))  # stable: keeps chronology inside group
    rows_g = rows_sorted[perm]
    codes_g = codes[perm]
    bounds = np.flatnonzero(np.diff(codes_g)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(codes_g)]])
    idx_parts, gid_parts = [], []
    for g, (a, b) in enumerate(zip(starts, ends)):
        off = window_offsets(int(b - a))
        idx_parts.append(rows_g[a + off])
        gid_parts.append(np.full(len(off), group_offset + g, dtype=np.int64))
    return np.concatenate(idx_parts), np.concatenate(gid_parts)


def window_labels(index: np.ndarray, y: np.ndarray, type_codes: np.ndarray, benign_code: int,
                  n_types: int) -> Tuple[np.ndarray, np.ndarray]:
    """Majority-vote window labels (ties -> attack) and majority attack type (chunked).

    Args:
        index: ``(N_w, W)`` row ids.
        y: Flow-level ``is_attack``.
        type_codes: Flow-level attack-type codes.
        benign_code: Code of ``BENIGN``.
        n_types: Number of attack-type codes.

    Returns:
        ``(y_window, type_window_codes)``.
    """
    label = np.empty(len(index), dtype=np.int64)
    tcode = np.empty(len(index), dtype=np.int64)
    for s in range(0, len(index), C.WINDOW_WRITE_CHUNK):
        ix = index[s:s + C.WINDOW_WRITE_CHUNK]
        yw = y[ix]
        lab = (2 * yw.sum(axis=1) >= ix.shape[1]).astype(np.int64)
        tw = type_codes[ix]
        counts = np.stack([((tw == k) & (yw == 1)).sum(axis=1) for k in range(n_types)], axis=1)
        counts[:, benign_code] = -1
        tc = counts.argmax(axis=1)
        tc[lab == 0] = benign_code
        label[s:s + len(ix)], tcode[s:s + len(ix)] = lab, tc
    return label, tcode


def write_windows(path: Path, X: np.ndarray, index: np.ndarray) -> None:
    """Materialise windows ``X[index]`` into an ``.npy`` file chunk by chunk.

    Args:
        path: Output ``.npy`` path.
        X: Scaled flow matrix ``(n, F)`` (array or memmap).
        index: ``(N_w, W)`` row ids.
    """
    mm = open_memmap(path, mode="w+", dtype=np.float32, shape=(len(index), index.shape[1], X.shape[1]))
    for s in range(0, len(index), C.WINDOW_WRITE_CHUNK):
        e = min(s + C.WINDOW_WRITE_CHUNK, len(index))
        mm[s:e] = X[index[s:e]]
    mm.flush()
    del mm


def write_rows(path: Path, X: np.ndarray, rows: np.ndarray) -> None:
    """Write ``X[rows]`` to an ``.npy`` file chunk by chunk.

    Args:
        path: Output path.
        X: Source matrix / memmap.
        rows: Row ids.
    """
    mm = open_memmap(path, mode="w+", dtype=np.float32, shape=(len(rows), X.shape[1]))
    for s in range(0, len(rows), C.WINDOW_WRITE_CHUNK):
        e = min(s + C.WINDOW_WRITE_CHUNK, len(rows))
        mm[s:e] = X[rows[s:e]]
    mm.flush()
    del mm


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _class_balance(y: np.ndarray) -> Dict[str, object]:
    n = int(len(y))
    a = int(np.asarray(y).sum())
    return {"n": n, "benign": n - a, "attack": a, "attack_pct": round(100.0 * a / n, 3) if n else 0.0}


def _check_disk(path: Path) -> None:
    free_gb = shutil.disk_usage(path).free / 1024 ** 3
    msg = f"Free disk under {path}: {free_gb:.1f} GB"
    if free_gb < C.MIN_FREE_DISK_GB:
        log.warning(msg + f" - the full CICIDS2017+2018 run needs about {C.MIN_FREE_DISK_GB} GB "
                    "(temporary files included). Free some space if preprocessing fails.")
    else:
        log.info(msg)


def run(force: bool = False, max_rows: Optional[int] = C.MAX_ROWS_PER_FILE) -> Dict[str, object]:
    """Execute the full Step-1 pipeline and write every artifact.

    Args:
        force: Overwrite existing artifacts.
        max_rows: Optional per-file row cap (debug only).

    Returns:
        The summary dict (also saved to ``preprocess_summary.json``).
    """
    set_seed(C.SEED)
    A = C.ARTIFACT_DIR
    A.mkdir(parents=True, exist_ok=True)
    if C.artifact_path("X_train").exists() and not force:
        log.error("Preprocessed artifacts already exist. Use `python -m src.preprocess --force` to rebuild.")
        raise SystemExit(1)
    for key in ("X_train", "X_test", "X_flows_train"):
        C.artifact_path(key).unlink(missing_ok=True)
    if C.EVAL_DIR.exists():
        shutil.rmtree(C.EVAL_DIR)
    tmp = A / "_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    _check_disk(A)

    try:
        summary = _run_inner(max_rows, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return summary


def _run_inner(max_rows: Optional[int], tmp: Path) -> Dict[str, object]:
    res = load_stream(max_rows, tmp)
    Xraw = raw_matrix(res)
    meta = res.meta

    # ---- exact-duplicate removal (64-bit row hash; first occurrence kept) ----
    dup = pd.Series(meta.pop("hash")).duplicated(keep="first").to_numpy()
    kept = np.flatnonzero(~dup)
    meta = {k: v[kept] for k, v in meta.items()}
    n = len(kept)
    stats = {"rows_raw": int(sum(res.rows_raw.values())), "rows_dropped_nan": res.rows_dropped_nan,
             "rows_dropped_duplicate": int(dup.sum()), "rows_clean": int(n)}
    del dup
    log.info(f"Cleaned: {n:,} rows kept ({stats['rows_dropped_duplicate']:,} duplicates, "
             f"{stats['rows_dropped_nan']:,} NaN/inf/invalid rows dropped)")

    # ---- constant columns ----
    feats_all = list(S.CICIDS2017_FEATURES)
    dropped = constant_columns_chunked(Xraw, kept, feats_all)
    keep_cols = np.array([i for i, c in enumerate(feats_all) if c not in dropped], dtype=np.int64)
    feats = [feats_all[i] for i in keep_cols]
    save_json(dropped, C.artifact_path("dropped_columns"))
    save_json(feats, C.artifact_path("feature_columns"))

    # ---- labels ----
    lab_str = np.array(res.label_strings, dtype=object)
    lab_is_attack = np.array([s.strip().lower() != "benign" for s in lab_str], dtype=np.int64)
    lab_type = np.array([S.canonical_attack_type(s) for s in lab_str], dtype=object)
    types_present = np.unique(np.append(lab_type[np.unique(meta["label"])].astype(str), "BENIGN"))
    le_type = LabelEncoder().fit(types_present)
    lab_type_code = le_type.transform(lab_type.astype(str)).astype(np.int64)
    y = lab_is_attack[meta["label"]]
    type_codes = lab_type_code[meta["label"]]
    benign_code = int(le_type.transform(["BENIGN"])[0])
    joblib.dump({"attack_type": le_type, "binary": {"BENIGN": 0, "ATTACK": 1}}, C.artifact_path("label_encoders"))

    # ---- chronological split ----
    mdf = pd.DataFrame({
        "source": meta["source"], "day": meta["day"], "ts_ns": meta["ts_ns"], "file_idx": meta["file_idx"],
        "seq": meta["seq"], "attack_type": type_codes.astype(np.int16), "win_key": meta["win_key"],
    })
    order, is_train, is_inner_val = chronological_split(mdf)
    is_test = ~is_train
    np.savez_compressed(
        C.artifact_path("split_indices"),
        train_idx=np.flatnonzero(is_train), test_idx=np.flatnonzero(is_test),
        inner_val_idx=np.flatnonzero(is_inner_val),
        row_file_idx=meta["file_idx"], row_seq_in_file=meta["seq"],
    )

    # ---- scaler (train rows only), then a scaled on-disk flow matrix ----
    scaler = StandardScaler()
    train_rows = np.flatnonzero(is_train)
    for s in range(0, len(train_rows), C.LOAD_CHUNK_ROWS):
        blk = Xraw[kept[train_rows[s:s + C.LOAD_CHUNK_ROWS]]][:, keep_cols]
        scaler.partial_fit(blk.astype(np.float64))
    scaler.feature_names_ordered_ = np.array(feats)
    joblib.dump(scaler, C.artifact_path("scaler"))
    Xs = open_memmap(tmp / "flows_scaled.npy", mode="w+", dtype=np.float32, shape=(n, len(feats)))
    for s in range(0, n, C.LOAD_CHUNK_ROWS):
        blk = Xraw[kept[s:s + C.LOAD_CHUNK_ROWS]][:, keep_cols]
        Xs[s:s + len(blk)] = scaler.transform(blk.astype(np.float64)).astype(np.float32)
    Xs.flush()
    del Xraw
    res.raw_path.unlink(missing_ok=True)  # free disk early
    log.info("Scaled flow matrix written; building windows ...")

    # ---- windows ----
    seg_inner_train = order[(is_train & ~is_inner_val)[order]]
    seg_inner_val = order[is_inner_val[order]]
    seg_test = order[is_test[order]]
    idx_it, g_it = build_window_index(seg_inner_train, mdf, 0)
    idx_iv, g_iv = build_window_index(seg_inner_val, mdf, int(g_it.max(initial=-1)) + 1)
    idx_te, g_te = build_window_index(seg_test, mdf, 0)
    del seg_inner_train, seg_inner_val, seg_test, mdf
    idx_tr = np.concatenate([idx_it, idx_iv])
    g_tr = np.concatenate([g_it, g_iv])
    inner_val_mask = np.concatenate([np.zeros(len(idx_it), bool), np.ones(len(idx_iv), bool)])
    del idx_it, idx_iv, g_it, g_iv

    n_types = len(le_type.classes_)
    y_tr, t_tr = window_labels(idx_tr, y, type_codes, benign_code, n_types)
    y_te, t_te = window_labels(idx_te, y, type_codes, benign_code, n_types)

    write_windows(C.artifact_path("X_train"), Xs, idx_tr)
    log.info(f"X_train.npy written: {len(idx_tr):,} windows")
    write_windows(C.artifact_path("X_test"), Xs, idx_te)
    log.info(f"X_test.npy written: {len(idx_te):,} windows")
    np.save(C.artifact_path("y_train"), y_tr)
    np.save(C.artifact_path("y_test"), y_te)
    np.save(C.artifact_path("train_attack_type"), le_type.classes_[t_tr].astype(str))
    np.save(C.artifact_path("test_attack_type"), le_type.classes_[t_te].astype(str))
    np.save(C.artifact_path("train_inner_val"), inner_val_mask)
    np.save(C.artifact_path("train_group"), g_tr)
    np.save(C.artifact_path("test_group"), g_te)
    src_str = np.array(SOURCES)
    np.save(C.artifact_path("test_source"), src_str[meta["source"][idx_te[:, -1]]])

    # ---- flow-level train data (RF is fit on the full 80 % at flow level) ----
    flows_tr = order[is_train[order]]
    write_rows(C.artifact_path("X_flows_train"), Xs, flows_tr)
    np.save(C.artifact_path("y_flows_train"), y[flows_tr])
    np.save(C.artifact_path("flows_train_inner_val"), is_inner_val[flows_tr])
    np.save(C.artifact_path("flows_train_source"), src_str[meta["source"][flows_tr]])
    del Xs

    # ---- class weights ----
    counts = np.bincount(y_tr, minlength=2).astype(np.float64)
    if (counts == 0).any():
        log.warning(f"A class is absent from the training windows (counts={counts.tolist()}); weight set to 1.0.")
    cw = np.where(counts > 0, counts.sum() / (2.0 * np.maximum(counts, 1)), 1.0).astype(np.float32)
    np.save(C.artifact_path("class_weights"), cw)

    clean_per_src = np.bincount(meta["source"].astype(np.int64), minlength=2)
    summary: Dict[str, object] = {
        "files": res.files,
        "rows_per_source_raw": {s: int(v) for s, v in res.rows_raw.items() if v},
        "rows_per_source_clean": {s: int(clean_per_src[i]) for i, s in enumerate(SOURCES) if clean_per_src[i]},
        **stats,
        "n_features": len(feats),
        "dropped_constant_columns": dropped,
        "flow_split": {
            "train": _class_balance(y[is_train]), "test": _class_balance(y[is_test]),
            "inner_val (subset of train)": _class_balance(y[is_inner_val]),
        },
        "window_counts": {
            "train": int(len(idx_tr)), "train_inner_val": int(inner_val_mask.sum()), "test": int(len(idx_te)),
        },
        "window_split": {"train": _class_balance(y_tr), "test": _class_balance(y_te)},
        "attack_types_test_windows": {str(k): int(v) for k, v in
                                      zip(*np.unique(le_type.classes_[t_te], return_counts=True))},
        "class_weights": cw.tolist(),
        "window_size": C.WINDOW_SIZE, "stride": C.STRIDE,
        "chronology_note": "2018: Timestamp; 2017: file/row order used as chronological proxy (one CSV = one day).",
        "max_rows_per_file": max_rows,
    }
    save_json(summary, C.artifact_path("preprocess_summary"))
    _print_summary(summary)
    return summary


def _print_summary(s: Dict[str, object]) -> None:
    log.info("=" * 72)
    log.info("PREPROCESS SUMMARY")
    log.info(f"Rows per source (raw):   {s['rows_per_source_raw']}")
    log.info(f"Rows per source (clean): {s['rows_per_source_clean']}")
    log.info(f"Dropped NaN/inf rows: {s['rows_dropped_nan']:,} | duplicate rows: {s['rows_dropped_duplicate']:,}")
    log.info(f"Feature count: {s['n_features']} | dropped constant columns: {s['dropped_constant_columns']}")
    for k, v in s["flow_split"].items():  # type: ignore[union-attr]
        log.info(f"Flow split {k:<26}: {v}")
    log.info(f"Window counts: {s['window_counts']}")
    for k, v in s["window_split"].items():  # type: ignore[union-attr]
        log.info(f"Window split {k:<6}: {v}")
    log.info(f"Class weights: {s['class_weights']}")
    log.info("=" * 72)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point: ``python -m src.preprocess [--force] [--max-rows-per-file N]``."""
    p = argparse.ArgumentParser(description="Step 1: preprocess CICIDS2017 + CICIDS2018.")
    p.add_argument("--force", action="store_true", help="Rebuild all preprocessing artifacts.")
    p.add_argument("--max-rows-per-file", type=int, default=C.MAX_ROWS_PER_FILE,
                   help="Debug-only row cap per CSV (default: None = all rows).")
    a = p.parse_args(argv)
    if a.max_rows_per_file is not None:
        log.warning(f"--max-rows-per-file={a.max_rows_per_file}: NOT the full dataset (debug mode).")
    run(force=a.force, max_rows=a.max_rows_per_file)


if __name__ == "__main__":
    main(sys.argv[1:])
