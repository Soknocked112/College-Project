"""Canonical CICIDS2017 / CICIDS2018 column lists and the 2018 -> 2017 mapping.

The lists below are copied verbatim from the project specification. The 2017
list contains ``"Fwd Header Length"`` twice; :data:`CICIDS2017_FEATURES` is the
de-duplicated version (first occurrence kept) and is the canonical feature
order. After constant-column removal the final order is persisted to
``artifacts/feature_columns.json`` - the exact array the Phase-2 XDP feature
extractor must emit.
"""
from __future__ import annotations

from typing import Dict, List

CICIDS2017_COLUMNS = [
    "Destination Port","Flow Duration","Total Fwd Packets","Total Backward Packets",
    "Total Length of Fwd Packets","Total Length of Bwd Packets",
    "Fwd Packet Length Max","Fwd Packet Length Min","Fwd Packet Length Mean","Fwd Packet Length Std",
    "Bwd Packet Length Max","Bwd Packet Length Min","Bwd Packet Length Mean","Bwd Packet Length Std",
    "Flow Bytes/s","Flow Packets/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min",
    "Fwd IAT Total","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min",
    "Bwd IAT Total","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min",
    "Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags",
    "Fwd Header Length","Bwd Header Length","Fwd Packets/s","Bwd Packets/s",
    "Min Packet Length","Max Packet Length","Packet Length Mean","Packet Length Std","Packet Length Variance",
    "FIN Flag Count","SYN Flag Count","RST Flag Count","PSH Flag Count","ACK Flag Count","URG Flag Count",
    "CWE Flag Count","ECE Flag Count","Down/Up Ratio","Average Packet Size",
    "Avg Fwd Segment Size","Avg Bwd Segment Size","Fwd Header Length",
    "Fwd Avg Bytes/Bulk","Fwd Avg Packets/Bulk","Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk","Bwd Avg Packets/Bulk","Bwd Avg Bulk Rate",
    "Subflow Fwd Packets","Subflow Fwd Bytes","Subflow Bwd Packets","Subflow Bwd Bytes",
    "Init_Win_bytes_forward","Init_Win_bytes_backward","act_data_pkt_fwd","min_seg_size_forward",
    "Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min",
]

CICIDS2018_COLUMNS = [
    "Dst Port","Protocol","Timestamp","Flow Duration","Tot Fwd Pkts","Tot Bwd Pkts",
    "TotLen Fwd Pkts","TotLen Bwd Pkts","Fwd Pkt Len Max","Fwd Pkt Len Min","Fwd Pkt Len Mean","Fwd Pkt Len Std",
    "Bwd Pkt Len Max","Bwd Pkt Len Min","Bwd Pkt Len Mean","Bwd Pkt Len Std",
    "Flow Byts/s","Flow Pkts/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min",
    "Fwd IAT Tot","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min",
    "Bwd IAT Tot","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min",
    "Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags",
    "Fwd Header Len","Bwd Header Len","Fwd Pkts/s","Bwd Pkts/s",
    "Pkt Len Min","Pkt Len Max","Pkt Len Mean","Pkt Len Std","Pkt Len Var",
    "FIN Flag Cnt","SYN Flag Cnt","RST Flag Cnt","PSH Flag Cnt","ACK Flag Cnt","URG Flag Cnt",
    "CWE Flag Count","ECE Flag Cnt","Down/Up Ratio","Pkt Size Avg",
    "Fwd Seg Size Avg","Bwd Seg Size Avg",
    "Fwd Byts/b Avg","Fwd Pkts/b Avg","Fwd Blk Rate Avg",
    "Bwd Byts/b Avg","Bwd Pkts/b Avg","Bwd Blk Rate Avg",
    "Subflow Fwd Pkts","Subflow Fwd Byts","Subflow Bwd Pkts","Subflow Bwd Byts",
    "Init Fwd Win Byts","Init Bwd Win Byts","Fwd Act Data Pkts","Fwd Seg Size Min",
    "Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min",
]

MAP_2018_TO_2017 = {
    "Dst Port":"Destination Port", "Tot Fwd Pkts":"Total Fwd Packets",
    "Tot Bwd Pkts":"Total Backward Packets", "TotLen Fwd Pkts":"Total Length of Fwd Packets",
    "TotLen Bwd Pkts":"Total Length of Bwd Packets",
    "Fwd Pkt Len Max":"Fwd Packet Length Max", "Fwd Pkt Len Min":"Fwd Packet Length Min",
    "Fwd Pkt Len Mean":"Fwd Packet Length Mean", "Fwd Pkt Len Std":"Fwd Packet Length Std",
    "Bwd Pkt Len Max":"Bwd Packet Length Max", "Bwd Pkt Len Min":"Bwd Packet Length Min",
    "Bwd Pkt Len Mean":"Bwd Packet Length Mean", "Bwd Pkt Len Std":"Bwd Packet Length Std",
    "Flow Byts/s":"Flow Bytes/s", "Flow Pkts/s":"Flow Packets/s",
    "Fwd IAT Tot":"Fwd IAT Total", "Bwd IAT Tot":"Bwd IAT Total",
    "Fwd Header Len":"Fwd Header Length", "Bwd Header Len":"Bwd Header Length",
    "Fwd Pkts/s":"Fwd Packets/s", "Bwd Pkts/s":"Bwd Packets/s",
    "Pkt Len Min":"Min Packet Length", "Pkt Len Max":"Max Packet Length",
    "Pkt Len Mean":"Packet Length Mean", "Pkt Len Std":"Packet Length Std",
    "Pkt Len Var":"Packet Length Variance",
    "FIN Flag Cnt":"FIN Flag Count", "SYN Flag Cnt":"SYN Flag Count",
    "RST Flag Cnt":"RST Flag Count", "PSH Flag Cnt":"PSH Flag Count",
    "ACK Flag Cnt":"ACK Flag Count", "URG Flag Cnt":"URG Flag Count",
    "ECE Flag Cnt":"ECE Flag Count", "Pkt Size Avg":"Average Packet Size",
    "Fwd Seg Size Avg":"Avg Fwd Segment Size", "Bwd Seg Size Avg":"Avg Bwd Segment Size",
    "Fwd Byts/b Avg":"Fwd Avg Bytes/Bulk", "Fwd Pkts/b Avg":"Fwd Avg Packets/Bulk",
    "Fwd Blk Rate Avg":"Fwd Avg Bulk Rate", "Bwd Byts/b Avg":"Bwd Avg Bytes/Bulk",
    "Bwd Pkts/b Avg":"Bwd Avg Packets/Bulk", "Bwd Blk Rate Avg":"Bwd Avg Bulk Rate",
    "Subflow Fwd Pkts":"Subflow Fwd Packets", "Subflow Fwd Byts":"Subflow Fwd Bytes",
    "Subflow Bwd Pkts":"Subflow Bwd Packets", "Subflow Bwd Byts":"Subflow Bwd Bytes",
    "Init Fwd Win Byts":"Init_Win_bytes_forward", "Init Bwd Win Byts":"Init_Win_bytes_backward",
    "Fwd Act Data Pkts":"act_data_pkt_fwd", "Fwd Seg Size Min":"min_seg_size_forward",
}

LABEL_COLUMN: str = "Label"
TIMESTAMP_COLUMN: str = "Timestamp"
PROTOCOL_COLUMN: str = "Protocol"
META_2018: List[str] = [TIMESTAMP_COLUMN, PROTOCOL_COLUMN]


def dedupe_keep_first(columns: List[str]) -> List[str]:
    """Return ``columns`` with duplicates removed, keeping first occurrences.

    Args:
        columns: Column names, possibly with repeats.

    Returns:
        Order-preserving de-duplicated list.
    """
    seen: set = set()
    out: List[str] = []
    for c in columns:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


#: Canonical (pre constant-drop) feature order - de-duplicated 2017 schema.
CICIDS2017_FEATURES: List[str] = dedupe_keep_first(CICIDS2017_COLUMNS)

#: 2018 feature columns (raw names), i.e. the 2018 schema minus metadata.
CICIDS2018_FEATURES_RAW: List[str] = [c for c in CICIDS2018_COLUMNS if c not in META_2018]


def rename_2018(name: str) -> str:
    """Map a 2018 column name to its 2017 canonical name (identity if shared).

    Args:
        name: A CICIDS2018 column name.

    Returns:
        The CICIDS2017 canonical name.
    """
    return MAP_2018_TO_2017.get(name, name)


def normalize_column_name(name: str) -> str:
    """Strip a UTF-8 BOM and surrounding whitespace from a CSV header.

    Args:
        name: Raw header string.

    Returns:
        Cleaned header string.
    """
    return str(name).replace("﻿", "").strip()


def validate_schema() -> Dict[str, int]:
    """Check that the renamed 2018 features equal the 2017 canonical features.

    Raises:
        ValueError: If the two schemas do not align one-to-one.

    Returns:
        Small dict of counts for logging.
    """
    renamed = [rename_2018(c) for c in CICIDS2018_FEATURES_RAW]
    a, b = set(renamed), set(CICIDS2017_FEATURES)
    if a != b or len(renamed) != len(set(renamed)):
        raise ValueError(
            "CICIDS2018->2017 mapping mismatch. Only in 2018: "
            f"{sorted(a - b)}; only in 2017: {sorted(b - a)}"
        )
    return {
        "n_2017_raw": len(CICIDS2017_COLUMNS),
        "n_2017_dedup": len(CICIDS2017_FEATURES),
        "n_2018_features": len(CICIDS2018_FEATURES_RAW),
    }


def canonical_attack_type(label: str) -> str:
    """Normalise a raw CICIDS label into a canonical attack-type string.

    Merges ``Web Attack - Brute Force / XSS / Sql Injection`` (2017, any dash
    encoding) and ``Brute Force -Web / -XSS / SQL Injection`` (2018) into
    ``WebAttack``; fixes ``Infilteration`` -> ``Infiltration``; ``BENIGN`` and
    ``Benign`` -> ``BENIGN``.

    Args:
        label: Raw label string.

    Returns:
        Canonical attack-type string.
    """
    s = str(label).strip()
    low = s.lower()
    if low == "benign":
        return "BENIGN"
    if low.startswith("web attack") or low in {
        "brute force -web", "brute force -xss", "sql injection"
    }:
        return "WebAttack"
    if low.startswith("infilt"):
        return "Infiltration"
    return s
