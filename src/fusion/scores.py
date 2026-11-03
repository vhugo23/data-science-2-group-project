"""Adapter that turns teammates' score tables into the shared contract.

Contract columns: sample_id, split, label, score. Extra columns such as fold,
turn_fraction, condition or attack are kept after the contract columns.
"""
from pathlib import Path

import numpy as np
import pandas as pd

CONTRACT_COLUMNS = ["sample_id", "split", "label", "score"]
SPLITS = ("train", "validation", "test")

# Column names each track uses today, mapped to the contract names.
_RENAMES = {
    "audio": {},
    "text": {"dialogue_id": "sample_id", "scam_score": "score"},
}
# Columns that legitimately repeat a sample id (one row per condition).
_REPEAT_KEYS = ("turn_fraction", "condition")


def normalize_scores(table, kind, model=None, allow_external=False):
    """Return a copy of a score table in contract form, or raise ValueError."""
    if kind not in _RENAMES:
        raise ValueError(f"kind must be one of {sorted(_RENAMES)}, got {kind!r}")
    table = table.copy()

    if "model" in table.columns:
        names = sorted(table["model"].astype(str).unique())
        if model is None and len(names) > 1:
            raise ValueError(f"table holds several models {names}; pass model=")
        if model is not None:
            if model not in names:
                raise ValueError(f"model {model!r} not in table; available: {names}")
            table = table[table["model"].astype(str) == model]
        table = table.drop(columns="model")
    elif model is not None:
        raise ValueError("model was given but the table has no model column")

    table = table.rename(columns=_RENAMES[kind])
    missing = [column for column in CONTRACT_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"{kind} table is missing columns: {missing}")

    allowed = set(SPLITS) | ({"external"} if allow_external else set())
    unknown = sorted(set(table["split"].astype(str)) - allowed)
    if unknown:
        raise ValueError(f"unexpected split values {unknown}; allowed: {sorted(allowed)}")

    label = pd.to_numeric(table["label"], errors="coerce")
    if label.isna().any() or not label.isin([0, 1]).all():
        raise ValueError("label must be 0 or 1 in every row")
    score = pd.to_numeric(table["score"], errors="coerce")
    if not np.isfinite(score).all():
        raise ValueError("score must be a finite number in every row")

    table["sample_id"] = table["sample_id"].astype(str).str.strip()
    table["split"] = table["split"].astype(str)
    table["label"] = label.astype("int64")
    table["score"] = score.astype("float64")

    crossing = table.groupby("sample_id")["split"].nunique()
    if (crossing > 1).any():
        raise ValueError(f"sample_id appears in more than one split: {list(crossing[crossing > 1].index[:5])}")
    key = ["sample_id"] + [column for column in _REPEAT_KEYS if column in table.columns]
    if table.duplicated(key).any():
        raise ValueError(f"duplicate rows for key {key}")

    extras = [column for column in table.columns if column not in CONTRACT_COLUMNS]
    return table[CONTRACT_COLUMNS + extras].reset_index(drop=True)


def load_scores(path, kind, **kwargs):
    """Read a score CSV and normalize it."""
    return normalize_scores(pd.read_csv(Path(path)), kind, **kwargs)
