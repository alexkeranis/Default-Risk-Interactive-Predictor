"""Raw data loading, integrity hashing and the one-time development/test split."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from bnpl_risk.config import (
    CLASS_TO_INT,
    DATA_PATH,
    ID_COL,
    INPUT_COLS,
    SEED,
    SPLIT_PATH,
    TARGET_COL,
    TEST_SIZE,
)
from bnpl_risk.schema import assert_valid


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}")
    df = pd.read_csv(path)
    missing = [c for c in [ID_COL, TARGET_COL, *INPUT_COLS] if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data is missing columns: {missing}")
    if not df[ID_COL].is_unique:
        raise ValueError(f"{ID_COL} is not unique")
    unknown = set(df[TARGET_COL].unique()) - set(CLASS_TO_INT)
    if unknown:
        raise ValueError(f"Unknown target labels: {sorted(unknown)}")
    assert_valid(df[INPUT_COLS])  # the training data must pass the same gate as live traffic
    return df


def encode_target(labels: pd.Series | np.ndarray) -> np.ndarray:
    return pd.Series(labels).map(CLASS_TO_INT).to_numpy(dtype=np.int64)


@dataclass(frozen=True)
class Split:
    X_dev: pd.DataFrame
    y_dev: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    dev_ids: list[str]
    test_ids: list[str]


def make_split(df: pd.DataFrame, seed: int = SEED, test_size: float = TEST_SIZE) -> Split:
    """Stratified development/test split. The test partition is never used for tuning,
    calibration, threshold selection or EDA: only for the single final evaluation."""
    y = encode_target(df[TARGET_COL])
    dev_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=test_size, stratify=y, random_state=seed
    )
    dev_idx, test_idx = np.sort(dev_idx), np.sort(test_idx)
    dev, test = df.iloc[dev_idx], df.iloc[test_idx]
    # Training frames pass through the same schema coercion as live traffic, so the
    # fitted pipeline sees identical dtypes in training and serving.
    typed = assert_valid(df[INPUT_COLS].reset_index(drop=True)).frame
    return Split(
        X_dev=typed.iloc[dev_idx].reset_index(drop=True),
        y_dev=y[dev_idx],
        X_test=typed.iloc[test_idx].reset_index(drop=True),
        y_test=y[test_idx],
        dev_ids=dev[ID_COL].tolist(),
        test_ids=test[ID_COL].tolist(),
    )


def save_split(split: Split, data_sha256: str, path: Path = SPLIT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_sha256": data_sha256,
        "seed": SEED,
        "test_size": TEST_SIZE,
        "dev_ids": split.dev_ids,
        "test_ids": split.test_ids,
    }
    path.write_text(json.dumps(payload, indent=1))


def load_dev_frame(path: Path = DATA_PATH, split_path: Path = SPLIT_PATH) -> pd.DataFrame:
    """Development rows only, for EDA. Never exposes the held-out test rows."""
    df = pd.read_csv(path)
    dev_ids = set(json.loads(split_path.read_text())["dev_ids"])
    return df[df[ID_COL].isin(dev_ids)].reset_index(drop=True)
