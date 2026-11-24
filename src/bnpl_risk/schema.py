"""Input contract for training data and live traffic.

Policy:
- Hard errors (row rejected): missing column, non-numeric value, non-finite value,
  value outside ``HARD_LIMITS``, non-integer count, unknown or missing category,
  missing value in a non-nullable column.
- Allowed: missing ``Income_USD`` / ``Credit_Score`` (imputed exactly as in training).
- Flagged, still scored: values outside the *training support*, i.e. the model is
  extrapolating. Flags are surfaced in the output, never silenced.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bnpl_risk.config import (
    CATEGORICAL_COLS,
    CATEGORY_LEVELS,
    HARD_LIMITS,
    INPUT_COLS,
    INTEGER_COLS,
    NULLABLE_COLS,
    NUMERIC_COLS,
)


class SchemaError(ValueError):
    """Raised when input rows violate the input contract. Key ``-1`` = frame-level error."""

    def __init__(self, errors: dict[int, list[str]]):
        self.errors = errors
        if -1 in errors:
            super().__init__("; ".join(errors[-1]))
            return
        preview = "; ".join(f"row {i}: {', '.join(msgs)}" for i, msgs in list(errors.items())[:5])
        more = f" (+{len(errors) - 5} more rows)" if len(errors) > 5 else ""
        super().__init__(f"{len(errors)} invalid row(s): {preview}{more}")


@dataclass
class ValidationResult:
    frame: pd.DataFrame  # typed copy of the input columns (numeric coerced to float)
    errors: pd.Series  # list[str] per row, empty list = valid
    ood_flags: pd.Series  # list[str] per row: columns outside the training support

    @property
    def valid_mask(self) -> np.ndarray:
        return self.errors.map(len).eq(0).to_numpy()


def validate_frame(
    df: pd.DataFrame, support: Mapping[str, Mapping[str, float]] | None = None
) -> ValidationResult:
    """Validate and type-coerce input rows. ``support`` maps numeric columns to the
    ``{"min": .., "max": ..}`` range observed in training (from ``model_meta.json``)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")
    missing = [c for c in INPUT_COLS if c not in df.columns]
    if missing:
        raise SchemaError({-1: [f"missing required column(s) {missing}"]})

    n = len(df)
    errors: list[list[str]] = [[] for _ in range(n)]
    ood: list[list[str]] = [[] for _ in range(n)]
    out = pd.DataFrame(index=df.index)

    for col in NUMERIC_COLS:
        raw = df[col]
        values = pd.to_numeric(raw, errors="coerce").astype(float)
        raw_missing = raw.isna().to_numpy()
        coerced_bad = values.isna().to_numpy() & ~raw_missing
        arr = values.to_numpy()
        lo, hi = HARD_LIMITS[col]
        for i in range(n):
            if coerced_bad[i]:
                errors[i].append(f"{col}={raw.iloc[i]!r} is not numeric")
                continue
            v = arr[i]
            if raw_missing[i]:
                if col not in NULLABLE_COLS:
                    errors[i].append(f"{col} is required")
                continue
            if not np.isfinite(v):
                errors[i].append(f"{col}={v} is not finite")
                continue
            if v < lo or (hi is not None and v > hi):
                bound = f"[{lo}, {hi}]" if hi is not None else f">= {lo}"
                errors[i].append(f"{col}={v:g} outside valid range {bound}")
                continue
            if col in INTEGER_COLS and v != np.floor(v):
                errors[i].append(f"{col}={v:g} must be an integer")
                continue
            if support and col in support:
                s = support[col]
                if v < s["min"] or v > s["max"]:
                    ood[i].append(col)
        out[col] = values

    for col in CATEGORICAL_COLS:
        levels = CATEGORY_LEVELS[col]
        raw = df[col]
        for i in range(n):
            v = raw.iloc[i]
            if pd.isna(v):
                errors[i].append(f"{col} is required")
            elif str(v) not in levels:
                errors[i].append(f"{col}={v!r} not in {levels}")
        out[col] = (
            raw.astype(object).where(raw.notna(), None).map(lambda v: None if v is None else str(v))
        )

    return ValidationResult(
        frame=out[INPUT_COLS],
        errors=pd.Series(errors, index=df.index, dtype=object),
        ood_flags=pd.Series(ood, index=df.index, dtype=object),
    )


def assert_valid(
    df: pd.DataFrame, support: Mapping[str, Mapping[str, float]] | None = None
) -> ValidationResult:
    result = validate_frame(df, support)
    bad = {int(pos): msgs for pos, msgs in enumerate(result.errors.tolist()) if msgs}
    if bad:
        raise SchemaError(bad)
    return result


def training_support(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Observed min/max of each numeric input, used to flag extrapolation at serve time."""
    return {col: {"min": float(df[col].min()), "max": float(df[col].max())} for col in NUMERIC_COLS}
