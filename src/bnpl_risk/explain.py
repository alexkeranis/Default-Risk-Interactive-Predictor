"""Feature attributions.

- XGBoost: exact TreeSHAP from the booster itself (``pred_contribs=True``), with no
  dependency on the ``shap`` package. Local "Saabas" path attributions
  (``approx_contribs=True``) are what the browser demo reproduces.
- Logistic regression: exact linear contributions ``coef_k * x_scaled`` (the
  standardised feature is centred on the training mean, so contributions are relative to
  an average applicant).
- All families: permutation importance on the raw inputs, a model-agnostic common currency.

All attributions are in the base model's log-odds (margin) space, i.e. *before* post-hoc
calibration, which is a monotone per-class transform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from bnpl_risk.config import CLASSES, INPUT_COLS, SEED
from bnpl_risk.preprocessing import transform_features


def base_pipeline(model) -> Pipeline:
    """Unwrap ``CalibratedClassifierCV(ensemble=False)`` to the fitted base pipeline."""
    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator
    return model


def xgb_contributions(pipeline: Pipeline, X: pd.DataFrame, approx: bool = False) -> np.ndarray:
    """(n_samples, n_classes, n_features + 1); last column is the bias term."""
    Z = transform_features(pipeline, X)
    booster = pipeline[-1].booster_.get_booster()
    return booster.predict(xgb.DMatrix(Z), pred_contribs=True, approx_contribs=approx)


def linear_contributions(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    Z = transform_features(pipeline, X).to_numpy()
    lr = pipeline[-1]
    contrib = Z[:, None, :] * lr.coef_[None, :, :]
    bias = np.broadcast_to(lr.intercept_[None, :, None], (len(Z), len(CLASSES), 1))
    return np.concatenate([contrib, bias], axis=2)


def contributions(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = list(transform_features(pipeline, X.head(1)).columns)
    if hasattr(pipeline[-1], "booster_"):
        return xgb_contributions(pipeline, X), names
    if hasattr(pipeline[-1], "coef_"):
        return linear_contributions(pipeline, X), names
    raise TypeError(f"No exact attribution for {type(pipeline[-1]).__name__}")


def global_importance(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Mean |contribution| per model feature and class (bias excluded)."""
    phi, names = contributions(pipeline, X)
    mean_abs = np.abs(phi[:, :, :-1]).mean(axis=0)  # (classes, features)
    df = pd.DataFrame(mean_abs.T, index=names, columns=list(CLASSES))
    df["total"] = df.sum(axis=1)
    return df.sort_values("total", ascending=False)


def raw_permutation_importance(
    model, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 5
) -> pd.DataFrame:
    """Increase in log loss when one raw input column is shuffled."""
    r = permutation_importance(
        model, X, y, scoring="neg_log_loss", n_repeats=n_repeats, random_state=SEED, n_jobs=1
    )
    return pd.DataFrame(
        {"importance": r.importances_mean, "std": r.importances_std}, index=INPUT_COLS
    ).sort_values("importance", ascending=False)
