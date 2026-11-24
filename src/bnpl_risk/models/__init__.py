"""Model families. Each returns a scikit-learn classifier that plugs into
``preprocessing.build_pipeline`` so all families share one evaluation harness."""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from bnpl_risk.config import SEED


def make_estimator(family: str, **params) -> BaseEstimator:
    if family == "logreg":
        defaults = {"C": 1.0, "max_iter": 5000, "random_state": SEED}
        return LogisticRegression(**{**defaults, **params})
    if family == "xgboost":
        from bnpl_risk.models.gbdt import XGBRiskClassifier

        return XGBRiskClassifier(**{"random_state": SEED, **params})
    if family == "resnet":
        from bnpl_risk.models.tabular_nn import TabularResNetClassifier

        return TabularResNetClassifier(**{"random_state": SEED, **params})
    raise ValueError(f"Unknown model family {family!r}")


def make_model(family: str, features: str = "engineered", imputer: str = "median", **params):
    """Full pipeline (imputation -> features -> encoding -> estimator) for a family."""
    from bnpl_risk.preprocessing import build_pipeline

    return build_pipeline(family, make_estimator(family, **params), features, imputer)
