"""XGBoost wrapped as a plain scikit-learn classifier with an explicit ``class_weight``
parameter, so class weighting is tuned like any other hyperparameter."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


class XGBRiskClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        min_child_weight: float = 1.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        class_weight: str | None = None,
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        weights = compute_sample_weight(self.class_weight, y) if self.class_weight else None
        self.booster_ = XGBClassifier(
            objective="multi:softprob",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric="mlogloss",
        )
        self.booster_.fit(X, y, sample_weight=weights)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # XGBoost computes softmax in float32; renormalise in float64 so rows sum to 1.
        p = self.booster_.predict_proba(X).astype(np.float64)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
