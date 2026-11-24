"""The single preprocessing graph used by training, cross-validation and serving.

    raw inputs -> NumericImputer -> FinancialFeatureEngineer -> encoder -> estimator

Every stateful step is fit inside ``Pipeline.fit``, so within cross-validation it only
ever sees the training fold."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from bnpl_risk.config import CATEGORICAL_COLS, CATEGORY_LEVELS, NULLABLE_COLS, NUMERIC_COLS
from bnpl_risk.features import FinancialFeatureEngineer

INDICATOR_PREFIX = "missing_"


class NumericImputer(TransformerMixin, BaseEstimator):
    """Fills ``NULLABLE_COLS`` and always appends a missing-value indicator per nullable
    column (a stable output schema, even if a fold or a batch has no missing values).

    ``strategy="median"`` stores per-column training medians. ``strategy="knn"`` fits a
    standardised KNN imputer on all numeric inputs (the original repo's approach, kept as
    an ablation)."""

    def __init__(self, strategy: str = "median", n_neighbors: int = 5):
        self.strategy = strategy
        self.n_neighbors = n_neighbors

    def fit(self, X: pd.DataFrame, y=None):
        num = X[NUMERIC_COLS].astype(float)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.medians_ = {c: float(num[c].median()) for c in NULLABLE_COLS}
        if self.strategy == "knn":
            self.mean_ = num.mean().to_numpy()
            self.scale_ = num.std(ddof=0).replace(0, 1).to_numpy()
            self.knn_ = KNNImputer(n_neighbors=self.n_neighbors, weights="distance")
            self.knn_.fit((num.to_numpy() - self.mean_) / self.scale_)
        elif self.strategy != "median":
            raise ValueError(f"Unknown imputation strategy {self.strategy!r}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        indicators = {f"{INDICATOR_PREFIX}{c}": X[c].isna().astype(float) for c in NULLABLE_COLS}
        if self.strategy == "knn" and X[NULLABLE_COLS].isna().to_numpy().any():
            num = X[NUMERIC_COLS].astype(float).to_numpy()
            filled = (
                self.knn_.transform((num - self.mean_) / self.scale_) * self.scale_ + self.mean_
            )
            for j, c in enumerate(NUMERIC_COLS):
                if c in NULLABLE_COLS:
                    X[c] = np.where(np.isnan(num[:, j]), filled[:, j], num[:, j])
        else:
            for c in NULLABLE_COLS:
                X[c] = X[c].astype(float).fillna(self.medians_[c])
        for name, col in indicators.items():
            X[name] = col
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            [*self.feature_names_in_, *(f"{INDICATOR_PREFIX}{c}" for c in NULLABLE_COLS)]
        )


def make_encoder(family: str) -> ColumnTransformer:
    """Linear/NN: standardised numerics. Trees: raw numerics (scale-invariant).
    Linear/trees: one-hot categoricals with a fixed level order. NN: ordinal codes
    (-1 = unknown) feeding entity embeddings; categoricals come last by construction."""
    levels = [CATEGORY_LEVELS[c] for c in CATEGORICAL_COLS]
    numeric = make_column_selector(dtype_include=np.number)
    if family == "resnet":
        cat = OrdinalEncoder(
            categories=levels, handle_unknown="use_encoded_value", unknown_value=-1
        )
    else:
        cat = OneHotEncoder(categories=levels, handle_unknown="ignore", sparse_output=False)
    num = "passthrough" if family == "xgboost" else StandardScaler()
    return ColumnTransformer(
        [("num", num, numeric), ("cat", cat, CATEGORICAL_COLS)],
        verbose_feature_names_out=False,
    )


def build_pipeline(
    family: str,
    estimator: BaseEstimator,
    features: str = "engineered",
    imputer: str = "median",
) -> Pipeline:
    pipe = Pipeline(
        [
            ("impute", NumericImputer(strategy=imputer)),
            ("features", FinancialFeatureEngineer(mode=features)),
            ("encode", make_encoder(family)),
            ("model", estimator),
        ]
    )
    return pipe.set_output(transform="pandas")


def transform_features(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Model-ready matrix produced by a fitted pipeline (everything except the estimator)."""
    return pipeline[:-1].transform(X)
