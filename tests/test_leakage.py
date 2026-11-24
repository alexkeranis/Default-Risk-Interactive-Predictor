"""Every fitted statistic must come from the training rows only."""

import json
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold

from bnpl_risk.config import ID_COL, NULLABLE_COLS, SEED, SPLIT_PATH, TARGET_COL
from bnpl_risk.data import encode_target, make_split
from bnpl_risk.features import FinancialFeatureEngineer
from bnpl_risk.models import make_model
from bnpl_risk.preprocessing import NumericImputer
from bnpl_risk.train import run_cv


def test_dev_and_test_are_disjoint_and_complete(raw, split):
    dev, test = set(split.dev_ids), set(split.test_ids)
    assert not dev & test
    assert dev | test == set(raw[ID_COL])


def test_split_is_stratified_and_deterministic(raw, split):
    again = make_split(raw)
    assert again.test_ids == split.test_ids
    full = np.bincount(encode_target(raw[TARGET_COL])) / len(raw)
    test = np.bincount(split.y_test) / len(split.y_test)
    np.testing.assert_allclose(full, test, atol=0.002)


def test_committed_split_matches_code(split):
    if not SPLIT_PATH.exists():
        pytest.skip("split_indices.json is produced by the training run")
    saved = json.loads(SPLIT_PATH.read_text())
    assert saved["test_ids"] == split.test_ids


def test_fitted_statistics_equal_train_only_statistics(small):
    X, y, X_ho, _ = small
    pipe = make_model("logreg").fit(X, y)
    imp = pipe.named_steps["impute"]
    pooled = pd.concat([X, X_ho])
    for c in NULLABLE_COLS:
        assert imp.medians_[c] == pytest.approx(X[c].median())
    # The probe is only meaningful if pooling would have changed the statistic.
    assert pooled["Income_USD"].median() != X["Income_USD"].median()
    # The scaler sees the engineered training features, nothing else.
    engineered = FinancialFeatureEngineer().fit(X).transform(imp.transform(X))
    scaler = pipe.named_steps["encode"].named_transformers_["num"]
    expected = engineered[list(scaler.feature_names_in_)].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, expected, rtol=1e-12)


class _SpyClassifier(ClassifierMixin, BaseEstimator):
    """Records the row labels it is trained on (class-level, survives cloning)."""

    seen: ClassVar[list[set]] = []

    def fit(self, X, y):
        _SpyClassifier.seen.append(set(X.index))
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return np.full((len(X), len(self.classes_)), 1 / len(self.classes_))


def test_cv_harness_never_trains_on_validation_rows(small):
    X, y, _, _ = small
    splits = list(StratifiedKFold(4, shuffle=True, random_state=SEED).split(X, y))
    _SpyClassifier.seen.clear()
    run_cv(_SpyClassifier(), X, y, splits, n_jobs=1)
    assert len(_SpyClassifier.seen) == len(splits)
    for seen, (tr, va) in zip(_SpyClassifier.seen, splits, strict=True):
        assert seen == set(X.index[tr])
        assert not seen & set(X.index[va])


def test_cv_folds_refit_preprocessing_per_fold(small):
    """Each fold's imputer medians must equal that fold's training medians."""
    X, y, _, _ = small
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    medians = []
    for tr, _ in cv.split(X, y):
        imp = NumericImputer().fit(X.iloc[tr])
        assert imp.medians_["Income_USD"] == pytest.approx(X.iloc[tr]["Income_USD"].median())
        medians.append(imp.medians_["Income_USD"])
    assert len(set(medians)) > 1  # genuinely fold-specific


def test_transform_is_row_independent(fitted, small):
    _, _, X_ho, _ = small
    model = fitted["xgboost"]
    batch = model.predict_proba(X_ho)
    one_by_one = np.vstack([model.predict_proba(X_ho.iloc[[i]]) for i in range(20)])
    np.testing.assert_allclose(batch[:20], one_by_one, rtol=0, atol=1e-12)
