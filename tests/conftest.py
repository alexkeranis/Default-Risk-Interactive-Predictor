"""Shared fixtures. Models are trained here on a small, seeded subset with tiny budgets,
so the suite runs in CI without the (git-ignored) production artifacts."""

from __future__ import annotations

import joblib
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from bnpl_risk.config import INPUT_COLS, SEED
from bnpl_risk.data import load_raw, make_split
from bnpl_risk.explain import base_pipeline
from bnpl_risk.inference import RiskScorer
from bnpl_risk.preprocessing import transform_features
from bnpl_risk.schema import training_support
from bnpl_risk.seed import seed_everything
from bnpl_risk.train import build_meta, fit_calibrated

FAST_PARAMS = {
    "logreg": {"C": 1.0},
    "xgboost": {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.2},
    "resnet": {"max_epochs": 4, "patience": 2, "d_block": 32, "n_blocks": 1},
}
# One calibration method per family so every calibration branch is exercised.
FAST_CALIBRATION = {"logreg": "temperature", "xgboost": "isotonic", "resnet": "sigmoid"}


@pytest.fixture(scope="session", autouse=True)
def _seeded():
    seed_everything()


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return load_raw()


@pytest.fixture(scope="session")
def split(raw):
    return make_split(raw)


@pytest.fixture(scope="session")
def small(split):
    """(X_train, y_train, X_holdout, y_holdout) drawn from the development split only."""
    X_tr, X_ho, y_tr, y_ho = train_test_split(
        split.X_dev,
        split.y_dev,
        train_size=2500,
        test_size=500,
        stratify=split.y_dev,
        random_state=SEED,
    )
    return (X_tr.reset_index(drop=True), y_tr, X_ho.reset_index(drop=True), y_ho)


@pytest.fixture(scope="session")
def fitted(small) -> dict:
    X, y, _, _ = small
    return {
        fam: fit_calibrated(fam, FAST_PARAMS[fam], X, y, FAST_CALIBRATION[fam])
        for fam in FAST_PARAMS
    }


@pytest.fixture(scope="session")
def scorer(fitted, small, tmp_path_factory) -> RiskScorer:
    X, _, _, _ = small
    d = tmp_path_factory.mktemp("artifacts")
    model = fitted["xgboost"]
    meta = build_meta(
        family="xgboost",
        params=FAST_PARAMS["xgboost"],
        calibration=FAST_CALIBRATION["xgboost"],
        support=training_support(X),
        model_features=list(transform_features(base_pipeline(model), X.head(1)).columns),
        cutoff=0.05,
    )
    import json

    joblib.dump(model, d / "risk_model.joblib")
    (d / "model_meta.json").write_text(json.dumps(meta))
    return RiskScorer.load(d / "risk_model.joblib", d / "model_meta.json")


@pytest.fixture
def good_row() -> dict:
    return {
        "Age": 31,
        "Income_USD": 42000.0,
        "Credit_Score": 640.0,
        "Total_BNPL_Active_Loans": 2,
        "Total_BNPL_Debt_USD": 380,
        "Average_Transaction_Value_USD": 220,
        "Employment_Status": "Freelancer",
        "Late_Payment_History": "No",
        "Shopping_Category_Most_Frequent": "Fashion",
    }


@pytest.fixture
def good_frame(good_row) -> pd.DataFrame:
    return pd.DataFrame([good_row])[INPUT_COLS]
