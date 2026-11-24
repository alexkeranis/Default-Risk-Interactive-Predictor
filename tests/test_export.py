"""The JSON exported for the browser demo must reproduce the fitted Python models.
``reference_predict`` mirrors docs/assets/model.js; the Node test then checks the JS
against golden cases produced from the same fitted pipelines."""

import json

import numpy as np
import pytest

from bnpl_risk.config import DOCS_DATA_DIR
from bnpl_risk.explain import base_pipeline
from bnpl_risk.export_web import (
    export_linear,
    export_xgb,
    model_contributions,
    model_feature_frame,
    reference_predict,
)
from bnpl_risk.preprocessing import transform_features
from bnpl_risk.train import fit_calibrated
from tests.conftest import FAST_PARAMS

METHODS = ["none", "sigmoid", "isotonic", "temperature"]


@pytest.fixture(scope="module")
def probe(small):
    X, y, X_ho, _ = small
    return X, y, X_ho.head(120)


@pytest.mark.parametrize("method", METHODS)
def test_linear_export_reproduces_pipeline(probe, method):
    X, y, X_eval = probe
    model = fit_calibrated("logreg", FAST_PARAMS["logreg"], X, y, method)
    spec = json.loads(json.dumps(export_linear(model)))  # must survive JSON
    p, phi = reference_predict(spec, X_eval)
    np.testing.assert_allclose(p, model.predict_proba(X_eval), rtol=0, atol=1e-9)
    np.testing.assert_allclose(phi, model_contributions(model, X_eval), rtol=0, atol=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_tree_export_reproduces_pipeline(probe, method):
    X, y, X_eval = probe
    model = fit_calibrated("xgboost", FAST_PARAMS["xgboost"], X, y, method)
    spec = json.loads(json.dumps(export_xgb(model, X.dropna().head(100))))
    p, phi = reference_predict(spec, X_eval)
    # XGBoost accumulates margins in float32, the reference in float64.
    np.testing.assert_allclose(p, model.predict_proba(X_eval), rtol=0, atol=1e-5)
    np.testing.assert_allclose(phi, model_contributions(model, X_eval), rtol=0, atol=1e-4)


def test_saabas_contributions_sum_to_margin(fitted, probe):
    """Local attributions are complete: features + bias = the model's raw margin."""
    _, _, X_eval = probe
    model = fitted["xgboost"]
    phi = model_contributions(model, X_eval)
    pipe = base_pipeline(model)
    margin = pipe[-1].booster_.predict(transform_features(pipe, X_eval), output_margin=True)
    np.testing.assert_allclose(phi.sum(axis=2), margin, atol=1e-4)


def test_reference_features_match_pipeline_before_scaling(fitted, probe):
    _, _, X_eval = probe
    pipe = base_pipeline(fitted["xgboost"])  # xgboost encoder = passthrough numerics
    expected = transform_features(pipe, X_eval)
    got = model_feature_frame(X_eval, pipe.named_steps["impute"].medians_)[list(expected.columns)]
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-12)


def test_published_golden_cases_are_consistent():
    """If the docs bundle exists, its golden cases must match its own model.json."""
    model_path, golden_path = DOCS_DATA_DIR / "model.json", DOCS_DATA_DIR / "golden_cases.json"
    if not (model_path.exists() and golden_path.exists()):
        pytest.skip("docs data not exported yet")
    import pandas as pd

    spec = json.loads(model_path.read_text())
    golden = json.loads(golden_path.read_text())
    inputs = pd.DataFrame(golden["inputs"])
    for fam, model_spec in spec["models"].items():
        p, _ = reference_predict(model_spec, inputs)
        np.testing.assert_allclose(p, golden["models"][fam]["proba"], atol=1e-5)
