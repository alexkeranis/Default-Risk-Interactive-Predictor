import numpy as np
import pytest

from bnpl_risk.models import make_model
from tests.conftest import FAST_PARAMS


@pytest.mark.parametrize("family", ["logreg", "xgboost", "resnet"])
def test_same_seed_same_model(small, family):
    X, y, X_ho, _ = small
    a = make_model(family, **FAST_PARAMS[family]).fit(X, y).predict_proba(X_ho)
    b = make_model(family, **FAST_PARAMS[family]).fit(X, y).predict_proba(X_ho)
    np.testing.assert_array_equal(a, b)


def test_different_seed_changes_stochastic_models(small):
    X, y, X_ho, _ = small
    a = make_model("resnet", **FAST_PARAMS["resnet"], random_state=1).fit(X, y)
    b = make_model("resnet", **FAST_PARAMS["resnet"], random_state=2).fit(X, y)
    assert not np.array_equal(a.predict_proba(X_ho), b.predict_proba(X_ho))


def test_repeated_prediction_is_stable(fitted, small):
    _, _, X_ho, _ = small
    for model in fitted.values():
        np.testing.assert_array_equal(model.predict_proba(X_ho), model.predict_proba(X_ho))


def test_resnet_roundtrips_through_pickle(fitted, small, tmp_path):
    import joblib

    _, _, X_ho, _ = small
    joblib.dump(fitted["resnet"], tmp_path / "m.joblib")
    loaded = joblib.load(tmp_path / "m.joblib")
    np.testing.assert_array_equal(loaded.predict_proba(X_ho), fitted["resnet"].predict_proba(X_ho))
