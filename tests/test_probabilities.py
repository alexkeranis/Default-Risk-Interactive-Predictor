import numpy as np
import pytest

from bnpl_risk.explain import base_pipeline


@pytest.mark.parametrize("family", ["logreg", "xgboost", "resnet"])
@pytest.mark.parametrize("stage", ["calibrated", "uncalibrated"])
def test_probabilities_are_valid_distributions(fitted, small, family, stage):
    _, _, X_ho, _ = small
    model = fitted[family] if stage == "calibrated" else base_pipeline(fitted[family])
    p = model.predict_proba(X_ho)
    assert p.shape == (len(X_ho), 3)
    assert np.isfinite(p).all()
    assert (p >= 0).all()
    assert (p <= 1).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-9)


def test_scorer_output_columns_are_consistent(scorer, small):
    _, _, X_ho, _ = small
    out = scorer.score(X_ho)
    probs = out[["Prob_Low", "Prob_Medium", "Prob_High"]].to_numpy()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    pd_eff = probs @ np.asarray(scorer.cost.pd_by_tier)
    np.testing.assert_allclose(out["Expected_PD"], pd_eff)
    expected_loss = X_ho["Average_Transaction_Value_USD"] * scorer.cost.lgd * pd_eff
    np.testing.assert_allclose(out["Expected_Loss_USD"], expected_loss)
    assert set(out["Decision"]) <= {"APPROVE", "DECLINE"}
    assert ((out["Expected_PD"] <= scorer.cutoff) == (out["Decision"] == "APPROVE")).all()
    assert set(out["Risk_Tier"]) <= {"Low", "Medium", "High"}


def test_model_ranks_risk_sensibly(fitted, small):
    """Sanity, not performance: a fast model must still separate the extreme tiers."""
    from sklearn.metrics import roc_auc_score

    _, _, X_ho, y_ho = small
    p = fitted["xgboost"].predict_proba(X_ho)
    assert roc_auc_score(y_ho == 2, p[:, 2]) > 0.9


def test_prior_shift_correction_recovers_true_prior():
    from bnpl_risk.evaluation import prior_shift_correction

    true_prior = np.array([0.88, 0.068, 0.052])
    uniform = np.full((4, 3), 1 / 3)  # what a balanced-weight model predicts with no signal
    corrected = prior_shift_correction(uniform, true_prior, np.ones(3))
    np.testing.assert_allclose(corrected, np.tile(true_prior, (4, 1)))
    np.testing.assert_allclose(corrected.sum(axis=1), 1.0)
