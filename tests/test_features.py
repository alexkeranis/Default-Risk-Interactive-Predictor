import numpy as np
import pandas as pd
import pytest

from bnpl_risk.config import INCOME_FLOOR_USD
from bnpl_risk.features import ENGINEERED_FEATURES, FinancialFeatureEngineer, engineer
from bnpl_risk.preprocessing import NumericImputer


def test_hand_computed_values(good_frame):
    f = engineer(good_frame).iloc[0]
    assert f["bnpl_dti"] == pytest.approx(380 / 42000)
    assert f["ticket_to_monthly_income"] == pytest.approx(220 / (42000 / 12))
    assert f["debt_per_loan"] == pytest.approx(190)
    assert f["loan_stacking"] == 0
    assert f["subprime"] == 0
    assert f["near_prime"] == 1  # 640 is in [580, 670)
    assert f["unstable_income"] == 0
    assert f["late_x_dti"] == 0
    assert f["log_income"] == pytest.approx(np.log1p(42000))
    assert f["young_borrower"] == 0


def test_zero_income_maps_to_maximum_leverage_not_zero(good_frame):
    """Income 0 must never yield DTI 0, which would score as the 'safest' applicant."""
    f = engineer(good_frame.assign(Income_USD=0.0)).iloc[0]
    assert f["bnpl_dti"] == pytest.approx(380 / INCOME_FLOOR_USD)
    assert f["bnpl_dti"] > engineer(good_frame).iloc[0]["bnpl_dti"]


def test_zero_loans_and_extremes_give_finite_features(good_frame):
    df = pd.concat(
        [
            good_frame.assign(Total_BNPL_Active_Loans=0, Total_BNPL_Debt_USD=0),
            good_frame.assign(Income_USD=0.0, Average_Transaction_Value_USD=1e6),
        ],
        ignore_index=True,
    )
    f = engineer(df)
    assert np.isfinite(f.to_numpy()).all()
    assert f.loc[0, "debt_per_loan"] == 0


def test_interactions(good_frame):
    df = good_frame.assign(
        Late_Payment_History="Yes", Credit_Score=550.0, Employment_Status="Student"
    )
    f = engineer(df).iloc[0]
    dti = 380 / 42000
    assert f["late_x_dti"] == pytest.approx(dti)
    assert f["unstable_x_dti"] == pytest.approx(dti)
    assert f["late_x_subprime"] == 1


def test_transformer_is_stateless(small):
    X, _, X_ho, _ = small
    a = FinancialFeatureEngineer().fit(X).transform(X_ho.fillna(0))
    b = FinancialFeatureEngineer().fit(X_ho).transform(X_ho.fillna(0))
    pd.testing.assert_frame_equal(a, b)


def test_raw_mode_is_passthrough(good_frame):
    out = FinancialFeatureEngineer(mode="raw").fit(good_frame).transform(good_frame)
    pd.testing.assert_frame_equal(out, good_frame)


def test_feature_names_out(good_frame):
    fe = FinancialFeatureEngineer().fit(good_frame)
    assert list(fe.get_feature_names_out()) == [*good_frame.columns, *ENGINEERED_FEATURES]
    assert list(fe.transform(good_frame).columns) == list(fe.get_feature_names_out())


def test_imputer_always_emits_indicators(good_frame):
    imp = NumericImputer().fit(good_frame)
    out = imp.transform(good_frame)
    assert {"missing_Income_USD", "missing_Credit_Score"} <= set(out.columns)
    assert list(out.columns) == list(imp.get_feature_names_out())
