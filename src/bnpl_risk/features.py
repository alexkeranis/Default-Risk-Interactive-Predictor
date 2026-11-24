"""Domain feature engineering. Stateless: every feature is a deterministic function of a
single row, so ``fit`` learns nothing and cannot leak information across rows or folds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from bnpl_risk.config import INCOME_FLOOR_USD

# Name -> financial rationale. Rendered in the README and stored in model_meta.json.
FEATURE_RATIONALE: dict[str, str] = {
    "bnpl_dti": "Outstanding BNPL debt / annual income. Core affordability and leverage "
    "measure. Income is floored, so zero income means maximum leverage, never zero.",
    "ticket_to_monthly_income": "Typical purchase size / monthly income. How large each new "
    "instalment plan is relative to the cash flow that must service it.",
    "debt_per_loan": "Average balance per active plan. Separates a few large commitments from "
    "many small ones; 0 when there are no active plans.",
    "loan_stacking": "Flag for 3 or more concurrent BNPL plans. Stacking across providers "
    "is a known stress signal that bureau data often misses.",
    "subprime": "Credit score below 580 (FICO 'poor' band). A non-linear threshold a linear "
    "model cannot represent from the raw score.",
    "near_prime": "Credit score 580-669 (FICO 'fair' band).",
    "unstable_income": "Student or unemployed: income that is irregular or transfer-based.",
    "late_x_dti": "Late payment history x leverage. Past delinquency matters more when the "
    "customer is already stretched.",
    "unstable_x_dti": "Unstable income x leverage. Debt is riskier when income is unreliable.",
    "late_x_subprime": "Late payment history x subprime score: compounding delinquency signals.",
    "log_income": "log(1 + income). Diminishing marginal effect of income on repayment capacity.",
    "log_debt": "log(1 + BNPL debt). Diminishing marginal effect of balance size.",
    "young_borrower": "Age under 25: a proxy for a thin credit file.",
}
ENGINEERED_FEATURES: list[str] = list(FEATURE_RATIONALE)


def engineer(df: pd.DataFrame, income_floor: float = INCOME_FLOOR_USD) -> pd.DataFrame:
    """Return the engineered columns for an imputed input frame (no NaN in numerics)."""
    income = np.maximum(df["Income_USD"].to_numpy(dtype=float), income_floor)
    debt = df["Total_BNPL_Debt_USD"].to_numpy(dtype=float)
    loans = df["Total_BNPL_Active_Loans"].to_numpy(dtype=float)
    ticket = df["Average_Transaction_Value_USD"].to_numpy(dtype=float)
    score = df["Credit_Score"].to_numpy(dtype=float)
    late = (df["Late_Payment_History"] == "Yes").to_numpy(dtype=float)
    unstable = df["Employment_Status"].isin(["Student", "Unemployed"]).to_numpy(dtype=float)

    dti = debt / income
    subprime = (score < 580).astype(float)
    out = {
        "bnpl_dti": dti,
        "ticket_to_monthly_income": ticket / (income / 12.0),
        "debt_per_loan": np.divide(debt, loans, out=np.zeros_like(debt), where=loans > 0),
        "loan_stacking": (loans >= 3).astype(float),
        "subprime": subprime,
        "near_prime": ((score >= 580) & (score < 670)).astype(float),
        "unstable_income": unstable,
        "late_x_dti": late * dti,
        "unstable_x_dti": unstable * dti,
        "late_x_subprime": late * subprime,
        "log_income": np.log1p(income),
        "log_debt": np.log1p(debt),
        "young_borrower": (df["Age"].to_numpy(dtype=float) < 25).astype(float),
    }
    return pd.DataFrame(out, index=df.index)[ENGINEERED_FEATURES]


class FinancialFeatureEngineer(TransformerMixin, BaseEstimator):
    """Appends ``ENGINEERED_FEATURES`` to the frame (``mode="engineered"``) or passes it
    through untouched (``mode="raw"``, used for the feature ablation)."""

    def __init__(self, mode: str = "engineered", income_floor: float = INCOME_FLOOR_USD):
        self.mode = mode
        self.income_floor = income_floor

    def fit(self, X: pd.DataFrame, y=None):
        if self.mode not in ("engineered", "raw"):
            raise ValueError(f"mode must be 'engineered' or 'raw', got {self.mode!r}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.mode == "raw":
            return X
        return pd.concat([X, engineer(X, self.income_floor)], axis=1)

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        return np.asarray(names + (ENGINEERED_FEATURES if self.mode == "engineered" else []))
