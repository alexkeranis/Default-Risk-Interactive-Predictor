"""Single source of truth for paths, schema, experiment design and cost assumptions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "BNPL_Financial_Default_Risk_Dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DATA_DIR = PROJECT_ROOT / "docs" / "data"

MODEL_PATH = MODELS_DIR / "risk_model.joblib"
META_PATH = MODELS_DIR / "model_meta.json"
SPLIT_PATH = REPORTS_DIR / "split_indices.json"

SEED = 42

# --------------------------------------------------------------------------- schema
ID_COL = "Customer_ID"
TARGET_COL = "Default_Risk"
CLASSES: tuple[str, ...] = ("Low", "Medium", "High")
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}

NUMERIC_COLS: list[str] = [
    "Age",
    "Income_USD",
    "Credit_Score",
    "Total_BNPL_Active_Loans",
    "Total_BNPL_Debt_USD",
    "Average_Transaction_Value_USD",
]
# Only these two columns are missing in the source data (~3%, MCAR); they are imputed.
NULLABLE_COLS: list[str] = ["Income_USD", "Credit_Score"]
INTEGER_COLS: list[str] = ["Age", "Total_BNPL_Active_Loans"]

CATEGORY_LEVELS: dict[str, list[str]] = {
    "Employment_Status": ["Employed", "Freelancer", "Student", "Unemployed"],
    "Late_Payment_History": ["No", "Yes"],
    "Shopping_Category_Most_Frequent": [
        "Electronics",
        "Fashion",
        "Groceries/Essentials",
        "Home/Furniture",
        "Travel",
    ],
}
CATEGORICAL_COLS: list[str] = list(CATEGORY_LEVELS)
INPUT_COLS: list[str] = NUMERIC_COLS + CATEGORICAL_COLS

# Hard validity limits: values outside are rejected, not scored.
# Upper bound None = only finiteness is required (the training-support check flags extremes).
HARD_LIMITS: dict[str, tuple[float, float | None]] = {
    "Age": (18, 100),
    "Income_USD": (0, None),
    "Credit_Score": (300, 850),
    "Total_BNPL_Active_Loans": (0, 50),
    "Total_BNPL_Debt_USD": (0, None),
    "Average_Transaction_Value_USD": (0, None),
}

# Income below this is treated as this value inside ratio features. It equals the
# floor observed in the source data, so a zero/near-zero income maps to *maximum*
# leverage rather than to a division-by-zero fallback of 0 (the original bug).
INCOME_FLOOR_USD = 5_000.0

# ---------------------------------------------------------------- experiment design
TEST_SIZE = 0.20  # held out once, touched only by the final evaluation
CV_FOLDS = 5
CV_REPEATS = 3
TUNE_FOLDS = 5
N_ITER = 30  # equal random-search budget for XGBoost and the neural network
CALIBRATION_FOLDS = 5
BOOTSTRAP_ROUNDS = 1000
MODEL_FAMILIES: tuple[str, ...] = ("logreg", "xgboost", "resnet")
# Simplicity order used by the 1-SE selection rule (lower index = simpler).
SIMPLICITY_ORDER = {"logreg": 0, "xgboost": 1, "resnet": 2}


# ------------------------------------------------------------------ cost assumptions
@dataclass(frozen=True)
class CostAssumptions:
    """Illustrative unit economics. The labels are risk *tiers*, not observed defaults,
    so turning a tier into money requires an explicit, documented assumption.

    - ``pd_by_tier``: assumed 12-month probability of default for each tier.
    - ``lgd``: loss given default on an unsecured BNPL balance.
    - ``merchant_fee``: revenue per approved purchase as a fraction of its value.
    - ``cost_matrix[true][pred]``: relative cost of assigning tier ``pred`` to a
      customer whose true tier is ``true``. Under-calling risk costs more than
      over-calling it.
    """

    pd_by_tier: tuple[float, float, float] = (0.02, 0.15, 0.45)
    lgd: float = 0.85
    merchant_fee: float = 0.04
    cost_matrix: tuple[tuple[float, ...], ...] = field(
        default=(
            (0.0, 1.0, 2.0),
            (5.0, 0.0, 1.0),
            (10.0, 3.0, 0.0),
        )
    )

    def to_dict(self) -> dict:
        return asdict(self)


COST = CostAssumptions()
