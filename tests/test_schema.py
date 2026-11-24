import numpy as np
import pandas as pd
import pytest

from bnpl_risk.config import INPUT_COLS
from bnpl_risk.schema import SchemaError, assert_valid, training_support, validate_frame


def _with(row: dict, **changes) -> pd.DataFrame:
    return pd.DataFrame([{**row, **changes}])[INPUT_COLS]


def test_valid_row_passes(good_frame):
    res = assert_valid(good_frame)
    assert res.valid_mask.all()
    assert res.frame["Age"].dtype == float


def test_full_training_data_passes(raw):
    assert assert_valid(raw[INPUT_COLS]).valid_mask.all()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"Age": "thirty"}, "Age='thirty' is not numeric"),
        ({"Age": 12}, "Age=12 outside valid range"),
        ({"Age": 30.5}, "Age=30.5 must be an integer"),
        ({"Credit_Score": 900}, "Credit_Score=900 outside valid range"),
        ({"Credit_Score": 250}, "Credit_Score=250 outside valid range"),
        ({"Total_BNPL_Debt_USD": -10}, "Total_BNPL_Debt_USD=-10 outside valid range"),
        ({"Income_USD": -1}, "Income_USD=-1 outside valid range"),
        ({"Total_BNPL_Active_Loans": 2.5}, "must be an integer"),
        ({"Average_Transaction_Value_USD": np.inf}, "is not finite"),
        ({"Age": None}, "Age is required"),
        ({"Employment_Status": "Astronaut"}, "Employment_Status='Astronaut' not in"),
        ({"Late_Payment_History": "yes"}, "Late_Payment_History='yes' not in"),
        ({"Shopping_Category_Most_Frequent": None}, "is required"),
    ],
)
def test_malformed_rows_rejected_with_clear_message(good_row, changes, message):
    df = _with(good_row, **changes)
    res = validate_frame(df)
    assert not res.valid_mask[0]
    assert any(message in e for e in res.errors.iloc[0]), res.errors.iloc[0]
    with pytest.raises(SchemaError, match="1 invalid row"):
        assert_valid(df)


def test_missing_column_fails_fast(good_frame):
    with pytest.raises(SchemaError, match="missing required column"):
        validate_frame(good_frame.drop(columns=["Credit_Score"]))


def test_non_dataframe_rejected(good_row):
    with pytest.raises(TypeError):
        validate_frame([good_row])


@pytest.mark.parametrize("col", ["Income_USD", "Credit_Score"])
def test_nullable_columns_accept_missing(good_row, col):
    res = validate_frame(_with(good_row, **{col: np.nan}))
    assert res.valid_mask.all()


def test_extra_columns_are_ignored(good_frame):
    df = good_frame.assign(Customer_ID="C1", Unused=3)
    assert list(assert_valid(df).frame.columns) == INPUT_COLS


def test_out_of_support_values_are_flagged_not_silenced(good_row, raw):
    support = training_support(raw)
    df = _with(good_row, Total_BNPL_Debt_USD=23_500, Age=90)  # 8.6x the training max
    res = validate_frame(df, support)
    assert res.valid_mask.all()  # still a legal input...
    assert set(res.ood_flags.iloc[0]) == {"Total_BNPL_Debt_USD", "Age"}  # ...but flagged


def test_errors_are_reported_per_row(good_row):
    df = pd.concat([_with(good_row), _with(good_row, Age=5), _with(good_row)], ignore_index=True)
    res = validate_frame(df)
    assert res.valid_mask.tolist() == [True, False, True]
    with pytest.raises(SchemaError) as exc:
        assert_valid(df)
    assert list(exc.value.errors) == [1]
