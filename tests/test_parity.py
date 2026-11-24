"""Train/serve parity: live single-record and batch inference must produce exactly the
features and probabilities of the training pipeline."""

import json

import numpy as np
import pandas as pd
import pytest

from bnpl_risk import inference
from bnpl_risk.explain import base_pipeline
from bnpl_risk.preprocessing import transform_features
from bnpl_risk.schema import SchemaError


def test_serving_features_equal_training_pipeline_features(scorer, small):
    _, _, X_ho, _ = small
    training_view = transform_features(base_pipeline(scorer.model), X_ho)
    serving_view = scorer.transform(X_ho)
    pd.testing.assert_frame_equal(serving_view, training_view)
    assert list(serving_view.columns) == scorer.meta["model_features"]


def test_single_equals_batch(scorer, small):
    _, _, X_ho, _ = small
    batch_feats = scorer.transform(X_ho).to_numpy()
    batch_proba = scorer.predict_proba(X_ho)
    for i in range(25):
        row = X_ho.iloc[[i]]
        np.testing.assert_array_equal(scorer.transform(row).to_numpy()[0], batch_feats[i])
        np.testing.assert_allclose(scorer.predict_proba(row)[0], batch_proba[i], rtol=0, atol=1e-12)
        single = scorer.score_one(X_ho.iloc[i].to_dict())
        np.testing.assert_allclose(
            [single["probabilities"][c] for c in ("Low", "Medium", "High")],
            batch_proba[i],
            rtol=0,
            atol=1e-12,
        )


def test_row_order_does_not_matter(scorer, small):
    _, _, X_ho, _ = small
    perm = np.random.default_rng(0).permutation(len(X_ho))
    p = scorer.predict_proba(X_ho)
    p_perm = scorer.predict_proba(X_ho.iloc[perm])
    np.testing.assert_allclose(p_perm, p[perm], rtol=0, atol=1e-12)


def test_missing_values_are_imputed_like_training(scorer, small):
    _, _, X_ho, _ = small
    row = X_ho.iloc[[0]].assign(Income_USD=np.nan)
    medians = base_pipeline(scorer.model).named_steps["impute"].medians_
    feats = scorer.transform(row).iloc[0]
    assert feats["Income_USD"] == medians["Income_USD"]
    assert feats["missing_Income_USD"] == 1.0


def test_invalid_rows_are_rejected_not_scored(scorer, small):
    _, _, X_ho, _ = small
    df = X_ho.head(3).copy()
    df.loc[1, "Employment_Status"] = "Astronaut"
    out = scorer.score(df)
    assert out.loc[1, "Decision"] == "REJECTED_INPUT"
    assert np.isnan(out.loc[1, "Prob_High"])
    assert "Astronaut" in out.loc[1, "Errors"]
    assert (out.loc[[0, 2], "Decision"] != "REJECTED_INPUT").all()
    # Valid rows are unaffected by the invalid neighbour.
    np.testing.assert_allclose(
        out.loc[[0, 2], ["Prob_Low", "Prob_Medium", "Prob_High"]].to_numpy(),
        scorer.predict_proba(df.iloc[[0, 2]]),
        atol=1e-12,
    )
    with pytest.raises(SchemaError):
        scorer.score(df, strict=True)


def _artifact_args(scorer, tmp_path):
    import joblib

    joblib.dump(scorer.model, tmp_path / "m.joblib")
    (tmp_path / "meta.json").write_text(json.dumps(scorer.meta))
    return ["--model", str(tmp_path / "m.joblib"), "--meta", str(tmp_path / "meta.json")]


def test_cli_batch_matches_in_process(scorer, small, tmp_path):
    _, _, X_ho, _ = small
    src = tmp_path / "batch.csv"
    X_ho.head(50).to_csv(src, index=False)
    rc = inference.main(["--csv", str(src), *_artifact_args(scorer, tmp_path)])
    assert rc == 0
    out = pd.read_csv(tmp_path / "batch_scored.csv")
    expected = scorer.score(X_ho.head(50))
    np.testing.assert_allclose(
        out[["Prob_Low", "Prob_Medium", "Prob_High"]].to_numpy(),
        expected[["Prob_Low", "Prob_Medium", "Prob_High"]].to_numpy(),
        rtol=0,
        atol=1e-12,
    )
    assert out["Decision"].tolist() == expected["Decision"].tolist()


def test_cli_single_and_errors(scorer, good_row, tmp_path, capsys):
    args = _artifact_args(scorer, tmp_path)
    assert inference.main(["--json", json.dumps(good_row), *args]) == 0
    result = json.loads(capsys.readouterr().out)
    assert set(result["probabilities"]) == {"Low", "Medium", "High"}

    bad = {**good_row, "Credit_Score": 990}
    assert inference.main(["--json", json.dumps(bad), *args]) == 2
    assert "Credit_Score=990 outside valid range" in capsys.readouterr().err

    partial = {k: v for k, v in good_row.items() if k != "Age"}
    assert inference.main(["--json", json.dumps(partial), *args]) == 2


def test_cli_missing_column_fails_cleanly(scorer, small, tmp_path, capsys):
    _, _, X_ho, _ = small
    src = tmp_path / "broken.csv"
    X_ho.head(5).drop(columns=["Credit_Score"]).to_csv(src, index=False)
    assert inference.main(["--csv", str(src), *_artifact_args(scorer, tmp_path)]) == 2
    assert "missing required column" in capsys.readouterr().err
    assert not (tmp_path / "broken_scored.csv").exists()
