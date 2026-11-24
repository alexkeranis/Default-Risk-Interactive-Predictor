"""Export fitted models and evaluation results as plain JSON for the static demo in docs/.

The browser re-implements only *arithmetic* (feature formulas, a linear model or tree
traversal, the calibration map). ``reference_predict`` is a line-for-line Python mirror of
``docs/assets/model.js``. pytest checks it against the real fitted pipelines, and the Node
test checks the JS against ``golden_cases.json`` generated from those same pipelines.

    python -m bnpl_risk.export_web
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from bnpl_risk.config import (
    CATEGORICAL_COLS,
    CATEGORY_LEVELS,
    CLASSES,
    COST,
    DOCS_DATA_DIR,
    INCOME_FLOOR_USD,
    INPUT_COLS,
    META_PATH,
    MODELS_DIR,
    NULLABLE_COLS,
    NUMERIC_COLS,
    REPORTS_DIR,
)
from bnpl_risk.data import load_raw, make_split
from bnpl_risk.explain import base_pipeline, xgb_contributions
from bnpl_risk.features import ENGINEERED_FEATURES, engineer
from bnpl_risk.preprocessing import INDICATOR_PREFIX, transform_features

LOGIT_EPS = 1e-12  # sklearn's _convert_to_logits eps for probability inputs

FEATURE_LABELS = {
    "Age": "Age",
    "Income_USD": "Income",
    "Credit_Score": "Credit score",
    "Total_BNPL_Active_Loans": "Active BNPL plans",
    "Total_BNPL_Debt_USD": "BNPL debt",
    "Average_Transaction_Value_USD": "Avg purchase size",
    "missing_Income_USD": "Income missing",
    "missing_Credit_Score": "Credit score missing",
    "bnpl_dti": "BNPL debt-to-income",
    "ticket_to_monthly_income": "Purchase / monthly income",
    "debt_per_loan": "Debt per plan",
    "loan_stacking": "Loan stacking (3+ plans)",
    "subprime": "Subprime score (<580)",
    "near_prime": "Near-prime score (580-669)",
    "unstable_income": "Student / unemployed",
    "late_x_dti": "Late payments × DTI",
    "unstable_x_dti": "Unstable income × DTI",
    "late_x_subprime": "Late payments × subprime",
    "log_income": "log income",
    "log_debt": "log debt",
    "young_borrower": "Under 25",
}
for _col, _levels in CATEGORY_LEVELS.items():
    _short = {
        "Employment_Status": "Employment",
        "Late_Payment_History": "Late payments",
        "Shopping_Category_Most_Frequent": "Category",
    }[_col]
    for _lvl in _levels:
        FEATURE_LABELS[f"{_col}_{_lvl}"] = f"{_short}: {_lvl}"


# ------------------------------------------------------------------------ model export
def export_calibration(model) -> dict:
    if not hasattr(model, "calibrated_classifiers_"):
        return {"method": "none"}
    cc = model.calibrated_classifiers_[0]
    base = cc.estimator
    response = "decision_function" if hasattr(base, "decision_function") else "predict_proba"
    method = model.method
    spec: dict = {"method": method, "input": response}
    if method == "isotonic":
        spec["per_class"] = [
            {"x": c.X_thresholds_.tolist(), "y": c.y_thresholds_.tolist()} for c in cc.calibrators
        ]
    elif method == "sigmoid":
        spec["per_class"] = [{"a": float(c.a_), "b": float(c.b_)} for c in cc.calibrators]
    elif method == "temperature":
        spec["beta"] = float(cc.calibrators[0].beta_)
    else:
        raise ValueError(method)
    return spec


def export_linear(model) -> dict:
    pipe = base_pipeline(model)
    scaler = pipe.named_steps["encode"].named_transformers_["num"]
    lr = pipe[-1]
    names = list(pipe.named_steps["encode"].get_feature_names_out())
    mean = dict(zip(scaler.feature_names_in_, scaler.mean_, strict=True))
    scale = dict(zip(scaler.feature_names_in_, scaler.scale_, strict=True))
    return {
        "type": "linear",
        "features": names,
        "mean": [float(mean.get(n, 0.0)) for n in names],
        "scale": [float(scale.get(n, 1.0)) for n in names],
        "coef": lr.coef_.tolist(),
        "intercept": lr.intercept_.tolist(),
        "calibration": export_calibration(model),
        "imputer_medians": pipe.named_steps["impute"].medians_,
    }


def _flatten_tree(tree: dict, feat_index: dict[str, int]) -> dict:
    nodes: dict[int, dict] = {}

    def walk(n):
        nodes[n["nodeid"]] = n
        for c in n.get("children", []):
            walk(c)

    walk(tree)
    size = max(nodes) + 1
    feat = [-1] * size
    thr = [0.0] * size
    yes = [-1] * size
    no = [-1] * size
    missing = [-1] * size
    value = [0.0] * size

    def mean_value(nid: int) -> tuple[float, float]:
        n = nodes[nid]
        if "leaf" in n:
            value[nid] = float(n["leaf"])
            return value[nid], float(n["cover"])
        feat[nid] = feat_index[n["split"]]
        thr[nid] = float(n["split_condition"])
        yes[nid], no[nid], missing[nid] = n["yes"], n["no"], n["missing"]
        vy, cy = mean_value(n["yes"])
        vn, cn = mean_value(n["no"])
        value[nid] = (vy * cy + vn * cn) / (cy + cn)  # cover-weighted mean (XGBoost Saabas)
        return value[nid], cy + cn

    mean_value(tree["nodeid"])
    return {"feat": feat, "thr": thr, "yes": yes, "no": no, "missing": missing, "value": value}


def export_xgb(model, X_probe: pd.DataFrame) -> dict:
    pipe = base_pipeline(model)
    booster: xgb.Booster = pipe[-1].booster_.get_booster()
    names = list(booster.feature_names)
    feat_index = {n: i for i, n in enumerate(names)}
    trees = [
        _flatten_tree(json.loads(d), feat_index)
        for d in booster.get_dump(dump_format="json", with_stats=True)
    ]
    n_classes = len(CLASSES)
    spec = {
        "type": "trees",
        "features": names,
        "n_classes": n_classes,
        "trees": trees,
        "base_margin": [0.0] * n_classes,
        "calibration": export_calibration(model),
        "imputer_medians": pipe.named_steps["impute"].medians_,
    }
    # Recover the per-class intercept empirically (robust to how the XGBoost version
    # stores base_score): output margin minus the sum of leaf values.
    Z = transform_features(pipe, X_probe)
    margin = booster.predict(xgb.DMatrix(Z), output_margin=True)
    leaves = _tree_margins(spec, Z.to_numpy())
    spec["base_margin"] = (margin - leaves).mean(axis=0).tolist()
    return spec


# ----------------------------------------------------------------- reference evaluator
def model_feature_frame(X: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    """Raw inputs -> named model features before scaling. Mirrors model.js:buildFeatures."""
    X = X.copy()
    feats = {}
    for c in NULLABLE_COLS:
        feats[f"{INDICATOR_PREFIX}{c}"] = X[c].isna().astype(float)
        X[c] = X[c].astype(float).fillna(medians[c])
    for c in NUMERIC_COLS:
        feats[c] = X[c].astype(float)
    eng = engineer(X)
    for c in ENGINEERED_FEATURES:
        feats[c] = eng[c]
    for col in CATEGORICAL_COLS:
        for lvl in CATEGORY_LEVELS[col]:
            feats[f"{col}_{lvl}"] = (X[col] == lvl).astype(float)
    return pd.DataFrame(feats, index=X.index)


def _tree_margins(spec: dict, Z: np.ndarray, contribs: np.ndarray | None = None) -> np.ndarray:
    n_classes = spec["n_classes"]
    out = np.zeros((len(Z), n_classes))
    Z32 = Z.astype(np.float32)
    for t, tree in enumerate(spec["trees"]):
        k = t % n_classes
        feat, thr, value = tree["feat"], np.float32(tree["thr"]), tree["value"]
        for i in range(len(Z)):
            node = 0
            if contribs is not None:
                contribs[i, k, -1] += value[0]
            while feat[node] >= 0:
                x = Z32[i, feat[node]]
                if np.isnan(x):
                    nxt = tree["missing"][node]
                else:
                    nxt = tree["yes"][node] if x < thr[node] else tree["no"][node]
                if contribs is not None:
                    contribs[i, k, feat[node]] += value[nxt] - value[node]
                node = nxt
            out[i, k] += value[node]
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _apply_calibration(cal: dict, raw: np.ndarray, base_proba: np.ndarray) -> np.ndarray:
    method = cal["method"]
    if method == "none":
        return base_proba
    pred = raw if cal["input"] == "decision_function" else base_proba
    if method == "temperature":
        logits = pred if cal["input"] == "decision_function" else np.log(pred + LOGIT_EPS)
        return _softmax(cal["beta"] * logits)
    p = np.zeros_like(pred)
    for k, c in enumerate(cal["per_class"]):
        if method == "isotonic":
            x, y = np.asarray(c["x"]), np.asarray(c["y"])
            p[:, k] = np.interp(np.clip(pred[:, k], x[0], x[-1]), x, y)
        else:
            p[:, k] = 1.0 / (1.0 + np.exp(c["a"] * pred[:, k] + c["b"]))
    s = p.sum(axis=1, keepdims=True)
    return np.where(s > 0, p / np.where(s > 0, s, 1), 1.0 / p.shape[1])


def reference_predict(spec: dict, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(calibrated probabilities, per-class contributions incl. bias as last column)."""
    F = model_feature_frame(X, spec["imputer_medians"])[spec["features"]].to_numpy()
    n_classes = len(CLASSES)
    contribs = np.zeros((len(F), n_classes, len(spec["features"]) + 1))
    if spec["type"] == "linear":
        Zs = (F - np.asarray(spec["mean"])) / np.asarray(spec["scale"])
        W, b = np.asarray(spec["coef"]), np.asarray(spec["intercept"])
        raw = Zs @ W.T + b
        contribs[:, :, :-1] = Zs[:, None, :] * W[None, :, :]
        contribs[:, :, -1] = b
        base = _softmax(raw)
    else:
        raw = _tree_margins(spec, F, contribs) + np.asarray(spec["base_margin"])
        contribs[:, :, -1] += np.asarray(spec["base_margin"])
        base = _softmax(raw)
    return _apply_calibration(spec["calibration"], raw, base), contribs


def model_contributions(model, X: pd.DataFrame) -> np.ndarray:
    """Ground truth for the contributions the browser shows (exact LR / XGBoost Saabas)."""
    pipe = base_pipeline(model)
    if hasattr(pipe[-1], "booster_"):
        return xgb_contributions(pipe, X, approx=True)
    from bnpl_risk.explain import linear_contributions

    return linear_contributions(pipe, X)


# ---------------------------------------------------------------------------- bundling
def golden_records(X_test: pd.DataFrame, y_test: np.ndarray, n: int = 45) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    picks = np.concatenate(
        [rng.choice(np.flatnonzero(y_test == k), size=n // 3, replace=False) for k in range(3)]
    )
    rows = X_test.iloc[np.sort(picks)].dropna()
    edges = pd.DataFrame(
        [
            {
                "Age": 18,
                "Income_USD": 0,
                "Credit_Score": 300,
                "Total_BNPL_Active_Loans": 10,
                "Total_BNPL_Debt_USD": 2731,
                "Average_Transaction_Value_USD": 1420,
                "Employment_Status": "Unemployed",
                "Late_Payment_History": "Yes",
                "Shopping_Category_Most_Frequent": "Travel",
            },
            {
                "Age": 64,
                "Income_USD": 139452,
                "Credit_Score": 850,
                "Total_BNPL_Active_Loans": 0,
                "Total_BNPL_Debt_USD": 0,
                "Average_Transaction_Value_USD": 10,
                "Employment_Status": "Employed",
                "Late_Payment_History": "No",
                "Shopping_Category_Most_Frequent": "Groceries/Essentials",
            },
            {
                "Age": 24,
                "Income_USD": 5000,
                "Credit_Score": 579,
                "Total_BNPL_Active_Loans": 3,
                "Total_BNPL_Debt_USD": 450,
                "Average_Transaction_Value_USD": 300,
                "Employment_Status": "Student",
                "Late_Payment_History": "No",
                "Shopping_Category_Most_Frequent": "Electronics",
            },
        ]
    )
    return pd.concat([rows, edges], ignore_index=True)[INPUT_COLS]


def _records(df: pd.DataFrame) -> list[dict]:
    return [
        {k: (v.item() if isinstance(v, np.generic) else v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def build_golden(specs: dict, models: dict, records: pd.DataFrame) -> dict:
    cases = {"inputs": _records(records), "models": {}}
    for fam, model in models.items():
        cases["models"][fam] = {
            "proba": model.predict_proba(records).tolist(),
            "contributions": model_contributions(model, records).tolist(),
            "features": model_feature_frame(records, specs[fam]["imputer_medians"])[
                specs[fam]["features"]
            ]
            .to_numpy()
            .tolist(),
        }
    return cases


def build_dashboard(results: dict, preds: dict, meta: dict) -> dict:
    fams = ["logreg", "xgboost", "resnet"]
    return {
        "selected": results["selection"]["selected"],
        "selection": results["selection"],
        "experiment": results["experiment"],
        "cv": {f: results["cv"][f] for f in fams},
        "test": {f: results["test"][f] for f in fams},
        "ablations": results["ablations"],
        "comparisons": results["comparisons"],
        "calibration": results["calibration"],
        "reliability_test": results["reliability_test"],
        "decisions": results["decisions"],
        "importance": results["importance"],
        "permutation_importance": results["permutation_importance"],
        "feature_labels": FEATURE_LABELS,
        "test_predictions": {
            "y": preds["y_test"].tolist(),
            "exposure": preds["exposure_test"].tolist(),
            **{f: np.round(preds[f"test_{f}"], 6).tolist() for f in fams},
        },
        "cost_assumptions": COST.to_dict(),
        "pd_cutoffs": {f: results["decisions"][f]["cutoff"] for f in fams},
    }


def export_all(
    models_dir: Path = MODELS_DIR, reports_dir: Path = REPORTS_DIR, out_dir: Path = DOCS_DATA_DIR
) -> None:
    meta = json.loads((models_dir / META_PATH.name).read_text())
    results = json.loads((reports_dir / "results.json").read_text())
    preds = dict(np.load(models_dir / "predictions.npz"))
    models = {f: joblib.load(models_dir / f"{f}.joblib") for f in ("logreg", "xgboost")}

    split = make_split(load_raw())
    probe = split.X_dev.dropna().head(200)
    specs = {
        "logreg": export_linear(models["logreg"]),
        "xgboost": export_xgb(models["xgboost"], probe),
    }

    model_json = {
        "classes": list(CLASSES),
        "selected": meta["model"]["family"],
        "income_floor": INCOME_FLOOR_USD,
        "input_schema": meta["input_schema"],
        "cost_assumptions": COST.to_dict(),
        "pd_cutoff": {f: results["decisions"][f]["cutoff"] for f in specs},
        "feature_labels": FEATURE_LABELS,
        "models": specs,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.json").write_text(json.dumps(model_json, separators=(",", ":")))
    (out_dir / "dashboard.json").write_text(
        json.dumps(build_dashboard(results, preds, meta), separators=(",", ":"), default=str)
    )
    golden = build_golden(specs, models, golden_records(split.X_test, split.y_test))
    (out_dir / "golden_cases.json").write_text(json.dumps(golden, separators=(",", ":")))

    # Self-check: the reference evaluator must reproduce the fitted pipelines.
    recs = pd.DataFrame(golden["inputs"])
    for fam, spec in specs.items():
        p, c = reference_predict(spec, recs)
        exp_p = np.asarray(golden["models"][fam]["proba"])
        exp_c = np.asarray(golden["models"][fam]["contributions"])
        print(
            f"{fam}: max |dp|={np.abs(p - exp_p).max():.2e}  max |dphi|={np.abs(c - exp_c).max():.2e}"
        )
    for name in ("model.json", "dashboard.json", "golden_cases.json"):
        print(f"wrote {out_dir / name} ({(out_dir / name).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    export_all()
