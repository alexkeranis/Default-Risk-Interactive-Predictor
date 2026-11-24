"""End-to-end experiment: split -> tune -> repeated CV -> ablations -> selection ->
calibration -> single test evaluation -> decision analysis -> artifacts.

    python -m bnpl_risk.train           # full run (~15-25 min on 16 CPU cores)
    python -m bnpl_risk.train --fast    # smoke run into models/_fast, reports/_fast
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

from bnpl_risk import __version__
from bnpl_risk.config import (
    BOOTSTRAP_ROUNDS,
    CALIBRATION_FOLDS,
    CATEGORICAL_COLS,
    CATEGORY_LEVELS,
    CLASSES,
    COST,
    CV_FOLDS,
    CV_REPEATS,
    DATA_PATH,
    HARD_LIMITS,
    INCOME_FLOOR_USD,
    INPUT_COLS,
    INTEGER_COLS,
    MODEL_FAMILIES,
    MODELS_DIR,
    N_ITER,
    NULLABLE_COLS,
    NUMERIC_COLS,
    REPORTS_DIR,
    SEED,
    SIMPLICITY_ORDER,
    TEST_SIZE,
)
from bnpl_risk.data import file_sha256, load_raw, make_split, save_split
from bnpl_risk.evaluation import (
    HEADLINE_METRICS,
    approval_curve,
    bootstrap_ci,
    choose_cutoff,
    compute_metrics,
    corrected_resampled_ttest,
    corrected_se,
    decision_summary,
    one_hot,
    prior_shift_correction,
    reliability_curve,
    summarize_folds,
)
from bnpl_risk.explain import base_pipeline, global_importance, raw_permutation_importance
from bnpl_risk.features import ENGINEERED_FEATURES, FEATURE_RATIONALE
from bnpl_risk.models import make_model
from bnpl_risk.models.gbdt import XGBRiskClassifier
from bnpl_risk.preprocessing import NumericImputer, transform_features
from bnpl_risk.schema import training_support
from bnpl_risk.seed import seed_everything
from bnpl_risk.tuning import tune

CALIBRATION_METHODS = ("none", "sigmoid", "isotonic", "temperature")


@dataclass
class RunConfig:
    fast: bool = False
    models_dir: Path = MODELS_DIR
    reports_dir: Path = REPORTS_DIR
    n_iter: int = N_ITER
    cv_repeats: int = CV_REPEATS
    bootstrap_rounds: int = BOOTSTRAP_ROUNDS
    n_jobs: int = -1

    @classmethod
    def make(cls, fast: bool) -> RunConfig:
        if not fast:
            return cls()
        return cls(
            fast=True,
            models_dir=MODELS_DIR / "_fast",
            reports_dir=REPORTS_DIR / "_fast",
            n_iter=3,
            cv_repeats=1,
            bootstrap_rounds=50,
        )


# ------------------------------------------------------------------------- building blocks
def fit_calibrated(family: str, params: dict, X, y, method: str):
    """The deployable model: tuned pipeline + post-hoc calibration. With
    ``ensemble=False`` the base pipeline is refit on all of ``X`` and the calibrator is
    fit on its out-of-fold predictions, so calibration never sees in-sample scores."""
    base = make_model(family, **params)
    if method == "none":
        return base.fit(X, y)
    cv = StratifiedKFold(CALIBRATION_FOLDS, shuffle=True, random_state=SEED)
    return CalibratedClassifierCV(base, method=method, cv=cv, ensemble=False).fit(X, y)


def _fit_predict(model, X, y, tr, va):
    m = clone(model).fit(X.iloc[tr], y[tr])
    return va, m.predict_proba(X.iloc[va])


def run_cv(model, X, y, splits, n_jobs=-1) -> tuple[list[dict], np.ndarray]:
    res = Parallel(n_jobs=n_jobs)(delayed(_fit_predict)(model, X, y, tr, va) for tr, va in splits)
    folds = [compute_metrics(y[va], p) for va, p in res]
    per_repeat = round(len(y) / len(splits[0][1]))  # k in a (repeated) k-fold scheme
    oof = np.zeros((max(len(splits) // per_repeat, 1), len(y), len(CLASSES)))
    for i, (va, p) in enumerate(res):
        oof[i // per_repeat, va] = p
    return folds, oof


def _calibration_job(family, params, method, X, y, tr, va):
    if method == "prior_shift":
        # Analytic Bayes prior correction (Saerens et al., 2002) for a model trained with
        # class weights: the weighted loss implies priors pi' ~ pi * w; map back to pi.
        m = fit_calibrated(family, params, X.iloc[tr], y[tr], "none")
        prior = np.bincount(y[tr], minlength=len(CLASSES)) / len(tr)
        weights = compute_class_weight(params["class_weight"], classes=np.unique(y[tr]), y=y[tr])
        return (
            method,
            va,
            prior_shift_correction(m.predict_proba(X.iloc[va]), prior, prior * weights),
        )
    m = fit_calibrated(family, params, X.iloc[tr], y[tr], method)
    return method, va, m.predict_proba(X.iloc[va])


def fixed_config_xgb(with_categoricals: bool):
    """Categorical-input ablation: an untuned, fixed XGBoost configuration evaluated on the
    numeric inputs (+ debt-to-income) alone, or with the one-hot categorical inputs added."""
    cols = [*NUMERIC_COLS, "bnpl_dti"]
    transformers = [("num", "passthrough", cols)]
    if with_categoricals:
        levels = [CATEGORY_LEVELS[c] for c in CATEGORICAL_COLS]
        transformers.append(
            ("cat", OneHotEncoder(categories=levels, sparse_output=False), CATEGORICAL_COLS)
        )
    from sklearn.pipeline import Pipeline

    from bnpl_risk.features import FinancialFeatureEngineer

    return Pipeline(
        [
            ("impute", NumericImputer()),
            ("features", FinancialFeatureEngineer()),
            ("select", ColumnTransformer(transformers, verbose_feature_names_out=False)),
            (
                "model",
                XGBRiskClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    ).set_output(transform="pandas")


def select_family(cv_folds: dict[str, list[dict]], n_train: int, n_test: int) -> dict:
    """Pre-registered rule: lowest mean CV log loss, but choose the simplest family whose
    mean is within one (corrected) standard error of the best."""
    ll = {f: np.array([m["log_loss"] for m in folds]) for f, folds in cv_folds.items()}
    best = min(ll, key=lambda f: ll[f].mean())
    se = corrected_se(ll[best], n_train, n_test)
    eligible = [f for f in ll if ll[f].mean() <= ll[best].mean() + se]
    selected = min(eligible, key=lambda f: SIMPLICITY_ORDER[f])
    return {
        "rule": "lowest mean CV log loss; simplest family within 1 corrected SE of the best",
        "best_by_log_loss": best,
        "threshold": float(ll[best].mean() + se),
        "corrected_se": se,
        "eligible": eligible,
        "selected": selected,
    }


def git_revision() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], check=False).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def library_versions() -> dict:
    import sklearn
    import torch
    import xgboost

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "torch": torch.__version__,
    }


def input_schema(support: dict) -> dict:
    return {
        "numeric": {
            c: {
                "nullable": c in NULLABLE_COLS,
                "integer": c in INTEGER_COLS,
                "hard_min": HARD_LIMITS[c][0],
                "hard_max": HARD_LIMITS[c][1],
                "train_min": support[c]["min"],
                "train_max": support[c]["max"],
            }
            for c in NUMERIC_COLS
        },
        "categorical": CATEGORY_LEVELS,
        "input_order": INPUT_COLS,
    }


def build_meta(
    *,
    family: str,
    params: dict,
    calibration: str,
    support: dict,
    model_features: list[str],
    cutoff: float,
    extra: dict | None = None,
) -> dict:
    """Everything needed to serve and audit the model, without unpickling it."""
    meta = {
        "package_version": __version__,
        "classes": list(CLASSES),
        "model": {"family": family, "params": params, "calibration": calibration},
        "input_schema": input_schema(support),
        "support": support,
        "model_features": model_features,
        "engineered_features": FEATURE_RATIONALE,
        "income_floor_usd": INCOME_FLOOR_USD,
        "decision": {
            "rule": "approve if expected PD = sum_k p_k * PD_k <= cutoff; tier = argmin expected cost",
            "pd_cutoff": cutoff,
            "cost_assumptions": COST.to_dict(),
        },
    }
    meta.update(extra or {})
    return meta


def _round(obj, nd=6):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_round(v, nd) for v in obj]
    return obj


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_round(obj), indent=2, default=str))


# ------------------------------------------------------------------------------- pipeline
def run(cfg: RunConfig) -> dict:
    seed_everything()
    t_start = time.time()
    log = lambda msg: print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)  # noqa: E731

    df = load_raw()
    sha = file_sha256(DATA_PATH)
    split = make_split(df)
    if not cfg.fast:
        save_split(split, sha)
    X, y = split.X_dev, split.y_dev
    X_test, y_test = split.X_test, split.y_test
    support = training_support(X)
    log(f"dev={len(X)} test={len(X_test)} sha256={sha[:12]}")

    fixed = {"resnet": {"max_epochs": 12, "patience": 4}} if cfg.fast else {}

    # 1. Tuning (inner CV on dev only, equal budget, log-loss scoring)
    params, tuning_logs = {}, {}
    for fam in MODEL_FAMILIES:
        params[fam], tuning_logs[fam] = tune(
            fam, X, y, n_iter=cfg.n_iter, n_jobs=cfg.n_jobs, fixed_params=fixed.get(fam)
        )
        log(f"tuned {fam}: {params[fam]}")

    # 2. Repeated stratified CV with identical folds for every family
    rskf = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS, n_repeats=cfg.cv_repeats, random_state=SEED + 1
    )
    splits = list(rskf.split(X, y))
    n_tr, n_va = len(splits[0][0]), len(splits[0][1])
    cv_folds, oof = {}, {}
    for fam in MODEL_FAMILIES:
        cv_folds[fam], oof[fam] = run_cv(make_model(fam, **params[fam]), X, y, splits, cfg.n_jobs)
        log(
            f"CV {fam}: log_loss={np.mean([m['log_loss'] for m in cv_folds[fam]]):.4f} "
            f"macro_f1={np.mean([m['macro_f1'] for m in cv_folds[fam]]):.4f}"
        )

    # 3. Ablations on the same folds (feature set, imputer, categorical inputs)
    ablations = {}
    for fam in ("logreg", "xgboost"):
        for name, kw in {
            "raw_features": {"features": "raw"},
            "knn_imputer": {"imputer": "knn"},
        }.items():
            folds, _ = run_cv(make_model(fam, **kw, **params[fam]), X, y, splits, cfg.n_jobs)
            ablations[f"{fam}/{name}"] = folds
    for with_cat in (False, True):
        key = "categorical_ablation/" + ("all_inputs" if with_cat else "numeric_only")
        ablations[key], _ = run_cv(fixed_config_xgb(with_cat), X, y, splits, cfg.n_jobs)
    log("ablations done")

    # 4. Paired comparisons (Nadeau-Bengio corrected t-test)
    comparisons = []
    fams = list(MODEL_FAMILIES)
    for i, a in enumerate(fams):
        for b in fams[i + 1 :]:
            for metric in ("log_loss", "macro_f1", "brier", "pr_auc_High"):
                d = np.array(
                    [
                        fa[metric] - fb[metric]
                        for fa, fb in zip(cv_folds[a], cv_folds[b], strict=True)
                    ]
                )
                t, p = corrected_resampled_ttest(d, n_tr, n_va)
                comparisons.append(
                    {"a": a, "b": b, "metric": metric, "mean_diff": float(d.mean()), "t": t, "p": p}
                )
    for fam in ("logreg", "xgboost"):
        for name in ("raw_features", "knn_imputer"):
            d = np.array(
                [
                    fa["log_loss"] - fb["log_loss"]
                    for fa, fb in zip(ablations[f"{fam}/{name}"], cv_folds[fam], strict=True)
                ]
            )
            t, p = corrected_resampled_ttest(d, n_tr, n_va)
            comparisons.append(
                {
                    "a": f"{fam}/{name}",
                    "b": fam,
                    "metric": "log_loss",
                    "mean_diff": float(d.mean()),
                    "t": t,
                    "p": p,
                }
            )

    # 5. Model selection (pre-registered 1-SE rule)
    selection = select_family(cv_folds, n_tr, n_va)
    log(f"selection: {selection['selected']} (best by log loss: {selection['best_by_log_loss']})")

    # 6. Calibration method choice on out-of-fold predictions (first CV repeat)
    first = splits[:CV_FOLDS]
    jobs = [
        delayed(_calibration_job)(fam, params[fam], m, X, y, tr, va)
        for fam in MODEL_FAMILIES
        for m in CALIBRATION_METHODS[1:]
        for tr, va in first
    ]
    cal_oof = {fam: {"none": oof[fam][0]} for fam in MODEL_FAMILIES}
    results = Parallel(n_jobs=cfg.n_jobs)(jobs)
    for job, (method, va, p) in zip(jobs, results, strict=True):
        fam = job[1][0]
        cal_oof[fam].setdefault(method, np.zeros((len(y), len(CLASSES))))[va] = p
    calibration = {}
    for fam in MODEL_FAMILIES:
        scores = {m: compute_metrics(y, cal_oof[fam][m]) for m in CALIBRATION_METHODS}
        chosen = min(scores, key=lambda m: scores[m]["brier"])
        calibration[fam] = {
            "chosen": chosen,
            "criterion": "lowest out-of-fold multiclass Brier score",
            "oof": {
                m: {
                    k: s[k]
                    for k in ("brier", "log_loss", "ece_Low", "ece_Medium", "ece_High", "macro_f1")
                }
                for m, s in scores.items()
            },
        }
        log(f"calibration {fam}: {chosen}")

    # 6b. Class-weighted training (inverse-frequency weights) distorts the implied priors.
    # Same LR, balanced weights, every post-hoc repair.
    cw_params = {**params["logreg"], "class_weight": "balanced"}
    cw_jobs = [
        delayed(_calibration_job)("logreg", cw_params, m, X, y, tr, va)
        for m in (*CALIBRATION_METHODS, "prior_shift")
        for tr, va in first
    ]
    cw_oof: dict[str, np.ndarray] = {}
    for method, va, p in Parallel(n_jobs=cfg.n_jobs)(cw_jobs):
        cw_oof.setdefault(method, np.zeros((len(y), len(CLASSES))))[va] = p
    cw_scores = {m: compute_metrics(y, p) for m, p in cw_oof.items()}
    cw_best = min((m for m in cw_scores if m != "none"), key=lambda m: cw_scores[m]["brier"])
    calibration_weighted = {
        "params": cw_params,
        "best": cw_best,
        "oof": {
            m: {
                k: s[k]
                for k in (
                    "brier",
                    "log_loss",
                    "ece_Low",
                    "ece_Medium",
                    "ece_High",
                    "macro_f1",
                    "recall_High",
                    "precision_High",
                )
            }
            for m, s in cw_scores.items()
        },
        "mean_pred_vs_base_rate": {
            m: {
                "mean_pred": cw_oof[m].mean(axis=0).tolist(),
                "base_rate": (np.bincount(y, minlength=3) / len(y)).tolist(),
            }
            for m in cw_oof
        },
        "reliability_oof": {
            m: {
                cls: reliability_curve(one_hot(y)[:, k], cw_oof[m][:, k]).to_dict(orient="list")
                for k, cls in enumerate(CLASSES)
            }
            for m in ("none", cw_best)
        },
    }
    log(f"class-weighted calibration demo: best={cw_best}")

    # 7. Final fit on all dev data; the test set is scored exactly once here
    finals = Parallel(n_jobs=min(cfg.n_jobs if cfg.n_jobs > 0 else 6, 6))(
        delayed(fit_calibrated)(fam, params[fam], X, y, m)
        for fam in MODEL_FAMILIES
        for m in ("none", calibration[fam]["chosen"])
    )
    final_models, raw_models = {}, {}
    for i, fam in enumerate(MODEL_FAMILIES):
        raw_models[fam], final_models[fam] = finals[2 * i], finals[2 * i + 1]
    test_proba = {fam: final_models[fam].predict_proba(X_test) for fam in MODEL_FAMILIES}
    test_proba_raw = {fam: raw_models[fam].predict_proba(X_test) for fam in MODEL_FAMILIES}
    test_metrics = {}
    for fam in MODEL_FAMILIES:
        test_metrics[fam] = {
            "calibrated": compute_metrics(y_test, test_proba[fam]),
            "uncalibrated": compute_metrics(y_test, test_proba_raw[fam]),
            "ci95": bootstrap_ci(y_test, test_proba[fam], n_rounds=cfg.bootstrap_rounds, seed=SEED),
        }
    log("test evaluation done")

    # 8. Decision analysis: cutoff chosen on calibrated OOF dev predictions, applied to test
    exposure_dev = X["Average_Transaction_Value_USD"].to_numpy(dtype=float)
    exposure_test = X_test["Average_Transaction_Value_USD"].to_numpy(dtype=float)
    decisions = {}
    for fam in MODEL_FAMILIES:
        p_oof = cal_oof[fam][calibration[fam]["chosen"]]
        cutoff = choose_cutoff(approval_curve(p_oof, y, exposure_dev))
        decisions[fam] = {
            "cutoff": cutoff,
            "dev_oof": decision_summary(p_oof, y, exposure_dev, cutoff),
            "test": decision_summary(test_proba[fam], y_test, exposure_test, cutoff),
        }
    # Reference policies for context: approve everyone / perfect tier knowledge
    perfect = one_hot(y_test)
    decisions["reference"] = {
        "approve_all": decision_summary(perfect, y_test, exposure_test, cutoff=1.0),
        "oracle": decision_summary(
            perfect, y_test, exposure_test, cutoff=decisions[selection["selected"]]["cutoff"]
        ),
    }
    log("decision analysis done")

    # 9. Explanations
    importance = {}
    for fam in ("logreg", "xgboost"):
        importance[fam] = global_importance(base_pipeline(final_models[fam]), X_test)
    permutation = {
        fam: raw_permutation_importance(final_models[fam], X_test, y_test) for fam in MODEL_FAMILIES
    }
    log("explanations done")

    # 10. Artifacts
    sel = selection["selected"]
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    for fam in MODEL_FAMILIES:
        joblib.dump(final_models[fam], cfg.models_dir / f"{fam}.joblib")
    joblib.dump(final_models[sel], cfg.models_dir / "risk_model.joblib")
    model_features = list(transform_features(base_pipeline(final_models[sel]), X.head(1)).columns)

    cv_summary = {f: summarize_folds(cv_folds[f]) for f in MODEL_FAMILIES}
    abl_summary = {k: summarize_folds(v) for k, v in ablations.items()}
    reliability = {
        fam: {
            stage: {
                cls: reliability_curve(one_hot(y_test)[:, k], p[:, k]).to_dict(orient="list")
                for k, cls in enumerate(CLASSES)
            }
            for stage, p in (("uncalibrated", test_proba_raw[fam]), ("calibrated", test_proba[fam]))
        }
        for fam in MODEL_FAMILIES
    }
    experiment = {
        "data": {
            "path": DATA_PATH.name,
            "sha256": sha,
            "n_rows": len(df),
            "n_dev": len(X),
            "n_test": len(X_test),
            "test_size": TEST_SIZE,
            "class_counts_dev": np.bincount(y).tolist(),
            "class_counts_test": np.bincount(y_test).tolist(),
        },
        "seed": SEED,
        "cv": {
            "scheme": f"RepeatedStratifiedKFold({CV_FOLDS}x{cfg.cv_repeats})",
            "n_train_per_fold": n_tr,
            "n_val_per_fold": n_va,
        },
        "tuning": {
            "scoring": "neg_log_loss",
            "inner_cv_folds": 5,
            "n_iter": cfg.n_iter,
            "best_params": params,
        },
        "fast_mode": cfg.fast,
    }
    results = {
        "experiment": experiment,
        "cv": cv_summary,
        "cv_folds": cv_folds,
        "ablations": abl_summary,
        "comparisons": comparisons,
        "selection": selection,
        "calibration": calibration,
        "calibration_class_weighted": calibration_weighted,
        "test": test_metrics,
        "decisions": decisions,
        "reliability_test": reliability,
        "importance": {
            f: d.reset_index(names="feature").to_dict(orient="records")
            for f, d in importance.items()
        },
        "permutation_importance": {
            f: d.reset_index(names="feature").to_dict(orient="records")
            for f, d in permutation.items()
        },
        "headline_metrics": HEADLINE_METRICS,
        "runtime_seconds": time.time() - t_start,
    }
    write_json(results, cfg.reports_dir / "results.json")
    for fam in MODEL_FAMILIES:
        tuning_logs[fam].to_csv(cfg.reports_dir / f"tuning_{fam}.csv", index=False)
    pd.DataFrame(
        [
            {"family": f, "fold": i, **m}
            for f, folds in cv_folds.items()
            for i, m in enumerate(folds)
        ]
    ).to_csv(cfg.reports_dir / "cv_results.csv", index=False)
    np.savez_compressed(
        cfg.models_dir / "predictions.npz",
        y_dev=y,
        y_test=y_test,
        exposure_test=exposure_test,
        **{f"oof_{f}": cal_oof[f][calibration[f]["chosen"]] for f in MODEL_FAMILIES},
        **{f"oof_raw_{f}": oof[f][0] for f in MODEL_FAMILIES},
        **{f"test_{f}": test_proba[f] for f in MODEL_FAMILIES},
        **{f"test_raw_{f}": test_proba_raw[f] for f in MODEL_FAMILIES},
    )

    meta = build_meta(
        family=sel,
        params=params[sel],
        calibration=calibration[sel]["chosen"],
        support=support,
        model_features=model_features,
        cutoff=decisions[sel]["cutoff"],
        extra={
            "experiment": experiment,
            "selection": selection,
            "cv_metrics": cv_summary[sel],
            "cv_metrics_all_families": {
                f: {k: cv_summary[f][k] for k in HEADLINE_METRICS} for f in MODEL_FAMILIES
            },
            "test_metrics": test_metrics[sel]["calibrated"],
            "test_ci95": test_metrics[sel]["ci95"],
            "calibration_curves_test": reliability[sel],
            "decision_test": decisions[sel]["test"],
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_revision": git_revision(),
            "library_versions": library_versions(),
            "engineered_feature_names": ENGINEERED_FEATURES,
        },
    )
    write_json(meta, cfg.models_dir / "model_meta.json")
    log(f"done: selected={sel}, artifacts in {cfg.models_dir}")
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fast", action="store_true", help="tiny budgets; writes to */_fast")
    args = ap.parse_args(argv)
    run(RunConfig.make(args.fast))


if __name__ == "__main__":
    main()
