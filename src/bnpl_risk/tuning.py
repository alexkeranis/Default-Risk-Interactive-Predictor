"""Hyperparameter search with an equal budget per family, scored by log loss (a proper
scoring rule) with inner cross-validation on the development set only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from bnpl_risk.config import N_ITER, SEED, TUNE_FOLDS
from bnpl_risk.models import make_model


def search_space(family: str) -> dict:
    if family == "logreg":
        # 12 x 2 = 24 configurations: exhaustive grid, within the same budget as the others.
        return {
            "model__C": list(np.logspace(-3, 2, 12)),
            "model__class_weight": [None, "balanced"],
        }
    if family == "xgboost":
        return {
            "model__n_estimators": randint(100, 1201),
            "model__max_depth": randint(2, 8),
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__min_child_weight": loguniform(0.5, 20),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.5, 0.5),
            "model__reg_lambda": loguniform(0.1, 20),
            "model__gamma": [0.0, 0.1, 0.5, 1.0],
            "model__class_weight": [None, "balanced"],
        }
    if family == "resnet":
        return {
            "model__d_block": [64, 128, 256],
            "model__n_blocks": [1, 2, 3],
            "model__emb_dim": [2, 4, 8],
            "model__dropout": uniform(0.0, 0.4),
            "model__lr": loguniform(3e-4, 3e-3),
            "model__weight_decay": loguniform(1e-6, 1e-2),
            "model__batch_size": [128, 256, 512],
            "model__cat_dropout": [0.0, 0.02, 0.05],
            "model__class_weight": [None, "balanced"],
        }
    raise ValueError(family)


def tune(
    family: str,
    X: pd.DataFrame,
    y: np.ndarray,
    n_iter: int = N_ITER,
    n_jobs: int = -1,
    fixed_params: dict | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Return the best estimator params (without the ``model__`` prefix) and the search log."""
    pipe = make_model(family, **(fixed_params or {}))
    cv = StratifiedKFold(TUNE_FOLDS, shuffle=True, random_state=SEED)
    space = search_space(family)
    if family == "logreg":
        search = GridSearchCV(pipe, space, scoring="neg_log_loss", cv=cv, n_jobs=n_jobs)
    else:
        search = RandomizedSearchCV(
            pipe,
            space,
            n_iter=n_iter,
            scoring="neg_log_loss",
            cv=cv,
            n_jobs=n_jobs,
            random_state=SEED,
        )
    search.fit(X, y)
    best = {k.removeprefix("model__"): _py(v) for k, v in search.best_params_.items()}
    log = pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time"]
    ].sort_values("rank_test_score")
    return {**(fixed_params or {}), **best}, log


def _py(v):
    """numpy scalars -> plain Python so params serialise to JSON."""
    return v.item() if isinstance(v, np.generic) else v
