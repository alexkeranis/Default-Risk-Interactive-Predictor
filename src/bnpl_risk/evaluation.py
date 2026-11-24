"""Metrics, uncertainty quantification and cost-sensitive decision analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from bnpl_risk.config import CLASSES, COST, CostAssumptions

N_CLASSES = len(CLASSES)


# ----------------------------------------------------------------- probability quality
def one_hot(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    return np.eye(n_classes)[np.asarray(y)]


def brier_multiclass(y: np.ndarray, proba: np.ndarray) -> float:
    """Mean over samples of sum_k (p_k - 1[y=k])^2 (range 0-2; lower is better)."""
    return float(np.mean(np.sum((proba - one_hot(y, proba.shape[1])) ** 2, axis=1)))


def reliability_curve(y_bin: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width bins over [0, 1] for plotting (empty bins dropped; ``n`` per bin is
    kept so sparse high-probability bins can be read with appropriate caution)."""
    idx = np.minimum((np.asarray(p) * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append(
                {
                    "mean_pred": float(p[m].mean()),
                    "frac_pos": float(y_bin[m].mean()),
                    "n": int(m.sum()),
                }
            )
    return pd.DataFrame(rows, columns=["mean_pred", "frac_pos", "n"])


def expected_calibration_error(y_bin: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Adaptive ECE for one class (one-vs-rest): equal-mass bins formed by splitting the
    sorted predictions, which stays well-defined when most predictions are tied near 0."""
    order = np.argsort(p, kind="stable")
    total = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            total += len(chunk) * abs(float(y_bin[chunk].mean()) - float(p[chunk].mean()))
    return total / len(p)


# ------------------------------------------------------------------------ full metrics
def compute_metrics(y: np.ndarray, proba: np.ndarray, y_pred: np.ndarray | None = None) -> dict:
    y = np.asarray(y)
    if y_pred is None:
        y_pred = np.argmax(proba, axis=1)
    labels = list(range(N_CLASSES))
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, labels=labels, zero_division=0)
    yb = one_hot(y)
    out = {
        "macro_f1": float(f1_score(y, y_pred, average="macro", labels=labels, zero_division=0)),
        "accuracy": float(np.mean(y == y_pred)),
        "log_loss": float(log_loss(y, np.clip(proba, 1e-15, 1), labels=labels)),
        "brier": brier_multiclass(y, proba),
        "roc_auc_macro": float(roc_auc_score(y, proba, multi_class="ovr", average="macro")),
        "pr_auc_macro": float(
            np.mean([average_precision_score(yb[:, k], proba[:, k]) for k in labels])
        ),
    }
    for k, name in enumerate(CLASSES):
        out[f"precision_{name}"] = float(prec[k])
        out[f"recall_{name}"] = float(rec[k])
        out[f"f1_{name}"] = float(f1[k])
        out[f"pr_auc_{name}"] = float(average_precision_score(yb[:, k], proba[:, k]))
        out[f"roc_auc_{name}"] = float(roc_auc_score(yb[:, k], proba[:, k]))
        out[f"ece_{name}"] = expected_calibration_error(yb[:, k], proba[:, k])
    return out


HEADLINE_METRICS = [
    "macro_f1",
    "log_loss",
    "brier",
    "roc_auc_macro",
    "pr_auc_macro",
    "pr_auc_High",
    "recall_High",
    "precision_High",
    "f1_High",
    "ece_High",
]


def bootstrap_ci(
    y: np.ndarray,
    proba: np.ndarray,
    y_pred: np.ndarray | None = None,
    metrics: list[str] = HEADLINE_METRICS,
    n_rounds: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Stratified percentile bootstrap CIs on a fixed evaluation set."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    if y_pred is None:
        y_pred = np.argmax(proba, axis=1)
    by_class = [np.flatnonzero(y == k) for k in range(N_CLASSES)]
    draws: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_rounds):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in by_class])
        m = compute_metrics(y[idx], proba[idx], y_pred[idx])
        for name in metrics:
            draws[name].append(m[name])
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {k: tuple(float(x) for x in np.percentile(v, q)) for k, v in draws.items()}


# --------------------------------------------------------------- model comparison stats
def corrected_resampled_ttest(diffs: np.ndarray, n_train: int, n_test: int) -> tuple[float, float]:
    """Nadeau & Bengio (2003) corrected repeated k-fold t-test. Fold scores share training
    data, so the naive t-test is badly anti-conservative; the variance is inflated by
    ``n_test / n_train``. Returns (t statistic, two-sided p-value)."""
    diffs = np.asarray(diffs, dtype=float)
    j = len(diffs)
    var = np.var(diffs, ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = diffs.mean() / np.sqrt((1 / j + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=j - 1)
    return float(t), float(p)


def corrected_se(scores: np.ndarray, n_train: int, n_test: int) -> float:
    scores = np.asarray(scores, dtype=float)
    return float(np.sqrt((1 / len(scores) + n_test / n_train) * np.var(scores, ddof=1)))


# ------------------------------------------------------------------- prior correction
def prior_shift_correction(
    proba: np.ndarray, true_prior: np.ndarray, model_prior: np.ndarray
) -> np.ndarray:
    """Bayes prior-shift correction (Saerens, Latinne & Decaestecker, 2002):
    p(k|x) ∝ p'(k|x) * pi_k / pi'_k. Undoes the prior distortion of class-weighted training,
    where the effective prior is pi'_k ∝ pi_k * w_k."""
    model_prior = np.asarray(model_prior, dtype=float) / np.sum(model_prior)
    p = proba * (np.asarray(true_prior, dtype=float) / model_prior)
    return p / p.sum(axis=1, keepdims=True)


# --------------------------------------------------------------- cost-sensitive decisions
def expected_pd(proba: np.ndarray, cost: CostAssumptions = COST) -> np.ndarray:
    """Effective probability of default implied by the tier probabilities."""
    return proba @ np.asarray(cost.pd_by_tier)


def cost_optimal_tier(proba: np.ndarray, cost: CostAssumptions = COST) -> np.ndarray:
    """Bayes decision: the tier with the lowest expected misclassification cost,
    argmin_j sum_k p_k C[k, j], instead of argmax_k p_k."""
    return np.argmin(proba @ np.asarray(cost.cost_matrix), axis=1)


def misclassification_cost(
    y: np.ndarray, y_pred: np.ndarray, cost: CostAssumptions = COST
) -> float:
    C = np.asarray(cost.cost_matrix)
    return float(C[np.asarray(y), np.asarray(y_pred)].mean())


def approval_curve(
    proba: np.ndarray,
    y: np.ndarray,
    exposure: np.ndarray,
    cost: CostAssumptions = COST,
    cutoffs: np.ndarray | None = None,
) -> pd.DataFrame:
    """Approve when expected PD <= cutoff. For each cutoff report volume, bad rates and
    money. ``loss_proxy`` applies the assumed tier PD to the *true* tier of each approved
    customer: the realised-loss proxy that labelled data allows."""
    pd_eff = expected_pd(proba, cost)
    pd_true = np.asarray(cost.pd_by_tier)[np.asarray(y)]
    if cutoffs is None:
        cutoffs = np.unique(np.concatenate([[0.0], np.quantile(pd_eff, np.linspace(0, 1, 201))]))
    n = len(y)
    rows = []
    for c in cutoffs:
        a = pd_eff <= c
        k = int(a.sum())
        revenue = float((exposure[a] * cost.merchant_fee).sum())
        exp_loss = float((exposure[a] * cost.lgd * pd_eff[a]).sum())
        loss_proxy = float((exposure[a] * cost.lgd * pd_true[a]).sum())
        rows.append(
            {
                "cutoff": float(c),
                "approval_rate": k / n,
                "high_rate_approved": float((y[a] == 2).mean()) if k else 0.0,
                "medhigh_rate_approved": float((y[a] >= 1).mean()) if k else 0.0,
                "high_captured": float((~a & (y == 2)).sum() / max((y == 2).sum(), 1)),
                "revenue_per_applicant": revenue / n,
                "expected_loss_per_applicant": exp_loss / n,
                "loss_proxy_per_applicant": loss_proxy / n,
                "profit_per_applicant": (revenue - loss_proxy) / n,
            }
        )
    return pd.DataFrame(rows)


def choose_cutoff(curve: pd.DataFrame) -> float:
    """Profit-maximising cutoff (ties broken toward the more conservative cutoff)."""
    best = curve["profit_per_applicant"].max()
    return float(curve.loc[np.isclose(curve["profit_per_applicant"], best), "cutoff"].min())


def decision_summary(
    proba: np.ndarray,
    y: np.ndarray,
    exposure: np.ndarray,
    cutoff: float,
    cost: CostAssumptions = COST,
) -> dict:
    row = approval_curve(proba, y, exposure, cost, cutoffs=np.array([cutoff])).iloc[0].to_dict()
    tier_argmax = np.argmax(proba, axis=1)
    tier_cost = cost_optimal_tier(proba, cost)
    row.update(
        {
            "misclassification_cost_argmax": misclassification_cost(y, tier_argmax, cost),
            "misclassification_cost_bayes": misclassification_cost(y, tier_cost, cost),
            "confusion_argmax": confusion_matrix(y, tier_argmax, labels=range(N_CLASSES)).tolist(),
            "confusion_bayes": confusion_matrix(y, tier_cost, labels=range(N_CLASSES)).tolist(),
            "macro_f1_bayes": float(f1_score(y, tier_cost, average="macro")),
            "recall_High_bayes": float(np.mean(tier_cost[y == 2] == 2)),
        }
    )
    return row


def summarize_folds(fold_metrics: list[dict], keys: list[str] | None = None) -> dict[str, dict]:
    df = pd.DataFrame(fold_metrics)
    keys = keys or list(df.columns)
    return {k: {"mean": float(df[k].mean()), "std": float(df[k].std(ddof=1))} for k in keys}
