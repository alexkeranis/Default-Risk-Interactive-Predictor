"""Render reports/benchmark.md and reports/figures/*.png from a training run.

python -m bnpl_risk.reporting
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bnpl_risk.config import CLASSES, COST, FIGURES_DIR, MODEL_FAMILIES, MODELS_DIR, REPORTS_DIR
from bnpl_risk.evaluation import approval_curve
from bnpl_risk.export_web import FEATURE_LABELS

NAMES = {"logreg": "Logistic regression", "xgboost": "XGBoost", "resnet": "Tabular ResNet"}
# Reference data-viz palette (light mode), categorical slots 1-3 in fixed order.
SERIES = {"logreg": "#2a78d6", "xgboost": "#eb6834", "resnet": "#1baf7a"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    "#fcfcfb",
)
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.titlelocation": "left",
        }
    )


def _save(fig, name: str, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------ figures
def fig_cv(results: dict, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
    for ax, (metric, label) in zip(
        axes,
        [
            ("log_loss", "Log loss (lower is better)"),
            ("macro_f1", "Macro F1"),
            ("pr_auc_High", "PR-AUC, High tier"),
        ],
        strict=True,
    ):
        for i, fam in enumerate(MODEL_FAMILIES):
            vals = [f[metric] for f in results["cv_folds"][fam]]
            jitter = np.linspace(-0.12, 0.12, len(vals))
            ax.scatter(vals, i + jitter, s=14, color=SERIES[fam], alpha=0.45, linewidths=0)
            m, s = np.mean(vals), np.std(vals, ddof=1)
            ax.errorbar(
                m,
                i,
                xerr=s,
                fmt="o",
                color=SERIES[fam],
                ms=8,
                mec=SURFACE,
                mew=2,
                elinewidth=2,
                capsize=0,
            )
        ax.set_yticks(range(len(MODEL_FAMILIES)), [NAMES[f] for f in MODEL_FAMILIES])
        ax.set_ylim(-0.6, len(MODEL_FAMILIES) - 0.4)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(label)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    fig.suptitle(
        f"Repeated stratified CV ({results['experiment']['cv']['scheme']}): "
        "each dot is one fold, marker = mean ± 1 sd",
        x=0.01,
        ha="left",
        fontsize=9,
        color=INK2,
        y=1.04,
    )
    _save(fig, "cv_benchmark.png", out)


def _reliability_panels(curves: list[tuple[dict, str, str]], title: str, name: str, out: Path):
    """curves: (per-class curve dict, color, label). Marker area ~ bin count."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, cls in zip(axes, CLASSES, strict=True):
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1, zorder=0)
        for rel, color, label in curves:
            c = rel[cls]
            sizes = 12 + 140 * np.sqrt(np.asarray(c["n"]) / max(c["n"]))
            ax.plot(c["mean_pred"], c["frac_pos"], "-", color=color, label=label)
            ax.scatter(
                c["mean_pred"],
                c["frac_pos"],
                s=sizes,
                color=color,
                edgecolors=SURFACE,
                linewidths=1.5,
                zorder=3,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"P({cls})")
        ax.set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed frequency")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=9, color=INK2, y=1.0)
    _save(fig, name, out)


def fig_reliability(results: dict, fam: str, out: Path):
    rel = results["reliability_test"][fam]
    method = results["calibration"][fam]["chosen"]
    _reliability_panels(
        [
            (rel["uncalibrated"], MUTED, "Uncalibrated"),
            (rel["calibrated"], SERIES[fam], f"Calibrated ({method})"),
        ],
        f"{NAMES[fam]} reliability on the held-out test set "
        "(10 equal-width bins; marker area ∝ bin count; diagonal = perfect calibration)",
        "reliability.png",
        out,
    )


def fig_calibration_weighted(results: dict, out: Path):
    cw = results["calibration_class_weighted"]
    best = cw["best"]
    _reliability_panels(
        [
            (cw["reliability_oof"]["none"], SERIES["xgboost"], "Balanced class weights, raw"),
            (
                cw["reliability_oof"][best],
                SERIES["logreg"],
                "+ Bayes prior correction" if best == "prior_shift" else f"+ {best} calibration",
            ),
        ],
        "Logistic regression with inverse-frequency class weights (the original design): "
        "raw vs repaired probabilities, out-of-fold on dev",
        "calibration_weighted.png",
        out,
    )


def fig_confusion(results: dict, fam: str, out: Path):
    d = results["decisions"][fam]["test"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, key, title in (
        (axes[0], "confusion_argmax", "Argmax tier"),
        (axes[1], "confusion_bayes", "Cost-optimal tier"),
    ):
        cm = np.asarray(d[key], dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", [SURFACE, *BLUES])
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    f"{norm[i, j]:.0%}\n({int(cm[i, j])})",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#ffffff" if norm[i, j] > 0.55 else INK,
                )
        ax.set_xticks(range(3), CLASSES)
        ax.set_yticks(range(3), CLASSES)
        ax.set_xlabel("Predicted")
        ax.grid(False)
        ax.set_title(title)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("Actual")
    fig.suptitle(
        f"{NAMES[fam]} on the test set, row-normalised (counts)",
        x=0.01,
        ha="left",
        fontsize=9,
        color=INK2,
        y=1.0,
    )
    _save(fig, "confusion.png", out)


def fig_importance(results: dict, out: Path, top: int = 12):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, fam, title in (
        (axes[0], "xgboost", "XGBoost: mean |TreeSHAP|, High tier"),
        (axes[1], "logreg", "Logistic regression: mean |coef × x|, High tier"),
    ):
        rows = sorted(results["importance"][fam], key=lambda r: r["High"], reverse=True)[:top]
        labels = [FEATURE_LABELS.get(r["feature"], r["feature"]) for r in rows]
        vals = [r["High"] for r in rows]
        ax.barh(range(len(rows)), vals, height=0.6, color=SERIES[fam])
        ax.set_yticks(range(len(rows)), labels)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
        ax.set_xlabel("log-odds units")
    fig.suptitle(
        "Global drivers of the High-risk score on the test set",
        x=0.01,
        ha="left",
        fontsize=9,
        color=INK2,
        y=1.0,
    )
    fig.tight_layout()
    _save(fig, "importance.png", out)


def fig_cost(results: dict, preds: dict, out: Path):
    y, exposure = preds["y_test"], preds["exposure_test"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    visible = []
    for fam in MODEL_FAMILIES:
        curve = approval_curve(preds[f"test_{fam}"], y, exposure)
        visible.extend(curve.loc[curve["approval_rate"] >= 0.6, "profit_per_applicant"])
        axes[0].plot(
            curve["approval_rate"],
            curve["medhigh_rate_approved"],
            color=SERIES[fam],
            label=NAMES[fam],
        )
        axes[1].plot(
            curve["approval_rate"],
            curve["profit_per_applicant"],
            color=SERIES[fam],
            label=NAMES[fam],
        )
        d = results["decisions"][fam]["test"]
        axes[1].plot(
            d["approval_rate"],
            d["profit_per_applicant"],
            "o",
            color=SERIES[fam],
            ms=8,
            mec=SURFACE,
            mew=2,
        )
        axes[0].plot(
            d["approval_rate"],
            d["medhigh_rate_approved"],
            "o",
            color=SERIES[fam],
            ms=8,
            mec=SURFACE,
            mew=2,
        )
    oracle = results["decisions"]["reference"]["oracle"]["approval_rate"]
    for ax in axes:
        ax.axvline(oracle, color=AXIS, lw=1, zorder=0)
    axes[0].text(oracle, 0.11, " share of Low tier\n (oracle approval)", fontsize=8, color=MUTED)
    axes[0].set_title("Bad rate among approved")
    axes[0].set_ylabel("Medium or High tier among approved")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    axes[0].set_ylim(0, 0.125)
    pad = 0.05 * (max(visible) - min(visible))
    axes[1].set_ylim(min(visible) - pad, max(visible) + pad)
    axes[1].axhline(0, color=AXIS, lw=1, zorder=0)
    axes[1].set_title("Unit economics (dots = cutoff chosen on dev out-of-fold)")
    axes[1].set_ylabel("Profit proxy per applicant (USD)")
    for ax in axes:
        ax.set_xlabel("Approval rate")
        ax.set_xlim(0.6, 1.0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    axes[1].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        f"Approval cutoff sweep on the test set. Assumed PD by tier {COST.pd_by_tier}, "
        f"LGD {COST.lgd}, merchant fee {COST.merchant_fee:.0%}",
        x=0.01,
        ha="left",
        fontsize=9,
        color=INK2,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "cost_curve.png", out)


# ----------------------------------------------------------------------------- markdown
def _pm(d: dict, k: str, nd: int = 3) -> str:
    return f"{d[k]['mean']:.{nd}f} ± {d[k]['std']:.{nd}f}"


def _ci(test: dict, k: str, nd: int = 3) -> str:
    lo, hi = test["ci95"][k]
    return f"{test['calibrated'][k]:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def benchmark_markdown(results: dict, meta: dict) -> str:
    sel = results["selection"]["selected"]
    exp = results["experiment"]
    L = []
    w = L.append
    w("# Benchmark report\n")
    w(
        f"Generated by `python -m bnpl_risk.train` + `python -m bnpl_risk.reporting`. "
        f"Data sha256 `{exp['data']['sha256'][:16]}…`, seed {exp['seed']}, "
        f"dev n = {exp['data']['n_dev']}, held-out test n = {exp['data']['n_test']} "
        f"(class counts {exp['data']['class_counts_test']} for Low/Medium/High). "
        f"Git revision `{meta.get('git_revision', '?')}`.\n"
    )
    w("## Protocol\n")
    w(
        f"- Every family is tuned with the same inner {exp['tuning']['inner_cv_folds']}-fold CV on the "
        f"development set only, scored by **log loss**: 24-point grid for logistic regression, "
        f"{exp['tuning']['n_iter']} random-search draws each for XGBoost and the tabular ResNet."
    )
    w(
        f"- Tuned configurations are then compared on identical folds: {exp['cv']['scheme']}. "
        "Paired differences use the Nadeau–Bengio corrected resampled t-test."
    )
    w(
        "- Calibration method (none / sigmoid / isotonic / temperature) is chosen per family by "
        "out-of-fold Brier score; decision cutoffs are chosen on out-of-fold development predictions."
    )
    w(
        "- The test set is scored once, at the end. 95% CIs: stratified bootstrap, "
        f"{1000 if not exp['fast_mode'] else 50} resamples.\n"
    )

    w("## Cross-validated performance (dev, mean ± sd over folds, uncalibrated)\n")
    w(
        "| Model | Log loss | Brier | Macro F1 | ROC-AUC (macro) | PR-AUC (macro) | PR-AUC High | Recall High | Precision High |"
    )
    w("|---|---|---|---|---|---|---|---|---|")
    for f in MODEL_FAMILIES:
        c = results["cv"][f]
        w(
            f"| {NAMES[f]}{' **(selected)**' if f == sel else ''} | {_pm(c, 'log_loss', 4)} | "
            f"{_pm(c, 'brier', 4)} | {_pm(c, 'macro_f1')} | {_pm(c, 'roc_auc_macro', 4)} | "
            f"{_pm(c, 'pr_auc_macro')} | {_pm(c, 'pr_auc_High')} | {_pm(c, 'recall_High')} | "
            f"{_pm(c, 'precision_High')} |"
        )
    s = results["selection"]
    w(
        f"\nSelection rule: {s['rule']}. Best by log loss: **{NAMES[s['best_by_log_loss']]}**; "
        f"threshold {s['threshold']:.4f} (corrected SE {s['corrected_se']:.4f}); eligible: "
        f"{', '.join(NAMES[e] for e in s['eligible'])}; selected: **{NAMES[sel]}**.\n"
    )

    w("### Paired comparisons (corrected t-test over the CV folds)\n")
    w("| A | B | Metric | Mean(A − B) | t | p |")
    w("|---|---|---|---|---|---|")
    for c in results["comparisons"]:
        a = NAMES.get(c["a"], c["a"])
        b = NAMES.get(c["b"], c["b"])
        w(f"| {a} | {b} | {c['metric']} | {c['mean_diff']:+.4f} | {c['t']:.2f} | {c['p']:.3f} |")

    w("\n## Held-out test set (calibrated, point estimate [95% bootstrap CI])\n")
    w(
        "| Model | Calibration | Log loss | Brier | Macro F1 | PR-AUC High | Recall High | Precision High | ECE High |"
    )
    w("|---|---|---|---|---|---|---|---|---|")
    for f in MODEL_FAMILIES:
        t = results["test"][f]
        w(
            f"| {NAMES[f]} | {results['calibration'][f]['chosen']} | {_ci(t, 'log_loss', 4)} | "
            f"{_ci(t, 'brier', 4)} | {_ci(t, 'macro_f1')} | {_ci(t, 'pr_auc_High')} | "
            f"{_ci(t, 'recall_High')} | {_ci(t, 'precision_High')} | {_ci(t, 'ece_High')} |"
        )

    w("\nPer-class test metrics, selected model (calibrated, argmax):\n")
    w("| Tier | Precision | Recall | F1 | PR-AUC (OvR) | ROC-AUC (OvR) | ECE |")
    w("|---|---|---|---|---|---|---|")
    tc = results["test"][sel]["calibrated"]
    for c in CLASSES:
        w(
            f"| {c} | {tc[f'precision_{c}']:.3f} | {tc[f'recall_{c}']:.3f} | {tc[f'f1_{c}']:.3f} | "
            f"{tc[f'pr_auc_{c}']:.3f} | {tc[f'roc_auc_{c}']:.4f} | {tc[f'ece_{c}']:.4f} |"
        )

    w("\n## Calibration (out-of-fold on dev)\n")
    w("| Model | Method | Brier | Log loss | ECE Low | ECE Medium | ECE High |")
    w("|---|---|---|---|---|---|---|")
    for f in MODEL_FAMILIES:
        cal = results["calibration"][f]
        for m, v in cal["oof"].items():
            mark = " ✔" if m == cal["chosen"] else ""
            w(
                f"| {NAMES[f]} | {m}{mark} | {v['brier']:.4f} | {v['log_loss']:.4f} | "
                f"{v['ece_Low']:.4f} | {v['ece_Medium']:.4f} | {v['ece_High']:.4f} |"
            )

    cw = results["calibration_class_weighted"]
    w("\n### Repairing class-weighted probabilities (the original design)\n")
    w(
        "Same logistic regression, `class_weight='balanced'`, out-of-fold on dev. Inverse-frequency "
        "weights make the implied class priors uniform, so raw probabilities overstate Medium risk "
        "and understate Low. Findings: one temperature cannot shift class priors; one-vs-rest "
        "sigmoid/isotonic on a multinomial model's decision function are actively harmful (each "
        "column is calibrated in isolation although softmax logits are only defined up to a "
        "shared shift); the analytic Bayes prior correction "
        "p ∝ p_w · π/π_w restores both calibration and the base rates. The cleaner fix, used by "
        "the deployed pipeline, is upstream: tuning on log loss selected *no* class weights for "
        "every family, and class asymmetry is handled in the decision layer (cost matrix, cutoff).\n"
    )
    w(
        "| Variant | Brier | Log loss | ECE Medium | ECE High | Mean P(Medium) | Mean P(High) | Macro F1 | Recall High |"
    )
    w("|---|---|---|---|---|---|---|---|---|")
    base = cw["mean_pred_vs_base_rate"]["none"]["base_rate"]
    labels = {
        "none": "balanced weights, uncalibrated",
        "prior_shift": "balanced weights + Bayes prior correction",
    }
    for m in cw["oof"]:
        v, mp = cw["oof"][m], cw["mean_pred_vs_base_rate"][m]["mean_pred"]
        label = labels.get(m, f"balanced weights + {m}")
        if m == cw["best"]:
            label += " ✔"
        w(
            f"| {label} | {v['brier']:.4f} | {v['log_loss']:.4f} | {v['ece_Medium']:.4f} | "
            f"{v['ece_High']:.4f} | {mp[1]:.3f} | {mp[2]:.3f} | {v['macro_f1']:.3f} | {v['recall_High']:.3f} |"
        )
    w(f"| *observed base rate* | | | | | {base[1]:.3f} | {base[2]:.3f} | | |")

    w("\n## Ablations (same folds)\n")
    w("| Variant | Log loss | Macro F1 | PR-AUC High |")
    w("|---|---|---|---|")
    for f in ("logreg", "xgboost"):
        w(
            f"| {NAMES[f]}: engineered features, median imputation (baseline) | "
            f"{_pm(results['cv'][f], 'log_loss', 4)} | {_pm(results['cv'][f], 'macro_f1')} | "
            f"{_pm(results['cv'][f], 'pr_auc_High')} |"
        )
        for k, label in (("raw_features", "raw features only"), ("knn_imputer", "KNN imputation")):
            a = results["ablations"][f"{f}/{k}"]
            w(
                f"| {NAMES[f]}: {label} | {_pm(a, 'log_loss', 4)} | {_pm(a, 'macro_f1')} | "
                f"{_pm(a, 'pr_auc_High')} |"
            )
    for k, label in (
        ("numeric_only", "Original XGBoost config, numeric features only (as in the old notebook)"),
        ("with_categoricals", "Original XGBoost config, with categorical features"),
    ):
        a = results["ablations"][f"legacy_xgb/{k}"]
        w(f"| {label} | {_pm(a, 'log_loss', 4)} | {_pm(a, 'macro_f1')} | {_pm(a, 'pr_auc_High')} |")

    w("\n## Cost-sensitive decisions (test set; cutoff chosen on dev out-of-fold predictions)\n")
    w(
        f"Assumptions (illustrative, configurable in `config.CostAssumptions`): PD by tier "
        f"{COST.pd_by_tier}, LGD {COST.lgd}, merchant fee {COST.merchant_fee:.0%} of purchase value, "
        "exposure = average transaction value. Break-even PD = fee / LGD = "
        f"{COST.merchant_fee / COST.lgd:.3f}.\n"
    )
    w(
        "| Policy | PD cutoff | Approval rate | High share of approved | Medium+High share of approved | High tier declined | Profit proxy / applicant | Misclass. cost (argmax → Bayes) |"
    )
    w("|---|---|---|---|---|---|---|---|")
    ref = results["decisions"]["reference"]
    rows = [
        (NAMES[f], results["decisions"][f]["cutoff"], results["decisions"][f]["test"])
        for f in MODEL_FAMILIES
    ]
    rows += [
        ("Approve everyone", 1.0, ref["approve_all"]),
        ("Oracle (true tier known)", None, ref["oracle"]),
    ]
    for name, cut, d in rows:
        cut_s = f"{cut:.4f}" if cut is not None else "n/a"
        w(
            f"| {name} | {cut_s} | {d['approval_rate']:.1%} | {d['high_rate_approved']:.2%} | "
            f"{d['medhigh_rate_approved']:.2%} | {d['high_captured']:.1%} | "
            f"${d['profit_per_applicant']:.2f} | {d['misclassification_cost_argmax']:.3f} → "
            f"{d['misclassification_cost_bayes']:.3f} |"
        )

    w("\n## Feature importance (test set)\n")
    w("| Rank | XGBoost mean abs TreeSHAP (High) | Logistic regression mean abs coef × x (High) |")
    w("|---|---|---|")
    xs = sorted(results["importance"]["xgboost"], key=lambda r: r["High"], reverse=True)[:10]
    ls = sorted(results["importance"]["logreg"], key=lambda r: r["High"], reverse=True)[:10]
    for i, (a, b) in enumerate(zip(xs, ls, strict=True), 1):
        w(
            f"| {i} | {FEATURE_LABELS.get(a['feature'], a['feature'])} ({a['High']:.3f}) | "
            f"{FEATURE_LABELS.get(b['feature'], b['feature'])} ({b['High']:.3f}) |"
        )
    w("\nPermutation importance on raw inputs (increase in test log loss when shuffled):\n")
    w("| Input | " + " | ".join(NAMES[f] for f in MODEL_FAMILIES) + " |")
    w("|---|" + "---|" * len(MODEL_FAMILIES))
    perm = {
        f: {r["feature"]: r["importance"] for r in results["permutation_importance"][f]}
        for f in MODEL_FAMILIES
    }
    order = sorted(perm[sel], key=perm[sel].get, reverse=True)
    for feat in order:
        w(f"| {feat} | " + " | ".join(f"{perm[f][feat]:.4f}" for f in MODEL_FAMILIES) + " |")

    w("\n## Figures\n")
    for fig in (
        "cv_benchmark",
        "reliability",
        "calibration_weighted",
        "confusion",
        "cost_curve",
        "importance",
    ):
        w(f"![{fig}](figures/{fig}.png)\n")
    return "\n".join(L) + "\n"


def render(models_dir: Path = MODELS_DIR, reports_dir: Path = REPORTS_DIR) -> None:
    _style()
    results = json.loads((reports_dir / "results.json").read_text())
    meta = json.loads((models_dir / "model_meta.json").read_text())
    preds = dict(np.load(models_dir / "predictions.npz"))
    sel = results["selection"]["selected"]
    figs = reports_dir / "figures" if reports_dir != REPORTS_DIR else FIGURES_DIR
    fig_cv(results, figs)
    fig_reliability(results, sel, figs)
    fig_calibration_weighted(results, figs)
    fig_confusion(results, sel, figs)
    fig_importance(results, figs)
    fig_cost(results, preds, figs)
    (reports_dir / "benchmark.md").write_text(benchmark_markdown(results, meta), encoding="utf-8")
    print(f"wrote {reports_dir / 'benchmark.md'} and figures in {figs}")


if __name__ == "__main__":
    render()
