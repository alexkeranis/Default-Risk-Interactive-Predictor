# Methodology Audit of the Original Pipeline

This audit covers the repository at commit `875b381`, before the overhaul. It lists every methodological error, leakage path and unrealistic assumption found in `src/`, `notebooks/` and `data/`, and estimates how each one distorts the README claim: **"PyTorch macro F1 0.84 vs XGBoost 0.75, a 9 percentage point improvement."**

Every probe below was run on the raw CSV with leak-free scikit-learn pipelines and stratified 5-fold CV (seed 42).

## Verdict

The comparison is invalid in both directions. The neural network's score is inflated because the model was selected on the test set. The XGBoost score is deflated because it was trained without the categorical features. On equal footing, a default gradient-boosted tree model with all features reaches **0.880 ± 0.008** macro F1 in 5-fold CV. That is above the reported PyTorch figure. The rerun benchmark is in [`benchmark.md`](benchmark.md).

---

## A. Leakage and invalid evaluation

| # | Finding | Where | Effect on the 0.84 / 0.75 numbers |
|---|---|---|---|
| A1 | **The NN was selected on the test set.** The notebook passes `test_loader` as the validation loader, so test macro F1 drives early stopping (patience 15), `ReduceLROnPlateau` and best-checkpoint restore. The validation split built in `prepare_and_save_data` is never used. | `notebooks/02_model_training.ipynb` cell 14; `src/utils.py:187-243` | **Inflates the NN.** The reported 0.8455 is the maximum test F1 over 55 epochs (epoch 40), so it is an optimistically biased order statistic, not an out-of-sample estimate. |
| A2 | **XGBoost never saw a categorical feature.** `run_xgboost_baseline(X_train, ...)` receives only the 7 scaled numeric columns (`X_train.shape == (7000, 7)`). `Employment_Status` and `Late_Payment_History` are omitted. They are the two strongest predictors: Unemployed customers are 36.8% High vs 0.2% for Employed. | `notebooks/02` cell 6; `src/notebook_utils.py:156` | **Deflates XGBoost by about 10 pp.** Probe with the same GBDT: numeric only **0.783 ± 0.009**, numeric + categoricals **0.880 ± 0.008**. This alone accounts for the whole claimed +9 pp NN advantage. |
| A3 | **Unequal tuning effort.** The NN's loss (focal γ = 1 and 2 vs weighted CE), architecture and schedule were iterated on the same split, per the notebook prose. XGBoost ran one hand-picked configuration. XGBoost was also given `eval_set=[(X_test, y_test)]`; this is monitoring only, but it is still test-set exposure. | `notebooks/02` markdown cell 0; `src/notebook_utils.py:212-223` | Favours the NN. The garden of forking paths on a single split inflates whichever model received the attention. |
| A4 | **One split, no uncertainty.** A single 15% test split holds only 78 High and 102 Medium rows. No CV, no confidence intervals, no repeated seeds. | `src/config.py:47-49` | Probe: the same model on 20 random 15% splits ranged **0.864–0.908** macro F1 (sd 0.011). A 9 pp gap from one split, with selection on that same split, carries no evidential weight. |
| A5 | **EDA on the full dataset.** Imputation diagnostics, correlations and feature decisions used all 10,000 rows, including future test rows. | `notebooks/01` cells 14–24 | Mild selection leakage: modelling choices are informed by test rows. |

## B. Train/serve skew and serving defects

| # | Finding | Where | Consequence |
|---|---|---|---|
| B1 | **Different imputers in training and serving.** Training fills missing values with a KNN imputer fit on train. Batch inference fills NaN with the RobustScaler's medians. The KNN imputer is never serialized. | `src/utils.py:28-60` vs `src/inference.py:97-118` | Served predictions for incomplete rows come from a different input distribution than the model was trained on. |
| B2 | **Zero or invalid income becomes the safest possible applicant.** `Debt_to_Income_Ratio = debt / income if income != 0 else 0.0`. DTI is the dominant risk driver, so an applicant with no income is scored as having no leverage. | `src/inference.py:20`; `src/utils.py:88-92` | Silent, catastrophic under-pricing of risk on bad inputs. |
| B3 | **Weak input validation.** Single-record mode only *warns* on out-of-range values. Batch mode does no range checks at all. Non-numeric values are replaced silently. | `src/inference.py:22-26, 106-123` | Malformed rows get scored instead of rejected. |
| B4 | **Serving ranges don't match the training support.** `RANGES` allows debt ≤ $50,000, transaction ≤ $2,000, income ≤ $150,000 and age ≤ 100. The training maxima are $2,731, $1,420, $139,452 and 64. The README's own demo (notebook 03) scores a debt of **$23,500**, 8.6× the training maximum, and prints "91.71% confidence". | `src/config.py:38-45`; `notebooks/03` cell 6 | Pure extrapolation is presented as a confident prediction. |
| B5 | **Unseen categories map to embedding row 0, which is never trained.** Codes start at 1 and the training data has no categorical NaN, so row 0 keeps its random initialization. | `src/utils.py:63-67, 285-299` | The notebook 04 result for "Astronaut" (→ Low) is arbitrary, not graceful handling. |
| B6 | **Committed outputs can't be reproduced.** Notebook 03 prints `Embedding(4,8)`, `(2,8)` and `(5,8)`, but the current preprocessing produces cardinalities `[5, 3, 6]`, so this checkpoint cannot come from this code. Notebook paths point to `C:\Users\Alexz\BNPL_ML`. | `notebooks/03` cell 4; `notebooks/04` | The published numbers can't be traced to the published code. |

## C. Unrealistic assumptions and reporting gaps

| # | Finding | Consequence |
|---|---|---|
| C1 | **Softmax outputs from inverse-frequency-weighted cross-entropy are presented as "confidence".** Inverse-frequency weights make the implied class priors uniform (1/3 each), inflating the Medium and High base rates by about 4.9× and 6.4×, so these are not calibrated probabilities. | Risk-based pricing or expected-loss estimates built on these numbers would be biased upward for the minority classes and could not be audited. |
| C2 | **Macro F1 at argmax is the only metric.** No PR-AUC, ROC-AUC, Brier score, log loss, calibration curve or cost analysis. | In underwriting, approving a High-risk applicant costs far more than declining a Low-risk one. Argmax and macro F1 weight these errors as if they were equally costly. |
| C3 | **Incomplete determinism.** No `torch.use_deterministic_algorithms`, no seeded DataLoader generator, CUDA training, and the notebook seeds *after* the XGBoost run. | Reruns drift by about 1 F1 point, as the notebook itself admits. |
| C4 | **The dataset's synthetic nature is undocumented.** The labels are close to a deterministic rule: High median BNPL-debt/income is 0.067 vs 0.004 for Low, and late payment history and employment status separate the rest. A **depth-4 decision tree reaches 0.84 macro F1**, the same as the headline NN. Income is censored at a $5,000 floor (exactly the 5th percentile of the High class). Missingness is about 3% and missing completely at random (MCAR) across classes, so it carries no signal. `Customer_ID` is clean (correlation with the target 0.003, no duplicates). Debt and active loans are correlated at r = 0.886. | Absolute scores on this dataset say little about real BNPL portfolios. They are benchmark numbers on a synthetic generator, and the README must say so. |
| C5 | **"Default risk tier" is treated as if it were observed default.** The labels are risk tiers, not realised defaults. | Any expected-loss figure requires an explicit, documented mapping from tier to probability of default (PD). |

## Remediation map

| Finding | Fixed by |
|---|---|
| A1, A3, A4 | `bnpl_risk.tuning` / `bnpl_risk.evaluation`: a held-out test set touched once; equal-budget inner-CV tuning for every family; repeated stratified 5×3 CV with mean ± std; corrected paired t-tests; bootstrap CIs. The NN's early stopping uses a slice of each *training fold*. |
| A2 | Every model receives the same raw inputs through the same `Pipeline`. |
| A5 | `notebooks/01_eda.ipynb` works on the development split only. |
| B1–B3 | One serialized, fitted `Pipeline` used for training, CV, single-record and batch inference. `bnpl_risk.schema` rejects malformed rows with explicit errors. The income floor guard is part of the feature engineering. |
| B4 | Training-support ranges are stored in `models/model_meta.json`. Out-of-support values are flagged at inference, and the demo's sliders are bounded to the support. |
| B5 | Unknown categories are rejected by the schema. The NN reserves an unknown index that is trained through category dropout. |
| B6 | Notebooks are rewritten to call the package and re-executed. 03/04 are replaced by the CLI and `tests/`. |
| C1 | Post-hoc isotonic/sigmoid calibration chosen on out-of-fold predictions, with reliability curves and ECE reported. |
| C2, C5 | Full metric suite plus cost-sensitive decision analysis under explicit, configurable PD/LGD/fee assumptions. |
| C3 | `bnpl_risk.seed.seed_everything()` is called at every entry point; CPU, deterministic torch algorithms, seeded generators; determinism tests. |
| C4 | Dataset limitations section in the README and a banner in the demo. |
