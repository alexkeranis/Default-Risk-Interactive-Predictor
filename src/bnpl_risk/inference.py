"""Serving: one code path for single records and batches.

    python -m bnpl_risk.inference --json '{"Age": 31, "Income_USD": 42000, ...}'
    python -m bnpl_risk.inference --csv customers.csv [--out scored.csv] [--strict]

The fitted object loaded here is the exact pipeline evaluated in training (imputation,
feature engineering, encoding, model, calibration), so there is no re-implemented
preprocessing to drift out of sync.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from bnpl_risk.config import CLASSES, INPUT_COLS, META_PATH, MODEL_PATH, CostAssumptions
from bnpl_risk.evaluation import cost_optimal_tier, expected_pd
from bnpl_risk.explain import base_pipeline
from bnpl_risk.preprocessing import transform_features
from bnpl_risk.schema import SchemaError, ValidationResult, assert_valid, validate_frame
from bnpl_risk.seed import seed_everything

PROB_COLS = [f"Prob_{c}" for c in CLASSES]


class RiskScorer:
    def __init__(self, model, meta: dict):
        self.model = model
        self.meta = meta
        self.support = meta["support"]
        d = meta["decision"]
        c = d["cost_assumptions"]
        self.cutoff = float(d["pd_cutoff"])
        self.cost = CostAssumptions(
            pd_by_tier=tuple(c["pd_by_tier"]),
            lgd=c["lgd"],
            merchant_fee=c["merchant_fee"],
            cost_matrix=tuple(tuple(r) for r in c["cost_matrix"]),
        )

    @classmethod
    def load(cls, model_path: Path = MODEL_PATH, meta_path: Path = META_PATH) -> RiskScorer:
        for p in (model_path, meta_path):
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing artifact {p}. Run `python -m bnpl_risk.train`.")
        return cls(joblib.load(model_path), json.loads(Path(meta_path).read_text()))

    # ------------------------------------------------------------------ core API
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        return validate_frame(df, self.support)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Strict: raises ``SchemaError`` if any row is invalid."""
        frame = assert_valid(df, self.support).frame
        return self.model.predict_proba(frame)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Model-ready feature matrix (for parity checks and debugging)."""
        frame = assert_valid(df, self.support).frame
        return transform_features(base_pipeline(self.model), frame)

    def score(self, df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
        """Probabilities plus the business decision for every row. Invalid rows are kept,
        marked ``REJECTED_INPUT`` with their errors, and never scored (unless ``strict``,
        in which case any invalid row raises)."""
        res = self.validate(df)
        valid = res.valid_mask
        if strict and not valid.all():
            raise SchemaError({int(i): e for i, e in enumerate(res.errors.tolist()) if e})
        out = df.reset_index(drop=True).copy()
        proba = np.full((len(df), len(CLASSES)), np.nan)
        if valid.any():
            proba[valid] = self.model.predict_proba(res.frame[valid])
        pd_eff = np.full(len(df), np.nan)
        tier = np.array([None] * len(df), dtype=object)
        decision = np.array(["REJECTED_INPUT"] * len(df), dtype=object)
        if valid.any():
            pd_eff[valid] = expected_pd(proba[valid], self.cost)
            tier[valid] = np.asarray(CLASSES)[cost_optimal_tier(proba[valid], self.cost)]
            decision[valid] = np.where(pd_eff[valid] <= self.cutoff, "APPROVE", "DECLINE")
        exposure = pd.to_numeric(out["Average_Transaction_Value_USD"], errors="coerce")
        for k, col in enumerate(PROB_COLS):
            out[col] = proba[:, k]
        out["Expected_PD"] = pd_eff
        out["Expected_Loss_USD"] = exposure.to_numpy() * self.cost.lgd * pd_eff
        out["Risk_Tier"] = tier
        out["Decision"] = decision
        out["OOD_Flags"] = res.ood_flags.map(";".join).to_numpy()
        out["Errors"] = res.errors.map("; ".join).to_numpy()
        return out

    def score_one(self, record: dict) -> dict:
        row = self.score(pd.DataFrame([record]), strict=True).iloc[0]
        return {
            "probabilities": {c: float(row[f"Prob_{c}"]) for c in CLASSES},
            "expected_pd": float(row["Expected_PD"]),
            "expected_loss_usd": float(row["Expected_Loss_USD"]),
            "risk_tier": row["Risk_Tier"],
            "decision": row["Decision"],
            "out_of_distribution": [f for f in row["OOD_Flags"].split(";") if f],
        }


def main(argv: list[str] | None = None) -> int:
    seed_everything()
    ap = argparse.ArgumentParser(description="BNPL default-risk scoring")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="batch input CSV")
    src.add_argument("--json", help="single applicant as a JSON object")
    ap.add_argument("--out", type=Path, help="batch output path (default: <input>_scored.csv)")
    ap.add_argument("--strict", action="store_true", help="fail the batch on any invalid row")
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    ap.add_argument("--meta", type=Path, default=META_PATH)
    args = ap.parse_args(argv)

    try:
        scorer = RiskScorer.load(args.model, args.meta)
        if args.json:
            record = json.loads(args.json)
            missing = [c for c in INPUT_COLS if c not in record]
            if missing:
                raise SchemaError({0: [f"missing field(s) {missing}"]})
            print(json.dumps(scorer.score_one(record), indent=2))
            return 0
        df = pd.read_csv(args.csv)
        scored = scorer.score(df, strict=args.strict)
        out = args.out or args.csv.with_name(f"{args.csv.stem}_scored.csv")
        scored.to_csv(out, index=False)
        n_bad = int((scored["Decision"] == "REJECTED_INPUT").sum())
        n_ood = int((scored["OOD_Flags"] != "").sum())
        print(f"Scored {len(scored) - n_bad}/{len(scored)} rows -> {out}")
        if n_bad:
            print(f"{n_bad} row(s) rejected for invalid input (see the Errors column)")
        if n_ood:
            print(f"{n_ood} row(s) outside the training support (see OOD_Flags)")
        return 0
    except SchemaError as e:
        print(f"Input error: {e}", file=sys.stderr)
        return 2
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
