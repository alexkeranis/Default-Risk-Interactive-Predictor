// JS <-> Python parity: the browser model must reproduce the fitted Python pipelines.
// Run: node --test docs/tests/parity.test.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { costOptimalTier, expectedPD, portfolioAtCutoff, predict } from "../assets/model.js";

const load = (name) => JSON.parse(readFileSync(fileURLToPath(new URL(`../data/${name}`, import.meta.url))));
const bundle = load("model.json");
const golden = load("golden_cases.json");

// Linear models match to float64 round-off. Trees: XGBoost accumulates in float32.
const TOL = { logreg: { p: 1e-9, phi: 1e-9 }, xgboost: { p: 1e-5, phi: 1e-4 } };

function maxAbsDiff(a, b) {
  return Math.max(...a.flat().map((v, i) => Math.abs(v - b.flat()[i])));
}

for (const [family, expected] of Object.entries(golden.models)) {
  test(`${family}: features, probabilities and contributions match Python`, () => {
    const spec = bundle.models[family];
    golden.inputs.forEach((input, i) => {
      const out = predict(spec, input, bundle);
      assert.ok(maxAbsDiff(out.features, expected.features[i]) < 1e-9, `features row ${i}`);
      const dp = maxAbsDiff(out.proba, expected.proba[i]);
      assert.ok(dp < TOL[family].p, `proba row ${i}: |diff| = ${dp}`);
      const dphi = maxAbsDiff(out.contributions, expected.contributions[i]);
      assert.ok(dphi < TOL[family].phi, `contributions row ${i}: |diff| = ${dphi}`);
      assert.ok(Math.abs(out.proba.reduce((a, b) => a + b, 0) - 1) < 1e-12, `sum row ${i}`);
    });
  });
}

test("decision helpers", () => {
  const cost = bundle.cost_assumptions;
  assert.equal(expectedPD([1, 0, 0], cost), cost.pd_by_tier[0]);
  assert.equal(costOptimalTier([0.6, 0.1, 0.3], cost), 2); // high-risk mass dominates expected cost
  const pf = portfolioAtCutoff([[1, 0, 0], [0, 0, 1]], [0, 2], [100, 100], cost, 0.1);
  assert.equal(pf.approval_rate, 0.5);
  assert.equal(pf.high_captured, 1);
});
