// Browser-side scoring. A line-for-line mirror of bnpl_risk.export_web.reference_predict,
// verified in CI against golden cases produced by the fitted Python pipelines
// (docs/tests/parity.test.mjs). Pure functions, no DOM: shared by the page and Node.

export const NUMERIC = [
  "Age",
  "Income_USD",
  "Credit_Score",
  "Total_BNPL_Active_Loans",
  "Total_BNPL_Debt_USD",
  "Average_Transaction_Value_USD",
];
export const NULLABLE = ["Income_USD", "Credit_Score"];
const LOGIT_EPS = 1e-12;

const isMissing = (v) => v === null || v === undefined || Number.isNaN(v);

/** Raw applicant -> named model features (before scaling). Mirrors model_feature_frame. */
export function buildFeatureMap(input, spec, bundle) {
  const x = { ...input };
  const f = {};
  for (const c of NULLABLE) {
    const miss = isMissing(x[c]);
    f[`missing_${c}`] = miss ? 1 : 0;
    if (miss) x[c] = spec.imputer_medians[c];
  }
  for (const c of NUMERIC) f[c] = Number(x[c]);

  const income = Math.max(Number(x.Income_USD), bundle.income_floor);
  const debt = Number(x.Total_BNPL_Debt_USD);
  const loans = Number(x.Total_BNPL_Active_Loans);
  const ticket = Number(x.Average_Transaction_Value_USD);
  const score = Number(x.Credit_Score);
  const late = x.Late_Payment_History === "Yes" ? 1 : 0;
  const unstable = x.Employment_Status === "Student" || x.Employment_Status === "Unemployed" ? 1 : 0;
  const dti = debt / income;
  const subprime = score < 580 ? 1 : 0;

  f.bnpl_dti = dti;
  f.ticket_to_monthly_income = ticket / (income / 12.0);
  f.debt_per_loan = loans > 0 ? debt / loans : 0;
  f.loan_stacking = loans >= 3 ? 1 : 0;
  f.subprime = subprime;
  f.near_prime = score >= 580 && score < 670 ? 1 : 0;
  f.unstable_income = unstable;
  f.late_x_dti = late * dti;
  f.unstable_x_dti = unstable * dti;
  f.late_x_subprime = late * subprime;
  f.log_income = Math.log1p(income);
  f.log_debt = Math.log1p(debt);
  f.young_borrower = Number(x.Age) < 25 ? 1 : 0;

  for (const [col, levels] of Object.entries(bundle.input_schema.categorical)) {
    for (const lvl of levels) f[`${col}_${lvl}`] = x[col] === lvl ? 1 : 0;
  }
  return f;
}

export function softmax(z) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

function interp(x, xs, ys) {
  // np.interp on a clipped input (IsotonicRegression with out_of_bounds="clip").
  if (x <= xs[0]) return ys[0];
  const n = xs.length;
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  if (xs[hi] === xs[lo]) return ys[hi];
  return ys[lo] + ((x - xs[lo]) * (ys[hi] - ys[lo])) / (xs[hi] - xs[lo]);
}

export function calibrate(cal, raw, base) {
  if (cal.method === "none") return base;
  const pred = cal.input === "decision_function" ? raw : base;
  if (cal.method === "temperature") {
    const logits = cal.input === "decision_function" ? pred : pred.map((p) => Math.log(p + LOGIT_EPS));
    return softmax(logits.map((v) => cal.beta * v));
  }
  const p = pred.map((v, k) => {
    const c = cal.per_class[k];
    if (cal.method === "isotonic") return interp(v, c.x, c.y);
    return 1 / (1 + Math.exp(c.a * v + c.b));
  });
  const s = p.reduce((a, b) => a + b, 0);
  return s > 0 ? p.map((v) => v / s) : p.map(() => 1 / p.length);
}

/**
 * Score one applicant with one exported model.
 * Returns { proba, raw, contributions: [class][feature + bias], features: number[] }.
 * Contributions are exact for the linear model and Saabas path attributions for trees,
 * both in the base model's log-odds space (before calibration).
 */
export function predict(spec, input, bundle) {
  const fmap = buildFeatureMap(input, spec, bundle);
  const x = spec.features.map((n) => fmap[n]);
  const K = bundle.classes.length;
  const P = x.length;
  const contrib = Array.from({ length: K }, () => new Array(P + 1).fill(0));
  let raw;
  if (spec.type === "linear") {
    const z = x.map((v, j) => (v - spec.mean[j]) / spec.scale[j]);
    raw = spec.intercept.map((b, k) => {
      let s = 0;
      for (let j = 0; j < P; j++) {
        const c = z[j] * spec.coef[k][j];
        contrib[k][j] = c;
        s += c;
      }
      contrib[k][P] = b;
      return s + b;
    });
  } else {
    raw = new Array(K).fill(0);
    const xf = x.map((v) => Math.fround(v));
    spec.trees.forEach((t, i) => {
      const k = i % spec.n_classes;
      let node = 0;
      contrib[k][P] += t.value[0];
      while (t.feat[node] >= 0) {
        const v = xf[t.feat[node]];
        const next = Number.isNaN(v) ? t.missing[node] : v < Math.fround(t.thr[node]) ? t.yes[node] : t.no[node];
        contrib[k][t.feat[node]] += t.value[next] - t.value[node];
        node = next;
      }
      raw[k] += t.value[node];
    });
    raw = raw.map((r, k) => r + spec.base_margin[k]);
    spec.base_margin.forEach((b, k) => (contrib[k][P] += b));
  }
  const proba = calibrate(spec.calibration, raw, softmax(raw));
  return { proba, raw, contributions: contrib, features: x };
}

// ------------------------------------------------------------------ decision economics
export const expectedPD = (proba, cost) => proba.reduce((s, p, k) => s + p * cost.pd_by_tier[k], 0);

export function costOptimalTier(proba, cost) {
  let best = 0;
  let bestCost = Infinity;
  for (let j = 0; j < proba.length; j++) {
    let c = 0;
    for (let k = 0; k < proba.length; k++) c += proba[k] * cost.cost_matrix[k][j];
    if (c < bestCost - 1e-15) {
      bestCost = c;
      best = j;
    }
  }
  return best;
}

/** Portfolio statistics for one cutoff. Mirrors evaluation.approval_curve. */
export function portfolioAtCutoff(probas, y, exposure, cost, cutoff) {
  const n = y.length;
  let k = 0;
  let high = 0;
  let medHigh = 0;
  let highDeclined = 0;
  let highTotal = 0;
  let revenue = 0;
  let lossProxy = 0;
  let expLoss = 0;
  for (let i = 0; i < n; i++) {
    const pd = expectedPD(probas[i], cost);
    if (y[i] === 2) highTotal++;
    if (pd <= cutoff) {
      k++;
      if (y[i] === 2) high++;
      if (y[i] >= 1) medHigh++;
      revenue += exposure[i] * cost.merchant_fee;
      lossProxy += exposure[i] * cost.lgd * cost.pd_by_tier[y[i]];
      expLoss += exposure[i] * cost.lgd * pd;
    } else if (y[i] === 2) {
      highDeclined++;
    }
  }
  return {
    cutoff,
    approval_rate: k / n,
    high_rate_approved: k ? high / k : 0,
    medhigh_rate_approved: k ? medHigh / k : 0,
    high_captured: highTotal ? highDeclined / highTotal : 0,
    revenue_per_applicant: revenue / n,
    expected_loss_per_applicant: expLoss / n,
    loss_proxy_per_applicant: lossProxy / n,
    profit_per_applicant: (revenue - lossProxy) / n,
  };
}
