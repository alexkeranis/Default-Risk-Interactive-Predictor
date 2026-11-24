import { costOptimalTier, expectedPD, portfolioAtCutoff, predict } from "./model.js";

const REPO = "https://github.com/alexkeranis/Buy-Now-Pay-Later---Default-Risk-Assessment/blob/main/";
const TIERS = ["Low", "Medium", "High"];
const NAMES = { logreg: "Logistic regression", xgboost: "XGBoost", resnet: "Tabular ResNet" };
const SLOT = { logreg: "--series-1", xgboost: "--series-2", resnet: "--series-3" };

const NUM_FIELDS = [
  { key: "Age", label: "Age", step: 1, fmt: (v) => `${v}` },
  { key: "Income_USD", label: "Annual income", step: 500, fmt: (v) => usd(v) },
  { key: "Credit_Score", label: "Credit score", step: 1, fmt: (v) => `${v}` },
  { key: "Total_BNPL_Active_Loans", label: "Active BNPL plans", step: 1, fmt: (v) => `${v}` },
  { key: "Total_BNPL_Debt_USD", label: "Outstanding BNPL debt", step: 10, fmt: (v) => usd(v) },
  { key: "Average_Transaction_Value_USD", label: "This purchase", step: 10, fmt: (v) => usd(v) },
];
const CAT_FIELDS = [
  { key: "Employment_Status", label: "Employment" },
  { key: "Late_Payment_History", label: "Late payment history" },
  { key: "Shopping_Category_Most_Frequent", label: "Most frequent category" },
];
const PRESETS = {
  prime: { Age: 41, Income_USD: 85000, Credit_Score: 760, Total_BNPL_Active_Loans: 1, Total_BNPL_Debt_USD: 120,
    Average_Transaction_Value_USD: 250, Employment_Status: "Employed", Late_Payment_History: "No",
    Shopping_Category_Most_Frequent: "Home/Furniture" },
  stretched: { Age: 22, Income_USD: 14000, Credit_Score: 610, Total_BNPL_Active_Loans: 4, Total_BNPL_Debt_USD: 650,
    Average_Transaction_Value_USD: 380, Employment_Status: "Student", Late_Payment_History: "No",
    Shopping_Category_Most_Frequent: "Electronics" },
  delinquent: { Age: 29, Income_USD: 9000, Credit_Score: 540, Total_BNPL_Active_Loans: 6, Total_BNPL_Debt_USD: 1100,
    Average_Transaction_Value_USD: 520, Employment_Status: "Unemployed", Late_Payment_History: "Yes",
    Shopping_Category_Most_Frequent: "Fashion" },
};
const BINARY = new Set([
  "loan_stacking", "subprime", "near_prime", "unstable_income", "late_x_subprime", "young_borrower",
  "missing_Income_USD", "missing_Credit_Score",
]);
const USD_FEATURES = new Set(["Income_USD", "Total_BNPL_Debt_USD", "Average_Transaction_Value_USD", "debt_per_loan"]);

// ---------------------------------------------------------------- helpers
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const usd = (v, d = 0) => {
  const s = Math.abs(Number(v)).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  return `${Number(v) < 0 && Number(s.replace(/,/g, "")) !== 0 ? "−" : ""}$${s}`;
};
const pct = (v, d) => `${(v * 100).toFixed(d ?? (Math.abs(v) < 0.01 && v !== 0 ? 2 : 1))}%`;
const fx = (v, d = 3) => Number(v).toFixed(d);

function el(tag, props = {}, children = []) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "text") n.textContent = v;
    else if (k === "class") n.className = v;
    else if (k === "style") Object.assign(n.style, v);
    else n.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c != null) n.append(c);
  return n;
}

function luminance(hex) {
  const m = hex.replace("#", "").match(/.{2}/g);
  if (!m) return 1;
  const [r, g, b] = m.map((h) => {
    const c = parseInt(h, 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

// ---------------------------------------------------------------- state
let bundle, dash;
const state = { input: { ...PRESETS.stretched }, simModel: null, diagModel: null, cm: "confusion_argmax", cost: null };
const charts = {};

function defaultCost() {
  const c = bundle.cost_assumptions;
  return {
    pd_by_tier: [...c.pd_by_tier],
    lgd: c.lgd,
    merchant_fee: c.merchant_fee,
    cost_matrix: c.cost_matrix,
    cutoff: bundle.pd_cutoff[bundle.selected],
  };
}

// ---------------------------------------------------------------- charts
function chartDefaults() {
  if (!window.Chart) return false;
  const C = window.Chart;
  C.defaults.font.family = css("--font") || "system-ui";
  C.defaults.font.size = 12;
  C.defaults.color = css("--text-secondary");
  C.defaults.borderColor = css("--grid");
  C.defaults.animation = false;
  C.defaults.maintainAspectRatio = false;
  C.defaults.plugins.legend.display = false;
  Object.assign(C.defaults.plugins.tooltip, {
    backgroundColor: css("--surface-1"),
    titleColor: css("--text-secondary"),
    bodyColor: css("--text-primary"),
    borderColor: css("--border"),
    borderWidth: 1,
    padding: 8,
    displayColors: false,
    titleFont: { weight: "normal" },
    bodyFont: { weight: "600" },
  });
  return true;
}

function upsertChart(id, config) {
  if (!window.Chart) return;
  if (charts[id]) charts[id].destroy();
  charts[id] = new window.Chart($(id), config);
}

const axis = (extra = {}) => ({
  grid: { color: css("--grid"), drawTicks: false },
  border: { color: css("--axis") },
  ticks: { color: css("--text-secondary"), padding: 6 },
  ...extra,
});

// ---------------------------------------------------------------- simulator
function buildForm() {
  const box = $("fields");
  box.replaceChildren();
  const schema = bundle.input_schema;
  for (const f of NUM_FIELDS) {
    const s = schema.numeric[f.key];
    const out = el("output", { id: `o-${f.key}` });
    const input = el("input", {
      type: "range", id: `f-${f.key}`, min: s.train_min, max: s.train_max, step: f.step,
      "aria-label": f.label,
    });
    input.addEventListener("input", () => {
      state.input[f.key] = Number(input.value);
      out.textContent = f.fmt(input.value);
      renderSimulation();
    });
    box.append(el("div", { class: "field" }, [el("label", { for: input.id }, [el("span", { text: f.label }), out]), input]));
  }
  for (const f of CAT_FIELDS) {
    const sel = el("select", { id: `f-${f.key}` }, schema.categorical[f.key].map((v) => el("option", { value: v, text: v })));
    sel.addEventListener("change", () => {
      state.input[f.key] = sel.value;
      renderSimulation();
    });
    box.append(el("div", { class: "field" }, [el("label", { for: sel.id, text: f.label }), sel]));
  }
  syncForm();
}

function syncForm() {
  for (const f of NUM_FIELDS) {
    $(`f-${f.key}`).value = state.input[f.key];
    $(`o-${f.key}`).textContent = f.fmt(state.input[f.key]);
  }
  for (const f of CAT_FIELDS) $(`f-${f.key}`).value = state.input[f.key];
}

function scoreAll() {
  const out = {};
  for (const [fam, spec] of Object.entries(bundle.models)) out[fam] = predict(spec, state.input, bundle);
  return out;
}

function renderSimulation() {
  const results = scoreAll();
  const fam = state.simModel;
  const r = results[fam];
  const cost = state.cost;
  const pd = expectedPD(r.proba, cost);
  const approve = pd <= cost.cutoff;

  const dec = $("decision");
  dec.className = `decision ${approve ? "approve" : "decline"}`;
  dec.replaceChildren(
    el("span", { class: "icon", "aria-hidden": "true", text: approve ? "✓" : "✕" }),
    el("span", { text: approve ? "Approve" : "Decline" }),
  );
  $("hero-pd").textContent = pct(pd);
  $("hero-caption").textContent = `expected probability of default · approve if ≤ ${pct(cost.cutoff)}`;

  // Tall segmented bar of tier probabilities (ordinal ramp).
  const bar = $("segbar");
  bar.replaceChildren();
  bar.setAttribute("aria-label", TIERS.map((t, k) => `${t} ${pct(r.proba[k])}`).join(", "));
  r.proba.forEach((p, k) => {
    if (p <= 0) return;
    const seg = el("div", {
      title: `${TIERS[k]}: ${pct(p)}`,
      style: { flex: `${p} 1 0`, background: css(`--tier-${k}`), color: css(`--tier-ink-${k}`) },
    });
    if (p >= 0.14) seg.textContent = `${TIERS[k]} ${pct(p, 0)}`;
    bar.append(seg);
  });
  $("seg-legend").replaceChildren(
    ...TIERS.map((t, k) =>
      el("span", {}, [el("i", { class: "swatch", style: { background: css(`--tier-${k}`) } }), `${t} ${pct(r.proba[k])}`]),
    ),
  );

  const exposure = state.input.Average_Transaction_Value_USD;
  $("t-loss").textContent = usd(exposure * cost.lgd * pd, 2);
  $("t-rev").textContent = usd(exposure * cost.merchant_fee, 2);
  $("t-tier").textContent = TIERS[costOptimalTier(r.proba, cost)];

  // Every exported model, side by side.
  const tbl = $("compare");
  tbl.replaceChildren(
    el("tr", {}, ["Model", "P(Low)", "P(Medium)", "P(High)", "Expected PD", "Decision"].map((h, i) =>
      el("th", { class: i ? "num" : "", text: h }))),
    ...Object.entries(results).map(([f, res]) => {
      const p = expectedPD(res.proba, cost);
      return el("tr", { class: f === fam ? "selected" : "" }, [
        el("td", {}, el("span", { class: "key-cell" }, [
          el("i", { class: "line", style: { background: css(SLOT[f]) } }),
          `${NAMES[f]}${f === bundle.selected ? " (deployed)" : ""}`,
        ])),
        ...res.proba.map((v) => el("td", { class: "num", text: pct(v) })),
        el("td", { class: "num", text: pct(p) }),
        el("td", { class: "num", text: p <= cost.cutoff ? "Approve" : "Decline" }),
      ]);
    }),
  );
  renderDrivers(r, fam);
}

function renderDrivers(r, fam) {
  const spec = bundle.models[fam];
  const P = spec.features.length;
  const d = spec.features.map((name, j) => ({
    name,
    value: r.features[j],
    delta: r.contributions[2][j] - r.contributions[0][j],
  }));
  const top = d.filter((x) => Math.abs(x.delta) > 1e-9).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)).slice(0, 8);
  const narrow = $("drivers-chart").parentElement.clientWidth < 480; // value moves to the tooltip
  const labelOf = (x) => {
    const base = bundle.feature_labels[x.name] ?? x.name;
    if (BINARY.has(x.name) || base.includes(": ") || narrow) return base;
    const v = USD_FEATURES.has(x.name) ? usd(x.value) : Math.abs(x.value) < 1 ? fx(x.value, 3) : fx(x.value, x.value % 1 ? 2 : 0);
    return `${base} = ${v}`;
  };
  const bias = r.contributions[2][P] - r.contributions[0][P];
  $("drivers-caption").textContent =
    `Contribution to the log-odds of High vs Low tier (before calibration; baseline ${fx(bias, 2)}). ` +
    (spec.type === "linear" ? "Exact for logistic regression." : "Saabas path attributions for XGBoost.");
  upsertChart("drivers-chart", {
    type: "bar",
    data: {
      labels: top.map(labelOf),
      datasets: [{
        data: top.map((x) => x.delta),
        backgroundColor: top.map((x) => (x.delta > 0 ? css("--neg") : css("--pos"))),
        borderRadius: 4,
        borderSkipped: "start",
        barThickness: 16,
      }],
    },
    options: {
      indexAxis: "y",
      scales: {
        x: axis({ title: { display: true, text: "log-odds (High vs Low)", color: css("--muted") } }),
        y: axis({ grid: { display: false }, ticks: { color: css("--text-primary"), autoSkip: false } }),
      },
      plugins: {
        tooltip: {
          callbacks: {
            title: (items) => {
              const x = top[items[0].dataIndex];
              return BINARY.has(x.name) || (bundle.feature_labels[x.name] ?? "").includes(": ") ? labelOf(x) : `${bundle.feature_labels[x.name] ?? x.name} = ${fx(x.value, 3)}`;
            },
            label: (c) => `${c.raw > 0 ? "+" : ""}${fx(c.raw, 3)} log-odds`,
          },
        },
      },
    },
  });
}

// ---------------------------------------------------------------- assumptions
function bindAssumptions() {
  const ids = { "a-pd0": ["pd", 0], "a-pd1": ["pd", 1], "a-pd2": ["pd", 2], "a-lgd": ["lgd"], "a-fee": ["merchant_fee"], "a-cut": ["cutoff"] };
  for (const [id, [key, k]] of Object.entries(ids)) {
    $(id).addEventListener("input", (e) => {
      const v = Number(e.target.value);
      if (!Number.isFinite(v)) return;
      if (key === "pd") state.cost.pd_by_tier[k] = v;
      else state.cost[key] = v;
      onCostChange(id !== "a-cut");
    });
  }
  $("a-reset").addEventListener("click", () => {
    state.cost = defaultCost();
    syncAssumptions();
    onCostChange(true);
  });
}

function syncAssumptions() {
  const c = state.cost;
  $("a-pd0").value = c.pd_by_tier[0];
  $("a-pd1").value = c.pd_by_tier[1];
  $("a-pd2").value = c.pd_by_tier[2];
  $("a-lgd").value = c.lgd;
  $("a-fee").value = c.merchant_fee;
  $("a-cut").value = Number(c.cutoff.toFixed(4));
  $("a-breakeven").textContent = c.lgd > 0 ? pct(c.merchant_fee / c.lgd) : "n/a";
}

function onCostChange(recomputeCurve) {
  $("a-breakeven").textContent = state.cost.lgd > 0 ? pct(state.cost.merchant_fee / state.cost.lgd) : "n/a";
  renderSimulation();
  if (recomputeCurve) renderCostCurve();
  else renderCutoff();
}

// ---------------------------------------------------------------- diagnostics
function renderBenchmark() {
  const fams = ["logreg", "xgboost", "resnet"];
  const pm = (m, k, d = 3) => `${fx(m[k].mean, d)} ± ${fx(m[k].std, d)}`;
  const ci = (t, k, d = 3) => `${fx(t.calibrated[k], d)} [${fx(t.ci95[k][0], d)}, ${fx(t.ci95[k][1], d)}]`;
  const head = ["Model", "CV log loss", "CV Brier", "CV macro F1", "CV PR-AUC High", "Test macro F1 [95% CI]", "Test recall High [95% CI]", "Calibration"];
  $("bench").replaceChildren(
    el("tr", {}, head.map((h, i) => el("th", { class: i ? "num" : "", text: h }))),
    ...fams.map((f) => {
      const c = dash.cv[f];
      const t = dash.test[f];
      return el("tr", { class: f === dash.selected ? "selected" : "" }, [
        el("td", {}, el("span", { class: "key-cell" }, [
          el("i", { class: "line", style: { background: css(SLOT[f]) } }),
          NAMES[f],
          f === dash.selected ? el("span", { class: "tag", text: "deployed" }) : null,
        ])),
        el("td", { class: "num", text: pm(c, "log_loss", 4) }),
        el("td", { class: "num", text: pm(c, "brier", 4) }),
        el("td", { class: "num", text: pm(c, "macro_f1") }),
        el("td", { class: "num", text: pm(c, "pr_auc_High") }),
        el("td", { class: "num", text: ci(t, "macro_f1") }),
        el("td", { class: "num", text: ci(t, "recall_High") }),
        el("td", { class: "num", text: dash.calibration[f].chosen }),
      ]);
    }),
  );
  const e = dash.experiment;
  $("bench-caption").textContent = `${e.cv.scheme} on ${e.data.n_dev} development rows; test n = ${e.data.n_test}`;

  const legacyNum = dash.ablations["legacy_xgb/numeric_only"].macro_f1.mean;
  const legacyCat = dash.ablations["legacy_xgb/with_categoricals"].macro_f1.mean;
  $("legacy").replaceChildren(
    el("strong", { text: "The old 0.84 vs 0.75 claim, corrected. " }),
    `Re-running the original XGBoost configuration under this protocol gives macro F1 ${fx(legacyNum)} on numeric ` +
      `features only (what the old notebook fed it) and ${fx(legacyCat)} once the categorical features are included. ` +
      "Withholding two columns, not the neural network, produced the gap.",
  );

  const f1s = fams.map((f) => dash.cv[f].macro_f1.mean);
  const sig = dash.comparisons.filter((c) => c.metric === "macro_f1" && NAMES[c.a] && NAMES[c.b] && c.p < 0.05);
  $("diag-lede").textContent =
    `Cross-validated macro F1 ranges from ${fx(Math.min(...f1s))} to ${fx(Math.max(...f1s))} across the three families; ` +
    (sig.length ? `${sig.length} pairwise difference(s) are significant at 5% (corrected t-test). `
      : "no pairwise difference is significant at 5% (corrected t-test). ") +
    `${NAMES[dash.selected]} is deployed under the pre-registered rule: the simplest model within one standard error of the best log loss.`;
}

function renderReliability() {
  const fam = state.diagModel;
  const rel = dash.reliability_test[fam];
  const color = css(SLOT[fam]);
  $("rel-key").style.background = color;
  $("rel-caption").textContent = `${NAMES[fam]}, ${dash.calibration[fam].chosen} calibration; 10 equal-width bins, marker size ∝ bin count`;
  TIERS.forEach((tier, k) => {
    const pts = (stage) => rel[stage][tier].mean_pred.map((x, i) => ({ x, y: rel[stage][tier].frac_pos[i], n: rel[stage][tier].n[i] }));
    const radius = (stage) => {
      const ns = rel[stage][tier].n;
      const max = Math.max(...ns);
      return ns.map((n) => 3 + 5 * Math.sqrt(n / max));
    };
    upsertChart(`rel-${k}`, {
      type: "scatter",
      data: {
        datasets: [
          { data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], showLine: true, borderColor: css("--axis"), borderWidth: 1, pointRadius: 0, label: "perfect" },
          { data: pts("uncalibrated"), showLine: true, borderColor: css("--muted"), backgroundColor: css("--muted"), borderWidth: 2,
            pointRadius: radius("uncalibrated"), pointBorderColor: css("--surface-1"), pointBorderWidth: 2, label: "uncalibrated" },
          { data: pts("calibrated"), showLine: true, borderColor: color, backgroundColor: color, borderWidth: 2,
            pointRadius: radius("calibrated"), pointBorderColor: css("--surface-1"), pointBorderWidth: 2, pointHitRadius: 12, label: "calibrated" },
        ],
      },
      options: {
        scales: {
          x: axis({ min: 0, max: 1, ticks: { stepSize: 0.2, callback: (v) => Number(v).toFixed(1) } }),
          y: axis({ min: 0, max: 1, ticks: { stepSize: 0.2, callback: (v) => Number(v).toFixed(1) } }),
        },
        plugins: {
          tooltip: {
            filter: (c) => c.datasetIndex > 0,
            callbacks: {
              title: (items) => items[0].dataset.label,
              label: (c) => `observed ${pct(c.raw.y)} vs predicted ${pct(c.raw.x)} (n=${c.raw.n})`,
            },
          },
        },
      },
    });
  });
}

function renderConfusion() {
  const fam = state.diagModel;
  const d = dash.decisions[fam].test;
  const cm = d[state.cm];
  const steps = ["--seq-0", "--seq-1", "--seq-2", "--seq-3", "--seq-4"].map(css);
  const rows = cm.map((row) => {
    const s = row.reduce((a, b) => a + b, 0);
    return row.map((v) => ({ v, r: s ? v / s : 0 }));
  });
  $("cm").replaceChildren(
    el("tr", {}, [el("th", { class: "row", text: "Actual ↓ / Predicted →" }), ...TIERS.map((t) => el("th", { text: t }))]),
    ...rows.map((row, i) =>
      el("tr", {}, [
        el("th", { class: "row", text: TIERS[i] }),
        ...row.map(({ v, r }) => {
          const bg = steps[Math.min(4, Math.floor(r * 4.999))];
          const ink = luminance(bg) > 0.4 ? "#0b0b0b" : "#ffffff";
          return el("td", { style: { background: bg, color: ink }, title: `${v} applicants` }, [pct(r, 0), el("small", { text: `${v}` })]);
        }),
      ]),
    ),
  );
  const rule = state.cm === "confusion_argmax" ? "argmax tier" : "cost-optimal tier (minimises expected misclassification cost)";
  $("cm-caption").textContent =
    `${NAMES[fam]}, ${rule}; row-normalised. Expected misclassification cost per applicant: ` +
    `argmax ${fx(d.misclassification_cost_argmax)} vs cost-optimal ${fx(d.misclassification_cost_bayes)}.`;
}

function renderImportance() {
  const fam = state.diagModel;
  let rows;
  let caption;
  if (dash.importance[fam]) {
    rows = [...dash.importance[fam]].sort((a, b) => b.High - a.High).slice(0, 10).map((r) => ({ label: bundle.feature_labels[r.feature] ?? r.feature, v: r.High }));
    caption = fam === "xgboost" ? "Mean |TreeSHAP| for the High tier, test set (log-odds)." : "Mean |coefficient × standardised value| for the High tier, test set (log-odds).";
  } else {
    rows = dash.permutation_importance[fam].slice(0, 9).map((r) => ({ label: r.feature, v: r.importance }));
    caption = "Permutation importance: increase in test log loss when one raw input is shuffled (no exact attribution for the network).";
  }
  $("imp-caption").textContent = caption;
  upsertChart("imp-chart", {
    type: "bar",
    data: { labels: rows.map((r) => r.label), datasets: [{ data: rows.map((r) => r.v), backgroundColor: css(SLOT[fam]), borderRadius: 4, borderSkipped: "start", barThickness: 14 }] },
    options: {
      indexAxis: "y",
      scales: { x: axis({ beginAtZero: true }), y: axis({ grid: { display: false }, ticks: { color: css("--text-primary"), autoSkip: false } }) },
      plugins: { tooltip: { callbacks: { label: (c) => fx(c.raw, 3) } } },
    },
  });
}

// cost curve ------------------------------------------------------------
const sliderToCutoff = (t) => 0.5 * (t / 1000) ** 2;
const cutoffToSlider = (c) => Math.round(1000 * Math.sqrt(Math.min(c, 0.5) / 0.5));

function renderCostCurve() {
  const fam = state.diagModel;
  const tp = dash.test_predictions;
  const probas = tp[fam];
  const pds = probas.map((p) => expectedPD(p, state.cost)).sort((a, b) => a - b);
  const cuts = [0];
  for (let i = 0; i <= 150; i++) cuts.push(pds[Math.min(pds.length - 1, Math.floor((i / 150) * (pds.length - 1)))]);
  const pts = [...new Set(cuts)].map((c) => portfolioAtCutoff(probas, tp.y, tp.exposure, state.cost, c));
  const color = css(SLOT[fam]);
  const line = (key) => ({
    data: pts.map((p) => ({ x: p.approval_rate, y: p[key] })),
    showLine: true, borderColor: color, borderWidth: 2, pointRadius: 0, pointHitRadius: 8,
  });
  const marker = { data: [], pointRadius: 6, pointBackgroundColor: color, pointBorderColor: css("--surface-1"), pointBorderWidth: 2 };
  const opts = (yfmt) => ({
    scales: {
      x: axis({ min: 0.6, max: 1, ticks: { callback: (v) => pct(v, 0) }, title: { display: true, text: "approval rate", color: css("--muted") } }),
      y: axis({ ticks: { callback: yfmt } }),
    },
    plugins: { tooltip: { callbacks: { title: (i) => `approval ${pct(i[0].raw.x)}`, label: (c) => yfmt(c.raw.y) } } },
  });
  upsertChart("curve-bad", { type: "scatter", data: { datasets: [line("medhigh_rate_approved"), { ...marker }] }, options: opts((v) => pct(v, 1)) });
  upsertChart("curve-profit", { type: "scatter", data: { datasets: [line("profit_per_applicant"), { ...marker }] }, options: opts((v) => usd(v, 2)) });
  renderCutoff();
}

function renderCutoff() {
  const fam = state.diagModel;
  const tp = dash.test_predictions;
  const c = state.cost.cutoff;
  const p = portfolioAtCutoff(tp[fam], tp.y, tp.exposure, state.cost, c);
  $("cut-slider").value = cutoffToSlider(c);
  $("cut-value").textContent = pct(c, 2);
  $("c-appr").textContent = pct(p.approval_rate);
  $("c-bad").textContent = pct(p.medhigh_rate_approved, 2);
  $("c-cap").textContent = pct(p.high_captured);
  $("c-profit").textContent = usd(p.profit_per_applicant, 2);
  for (const [id, key] of [["curve-bad", "medhigh_rate_approved"], ["curve-profit", "profit_per_applicant"]]) {
    const ch = charts[id];
    if (!ch) continue;
    ch.data.datasets[1].data = [{ x: p.approval_rate, y: p[key] }];
    ch.update("none");
  }
}

// ---------------------------------------------------------------- wiring
function renderDiagnostics() {
  renderReliability();
  renderConfusion();
  renderImportance();
  renderCostCurve();
}

function renderAll() {
  chartDefaults();
  renderBenchmark();
  renderSimulation();
  renderDiagnostics();
}

function initTheme() {
  try {
    const saved = localStorage.getItem("theme");
    if (saved) document.documentElement.dataset.theme = saved;
  } catch {
    /* storage unavailable: follow the OS setting */
  }
  $("theme-toggle").addEventListener("click", () => {
    const current = document.documentElement.dataset.theme
      || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch { /* ignore */ }
    renderAll();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (!document.documentElement.dataset.theme) renderAll();
  });
}

async function main() {
  initTheme();
  try {
    [bundle, dash] = await Promise.all(["data/model.json", "data/dashboard.json"].map(async (u) => {
      const r = await fetch(u);
      if (!r.ok) throw new Error(`${u}: ${r.status}`);
      return r.json();
    }));
  } catch (err) {
    console.error(err);
    $("load-error").hidden = false;
    return;
  }
  $("audit-link").href = `${REPO}reports/AUDIT.md`;
  $("bench-link").href = `${REPO}reports/benchmark.md`;
  state.cost = defaultCost();
  state.simModel = bundle.selected;
  state.diagModel = dash.selected;

  const simSel = $("sim-model");
  simSel.replaceChildren(...Object.keys(bundle.models).map((f) => el("option", { value: f, text: `${NAMES[f]}${f === bundle.selected ? " (deployed)" : ""}` })));
  simSel.value = state.simModel;
  simSel.addEventListener("change", () => { state.simModel = simSel.value; renderSimulation(); });

  const diagSel = $("diag-model");
  diagSel.replaceChildren(...["logreg", "xgboost", "resnet"].map((f) => el("option", { value: f, text: NAMES[f] })));
  diagSel.value = state.diagModel;
  diagSel.addEventListener("change", () => { state.diagModel = diagSel.value; renderDiagnostics(); });

  for (const b of document.querySelectorAll("[data-preset]")) {
    b.addEventListener("click", () => {
      state.input = { ...PRESETS[b.dataset.preset] };
      syncForm();
      renderSimulation();
    });
  }
  for (const b of document.querySelectorAll("[data-cm]")) {
    b.addEventListener("click", () => {
      state.cm = b.dataset.cm;
      document.querySelectorAll("[data-cm]").forEach((x) => x.classList.toggle("on", x === b));
      renderConfusion();
    });
  }
  $("cut-slider").addEventListener("input", (e) => {
    state.cost.cutoff = sliderToCutoff(Number(e.target.value));
    $("a-cut").value = Number(state.cost.cutoff.toFixed(4));
    renderCutoff();
    renderSimulation();
  });

  buildForm();
  syncAssumptions();
  bindAssumptions();
  renderAll();
}

main();
