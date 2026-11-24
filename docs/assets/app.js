import { costOptimalTier, expectedPD, portfolioAtCutoff, predict } from "./model.js";

// On GitHub Pages (<owner>.github.io/<repo>/) the source link is derived from the URL, so renaming the
// repository needs no code change. The fallback is only used when the page is served locally.
const REPO = location.hostname.endsWith(".github.io")
  ? `https://github.com/${location.hostname.split(".")[0]}/${location.pathname.split("/")[1]}`
  : "https://github.com/alexkeranis/Default-Risk-Interactive-Predictor";
// The Pages workflow loads this module as app.js?v=<commit>; reuse that query on data fetches so the
// JSON always matches the deployed code. Empty when served locally, which is fine.
const VERSION = new URL(import.meta.url).search;
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
const LABELS = Object.fromEntries([...NUM_FIELDS, ...CAT_FIELDS].map((f) => [f.key, f.label]));
const FMT = Object.fromEntries(NUM_FIELDS.map((f) => [f.key, f.fmt]));
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
const PRESET_NAMES = { prime: "Prime", stretched: "Stretched student", delinquent: "Delinquent" };
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

/** Paint the filled part of a range track (CSS reads --fill). */
function setFill(input) {
  const min = Number(input.min);
  const max = Number(input.max);
  const v = Number(input.value);
  input.style.setProperty("--fill", `${max > min ? ((v - min) / (max - min)) * 100 : 0}%`);
}

// ---------------------------------------------------------------- state
let bundle, dash;
const state = { input: { ...PRESETS.stretched }, simModel: null, diagModel: null, cost: null };
const charts = {};
// "What changed" tracking: a gesture groups consecutive edits of the same control.
let last = null; // { pd, approve } of the previous render
let gesture = null; // { key, label, from, pd, approve, t }
let pendingChange = null; // { key, label, from, to } set by a control before re-rendering

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
    borderColor: css("--border-strong"),
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
      const from = state.input[f.key];
      state.input[f.key] = Number(input.value);
      out.textContent = f.fmt(input.value);
      setFill(input);
      pendingChange = { key: f.key, label: f.label, from: f.fmt(from), to: f.fmt(input.value) };
      renderSimulation();
    });
    box.append(el("div", { class: "field" }, [el("label", { for: input.id }, [el("span", { text: f.label }), out]), input]));
  }
  for (const f of CAT_FIELDS) {
    const sel = el("select", { id: `f-${f.key}` }, schema.categorical[f.key].map((v) => el("option", { value: v, text: v })));
    sel.addEventListener("change", () => {
      pendingChange = { key: f.key, label: f.label, from: state.input[f.key], to: sel.value };
      state.input[f.key] = sel.value;
      renderSimulation();
    });
    box.append(el("div", { class: "field" }, [el("label", { for: sel.id, text: f.label }), sel]));
  }
  syncForm();
}

function syncForm() {
  for (const f of NUM_FIELDS) {
    const input = $(`f-${f.key}`);
    input.value = state.input[f.key];
    setFill(input);
    $(`o-${f.key}`).textContent = f.fmt(state.input[f.key]);
  }
  for (const f of CAT_FIELDS) $(`f-${f.key}`).value = state.input[f.key];
}

function scoreAll() {
  const out = {};
  for (const [fam, spec] of Object.entries(bundle.models)) out[fam] = predict(spec, state.input, bundle);
  return out;
}

/** Live "what changed" readout: groups a slider drag into one before -> after statement. */
function renderChange(pd, approve) {
  const box = $("change");
  const now = performance.now();
  if (pendingChange) {
    const sameGesture = gesture && gesture.key === pendingChange.key && now - gesture.t < 1500;
    if (!sameGesture && last) {
      gesture = { key: pendingChange.key, label: pendingChange.label, from: pendingChange.from, pd: last.pd, approve: last.approve };
    }
    if (gesture) {
      gesture.t = now;
      const diff = (pd - gesture.pd) * 100;
      const flipped = approve !== gesture.approve;
      const arrow = Math.abs(diff) < 0.005 ? "unchanged" : `${diff > 0 ? "+" : "−"}${Math.abs(diff).toFixed(1)} pts`;
      const what = pendingChange.key === "preset" ? `Loaded the “${pendingChange.to}” applicant`
        : pendingChange.key === "reset" ? "Assumptions reset to defaults"
          : gesture.from === pendingChange.to ? `${gesture.label} back to ${pendingChange.to}`
            : `${gesture.label}: ${gesture.from} → ${pendingChange.to}`;
      box.className = `change ${flipped ? "flip" : diff > 0.005 ? "up" : diff < -0.005 ? "down" : ""}`;
      box.replaceChildren(
        el("strong", { text: what }),
        ` · expected PD ${pct(gesture.pd)} → ${pct(pd)} (${arrow})`,
        flipped ? el("strong", { text: ` · decision flips to ${approve ? "Approve" : "Decline"}` }) : " · decision unchanged",
      );
    }
    pendingChange = null;
  }
  last = { pd, approve };
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
  renderChange(pd, approve);

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
  $("compare").replaceChildren(
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
  return { pd, approve };
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
  const narrow = $("drivers-chart").parentElement.clientWidth < 480; // values move to the tooltip
  const isCategory = (x) => BINARY.has(x.name) || (bundle.feature_labels[x.name] ?? "").includes(": ");
  const valueOf = (x) => (USD_FEATURES.has(x.name) ? usd(x.value)
    : Math.abs(x.value) < 1 ? fx(x.value, 3) : fx(x.value, x.value % 1 ? 2 : 0));
  const nameOf = (x) => bundle.feature_labels[x.name] ?? x.name;
  const labelOf = (x) => (isCategory(x) || narrow ? nameOf(x) : `${nameOf(x)} = ${valueOf(x)}`);
  const bias = r.contributions[2][P] - r.contributions[0][P];
  $("drivers-caption").textContent =
    `Contribution of each factor to the log-odds of High vs Low tier (before calibration; baseline ${fx(bias, 2)}). ` +
    (spec.type === "linear" ? "Exact for logistic regression." : "Saabas path attributions for XGBoost.");

  const labels = top.map(labelOf);
  const values = top.map((x) => x.delta);
  const colors = top.map((x) => (x.delta > 0 ? css("--neg") : css("--pos")));
  const ch = charts["drivers-chart"];
  if (ch && ch.$family === fam && ch.$narrow === narrow) {
    // Update in place while dragging: no flicker, no re-layout.
    ch.data.labels = labels;
    ch.data.datasets[0].data = values;
    ch.data.datasets[0].backgroundColor = colors;
    ch.$top = top;
    ch.update("none");
    return;
  }
  upsertChart("drivers-chart", {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 4, borderSkipped: "start", barThickness: 16 }] },
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
              const x = charts["drivers-chart"].$top[items[0].dataIndex];
              return isCategory(x) ? nameOf(x) : `${nameOf(x)} = ${valueOf(x)}`;
            },
            label: (c) => `${c.raw > 0 ? "+" : ""}${fx(c.raw, 3)} log-odds`,
          },
        },
      },
    },
  });
  if (charts["drivers-chart"]) Object.assign(charts["drivers-chart"], { $family: fam, $narrow: narrow, $top: top });
}

// ---------------------------------------------------------------- "try this" prompts
function scrollToDecision() {
  const card = document.querySelector(".decision-card");
  const r = card.getBoundingClientRect();
  if (r.top < 60 || r.top > window.innerHeight * 0.6) card.scrollIntoView({ behavior: "smooth", block: "start" });
}

function setField(key, value) {
  pendingChange = {
    key, label: LABELS[key],
    from: FMT[key] ? FMT[key](state.input[key]) : state.input[key],
    to: FMT[key] ? FMT[key](value) : value,
  };
  state.input[key] = value;
  syncForm();
  return renderSimulation();
}

let animating = null;
function animateDebtUntilFlip() {
  clearTimeout(animating);
  const s = bundle.input_schema.numeric.Total_BNPL_Debt_USD;
  const startApprove = last ? last.approve : true;
  const dir = startApprove ? 1 : -1; // approved -> add debt until declined, and vice versa
  const step = () => {
    const next = Math.min(s.train_max, Math.max(s.train_min, state.input.Total_BNPL_Debt_USD + dir * 20));
    if (next === state.input.Total_BNPL_Debt_USD) {
      // Reached the edge of the training range without a flip: say so explicitly.
      animating = null;
      $("change").append(el("span", {
        text: ` · no flip anywhere in the training range: for this applicant, debt alone ${startApprove ? "can't trigger a decline" : "can't earn an approval"}.`,
      }));
      return;
    }
    const { approve } = setField("Total_BNPL_Debt_USD", next);
    animating = approve !== startApprove ? null : setTimeout(step, 22);
  };
  scrollToDecision();
  step();
}

function bindTries() {
  const actions = {
    debt: animateDebtUntilFlip,
    late: () => {
      scrollToDecision();
      setField("Late_Payment_History", state.input.Late_Payment_History === "Yes" ? "No" : "Yes");
    },
    income: () => {
      const max = bundle.input_schema.numeric.Income_USD.train_max;
      scrollToDecision();
      setField("Income_USD", Math.min(max, Math.round((state.input.Income_USD * 2) / 500) * 500));
    },
    model: () => {
      const next = state.simModel === "xgboost" ? bundle.selected : "xgboost";
      pendingChange = { key: "model", label: "Model", from: NAMES[state.simModel], to: NAMES[next] };
      state.simModel = next;
      $("sim-model").value = next;
      scrollToDecision();
      renderSimulation();
    },
  };
  for (const b of document.querySelectorAll("[data-try]")) b.addEventListener("click", () => actions[b.dataset.try]());
}

// ---------------------------------------------------------------- assumptions
function bindAssumptions() {
  const ids = {
    "a-pd0": ["pd", 0, "Assumed PD (Low)"], "a-pd1": ["pd", 1, "Assumed PD (Medium)"], "a-pd2": ["pd", 2, "Assumed PD (High)"],
    "a-lgd": ["lgd", null, "Loss given default"], "a-fee": ["merchant_fee", null, "Merchant fee"], "a-cut": ["cutoff", null, "Approval cutoff"],
  };
  for (const [id, [key, k, label]] of Object.entries(ids)) {
    $(id).addEventListener("input", (e) => {
      const v = Number(e.target.value);
      if (!Number.isFinite(v) || e.target.value === "") return;
      const before = key === "pd" ? state.cost.pd_by_tier[k] : state.cost[key];
      if (key === "pd") state.cost.pd_by_tier[k] = v;
      else state.cost[key] = v;
      pendingChange = { key: id, label, from: `${before}`, to: `${v}` };
      onCostChange(id !== "a-cut");
    });
  }
  $("a-reset").addEventListener("click", () => {
    state.cost = defaultCost();
    syncAssumptions();
    pendingChange = { key: "reset", label: "Assumptions", from: "edited", to: "defaults" };
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

// ---------------------------------------------------------------- masthead + diagnostics
function renderStats() {
  const sel = dash.selected;
  const t = dash.test[sel];
  const stat = (label, value, note) =>
    el("div", { class: "stat" }, [el("div", { class: "stat-label", text: label }), el("div", { class: "stat-value", text: value }), el("div", { class: "stat-note", text: note })]);
  const d = dash.decisions[sel].test;
  $("stat-strip").replaceChildren(
    stat("Deployed model", NAMES[sel], "simplest within 1 SE of best log loss"),
    stat("Test macro F1", fx(t.calibrated.macro_f1), `95% CI ${fx(t.ci95.macro_f1[0])}–${fx(t.ci95.macro_f1[1])}`),
    stat("High-risk recall", pct(t.calibrated.recall_High, 0), `precision ${pct(t.calibrated.precision_High, 0)}`),
    stat("Approval at trained cutoff", pct(d.approval_rate, 0), `${pct(d.high_rate_approved, 1)} High tier approved`),
  );
}

function renderBenchmark() {
  const fams = ["logreg", "xgboost", "resnet"];
  const pm = (m, k, d = 3) => `${fx(m[k].mean, d)} ± ${fx(m[k].std, d)}`;
  const ci = (t, k, d = 3) => `${fx(t.calibrated[k], d)} [${fx(t.ci95[k][0], d)}, ${fx(t.ci95[k][1], d)}]`;
  const head = ["Model", "CV log loss", "CV Brier", "CV macro F1", "CV PR-AUC High", "Test macro F1 [95% CI]", "Test recall High [95% CI]"];
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
      ]);
    }),
  );
  const e = dash.experiment;
  $("bench-caption").textContent = `${e.cv.scheme} on ${e.data.n_dev} development rows; test n = ${e.data.n_test}`;

  const a = dash.ablations;
  $("insight").replaceChildren(
    el("strong", { text: "Effect of feature engineering. " }),
    `Engineered ratios cut cross-validated log loss from ${fx(a["logreg/raw_features"].log_loss.mean)} to ` +
      `${fx(dash.cv.logreg.log_loss.mean)} for logistic regression (XGBoost: ${fx(a["xgboost/raw_features"].log_loss.mean)} → ` +
      `${fx(dash.cv.xgboost.log_loss.mean)}). Employment status and late-payment history are essential: without them a ` +
      `fixed-config XGBoost reaches macro F1 ${fx(a["categorical_ablation/numeric_only"].macro_f1.mean)}, versus ` +
      `${fx(a["categorical_ablation/all_inputs"].macro_f1.mean)} with them.`,
  );

  const f1s = fams.map((f) => dash.cv[f].macro_f1.mean);
  const sig = dash.comparisons.filter((c) => c.metric === "macro_f1" && NAMES[c.a] && NAMES[c.b] && c.p < 0.05);
  $("diag-lede").textContent =
    `Cross-validated macro F1 ranges from ${fx(Math.min(...f1s))} to ${fx(Math.max(...f1s))} across the three model families; ` +
    (sig.length ? `${sig.length} pairwise difference(s) are significant at 5% (corrected t-test). `
      : "no pairwise difference is significant at 5% (corrected t-test). ") +
    `${NAMES[dash.selected]} is deployed under a pre-registered rule: the simplest model within one standard error of the best log loss.`;
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
    caption = "Permutation importance: increase in test log loss when one raw input is shuffled.";
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
  const marker = { data: [], pointRadius: 7, pointBackgroundColor: css("--accent-amber"), pointBorderColor: css("--surface-1"), pointBorderWidth: 2 };
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
  const slider = $("cut-slider");
  slider.value = cutoffToSlider(c);
  setFill(slider);
  $("cut-value").textContent = pct(c, 2);
  const [p0, p1, p2] = state.cost.pd_by_tier.map((v) => pct(v, 0));
  $("cut-explain").textContent =
    "The cutoff is the highest expected probability of default (PD) the lender accepts. " +
    `Expected PD = P(Low) × ${p0} + P(Medium) × ${p1} + P(High) × ${p2}, using the tier default rates from the assumptions above. ` +
    `Applicants at or below the cutoff are approved and the rest are declined. The trained value, ${pct(bundle.pd_cutoff[bundle.selected], 2)}, ` +
    "maximises the profit proxy on development data.";
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
  renderImportance();
  renderCostCurve();
}

async function main() {
  try {
    [bundle, dash] = await Promise.all(["data/model.json", "data/dashboard.json"].map(async (u) => {
      const r = await fetch(`${u}${VERSION}`);
      if (!r.ok) throw new Error(`${u}: ${r.status}`);
      return r.json();
    }));
  } catch (err) {
    console.error(err);
    $("load-error").hidden = false;
    return;
  }
  $("bench-link").href = `${REPO}/blob/main/reports/benchmark.md`;
  $("repo-link").href = REPO;
  state.cost = defaultCost();
  state.simModel = bundle.selected;
  state.diagModel = dash.selected;

  const simSel = $("sim-model");
  simSel.replaceChildren(...Object.keys(bundle.models).map((f) => el("option", { value: f, text: `${NAMES[f]}${f === bundle.selected ? " (deployed)" : ""}` })));
  simSel.value = state.simModel;
  simSel.addEventListener("change", () => {
    pendingChange = { key: "model", label: "Model", from: NAMES[state.simModel], to: NAMES[simSel.value] };
    state.simModel = simSel.value;
    renderSimulation();
  });

  const diagSel = $("diag-model");
  diagSel.replaceChildren(...["logreg", "xgboost", "resnet"].map((f) => el("option", { value: f, text: NAMES[f] })));
  diagSel.value = state.diagModel;
  diagSel.addEventListener("change", () => { state.diagModel = diagSel.value; renderDiagnostics(); });

  for (const b of document.querySelectorAll("[data-preset]")) {
    b.addEventListener("click", () => {
      pendingChange = { key: "preset", label: "Applicant", from: "current", to: PRESET_NAMES[b.dataset.preset] };
      gesture = null;
      state.input = { ...PRESETS[b.dataset.preset] };
      syncForm();
      renderSimulation();
    });
  }
  $("cut-slider").addEventListener("input", (e) => {
    const before = state.cost.cutoff;
    state.cost.cutoff = sliderToCutoff(Number(e.target.value));
    $("a-cut").value = Number(state.cost.cutoff.toFixed(4));
    pendingChange = { key: "cutoff", label: "Approval cutoff", from: pct(before, 2), to: pct(state.cost.cutoff, 2) };
    renderCutoff();
    renderSimulation();
  });

  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(renderSimulation, 150); // re-fit driver labels to the new width
  });

  chartDefaults();
  buildForm();
  syncAssumptions();
  bindAssumptions();
  bindTries();
  renderStats();
  renderBenchmark();
  renderSimulation();
  renderDiagnostics();
}

main();
