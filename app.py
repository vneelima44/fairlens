"""
FairLens â Credit Decisioning Tradeoffs

v0.3 â adds subgroup breakdown (White / Black / Hispanic / Asian).

Data:   HMDA 2021 NY mortgage applications, ~95K test records.
Models: pre-trained probabilities from artifacts/test_probs.parquet.
CIs:    pre-computed at default settings from artifacts/bootstrap_cis.json.
Sub:    test-aligned race/ethnicity from artifacts/race_eth_test.parquet.

Sections (in order):
  1. Key Finding callout (binary Fairness Paradox)
  2. All-models comparison table at current settings (binary, with CIs)
  3. Detail view for selected model (binary)
  4. Subgroup breakdown (4-group, model-specific)   â NEW in v0.3
  5. Multi-model gap-vs-threshold curve (binary)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score
# ── Default parameter values (single source of truth) ─────────────────────────
DEFAULT_THRESHOLD = 0.5
DEFAULT_LGD       = 0.40
DEFAULT_DISCOUNT  = 0.030


# ââ Page config ââââââââââââââââââââââââââââââââââââââââââââââââââââ
st.set_page_config(
    page_title="FairLens â Credit Decisioning Tradeoffs",
    page_icon="ð",
    layout="wide",
)


# ââ Data loading âââââââââââââââââââââââââââââââââââââââââââââââââââ
@st.cache_data
def load_data():
    """Load model probabilities and test metadata from artifact files."""
    try:
        probs = pd.read_parquet("artifacts/test_probs.parquet")
        meta  = pd.read_parquet("artifacts/test_meta.parquet")
        cost  = pd.read_parquet("artifacts/cost_meta.parquet")
        return probs, meta, cost
    except FileNotFoundError as e:
        st.error(f"Missing artifact: {e}. Re-run the modeling notebook to regenerate.")
        st.stop()


@st.cache_data
def load_cis():
    """Load pre-computed 95% bootstrap CIs, or None if unavailable."""
    path = Path("artifacts/bootstrap_cis.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_subgroups():
    """Load test-aligned race/ethnicity labels for 4-group analysis."""
    path = Path("artifacts/race_eth_test.parquet")
    if not path.exists():
        return None
    return pd.read_parquet(path)['race_ethnicity'].values


probs, meta, cost = load_data()
cis = load_cis()
subgroups = load_subgroups()

y_true_arr = meta["y_true"].values.astype(int)
race_arr   = meta["race_binary"].values.astype(int)
loans_arr  = cost["loan_amount"].values.astype(float)
rates_arr  = cost["interest_rate"].values.astype(float)
terms_arr  = cost["loan_term_years"].values.astype(float)


# ââ Vectorized cost framework ââââââââââââââââââââââââââââââââââââââ
def amortized_interest_npv_vec(loans, rates, terms, discount_rate):
    r = rates / 100.0 / 12.0
    d = discount_rate / 12.0
    n = (terms * 12).astype(int)
    valid = (r > 0) & (n > 0)
    npv = np.zeros_like(loans, dtype=float)

    one_plus_r_n = np.zeros_like(loans, dtype=float)
    one_plus_r_n[valid] = (1 + r[valid]) ** n[valid]
    mp = np.zeros_like(loans, dtype=float)
    mp[valid] = (
        loans[valid] * (r[valid] * one_plus_r_n[valid])
        / (one_plus_r_n[valid] - 1)
    )
    npv[valid] = (
        mp[valid] * (1 - (1 + d) ** (-n[valid])) / d
        - loans[valid]
    )
    return npv


def remaining_balance_vec(loans, rates, terms, default_years):
    r = rates / 100.0 / 12.0
    n = (terms * 12).astype(int)
    k = (default_years * 12).astype(int)
    valid = (r > 0) & (n > 0)
    rb = loans.copy().astype(float)

    one_plus_r_n = np.zeros_like(loans, dtype=float)
    one_plus_r_n[valid] = (1 + r[valid]) ** n[valid]
    mp = np.zeros_like(loans, dtype=float)
    mp[valid] = (
        loans[valid] * (r[valid] * one_plus_r_n[valid])
        / (one_plus_r_n[valid] - 1)
    )

    one_plus_r_k = np.zeros_like(loans, dtype=float)
    one_plus_r_k[valid] = (1 + r[valid]) ** k[valid]
    rb[valid] = np.maximum(
        loans[valid] * one_plus_r_k[valid]
        - mp[valid] * (one_plus_r_k[valid] - 1) / r[valid],
        0.0,
    )
    return rb


@st.cache_data
def compute_metrics(probs_array, threshold, lgd, discount_rate):
    preds = (probs_array >= threshold).astype(int)
    results = {}

    for group, label in [(1, "White"), (0, "NonWhite")]:
        mask = race_arr == group
        yt = y_true_arr[mask]
        yp = preds[mask]
        l  = loans_arr[mask]
        r_ = rates_arr[mask]
        t_ = terms_arr[mask]

        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        n  = int(mask.sum())

        fn_mask = (yt == 1) & (yp == 0)
        fp_mask = (yt == 0) & (yp == 1)

        fn_cost = float(amortized_interest_npv_vec(
            l[fn_mask], r_[fn_mask], t_[fn_mask], discount_rate
        ).sum()) if fn_mask.any() else 0.0

        fp_cost = float((remaining_balance_vec(
            l[fp_mask], r_[fp_mask], t_[fp_mask], t_[fp_mask] / 2.0
        ) * lgd).sum()) if fp_mask.any() else 0.0

        results[label] = {
            "n":             n,
            "approval_rate": float(yp.mean()),
            "tpr":           float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "fpr":           float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "fn_count":      fn,
            "fp_count":      fp,
            "fn_cost":       fn_cost,
            "fp_cost":       fp_cost,
            "total_cost":    fn_cost + fp_cost,
            "per_applicant": (fn_cost + fp_cost) / n if n > 0 else 0.0,
        }

    di_denom = max(results["White"]["approval_rate"], 1e-9)
    di       = results["NonWhite"]["approval_rate"] / di_denom
    spd      = results["NonWhite"]["approval_rate"] - results["White"]["approval_rate"]
    eod      = results["NonWhite"]["tpr"] - results["White"]["tpr"]
    fpr_diff = results["NonWhite"]["fpr"] - results["White"]["fpr"]
    gap      = results["NonWhite"]["per_applicant"] - results["White"]["per_applicant"]

    return {
        "White":    results["White"],
        "NonWhite": results["NonWhite"],
        "di":       di,
        "spd":      spd,
        "eod":      eod,
        "fpr_diff": fpr_diff,
        "gap":      gap,
    }


@st.cache_data
def compute_subgroup_metrics(probs_array, threshold, lgd, discount_rate, _subgroups_tuple):
    """4-group analysis (White / Black / Hispanic / Asian). Excludes Other/Unknown."""
    preds = (probs_array >= threshold).astype(int)
    subgroups_arr = np.array(_subgroups_tuple)
    groups = ["White", "Black or African American", "Hispanic or Latino", "Asian"]

    results = {}
    for grp in groups:
        mask = subgroups_arr == grp
        if mask.sum() < 50:
            continue

        yt = y_true_arr[mask]
        yp = preds[mask]
        l  = loans_arr[mask]
        r_ = rates_arr[mask]
        t_ = terms_arr[mask]

        fn_mask = (yt == 1) & (yp == 0)
        fp_mask = (yt == 0) & (yp == 1)

        fn_cost = float(amortized_interest_npv_vec(
            l[fn_mask], r_[fn_mask], t_[fn_mask], discount_rate
        ).sum()) if fn_mask.any() else 0.0

        fp_cost = float((remaining_balance_vec(
            l[fp_mask], r_[fp_mask], t_[fp_mask], t_[fp_mask] / 2.0
        ) * lgd).sum()) if fp_mask.any() else 0.0

        n = int(mask.sum())
        avg_denied_loan = float(l[fn_mask].mean()) if fn_mask.any() else 0.0
        results[grp] = {
            "n":               n,
            "approval_rate":   float(yp.mean()),
            "fn_count":        int(fn_mask.sum()),
            "fn_rate":         float(fn_mask.sum() / n) if n > 0 else 0.0,
            "avg_denied_loan": avg_denied_loan,
            "per_applicant":   (fn_cost + fp_cost) / n if n > 0 else 0.0,
        }

    return results


# ââ Header âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
st.title("FairLens")
st.markdown(
    "**Translating fairness metrics into the dollar units that "
    "threshold decisions are actually made in.**"
)
st.caption(
    "HMDA 2021 NY mortgage applications â¢ 95,588 test records â¢ interactive demo"
)

# ââ Methodology Caveats (proactive disclosure) âââââââââââââââââââââ
with st.expander("âï¸ Methodology caveats & known limitations (click to expand)", expanded=False):
    st.markdown(
        """
This dashboard is a **policy-translation prototype**, not a production credit-decisioning tool.
Surfacing the limitations upfront because regulators, compliance teams, and risk teams
should know what these numbers do â and don't â represent.

- **LGD = 0.40 is a placeholder.** Production calibration would use the bank's
  internal recovery data (typical prime-mortgage range: 35â45%). The "$M of disparate cost"
  headline scales with LGD; treat aggregate dollar figures as illustrative magnitudes,
  not point estimates. Sensitivity analysis across LGD scenarios is in the methodology paper â
  model rankings are stable in 3 of 4 scenarios tested.
- **FN cost = NPV of foregone interest is an upper bound.** It does not account for
  alternative credit access (denied applicants often obtain loans elsewhere) or the
  counterfactual default probability of the denied loan. Realized harm is lower than
  this maximum-theoretical figure. The framework choice is explicit and conservative
  in the regulator's direction.
- **The aggregate $M figure is a test-set projection, not annual real-world impact.**
  Computed as per-applicant gap Ã test-set size (95,588). It indicates the magnitude
  of disparate cost that rate-parity metrics miss, not a realized loss number.
- **LR Baseline (94.7% overall approval) is not target leakage â it's class imbalance.**
  AUC = 0.76 (in line with other linear models). The high overall approval rate reflects
  a model that approves ~98% of the majority group and only ~23% of the minority group.
  DI = 0.979 because DI is a *ratio* â it stays near 1 even when both rates approach
  their respective ceilings. This is precisely the rate-parity blind spot the dollar
  translation surfaces: a model can pass ECOA's 80% rule while declining 77% of the
  minority group, and the dollar gap (\\$4,711/applicant) makes that visible.
- **Three models shown; ten models analyzed.** The dashboard surfaces three (LR Baseline,
  LR Manual RW, NN + Focal Loss) chosen to span the policy tradeoff space. Full benchmark
  (10 models Ã multiple imputation strategies) is in the methodology paper.
- **Four fairness metrics shown** (Disparate Impact, Statistical Parity Diff, Equalized Odds
  Diff, FPR Difference) represent the standard ECOA + Separation toolkit. Calibration
  parity, predictive parity, and counterfactual fairness are addressed in the methodology
  paper. The core finding â that dollar translation reveals rate-parity blind spots â
  doesn't require additional rate-based metrics to demonstrate.
- **Bootstrap inference is on the test set, conditional on the trained models.**
  CIs reflect sampling variability in the held-out evaluation, not training-set
  uncertainty or cross-validation variance.
        """
    )


# ââ Sidebar controls âââââââââââââââââââââââââââââââââââââââââââââââ
st.sidebar.header("Controls")

model_choice = st.sidebar.selectbox(
    "Model (for detail views below)",
    options=list(probs.columns),
    index=len(probs.columns) - 1,
)

threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.0, max_value=1.0, value=DEFAULT_THRESHOLD, step=0.01,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Economic assumptions**")

lgd = st.sidebar.slider(
    "Loss Given Default (LGD)",
    min_value=0.10, max_value=0.90, value=DEFAULT_LGD, step=0.05,
)

discount = st.sidebar.slider(
    "Discount rate",
    min_value=0.01, max_value=0.10, value=DEFAULT_DISCOUNT, step=0.005,
    format="%.3f",
)

at_defaults = (
    abs(threshold - DEFAULT_THRESHOLD) < 0.005
    and abs(lgd - DEFAULT_LGD) < 0.005
    and abs(discount - DEFAULT_DISCOUNT) < 0.001
)


# ââ Compute binary metrics for all models âââââââââââââââââââââââââ
all_results = {
    name: compute_metrics(probs[name].values, threshold, lgd, discount)
    for name in probs.columns
}

highest_di_model   = max(all_results, key=lambda k: all_results[k]["di"])
smallest_gap_model = min(all_results, key=lambda k: abs(all_results[k]["gap"]))

highest_di     = all_results[highest_di_model]["di"]
highest_di_gap = all_results[highest_di_model]["gap"]
smallest_gap   = all_results[smallest_gap_model]["gap"]
smallest_di    = all_results[smallest_gap_model]["di"]

total_n = (
    all_results[smallest_gap_model]["White"]["n"]
    + all_results[smallest_gap_model]["NonWhite"]["n"]
)


def fmt_with_ci(point, metric_name, model_name, kind="number"):
    """Format a metric with CI brackets if available at default settings."""
    if not at_defaults or cis is None:
        return (f"${point:+,.0f}" if kind == "money" else f"{point:.3f}")
    model_cis = cis.get("models", {}).get(model_name, {})
    metric    = model_cis.get(metric_name)
    if metric is None:
        return (f"${point:+,.0f}" if kind == "money" else f"{point:.3f}")
    lo, hi = metric["ci_lower"], metric["ci_upper"]
    if kind == "money":
        return f"${point:+,.0f} [${lo:+,.0f}, ${hi:+,.0f}]"
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


# ââ Key Finding callout âââââââââââââââââââââââââââââââââââââââââââ
if highest_di_model != smallest_gap_model:
    cost_diff_per_app = abs(highest_di_gap) - abs(smallest_gap)
    cost_diff_total_M = cost_diff_per_app * total_n / 1e6

    p_str = ""
    if at_defaults and cis is not None:
        pt = cis.get("paradox_test", {})
        if (
            "p_value_one_sided" in pt
            and pt.get("highest_di_model") == highest_di_model
            and pt.get("smallest_gap_model") == smallest_gap_model
        ):
            p = pt["p_value_one_sided"]
            n_holds = pt["n_iterations_paradox_holds"]
            n_iter = cis["metadata"]["n_iterations"]
            if p < 1 / n_iter:
                p_str = (f" Paradox holds in **{n_holds}/{n_iter} bootstrap samples** "
                         f"(one-sided p < {1/n_iter:.4f}).")
            else:
                p_str = (f" Paradox holds in **{n_holds}/{n_iter} bootstrap samples** "
                         f"(one-sided p = {p:.4f}).")

    st.info(
        f"ð **The Fairness Paradox.** At your current settings, "
        f"**{highest_di_model}** has the highest Disparate Impact "
        f"({highest_di:.3f}) â the model a compliance team would pick as 'most fair'. "
        f"But **{smallest_gap_model}** (DI {smallest_di:.3f}) has the "
        f"**smallest per-applicant racial cost gap** "
        f"(\\${smallest_gap:+,.0f} vs \\${highest_di_gap:+,.0f}). "
        f"Picking by DI alone leaves **\\${cost_diff_per_app:,.0f} per applicant** "
        f"of disparate cost on the table â **\\${cost_diff_total_M:,.0f}M** projected "
        f"across the {total_n:,} applicants in this test set "
        f"*(illustrative scaling, not realized annual impact â see methodology caveats)*."
        + p_str
    )
else:
    st.success(
        f"â At current settings, **{highest_di_model}** is best on both DI "
        f"and dollar-cost gap â no paradox here. Try moving sliders to see where they diverge."
    )


# ââ Model comparison table âââââââââââââââââââââââââââââââââââââââââ
st.subheader("All models at current settings")
caption_text = "Sorted by absolute racial gap (smallest first)."
if at_defaults and cis is not None:
    caption_text += (f"  CIs from {cis['metadata']['n_iterations']} bootstrap "
                     f"iterations at default settings.")
else:
    caption_text += ("  (Move sliders to defaults â threshold 0.50, "
                     "LGD 0.40, discount 0.030 â to see 95% confidence intervals.)")
st.caption(caption_text)

# Per-model AUC (threshold-independent â compute once, cache)
@st.cache_data
def compute_all_aucs(_probs_columns):
    """Compute AUC for each model. Cached by column tuple."""
    return {name: float(roc_auc_score(y_true_arr, probs[name].values))
            for name in _probs_columns}

all_aucs = compute_all_aucs(tuple(probs.columns))

comp_rows = []
for name, r in sorted(all_results.items(), key=lambda x: abs(x[1]["gap"])):
    n_total = r["White"]["n"] + r["NonWhite"]["n"]
    approval = (
        r["White"]["approval_rate"] * r["White"]["n"]
        + r["NonWhite"]["approval_rate"] * r["NonWhite"]["n"]
    ) / n_total
    annotation = ""
    if name == smallest_gap_model:
        annotation = " â smallest gap"
    elif name == highest_di_model and highest_di_model != smallest_gap_model:
        annotation = " â highest DI"
    comp_rows.append({
        "Model":             name + annotation,
        "AUC":               f"{all_aucs[name]:.3f}",
        "Disparate Impact":  fmt_with_ci(r["di"], "di", name),
        "Per-applicant gap": fmt_with_ci(r["gap"], "gap", name, kind="money"),
        "Approval rate":     f"{approval:.1%}",
        "ECOA (DIâ¥0.8)":     "â Pass" if r["di"] >= 0.8 else "â Fail",
    })

st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)
st.caption(
    "AUC reflects model discrimination across all thresholds. Approval rate at current "
    "threshold setting. Higher AUC + smaller gap = better on both performance and fairness."
)
st.divider()


# ââ Detail view: hero metrics for selected model âââââââââââââââââââ
st.subheader(f"Detail view: {model_choice}")
st.caption("Change model in the sidebar to compare. Numbers below reflect the selected model.")

m = all_results[model_choice]
total_n_sel = m["White"]["n"] + m["NonWhite"]["n"]
avg_approval = (
    m["White"]["approval_rate"] * m["White"]["n"]
    + m["NonWhite"]["approval_rate"] * m["NonWhite"]["n"]
) / total_n_sel


def metric_delta(metric_name, kind="number"):
    """Return 95% CI string for st.metric delta, or None if unavailable."""
    if not at_defaults or cis is None:
        return None
    model_cis = cis.get("models", {}).get(model_choice, {})
    metric = model_cis.get(metric_name)
    if metric is None:
        return None
    lo, hi = metric["ci_lower"], metric["ci_upper"]
    if kind == "money":
        return f"95% CI: [${lo:+,.0f}, ${hi:+,.0f}]"
    return f"95% CI: [{lo:.3f}, {hi:.3f}]"


col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Per-applicant racial gap", f"${m['gap']:,.0f}",
              delta=metric_delta("gap", kind="money"), delta_color="off")
with col2:
    di_delta = metric_delta("di")
    st.metric("Disparate Impact", f"{m['di']:.3f}",
              delta=di_delta if di_delta else ("ECOA pass" if m["di"] >= 0.8 else "ECOA fail"),
              delta_color="off" if di_delta else ("normal" if m["di"] >= 0.8 else "inverse"))
with col3:
    st.metric("Overall approval rate", f"{avg_approval:.1%}")
with col4:
    st.metric("Equalized Odds Diff", f"{m['eod']:+.3f}",
              delta=metric_delta("eod"), delta_color="off")

st.divider()


# ââ Bar chart + fairness table âââââââââââââââââââââââââââââââââââââ
left, right = st.columns([2, 1])
with left:
    st.subheader("Per-applicant cost by group (binary)")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["White", "NonWhite"],
        y=[m["White"]["per_applicant"], m["NonWhite"]["per_applicant"]],
        marker_color=["#4A90E2", "#D97757"],
        text=[f"${m['White']['per_applicant']:,.0f}",
              f"${m['NonWhite']['per_applicant']:,.0f}"],
        textposition="outside", textfont=dict(size=14),
    ))
    fig.update_layout(yaxis_title="$ per applicant", showlegend=False,
                      height=400, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Fairness metrics")
    metrics_df = pd.DataFrame({
        "Metric": ["Disparate Impact", "Statistical Parity Diff",
                   "Equalized Odds Diff", "FPR Difference"],
        "Value":  [fmt_with_ci(m["di"], "di", model_choice),
                   fmt_with_ci(m["spd"], "spd", model_choice),
                   fmt_with_ci(m["eod"], "eod", model_choice),
                   f"{m['fpr_diff']:+.3f}"],
        "Status": ["â Pass" if m["di"] >= 0.8 else "â Fail",
                   "â" if abs(m["spd"]) < 0.05 else "â ï¸",
                   "â" if abs(m["eod"]) < 0.05 else "â ï¸",
                   "â" if abs(m["fpr_diff"]) < 0.05 else "â ï¸"],
    })
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)


# ââ SUBGROUP BREAKDOWN âââââââââââââââââââââââââââââââââââââââââââââ
if subgroups is not None:
    st.divider()
    st.subheader("Subgroup breakdown â beyond the binary")
    st.caption(
        "Same model and settings as above, broken into 4 race/ethnicity groups. "
        "Reveals patterns the binary White/NonWhite framing hides."
    )

    sub_results = compute_subgroup_metrics(
        probs[model_choice].values, threshold, lgd, discount, tuple(subgroups)
    )

    if "White" in sub_results:
        white = sub_results["White"]

        # Build the table
        sub_rows = []
        for grp in ["White", "Black or African American", "Hispanic or Latino", "Asian"]:
            if grp not in sub_results:
                continue
            r = sub_results[grp]
            if grp == "White":
                di_str = "(ref)"
                gap_str = "(baseline)"
            else:
                di     = r["approval_rate"] / max(white["approval_rate"], 1e-9)
                gap    = r["per_applicant"] - white["per_applicant"]
                di_flag = " â ï¸" if di < 0.8 else (" ð©" if di > 1.0 and gap > 0 else "")
                di_str  = f"{di:.3f}{di_flag}"
                gap_str = f"${gap:+,.0f}"

            sub_rows.append({
                "Group":          grp.replace("Black or African American", "Black")
                                       .replace("Hispanic or Latino", "Hispanic"),
                "N":              f"{r['n']:,}",
                "Approval rate":  f"{r['approval_rate']:.1%}",
                "FN rate":        f"{r['fn_rate']:.1%}",
                "Avg denied loan": f"${r['avg_denied_loan']:,.0f}",
                "Per-applicant cost": f"${r['per_applicant']:,.0f}",
                "DI vs White":    di_str,
                "Gap vs White":   gap_str,
            })

        st.dataframe(pd.DataFrame(sub_rows), hide_index=True, use_container_width=True)
        st.caption("â ï¸ = DI < 0.8 (ECOA fail). ð© = DI > 1.0 (passes ECOA) but dollar gap > 0 (subgroup-level paradox).")

        # Subgroup-level paradox callout
        flagged_groups = []
        for grp in ["Black or African American", "Hispanic or Latino", "Asian"]:
            if grp not in sub_results:
                continue
            r = sub_results[grp]
            di  = r["approval_rate"] / max(white["approval_rate"], 1e-9)
            gap = r["per_applicant"] - white["per_applicant"]
            if di > 1.0 and gap > 0:
                flagged_groups.append((grp, di, gap, r["avg_denied_loan"]))

        if flagged_groups:
            grp, di, gap, avg_loan = flagged_groups[0]
            grp_short = grp.replace("Black or African American", "Black").replace("Hispanic or Latino", "Hispanic")
            loan_amp = (avg_loan / white["avg_denied_loan"] - 1) * 100 if white["avg_denied_loan"] > 0 else 0
            st.warning(
                f"ð© **Subgroup-level paradox.** **{grp_short}** applicants have "
                f"DI = {di:.3f} â *above the parity line*, so a compliance team using "
                f"ECOA's 80% rule would consider this group 'more than fair'. "
                f"But the per-applicant dollar gap is **+\\${gap:,.0f}** vs White. "
                f"Why? {grp_short} wrongful-denial loans average **\\${avg_loan:,.0f}** "
                f"â **{loan_amp:+.0f}% vs White's \\${white['avg_denied_loan']:,.0f}**. "
                f"Rate-parity metrics don't see loan size. This is the Fairness Paradox at "
                f"the subgroup level."
            )

        # Loan-size amplification callout (always shown, model-agnostic insight)
        st.info(
            "ð¡ **Loan-size amplification.** A group can have a *lower* wrongful-denial rate "
            "than another and still bear a *larger* per-applicant dollar cost â if their "
            "wrongful-denied loans are bigger. NPV of foregone interest scales with loan size, "
            "not with denial rate. This invisible-to-DI effect is robust across all 3 models in "
            "this demo and is documented in the methodology writeup."
        )
else:
    st.divider()
    st.caption(
        "*Subgroup breakdown unavailable â `artifacts/race_eth_test.parquet` not found. "
        "Run the race analysis notebook to generate it.*"
    )


# ââ Detailed breakdown by binary group âââââââââââââââââââââââââââââ
st.divider()
st.subheader("Detailed breakdown by group (binary)")
breakdown_df = pd.DataFrame({
    "Group":                 ["White", "NonWhite"],
    "N applicants":          [m["White"]["n"], m["NonWhite"]["n"]],
    "Approval rate":         [f"{m['White']['approval_rate']:.1%}",
                              f"{m['NonWhite']['approval_rate']:.1%}"],
    "Wrongful denials (FN)": [m["White"]["fn_count"], m["NonWhite"]["fn_count"]],
    "Missed defaults (FP)":  [m["White"]["fp_count"], m["NonWhite"]["fp_count"]],
    "FN cost ($M)":          [f"${m['White']['fn_cost']/1e6:,.2f}",
                              f"${m['NonWhite']['fn_cost']/1e6:,.2f}"],
    "FP cost ($M)":          [f"${m['White']['fp_cost']/1e6:,.2f}",
                              f"${m['NonWhite']['fp_cost']/1e6:,.2f}"],
    "Per applicant ($)":     [f"${m['White']['per_applicant']:,.2f}",
                              f"${m['NonWhite']['per_applicant']:,.2f}"],
})
st.dataframe(breakdown_df, hide_index=True, use_container_width=True)


# ââ Multi-model gap-vs-threshold curve âââââââââââââââââââââââââââââ
st.divider()
st.subheader("Gap vs. threshold â all models compared (binary)")
st.caption(
    "Each line is a model. Lower = smaller racial gap. Dotted vertical line marks "
    "your current threshold."
)


@st.cache_data
def compute_curve_all_models(_probs_columns, lgd_val, discount_val):
    thresholds = np.linspace(0.05, 0.95, 30)
    curves = {}
    for name in _probs_columns:
        probs_arr = probs[name].values
        gaps = [compute_metrics(probs_arr, float(t), lgd_val, discount_val)["gap"]
                for t in thresholds]
        curves[name] = np.array(gaps)
    return thresholds, curves


thresholds_arr, curves = compute_curve_all_models(tuple(probs.columns), lgd, discount)

colors = {
    "LR Baseline":     "#4A90E2",
    "LR Manual RW":    "#7B61FF",
    "NN + Focal Loss": "#D97757",
}

fig = go.Figure()
for name, gaps in curves.items():
    is_selected = (name == model_choice)
    fig.add_trace(go.Scatter(
        x=thresholds_arr, y=gaps, name=name,
        line=dict(color=colors.get(name, "#888"),
                  width=3 if is_selected else 2,
                  dash="solid" if is_selected else "dash"),
        mode="lines",
        opacity=1.0 if is_selected else 0.7,
    ))

fig.add_vline(x=threshold, line_dash="dot", line_color="gray",
              annotation_text=f"Current: {threshold:.2f}",
              annotation_position="bottom right")
fig.add_hline(y=0, line_dash="dot", line_color="black")
fig.update_xaxes(title_text="Decision threshold")
fig.update_yaxes(title_text="Racial gap ($, NonWhite â White)")
fig.update_layout(height=450, hovermode="x unified",
                  margin=dict(t=30, b=30),
                  legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
st.plotly_chart(fig, use_container_width=True)


# ââ SENSITIVITY ANALYSIS â model rankings under different economic scenarios â
st.divider()
st.subheader("Sensitivity analysis â does the paradox survive different economic assumptions?")
st.caption(
    "Per-applicant racial gap recomputed under 3 economic scenarios. "
    "If the Fairness Paradox is real, the smallest-gap model should remain the smallest-gap model "
    "(or close to it) when LGD and discount-rate assumptions change. "
    "Computed at fixed threshold 0.50 across all scenarios for comparability."
)


@st.cache_data
def compute_sensitivity_table(_probs_columns):
    """Compute per-applicant gap for each model under 3 economic scenarios.
    Threshold fixed at 0.50 so we isolate the effect of LGD/rate assumptions."""
    scenarios = [
        ("Base (LGD 0.40, rate 3.0%)", 0.40, 0.030),
        ("High rate (LGD 0.40, rate 5.0%)", 0.40, 0.050),
        ("High LGD (LGD 0.55, rate 3.0%)", 0.55, 0.030),
    ]
    rows = []
    raw_gaps = []  # raw floats for ranking
    for name in _probs_columns:
        row = {"Model": name}
        raw = {}
        gaps = []
        for label, lgd_s, disc_s in scenarios:
            m_s = compute_metrics(probs[name].values, 0.50, lgd_s, disc_s)
            row[label] = f"${m_s['gap']:+,.0f}"
            raw[label] = m_s["gap"]
            gaps.append(m_s["gap"])
        # Stability flag â sign of gap consistent across scenarios?
        same_sign = all(g > 0 for g in gaps) or all(g < 0 for g in gaps)
        row["Sign stable?"] = "â Yes" if same_sign else "â ï¸ Flips"
        rows.append(row)
        raw_gaps.append(raw)
    return rows, raw_gaps


sens_rows, sens_raw = compute_sensitivity_table(tuple(probs.columns))

# Add ranking columns â which model is #1 (smallest gap) in each scenario?
scenarios_cols = [c for c in sens_rows[0].keys() if c not in ("Model", "Sign stable?")]
rank_rows = []
for col in scenarios_cols:
    # Use raw float values directly (no string parsing needed)
    values = [abs(raw[col]) for raw in sens_raw]
    sorted_idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0] * len(values)
    for rank_pos, idx in enumerate(sorted_idx):
        ranks[idx] = rank_pos + 1
    rank_rows.append((col, ranks))

# Inject rank into the display
for i, r in enumerate(sens_rows):
    for col, ranks in rank_rows:
        r[col] = f"{r[col]}  (#{ranks[i]})"

st.dataframe(pd.DataFrame(sens_rows), hide_index=True, use_container_width=True)
st.caption(
    "(#N) = rank within scenario, 1 = smallest absolute gap. "
    "**Sign stable** means the gap stays positive (NonWhite > White cost) across all 3 scenarios; "
    "models that flip sign indicate the gap is sensitive to economic assumptions and shouldn't "
    "be over-interpreted."
)

# Find best model across all scenarios
best_in_each = [ranks for _, ranks in rank_rows]
best_model_idx = min(range(len(sens_rows)),
                     key=lambda i: sum(rank[i] for rank in best_in_each))
best_model_name = sens_rows[best_model_idx]["Model"]

st.info(
    f"ð **Robustness read.** Across 3 economic scenarios, **{best_model_name}** has the lowest "
    f"average rank on per-applicant racial gap. Model rankings shift between scenarios â "
    f"which is exactly why DI alone is insufficient for credit policy. Production deployment "
    f"would calibrate LGD to internal data and could re-run this table on demand. "
    f"Dynamic LGD scenario (LTV-based) shown in the methodology paper requires "
    f"`property_value` field not exposed in this demo's cost framework."
)



st.divider()
st.markdown(
    """
**Method note.** Costs from HMDA 2021 NY applications: false negatives valued at NPV of foregone
interest; false positives at remaining balance Ã LGD. CIs shown when sliders are at defaults are
95% percentile CIs from 1,000 bootstrap iterations. Subgroup breakdown groups derive from
HMDA `derived_race` + `derived_ethnicity` (Hispanic overrides race per HMDA convention).
The "Other" subgroup (<1% of test set) is omitted from the table for clarity.
    """
)
