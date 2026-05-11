"""
FairLens — Credit Decisioning Tradeoffs

v0.2 — adds bootstrap CIs and paradox significance test on the headline finding.

Data:   HMDA 2021 NY mortgage applications, ~95K test records.
Models: pre-trained, probabilities loaded from artifacts/test_probs.parquet.
CIs:    pre-computed at default settings (threshold=0.5, LGD=0.4, discount=0.03)
        from 1,000 bootstrap iterations; loaded from artifacts/bootstrap_cis.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="FairLens — Credit Decisioning Tradeoffs",
    page_icon="📊",
    layout="wide",
)


# ── Data loading ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    probs = pd.read_parquet("artifacts/test_probs.parquet")
    meta  = pd.read_parquet("artifacts/test_meta.parquet")
    cost  = pd.read_parquet("artifacts/cost_meta.parquet")
    return probs, meta, cost


@st.cache_data
def load_cis():
    """Load pre-computed bootstrap CIs if available."""
    path = Path("artifacts/bootstrap_cis.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


probs, meta, cost = load_data()
cis = load_cis()

y_true_arr = meta["y_true"].values.astype(int)
race_arr   = meta["race_binary"].values.astype(int)
loans_arr  = cost["loan_amount"].values.astype(float)
rates_arr  = cost["interest_rate"].values.astype(float)
terms_arr  = cost["loan_term_years"].values.astype(float)


# ── Vectorized cost framework ──────────────────────────────────────
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


# ── Header ─────────────────────────────────────────────────────────
st.title("FairLens")
st.markdown(
    "**Translating fairness metrics into the dollar units that "
    "threshold decisions are actually made in.**"
)
st.caption(
    "HMDA 2021 NY mortgage applications • 95,588 test records • interactive demo"
)


# ── Sidebar controls ───────────────────────────────────────────────
st.sidebar.header("Controls")

model_choice = st.sidebar.selectbox(
    "Model (for detail view below)",
    options=list(probs.columns),
    index=len(probs.columns) - 1,
    help="Selects the model whose detailed breakdown is shown in the lower section.",
)

threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.0, max_value=1.0, value=0.5, step=0.01,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Economic assumptions**")

lgd = st.sidebar.slider(
    "Loss Given Default (LGD)",
    min_value=0.10, max_value=0.90, value=0.40, step=0.05,
)

discount = st.sidebar.slider(
    "Discount rate",
    min_value=0.01, max_value=0.10, value=0.03, step=0.005,
    format="%.3f",
)

# Are we at canonical settings? (CIs only valid there)
at_defaults = (
    abs(threshold - 0.5) < 0.005
    and abs(lgd - 0.4) < 0.005
    and abs(discount - 0.03) < 0.001
)


# ── Compute metrics for ALL models at current settings ─────────────
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


# ── Helper to format a metric with CI when at defaults ────────────
def fmt_with_ci(point, metric_name, model_name, kind="number"):
    """Format a point estimate with bootstrap CI when at canonical settings."""
    if not at_defaults or cis is None:
        if kind == "money":
            return f"${point:+,.0f}"
        return f"{point:.3f}"

    model_cis = cis.get("models", {}).get(model_name, {})
    metric    = model_cis.get(metric_name)
    if metric is None:
        if kind == "money":
            return f"${point:+,.0f}"
        return f"{point:.3f}"

    lo, hi = metric["ci_lower"], metric["ci_upper"]
    if kind == "money":
        return f"${point:+,.0f} [${lo:+,.0f}, ${hi:+,.0f}]"
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


# ── Key Finding callout ────────────────────────────────────────────
if highest_di_model != smallest_gap_model:
    cost_diff_per_app = abs(highest_di_gap) - abs(smallest_gap)
    cost_diff_total_M = cost_diff_per_app * total_n / 1e6

    # P-value addition if we have CIs and are at defaults
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
                p_str = (
                    f" Paradox holds in **{n_holds}/{n_iter} bootstrap samples** "
                    f"(one-sided p < {1/n_iter:.4f})."
                )
            else:
                p_str = (
                    f" Paradox holds in **{n_holds}/{n_iter} bootstrap samples** "
                    f"(one-sided p = {p:.4f})."
                )

    st.info(
        f"📌 **The Fairness Paradox.** At your current settings, "
        f"**{highest_di_model}** has the highest Disparate Impact "
        f"({highest_di:.3f}) — the model a compliance team would pick as 'most fair'. "
        f"But **{smallest_gap_model}** (DI {smallest_di:.3f}) has the "
        f"**smallest per-applicant racial cost gap** "
        f"(\\${smallest_gap:+,.0f} vs \\${highest_di_gap:+,.0f}). "
        f"Picking by DI alone leaves **\\${cost_diff_per_app:,.0f} per applicant** "
        f"of disparate cost on the table — about **\\${cost_diff_total_M:,.0f}M** "
        f"over the {total_n:,} applicants in this test set." + p_str
    )
else:
    st.success(
        f"✅ At current settings, **{highest_di_model}** is best on both DI "
        f"and dollar-cost gap — no paradox here. Try moving the threshold or "
        f"LGD slider to see where they diverge."
    )


# ── Model comparison table ─────────────────────────────────────────
st.subheader("All models at current settings")
caption_text = "Sorted by absolute racial gap (smallest first)."
if at_defaults and cis is not None:
    caption_text += (
        f"  CIs from {cis['metadata']['n_iterations']} bootstrap iterations "
        f"at default settings."
    )
else:
    caption_text += "  (Move sliders back to defaults — threshold 0.50, "
    caption_text += "LGD 0.40, discount 0.030 — to see 95% confidence intervals.)"
st.caption(caption_text)

comp_rows = []
for name, r in sorted(all_results.items(), key=lambda x: abs(x[1]["gap"])):
    n_total = r["White"]["n"] + r["NonWhite"]["n"]
    approval = (
        r["White"]["approval_rate"] * r["White"]["n"]
        + r["NonWhite"]["approval_rate"] * r["NonWhite"]["n"]
    ) / n_total

    annotation = ""
    if name == smallest_gap_model:
        annotation = " ← smallest gap"
    elif name == highest_di_model and highest_di_model != smallest_gap_model:
        annotation = " ← highest DI"

    comp_rows.append({
        "Model":             name + annotation,
        "Disparate Impact":  fmt_with_ci(r["di"], "di", name),
        "Per-applicant gap": fmt_with_ci(r["gap"], "gap", name, kind="money"),
        "Approval rate":     f"{approval:.1%}",
        "ECOA (DI≥0.8)":     "✅ Pass" if r["di"] >= 0.8 else "❌ Fail",
    })

comp_df = pd.DataFrame(comp_rows)
st.dataframe(comp_df, hide_index=True, use_container_width=True)

st.divider()


# ── Detail view: hero metrics for selected model ───────────────────
st.subheader(f"Detail view: {model_choice}")
st.caption("Change model in the sidebar to compare. Numbers below reflect the selected model.")

m = all_results[model_choice]

total_n_sel = m["White"]["n"] + m["NonWhite"]["n"]
avg_approval = (
    m["White"]["approval_rate"] * m["White"]["n"]
    + m["NonWhite"]["approval_rate"] * m["NonWhite"]["n"]
) / total_n_sel

col1, col2, col3, col4 = st.columns(4)

# Show CI as the metric's "delta" (subtitle) when at defaults
def metric_delta(metric_name, kind="number"):
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

with col1:
    st.metric(
        "Per-applicant racial gap",
        f"${m['gap']:,.0f}",
        delta=metric_delta("gap", kind="money"),
        delta_color="off",
    )

with col2:
    di_delta = metric_delta("di")
    st.metric(
        "Disparate Impact",
        f"{m['di']:.3f}",
        delta=di_delta if di_delta else ("ECOA pass" if m["di"] >= 0.8 else "ECOA fail"),
        delta_color="off" if di_delta else ("normal" if m["di"] >= 0.8 else "inverse"),
    )

with col3:
    st.metric("Overall approval rate", f"{avg_approval:.1%}")

with col4:
    st.metric(
        "Equalized Odds Diff",
        f"{m['eod']:+.3f}",
        delta=metric_delta("eod"),
        delta_color="off",
    )

st.divider()


# ── Bar chart + fairness table ─────────────────────────────────────
left, right = st.columns([2, 1])

with left:
    st.subheader("Per-applicant cost by group")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["White", "NonWhite"],
        y=[m["White"]["per_applicant"], m["NonWhite"]["per_applicant"]],
        marker_color=["#4A90E2", "#D97757"],
        text=[
            f"${m['White']['per_applicant']:,.0f}",
            f"${m['NonWhite']['per_applicant']:,.0f}",
        ],
        textposition="outside",
        textfont=dict(size=14),
    ))
    fig.update_layout(
        yaxis_title="$ per applicant",
        showlegend=False,
        height=400,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Fairness metrics")
    metrics_df = pd.DataFrame({
        "Metric": [
            "Disparate Impact",
            "Statistical Parity Diff",
            "Equalized Odds Diff",
            "FPR Difference",
        ],
        "Value": [
            fmt_with_ci(m["di"],  "di",  model_choice),
            fmt_with_ci(m["spd"], "spd", model_choice),
            fmt_with_ci(m["eod"], "eod", model_choice),
            f"{m['fpr_diff']:+.3f}",  # not bootstrapped
        ],
        "Status": [
            "✅ Pass" if m["di"] >= 0.8 else "❌ Fail",
            "✅" if abs(m["spd"]) < 0.05 else "⚠️",
            "✅" if abs(m["eod"]) < 0.05 else "⚠️",
            "✅" if abs(m["fpr_diff"]) < 0.05 else "⚠️",
        ],
    })
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)


# ── Detailed breakdown ─────────────────────────────────────────────
st.divider()
st.subheader("Detailed breakdown by group")
breakdown_df = pd.DataFrame({
    "Group":                 ["White", "NonWhite"],
    "N applicants":          [m["White"]["n"], m["NonWhite"]["n"]],
    "Approval rate":         [
        f"{m['White']['approval_rate']:.1%}",
        f"{m['NonWhite']['approval_rate']:.1%}",
    ],
    "Wrongful denials (FN)": [m["White"]["fn_count"], m["NonWhite"]["fn_count"]],
    "Missed defaults (FP)":  [m["White"]["fp_count"], m["NonWhite"]["fp_count"]],
    "FN cost ($M)":          [
        f"${m['White']['fn_cost']/1e6:,.2f}",
        f"${m['NonWhite']['fn_cost']/1e6:,.2f}",
    ],
    "FP cost ($M)":          [
        f"${m['White']['fp_cost']/1e6:,.2f}",
        f"${m['NonWhite']['fp_cost']/1e6:,.2f}",
    ],
    "Per applicant ($)":     [
        f"${m['White']['per_applicant']:,.2f}",
        f"${m['NonWhite']['per_applicant']:,.2f}",
    ],
})
st.dataframe(breakdown_df, hide_index=True, use_container_width=True)


# ── Multi-model gap-vs-threshold curve ─────────────────────────────
st.divider()
st.subheader("Gap vs. threshold — all models compared")
st.caption(
    "Each line is a model. Lower = smaller racial gap. The threshold where "
    "each line bottoms out is that model's 'best' operating point. "
    "The dotted vertical line marks your current slider position."
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


thresholds_arr, curves = compute_curve_all_models(
    tuple(probs.columns), lgd, discount
)

colors = {
    "LR Baseline":     "#4A90E2",
    "LR Manual RW":    "#7B61FF",
    "NN + Focal Loss": "#D97757",
}

fig = go.Figure()
for name, gaps in curves.items():
    is_selected = (name == model_choice)
    fig.add_trace(go.Scatter(
        x=thresholds_arr, y=gaps,
        name=name,
        line=dict(
            color=colors.get(name, "#888"),
            width=3 if is_selected else 2,
            dash="solid" if is_selected else "dash",
        ),
        mode="lines",
        opacity=1.0 if is_selected else 0.7,
    ))

fig.add_vline(
    x=threshold, line_dash="dot", line_color="gray",
    annotation_text=f"Current: {threshold:.2f}",
    annotation_position="bottom right",
)
fig.add_hline(y=0, line_dash="dot", line_color="black")
fig.update_xaxes(title_text="Decision threshold")
fig.update_yaxes(title_text="Racial gap ($, NonWhite − White)")
fig.update_layout(
    height=450,
    hovermode="x unified",
    margin=dict(t=30, b=30),
    legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
)
st.plotly_chart(fig, use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
**Method note.** Costs computed from HMDA 2021 NY applications using a per-applicant
cost framework: false negatives valued at NPV of foregone interest (lender's lost
income on a creditworthy applicant denied); false positives at remaining balance ×
LGD (lender's loss on a default they should have caught). Confidence intervals
displayed when sliders are at defaults are 95% percentile CIs from 1,000 bootstrap
iterations on the test set. Cost values are sensitive to discount rate and LGD
assumptions — slide them above to see the impact.
    """
)
