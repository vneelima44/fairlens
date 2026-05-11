# FairLens

**Interactive credit-decisioning fairness tool. Translates statistical fairness metrics into the dollar units that threshold decisions are actually made in.**

🔗 **[Live demo](https://fairlens-vneelima.streamlit.app/)** • Built on HMDA 2021 NY mortgage applications (~95K test records)

---

## The problem

At banks, fairness analysis sits with compliance teams who report metrics like Disparate Impact (DI), Equalized Odds, and Statistical Parity. These are *statistical* units. But the people who actually set model thresholds — chief risk officers, model risk committees, fair lending officers — think in *dollars*: expected loss, approval volume, regulatory exposure.

The two groups don't share a vocabulary, so fairness analysis often becomes a quarterly compliance checkbox that doesn't influence the threshold decisions where disparate harm gets baked in.

**FairLens translates fairness into the same units those decision-makers already use.** Slide the threshold; watch fairness metrics *and* the per-applicant dollar gap update simultaneously.

---

## The headline finding: the Fairness Paradox

On 95,588 HMDA 2021 NY mortgage applications, at the standard 0.5 decision threshold:

| Model | Disparate Impact | Per-applicant racial cost gap |
|---|---|---|
| LR Baseline | **0.979** (highest) | **$4,711** |
| LR Manual RW | 0.899 | $2,916 |
| **NN + Focal Loss** | 0.970 | **$2,333** (smallest) |

The model that scores highest on the traditional fairness metric (LR Baseline, DI 0.979) — the model a compliance team would call "most fair" by ECOA's 80% rule — actually leaves **~2× the per-applicant racial cost gap** of a neural network with focal loss (DI 0.970).

Picking by DI alone leaves **$2,378 per applicant** of disparate financial harm on the table — about **$227M** over the 95,588 applicants in this test set. Compliance teams choosing models by statistical fairness metrics systematically miss the real-dollar impact on protected groups.

---

## How the cost framework works

For each applicant decision, FairLens assigns a dollar cost:

- **False negative** (creditworthy applicant denied) → lender's foregone interest income, valued as NPV of all monthly payments minus principal, at a chosen discount rate.
- **False positive** (applicant approved who would default) → lender's loss on default, valued as remaining balance at the assumed default time × Loss Given Default (LGD).

Per-applicant cost is then aggregated by protected group (White / Non-White, as classified in HMDA), and the **racial gap** is `(Non-White per-applicant cost) − (White per-applicant cost)`.

The interactive demo lets users tune:
- **Decision threshold** (0.0–1.0)
- **LGD** (0.10–0.90, industry default 0.40)
- **Discount rate** (0.01–0.10, default 0.030 to match 2021 mortgage rate environment)

Cost values are assumption-sensitive. Sliding LGD or discount rate shows the user exactly how robust each model's gap is to economic regime changes.

---

## Methodology

### Models trained (10 total)

In the [Colab notebook](#) (not in the live demo, which uses three representative models for narrative clarity):

- LR Baseline, LR Balanced, LR with manual reweighing, LR with proxy-feature removal, LR with reweighing + proxy removal, LR with AIF360 reweighing
- SVM Linear (calibrated)
- Decision Tree (constrained), Random Forest
- Neural network with three loss functions: Cross-Entropy, Focal Loss, Combined

### Fairness interventions evaluated

- **Sample reweighing** (manual implementation + AIF360 cross-validation; weight vectors correlated 1.000)
- **Proxy feature removal** (5 features: credit score type indicators, etc.)
- **Asymmetric loss** (focal loss for the under-represented denied class)

### Target leakage diagnostic

An early version of the model achieved suspicious AUCs of 0.99 across tree-based and neural net models. A single-feature drop diagnostic identified **`interest_rate_log`** as the source: interest rates don't exist for denied applications and were median-imputed, creating a near-perfect target signal. Dropping the feature brought AUCs to the credible 0.80–0.83 range expected for credit decisioning.

This caught-my-own-leak step is one of the project's most important methodological signals.

### Sensitivity analysis

The Fairness Paradox holds across four economic-assumption scenarios (varying discount rate and LGD). Linear models with fairness reweighing show **rate fragility** — their gap flips sign as discount rate rises — while tree-based and NN models stay stable.

---

## Data

- **Source:** Home Mortgage Disclosure Act (HMDA) Modified Loan Application Register, 2021, New York State
- **Provider:** Consumer Financial Protection Bureau (CFPB), via FFIEC public API
- **Subset:** Single-family conventional purchase applications, action_taken ∈ {1, 2, 3}
- **N:** 477,937 applications (382,349 train / 95,588 test, stratified)
- **Approval rate:** 83.1% (refi-boom year; affects cost-framework calibration)

---

## Tech stack

- **Modeling:** scikit-learn, PyTorch, AIF360, Fairlearn
- **Data:** pandas, NumPy, pyarrow, public CFPB API
- **App:** Streamlit, Plotly
- **Deployment:** Streamlit Cloud, GitHub

---

## Reproducing the analysis

```bash
git clone https://github.com/vneelima44/fairlens.git
cd fairlens
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

To regenerate model probabilities from raw HMDA data, see `notebooks/` (the EDA notebook downloads from the CFPB public API; the modeling notebook trains all 10 models and exports artifacts for the app).

---

## Limitations

- **Approved-loan rates are imputed for denied applicants** in the cost framework. The "what would the interest income have been" question requires a counterfactual rate; I use the actual rate for approved loans (median-impute for denied). Reasonable but assumption-laden.
- **No actual default outcomes.** HMDA reports application decisions, not loan performance. False-positive (default) costs are theoretical — based on the model's predicted defaults, not observed ones. Real FP cost would require loan performance data (Fannie/Freddie or Lending Club).
- **Refi-boom rate environment.** 2021 mortgage rates were ~3%, unusually low. Calibrating the cost framework with a 3% discount rate is defensible for 2021 but may not generalize. The sensitivity analysis surfaces this explicitly.
- **Race binarization.** HMDA's racial categories were collapsed to White / Non-White for the demo. A subgroup analysis (Black / Hispanic / Asian / White) exists in the notebook for 2020 data and reveals a *loan-size amplification* effect not visible in the binary view; regenerating it for 2021 is in the next-iteration roadmap.
- **No bootstrap confidence intervals** on the displayed fairness metrics or dollar gaps yet. Point estimates only. Adding CIs is high-priority follow-up work.

---

## Roadmap

- Bootstrap CIs on all metrics
- Calibration plots (reliability diagrams, Brier score, isotonic recalibration)
- Subgroup analysis for 2021 (Black / Hispanic / Asian / White) to recover the loan-size amplification finding
- LLM-generated decision explanations, with faithfulness evaluation against SHAP
- Multi-year temporal validation (2019 → 2021)

---

## License

MIT
