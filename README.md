# Causal Uplift & Targeting Engine

**Production-grade heterogeneous treatment effect estimation with formal causal identification, refutation, sensitivity analysis, and budget-constrained policy optimization.**

---

## The Business Problem

Marketing, pricing, and product teams routinely ask: *"Who should we target?"*

The standard answer — train a response model, target the most likely converters — is wrong.  
It conflates *who will convert* with *who will convert **because of** the intervention*.

A customer who buys regardless of the promo is zero ROI. A customer who is annoyed by it is negative ROI. Spending budget on "sure things" and "do-not-disturbs" is the default failure mode of response modeling.

**Uplift modeling** solves this by estimating the **Conditional Average Treatment Effect (CATE)**:

> τ(x) = E[ Y(1) − Y(0) | X = x ]

— the causal increment in outcome caused by the intervention, per customer. Rank by τ(x), treat the highest-uplift customers, stay within budget. That's the playbook.

---

## Headline Results

| Strategy | Net value / targeted unit | vs. Random |
|---|---|---|
| **Causal uplift (LinearDML)** | **+2.99** | **+576%** |
| Response model (GBM) | +1.32 | +198% |
| Random targeting | +0.44 | baseline |
| Treat everyone | +0.39 | −11% |

> Causal targeting at 30% budget with cost=1.0/unit delivers **2.3× the ROI of a response model** and **6.8× the ROI of random targeting**.

**Do-not-disturb**: 24% of the population has τ̂(x) < 0 — contacting them is net harmful. The causal policy avoids them; the response model does not.

---

## Causal Assumptions & Validation Story

This project follows the full **DoWhy workflow**: model → identify → estimate → refute.

```
X ──→ T          confounding: observed covariates drive treatment assignment
X ──→ Y          observed covariates also drive baseline outcome
T ──→ Y          treatment effect — what we estimate
U ⇢ T, Y         unobserved confounders — sensitivity-tested below
```

**Identification**: backdoor adjustment on observed X (conditional ignorability).  
**Overlap**: propensity clipped to [0.10, 0.90]; 0.2% of extreme units trimmed.  
**Balance**: age/income/recency show SMD > 0.1 before trimming (intentional confounding).

### Refutation (DoWhy)
| Test | Result |
|---|---|
| Placebo treatment (permuted T) | ATE ≈ 0.001 ✅ |
| Random common cause | Estimate stable ✅ |
| Data subset (80%) | Estimate stable ✅ |
| Permutation placebo (×5 runs) | Mean ≈ −0.008, 70× signal-to-noise ✅ |

### Sensitivity (E-Value)
An unmeasured confounder would need risk-ratio ≥ **3.31** with *both* T and Y to fully explain away the ATE. Cohen's d = 0.37 (moderate effect).

### Synthetic → RCT validation
- **Synthetic**: known ground-truth τ(x); LinearDML recovers ATE within 3% and achieves r = 0.999 with true individual CATEs.
- **Hillstrom email RCT**: real A/B experiment (64k customers, email vs. no-email); Qini > random, spend ATE = +$0.60.

---

## Estimators

| Estimator | Notes |
|---|---|
| S-Learner | Single model on [X, T]; simplest baseline |
| T-Learner | Separate models per arm |
| X-Learner | T-Learner + imputed effects + propensity weighting; best for imbalanced T |
| R-Learner | Residual-on-residual; robust to nuisance misspecification |
| LinearDML (EconML) | Double/debiased ML with linear CATE; best MSE on synthetic |
| CausalForestDML (EconML) | Best ATE; native confidence intervals |
| DRLearner (EconML) | Doubly robust |

---

## Project Structure

```
├── src/
│   ├── config.py               # typed Pydantic config loader
│   ├── data/
│   │   ├── synthetic.py        # DGP with known ground-truth τ(x)
│   │   └── hillstrom.py        # real RCT loader (graceful fallback)
│   ├── causal/
│   │   ├── graph.py            # DoWhy DAG + estimand identification
│   │   ├── diagnostics.py      # propensity, overlap, balance, trimming
│   │   ├── estimators.py       # all 7 estimators + MLflow logging
│   │   ├── refute.py           # DoWhy refutation suite
│   │   └── sensitivity.py      # E-value + DoWhy sensitivity
│   ├── eval/
│   │   └── uplift.py           # Qini, AUUC, decile chart, policy value + CIs
│   ├── policy/
│   │   └── targeting.py        # net-value policy, PolicyTree, response-model comparison
│   ├── explain/
│   │   └── cate_shap.py        # SHAP on CATE model
│   └── api/
│       ├── main.py             # FastAPI: /health, /uplift
│       └── schemas.py          # Pydantic request/response schemas
├── app/
│   └── streamlit_app.py        # interactive dashboard (all panels)
├── tests/                      # pytest: recovery, refutation, policy gates
├── config/config.yaml          # all hyperparams + data config
├── MODEL_CARD.md
└── Dockerfile
```

---

## Why Synthetic Data?

With real observational data, we never observe both Y(0) and Y(1) for the same person — the fundamental problem of causal inference. Synthetic data gives us the **known τ(x) for every unit**, enabling us to:

- Directly measure how accurately each estimator recovers individual CATEs (not just average effects)
- Validate that ranking by τ̂(x) actually captures the highest-true-uplift units
- Confirm the placebo refutation correctly collapses to zero

This is a rigorous, honest methodology showcase. The algorithms generalize to any real domain; the synthetic ground truth is the validation tool.

---

## Run Instructions

```bash
# Install
pip install -r requirements.txt

# Data smoke tests
python -m src.data.synthetic          # prints true ATE, saves histogram
python -m src.data.hillstrom          # loads real RCT, prints ATEs

# Causal pipeline
python -m src.causal.estimators       # fits all 7 estimators, logs to MLflow
python -m src.causal.refute           # DoWhy refutation suite
python -m src.causal.sensitivity      # E-value sensitivity analysis

# Evaluation & policy
python -m src.eval.uplift             # Qini curves, AUUC, decile chart
python -m src.policy.targeting        # policy comparison, PolicyTree rules

# Explainability
python -m src.explain.cate_shap       # SHAP feature importance

# Serve
uvicorn src.api.main:app --reload     # REST API: POST /uplift
streamlit run app/streamlit_app.py    # interactive dashboard

# Quality gate
pytest && ruff check .
```

---

## Tech Stack

Python 3.12 · numpy · pandas · scikit-learn · **DoWhy** · **EconML** · **SHAP** · MLflow · FastAPI · Streamlit · Plotly · pytest · ruff
