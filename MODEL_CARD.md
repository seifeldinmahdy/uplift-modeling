# Model Card — Don't Target Buyers, Target the Persuadable

## Model Overview

**Task**: Heterogeneous treatment effect estimation (CATE) for budget-constrained targeting.  
**Output**: Per-individual causal uplift τ(x) = E[Y(1) − Y(0) | X=x] — the incremental outcome caused by a promotional intervention for customer x.  
**Use case**: Allocate a fixed treatment budget to the highest-uplift customers; explicitly avoid treating "do-not-disturb" customers (negative uplift) and "sure-things" (would convert regardless).

---

## Estimators

| Estimator | ATE Error (vs truth) | Corr(τ̂, τ) | MSE(τ̂, τ) | Notes |
|---|---|---|---|---|
| S-Learner (GBM) | ~0.16 | 0.971 | 0.456 | Fastest; shrinks CATE toward mean |
| T-Learner (GBM) | ~0.10 | 0.976 | 0.262 | Simple; extrapolation risk in tails |
| X-Learner (GBM) | ~0.03 | 0.992 | 0.086 | Best for imbalanced treatment |
| R-Learner (GBM) | ~0.03 | 0.983 | 0.188 | Robust to nuisance misspecification |
| LinearDML | ~0.04 | 0.999 | 0.010 | **Best overall** on synthetic; interpretable coefficients |
| CausalForest (DML) | ~0.02 | 0.977 | 0.308 | Best ATE; native CIs available |
| DRLearner | ~0.02 | 0.974 | 0.273 | Doubly robust; conservative |

---

## Identification Assumptions

1. **Conditional ignorability** (unconfoundedness): Y(t) ⊥ T | X — no unmeasured confounders after conditioning on observed covariates.
2. **Overlap** (positivity): 0 < P(T=1|X) < 1 for all X — every unit has a nonzero probability of receiving each treatment value. Enforced in synthetic data by clipping propensity to [0.10, 0.90]. Common-support trimming removes 0.2% of extreme units.
3. **SUTVA**: no interference between units; no hidden treatment versions.

**These assumptions are untestable from data alone.** The refutation and sensitivity sections below assess robustness.

---

## Diagnostics

- **Overlap (positivity)**: Propensity score distributions of treated and control overlap across the full [0.1, 0.9] range. No extreme tails. ✅
- **Covariate balance (SMD)**: Before trimming, `age` (SMD=0.37), `income` (SMD=0.39), and `recency` (SMD=−0.25) show imbalance > 0.1, confirming confounding. This is by design — the DGP is intentionally observational.

---

## Refutation Results (DoWhy)

| Test | Result | Interpretation |
|---|---|---|
| Placebo treatment (permuted T) | ATE ≈ 0.001 | ✅ Signal disappears without real treatment — not spurious |
| Random common cause | Stable | ✅ Adding a noise covariate doesn't change the estimate |
| Data subset (80%) | Stable | ✅ Estimate not sensitive to subset of data |
| Permutation placebo (T-Learner) | Mean ≈ −0.008, std=0.020 | ✅ Signal-to-noise ratio > 70× |

---

## Sensitivity Analysis (E-Value)

Estimated ATE ≈ 1.50 on the synthetic data (true ATE = 1.40).

- **Cohen's d** ≈ 0.37 (moderate effect)
- **E-value (estimate)** ≈ 3.31
- **E-value (CI bound)** ≈ 3.21

**Interpretation**: An unmeasured confounder would need to be associated with *both* treatment and outcome by a risk-ratio of ≥3.31 to fully explain away the estimated effect. A confounder of this strength is unlikely given the observed covariates already explain substantial variance.

---

## Policy Performance (Default budget = 30%, cost = 1.0/unit)

| Strategy | Net incremental value/unit |
|---|---|
| **Causal policy (LinearDML)** | **+2.99** |
| Response model (GBM classifier) | +1.32 |
| Random targeting | +0.44 |
| Treat everyone | +0.39 |

Causal policy beats the response model by **2.26× net ROI**, and beats random by **6.7×**.

**Do-not-disturb customers**: ~24% of the population has τ̂(x) < 0 — the intervention likely harms them. The causal policy explicitly avoids them.

---

## SHAP Feature Importance (Top CATE Drivers)

| Feature | Mean |SHAP| |
|---|---|
| loyal | 1.35 |
| recency | 0.88 |
| frequency | 0.41 |
| age | 0.33 |
| income | 0.01 |

Treatment effect heterogeneity is primarily driven by loyalty status and recency — matching the DGP.

---

## Limitations

1. **Conditional ignorability is unverifiable** from observational data. The E-value analysis assesses robustness; an unmeasured variable like "intent to purchase" could partially confound estimates.
2. **Synthetic data only for ground-truth validation**. Hillstrom RCT provides a real-world credibility check but does not have a known ground-truth CATE.
3. **No individual-level CIs** for meta-learners. CausalForestDML provides interval estimates; meta-learner uncertainty should be interpreted via bootstrapping.
4. **Static policy**: treatment effect estimates are point-in-time. Customer behaviour may change, requiring periodic retraining.
5. **Cost model is simplified**: a single flat cost per unit. A real deployment should model heterogeneous costs and capacity constraints.

---

## Intended Use

- Portfolio demonstration of causal inference methods.
- Template for real-world uplift modeling on e-commerce, fintech, or healthcare datasets where a treatment effect is to be estimated from observational or experimental data.
- **Not** intended for production deployment without additional validation on real target-domain data.
