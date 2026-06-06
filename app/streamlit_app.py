"""
Causal Uplift Engine — Streamlit Dashboard

Panels
------
1. KPIs: true ATE, recovered ATE, Qini, ROI vs random/response model
2. Qini curves: all estimators vs random
3. True vs predicted CATE scatter
4. Uplift by decile
5. Policy: budget slider → live net ROI comparison
6. Overlap + balance diagnostics
7. Refutation + placebo results
8. Sensitivity (E-value)
9. SHAP feature importance
10. Do-not-disturb segment
11. PolicyTree rules
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

from src.config import cfg
from src.data.synthetic import FEATURE_COLS, generate

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Causal Uplift Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Causal Uplift & Targeting Engine")
st.markdown(
    "_Heterogeneous treatment effect estimation · budget-constrained targeting · "
    "formal refutation · sensitivity analysis_"
)

ARTIFACTS = Path("artifacts")


# ── Data + model cache ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Fitting models (first load ~45 s)…")
def load_all():
    from src.causal.diagnostics import estimate_propensity, standardized_mean_diff
    from src.causal.estimators import fit_all
    from src.eval.uplift import compute_qini, policy_value_with_ci, uplift_by_decile
    from src.policy.targeting import (
        fit_policy_tree,
        net_value_policy,
        print_policy_rules,
        response_model_policy,
    )

    df, true_tau, true_ate = generate()
    X = df[FEATURE_COLS]; T = df["treatment"].values; Y = df["outcome"].values
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=cfg.data.test_size, random_state=cfg.seed)
    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    T_train, T_test = T[tr_idx], T[te_idx]
    Y_train, Y_test = Y[tr_idx], Y[te_idx]
    tau_test = true_tau[te_idx]

    # Diagnostics
    prop = estimate_propensity(df, FEATURE_COLS)
    smd  = standardized_mean_diff(df, FEATURE_COLS)

    # Estimators
    fitted = fit_all(X_train, T_train, Y_train, verbose=False)

    eval_results = {}
    for name, est in fitted.items():
        tau_hat = est.predict_cate(X_test)
        fracs, mg, rg, q = compute_qini(tau_hat, T_test, Y_test)
        pv = policy_value_with_ci(tau_hat, tau_test, cfg.treatment.budget_fraction)
        eval_results[name] = {
            "tau_hat": tau_hat, "fracs": fracs,
            "model_gains": mg, "random_gains": rg,
            "qini": q, "policy_value": pv,
            "recovered_ate": float(tau_hat.mean()),
            "corr": float(np.corrcoef(tau_hat, tau_test)[0, 1]),
            "mse":  float(np.mean((tau_hat - tau_test) ** 2)),
        }

    best_name = max(eval_results, key=lambda k: eval_results[k]["qini"])
    best_tau  = eval_results[best_name]["tau_hat"]
    decile_df = uplift_by_decile(best_tau, T_test, Y_test, true_tau=tau_test)

    # Policy
    ldml_tau = eval_results["LinearDML"]["tau_hat"]
    ldml_tau_train = fitted["LinearDML"].predict_cate(X_train)
    default_policy = net_value_policy(ldml_tau, tau_test)
    resp = response_model_policy(X_train, Y_train, T_train, X_test, tau_test)
    tree = fit_policy_tree(X_train, ldml_tau_train)
    rules = print_policy_rules(tree, FEATURE_COLS)

    # Placebo
    from src.causal.refute import manual_placebo_refutation
    placebo = manual_placebo_refutation(X_train, T_train, Y_train, X_test, n_runs=5)

    # Sensitivity
    from src.causal.sensitivity import evalue_continuous
    ate_se = float(ldml_tau.std() / np.sqrt(len(ldml_tau)))
    evalue = evalue_continuous(float(ldml_tau.mean()), df["outcome"].std(), ate_se)

    return {
        "df": df, "true_tau": true_tau, "true_ate": true_ate,
        "tau_test": tau_test, "T_test": T_test, "Y_test": Y_test,
        "X_train": X_train, "X_test": X_test,
        "prop": prop, "smd": smd,
        "eval_results": eval_results, "best_name": best_name,
        "decile_df": decile_df,
        "default_policy": default_policy, "resp": resp,
        "rules": rules,
        "placebo": placebo, "evalue": evalue,
        "ldml_tau": ldml_tau,
    }


data = load_all()
eval_results  = data["eval_results"]
true_ate      = data["true_ate"]
tau_test      = data["tau_test"]
best_name     = data["best_name"]
default_policy = data["default_policy"]
resp          = data["resp"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Controls")
budget = st.sidebar.slider("Treatment budget (%)", 5, 100, int(cfg.treatment.budget_fraction * 100), 5) / 100
cost   = st.sidebar.number_input("Treatment cost per unit", value=cfg.treatment.cost_per_unit, step=0.1)
sel    = st.sidebar.selectbox("Active estimator", list(eval_results.keys()))

# ── KPI row ───────────────────────────────────────────────────────────────────
r = eval_results[sel]
from src.policy.targeting import net_value_policy, response_model_policy  # noqa: E402

live_policy = net_value_policy(r["tau_hat"], tau_test, cost=cost, budget_frac=budget)
live_resp   = response_model_policy(
    data["X_train"], data["Y_test"][:len(data["X_train"])],
    data["T_test"][:len(data["X_train"])],
    data["X_test"], tau_test, budget_frac=budget, cost=cost,
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("True ATE", f"{true_ate:.3f}")
k2.metric("Recovered ATE", f"{r['recovered_ate']:.3f}",
          delta=f"{r['recovered_ate'] - true_ate:+.3f}")
k3.metric("Qini Coefficient", f"{r['qini']:.4f}")
k4.metric("Net Value (causal)", f"{live_policy['model_net_pv']:.3f}")
k5.metric("Net Value (response model)", f"{live_resp['response_net_pv']:.3f}",
          delta=f"{live_policy['model_net_pv'] - live_resp['response_net_pv']:+.3f} causal advantage")

st.divider()

# ── Row 1: Qini + True vs Predicted ──────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("Qini / Uplift Curves")
    fig = go.Figure()
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2"]
    for i, (name, res) in enumerate(eval_results.items()):
        lw = 4 if name == sel else 1.5
        fig.add_trace(go.Scatter(
            x=res["fracs"], y=res["model_gains"],
            name=f"{name} (Q={res['qini']:.3f})",
            line=dict(width=lw, color=colors[i % len(colors)])
        ))
    first = next(iter(eval_results.values()))
    fig.add_trace(go.Scatter(x=first["fracs"], y=first["random_gains"],
                             name="Random", line=dict(dash="dash", color="black", width=1.5)))
    fig.add_vline(x=budget, line_dash="dot", line_color="gray", annotation_text=f"Budget {budget:.0%}")
    fig.update_layout(xaxis_title="Fraction targeted", yaxis_title="Cumulative incremental outcome",
                      legend=dict(font=dict(size=9)), height=380)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader(f"True vs Predicted CATE — {sel}")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=tau_test, y=r["tau_hat"],
        mode="markers", marker=dict(size=3, opacity=0.3, color="steelblue"),
        name="Units"
    ))
    lim = [min(tau_test.min(), r["tau_hat"].min()), max(tau_test.max(), r["tau_hat"].max())]
    fig2.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                              line=dict(color="red", dash="dash"), name="Perfect"))
    fig2.update_layout(
        xaxis_title="True τ(x)", yaxis_title="Predicted τ(x)",
        title=f"r = {r['corr']:.3f}   MSE = {r['mse']:.3f}", height=380
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Row 2: Policy comparison + Decile chart ───────────────────────────────────
c3, c4 = st.columns(2)

with c3:
    st.subheader(f"Policy Comparison — Budget {budget:.0%}, Cost {cost:.2f}")

    budgets_sweep = np.linspace(0.05, 1.0, 40)
    causal_net_sweep, rand_net_sweep, resp_net_sweep = [], [], []
    ldml_tau = data["ldml_tau"]
    for b in budgets_sweep:
        p = net_value_policy(ldml_tau, tau_test, cost=cost, budget_frac=b)
        causal_net_sweep.append(p["model_net_pv"])
        rand_net_sweep.append(p["random_net_pv"])
        k = max(1, int(len(ldml_tau) * b))
        resp_net_sweep.append(float(tau_test[resp["resp_idx"][:k]].mean()) - cost)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=budgets_sweep, y=causal_net_sweep,
                              name="Causal (CATE)", line=dict(color="steelblue", width=3)))
    fig3.add_trace(go.Scatter(x=budgets_sweep, y=resp_net_sweep,
                              name="Response model", line=dict(color="orange", width=2)))
    fig3.add_trace(go.Scatter(x=budgets_sweep, y=rand_net_sweep,
                              name="Random", line=dict(color="gray", dash="dash")))
    treat_all = float((tau_test - cost).mean())
    fig3.add_hline(y=treat_all, line_dash="dot", line_color="red",
                   annotation_text=f"Treat-all ({treat_all:.2f})")
    fig3.add_vline(x=budget, line_dash="dot", line_color="gray")
    fig3.update_layout(xaxis_title="Budget (fraction)", yaxis_title="Net incremental value",
                       height=380)
    st.plotly_chart(fig3, use_container_width=True)
    dnd_pct = 100 * live_policy["n_dnd"] / len(ldml_tau)
    st.caption(f"Do-not-disturb customers (τ̂ < 0): **{live_policy['n_dnd']} "
               f"({dnd_pct:.1f}%)**. Causal policy avoids targeting them.")

with c4:
    st.subheader(f"Uplift by Decile — {best_name}")
    dd = data["decile_df"]
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=dd["decile"], y=dd["pred_cate_mean"],
                          name="Predicted CATE", marker_color="steelblue"))
    fig4.add_trace(go.Bar(x=dd["decile"], y=dd["observed_uplift"],
                          name="Observed uplift", marker_color="orange", opacity=0.8))
    if "true_cate_mean" in dd.columns:
        fig4.add_trace(go.Scatter(x=dd["decile"], y=dd["true_cate_mean"],
                                  name="True CATE", mode="lines+markers",
                                  line=dict(color="red", dash="dash")))
    fig4.add_hline(y=0, line_color="black", line_width=0.7)
    fig4.update_layout(xaxis_title="Decile (1=highest predicted uplift)",
                       yaxis_title="Treatment effect", barmode="group", height=380)
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ── Row 3: Diagnostics + Refutation ──────────────────────────────────────────
c5, c6 = st.columns(2)

with c5:
    st.subheader("Overlap & Covariate Balance")
    smd = data["smd"]
    prop = data["prop"]
    T_all = data["df"]["treatment"].values

    fig5, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].hist(prop[T_all==1], bins=40, alpha=0.6, density=True, color="steelblue", label="Treated")
    axes[0].hist(prop[T_all==0], bins=40, alpha=0.6, density=True, color="tomato",    label="Control")
    axes[0].set_title("Propensity Overlap"); axes[0].legend(fontsize=8)
    axes[0].set_xlabel("P(T=1|X)")

    axes[1].barh(smd.index, smd.values, color=["tomato" if abs(v)>0.1 else "steelblue" for v in smd])
    axes[1].axvline(0.1,  color="gray", linestyle="--", lw=0.8)
    axes[1].axvline(-0.1, color="gray", linestyle="--", lw=0.8)
    axes[1].axvline(0, color="black", lw=0.5)
    axes[1].set_title("Covariate Balance (SMD)")
    axes[1].set_xlabel("Standardised Mean Difference")
    plt.tight_layout()
    st.pyplot(fig5); plt.close(fig5)

with c6:
    st.subheader("Refutation & Sensitivity")
    placebo = data["placebo"]
    ev      = data["evalue"]

    st.markdown("**Permutation placebo** (T shuffled → ATE should collapse to ~0)")
    cols = st.columns(3)
    cols[0].metric("Placebo ATE", f"{placebo['mean']:.4f}", delta=f"std={placebo['std']:.4f}")
    cols[1].metric("True ATE", f"{true_ate:.4f}")
    cols[2].metric("Signal ratio", f"{abs(true_ate)/(abs(placebo['mean'])+1e-6):.1f}×")

    st.markdown("**Unobserved confounding (E-value)**")
    st.info(ev["interpretation"])
    cols2 = st.columns(2)
    cols2[0].metric("E-value (estimate)", f"{ev['evalue_estimate']}")
    cols2[1].metric("E-value (CI bound)", f"{ev['evalue_ci_bound']}")

st.divider()

# ── Row 4: Estimator comparison table + SHAP ─────────────────────────────────
c7, c8 = st.columns([1, 1])

with c7:
    st.subheader("Estimator Comparison")
    rows = []
    for name, res in eval_results.items():
        rows.append({
            "Estimator":    name,
            "ATE":          f"{res['recovered_ate']:.4f}",
            "Corr(τ̂,τ)":   f"{res['corr']:.3f}",
            "MSE(τ̂,τ)":    f"{res['mse']:.3f}",
            "Qini":         f"{res['qini']:.4f}",
            "Policy PV":    f"{res['policy_value']['model_pv']:.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("**PolicyTree Rules** (interpretable causal targeting policy)")
    st.code(data["rules"], language="text")

with c8:
    st.subheader("SHAP Feature Importance for CATE")
    shap_path = ARTIFACTS / "shap_summary.png"
    if shap_path.exists():
        st.image(str(shap_path), use_container_width=True)
        beeswarm = ARTIFACTS / "shap_summary_beeswarm.png"
        if beeswarm.exists():
            with st.expander("Beeswarm plot"):
                st.image(str(beeswarm), use_container_width=True)
    else:
        st.info("Run `python -m src.explain.cate_shap` to generate SHAP plots.")

st.divider()

# ── True CATE distribution ────────────────────────────────────────────────────
st.subheader("True CATE Distribution (Synthetic Ground Truth)")
fig6, ax = plt.subplots(figsize=(10, 3))
ax.hist(tau_test, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
ax.axvline(true_ate, color="red", linestyle="--", lw=2, label=f"True ATE={true_ate:.3f}")
ax.axvline(0, color="black", linestyle=":", lw=1, label="Zero effect")
neg_pct = 100*(tau_test<0).mean()
ax.set_xlabel("τ(x)"); ax.set_ylabel("Count")
ax.set_title(f"True individual treatment effects — {neg_pct:.1f}% negative (do-not-disturb)")
ax.legend()
st.pyplot(fig6); plt.close(fig6)
st.caption(
    "Data is synthetic with fully known ground-truth CATE — enabling direct validation "
    "of estimator recovery, an advantage unavailable with real observational data. "
    "Negative values represent customers who would be harmed by the intervention."
)
