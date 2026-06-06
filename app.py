"""
Streamlit dashboard for uplift modeling portfolio piece.
Run: streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.data import generate_data, FEATURE_COLS
from src.learners import fit_all
from src.evaluate import compute_qini, policy_report, placebo_test

st.set_page_config(page_title="Uplift Modeling Dashboard", layout="wide")
st.title("Uplift Modeling: Who is Causally Moved by an Intervention?")
st.markdown(
    "_This dashboard demonstrates heterogeneous treatment effect estimation (CATE) "
    "on synthetic data with known ground truth. Goal: rank individuals by causal uplift, "
    "not just predicted response — then target under a budget._"
)


@st.cache_data
def load_and_fit():
    df, true_tau, true_ate = generate_data()
    X = df[FEATURE_COLS]
    T = df["treatment"].values
    Y = df["outcome"].values

    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=0.3, random_state=42)

    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    T_train, T_test = T[tr_idx], T[te_idx]
    Y_train, Y_test = Y[tr_idx], Y[te_idx]
    tau_test = true_tau[te_idx]

    fitted = fit_all(X_train, T_train, Y_train)

    results = {}
    for name, learner in fitted.items():
        tau_hat = learner.predict_cate(X_test)
        fracs, mg, rg, qini = compute_qini(tau_hat, T_test, Y_test)
        results[name] = {
            "tau_hat": tau_hat,
            "recovered_ate": float(tau_hat.mean()),
            "corr": float(np.corrcoef(tau_hat, tau_test)[0, 1]),
            "mse": float(np.mean((tau_hat - tau_test) ** 2)),
            "qini": qini,
            "fracs": fracs, "model_gains": mg, "random_gains": rg,
        }

    placebo_ate, placebo_std = placebo_test(X_train, T_train, Y_train, X_test)

    return results, true_tau, true_ate, tau_test, T_test, Y_test, placebo_ate, placebo_std


with st.spinner("Fitting models (first load ~30 s) ..."):
    results, true_tau, true_ate, tau_test, T_test, Y_test, placebo_ate, placebo_std = load_and_fit()

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.header("Policy Controls")
budget = st.sidebar.slider("Treatment budget (% of population)", 5, 100, 30, 5) / 100.0
selected_learner = st.sidebar.selectbox("Active learner", list(results.keys()))

# ── KPI row ───────────────────────────────────────────────────────────────
r = results[selected_learner]
policy = policy_report(r["tau_hat"], tau_test, T_test, Y_test, budget_frac=budget)

c1, c2, c3, c4 = st.columns(4)
c1.metric("True ATE", f"{true_ate:.3f}")
c2.metric("Recovered ATE", f"{r['recovered_ate']:.3f}",
          delta=f"{r['recovered_ate'] - true_ate:+.3f} vs true")
c3.metric("Qini Coefficient", f"{r['qini']:.4f}")
c4.metric("Model vs Random Uplift", f"{policy['model_vs_random_lift']*100:+.1f}%",
          help="Extra incremental outcome per treated unit vs random targeting")

st.divider()

# ── Main charts ───────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Qini / Uplift Curves")
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, res in results.items():
        lw = 2.5 if name == selected_learner else 1.2
        ax.plot(res["fracs"], res["model_gains"], lw=lw,
                label=f"{name} (Q={res['qini']:.3f})")
    first = next(iter(results.values()))
    ax.plot(first["fracs"], first["random_gains"], "k--", lw=1.5, label="Random")
    ax.axvline(budget, color="gray", linestyle=":", alpha=0.7, label=f"Budget={budget:.0%}")
    ax.set_xlabel("Fraction targeted"); ax.set_ylabel("Cumulative incremental outcome")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    st.pyplot(fig); plt.close(fig)

with col_right:
    st.subheader("True vs Predicted CATE (selected learner)")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(tau_test, r["tau_hat"], alpha=0.06, s=5, color="steelblue")
    lim = [min(tau_test.min(), r["tau_hat"].min()), max(tau_test.max(), r["tau_hat"].max())]
    ax.plot(lim, lim, "r--", label="Perfect recovery")
    ax.set_xlabel("True tau(x)"); ax.set_ylabel("Predicted tau(x)")
    ax.set_title(f"r = {r['corr']:.3f}   MSE = {r['mse']:.3f}")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig); plt.close(fig)

st.divider()

# ── Policy breakdown ──────────────────────────────────────────────────────
st.subheader(f"Policy: treat top {budget:.0%} by predicted uplift")
pc1, pc2, pc3 = st.columns(3)
pc1.metric("Avg uplift — model targeting", f"{policy['model_avg_uplift']:.4f}")
pc2.metric("Avg uplift — random targeting", f"{policy['random_avg_uplift']:.4f}")
pc3.metric("Avg uplift — treat everyone",  f"{policy['all_avg_uplift']:.4f}")

st.caption(
    f"Model targeting at {budget:.0%} budget treats **{policy['n_treated']} units** "
    f"with avg incremental outcome **{policy['model_avg_uplift']:.4f}** vs "
    f"**{policy['random_avg_uplift']:.4f}** for random — a "
    f"**{policy['model_vs_random_lift']*100:+.1f}%** lift."
)

st.divider()

# ── Learner comparison table ──────────────────────────────────────────────
st.subheader("Learner Comparison")
rows = []
for name, res in results.items():
    rows.append({
        "Learner": name,
        "Recovered ATE": f"{res['recovered_ate']:.4f}",
        "Corr w/ true tau": f"{res['corr']:.3f}",
        "MSE vs true tau": f"{res['mse']:.3f}",
        "Qini": f"{res['qini']:.4f}",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── ATE bar chart ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 3))
names = list(results.keys()) + ["Placebo (shuffled T)"]
ates  = [res["recovered_ate"] for res in results.values()] + [placebo_ate]
colors = ["steelblue"] * len(results) + ["tomato"]
ax.bar(range(len(names)), ates, color=colors)
ax.axhline(true_ate, color="green", linestyle="--", lw=2, label=f"True ATE = {true_ate:.3f}")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=15, ha="right")
ax.set_ylabel("Estimated ATE"); ax.set_title("Recovered ATE per Learner + Placebo Refutation")
ax.legend()
st.pyplot(fig); plt.close(fig)

st.caption(
    f"**Placebo refutation**: shuffled treatment labels → ATE collapses to "
    f"{placebo_ate:.4f} ± {placebo_std:.4f} (expected ≈ 0). "
    "This confirms the real estimates are not spurious."
)

st.divider()
st.subheader("True CATE Distribution")
fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(tau_test, bins=70, color="steelblue", edgecolor="white", alpha=0.8)
ax.axvline(true_ate, color="red", linestyle="--", lw=2, label=f"True ATE={true_ate:.3f}")
ax.axvline(0, color="black", linestyle=":", lw=1, label="Zero effect")
ax.set_xlabel("tau(x)"); ax.set_ylabel("Count")
ax.set_title("Distribution of true individual treatment effects (ground truth)")
ax.legend()
st.pyplot(fig); plt.close(fig)
st.caption("Data is synthetic: we know the true CATE for every unit. "
           "Negative values = 'do-not-disturb' customers (intervention hurts them).")
