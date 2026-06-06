"""
Evaluation, Qini curves, policy analysis, and placebo refutation.

Run as: python -m src.evaluate
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.data import generate_data, FEATURE_COLS
from src.learners import fit_all, TLearner


# ── Qini curve ────────────────────────────────────────────────────────────
def qini_curve(tau_hat: np.ndarray, treatment: np.ndarray, outcome: np.ndarray,
               n_buckets: int = 100):
    """
    Returns (fractions, cumulative_gain, random_baseline).
    Gain = incremental outcome from targeting top-k% vs. no targeting.
    """
    order = np.argsort(-tau_hat)
    T = treatment[order]
    Y = outcome[order]
    n = len(T)

    gains, randoms = [0.0], [0.0]
    fractions = [0.0]

    cum_n1 = 0; cum_n0 = 0
    cum_y1 = 0.0; cum_y0 = 0.0

    for i in range(n):
        if T[i] == 1:
            cum_n1 += 1; cum_y1 += Y[i]
        else:
            cum_n0 += 1; cum_y0 += Y[i]

        if (i + 1) % max(1, n // n_buckets) == 0 or i == n - 1:
            frac = (i + 1) / n
            if cum_n1 > 0 and cum_n0 > 0:
                gain = cum_y1 / cum_n1 - cum_y0 / cum_n0
            else:
                gain = 0.0
            gains.append(gain * frac)
            randoms.append(gain * frac * 0.0)   # placeholder; we'll use ate * frac
            fractions.append(frac)

    return np.array(fractions), np.array(gains)


def qini_coefficient(fractions, gains, random_gains):
    """Area between model curve and random baseline (trapezoidal)."""
    from numpy import trapezoid
    return float(trapezoid(gains - random_gains, fractions))


def compute_qini(tau_hat, treatment, outcome, n_buckets=100):
    """Returns (fractions, model_gains, random_gains, qini_coef)."""
    order = np.argsort(-tau_hat)
    T = treatment[order]; Y = outcome[order]
    n = len(T)

    # overall ATE in test set (unbiased Horvitz-Thompson style: just use means)
    ate_est = Y[T == 1].mean() - Y[T == 0].mean()

    fracs, model_gains = [], []
    step = max(1, n // n_buckets)
    checkpoints = list(range(step - 1, n, step)) + ([n - 1] if (n - 1) % step != 0 else [])

    cum_n1 = cum_n0 = 0
    cum_y1 = cum_y0 = 0.0

    for i in range(n):
        if T[i] == 1:
            cum_n1 += 1; cum_y1 += Y[i]
        else:
            cum_n0 += 1; cum_y0 += Y[i]
        if i in checkpoints:
            frac = (i + 1) / n
            if cum_n1 > 0 and cum_n0 > 0:
                gain = (cum_y1 / cum_n1 - cum_y0 / cum_n0) * frac
            else:
                gain = 0.0
            fracs.append(frac)
            model_gains.append(gain)

    fracs = np.array([0.0] + fracs)
    model_gains = np.array([0.0] + model_gains)
    random_gains = ate_est * fracs

    from numpy import trapezoid
    qini = float(trapezoid(model_gains - random_gains, fracs))
    return fracs, model_gains, random_gains, qini


# ── Policy analysis ───────────────────────────────────────────────────────
def policy_report(tau_hat, true_tau, treatment, outcome, budget_frac=0.30):
    """
    Compare: treat top-k% by predicted uplift vs random vs treat-everyone.
    Uses true_tau for ground-truth incremental outcome calculation.
    """
    n = len(tau_hat)
    k = int(n * budget_frac)

    model_idx = np.argsort(-tau_hat)[:k]
    random_idx = np.random.default_rng(0).choice(n, k, replace=False)

    model_inc  = true_tau[model_idx].mean()
    random_inc = true_tau[random_idx].mean()
    all_inc    = true_tau.mean()

    report = {
        "budget_frac": budget_frac,
        "n_treated": k,
        "model_avg_uplift":  round(model_inc, 4),
        "random_avg_uplift": round(random_inc, 4),
        "all_avg_uplift":    round(all_inc, 4),
        "model_vs_random_lift": round((model_inc - random_inc) / abs(random_inc + 1e-9), 4),
    }
    return report


# ── Placebo refutation ────────────────────────────────────────────────────
def placebo_test(X_train, T_train, Y_train, X_test):
    """Shuffle T, refit T-learner, confirm ATE collapses toward 0."""
    rng = np.random.default_rng(99)
    T_shuffled = rng.permutation(T_train)
    model = TLearner()
    model.fit(X_train, T_shuffled, Y_train)
    tau_placebo = model.predict_cate(X_test)
    return float(tau_placebo.mean()), float(tau_placebo.std())


# ── Main ──────────────────────────────────────────────────────────────────
def run_evaluation(verbose=True):
    if verbose:
        print("Generating data...")
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

    if verbose:
        print(f"True ATE: {true_ate:.4f}")
        print("Fitting learners...")

    fitted = fit_all(X_train, T_train, Y_train)

    results = {}
    for name, learner in fitted.items():
        tau_hat = learner.predict_cate(X_test)
        recovered_ate = float(tau_hat.mean())
        corr = float(np.corrcoef(tau_hat, tau_test)[0, 1])
        mse  = float(np.mean((tau_hat - tau_test) ** 2))
        fracs, mg, rg, qini = compute_qini(tau_hat, T_test, Y_test)
        policy = policy_report(tau_hat, tau_test, T_test, Y_test)

        results[name] = {
            "tau_hat": tau_hat,
            "recovered_ate": recovered_ate,
            "corr_with_true": corr,
            "mse_vs_true": mse,
            "qini": qini,
            "fracs": fracs, "model_gains": mg, "random_gains": rg,
            "policy": policy,
        }
        if verbose:
            print(f"\n{name}")
            print(f"  Recovered ATE: {recovered_ate:.4f}  (true: {true_ate:.4f})")
            print(f"  Corr with true tau: {corr:.3f}   MSE: {mse:.3f}")
            print(f"  Qini coef: {qini:.4f}")
            print(f"  Policy @ 30%: model uplift={policy['model_avg_uplift']:.3f}, "
                  f"random={policy['random_avg_uplift']:.3f}")

    # placebo
    placebo_ate, placebo_std = placebo_test(X_train, T_train, Y_train, X_test)
    if verbose:
        print(f"\nPlacebo (shuffled T): ATE={placebo_ate:.4f}, std={placebo_std:.4f}  "
              f"(should be ~0)")

    return results, true_ate, tau_test, placebo_ate, placebo_std


def plot_results(results, true_ate, tau_test, placebo_ate, save_path="qini_plot.png"):
    n_learners = len(results)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1) Qini curves
    ax = axes[0]
    for name, r in results.items():
        ax.plot(r["fracs"], r["model_gains"], label=f"{name} (Q={r['qini']:.3f})")
    # random baseline from first result
    first = next(iter(results.values()))
    ax.plot(first["fracs"], first["random_gains"], "k--", label="Random")
    ax.set_xlabel("Fraction targeted")
    ax.set_ylabel("Cumulative incremental outcome")
    ax.set_title("Qini / Uplift Curves")
    ax.legend(fontsize=7)

    # 2) True vs predicted CATE (T-learner)
    ax = axes[1]
    name_t = "T-Learner"
    if name_t in results:
        r = results[name_t]
        ax.scatter(tau_test, r["tau_hat"], alpha=0.05, s=5, color="steelblue")
        lim = [min(tau_test.min(), r["tau_hat"].min()),
               max(tau_test.max(), r["tau_hat"].max())]
        ax.plot(lim, lim, "r--", label="Perfect")
        ax.set_xlabel("True tau(x)")
        ax.set_ylabel("Predicted tau(x)")
        ax.set_title(f"T-Learner: True vs Predicted CATE\n(r={r['corr_with_true']:.3f})")
        ax.legend()

    # 3) Recovered ATE comparison + placebo
    ax = axes[2]
    names = list(results.keys()) + ["Placebo"]
    ates  = [r["recovered_ate"] for r in results.values()] + [placebo_ate]
    colors = ["steelblue"] * len(results) + ["tomato"]
    bars = ax.bar(range(len(names)), ates, color=colors)
    ax.axhline(true_ate, color="green", linestyle="--", label=f"True ATE={true_ate:.2f}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_title("Recovered ATE vs True (+ Placebo)")
    ax.set_ylabel("Estimated ATE")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    print(f"Saved {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    results, true_ate, tau_test, p_ate, p_std = run_evaluation(verbose=True)
    plot_results(results, true_ate, tau_test, p_ate)
