"""
Uplift evaluation: Qini curve, AUUC, uplift-by-decile, policy value with CIs.
Works on both synthetic (true tau known) and Hillstrom data.

Run: python -m src.eval.uplift
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import cfg

logger = logging.getLogger(__name__)
ARTIFACTS_DIR = Path("artifacts")


# ── Core metrics ──────────────────────────────────────────────────────────────
def _ht_ate(Y: np.ndarray, T: np.ndarray, e: np.ndarray) -> float:
    """Horvitz-Thompson (IPW) ATE estimator."""
    e = np.clip(e, 0.05, 0.95)
    ipw = Y * T / e - Y * (1 - T) / (1 - e)
    return float(ipw.mean())


def compute_qini(
    tau_hat: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_buckets: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Qini curve and coefficient.

    Returns (fractions, model_gains, random_gains, qini_coef).
    Gain at k = incremental outcome from targeting top-k% vs. overall ATE * k.
    """
    order = np.argsort(-tau_hat)
    T, Y = treatment[order], outcome[order]
    n = len(T)

    step = max(1, n // n_buckets)
    checkpoints = set(range(step - 1, n, step))
    checkpoints.add(n - 1)

    cum_n1 = cum_n0 = 0
    cum_y1 = cum_y0 = 0.0
    fracs, gains = [0.0], [0.0]

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
            gains.append(gain)

    fracs = np.array(fracs)
    gains = np.array(gains)
    ate_obs = Y[T == 1].mean() - Y[T == 0].mean()
    random_gains = ate_obs * fracs

    from numpy import trapezoid
    qini = float(trapezoid(gains - random_gains, fracs))
    return fracs, gains, random_gains, qini


def auuc(fracs: np.ndarray, gains: np.ndarray, random_gains: np.ndarray) -> float:
    """Area Under the Uplift Curve (relative to random baseline)."""
    from numpy import trapezoid
    return float(trapezoid(gains - random_gains, fracs))


def uplift_by_decile(
    tau_hat: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    true_tau: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Split into deciles by predicted CATE; report observed uplift and (if available)
    true CATE per decile.
    """
    n = len(tau_hat)  # noqa: F841
    order = np.argsort(-tau_hat)   # descending
    decile_idx = np.array_split(order, 10)

    rows = []
    for d, idx in enumerate(decile_idx):
        T_d, Y_d = treatment[idx], outcome[idx]
        t_mask, c_mask = T_d == 1, T_d == 0
        obs_uplift = (Y_d[t_mask].mean() - Y_d[c_mask].mean()
                      if t_mask.sum() > 0 and c_mask.sum() > 0 else np.nan)
        row: dict = {
            "decile": d + 1,
            "n": len(idx),
            "pred_cate_mean": round(tau_hat[idx].mean(), 4),
            "observed_uplift": round(obs_uplift, 4) if not np.isnan(obs_uplift) else np.nan,
        }
        if true_tau is not None:
            row["true_cate_mean"] = round(true_tau[idx].mean(), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def policy_value_with_ci(
    tau_hat: np.ndarray,
    true_tau: np.ndarray,
    budget_frac: float = 0.30,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """
    Policy value (mean true CATE for targeted units) with bootstrap CIs.
    Also computes random and treat-all baselines for comparison.
    """
    n = len(tau_hat)
    k = max(1, int(n * budget_frac))
    model_idx = np.argsort(-tau_hat)[:k]

    rng = np.random.default_rng(seed)
    model_pv = float(true_tau[model_idx].mean())
    treat_all_pv = float(true_tau.mean())

    # Bootstrap CI on model policy value
    boot_vals = []
    for _ in range(n_bootstrap):
        b_idx = rng.choice(model_idx, size=k, replace=True)
        boot_vals.append(float(true_tau[b_idx].mean()))
    ci_lo = float(np.percentile(boot_vals, 2.5))
    ci_hi = float(np.percentile(boot_vals, 97.5))

    # Random baseline CI
    rand_vals = []
    for _ in range(n_bootstrap):
        r_idx = rng.choice(n, size=k, replace=False)
        rand_vals.append(float(true_tau[r_idx].mean()))
    rand_pv  = float(np.mean(rand_vals))
    rand_lo  = float(np.percentile(rand_vals, 2.5))
    rand_hi  = float(np.percentile(rand_vals, 97.5))

    return {
        "budget_frac":    budget_frac,
        "model_pv":       round(model_pv, 4),
        "model_ci_lo":    round(ci_lo, 4),
        "model_ci_hi":    round(ci_hi, 4),
        "random_pv":      round(rand_pv, 4),
        "random_ci_lo":   round(rand_lo, 4),
        "random_ci_hi":   round(rand_hi, 4),
        "treat_all_pv":   round(treat_all_pv, 4),
        "lift_vs_random": round((model_pv - rand_pv) / (abs(rand_pv) + 1e-9), 4),
    }


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_qini(
    results: dict[str, dict],
    save_path: str = "artifacts/qini_curves.png",
) -> None:
    Path(save_path).parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in results.items():
        ax.plot(r["fracs"], r["model_gains"], lw=2,
                label=f"{name} (Q={r['qini']:.4f})")
    first = next(iter(results.values()))
    ax.plot(first["fracs"], first["random_gains"], "k--", lw=1.5, label="Random")
    ax.set_xlabel("Fraction targeted")
    ax.set_ylabel("Cumulative incremental outcome")
    ax.set_title("Qini / Uplift Curves — Estimator Comparison")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    logger.info("Qini plot saved to %s", save_path)


def plot_decile(
    decile_df: pd.DataFrame,
    name: str = "Best Estimator",
    save_path: str = "artifacts/decile_chart.png",
) -> None:
    Path(save_path).parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = decile_df["decile"]
    ax.bar(x - 0.2, decile_df["pred_cate_mean"],    width=0.4, label="Pred CATE",     color="steelblue")
    ax.bar(x + 0.2, decile_df["observed_uplift"],   width=0.4, label="Observed uplift", color="orange", alpha=0.8)
    if "true_cate_mean" in decile_df.columns:
        ax.plot(x, decile_df["true_cate_mean"], "r--o", lw=1.5, ms=5, label="True CATE")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x); ax.set_xlabel("Decile (1 = highest predicted uplift)")
    ax.set_ylabel("Treatment effect"); ax.set_title(f"Uplift by Decile — {name}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────
def run_uplift_eval(verbose: bool = True) -> dict:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    from src.causal.estimators import fit_all
    from src.data.synthetic import FEATURE_COLS, generate

    df, true_tau, true_ate = generate()
    X = df[FEATURE_COLS]
    T = df["treatment"].values
    Y = df["outcome"].values

    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=cfg.data.test_size, random_state=cfg.seed)
    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    T_train, T_test = T[tr_idx], T[te_idx]
    Y_train, Y_test = Y[tr_idx], Y[te_idx]
    tau_test = true_tau[te_idx]

    if verbose: print("Fitting estimators...")
    fitted = fit_all(X_train, T_train, Y_train, verbose=verbose)

    eval_results: dict[str, dict] = {}
    for name, est in fitted.items():
        tau_hat = est.predict_cate(X_test)
        fracs, mg, rg, q = compute_qini(tau_hat, T_test, Y_test)
        pv = policy_value_with_ci(tau_hat, tau_test, cfg.treatment.budget_fraction)
        eval_results[name] = {
            "tau_hat": tau_hat,
            "fracs": fracs, "model_gains": mg, "random_gains": rg,
            "qini": q, "auuc": auuc(fracs, mg, rg),
            "policy_value": pv,
        }
        if verbose:
            print(f"  {name:20s}  Qini={q:.4f}  PolicyValue={pv['model_pv']:.4f} "
                  f"[{pv['model_ci_lo']:.4f},{pv['model_ci_hi']:.4f}]  "
                  f"vs random={pv['random_pv']:.4f}")

    # Best estimator decile chart (highest Qini)
    best_name = max(eval_results, key=lambda k: eval_results[k]["qini"])
    best_tau  = eval_results[best_name]["tau_hat"]
    decile_df = uplift_by_decile(best_tau, T_test, Y_test, true_tau=tau_test)
    decile_df.to_csv(ARTIFACTS_DIR / "decile_table.csv", index=False)
    if verbose:
        print(f"\nBest estimator by Qini: {best_name}")
        print(decile_df.to_string(index=False))

    plot_qini(eval_results)
    plot_decile(decile_df, name=best_name)

    # Hillstrom evaluation (if available)
    try:
        from src.data.hillstrom import FEATURE_COLS as H_FEATURES
        from src.data.hillstrom import load
        h_df = load()
        if h_df is not None:
            _run_hillstrom_eval(h_df, H_FEATURES, verbose=verbose)
    except Exception as exc:
        logger.warning("Hillstrom eval skipped: %s", exc)

    summary = {
        k: {
            "qini": v["qini"],
            "auuc": v["auuc"],
            "policy_value": v["policy_value"],
        }
        for k, v in eval_results.items()
    }
    with open(ARTIFACTS_DIR / "uplift_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return {"results": eval_results, "decile_df": decile_df,
            "best_name": best_name, "tau_test": tau_test,
            "T_test": T_test, "Y_test": Y_test}


def _run_hillstrom_eval(df: pd.DataFrame, feature_cols: list[str], verbose: bool = True) -> None:
    """Fit T-Learner on Hillstrom and plot Qini (no true tau — use observed outcomes)."""
    from src.causal.estimators import TLearner

    X = df[feature_cols]; T = df["treatment"].values; Y = df["spend"].values
    tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.3, random_state=cfg.seed)

    model = TLearner()
    model.fit(X.iloc[tr_idx], T[tr_idx], Y[tr_idx])
    tau_hat = model.predict_cate(X.iloc[te_idx])

    fracs, mg, rg, q = compute_qini(tau_hat, T[te_idx], Y[te_idx])
    if verbose:
        print(f"\nHillstrom (spend outcome): Qini={q:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fracs, mg, lw=2, label=f"T-Learner (Q={q:.4f})")
    ax.plot(fracs, rg, "k--", lw=1.5, label="Random")
    ax.set_xlabel("Fraction targeted"); ax.set_ylabel("Cumulative incremental spend")
    ax.set_title("Qini Curve — Hillstrom Real-World RCT")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "qini_hillstrom.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run_uplift_eval(verbose=True)
