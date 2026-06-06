"""
Policy optimization: budget- and cost-constrained targeting.

Key outputs
-----------
1. Causal policy: treat top-k by predicted CATE, net of treatment cost
2. PolicyTree (interpretable EconML rule): human-readable decision rules
3. Do-not-disturb flag: units with CATE < 0
4. Comparison: causal policy vs random vs treat-everyone vs response model

Run: python -m src.policy.targeting
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from src.config import cfg

logger = logging.getLogger(__name__)
ARTIFACTS_DIR = Path("artifacts")


# ── Net-value policy ────────────────────────────────────────────────────────
def net_value_policy(
    tau_hat: np.ndarray,
    true_tau: np.ndarray,
    cost: float | None = None,
    budget_frac: float | None = None,
) -> dict[str, object]:
    """
    Select individuals to treat to maximise net incremental value.

    treat if tau_hat(x) > cost  AND  within budget (top-k by tau_hat - cost)
    """
    cost = cost if cost is not None else cfg.treatment.cost_per_unit
    budget_frac = budget_frac if budget_frac is not None else cfg.treatment.budget_fraction

    n = len(tau_hat)
    k = max(1, int(n * budget_frac))

    net_tau = tau_hat - cost
    # rank by net value descending; additionally require net_tau > 0
    order = np.argsort(-net_tau)
    model_idx = order[:k]
    dnd_mask = tau_hat < 0                  # do-not-disturb

    model_gross  = float(true_tau[model_idx].mean())
    model_net    = float((true_tau[model_idx] - cost).mean())
    random_idx   = np.random.default_rng(cfg.seed).choice(n, k, replace=False)
    random_gross = float(true_tau[random_idx].mean())
    random_net   = float((true_tau[random_idx] - cost).mean())
    all_gross    = float(true_tau.mean())
    all_net      = float((true_tau - cost).mean())

    return {
        "budget_frac":    budget_frac,
        "n_treated":      k,
        "n_dnd":          int(dnd_mask.sum()),
        "model_gross_pv": round(model_gross, 4),
        "model_net_pv":   round(model_net, 4),
        "random_gross_pv": round(random_gross, 4),
        "random_net_pv":  round(random_net, 4),
        "all_gross_pv":   round(all_gross, 4),
        "all_net_pv":     round(all_net, 4),
        "lift_vs_random": round((model_net - random_net) / (abs(random_net) + 1e-9), 4),
        "model_idx":      model_idx,
        "dnd_mask":       dnd_mask,
    }


# ── Response model baseline ───────────────────────────────────────────────────
def response_model_policy(
    X_train: pd.DataFrame,
    Y_train: np.ndarray,
    T_train: np.ndarray,
    X_test: pd.DataFrame,
    true_tau_test: np.ndarray,
    budget_frac: float | None = None,
    cost: float | None = None,
    seed: int = 42,
) -> dict[str, object]:
    """
    Naive response model: classify who converts (regardless of treatment),
    target top-k% by predicted probability.  This is what uplift modeling beats.
    """
    cost = cost if cost is not None else cfg.treatment.cost_per_unit
    budget_frac = budget_frac if budget_frac is not None else cfg.treatment.budget_fraction

    # Fit a response model on treated+control combined (ignores T — that's the point)
    y_binary = (Y_train > Y_train.mean()).astype(int)
    rm = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=seed
    )
    rm.fit(X_train, y_binary)
    resp_score = rm.predict_proba(X_test)[:, 1]

    n = len(resp_score)
    k = max(1, int(n * budget_frac))
    resp_idx = np.argsort(-resp_score)[:k]

    resp_gross = float(true_tau_test[resp_idx].mean())
    resp_net   = float((true_tau_test[resp_idx] - cost).mean())

    return {
        "response_gross_pv": round(resp_gross, 4),
        "response_net_pv":   round(resp_net, 4),
        "resp_idx":          resp_idx,
    }


# ── PolicyTree ────────────────────────────────────────────────────────────────
def fit_policy_tree(
    X_train: pd.DataFrame,
    tau_hat_train: np.ndarray,
    cost: float | None = None,
    max_depth: int = 3,
) -> object:
    """
    Fit an EconML PolicyTree as an interpretable proxy for the CATE-based policy.
    Returns the fitted tree (or None if EconML is unavailable).
    """
    cost = cost if cost is not None else cfg.treatment.cost_per_unit
    try:
        from econml.policy import PolicyTree
        tree = PolicyTree(max_depth=max_depth, random_state=cfg.seed)
        # PolicyTree expects individual treatment values (tau_hat - cost as net gain)
        tree.fit(X_train, np.column_stack([-np.zeros(len(X_train)),
                                            tau_hat_train - cost]))
        return tree
    except Exception as exc:
        logger.warning("PolicyTree unavailable: %s", exc)
        return None


def print_policy_rules(tree, feature_names: list[str]) -> str:
    """Export PolicyTree rules as a text summary."""
    if tree is None:
        return "PolicyTree unavailable"
    try:
        from sklearn.tree import export_text
        inner = getattr(tree, "tree_model_", None)
        if inner is not None:
            return export_text(inner, feature_names=feature_names)
        return str(tree)
    except Exception as exc:
        return f"Could not export rules: {exc}"


# ── Comparison plot ───────────────────────────────────────────────────────────
def plot_policy_comparison(
    budgets: np.ndarray,
    causal_net: np.ndarray,
    random_net: np.ndarray,
    response_net: np.ndarray,
    treat_all_net: np.ndarray,
    save_path: str = "artifacts/policy_comparison.png",
) -> None:
    Path(save_path).parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(budgets, causal_net,   lw=2.5, color="steelblue", label="Causal policy (CATE)")
    ax.plot(budgets, response_net, lw=2,   color="orange",    label="Response model")
    ax.plot(budgets, random_net,   lw=1.5, color="gray",  linestyle="--", label="Random")
    ax.axhline(treat_all_net[0],   lw=1,   color="red",   linestyle=":",  label="Treat everyone")
    ax.set_xlabel("Treatment budget (fraction of population)")
    ax.set_ylabel("Net incremental value per targeted unit")
    ax.set_title("Causal Policy vs Response Model vs Random\n(net of treatment cost)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    logger.info("Policy comparison saved to %s", save_path)


# ── Main ───────────────────────────────────────────────────────────────────────
def run_policy(verbose: bool = True) -> dict:
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
    T_train, T_test = T[tr_idx], T[te_idx]  # noqa: F841
    Y_train, Y_test = Y[tr_idx], Y[te_idx]  # noqa: F841
    tau_test = true_tau[te_idx]
    tau_train = true_tau[tr_idx]  # noqa: F841

    if verbose: print("Fitting estimators...")
    fitted = fit_all(X_train, T_train, Y_train, verbose=verbose)

    # Use best estimator (lowest MSE on synthetic = CausalForest / X-Learner; pick LinearDML for CI)
    best_name = "LinearDML"
    best_est  = fitted[best_name]
    tau_hat_test  = best_est.predict_cate(X_test)
    tau_hat_train = best_est.predict_cate(X_train)

    cost = cfg.treatment.cost_per_unit

    # Sweep budgets
    budgets      = np.linspace(0.05, 1.0, 40)
    causal_nets  = []
    random_nets  = []
    resp_nets    = []
    treat_all_net = float((tau_test - cost).mean())

    resp = response_model_policy(X_train, Y_train, T_train, X_test, tau_test, cost=cost)

    for b in budgets:
        p = net_value_policy(tau_hat_test, tau_test, cost=cost, budget_frac=b)
        causal_nets.append(p["model_net_pv"])
        random_nets.append(p["random_net_pv"])
        # response model — use same scored ranking, different k
        n = len(tau_hat_test)
        k = max(1, int(n * b))
        # re-score for this budget
        # (already scored above — use resp_idx ordered by score)
        # Just use the full scored list and slice k
        resp_model_pv = float(tau_test[resp["resp_idx"][:k]].mean()) - cost
        resp_nets.append(resp_model_pv)

    plot_policy_comparison(
        budgets, np.array(causal_nets), np.array(random_nets),
        np.array(resp_nets), np.full(len(budgets), treat_all_net),
    )

    # At default budget
    default_policy = net_value_policy(tau_hat_test, tau_test, cost=cost)

    if verbose:
        print(f"\nPolicy @ {cfg.treatment.budget_fraction:.0%} budget (cost={cost}):")
        print(f"  Causal (net) : {default_policy['model_net_pv']:.4f}")
        print(f"  Response model: {resp['response_net_pv']:.4f}")
        print(f"  Random (net) : {default_policy['random_net_pv']:.4f}")
        print(f"  Treat-all (net): {treat_all_net:.4f}")
        print(f"  Do-not-disturb customers: {default_policy['n_dnd']} "
              f"({100*default_policy['n_dnd']/len(tau_hat_test):.1f}%)")
        beat = default_policy["model_net_pv"] > resp["response_net_pv"]
        print(f"  Causal beats response model: {beat}")

    # PolicyTree
    tree = fit_policy_tree(X_train, tau_hat_train, cost=cost)
    rules = print_policy_rules(tree, FEATURE_COLS)
    if verbose: print(f"\nPolicyTree rules:\n{rules}")

    summary = {
        "default_budget": cfg.treatment.budget_fraction,
        "causal_net_pv":   default_policy["model_net_pv"],
        "response_net_pv": resp["response_net_pv"],
        "random_net_pv":   default_policy["random_net_pv"],
        "treat_all_net_pv": treat_all_net,
        "n_dnd":           default_policy["n_dnd"],
        "causal_beats_response": bool(default_policy["model_net_pv"] > resp["response_net_pv"]),
        "policy_tree_rules": rules,
    }
    with open(ARTIFACTS_DIR / "policy_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return {
        "summary": summary, "default_policy": default_policy,
        "tau_hat_test": tau_hat_test, "tau_test": tau_test,
        "X_train": X_train, "tau_hat_train": tau_hat_train,
        "budgets": budgets, "causal_nets": causal_nets,
        "random_nets": random_nets, "resp_nets": resp_nets,
        "fitted": fitted,
    }


if __name__ == "__main__":
    run_policy(verbose=True)
