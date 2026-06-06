"""
Synthetic data generator for causal uplift modeling.

Ground truth tau(x) is fully observable — this is the key validation advantage of
synthetic data. The DGP encodes confounding (covariates drive both treatment assignment
and outcome), heterogeneous treatment effects (segments of strong responders, near-zero
responders, and do-not-disturb customers), and overlap (propensity clipped to [0.1, 0.9]).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import cfg

FEATURE_COLS = ["age", "income", "recency", "channel", "loyal", "frequency", "tenure"]


def generate(n: int | None = None, seed: int | None = None) -> tuple[pd.DataFrame, np.ndarray, float]:
    """
    Generate synthetic observational data with known heterogeneous treatment effects.

    Returns
    -------
    df : DataFrame with covariates, treatment, outcome, propensity
    true_tau : ndarray of shape (n,) — individual-level true CATE
    true_ate : float — population average treatment effect
    """
    n = n or cfg.data.n_samples
    seed = seed if seed is not None else cfg.seed
    rng = np.random.default_rng(seed)

    # ── Covariates ────────────────────────────────────────────────────────
    age = rng.uniform(18, 70, n)
    income = rng.lognormal(mean=10.5, sigma=0.6, size=n)
    recency = rng.exponential(scale=30, size=n)
    channel = rng.binomial(1, 0.55, n).astype(float)   # 1=online, 0=offline
    loyal = rng.binomial(1, 0.30, n).astype(float)
    frequency = rng.poisson(lam=5, size=n).astype(float)
    tenure = rng.uniform(0, 10, n)

    # Standardise continuous covariates (used in propensity + DGP)
    age_s = _zscore(age)
    income_s = _zscore(np.log(income))
    recency_s = _zscore(recency)
    freq_s = _zscore(frequency)
    tenure_s = _zscore(tenure)

    # ── Propensity (confounded: high-income, online, frequent → more likely treated) ──
    log_odds = (
        0.5 * income_s
        + 0.4 * age_s
        - 0.3 * recency_s
        + 0.3 * channel
        + 0.2 * freq_s
        - 0.1 * tenure_s
    )
    propensity = _sigmoid(log_odds)
    propensity = np.clip(propensity, 0.10, 0.90)   # enforce overlap

    treatment = rng.binomial(1, propensity).astype(float)

    # ── Baseline outcome (mu0): high income, loyal, frequent → high baseline ──
    mu0 = (
        10.0
        + 2.5 * income_s
        + 1.5 * age_s
        + 0.8 * loyal
        + 0.6 * freq_s
        - 0.5 * recency_s
        + 0.3 * tenure_s
    )

    # ── True CATE (heterogeneous) ─────────────────────────────────────────
    # Design: loyal + young + recent → strong positive uplift
    #         old + low-frequency + long-since-purchase → negative (do-not-disturb)
    true_tau = (
        3.0 * loyal                              # loyalty flag: strongest driver
        + 2.0 * (1 - (age - 18) / 52)           # younger → higher uplift
        - 1.5 * recency_s                        # recent → higher uplift
        + 0.8 * freq_s                           # high frequency → slightly more
        - 0.5                                    # global shift so ~10% are negative
    )

    noise = rng.normal(0, 1.5, n)
    outcome = mu0 + treatment * true_tau + noise

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "recency": recency,
        "channel": channel,
        "loyal": loyal,
        "frequency": frequency,
        "tenure": tenure,
        "treatment": treatment,
        "outcome": outcome,
        "propensity_true": propensity,
    })

    true_ate = float(true_tau.mean())
    return df, true_tau, true_ate


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df, true_tau, true_ate = generate()
    print(f"Rows            : {len(df):,}")
    print(f"Treatment rate  : {df['treatment'].mean():.3f}")
    print(f"True ATE        : {true_ate:.4f}")
    print(f"tau(x) range    : [{true_tau.min():.2f}, {true_tau.max():.2f}]")
    print(f"Fraction tau<0  : {(true_tau < 0).mean():.2%}")
    print(f"Outcome mean    : {df['outcome'].mean():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(true_tau, bins=60, color="steelblue", edgecolor="white")
    axes[0].axvline(true_ate, color="red", linestyle="--", label=f"ATE={true_ate:.2f}")
    axes[0].axvline(0, color="black", linestyle=":", label="Zero")
    axes[0].set_title("True CATE distribution")
    axes[0].set_xlabel("tau(x)")
    axes[0].legend()

    axes[1].hist(df["propensity_true"], bins=40, color="orange", edgecolor="white")
    axes[1].set_title("Propensity distribution (overlap check)")
    axes[1].set_xlabel("P(T=1|X)")
    plt.tight_layout()
    plt.savefig("true_tau_hist.png", dpi=120)
    print("Saved true_tau_hist.png")
