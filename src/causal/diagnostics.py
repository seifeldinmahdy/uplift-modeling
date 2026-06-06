"""
Causal diagnostics: propensity estimation, overlap/positivity plot,
covariate balance (Love plot), and common-support trimming.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def estimate_propensity(
    df: pd.DataFrame,
    feature_cols: list[str],
    treatment_col: str = "treatment",
    seed: int = 42,
) -> np.ndarray:
    """Fit a logistic regression propensity model and return P(T=1|X)."""
    X = df[feature_cols].values
    T = df[treatment_col].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    model.fit(X_s, T)
    return model.predict_proba(X_s)[:, 1]


def overlap_plot(
    df: pd.DataFrame,
    propensity: np.ndarray,
    treatment_col: str = "treatment",
    save_path: str = "artifacts/overlap.png",
) -> None:
    """Plot propensity distributions for treated and control — positivity check."""
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    t = propensity[df[treatment_col] == 1]
    c = propensity[df[treatment_col] == 0]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(t, bins=50, alpha=0.6, color="steelblue", density=True, label="Treated")
    ax.hist(c, bins=50, alpha=0.6, color="tomato",    density=True, label="Control")
    ax.set_xlabel("Estimated propensity P(T=1|X)")
    ax.set_ylabel("Density")
    ax.set_title("Propensity Overlap (Positivity Check)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info("Overlap plot saved to %s", save_path)


def standardized_mean_diff(
    df: pd.DataFrame,
    feature_cols: list[str],
    treatment_col: str = "treatment",
) -> pd.Series:
    """
    Compute standardized mean difference (SMD) per covariate.
    SMD = (mean_t - mean_c) / pooled_std.  |SMD| < 0.1 is the conventional balance threshold.
    """
    treated = df[df[treatment_col] == 1][feature_cols]
    control = df[df[treatment_col] == 0][feature_cols]

    mean_diff = treated.mean() - control.mean()
    pooled_std = np.sqrt((treated.var() + control.var()) / 2).replace(0, np.nan)
    smd = mean_diff / pooled_std
    return smd


def love_plot(
    smd_before: pd.Series,
    smd_after: pd.Series | None = None,
    save_path: str = "artifacts/balance.png",
) -> None:
    """Love plot: covariate balance before (and optionally after) weighting."""
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    features = smd_before.index.tolist()
    n = len(features)

    fig, ax = plt.subplots(figsize=(7, max(4, n * 0.4)))
    y_pos = range(n)

    ax.scatter(smd_before.values, y_pos, marker="o", color="tomato", zorder=3, label="Before")
    if smd_after is not None:
        ax.scatter(smd_after.values, y_pos, marker="D", color="steelblue", zorder=3, label="After IPW")
    ax.axvline(0.1,  color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(-0.1, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0,    color="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel("Standardized Mean Difference")
    ax.set_title("Covariate Balance (Love Plot)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info("Love plot saved to %s", save_path)


def trim_common_support(
    df: pd.DataFrame,
    propensity: np.ndarray,
    lo: float = 0.05,
    hi: float = 0.95,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Drop rows outside the [lo, hi] propensity range (extreme probability of treatment
    makes the counterfactual implausible — violation of positivity).
    """
    mask = (propensity >= lo) & (propensity <= hi)
    n_dropped = (~mask).sum()
    logger.info("Common-support trim: dropped %d rows (%.1f%%)", n_dropped, 100 * n_dropped / len(df))
    print(f"Common-support trim: dropped {n_dropped} rows ({100 * n_dropped / len(df):.1f}%)")
    return df[mask].reset_index(drop=True), propensity[mask]


if __name__ == "__main__":
    from src.data.synthetic import generate, FEATURE_COLS

    df, true_tau, true_ate = generate()
    prop = estimate_propensity(df, FEATURE_COLS)

    overlap_plot(df, prop)
    smd = standardized_mean_diff(df, FEATURE_COLS)
    print("SMD before trimming:\n", smd.round(3).to_string())

    df_trim, prop_trim = trim_common_support(df, prop)
    smd_after = standardized_mean_diff(df_trim, FEATURE_COLS)
    love_plot(smd, smd_after)
    print("Diagnostics saved to artifacts/")
