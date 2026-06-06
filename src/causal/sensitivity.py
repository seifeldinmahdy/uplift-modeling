"""
Sensitivity analysis for unobserved confounding.

Two complementary approaches:
1. E-value: minimum confounding strength needed to explain away the observed ATE.
   (VanderWeele & Ding 2017, adapted for continuous outcomes via SMD framing)
2. DoWhy sensitivity: add synthetic unobserved confounders and measure effect stability.
   Uses the sensitivity module from DoWhy when available.

Run: python -m src.causal.sensitivity
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import cfg

logger = logging.getLogger(__name__)
ARTIFACTS_DIR = Path("artifacts")


def evalue_continuous(
    ate: float,
    outcome_sd: float,
    ate_se: float = 0.0,
    null: float = 0.0,
) -> dict[str, float]:
    """
    Approximate E-value for a continuous outcome (VanderWeele & Ding 2017).

    Converts the standardised mean difference (SMD = ATE / outcome_sd) to an
    approximate odds-ratio using the Cox (1970) approximation, then computes the
    E-value: the minimum RR-style association an unmeasured confounder would need
    with both T and Y to fully explain away the effect.

    Parameters
    ----------
    ate        : estimated average treatment effect
    outcome_sd : standard deviation of the outcome (used for Cohen's d)
    ate_se     : standard error of the ATE (used for the CI-bound E-value)
    null       : null effect to test against (default 0)
    """
    # Cohen's d — the proper SMD, not the t-statistic
    d = abs(ate - null) / (outcome_sd + 1e-9)

    # 95% CI lower bound SMD (one-sided towards null)
    d_lower = max(0.0, abs(ate - null) - 1.96 * ate_se) / (outcome_sd + 1e-9)

    # Cox (1970) approximation: OR ≈ exp(1.81 * d)  [bounded to prevent overflow]
    d_clamped      = min(d,       5.0)
    d_lower_clamped = min(d_lower, 5.0)
    or_est  = float(np.exp(1.81 * d_clamped))
    or_null = float(np.exp(1.81 * d_lower_clamped))

    def _evalue_from_or(r: float) -> float:
        if r <= 1.0:
            return 1.0
        return r + np.sqrt(r * (r - 1))

    e_est  = _evalue_from_or(or_est)
    e_null = _evalue_from_or(max(or_null, 1.0))

    return {
        "ate": round(ate, 4),
        "cohens_d": round(d, 4),
        "evalue_estimate": round(e_est, 3),
        "evalue_ci_bound": round(e_null, 3),
        "interpretation": (
            f"An unmeasured confounder associated with both treatment and outcome "
            f"by a risk-ratio of ≥{e_est:.2f} could fully explain away the ATE. "
            f"The CI-bound E-value is {e_null:.2f}. "
            f"(Cohen's d = {d:.2f})"
        ),
    }


def dowhy_sensitivity(
    df: pd.DataFrame,
    feature_cols: list[str],
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    benchmark_covariate: str | None = None,
) -> dict:
    """
    Run DoWhy's sensitivity analysis (LinearSensitivityAnalyzer) if available.
    Falls back gracefully to just returning the E-value if unavailable.
    """
    benchmark_covariate = benchmark_covariate or cfg.sensitivity.benchmark_covariate

    try:
        from dowhy.causal_refuters.linear_sensitivity_analyzer import LinearSensitivityAnalyzer
        from src.causal.graph import build_dowhy_model

        model, estimand = build_dowhy_model(
            df, treatment_col=treatment_col, outcome_col=outcome_col,
            feature_cols=feature_cols,
        )
        estimate = model.estimate_effect(
            estimand,
            method_name="backdoor.linear_regression",
            control_value=0,
            treatment_value=1,
        )

        analyzer = LinearSensitivityAnalyzer(
            estimator=estimate,
            data=df,
            treatment_name=treatment_col,
            outcome_name=outcome_col,
            benchmark_covariate=benchmark_covariate if benchmark_covariate in df.columns else None,
            r2_y_values=cfg.sensitivity.r2_y_values,
            r2_t_values=cfg.sensitivity.r2_d_values,
        )
        sensitivity_results = analyzer.check_sensitivity(plot=False)
        return {"method": "LinearSensitivityAnalyzer", "results": str(sensitivity_results)}

    except Exception as exc:
        logger.info("DoWhy sensitivity analyzer unavailable (%s); using E-value only", exc)
        return {"method": "evalue_fallback", "note": str(exc)}


def run_full_sensitivity(
    ate: float,
    ate_se: float,
    df: pd.DataFrame,
    feature_cols: list[str],
    outcome_col: str = "outcome",
) -> dict:
    """Combine E-value and DoWhy sensitivity into one result dict."""
    outcome_sd = float(df[outcome_col].std())
    ev = evalue_continuous(ate, outcome_sd=outcome_sd, ate_se=ate_se)
    dw = dowhy_sensitivity(df, feature_cols)
    return {"evalue": ev, "dowhy_sensitivity": dw}


if __name__ == "__main__":
    from src.data.synthetic import FEATURE_COLS, generate
    from src.causal.estimators import TLearner
    from sklearn.model_selection import train_test_split

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    df, true_tau, true_ate = generate()
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=cfg.data.test_size, random_state=cfg.seed)
    X_train = df[FEATURE_COLS].iloc[tr_idx]
    T_train = df["treatment"].values[tr_idx]
    Y_train = df["outcome"].values[tr_idx]
    X_test  = df[FEATURE_COLS].iloc[te_idx]

    model = TLearner()
    model.fit(X_train, T_train, Y_train)
    tau_hat = model.predict_cate(X_test)
    ate_est = float(tau_hat.mean())
    ate_se  = float(tau_hat.std() / np.sqrt(len(tau_hat)))

    print(f"Estimated ATE: {ate_est:.4f}  SE: {ate_se:.4f}")

    results = run_full_sensitivity(ate_est, ate_se, df, FEATURE_COLS, outcome_col="outcome")

    ev = results["evalue"]
    print(f"\nE-value (estimate): {ev['evalue_estimate']}")
    print(f"E-value (CI bound): {ev['evalue_ci_bound']}")
    print(f"Interpretation: {ev['interpretation']}")

    with open(ARTIFACTS_DIR / "sensitivity_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved artifacts/sensitivity_results.json")
