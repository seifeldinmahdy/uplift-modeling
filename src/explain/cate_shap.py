"""
SHAP explainability for the CATE model.

Uses the LinearDML estimator (best MSE vs true tau) and SHAP's TreeExplainer
on its final-stage model to show which covariates drive treatment effect heterogeneity.

Run: python -m src.explain.cate_shap
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from src.config import cfg

logger = logging.getLogger(__name__)
ARTIFACTS_DIR = Path("artifacts")


def compute_shap_cate(
    estimator,
    X_test: pd.DataFrame,
    estimator_name: str = "CausalForest",
    max_display: int = 10,
    save_path: str = "artifacts/shap_summary.png",
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Compute SHAP values for CATE predictions and save summary plot.

    Returns (shap_values, X_explained) where both have the same number of rows.
    """
    Path(save_path).parent.mkdir(exist_ok=True)

    shap_values, X_plot = _try_tree_explainer(estimator, X_test)

    # Bar importance plot
    fig, _ = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_plot, max_display=max_display, show=False, plot_type="bar")
    plt.title(f"SHAP Feature Importance for CATE — {estimator_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close("all")
    logger.info("SHAP summary saved to %s", save_path)

    # Beeswarm plot
    beeswarm_path = save_path.replace(".png", "_beeswarm.png")
    fig2, _ = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_plot, max_display=max_display, show=False)
    plt.title(f"SHAP Beeswarm — {estimator_name}")
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=130, bbox_inches="tight")
    plt.close("all")

    return shap_values, X_plot


def _try_tree_explainer(estimator, X_test: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Try TreeExplainer on the CausalForestDML internal estimator;
    fall back to KernelExplainer on a sample.
    Returns (shap_values, X_used) — both have matching row counts.
    """
    inner_forest = getattr(estimator, "_model", None)

    if inner_forest is not None:
        try:
            explainer = shap.TreeExplainer(inner_forest)
            sv = explainer.shap_values(X_test)
            if isinstance(sv, list):
                sv = sv[0]
            return sv, X_test
        except Exception:
            pass

    # Fallback: KernelExplainer on a sample — X_plot = same sample
    sample_df = X_test.sample(min(200, len(X_test)), random_state=42)
    background = shap.sample(sample_df, min(50, len(sample_df)))
    explainer = shap.KernelExplainer(
        lambda x: estimator.predict_cate(pd.DataFrame(x, columns=X_test.columns)),
        background,
    )
    sv = explainer.shap_values(sample_df, nsamples=80)
    return sv, sample_df


if __name__ == "__main__":
    from src.causal.estimators import CausalForestEstimator
    from src.data.synthetic import FEATURE_COLS, generate

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    df, true_tau, true_ate = generate()
    X = df[FEATURE_COLS]
    T = df["treatment"].values
    Y = df["outcome"].values

    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=cfg.data.test_size, random_state=cfg.seed)
    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    T_train = T[tr_idx]; Y_train = Y[tr_idx]

    print("Fitting CausalForestDML...")
    est = CausalForestEstimator()
    est.fit(X_train, T_train, Y_train)

    print("Computing SHAP values...")
    sv, X_explained = compute_shap_cate(est, X_test, estimator_name="CausalForest (DML)")

    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X_explained.columns)
    print("\nMean |SHAP| per feature (CATE drivers):")
    print(mean_abs.sort_values(ascending=False).round(4).to_string())
    print("\nSHAP plots saved to artifacts/")
