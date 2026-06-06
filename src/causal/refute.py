"""
DoWhy refutation suite: placebo treatment, random common cause, data subset.

These tests are the "I'm not fooling myself" layer:
  - Placebo: shuffling T should collapse the estimated effect toward zero
  - Random common cause: adding a random noise covariate shouldn't change the estimate
  - Data subset: estimate on 80% of data should stay within a tolerance of full-data estimate

Run: python -m src.causal.refute
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


def run_dowhy_refutations(
    df: pd.DataFrame,
    feature_cols: list[str],
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    n_simulations: int | None = None,
) -> dict[str, dict]:
    """
    Run the full DoWhy refutation suite and return results dict.

    Refuters
    --------
    placebo_treatment   : permute T, refit → effect should vanish
    random_common_cause : add random noise covariate → estimate should be stable
    data_subset         : fit on a random subset → estimate should stay close
    """
    from src.causal.graph import build_dowhy_model

    n_sim = n_simulations or cfg.refutation.n_simulations
    model, estimand = build_dowhy_model(df, treatment_col=treatment_col,
                                        outcome_col=outcome_col,
                                        feature_cols=feature_cols)

    # Get a reference DoWhy estimate (linear regression for speed in refutation)
    ref_estimate = model.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression",
        control_value=0,
        treatment_value=1,
    )
    ref_ate = float(ref_estimate.value)
    logger.info("DoWhy reference ATE (linear regression): %.4f", ref_ate)

    results = {}

    # 1) Placebo treatment
    print("  Refuting: placebo treatment...", flush=True)
    try:
        placebo = model.refute_estimate(
            estimand,
            ref_estimate,
            method_name="placebo_treatment_refuter",
            placebo_type=cfg.refutation.placebo_type,
            num_simulations=n_sim,
        )
        results["placebo_treatment"] = {
            "original_ate": ref_ate,
            "new_effect":   float(placebo.new_effect),
            "p_value":      float(placebo.refutation_result.get("p_value", float("nan"))),
            "passed":       abs(float(placebo.new_effect)) < abs(ref_ate) * 0.3,
        }
        print(f"    Placebo ATE: {placebo.new_effect:.4f}  (original: {ref_ate:.4f})")
    except Exception as exc:
        logger.warning("Placebo refuter failed: %s", exc)
        results["placebo_treatment"] = {"error": str(exc)}

    # 2) Random common cause
    print("  Refuting: random common cause...", flush=True)
    try:
        rcc = model.refute_estimate(
            estimand,
            ref_estimate,
            method_name="random_common_cause",
            num_simulations=n_sim,
        )
        results["random_common_cause"] = {
            "original_ate": ref_ate,
            "new_effect":   float(rcc.new_effect),
            "passed":       abs(float(rcc.new_effect) - ref_ate) < abs(ref_ate) * 0.2,
        }
        print(f"    RCC ATE: {rcc.new_effect:.4f}  (original: {ref_ate:.4f})")
    except Exception as exc:
        logger.warning("Random common cause refuter failed: %s", exc)
        results["random_common_cause"] = {"error": str(exc)}

    # 3) Data subset
    print("  Refuting: data subset...", flush=True)
    try:
        subset = model.refute_estimate(
            estimand,
            ref_estimate,
            method_name="data_subset_refuter",
            subset_fraction=cfg.refutation.subset_fraction,
            num_simulations=n_sim,
        )
        results["data_subset"] = {
            "original_ate": ref_ate,
            "new_effect":   float(subset.new_effect),
            "passed":       abs(float(subset.new_effect) - ref_ate) < abs(ref_ate) * 0.2,
        }
        print(f"    Subset ATE: {subset.new_effect:.4f}  (original: {ref_ate:.4f})")
    except Exception as exc:
        logger.warning("Data subset refuter failed: %s", exc)
        results["data_subset"] = {"error": str(exc)}

    return results


def manual_placebo_refutation(
    X_train: pd.DataFrame,
    T_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: pd.DataFrame,
    n_runs: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """
    Fast permutation placebo: shuffle T, refit TLearner, report collapsed ATE.
    Used as the primary test gate (faster than DoWhy's full refutation).
    """
    from src.causal.estimators import TLearner

    rng = np.random.default_rng(seed)
    placebo_ates = []
    for _ in range(n_runs):
        T_shuffled = rng.permutation(T_train)
        m = TLearner()
        m.fit(X_train, T_shuffled, Y_train)
        placebo_ates.append(float(m.predict_cate(X_test).mean()))

    return {
        "mean": float(np.mean(placebo_ates)),
        "std": float(np.std(placebo_ates)),
        "values": placebo_ates,
    }


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from src.data.synthetic import FEATURE_COLS, generate

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    df, true_tau, true_ate = generate()
    print(f"True ATE: {true_ate:.4f}")

    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=cfg.data.test_size, random_state=cfg.seed)
    X_train = df[FEATURE_COLS].iloc[tr_idx]
    T_train = df["treatment"].values[tr_idx]
    Y_train = df["outcome"].values[tr_idx]
    X_test  = df[FEATURE_COLS].iloc[te_idx]

    # Fast permutation placebo
    print("\nFast permutation placebo (T-Learner):")
    placebo = manual_placebo_refutation(X_train, T_train, Y_train, X_test)
    print(f"  Placebo ATE mean={placebo['mean']:.4f}  std={placebo['std']:.4f}  (expected ≈0)")

    # DoWhy suite
    print("\nDoWhy refutation suite:")
    results = run_dowhy_refutations(
        df, FEATURE_COLS,
        n_simulations=cfg.refutation.n_simulations,
    )

    for name, r in results.items():
        status = "PASS" if r.get("passed", False) else "FAIL"
        if "error" not in r:
            print(f"  [{status}] {name}: new_effect={r['new_effect']:.4f}  original={r['original_ate']:.4f}")
        else:
            print(f"  [ERROR] {name}: {r['error']}")

    results["permutation_placebo"] = placebo
    with open(ARTIFACTS_DIR / "refutation_results.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "values"}
                   for k, v in results.items()}, f, indent=2, default=str)
    print("\nSaved artifacts/refutation_results.json")
