"""
Gate: placebo effect collapses toward zero; estimate is stable under random common cause.
"""
import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from src.causal.refute import manual_placebo_refutation
from src.config import cfg
from src.data.synthetic import FEATURE_COLS, generate


@pytest.fixture(scope="module")
def split_data():
    df, _, true_ate = generate(n=8_000, seed=cfg.seed)
    X = df[FEATURE_COLS]; T = df["treatment"].values; Y = df["outcome"].values
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.30, random_state=cfg.seed)
    return X.iloc[tr], T[tr], Y[tr], X.iloc[te], true_ate


def test_placebo_collapses_to_zero(split_data):
    """Shuffled-T ATE should be within ±0.3 of zero."""
    X_tr, T_tr, Y_tr, X_te, true_ate = split_data
    result = manual_placebo_refutation(X_tr, T_tr, Y_tr, X_te, n_runs=3, seed=cfg.seed)
    placebo_ate = result["mean"]
    assert abs(placebo_ate) < 0.3, (
        f"Placebo ATE {placebo_ate:.4f} too large — may indicate spurious signal"
    )


def test_placebo_much_smaller_than_true(split_data):
    """Placebo ATE should be at least 5× smaller than true ATE in absolute value."""
    X_tr, T_tr, Y_tr, X_te, true_ate = split_data
    result = manual_placebo_refutation(X_tr, T_tr, Y_tr, X_te, n_runs=3, seed=cfg.seed)
    ratio = abs(true_ate) / (abs(result["mean"]) + 1e-6)
    assert ratio >= 5.0, (
        f"Signal ratio {ratio:.1f}× too low (placebo={result['mean']:.4f}, true={true_ate:.4f})"
    )
