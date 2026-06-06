"""
Gate: DML and DR estimators recover the true ATE within tolerance.
Tolerance is set at 15% relative error (generous for a test, tight for a showcase).
"""
import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from src.causal.estimators import DRLearnerEstimator, LinearDMLEstimator, XLearner
from src.config import cfg
from src.data.synthetic import FEATURE_COLS, generate

ATE_TOLERANCE = 0.15   # 15% relative error


@pytest.fixture(scope="module")
def train_test_data():
    df, true_tau, true_ate = generate(n=10_000, seed=cfg.seed)
    X = df[FEATURE_COLS]; T = df["treatment"].values; Y = df["outcome"].values
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.30, random_state=cfg.seed)
    return (X.iloc[tr], T[tr], Y[tr],
            X.iloc[te], T[te], Y[te],
            true_tau[te], true_ate)


def _relative_error(estimated: float, true: float) -> float:
    return abs(estimated - true) / (abs(true) + 1e-9)


def test_linear_dml_ate_recovery(train_test_data):
    X_tr, T_tr, Y_tr, X_te, T_te, Y_te, tau_te, true_ate = train_test_data
    est = LinearDMLEstimator()
    est.fit(X_tr, T_tr, Y_tr)
    recovered = float(est.predict_cate(X_te).mean())
    err = _relative_error(recovered, true_ate)
    assert err < ATE_TOLERANCE, f"LinearDML ATE error {err:.3f} > {ATE_TOLERANCE}"


def test_dr_learner_ate_recovery(train_test_data):
    X_tr, T_tr, Y_tr, X_te, T_te, Y_te, tau_te, true_ate = train_test_data
    est = DRLearnerEstimator()
    est.fit(X_tr, T_tr, Y_tr)
    recovered = float(est.predict_cate(X_te).mean())
    err = _relative_error(recovered, true_ate)
    assert err < ATE_TOLERANCE, f"DRLearner ATE error {err:.3f} > {ATE_TOLERANCE}"


def test_x_learner_cate_correlation(train_test_data):
    """X-Learner should achieve ≥0.9 correlation with true individual CATE."""
    X_tr, T_tr, Y_tr, X_te, T_te, Y_te, tau_te, true_ate = train_test_data
    est = XLearner()
    est.fit(X_tr, T_tr, Y_tr)
    tau_hat = est.predict_cate(X_te)
    corr = float(np.corrcoef(tau_hat, tau_te)[0, 1])
    assert corr >= 0.90, f"X-Learner correlation {corr:.3f} < 0.90"


def test_true_ate_is_positive():
    """Sanity: synthetic DGP should produce a positive population ATE."""
    _, _, true_ate = generate(n=5_000, seed=cfg.seed)
    assert true_ate > 0, f"True ATE {true_ate:.4f} is not positive"
