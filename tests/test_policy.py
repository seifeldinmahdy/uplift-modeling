"""
Gate: policy never exceeds budget; causal targeting beats random and response model.
"""
import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from src.causal.estimators import LinearDMLEstimator
from src.config import cfg
from src.data.synthetic import FEATURE_COLS, generate
from src.policy.targeting import net_value_policy, response_model_policy


@pytest.fixture(scope="module")
def policy_data():
    df, true_tau, _ = generate(n=8_000, seed=cfg.seed)
    X = df[FEATURE_COLS]; T = df["treatment"].values; Y = df["outcome"].values
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.30, random_state=cfg.seed)
    X_tr, X_te = X.iloc[tr], X.iloc[te]
    T_tr, Y_tr = T[tr], Y[tr]
    tau_te = true_tau[te]

    est = LinearDMLEstimator()
    est.fit(X_tr, T_tr, Y_tr)
    tau_hat = est.predict_cate(X_te)

    return X_tr, T_tr, Y_tr, X_te, tau_hat, tau_te


def test_policy_respects_budget(policy_data):
    _, _, _, _, tau_hat, tau_te = policy_data
    budget = 0.30
    n = len(tau_hat)
    policy = net_value_policy(tau_hat, tau_te, budget_frac=budget)
    assert policy["n_treated"] <= int(n * budget) + 1, "Policy exceeds budget"


def test_causal_beats_random(policy_data):
    _, _, _, _, tau_hat, tau_te = policy_data
    policy = net_value_policy(tau_hat, tau_te, cost=cfg.treatment.cost_per_unit)
    assert policy["model_net_pv"] > policy["random_net_pv"], (
        f"Causal policy ({policy['model_net_pv']:.4f}) should beat random "
        f"({policy['random_net_pv']:.4f})"
    )


def test_causal_beats_response_model(policy_data):
    X_tr, T_tr, Y_tr, X_te, tau_hat, tau_te = policy_data
    policy  = net_value_policy(tau_hat, tau_te, cost=cfg.treatment.cost_per_unit)
    resp    = response_model_policy(X_tr, Y_tr, T_tr, X_te, tau_te,
                                    cost=cfg.treatment.cost_per_unit)
    assert policy["model_net_pv"] > resp["response_net_pv"], (
        f"Causal ({policy['model_net_pv']:.4f}) should beat response model "
        f"({resp['response_net_pv']:.4f})"
    )


def test_dnd_customers_flagged(policy_data):
    _, _, _, _, tau_hat, tau_te = policy_data
    policy = net_value_policy(tau_hat, tau_te)
    assert policy["n_dnd"] > 0, "No do-not-disturb customers identified — DGP should produce some"
