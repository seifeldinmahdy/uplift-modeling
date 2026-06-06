"""
CATE estimators: S/T/X meta-learners (sklearn GBM base) + EconML LinearDML,
CausalForestDML, DRLearner. All results logged to MLflow.

Run: python -m src.causal.estimators
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import cfg
from src.data.synthetic import FEATURE_COLS, generate

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")


# ── Base model factories ───────────────────────────────────────────────────
def _gbr() -> GradientBoostingRegressor:
    c = cfg.estimators
    return GradientBoostingRegressor(
        n_estimators=c.n_estimators,
        max_depth=c.max_depth,
        learning_rate=c.learning_rate,
        subsample=c.subsample,
        random_state=cfg.seed,
    )


def _gbc() -> GradientBoostingClassifier:
    c = cfg.estimators
    return GradientBoostingClassifier(
        n_estimators=c.n_estimators,
        max_depth=c.max_depth,
        learning_rate=c.learning_rate,
        subsample=c.subsample,
        random_state=cfg.seed,
    )


# ── Protocol ────────────────────────────────────────────────────────────────
class CATEEstimator(Protocol):
    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> "CATEEstimator": ...
    def predict_cate(self, X: pd.DataFrame) -> np.ndarray: ...


# ── S-Learner ────────────────────────────────────────────────────────────────
class SLearner:
    """Single model on [X, T]; CATE = f(x,1) - f(x,0)."""

    def __init__(self) -> None:
        self._model = _gbr()

    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> "SLearner":
        Xt = X.copy()
        Xt["__T__"] = T
        self._model.fit(Xt, Y)
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        X1, X0 = X.copy(), X.copy()
        X1["__T__"] = 1.0
        X0["__T__"] = 0.0
        return self._model.predict(X1) - self._model.predict(X0)


# ── T-Learner ────────────────────────────────────────────────────────────────
class TLearner:
    """Separate models per arm; CATE = f1(x) - f0(x)."""

    def __init__(self) -> None:
        self._m1 = _gbr()
        self._m0 = _gbr()

    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> "TLearner":
        mask = T == 1
        self._m1.fit(X[mask], Y[mask])
        self._m0.fit(X[~mask], Y[~mask])
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        return self._m1.predict(X) - self._m0.predict(X)


# ── X-Learner ────────────────────────────────────────────────────────────────
class XLearner:
    """
    T-learner → impute individual effects → model them → weight by propensity.
    Better than T-learner when treatment groups are imbalanced.
    """

    def __init__(self) -> None:
        self._m1 = _gbr()
        self._m0 = _gbr()
        self._mx1 = _gbr()
        self._mx0 = _gbr()
        self._prop = LogisticRegression(max_iter=500, C=1.0, random_state=cfg.seed)
        self._scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> "XLearner":
        mask = T == 1
        self._m1.fit(X[mask], Y[mask])
        self._m0.fit(X[~mask], Y[~mask])

        d1 = Y[mask] - self._m0.predict(X[mask])     # treated: Y(1) - hat{Y(0)}
        d0 = self._m1.predict(X[~mask]) - Y[~mask]   # control: hat{Y(1)} - Y(0)

        self._mx1.fit(X[mask], d1)
        self._mx0.fit(X[~mask], d0)

        Xs = self._scaler.fit_transform(X)
        self._prop.fit(Xs, T)
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self._scaler.transform(X)
        e = np.clip(self._prop.predict_proba(Xs)[:, 1], 0.05, 0.95)
        return e * self._mx0.predict(X) + (1 - e) * self._mx1.predict(X)


# ── R-Learner ────────────────────────────────────────────────────────────────
class RLearner:
    """
    Robinson (1988) / Nie & Wager (2021) residual-on-residual learner.
    Regresses (Y - m̂(x)) on (T - ê(x)) * τ(x) — robust to nuisance misspecification.
    """

    def __init__(self) -> None:
        self._outcome_model = _gbr()
        self._prop_model = _gbc()
        self._cate_model = _gbr()
        self._scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> "RLearner":
        Xs = self._scaler.fit_transform(X)
        # Step 1: fit nuisance models
        self._outcome_model.fit(X, Y)
        self._prop_model.fit(Xs, T.astype(int))

        m_hat = self._outcome_model.predict(X)
        e_hat = np.clip(self._prop_model.predict_proba(Xs)[:, 1], 0.05, 0.95)

        # Step 2: residuals
        y_resid = Y - m_hat
        t_resid = T - e_hat

        # Step 3: weight and fit CATE model
        weights = t_resid**2
        pseudo_outcome = y_resid / (t_resid + 1e-9)
        self._cate_model.fit(X, pseudo_outcome, sample_weight=weights)
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        return self._cate_model.predict(X)


# ── EconML wrappers ──────────────────────────────────────────────────────────
class LinearDMLEstimator:
    def __init__(self) -> None:
        from econml.dml import LinearDML
        # LinearDML requires a regressor for model_t (predicts propensity as probability)
        self._model = LinearDML(
            model_y=_gbr(),
            model_t=_gbr(),
            random_state=cfg.seed,
            cv=cfg.estimators.dml_cv_folds,
        )

    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> "LinearDMLEstimator":
        self._model.fit(Y, T.astype(float), X=X)
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.effect(X).ravel()


class CausalForestEstimator:
    def __init__(self) -> None:
        from econml.dml import CausalForestDML
        self._model = CausalForestDML(
            n_estimators=cfg.estimators.causal_forest_n_estimators,
            min_samples_leaf=cfg.estimators.min_samples_leaf,
            random_state=cfg.seed,
            cv=cfg.estimators.dml_cv_folds,
        )

    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> "CausalForestEstimator":
        self._model.fit(Y, T.astype(float), X=X)
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.effect(X).ravel()

    def predict_cate_interval(self, X: pd.DataFrame, alpha: float = 0.10):
        """Return (lower, upper) confidence intervals on CATE."""
        lb, ub = self._model.effect_interval(X, alpha=alpha)
        return lb.ravel(), ub.ravel()


class DRLearnerEstimator:
    def __init__(self) -> None:
        from econml.dr import DRLearner
        self._model = DRLearner(
            model_propensity=_gbc(),
            model_regression=_gbr(),
            model_final=_gbr(),
            random_state=cfg.seed,
            cv=cfg.estimators.dml_cv_folds,
        )

    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> "DRLearnerEstimator":
        self._model.fit(Y, T.astype(int), X=X)
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.effect(X).ravel()


# ── Fit all estimators ────────────────────────────────────────────────────────
def fit_all(
    X_train: pd.DataFrame,
    T_train: np.ndarray,
    Y_train: np.ndarray,
    verbose: bool = True,
) -> dict[str, CATEEstimator]:
    """Fit all available estimators and return a name→estimator dict."""
    estimators: dict[str, CATEEstimator] = {
        "S-Learner":   SLearner(),
        "T-Learner":   TLearner(),
        "X-Learner":   XLearner(),
        "R-Learner":   RLearner(),
        "LinearDML":   LinearDMLEstimator(),
        "CausalForest": CausalForestEstimator(),
        "DRLearner":   DRLearnerEstimator(),
    }

    fitted: dict[str, CATEEstimator] = {}
    for name, est in estimators.items():
        if verbose:
            print(f"  Fitting {name}...", flush=True)
        est.fit(X_train, T_train, Y_train)
        fitted[name] = est
    return fitted


# ── Main: fit + evaluate + log to MLflow ─────────────────────────────────────
def run_estimators(verbose: bool = True) -> dict:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
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

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    if verbose:
        print(f"True ATE: {true_ate:.4f}")
        print("Fitting estimators...")

    fitted = fit_all(X_train, T_train, Y_train, verbose=verbose)

    rows = []
    results = {}
    with mlflow.start_run(run_name="all_estimators"):
        mlflow.log_param("n_train", len(tr_idx))
        mlflow.log_param("true_ate", round(true_ate, 4))

        for name, est in fitted.items():
            tau_hat = est.predict_cate(X_test)
            rec_ate = float(tau_hat.mean())
            corr    = float(np.corrcoef(tau_hat, tau_test)[0, 1])
            mse     = float(np.mean((tau_hat - tau_test) ** 2))

            mlflow.log_metrics({
                f"{name}_recovered_ate": round(rec_ate, 4),
                f"{name}_corr_true_tau": round(corr, 4),
                f"{name}_mse_true_tau":  round(mse, 4),
            })

            row = {
                "Estimator":     name,
                "Recovered ATE": round(rec_ate, 4),
                "True ATE":      round(true_ate, 4),
                "ATE Error":     round(abs(rec_ate - true_ate), 4),
                "Corr(τ̂,τ)":    round(corr, 4),
                "MSE(τ̂,τ)":     round(mse, 4),
            }
            rows.append(row)
            results[name] = {"tau_hat": tau_hat, **row}

            if verbose:
                print(f"  {name:20s}  ATE={rec_ate:.4f}  corr={corr:.3f}  mse={mse:.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(ARTIFACTS_DIR / "estimator_comparison.csv", index=False)
    if verbose:
        print("\nEstimator comparison:")
        print(table.to_string(index=False))

    return {
        "fitted": fitted,
        "results": results,
        "true_ate": true_ate,
        "tau_test": tau_test,
        "X_test": X_test,
        "T_test": T_test,
        "Y_test": Y_test,
        "X_train": X_train,
        "T_train": T_train,
        "Y_train": Y_train,
    }


if __name__ == "__main__":
    run_estimators(verbose=True)
