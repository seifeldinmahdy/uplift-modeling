"""
Meta-learner implementations for CATE estimation.

S-learner : single model on [X, T]
T-learner : separate models for treated / control
X-learner : T-learner → impute effects → model effects, weighted by propensity
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data import FEATURE_COLS


def _default_base():
    return GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, random_state=42)


# ── S-learner ──────────────────────────────────────────────────────────────
class SLearner:
    def __init__(self, base_model=None):
        self.model = base_model or _default_base()

    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray):
        Xt = X.copy()
        Xt["__T__"] = T
        self.model.fit(Xt, Y)
        self._feature_cols = list(Xt.columns)
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        X1 = X.copy(); X1["__T__"] = 1.0
        X0 = X.copy(); X0["__T__"] = 0.0
        return self.model.predict(X1) - self.model.predict(X0)


# ── T-learner ──────────────────────────────────────────────────────────────
class TLearner:
    def __init__(self, base_model=None):
        self.m1 = base_model or _default_base()
        self.m0 = _default_base()

    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray):
        mask1 = T == 1
        self.m1.fit(X[mask1], Y[mask1])
        self.m0.fit(X[~mask1], Y[~mask1])
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        return self.m1.predict(X) - self.m0.predict(X)


# ── X-learner ──────────────────────────────────────────────────────────────
class XLearner:
    def __init__(self, base_model=None):
        self.m1 = base_model or _default_base()
        self.m0 = _default_base()
        self.mx1 = _default_base()
        self.mx0 = _default_base()
        self._prop_model = LogisticRegression(max_iter=500, C=1.0)
        self._scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray):
        mask1 = T == 1
        self.m1.fit(X[mask1], Y[mask1])
        self.m0.fit(X[~mask1], Y[~mask1])

        # impute counterfactual effects
        d1 = Y[mask1] - self.m0.predict(X[mask1])   # treated: Y(1) - hat{Y(0)}
        d0 = self.m1.predict(X[~mask1]) - Y[~mask1]  # control: hat{Y(1)} - Y(0)

        self.mx1.fit(X[mask1], d1)
        self.mx0.fit(X[~mask1], d0)

        # propensity model for weighting
        Xs = self._scaler.fit_transform(X)
        self._prop_model.fit(Xs, T)
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self._scaler.transform(X)
        e = self._prop_model.predict_proba(Xs)[:, 1]
        e = np.clip(e, 0.05, 0.95)
        tau1 = self.mx1.predict(X)
        tau0 = self.mx0.predict(X)
        return e * tau0 + (1 - e) * tau1


# ── EconML (optional) ─────────────────────────────────────────────────────
def try_load_econml():
    """Returns (CausalForestDML class, available=True) or (None, False)."""
    try:
        from econml.dml import CausalForestDML
        return CausalForestDML, True
    except ImportError:
        return None, False


class EconMLWrapper:
    """Thin wrapper so CausalForestDML has the same interface."""
    def __init__(self):
        CausalForestDML, ok = try_load_econml()
        if not ok:
            raise ImportError("econml not available")
        self._model = CausalForestDML(
            n_estimators=200, max_depth=5, min_samples_leaf=20,
            random_state=42, cv=3
        )

    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray):
        self._model.fit(Y, T.astype(float), X=X)
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.effect(X).ravel()


# ── Fit all available learners ────────────────────────────────────────────
def fit_all(X_train, T_train, Y_train):
    learners = {
        "S-Learner": SLearner(),
        "T-Learner": TLearner(),
        "X-Learner": XLearner(),
    }
    _, econml_ok = try_load_econml()
    if econml_ok:
        learners["CausalForest (EconML)"] = EconMLWrapper()

    fitted = {}
    for name, learner in learners.items():
        print(f"  Fitting {name}...", flush=True)
        learner.fit(X_train, T_train, Y_train)
        fitted[name] = learner
    return fitted
