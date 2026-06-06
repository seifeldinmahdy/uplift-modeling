"""
FastAPI scoring endpoint for real-time CATE prediction.

Endpoints
---------
GET  /health             → liveness check
POST /uplift             → CATE prediction + treat decision + incremental value

The estimator is loaded once at startup from a pre-fit artifact.
If no artifact exists, the model is fit on startup (slower first request).
"""
from __future__ import annotations

import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import UpliftRequest, UpliftResponse
from src.config import cfg
from src.data.synthetic import FEATURE_COLS

logger = logging.getLogger(__name__)

ARTIFACT_PATH = Path("artifacts/api_estimator.pkl")
_state: dict = {}


def _train_and_cache() -> object:
    """Fit LinearDML on synthetic data and pickle the estimator."""
    logger.info("Fitting estimator for API (no artifact found)...")
    from sklearn.model_selection import train_test_split

    from src.causal.estimators import LinearDMLEstimator
    from src.data.synthetic import generate

    df, _, _ = generate()
    X = df[FEATURE_COLS]; T = df["treatment"].values; Y = df["outcome"].values
    idx = np.arange(len(df))
    tr_idx, _ = train_test_split(idx, test_size=cfg.data.test_size, random_state=cfg.seed)
    est = LinearDMLEstimator()
    est.fit(X.iloc[tr_idx], T[tr_idx], Y[tr_idx])
    ARTIFACT_PATH.parent.mkdir(exist_ok=True)
    with open(ARTIFACT_PATH, "wb") as f:
        pickle.dump(est, f)
    return est


@asynccontextmanager
async def lifespan(app: FastAPI):
    if ARTIFACT_PATH.exists():
        with open(ARTIFACT_PATH, "rb") as f:
            _state["estimator"] = pickle.load(f)
        logger.info("Loaded estimator from %s", ARTIFACT_PATH)
    else:
        _state["estimator"] = _train_and_cache()
    yield
    _state.clear()


app = FastAPI(
    title="Causal Uplift Engine",
    description="Real-time CATE prediction for budget-constrained targeting",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "estimator": "LinearDML"}


@app.post("/uplift", response_model=UpliftResponse)
def predict_uplift(request: UpliftRequest) -> UpliftResponse:
    est = _state.get("estimator")
    if est is None:
        raise HTTPException(status_code=503, detail="Estimator not loaded")

    X = pd.DataFrame([request.model_dump()], columns=FEATURE_COLS)
    cate = float(est.predict_cate(X)[0])
    cost = cfg.treatment.cost_per_unit
    net  = cate - cost

    return UpliftResponse(
        cate=round(cate, 4),
        treat_decision=net > 0,
        expected_incremental_value=round(net, 4),
        do_not_disturb=cate < 0,
        estimator="LinearDML",
    )
