"""Pydantic schemas for the FastAPI scoring endpoint."""
from __future__ import annotations

from pydantic import BaseModel, Field


class UpliftRequest(BaseModel):
    age: float = Field(..., ge=18, le=100, description="Customer age in years")
    income: float = Field(..., gt=0, description="Annual income")
    recency: float = Field(..., ge=0, description="Days since last purchase")
    channel: int = Field(..., ge=0, le=1, description="1=online, 0=offline")
    loyal: int = Field(..., ge=0, le=1, description="1=loyalty member, 0=not")
    frequency: float = Field(..., ge=0, description="Number of past purchases")
    tenure: float = Field(..., ge=0, description="Customer tenure in years")


class UpliftResponse(BaseModel):
    cate: float = Field(..., description="Estimated conditional average treatment effect")
    treat_decision: bool = Field(..., description="True if CATE > treatment cost threshold")
    expected_incremental_value: float = Field(..., description="CATE minus treatment cost")
    do_not_disturb: bool = Field(..., description="True if CATE < 0 (treatment likely harmful)")
    estimator: str = Field(..., description="Name of the estimator used")
