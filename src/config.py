"""Typed config loader: reads config/config.yaml into a validated Pydantic model."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    n_samples: int = 20_000
    test_size: float = 0.30
    hillstrom_url: str = ""


class TreatmentConfig(BaseModel):
    cost_per_unit: float = 1.0
    budget_fraction: float = 0.30


class EstimatorConfig(BaseModel):
    n_estimators: int = 150
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    min_samples_leaf: int = 20
    dml_cv_folds: int = 3
    causal_forest_n_estimators: int = 200


class RefutationConfig(BaseModel):
    n_simulations: int = 50
    placebo_type: str = "permute"
    subset_fraction: float = 0.80


class SensitivityConfig(BaseModel):
    benchmark_covariate: str = "loyal"
    r2_y_values: list[float] = Field(default_factory=lambda: [0.1, 0.3, 0.5])
    r2_d_values: list[float] = Field(default_factory=lambda: [0.1, 0.3, 0.5])


class PolicyConfig(BaseModel):
    min_uplift_threshold: float = 0.0


class MLflowConfig(BaseModel):
    experiment_name: str = "causal-uplift-engine"
    tracking_uri: str = "mlruns"


class Config(BaseModel):
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    treatment: TreatmentConfig = Field(default_factory=TreatmentConfig)
    estimators: EstimatorConfig = Field(default_factory=EstimatorConfig)
    refutation: RefutationConfig = Field(default_factory=RefutationConfig)
    sensitivity: SensitivityConfig = Field(default_factory=SensitivityConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)


_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> Config:
    """Load and validate config from YAML."""
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    return Config.model_validate(raw or {})


# module-level singleton
cfg = load_config()
