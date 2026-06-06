"""
Hillstrom MineThatData email experiment loader.

Real A/B campaign data: men's email / women's email / no-email (control).
We expose a binary-treatment view: any-email vs. no-email.

If the download fails, a clear warning is printed and None is returned so the
pipeline continues with synthetic data only — the build is never blocked.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import cfg

logger = logging.getLogger(__name__)

CACHE_PATH = Path("data/hillstrom_raw.csv")
FEATURE_COLS = ["recency", "history", "mens", "womens", "zip_code", "newbie", "channel"]
COVARIATE_COLS = FEATURE_COLS  # alias


def load(force_download: bool = False) -> pd.DataFrame | None:
    """
    Load and clean the Hillstrom dataset.

    Returns a DataFrame with columns:
        covariates + treatment (0/1) + visit (0/1) + conversion (0/1) + spend (float)

    Returns None if the download fails (caller must handle gracefully).
    """
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not CACHE_PATH.exists() or force_download:
        try:
            raw = _download(cfg.data.hillstrom_url)
        except Exception as exc:
            logger.warning("Hillstrom download failed (%s). Proceeding with synthetic only.", exc)
            return None
        raw.to_csv(CACHE_PATH, index=False)
    else:
        raw = pd.read_csv(CACHE_PATH)

    return _clean(raw)


def _download(url: str) -> pd.DataFrame:
    import io
    import ssl
    import urllib.request

    logger.info("Downloading Hillstrom data from %s", url)
    # macOS ships without root certs for urllib; use an unverified context as fallback
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, timeout=30, context=ctx) as resp:
        content = resp.read().decode("latin-1")
    return pd.read_csv(io.StringIO(content))


def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # Binary treatment: any email vs. control
    df["treatment"] = (df["segment"] != "No E-Mail").astype(int)

    # Encode zip_code and channel as integers
    df["zip_code"] = df["zip_code"].map({"Urban": 2, "Suburban": 1, "Rural": 0}).fillna(1).astype(int)
    df["channel"] = df["channel"].map({"Web": 2, "Multichannel": 1, "Phone": 0}).fillna(1).astype(int)

    # Outcomes
    df["visit"] = df["visit"].astype(int)
    df["conversion"] = df["conversion"].astype(int)
    df["spend"] = df["spend"].astype(float)

    keep = FEATURE_COLS + ["treatment", "visit", "conversion", "spend"]
    df = df[keep].dropna()

    # Propensity should be ~2/3 (2 email groups vs 1 control) in the raw RCT
    # No confounding by design (true RCT), but we include covariates for CATE estimation
    return df.reset_index(drop=True)


def ate_estimates(df: pd.DataFrame) -> dict[str, float]:
    """Naive difference-in-means ATE for each outcome (valid because it's an RCT)."""
    treated = df[df["treatment"] == 1]
    control = df[df["treatment"] == 0]
    return {
        "visit":      float(treated["visit"].mean()      - control["visit"].mean()),
        "conversion": float(treated["conversion"].mean() - control["conversion"].mean()),
        "spend":      float(treated["spend"].mean()      - control["spend"].mean()),
    }


if __name__ == "__main__":
    df = load()
    if df is None:
        print("Hillstrom data unavailable — using synthetic only.")
    else:
        print(f"Rows            : {len(df):,}")
        print(f"Treatment rate  : {df['treatment'].mean():.3f}")
        ates = ate_estimates(df)
        print(f"ATE (visit)     : {ates['visit']:.4f}")
        print(f"ATE (conversion): {ates['conversion']:.4f}")
        print(f"ATE (spend)     : {ates['spend']:.4f}")
        print(df.head(3).to_string())
