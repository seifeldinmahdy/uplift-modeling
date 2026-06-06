"""
DAG construction and DoWhy causal model / identification.

Assumed DAG
-----------
  X → T       (confounding: covariates drive treatment)
  X → Y       (covariates drive outcome)
  T → Y       (treatment effect — what we want to estimate)
  U ⇢ T, Y    (unobserved confounders — acknowledged, later sensitivity-tested)

Identification strategy: backdoor adjustment on observed X (conditional ignorability).
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_dowhy_model(
    df: pd.DataFrame,
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    feature_cols: list[str] | None = None,
) -> tuple:
    """
    Build and return (causal_model, identified_estimand).

    Parameters
    ----------
    df            : DataFrame containing covariates, treatment, outcome
    treatment_col : name of the binary treatment column
    outcome_col   : name of the continuous/binary outcome column
    feature_cols  : list of observed covariate names (common causes)
    """
    try:
        from dowhy import CausalModel
    except ImportError as e:
        raise ImportError("dowhy is required for causal graph construction") from e

    if feature_cols is None:
        exclude = {treatment_col, outcome_col, "propensity_true", "propensity_hat"}
        feature_cols = [c for c in df.columns if c not in exclude]

    # Build graph as a networkx DiGraph to avoid GML parsing quirks
    import networkx as nx

    G = nx.DiGraph()
    all_nodes = feature_cols + [treatment_col, outcome_col]
    G.add_nodes_from(all_nodes)
    for col in feature_cols:
        G.add_edge(col, treatment_col)
        G.add_edge(col, outcome_col)
    G.add_edge(treatment_col, outcome_col)

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Constructed graph is not a DAG — check feature_cols for duplicates")

    model = CausalModel(
        data=df,
        treatment=treatment_col,
        outcome=outcome_col,
        graph=G,
        logging_level=logging.WARNING,
    )

    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    logger.info("Identified estimand:\n%s", estimand)
    return model, estimand
