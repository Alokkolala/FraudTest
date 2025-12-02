"""Simple forecasting utilities for 7-day fraud risk."""

from __future__ import annotations

import pandas as pd

from .config import DEFAULT_PIPELINE_CONFIG, PipelineConfig
from .models import AnomalyModel
from .rules import apply_rules


def forecast_next_week(
    model: AnomalyModel,
    df: pd.DataFrame,
    pipeline_config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
    threshold: float = 0.6,
):
    """Estimate per-customer fraud probability for the next 7 days using recent scores."""

    df_rules = apply_rules(df, pipeline_config=pipeline_config)
    scores = model.predict_scores(df_rules)
    df_rules["fraud_score"] = scores

    grouped = (
        df_rules.groupby(pipeline_config.customer_column)["fraud_score"]
        .agg(["mean", "max", "count"])
        .rename(columns={"mean": "avg_score", "max": "max_score", "count": "tx_count"})
    )
    grouped["prob_next_7d"] = grouped[["avg_score", "max_score"]].mean(axis=1)
    grouped["flagged_now"] = grouped["max_score"] >= threshold
    return grouped.reset_index()
