"""Rule-based anomaly checks."""

from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt

from .config import DEFAULT_PIPELINE_CONFIG, DEFAULT_RULE_CONFIG, PipelineConfig, RuleConfig


RULE_COLUMNS = ["rule_high_amount", "rule_risky_country", "rule_risky_category", "rule_velocity"]


def apply_rules(
    df: pd.DataFrame,
    rule_config: RuleConfig = DEFAULT_RULE_CONFIG,
    pipeline_config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
) -> pd.DataFrame:
    """Flag transactions using simple heuristic rules."""

    df = df.copy()
    amount = pipeline_config.amount_column
    country = pipeline_config.geography_column
    category = pipeline_config.category_column
    timestamp = pipeline_config.timestamp_column
    customer = pipeline_config.customer_column

    # Normalize dtypes to avoid runtime errors when inputs are strings or mixed types
    if timestamp in df.columns and not pdt.is_datetime64_any_dtype(df[timestamp]):
        df[timestamp] = pd.to_datetime(df[timestamp], errors="coerce")
    df[timestamp] = df[timestamp].fillna(pd.Timestamp(0))
    df[category] = df[category].astype(str)

    df["rule_high_amount"] = df[amount] > rule_config.high_amount_threshold
    df["rule_risky_country"] = df[country].isin(rule_config.risky_countries)
    df["rule_risky_category"] = df[category].str.lower().isin([c.lower() for c in rule_config.risky_categories])

    df.sort_values(by=[customer, timestamp], inplace=True)
    df["time_diff_minutes"] = df.groupby(customer)[timestamp].diff().dt.total_seconds().div(60).fillna(float("inf"))
    df["rule_velocity"] = df["time_diff_minutes"] < rule_config.velocity_time_minutes
    df.drop(columns=["time_diff_minutes"], inplace=True)

    df["rules_triggered"] = df[RULE_COLUMNS].sum(axis=1)
    return df
