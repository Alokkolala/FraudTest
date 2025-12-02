"""Feature engineering utilities for transaction data."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .config import DEFAULT_PIPELINE_CONFIG, PipelineConfig


def add_history_features(df: pd.DataFrame, config: PipelineConfig = DEFAULT_PIPELINE_CONFIG) -> pd.DataFrame:
    """Add rolling statistics per customer to encode transaction history."""

    df = df.copy()
    df.sort_values(by=[config.customer_column, config.timestamp_column], inplace=True)
    group = df.groupby(config.customer_column)
    rolling_amount = group[config.amount_column].rolling(config.history_window, min_periods=1)

    df["hist_amount_mean"] = rolling_amount.mean().reset_index(level=0, drop=True)
    df["hist_amount_std"] = rolling_amount.std(ddof=0).reset_index(level=0, drop=True).fillna(0.0)
    df["hist_amount_max"] = rolling_amount.max().reset_index(level=0, drop=True)
    df["hist_count"] = group.cumcount()

    # Velocity: transactions in last N minutes
    df["timestamp_unix"] = df[config.timestamp_column].astype("int64") // 10**9
    window_seconds = config.history_window * 3600
    df["hist_velocity"] = (
        group["timestamp_unix"].transform(lambda x: x.diff().fillna(0).rolling(config.history_window, min_periods=1).apply(lambda s: (s <= window_seconds).sum()))
    )

    df.drop(columns=["timestamp_unix"], inplace=True)
    return df


def build_preprocess_pipeline(df: pd.DataFrame, config: PipelineConfig = DEFAULT_PIPELINE_CONFIG) -> Pipeline:
    """Create a preprocessing pipeline with imputation and one-hot encoding."""

    timestamp_col = config.timestamp_column
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["hour"] = df[timestamp_col].dt.hour
    df["dayofweek"] = df[timestamp_col].dt.dayofweek

    numeric_features = [
        config.amount_column,
        "hist_amount_mean",
        "hist_amount_std",
        "hist_amount_max",
        "hist_count",
        "hist_velocity",
        "hour",
        "dayofweek",
    ]
    categorical_features = [config.geography_column, config.category_column]

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocess
