"""Feature engineering utilities for transaction data."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .config import DEFAULT_PIPELINE_CONFIG, PipelineConfig


def add_time_features(df: pd.DataFrame, config: PipelineConfig = DEFAULT_PIPELINE_CONFIG) -> pd.DataFrame:
    """Ensure timestamp is datetime and append hour/dayofweek columns."""

    df = df.copy()
    df[config.timestamp_column] = pd.to_datetime(df[config.timestamp_column])
    df["hour"] = df[config.timestamp_column].dt.hour
    df["dayofweek"] = df[config.timestamp_column].dt.dayofweek
    return df


def add_history_features(df: pd.DataFrame, config: PipelineConfig = DEFAULT_PIPELINE_CONFIG) -> pd.DataFrame:
    """Add rolling statistics per customer to encode transaction history."""

    df = df.copy()
    df[config.timestamp_column] = pd.to_datetime(df[config.timestamp_column])
    df.sort_values(by=[config.customer_column, config.timestamp_column], inplace=True)

    group = df.groupby(config.customer_column)
    rolling_amount = group[config.amount_column].rolling(config.history_window, min_periods=1)

    df["hist_amount_mean"] = rolling_amount.mean().reset_index(level=0, drop=True)
    df["hist_amount_std"] = rolling_amount.std(ddof=0).reset_index(level=0, drop=True).fillna(0.0)
    df["hist_amount_max"] = rolling_amount.max().reset_index(level=0, drop=True)
    df["hist_count"] = group.cumcount()

    # Velocity: count of transactions per customer in the last ``history_window`` hours
    window_seconds = config.history_window * 3600

    def _velocity(series: pd.Series) -> pd.Series:
        timestamps = (pd.to_datetime(series).astype("int64") // 10**9).to_numpy()
        counts = []
        start = 0
        for i, t in enumerate(timestamps):
            while start < len(timestamps) and t - timestamps[start] > window_seconds:
                start += 1
            counts.append(i - start + 1)
        return pd.Series(counts, index=series.index, dtype=float)

    df["hist_velocity"] = group[config.timestamp_column].transform(_velocity)

    # Ensure time-of-day features exist for downstream preprocessing
    df = add_time_features(df, config)
    return df


def build_preprocess_pipeline(df: pd.DataFrame, config: PipelineConfig = DEFAULT_PIPELINE_CONFIG) -> Pipeline:
    """Create a preprocessing pipeline with imputation and one-hot encoding."""

    df = add_time_features(df, config)

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
