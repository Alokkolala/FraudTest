"""Data loading and synthetic generation utilities."""

from __future__ import annotations

import io
import random
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_PIPELINE_CONFIG, PipelineConfig


_TIMESTAMP_CANDIDATES = [
    "timestamp",
    "time",
    "datetime",
    "date",
    "event_time",
    "event_timestamp",
    "transaction_time",
    "transaction_timestamp",
    "tx_datetime",
    "tx_timestamp",
    "step",
]

# Common alternative column names for transaction schemas such as PaySim
_CATEGORY_CANDIDATES = ["category", "type"]
_CUSTOMER_CANDIDATES = ["customer_id", "nameOrig", "customer"]
_LABEL_CANDIDATES = ["is_fraud", "isFraud", "fraud", "label"]


def _normalize_timestamp_column(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Rename or synthesize a timestamp column to the expected name.

    - If a matching timestamp-like column exists (case-insensitive), it is
      renamed to ``target``.
    - If the dataset uses the PaySim-style ``step`` column (hours since start),
      a synthetic datetime is created from that column.
    - If no match is found, a ValueError lists available columns to guide the
      user.
    """

    if target in df.columns:
        return df

    lower_map = {c.lower(): c for c in df.columns}
    for name in _TIMESTAMP_CANDIDATES:
        if name.lower() in lower_map:
            source = lower_map[name.lower()]
            if name.lower() == "step":
                # PaySim datasets store hours since a reference start time.
                base_time = pd.Timestamp("2025-01-01")
                df[target] = base_time + pd.to_timedelta(df[source].astype(float), unit="h")
                return df
            return df.rename(columns={source: target})

    available = ", ".join(df.columns)
    raise ValueError(
        f"Timestamp column '{target}' not found. "
        f"Available columns: {available}. "
        "Use a column named 'timestamp' or one of the common alternatives "
        "(datetime, event_time, transaction_time, step), or rename the column before loading."
    )


def _standardize_schema(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Align common transaction schemas to the internal expected column names.

    This adds or renames columns for datasets like PaySim that use fields such
    as ``step``, ``type``, ``nameOrig``, and ``isFraud``. Missing optional
    fields (e.g., country) are filled with placeholder values so downstream
    preprocessing and rules remain consistent.
    """

    df = df.copy()

    # Timestamp normalization first so dependent operations can rely on the column.
    df = _normalize_timestamp_column(df, config.timestamp_column)

    # Amount: allow case-insensitive match if the expected column is missing.
    if config.amount_column not in df.columns:
        lower_map = {c.lower(): c for c in df.columns}
        if config.amount_column.lower() in lower_map:
            df = df.rename(columns={lower_map[config.amount_column.lower()]: config.amount_column})

    # Category / operation type mapping.
    if config.category_column not in df.columns:
        for cand in _CATEGORY_CANDIDATES:
            if cand in df.columns:
                df = df.rename(columns={cand: config.category_column})
                break
    if config.category_column not in df.columns:
        df[config.category_column] = "unknown"

    # Customer identifier mapping.
    if config.customer_column not in df.columns:
        for cand in _CUSTOMER_CANDIDATES:
            if cand in df.columns:
                df = df.rename(columns={cand: config.customer_column})
                break
    if config.customer_column not in df.columns:
        df[config.customer_column] = "unknown_customer"

    # Geography is optional in some public datasets; ensure presence.
    if config.geography_column not in df.columns:
        df[config.geography_column] = "unknown"

    # Label mapping for supervised evaluation paths.
    if config.label_column not in df.columns:
        for cand in _LABEL_CANDIDATES:
            if cand in df.columns:
                df = df.rename(columns={cand: config.label_column})
                break

    # Transaction ID for exporting results.
    if config.transaction_id_column not in df.columns:
        df[config.transaction_id_column] = [f"txn_{i:07d}" for i in range(len(df))]

    return df


def load_transactions(path: str | Path, config: PipelineConfig = DEFAULT_PIPELINE_CONFIG) -> pd.DataFrame:
    """Load transactions from CSV or zipped CSV.

    The loader infers compression from the file name and ensures that the
    timestamp column is parsed as datetime.
    """

    path = Path(path)
    compression = None
    if path.suffix == ".zip":
        compression = "zip"
    df = pd.read_csv(path, compression=compression)
    df = _standardize_schema(df, config)
    df[config.timestamp_column] = pd.to_datetime(df[config.timestamp_column])
    return df


def unzip_and_load(zip_path: str | Path, inner_csv: Optional[str] = None, config: PipelineConfig = DEFAULT_PIPELINE_CONFIG) -> pd.DataFrame:
    """Extract a CSV from a ZIP archive and load it into a DataFrame."""

    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        name = inner_csv or zf.namelist()[0]
        with zf.open(name) as handle:
            content = handle.read()
    df = pd.read_csv(io.BytesIO(content))
    df = _standardize_schema(df, config)
    df[config.timestamp_column] = pd.to_datetime(df[config.timestamp_column])
    return df


def _random_geo() -> str:
    return random.choice(["US", "GB", "DE", "FR", "ES", "RU", "CN", "BR", "IN", "ZA", "AE"])


def _random_category() -> str:
    return random.choice([
        "retail",
        "grocery",
        "electronics",
        "travel",
        "gambling",
        "cryptocurrency",
        "services",
    ])


def generate_synthetic_transactions(
    n_rows: int = 5000,
    fraud_rate: float = 0.03,
    config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic transaction dataset with simple fraud patterns."""

    rng = np.random.default_rng(random_state)
    customer_ids = [f"C{rng.integers(1, 4000):05d}" for _ in range(n_rows)]
    base_time = pd.Timestamp.now().normalize()

    timestamps = [base_time + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24))) for _ in range(n_rows)]
    amounts = rng.normal(loc=120.0, scale=80.0, size=n_rows).clip(min=1.0)
    countries = [_random_geo() for _ in range(n_rows)]
    categories = [_random_category() for _ in range(n_rows)]

    fraud_flags = rng.random(n_rows) < fraud_rate

    # Inject anomalies: very high amounts, risky geos/categories, and bursts
    for i in range(n_rows):
        if fraud_flags[i]:
            amounts[i] *= rng.uniform(4, 12)
            if rng.random() < 0.5:
                countries[i] = random.choice(["RU", "IR", "KP", "SY", "NG", "UA"])
            if rng.random() < 0.5:
                categories[i] = random.choice(["cryptocurrency", "gambling"])

    df = pd.DataFrame(
        {
            config.transaction_id_column: [f"T{i:07d}" for i in range(n_rows)],
            config.customer_column: customer_ids,
            config.timestamp_column: timestamps,
            config.amount_column: amounts,
            config.geography_column: countries,
            config.category_column: categories,
            config.label_column: fraud_flags.astype(int),
        }
    )

    df.sort_values(by=[config.customer_column, config.timestamp_column], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
