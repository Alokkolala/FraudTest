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
    if config.timestamp_column in df.columns:
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
    if config.timestamp_column in df.columns:
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
