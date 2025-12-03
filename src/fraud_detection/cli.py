"""Command line interface for training, scoring, forecasting, and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .config import DEFAULT_MODEL_CONFIG, DEFAULT_PIPELINE_CONFIG, DEFAULT_RULE_CONFIG
from .models import AnomalyModel
from .data import generate_synthetic_transactions, load_transactions
from .forecast import forecast_next_week
from .pipeline import evaluate_model, export_results, score_transactions, train_model

app = typer.Typer(help="Fraud detection prototype utilities")


@app.command()
def generate_synthetic(output: str = "data/synthetic.csv", rows: int = 5000, fraud_rate: float = 0.03):
    """Generate a synthetic dataset for experimentation."""

    df = generate_synthetic_transactions(n_rows=rows, fraud_rate=fraud_rate, config=DEFAULT_PIPELINE_CONFIG)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    typer.echo(f"Saved synthetic dataset to {output} with {len(df)} rows")


@app.command()
def train(
    input_csv: Optional[str] = None,
    model_path: str = "artifacts/model.pkl",
    limit_rows: Optional[int] = typer.Option(None, help="Read only the first N rows from the CSV for quick tests."),
    sample_frac: Optional[float] = typer.Option(None, help="Randomly sample a fraction of rows (e.g., 0.1 for 10%)."),
    chunk_size: Optional[int] = typer.Option(None, help="Stream the CSV in chunks to train on large files without loading all rows."),
):
    """Train the anomaly model. If no input is provided, synthetic data is used."""

    if input_csv:
        df = load_transactions(
            input_csv,
            config=DEFAULT_PIPELINE_CONFIG,
            limit_rows=limit_rows,
            sample_frac=sample_frac,
            chunk_size=chunk_size,
        )
    else:
        df = generate_synthetic_transactions(config=DEFAULT_PIPELINE_CONFIG)
    model = train_model(df, model_config=DEFAULT_MODEL_CONFIG, pipeline_config=DEFAULT_PIPELINE_CONFIG)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model, model_path)
    typer.echo(f"Model saved to {model_path}")


@app.command()
def score(
    input_csv: str,
    output_csv: str = "data/results.csv",
    model_path: Optional[str] = None,
    threshold: float = 0.6,
    compute_importance: bool = False,
    limit_rows: Optional[int] = typer.Option(None, help="Read only the first N rows for a quick smoke test."),
    sample_frac: Optional[float] = typer.Option(None, help="Randomly sample a fraction of rows before scoring."),
    chunk_size: Optional[int] = typer.Option(None, help="Stream and sample large CSVs to keep memory usage low."),
):
    """Score transactions and export fraud results."""

    df = load_transactions(
        input_csv,
        config=DEFAULT_PIPELINE_CONFIG,
        limit_rows=limit_rows,
        sample_frac=sample_frac,
        chunk_size=chunk_size,
    )
    if model_path:
        import joblib

        model = joblib.load(model_path)
    else:
        model = train_model(pipeline_config=DEFAULT_PIPELINE_CONFIG)

    results = score_transactions(
        model,
        df,
        threshold=threshold,
        rule_config=DEFAULT_RULE_CONFIG,
        pipeline_config=DEFAULT_PIPELINE_CONFIG,
        compute_importance=compute_importance,
    )
    export_results(results, output_csv)
    typer.echo(f"Saved scoring results to {output_csv}")


@app.command()
def evaluate(
    input_csv: Optional[str] = None,
    threshold: float = 0.6,
    limit_rows: Optional[int] = typer.Option(None, help="Read only the first N rows for validation."),
    sample_frac: Optional[float] = typer.Option(None, help="Randomly sample a fraction of rows before evaluation."),
    chunk_size: Optional[int] = typer.Option(None, help="Stream large CSVs while sampling."),
):
    """Evaluate the model using ROC-AUC, PR-AUC, recall@10%, and flagged rate."""

    if input_csv:
        df = load_transactions(
            input_csv,
            config=DEFAULT_PIPELINE_CONFIG,
            limit_rows=limit_rows,
            sample_frac=sample_frac,
            chunk_size=chunk_size,
        )
    else:
        df = generate_synthetic_transactions(config=DEFAULT_PIPELINE_CONFIG)
    model = AnomalyModel(DEFAULT_MODEL_CONFIG, DEFAULT_PIPELINE_CONFIG)
    metrics = evaluate_model(model, df, threshold=threshold, pipeline_config=DEFAULT_PIPELINE_CONFIG)
    typer.echo(json.dumps(metrics, indent=2))


@app.command()
def forecast(
    input_csv: str,
    model_path: Optional[str] = None,
    threshold: float = 0.6,
    output_csv: str = "data/forecast.csv",
    limit_rows: Optional[int] = typer.Option(None, help="Read only the first N rows for fast forecasting."),
    sample_frac: Optional[float] = typer.Option(None, help="Randomly sample rows before forecasting."),
    chunk_size: Optional[int] = typer.Option(None, help="Stream large CSVs while sampling."),
):
    """Predict fraud propensity per customer for the next 7 days (naive baseline)."""

    df = load_transactions(
        input_csv,
        config=DEFAULT_PIPELINE_CONFIG,
        limit_rows=limit_rows,
        sample_frac=sample_frac,
        chunk_size=chunk_size,
    )
    if model_path:
        import joblib

        model = joblib.load(model_path)
    else:
        model = train_model(pipeline_config=DEFAULT_PIPELINE_CONFIG)

    forecast_df = forecast_next_week(model, df, pipeline_config=DEFAULT_PIPELINE_CONFIG, threshold=threshold)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(output_csv, index=False)
    typer.echo(f"Saved forecast to {output_csv}")


if __name__ == "__main__":
    app()
