"""End-to-end pipeline for training, scoring, and exporting fraud results."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import DEFAULT_MODEL_CONFIG, DEFAULT_PIPELINE_CONFIG, DEFAULT_RULE_CONFIG, ModelConfig, PipelineConfig, RuleConfig
from .data import generate_synthetic_transactions, load_transactions
from .explain import build_explanations, compute_feature_importance
from .models import AnomalyModel, train_validation_split
from .rules import apply_rules


def train_model(
    training_data: Optional[pd.DataFrame] = None,
    model_config: ModelConfig = DEFAULT_MODEL_CONFIG,
    pipeline_config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
) -> AnomalyModel:
    if training_data is None:
        training_data = generate_synthetic_transactions(config=pipeline_config)
    model = AnomalyModel(model_config, pipeline_config)
    model.fit(training_data)
    return model


def score_transactions(
    model: AnomalyModel,
    input_data: pd.DataFrame,
    threshold: float = 0.6,
    rule_config: RuleConfig = DEFAULT_RULE_CONFIG,
    pipeline_config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
    compute_importance: bool = False,
) -> pd.DataFrame:
    df_rules = apply_rules(input_data, rule_config, pipeline_config)
    scores = model.predict_scores(df_rules)

    feature_importance = None
    if compute_importance and model.pipeline is not None:
        feature_importance = compute_feature_importance(model.pipeline, df_rules, pipeline_config)

    results = build_explanations(df_rules, scores, threshold, feature_importance)
    required_cols = [pipeline_config.transaction_id_column, "fraud_score", "is_suspicious", "explanation"]
    return results[required_cols]


def export_results(df_results: pd.DataFrame, output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)


def evaluate_model(
    model: AnomalyModel,
    df: pd.DataFrame,
    threshold: float = 0.6,
    pipeline_config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
):
    train_df, test_df = train_validation_split(df, config=pipeline_config)
    model.fit(train_df)
    roc_auc, pr_auc, recall_at_k, flagged_rate = model.evaluate(test_df, threshold)
    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "recall_at_10pct": recall_at_k,
        "flagged_rate": flagged_rate,
    }
    return metrics
