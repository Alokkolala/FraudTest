"""Model training and scoring utilities."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import DEFAULT_MODEL_CONFIG, DEFAULT_PIPELINE_CONFIG, ModelConfig, PipelineConfig
from .features import add_history_features, build_preprocess_pipeline


class AnomalyModel:
    """Wrapper around IsolationForest with preprocessing."""

    def __init__(
        self,
        model_config: ModelConfig = DEFAULT_MODEL_CONFIG,
        pipeline_config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
    ) -> None:
        self.model_config = model_config
        self.pipeline_config = pipeline_config
        self.pipeline: Optional[Pipeline] = None

    def fit(self, df: pd.DataFrame) -> "AnomalyModel":
        df_proc = add_history_features(df, self.pipeline_config)
        preprocess = build_preprocess_pipeline(df_proc, self.pipeline_config)

        clf = IsolationForest(
            n_estimators=self.model_config.n_estimators,
            contamination=self.model_config.contamination,
            max_features=self.model_config.max_features,
            random_state=self.model_config.random_state,
        )

        self.pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", clf)])
        self.pipeline.fit(df_proc)
        return self

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise ValueError("Model has not been fit.")
        df_proc = add_history_features(df, self.pipeline_config)
        # IsolationForest returns anomaly scores where lower is more abnormal; invert and scale
        raw = self.pipeline.decision_function(df_proc)
        scores = (raw.max() - raw)  # higher means more anomalous
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return scores

    def predict_labels(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        scores = self.predict_scores(df)
        return (scores >= threshold).astype(int)

    def evaluate(
        self, df: pd.DataFrame, threshold: float = 0.5
    ) -> Tuple[float, float, float, float]:
        labels = df[self.pipeline_config.label_column]
        scores = self.predict_scores(df)

        roc_auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan")
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = np.trapz(recall, precision)
        recall_at_k = _recall_at_k(labels, scores, k=0.1)
        return roc_auc, pr_auc, recall_at_k, float(np.mean(scores >= threshold))


def train_validation_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, config: PipelineConfig = DEFAULT_PIPELINE_CONFIG
):
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[config.label_column])


def _recall_at_k(labels: pd.Series, scores: np.ndarray, k: float = 0.1) -> float:
    n = len(labels)
    top_k = max(1, int(n * k))
    order = np.argsort(scores)[::-1][:top_k]
    return float(labels.iloc[order].sum() / labels.sum()) if labels.sum() > 0 else float("nan")
