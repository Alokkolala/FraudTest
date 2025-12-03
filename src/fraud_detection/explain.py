"""Explainability helpers using permutation importance and rule descriptions."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .config import DEFAULT_PIPELINE_CONFIG, PipelineConfig
from .features import add_history_features
from .rules import RULE_COLUMNS


EXPLANATION_COLUMNS = ["fraud_score", "is_suspicious", "explanation"]


def compute_feature_importance(model, df: pd.DataFrame, config: PipelineConfig = DEFAULT_PIPELINE_CONFIG) -> Dict[str, float]:
    """Compute permutation importance for fitted pipeline."""

    df_proc = add_history_features(df, config)
    y_dummy = np.zeros(len(df_proc))

    def _anomaly_score(estimator, X, y=None):
        raw = estimator.decision_function(X)
        scores = (raw.max() - raw)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return float(np.mean(scores))
    result = permutation_importance(
        model,
        df_proc,
        y_dummy,
        scoring=_anomaly_score,
        n_repeats=3,
        random_state=42,
    )
    feature_names = model["preprocess"].get_feature_names_out()
    importance = dict(zip(feature_names, result.importances_mean))
    return importance


def build_explanations(
    df_with_rules: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    feature_importance: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """Attach human-readable explanations combining rules and top model features."""

    df = df_with_rules.copy()
    df["fraud_score"] = scores
    df["is_suspicious"] = (scores >= threshold) | (df[RULE_COLUMNS].any(axis=1))

    explanations: List[str] = []
    for _, row in df.iterrows():
        parts: List[str] = []
        if row["fraud_score"] >= threshold:
            parts.append(f"Model score {row['fraud_score']:.2f} >= {threshold:.2f}")
        for col in RULE_COLUMNS:
            if bool(row[col]):
                parts.append(col.replace("rule_", "rule: ").replace("_", " "))
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)[:3]
            formatted = ", ".join([f"{name} ({score:.3f})" for name, score in top_features])
            parts.append(f"Top features: {formatted}")
        if not parts:
            parts.append("No rule triggered; low model score")
        explanations.append("; ".join(parts))

    df["explanation"] = explanations
    return df
