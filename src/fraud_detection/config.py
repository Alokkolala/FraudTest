"""Configuration defaults for the fraud detection prototype."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    algorithm: str = "isolation_forest"
    contamination: float = 0.02
    n_estimators: int = 200
    random_state: int = 42
    max_features: float = 1.0


@dataclass
class RuleConfig:
    high_amount_threshold: float = 5000.0
    risky_countries: List[str] = field(default_factory=lambda: ["RU", "IR", "KP", "SY"])
    risky_categories: List[str] = field(default_factory=lambda: ["cryptocurrency", "gambling", "adult"])
    velocity_time_minutes: int = 30
    velocity_transaction_count: int = 5
    deviation_multiplier: float = 4.0


@dataclass
class PipelineConfig:
    timestamp_column: str = "timestamp"
    amount_column: str = "amount"
    geography_column: str = "country"
    category_column: str = "category"
    customer_column: str = "customer_id"
    label_column: str = "is_fraud"
    transaction_id_column: str = "transaction_id"
    history_window: int = 10


DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_RULE_CONFIG = RuleConfig()
DEFAULT_PIPELINE_CONFIG = PipelineConfig()
