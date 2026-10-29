"""
quant_ml_b3: Cross-Sectional Machine Learning for Stock Selection on the B3 (IBrX 50).
"""

from .data_loader import MarketData, load_local_market_data, download_b3_universe_yfinance
from .features import (
    FEATURE_COLS,
    compute_ticker_features,
    cross_sectional_scale,
    build_ml_panel,
    extract_current_features,
)
from .splits import TemporalSplit, split_temporal_70_20_10
from .metrics import (
    compute_ml_metrics,
    evaluate_top_k_cumulative_performance,
)
from .models import (
    MLModelArtifact,
    fit_ml_model,
    predict_ml_model,
    extract_feature_importance,
)

__all__ = [
    "MarketData",
    "load_local_market_data",
    "download_b3_universe_yfinance",
    "FEATURE_COLS",
    "compute_ticker_features",
    "cross_sectional_scale",
    "build_ml_panel",
    "extract_current_features",
    "TemporalSplit",
    "split_temporal_70_20_10",
    "compute_ml_metrics",
    "evaluate_top_k_cumulative_performance",
    "MLModelArtifact",
    "fit_ml_model",
    "predict_ml_model",
    "extract_feature_importance",
]
