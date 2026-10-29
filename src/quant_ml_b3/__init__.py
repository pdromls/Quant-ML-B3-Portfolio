"""
quant_ml_b3: Cross-Sectional Machine Learning for Stock Selection on the B3 (IBrX 50).
"""

from .data_loader import (
    MarketData,
    MarketDataValidationReport,
    load_local_market_data,
    download_b3_universe_yfinance,
    validate_market_data,
)
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
    evaluate_monthly_top_k_selection,
)
from .models import (
    MLModelArtifact,
    fit_ml_model,
    predict_ml_model,
    extract_feature_importance,
)
from .walk_forward import (
    WalkForwardResult,
    WeightingComparisonResult,
    run_walk_forward_simulation,
    run_weighting_schemes_comparison,
    compute_performance_metrics,
)
from .plotting import (
    plot_feature_importance,
    plot_test_metrics_summary,
    plot_top_k_cumulative_alpha,
    plot_walk_forward_comparison,
    plot_weighting_schemes_comparison,
)

__all__ = [
    "MarketData",
    "MarketDataValidationReport",
    "load_local_market_data",
    "download_b3_universe_yfinance",
    "validate_market_data",
    "FEATURE_COLS",
    "compute_ticker_features",
    "cross_sectional_scale",
    "build_ml_panel",
    "extract_current_features",
    "TemporalSplit",
    "split_temporal_70_20_10",
    "compute_ml_metrics",
    "evaluate_monthly_top_k_selection",
    "MLModelArtifact",
    "fit_ml_model",
    "predict_ml_model",
    "extract_feature_importance",
    "WalkForwardResult",
    "WeightingComparisonResult",
    "run_walk_forward_simulation",
    "run_weighting_schemes_comparison",
    "compute_performance_metrics",
    "plot_feature_importance",
    "plot_test_metrics_summary",
    "plot_top_k_cumulative_alpha",
    "plot_walk_forward_comparison",
    "plot_weighting_schemes_comparison",
]
