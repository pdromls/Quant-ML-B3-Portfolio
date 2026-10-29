"""
quant_ml_b3: Cross-Sectional Machine Learning & Event-Driven Backtesting on the B3 (IBrX 50).
"""

from .data_loader import MarketData, load_local_market_data, load_benchmark_series
from .features import FEATURE_COLS, compute_ticker_features, build_ml_panel, extract_current_features
from .splits import TemporalSplit, split_temporal_70_20_10
from .metrics import compute_ml_metrics, compute_financial_metrics, compute_trade_win_rate
from .models import MLModelArtifact, fit_ml_model, predict_ml_model, extract_feature_importance
from .backtest import (
    BacktestResult,
    BuyAndHoldStrategy,
    CrossSectionalMLStrategy,
    run_backtest,
    benchmark_equity_curve,
)

__all__ = [
    "MarketData",
    "load_local_market_data",
    "load_benchmark_series",
    "FEATURE_COLS",
    "compute_ticker_features",
    "build_ml_panel",
    "extract_current_features",
    "TemporalSplit",
    "split_temporal_70_20_10",
    "compute_ml_metrics",
    "compute_financial_metrics",
    "compute_trade_win_rate",
    "MLModelArtifact",
    "fit_ml_model",
    "predict_ml_model",
    "extract_feature_importance",
    "BacktestResult",
    "BuyAndHoldStrategy",
    "CrossSectionalMLStrategy",
    "run_backtest",
    "benchmark_equity_curve",
]
