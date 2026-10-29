"""
Unit tests for the Walk-Forward Validation and Periodic Retraining engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_ml_b3.data_loader import MarketData
from quant_ml_b3.walk_forward import compute_performance_metrics, run_walk_forward_simulation


def _create_synthetic_market(n_days: int = 800, n_tickers: int = 12) -> MarketData:
    """Generate deterministic synthetic market data for unit testing."""
    dates = pd.date_range("2014-01-01", periods=n_days, freq="B")
    tickers = [f"TICK{i:02d}" for i in range(1, n_tickers + 1)]

    np.random.seed(42)
    # Drift + random noise
    returns = np.random.normal(loc=0.0004, scale=0.015, size=(n_days, n_tickers))
    price_levels = 50.0 * np.exp(np.cumsum(returns, axis=0))

    close_df = pd.DataFrame(price_levels, index=dates, columns=tickers)
    open_df = close_df * 0.998
    vol_df = pd.DataFrame(1e6, index=dates, columns=tickers)

    return MarketData(dates=pd.DatetimeIndex(dates), tickers=tickers, open=open_df, close=close_df, volume=vol_df)


def test_compute_performance_metrics_cagr_and_drawdown():
    # 24 months of steady 1% monthly return
    ret_series = pd.Series([0.01] * 24)
    m = compute_performance_metrics(ret_series)

    # (1.01)^24 - 1 = 26.97%
    assert 26.0 < m["Retorno_Acumulado_pct"] < 28.0
    # Annualized CAGR ~ 12.68%
    assert 12.0 < m["CAGR_pct"] < 13.5
    # Max drawdown is 0% since it was strictly ascending
    assert m["Max_Drawdown_pct"] == 0.0


def test_run_walk_forward_simulation_executes_and_generates_comparison():
    mkt = _create_synthetic_market(n_days=950, n_tickers=15)
    start_date = mkt.dates[500].strftime("%Y-%m-%d")
    end_date = mkt.dates[-1].strftime("%Y-%m-%d")

    res = run_walk_forward_simulation(
        market=mkt,
        start_date=start_date,
        end_date=end_date,
        models=("ridge", "xgboost", "lightgbm"),
        retrain_every_months=3,  # Short retraining interval for test
        train_lookback_days=400,
        top_k=4,
        hysteresis_buffer=8,
        seed=42,
    )

    assert not res.monthly_returns.empty
    assert "Universe" in res.monthly_returns.columns
    assert "Ridge_WF" in res.monthly_returns.columns
    assert "Xgboost_WF" in res.monthly_returns.columns
    assert "Lightgbm_WF" in res.monthly_returns.columns
    assert "Ridge_Frozen" in res.monthly_returns.columns

    assert not res.summary_metrics.empty
    assert len(res.summary_metrics) == 7  # Universe + 3 models x 2 modes (WF and Frozen)
    assert len(res.retrain_log) >= 2


def test_run_weighting_schemes_comparison_executes():
    mkt = _create_synthetic_market(n_days=950, n_tickers=15)
    start_date = mkt.dates[500].strftime("%Y-%m-%d")
    end_date = mkt.dates[-1].strftime("%Y-%m-%d")

    from quant_ml_b3.walk_forward import run_weighting_schemes_comparison

    res = run_weighting_schemes_comparison(
        market=mkt,
        start_date=start_date,
        end_date=end_date,
        models=("ridge", "xgboost", "lightgbm"),
        retrain_every_months=3,
        train_lookback_days=400,
        top_k_hard=4,
        hysteresis_buffer=8,
        top_k_soft=6,
        seed=42,
    )

    assert not res.monthly_returns.empty
    assert "Ridge_HardCut" in res.monthly_returns.columns
    assert "Ridge_SoftWeight" in res.monthly_returns.columns
    assert "Xgboost_HardCut" in res.monthly_returns.columns
    assert "Xgboost_SoftWeight" in res.monthly_returns.columns
    assert "Lightgbm_HardCut" in res.monthly_returns.columns
    assert "Lightgbm_SoftWeight" in res.monthly_returns.columns

    assert not res.summary_metrics.empty
    assert "Esquema" in res.summary_metrics.columns
    assert "Turnover_Medio_Mensal_pct" in res.summary_metrics.columns
    assert len(res.summary_metrics) == 7  # 1 Universe + 3 models x 2 schemes (HardCut and SoftWeight)
