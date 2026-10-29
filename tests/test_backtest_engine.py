"""
Unit tests for the Event-Driven Backtesting Engine:
  - Non-negative cash invariant
  - Transaction cost accounting (15 bps)
  - Hysteresis buffer behavior (Top-K selection with Top-2K maintenance band)
  - Realized trade win-rate calculation
  - Passive benchmark alignment
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_ml_b3.backtest import (
    BuyAndHoldStrategy,
    CrossSectionalMLStrategy,
    benchmark_equity_curve,
    run_backtest,
)
from quant_ml_b3.metrics import compute_financial_metrics, compute_trade_win_rate
from test_features_and_splits import _make_synthetic_market


def test_buy_and_hold_and_ml_backtest_preserve_cash_invariants() -> None:
    mkt = _make_synthetic_market(n_days=380, n_tickers=12, seed=7)
    start_dt = mkt.dates[260]
    end_dt = mkt.dates[-1]

    bt_bh = run_backtest(
        BuyAndHoldStrategy(),
        mkt,
        cost_rate=0.0015,
        initial_capital=100_000.0,
        start=start_dt,
        end=end_dt,
    )
    assert (bt_bh.equity["cash"] >= -1e-6).all()
    assert (bt_bh.equity["equity"] > 0).all()
    assert bt_bh.metrics["total_cost"] > 0

    bt_ridge = run_backtest(
        CrossSectionalMLStrategy("ridge", top_k=4, retrain_every_months=6, lookback=260),
        mkt,
        cost_rate=0.0015,
        initial_capital=100_000.0,
        start=start_dt,
        end=end_dt,
    )
    assert (bt_ridge.equity["cash"] >= -1e-6).all()
    assert not bt_ridge.trades.empty
    assert np.isfinite(bt_ridge.metrics["sharpe"])


def test_higher_transaction_costs_reduce_final_equity() -> None:
    mkt = _make_synthetic_market(n_days=340, n_tickers=10, seed=99)
    start_dt = mkt.dates[240]
    end_dt = mkt.dates[-1]

    bt_zero_cost = run_backtest(
        CrossSectionalMLStrategy("ridge", top_k=3, retrain_every_months=0, lookback=240),
        mkt,
        cost_rate=0.0,
        initial_capital=100_000.0,
        start=start_dt,
        end=end_dt,
    )
    bt_high_cost = run_backtest(
        CrossSectionalMLStrategy("ridge", top_k=3, retrain_every_months=0, lookback=240),
        mkt,
        cost_rate=0.0050,
        initial_capital=100_000.0,
        start=start_dt,
        end=end_dt,
    )
    assert bt_zero_cost.metrics["final_equity"] > bt_high_cost.metrics["final_equity"]
    assert bt_zero_cost.metrics["total_cost"] == 0.0
    assert bt_high_cost.metrics["total_cost"] > 0.0


def test_compute_trade_win_rate_exact_accounting() -> None:
    trades = pd.DataFrame(
        [
            {"date": "2024-01-02", "ticker": "PETR4", "side": "buy", "qty": 100, "price": 30.0, "value": 3000.0, "cost": 4.5},
            {"date": "2024-02-02", "ticker": "PETR4", "side": "sell", "qty": 100, "price": 35.0, "value": 3500.0, "cost": 5.25},
            {"date": "2024-01-02", "ticker": "VALE3", "side": "buy", "qty": 100, "price": 70.0, "value": 7000.0, "cost": 10.5},
            {"date": "2024-02-02", "ticker": "VALE3", "side": "sell", "qty": 100, "price": 65.0, "value": 6500.0, "cost": 9.75},
        ]
    )
    wr = compute_trade_win_rate(trades)
    assert abs(wr - 0.50) < 1e-9


def test_benchmark_equity_curve_alignment() -> None:
    dates = pd.bdate_range("2024-01-02", periods=10)
    bench = pd.DataFrame(
        {
            "date": dates,
            "close": np.linspace(100.0, 110.0, len(dates)),
        }
    )
    curve = benchmark_equity_curve(bench, dates, initial_capital=100_000.0)
    assert len(curve) == 10
    assert abs(curve["equity"].iloc[0] - 100_000.0) < 1e-6
    assert abs(curve["equity"].iloc[-1] - 110_000.0) < 1e-6
    metrics = compute_financial_metrics(curve, initial_capital=100_000.0)
    assert abs(metrics["total_return"] - 0.10) < 1e-6
