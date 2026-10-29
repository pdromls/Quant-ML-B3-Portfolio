"""
Event-Driven Portfolio Backtesting Engine with Realistic Execution:
  - Decisions made at close of day `t` using solely historical data `<= t`
  - Orders executed at adjusted open of day `t + 1`
  - Integer share quantities, sell-before-buy cash management, never negative cash
  - Proportional transaction costs (10 bps brokerage + 5 bps slippage = 15 bps)
  - Hysteresis buffer (Top-K selection with Top-2K maintenance band) & Walk-Forward retraining
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from .data_loader import MarketData
from .features import FEATURE_COLS, build_ml_panel, extract_current_features
from .metrics import compute_financial_metrics
from .models import MLModelArtifact, fit_ml_model, predict_ml_model
from .splits import split_temporal_70_20_10


class Strategy(Protocol):
    name: str
    lookback: int | float

    def prepare(self, history: MarketData) -> None: ...
    def decide(self, window: MarketData, positions: dict[str, float]) -> dict[str, float] | None: ...


@dataclass
class BuyAndHoldStrategy:
    """Equal-weight (1/N) Buy & Hold strategy initialized on the first trading day."""

    name: str = "Buy & Hold 1/N (IBrX 50)"
    lookback: int = 1
    _invested: bool = False

    def prepare(self, history: MarketData) -> None:
        self._invested = False

    def decide(self, window: MarketData, positions: dict[str, float]) -> dict[str, float] | None:
        if self._invested:
            return None
        last_close = window.close.iloc[-1]
        available = [tk for tk in window.tickers if pd.notna(last_close[tk]) and last_close[tk] > 0]
        if not available:
            return None
        self._invested = True
        w = 1.0 / len(available)
        return {tk: w for tk in available}


@dataclass
class CrossSectionalMLStrategy:
    """
    Cross-Sectional ML Strategy (Ridge, XGBoost, or LightGBM) with monthly rebalancing,
    Walk-Forward periodic retraining, and Top-K / Top-2K hysteresis buffer.
    """

    model_type: str = "xgboost"
    top_k: int = 10
    horizon: int = 21
    retrain_every_months: int = 12
    lookback: int = 756
    seed: int = 42
    name: str = ""
    model_fit: MLModelArtifact | None = field(default=None, init=False)
    last_rebal_month: str | None = field(default=None, init=False)
    months_since_train: int = field(default=0, init=False)
    retrain_log: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        mtype = self.model_type.strip().lower()
        self.model_type = mtype
        if not self.name:
            labels = {
                "ridge": "ML Baseline: Regressao Ridge",
                "xgboost": "ML: XGBoost",
                "lightgbm": "ML: LightGBM",
            }
            self.name = labels.get(mtype, f"ML: {mtype.upper()}")

    def prepare(self, history: MarketData) -> None:
        self.model_fit = None
        self.last_rebal_month = None
        self.months_since_train = 0
        self.retrain_log = []

        n_h = len(history.dates)
        if n_h >= 180:
            if np.isfinite(self.lookback) and n_h > int(self.lookback):
                history = history.slice_iloc(np.arange(n_h - int(self.lookback), n_h))
            panel = build_ml_panel(history, horizon=self.horizon, step=5)
            if not panel.empty and len(panel) >= 200:
                splits = split_temporal_70_20_10(
                    panel, train_ratio=0.70, val_ratio=0.20, purge_days=self.horizon
                )
                self.model_fit = fit_ml_model(
                    self.model_type, splits, FEATURE_COLS, seed=self.seed
                )
                self.months_since_train = 0
                self.retrain_log.append(
                    {
                        "train_date": history.dates[-1],
                        "metrics": self.model_fit.metrics_table,
                    }
                )

    def decide(self, window: MarketData, positions: dict[str, float]) -> dict[str, float] | None:
        today = window.dates[-1]
        month_id = today.strftime("%Y-%m")
        if month_id == self.last_rebal_month:
            return None

        self.months_since_train += 1
        need_retrain = (self.model_fit is None) or (
            self.retrain_every_months > 0 and self.months_since_train >= self.retrain_every_months
        )

        if need_retrain and len(window.dates) >= 200:
            panel = build_ml_panel(window, horizon=self.horizon, step=5)
            if not panel.empty and len(panel) >= 200:
                splits = split_temporal_70_20_10(
                    panel, train_ratio=0.70, val_ratio=0.20, purge_days=self.horizon
                )
                self.model_fit = fit_ml_model(
                    self.model_type, splits, FEATURE_COLS, seed=self.seed
                )
                self.months_since_train = 0
                self.retrain_log.append(
                    {
                        "train_date": today,
                        "metrics": self.model_fit.metrics_table,
                    }
                )

        if self.model_fit is None:
            return None

        df_today = extract_current_features(window)
        if df_today.empty:
            return None

        preds = predict_ml_model(self.model_fit, df_today).dropna()
        if preds.empty:
            return None

        # Sort descending by predicted excess return
        ranked = list(preds.sort_values(ascending=False).index)
        buffer_rank = set(ranked[: min(2 * self.top_k, len(ranked))])
        current_held = [tk for tk, qty in positions.items() if qty > 0]
        kept = [tk for tk in current_held if tk in buffer_rank]

        if len(kept) >= self.top_k:
            chosen = kept[: self.top_k]
        else:
            new_picks = [tk for tk in ranked if tk not in kept]
            chosen = kept + new_picks[: self.top_k - len(kept)]

        self.last_rebal_month = month_id
        w = 1.0 / len(chosen)
        return {tk: w for tk in chosen}


@dataclass
class BacktestResult:
    """Result container returned by `run_backtest`."""

    strategy_name: str
    equity: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]
    initial_capital: float
    cost_rate: float


def _rebalance_portfolio(
    cash: float,
    positions: np.ndarray,
    tickers: list[str],
    target_weights: dict[str, float],
    open_prices: np.ndarray,
    last_prices: np.ndarray,
    cost_rate: float,
    date: pd.Timestamp,
) -> tuple[float, np.ndarray, list[dict]]:
    """
    Execute integer-share rebalancing at `open_prices` selling before buying,
    deducting proportional transaction costs (`cost_rate`), and preventing negative cash.
    """
    n_t = len(tickers)
    w_vec = np.zeros(n_t, dtype=float)
    tk_to_idx = {tk: idx for idx, tk in enumerate(tickers)}
    for tk, weight in target_weights.items():
        if tk in tk_to_idx:
            w_vec[tk_to_idx[tk]] = float(weight)

    pos_val = np.where(positions > 0, positions * last_prices, 0.0)
    equity = cash + float(np.nansum(pos_val))

    tradable = (~np.isnan(open_prices)) & (open_prices > 0.0)
    goal = positions.copy()
    goal[tradable] = np.floor(w_vec[tradable] * equity / open_prices[tradable] + 1e-9)
    delta = goal - positions

    trade_records: list[dict] = []

    # 1. Execute sells first
    sell_qty = np.maximum(-delta, 0.0)
    sell_idx = np.where(sell_qty > 0)[0]
    for idx in sell_idx:
        q = float(sell_qty[idx])
        p = float(open_prices[idx])
        val = q * p
        c = val * cost_rate
        cash += val - c
        positions[idx] -= q
        trade_records.append(
            {
                "date": date,
                "ticker": tickers[idx],
                "side": "sell",
                "qty": q,
                "price": p,
                "value": val,
                "cost": c,
            }
        )

    # 2. Execute buys with proportional scale-down if cash is insufficient
    wanted = np.maximum(delta, 0.0)
    if cash > 0 and np.any(wanted > 0):
        scale = 1.0
        while True:
            buy_qty = np.floor(wanted * scale)
            buy_vals = np.where(buy_qty > 0, buy_qty * open_prices, 0.0)
            buy_costs = buy_vals * cost_rate
            total_needed = float(np.sum(buy_vals + buy_costs))
            if total_needed <= cash or total_needed == 0.0:
                break
            scale *= min(0.999, cash / total_needed)

        buy_idx = np.where(buy_qty > 0)[0]
        for idx in buy_idx:
            q = float(buy_qty[idx])
            p = float(open_prices[idx])
            val = q * p
            c = val * cost_rate
            cash -= val + c
            positions[idx] += q
            trade_records.append(
                {
                    "date": date,
                    "ticker": tickers[idx],
                    "side": "buy",
                    "qty": q,
                    "price": p,
                    "value": val,
                    "cost": c,
                }
            )

    return cash, positions, trade_records


def run_backtest(
    strategy: Strategy,
    market: MarketData,
    cost_rate: float = 0.0015,
    initial_capital: float = 100_000.0,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> BacktestResult:
    """
    Run an event-driven simulation over `[start, end]`:
      - Pre-backtest history `< start` is passed to `strategy.prepare()`
      - At each trading day `t`:
          1. Execute pending orders from `t - 1` at `open[t]`
          2. Mark portfolio to market at `close[t]`
          3. Build strictly causal window `[max(0, t - lookback + 1) .. t]` and call `strategy.decide()`
    """
    start_ts = market.dates[0] if start is None else pd.Timestamp(start)
    end_ts = market.dates[-1] if end is None else pd.Timestamp(end)

    period_idx = np.where((market.dates >= start_ts) & (market.dates <= end_ts))[0]
    if len(period_idx) == 0:
        raise ValueError("No trading days found in the requested backtest period.")

    first_idx = int(period_idx[0])
    if first_idx > 0:
        strategy.prepare(market.slice_iloc(np.arange(0, first_idx)))
    else:
        strategy.prepare(market.slice_iloc(np.array([], dtype=int)))

    tickers = market.tickers
    n_t = len(tickers)
    n_days = len(period_idx)

    cash = float(initial_capital)
    positions = np.zeros(n_t, dtype=float)
    last_price = np.full(n_t, np.nan, dtype=float)

    open_mat = market.open.to_numpy(dtype=float)
    close_mat = market.close.to_numpy(dtype=float)

    equity_rows: list[dict] = []
    all_trades: list[dict] = []
    pending_weights: dict[str, float] | None = None

    for i, t in enumerate(period_idx):
        dt = market.dates[t]

        # 1. Execute pending orders decided at t - 1 using open[t]
        open_today = open_mat[t, :]
        valid_open = ~np.isnan(open_today)
        last_price[valid_open] = open_today[valid_open]

        if pending_weights is not None:
            cash, positions, day_trades = _rebalance_portfolio(
                cash=cash,
                positions=positions,
                tickers=tickers,
                target_weights=pending_weights,
                open_prices=open_today,
                last_prices=last_price,
                cost_rate=cost_rate,
                date=dt,
            )
            all_trades.extend(day_trades)
            pending_weights = None

        # 2. Mark to market at close[t]
        close_today = close_mat[t, :]
        valid_close = ~np.isnan(close_today)
        last_price[valid_close] = close_today[valid_close]

        invested = float(np.nansum(np.where(positions > 0, positions * last_price, 0.0)))
        equity_rows.append(
            {
                "date": dt,
                "cash": cash,
                "invested": invested,
                "equity": cash + invested,
            }
        )

        if i == n_days - 1:
            break

        # 3. Causal window up to t and strategy decision for t + 1
        win_start = 0 if np.isinf(strategy.lookback) else max(0, int(t) - int(strategy.lookback) + 1)
        window = market.slice_iloc(np.arange(win_start, int(t) + 1))
        pos_dict = {tickers[j]: float(positions[j]) for j in range(n_t)}
        pending_weights = strategy.decide(window, pos_dict)

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(
        all_trades,
        columns=["date", "ticker", "side", "qty", "price", "value", "cost"],
    )
    metrics = compute_financial_metrics(
        equity_df=equity_df,
        trades_df=trades_df,
        initial_capital=initial_capital,
    )

    return BacktestResult(
        strategy_name=strategy.name,
        equity=equity_df,
        trades=trades_df,
        metrics=metrics,
        initial_capital=initial_capital,
        cost_rate=cost_rate,
    )


def benchmark_equity_curve(
    benchmark_df: pd.DataFrame,
    dates: pd.DatetimeIndex | pd.Series,
    initial_capital: float = 100_000.0,
) -> pd.DataFrame:
    """
    Align a passive benchmark index/ETF (e.g., BRAX11) to the backtest trading dates,
    starting at `initial_capital` and forward-filling missing quotes.
    """
    bench = benchmark_df.sort_values("date").reset_index(drop=True)
    target_dates = pd.DataFrame({"date": pd.to_datetime(dates)})
    merged = pd.merge_asof(target_dates, bench[["date", "close"]], on="date", direction="backward")
    first_valid = merged["close"].first_valid_index()
    if first_valid is None:
        raise ValueError("Benchmark has no valid quotes in the backtest date range.")
    base_price = float(merged["close"].iloc[first_valid])
    merged["equity"] = initial_capital * (merged["close"] / base_price)
    return merged[["date", "equity"]]
