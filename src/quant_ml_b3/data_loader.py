"""
Data loading and preprocessing for the B3 high-liquidity stock universe (IBrX 50) and benchmarks.
Supports both instant local reproduction from processed CSVs and live updates via Yahoo Finance (`yfinance`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketData:
    """
    Aligned date x ticker matrices for adjusted open prices, adjusted close prices, and volume.
    `NaN` entries denote days on which a given ticker was not actively traded.
    """

    dates: pd.DatetimeIndex
    tickers: list[str]
    open: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    def __post_init__(self) -> None:
        if not (self.open.shape == self.close.shape == self.volume.shape):
            raise ValueError("open, close, and volume matrices must share identical dimensions.")
        if len(self.dates) != self.close.shape[0]:
            raise ValueError("Length of dates must match number of rows in price matrices.")

    def slice_dates(self, start: str | pd.Timestamp | None = None, end: str | pd.Timestamp | None = None) -> "MarketData":
        """Return a temporal slice of the market data between `start` and `end` (inclusive)."""
        mask = np.ones(len(self.dates), dtype=bool)
        if start is not None:
            mask &= self.dates >= pd.Timestamp(start)
        if end is not None:
            mask &= self.dates <= pd.Timestamp(end)
        return self.slice_iloc(np.where(mask)[0])

    def slice_iloc(self, row_indices: Sequence[int] | np.ndarray) -> "MarketData":
        """Return a subset of trading days by integer row positions."""
        idx = np.asarray(row_indices, dtype=int)
        return MarketData(
            dates=self.dates[idx],
            tickers=list(self.tickers),
            open=self.open.iloc[idx].reset_index(drop=True),
            close=self.close.iloc[idx].reset_index(drop=True),
            volume=self.volume.iloc[idx].reset_index(drop=True),
        )


def load_local_market_data(data_dir: str | Path = "data/processed") -> MarketData:
    """
    Load the pre-cleaned B3 IBrX 50 dataset (41 high-liquidity stocks, 2012-2026) from `data/processed/`.
    """
    base = Path(data_dir)
    open_df = pd.read_csv(base / "ibx50_open_adj.csv")
    close_df = pd.read_csv(base / "ibx50_close_adj.csv")
    vol_df = pd.read_csv(base / "ibx50_volume.csv")

    dates = pd.to_datetime(close_df["date"])
    tickers = [c for c in close_df.columns if c != "date"]

    open_mat = open_df[tickers].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    close_mat = close_df[tickers].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    vol_mat = vol_df[tickers].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)

    return MarketData(
        dates=pd.DatetimeIndex(dates),
        tickers=tickers,
        open=open_mat,
        close=close_mat,
        volume=vol_mat,
    )


def load_benchmark_series(
    data_dir: str | Path = "data/processed",
    filename: str = "benchmark_brax11.csv",
) -> pd.DataFrame:
    """Load the benchmark price series (e.g., BRAX11 ETF) with columns `['date', 'close']`."""
    path = Path(data_dir) / filename
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    return df[["date", "close"]]


def download_b3_universe_yfinance(
    tickers: Sequence[str],
    start: str = "2012-01-01",
    end: str = "2026-01-01",
    min_active_stocks_per_day: int = 20,
) -> MarketData:
    """
    Fetch historical daily OHLCV from Yahoo Finance (`.SA` suffix) and apply the project's
    data hygiene rules:
      1. Filter out spurious holiday dates with fewer than `min_active_stocks_per_day` active stocks.
      2. Mask days where `volume <= 0` as `NaN` to prevent artificial zero returns or invalid fills.
      3. Adjust `open` prices by the corporate action factor (`Adj Close / Close`).
    """
    import yfinance as yf

    clean_tickers = [t.replace(".SA", "").strip().upper() for t in tickers]
    yf_symbols = [f"{t}.SA" for t in clean_tickers]

    raw = yf.download(yf_symbols, start=start, end=end, auto_adjust=False, progress=False, group_by="ticker")

    frames_open = {}
    frames_close = {}
    frames_vol = {}

    for tk, yf_sym in zip(clean_tickers, yf_symbols):
        sub = raw[yf_sym].copy() if len(clean_tickers) > 1 else raw.copy()
        valid = sub["Volume"].notna() & (sub["Volume"] > 0) & sub["Close"].notna() & (sub["Close"] > 0)
        adj_factor = sub["Adj Close"] / sub["Close"]
        frames_open[tk] = np.where(valid, sub["Open"] * adj_factor, np.nan)
        frames_close[tk] = np.where(valid, sub["Adj Close"], np.nan)
        frames_vol[tk] = np.where(valid, sub["Volume"], np.nan)

    idx = pd.to_datetime(raw.index).tz_localize(None)
    open_df = pd.DataFrame(frames_open, index=idx)
    close_df = pd.DataFrame(frames_close, index=idx)
    vol_df = pd.DataFrame(frames_vol, index=idx)

    active_per_day = (vol_df > 0).sum(axis=1)
    valid_days = active_per_day >= min(min_active_stocks_per_day, max(1, len(clean_tickers) // 2))

    open_df = open_df.loc[valid_days]
    close_df = close_df.loc[valid_days]
    vol_df = vol_df.loc[valid_days]

    return MarketData(
        dates=pd.DatetimeIndex(close_df.index),
        tickers=clean_tickers,
        open=open_df.reset_index(drop=True),
        close=close_df.reset_index(drop=True),
        volume=vol_df.reset_index(drop=True),
    )
