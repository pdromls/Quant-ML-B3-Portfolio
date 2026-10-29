"""
Feature engineering and cross-sectional panel construction strictly free of look-ahead bias.
All 12 technical and statistical indicators at trading day `t` use solely historical observations `<= t`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import MarketData

FEATURE_COLS: list[str] = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_21d",
    "ret_63d",
    "vol_21d",
    "sma20_dist",
    "sma50_dist",
    "sma_ratio",
    "rsi_14",
    "bb_pos",
    "vol_ratio",
]


def _lag_ret(x: np.ndarray, k: int) -> np.ndarray:
    """Compute k-day cumulative return `x[t] / x[t - k] - 1`."""
    n = len(x)
    r = np.full(n, np.nan, dtype=float)
    if n > k:
        with np.errstate(divide="ignore", invalid="ignore"):
            r[k:] = x[k:] / x[:-k] - 1.0
    return r


def _roll_mean(x: np.ndarray, k: int) -> np.ndarray:
    """Exact simple moving average requiring all `k` observations in window to be non-NaN."""
    n = len(x)
    if n < k:
        return np.full(n, np.nan, dtype=float)
    return pd.Series(x, dtype=float).rolling(window=k, min_periods=k).mean().to_numpy()


def _roll_sd(x: np.ndarray, k: int) -> np.ndarray:
    """Exact sample standard deviation (ddof=1) requiring all `k` observations in window to be non-NaN."""
    n = len(x)
    if n < k:
        return np.full(n, np.nan, dtype=float)
    return pd.Series(x, dtype=float).rolling(window=k, min_periods=k).std(ddof=1).to_numpy()


def compute_ticker_features(close_vec: np.ndarray, vol_vec: np.ndarray) -> np.ndarray:
    """
    Compute the 12 technical indicators for a single stock time series.
    Returns an `(n_days, 12)` float array aligned with `FEATURE_COLS`.
    """
    close_arr = np.asarray(close_vec, dtype=float)
    vol_arr = np.asarray(vol_vec, dtype=float)
    n = len(close_arr)
    out = np.full((n, len(FEATURE_COLS)), np.nan, dtype=float)
    if n < 65:
        return out

    r1 = _lag_ret(close_arr, 1)
    r5 = _lag_ret(close_arr, 5)
    r10 = _lag_ret(close_arr, 10)
    r21 = _lag_ret(close_arr, 21)
    r63 = _lag_ret(close_arr, 63)

    vol21 = _roll_sd(r1, 21)
    sma20 = _roll_mean(close_arr, 20)
    sma50 = _roll_mean(close_arr, 50)
    sd20 = _roll_sd(close_arr, 20)

    with np.errstate(divide="ignore", invalid="ignore"):
        sma20_dist = close_arr / sma20 - 1.0
        sma50_dist = close_arr / sma50 - 1.0
        sma_ratio = sma20 / sma50 - 1.0

        valid_sd = (~np.isnan(sd20)) & (sd20 > 1e-8)
        bb_pos = np.where(valid_sd, (close_arr - sma20) / (2.0 * sd20), 0.0)
        bb_pos[np.isnan(close_arr) | np.isnan(sma20)] = np.nan

        # RSI(14) normalized to [-1, +1]
        delta = np.empty(n, dtype=float)
        delta[0] = np.nan
        delta[1:] = np.diff(close_arr)

        gain = np.where(np.isnan(delta), np.nan, np.where(delta > 0.0, delta, 0.0))
        loss = np.where(np.isnan(delta), np.nan, np.where(delta < 0.0, -delta, 0.0))
        avg_gain = _roll_mean(gain, 14)
        avg_loss = _roll_mean(loss, 14)

        rs = np.where(avg_loss < 1e-10, 100.0, avg_gain / avg_loss)
        rsi14 = (100.0 - (100.0 / (1.0 + rs)) - 50.0) / 50.0
        rsi14[np.isnan(avg_gain) | np.isnan(avg_loss)] = np.nan

        # Relative volume (5d vs 21d) clipped to [-2, +5]
        vol5_mean = _roll_mean(vol_arr, 5)
        vol21_mean = _roll_mean(vol_arr, 21)
        valid_vol = (~np.isnan(vol21_mean)) & (vol21_mean > 0.0)
        vol_ratio = np.where(valid_vol, vol5_mean / vol21_mean - 1.0, 0.0)
        vol_ratio = np.clip(vol_ratio, -2.0, 5.0)
        vol_ratio[np.isnan(vol5_mean) | np.isnan(vol21_mean)] = np.nan

    out[:, 0] = r1
    out[:, 1] = r5
    out[:, 2] = r10
    out[:, 3] = r21
    out[:, 4] = r63
    out[:, 5] = vol21
    out[:, 6] = sma20_dist
    out[:, 7] = sma50_dist
    out[:, 8] = sma_ratio
    out[:, 9] = rsi14
    out[:, 10] = bb_pos
    out[:, 11] = vol_ratio
    return out


def cross_sectional_scale(df_panel: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS) -> pd.DataFrame:
    """
    Standardize features cross-sectionally per trading date (Z-score across stocks on the same day),
    winsorized to `[-4, +4]`.
    """
    df = df_panel.copy()
    grouped = df.groupby("date", sort=False)
    for col in feature_cols:
        means = grouped[col].transform("mean")
        sds = grouped[col].transform(lambda s: s.std(ddof=1))
        sds = sds.where((sds.notna()) & (sds >= 1e-8), 1.0)
        z = (df[col] - means) / sds
        df[col] = z.clip(-4.0, 4.0)
    return df


def build_ml_panel(market: MarketData, horizon: int = 21, step: int = 5) -> pd.DataFrame:
    """
    Construct the longitudinal panel `(date x ticker)` with:
      - Standardized cross-sectional features `X` at day `t`
      - Forward return `target_raw = close[t + horizon] / close[t] - 1`
      - Cross-sectional excess return `target_excess = target_raw - mean_t(target_raw)`
    """
    n_dates = len(market.dates)
    # 0-based index 64 corresponds to the 65th trading day (matching R's `seq(65L, n_dates - horizon, by = step)`)
    valid_t = np.arange(64, n_dates - horizon, int(step), dtype=int)
    if len(valid_t) == 0:
        return pd.DataFrame()

    panel_list: list[pd.DataFrame] = []
    for tk in market.tickers:
        close_v = market.close[tk].to_numpy(dtype=float)
        vol_v = market.volume[tk].to_numpy(dtype=float)
        feats = compute_ticker_features(close_v, vol_v)

        with np.errstate(divide="ignore", invalid="ignore"):
            fwd_ret = close_v[valid_t + horizon] / close_v[valid_t] - 1.0

        df_tk = pd.DataFrame(feats[valid_t, :], columns=FEATURE_COLS)
        df_tk.insert(0, "target_raw", fwd_ret)
        df_tk.insert(0, "ticker", tk)
        df_tk.insert(0, "t_idx", valid_t)
        df_tk.insert(0, "date", market.dates[valid_t])
        panel_list.append(df_tk)

    panel = pd.concat(panel_list, ignore_index=True)
    panel = panel.dropna(subset=["target_raw"] + FEATURE_COLS)
    if panel.empty:
        return panel

    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    mean_ret_date = panel.groupby("date", sort=False)["target_raw"].transform("mean")
    panel["target_excess"] = panel["target_raw"] - mean_ret_date

    panel = cross_sectional_scale(panel, FEATURE_COLS)
    return panel


def extract_current_features(market_window: MarketData) -> pd.DataFrame:
    """
    Extract and cross-sectionally standardize features for the latest trading day (`t = -1`)
    of a historical `MarketData` window during live/simulated decision making.
    """
    n_rows = len(market_window.dates)
    if n_rows < 65:
        return pd.DataFrame()

    today = market_window.dates[-1]
    rows: list[dict] = []

    for tk in market_window.tickers:
        if np.isnan(market_window.close[tk].iloc[-1]):
            continue
        close_v = market_window.close[tk].to_numpy(dtype=float)
        vol_v = market_window.volume[tk].to_numpy(dtype=float)
        feats = compute_ticker_features(close_v, vol_v)
        last_row = feats[-1, :]
        if np.isnan(last_row).any():
            continue
        rec = {"date": today, "ticker": tk}
        for col_name, val in zip(FEATURE_COLS, last_row):
            rec[col_name] = float(val)
        rows.append(rec)

    if len(rows) < 5:
        return pd.DataFrame()

    df_today = pd.DataFrame(rows)
    df_today = cross_sectional_scale(df_today, FEATURE_COLS)
    return df_today
