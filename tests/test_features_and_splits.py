"""
Unit tests for causal feature engineering (zero look-ahead bias), cross-sectional Z-score scaling,
and purged chronological 70/20/10 splitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_ml_b3.data_loader import MarketData
from quant_ml_b3.features import (
    FEATURE_COLS,
    build_ml_panel,
    compute_ticker_features,
    cross_sectional_scale,
    extract_current_features,
)
from quant_ml_b3.splits import split_temporal_70_20_10


def _make_synthetic_market(n_days: int = 320, n_tickers: int = 12, seed: int = 42) -> MarketData:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    tickers = [f"TK{i:02d}" for i in range(1, n_tickers + 1)]

    rets = rng.normal(0.0005, 0.018, size=(n_days, n_tickers))
    close_arr = 20.0 * np.exp(np.cumsum(rets, axis=0))
    open_arr = close_arr * (1.0 + rng.normal(0.0, 0.003, size=(n_days, n_tickers)))
    vol_arr = rng.uniform(1e5, 5e6, size=(n_days, n_tickers))

    return MarketData(
        dates=dates,
        tickers=tickers,
        open=pd.DataFrame(open_arr, columns=tickers),
        close=pd.DataFrame(close_arr, columns=tickers),
        volume=pd.DataFrame(vol_arr, columns=tickers),
    )


def test_no_lookahead_bias_in_ticker_features() -> None:
    """Altering future prices or volumes after day `t` must NEVER change features at or before day `t`."""
    mkt = _make_synthetic_market(n_days=150, n_tickers=2)
    close_orig = mkt.close["TK01"].to_numpy().copy()
    vol_orig = mkt.volume["TK01"].to_numpy().copy()

    feats_orig = compute_ticker_features(close_orig, vol_orig)

    t_cutoff = 100
    close_shocked = close_orig.copy()
    vol_shocked = vol_orig.copy()
    close_shocked[t_cutoff + 1 :] *= 5.0
    vol_shocked[t_cutoff + 1 :] *= 10.0

    feats_shocked = compute_ticker_features(close_shocked, vol_shocked)

    np.testing.assert_allclose(
        feats_orig[: t_cutoff + 1],
        feats_shocked[: t_cutoff + 1],
        equal_nan=True,
    )


def test_cross_sectional_scale_zero_mean_and_bounds() -> None:
    mkt = _make_synthetic_market(n_days=140, n_tickers=15)
    panel = build_ml_panel(mkt, horizon=21, step=5)
    assert not panel.empty
    assert list(FEATURE_COLS) == [c for c in panel.columns if c in FEATURE_COLS]

    sample_date = panel["date"].iloc[0]
    sub = panel[panel["date"] == sample_date]
    for col in FEATURE_COLS:
        assert abs(sub[col].mean()) < 1e-6
        assert sub[col].min() >= -4.0 - 1e-9
        assert sub[col].max() <= 4.0 + 1e-9


def test_purged_temporal_split_strictly_chronological_with_embargo() -> None:
    mkt = _make_synthetic_market(n_days=400, n_tickers=10)
    panel = build_ml_panel(mkt, horizon=21, step=5)
    splits = split_temporal_70_20_10(panel, train_ratio=0.70, val_ratio=0.20, purge_days=21)

    max_train_date = splits.train["date"].max()
    min_val_date = splits.val["date"].min()
    max_val_date = splits.val["date"].max()
    min_test_date = splits.test["date"].min()

    assert max_train_date < min_val_date
    assert max_val_date < min_test_date

    # Verify purge removed at least 4 sampling steps (>= 20 trading days) at boundaries
    assert (min_val_date - max_train_date).days >= 20
    assert (min_test_date - max_val_date).days >= 20


def test_extract_current_features_returns_single_day_cross_section() -> None:
    mkt = _make_synthetic_market(n_days=90, n_tickers=10)
    df_today = extract_current_features(mkt)
    assert len(df_today) == 10
    assert (df_today["date"] == mkt.dates[-1]).all()
