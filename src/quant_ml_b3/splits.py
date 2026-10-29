"""
Chronological 70% Train / 20% Validation / 10% Test split with Purging (Embargo)
to prevent forward-return label leakage across partition boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Container for purged chronological train, validation, and test partitions."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    meta: dict


def split_temporal_70_20_10(
    panel: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    purge_days: int = 21,
) -> TemporalSplit:
    """
    Divide a cross-sectional panel chronologically into 70% Train, 20% Validation, and 10% Test,
    applying a temporal purge (embargo) of `floor(purge_days / 5)` sampling steps before the
    Train->Val and Val->Test boundaries so 21-day forward returns never overlap across splits.
    """
    unique_dates = sorted(panel["date"].unique())
    n_d = len(unique_dates)
    if n_d < 20:
        raise ValueError("Insufficient unique dates for 70/20/10 chronological split.")

    idx_train_end = math.floor(n_d * train_ratio)
    idx_val_end = math.floor(n_d * (train_ratio + val_ratio))

    train_dates_all = unique_dates[:idx_train_end]
    val_dates_all = unique_dates[idx_train_end:idx_val_end]
    test_dates_all = unique_dates[idx_val_end:]

    n_purge_steps = max(1, purge_days // 5)

    if len(train_dates_all) > n_purge_steps + 5:
        train_dates = train_dates_all[: len(train_dates_all) - n_purge_steps]
    else:
        train_dates = train_dates_all

    if len(val_dates_all) > n_purge_steps + 3:
        val_dates = val_dates_all[: len(val_dates_all) - n_purge_steps]
    else:
        val_dates = val_dates_all

    test_dates = test_dates_all

    train_df = panel[panel["date"].isin(train_dates)].reset_index(drop=True)
    val_df = panel[panel["date"].isin(val_dates)].reset_index(drop=True)
    test_df = panel[panel["date"].isin(test_dates)].reset_index(drop=True)

    meta = {
        "train_period": (pd.Timestamp(min(train_dates)), pd.Timestamp(max(train_dates))),
        "val_period": (pd.Timestamp(min(val_dates)), pd.Timestamp(max(val_dates))),
        "test_period": (pd.Timestamp(min(test_dates)), pd.Timestamp(max(test_dates))),
        "n_dates_train": len(train_dates),
        "n_dates_val": len(val_dates),
        "n_dates_test": len(test_dates),
    }

    return TemporalSplit(train=train_df, val=val_df, test=test_df, meta=meta)
