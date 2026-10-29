# -*- coding: utf-8 -*-
"""
Update local market dataset with live prices up to today (October 2026).
Appends recent trading days (2026-09-01 to today) for all 41 IBrX 50 tickers and BRAX11.
"""

from __future__ import annotations
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import numpy as np
import pandas as pd
import yfinance as yf

from quant_ml_b3.data_loader import download_b3_universe_yfinance, load_local_market_data, validate_market_data


def update_data() -> None:
    data_dir = ROOT_DIR / "data" / "processed"
    print("[1/5] Loading existing local market data...")
    mkt_existing = load_local_market_data(data_dir)
    tickers = mkt_existing.tickers

    # Download from 2026-08-25 to today to have clean overlap
    print(f"[2/5] Downloading live prices for {len(tickers)} tickers from 2026-08-25 to today...")
    live_mkt = download_b3_universe_yfinance(tickers, start="2026-08-25", end=None)
    print(f"Downloaded {len(live_mkt.dates)} trading days up to {live_mkt.dates[-1].strftime('%Y-%m-%d')}.")

    # Load existing CSVs
    open_file = data_dir / "ibx50_open_adj.csv"
    close_file = data_dir / "ibx50_close_adj.csv"
    vol_file = data_dir / "ibx50_volume.csv"

    df_open_old = pd.read_csv(open_file)
    df_close_old = pd.read_csv(close_file)
    df_vol_old = pd.read_csv(vol_file)

    cut_date = "2026-08-25"
    df_open_kept = df_open_old[df_open_old["date"] < cut_date].copy()
    df_close_kept = df_close_old[df_close_old["date"] < cut_date].copy()
    df_vol_kept = df_vol_old[df_vol_old["date"] < cut_date].copy()

    # Format live frames
    live_open = live_mkt.open.copy()
    live_open.insert(0, "date", live_mkt.dates.strftime("%Y-%m-%d"))

    live_close = live_mkt.close.copy()
    live_close.insert(0, "date", live_mkt.dates.strftime("%Y-%m-%d"))

    live_vol = live_mkt.volume.copy()
    live_vol.insert(0, "date", live_mkt.dates.strftime("%Y-%m-%d"))

    # Ensure column ordering matches
    cols = df_close_old.columns.tolist()
    live_open = live_open[[c for c in cols if c in live_open.columns]]
    live_close = live_close[[c for c in cols if c in live_close.columns]]
    live_vol = live_vol[[c for c in cols if c in live_vol.columns]]

    df_open_merged = pd.concat([df_open_kept, live_open], ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    df_close_merged = pd.concat([df_close_kept, live_close], ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    df_vol_merged = pd.concat([df_vol_kept, live_vol], ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    print(f"[3/5] Merged dataset now has {len(df_close_merged)} days from {df_close_merged['date'].iloc[0]} to {df_close_merged['date'].iloc[-1]}.")

    # Update benchmark_brax11.csv
    print("[4/5] Updating benchmark (BRAX11.SA)...")
    brax_file = data_dir / "benchmark_brax11.csv"
    df_brax_old = pd.read_csv(brax_file)
    df_brax_kept = df_brax_old[df_brax_old["date"] < cut_date].copy()

    raw_brax = yf.download("BRAX11.SA", start=cut_date, auto_adjust=False, progress=False)
    if not raw_brax.empty:
        idx_brax = pd.to_datetime(raw_brax.index).strftime("%Y-%m-%d")
        close_series = raw_brax["Adj Close"].squeeze() if "Adj Close" in raw_brax else raw_brax["Close"].squeeze()
        df_brax_new = pd.DataFrame({"date": idx_brax, "close": close_series.values})
        df_brax_merged = pd.concat([df_brax_kept, df_brax_new], ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        df_brax_merged.to_csv(brax_file, index=False)
        print(f"Benchmark updated to {df_brax_merged['date'].iloc[-1]} ({len(df_brax_merged)} days).")

    # Save merged files
    df_open_merged.to_csv(open_file, index=False)
    df_close_merged.to_csv(close_file, index=False)
    df_vol_merged.to_csv(vol_file, index=False)

    print("[5/5] Validating updated dataset...")
    mkt_updated = load_local_market_data(data_dir)
    report = validate_market_data(mkt_updated)
    print(report.summary())
    print(f"Successfully updated market data to {mkt_updated.dates[-1].strftime('%Y-%m-%d')}!")


if __name__ == "__main__":
    update_data()
