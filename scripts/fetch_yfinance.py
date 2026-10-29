"""
Optional CLI utility to refresh the B3 IBrX 50 historical dataset directly via Yahoo Finance (`yfinance`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from quant_ml_b3.data_loader import download_b3_universe_yfinance, load_local_market_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh B3 IBrX 50 market data from Yahoo Finance.")
    parser.add_argument("--start", default="2012-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT_DIR / "data" / "processed"),
        help="Output directory for CSV matrices",
    )
    args = parser.parse_args()

    existing = load_local_market_data(args.out_dir)
    tickers = existing.tickers
    print(f"Downloading {len(tickers)} B3 tickers ({args.start} to {args.end}) via yfinance...")
    market = download_b3_universe_yfinance(tickers, start=args.start, end=args.end)

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for name, mat in [("ibx50_open_adj.csv", market.open), ("ibx50_close_adj.csv", market.close), ("ibx50_volume.csv", market.volume)]:
        df_out = mat.copy()
        df_out.insert(0, "date", market.dates.strftime("%Y-%m-%d"))
        df_out.to_csv(out_path / name, index=False)

    print(f"Saved {len(market.dates)} valid trading days across {len(market.tickers)} tickers to {out_path}.")


if __name__ == "__main__":
    main()
