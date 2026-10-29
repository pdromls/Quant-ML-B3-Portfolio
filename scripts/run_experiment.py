"""
Main CLI entrypoint to reproduce the Quantitative Machine Learning Experiment on the B3 IBrX 50 Universe:
  - Part A: Data Science Evaluation (Chronological 70% Train / 20% Val / 10% Test with 21-day Purge)
  - Part B: Event-Driven Portfolio Backtesting (Frozen Out-of-Sample 10% & 11-Year Walk-Forward 2015-2025)
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from quant_ml_b3 import (
    FEATURE_COLS,
    BuyAndHoldStrategy,
    CrossSectionalMLStrategy,
    benchmark_equity_curve,
    build_ml_panel,
    compute_financial_metrics,
    extract_feature_importance,
    fit_ml_model,
    load_benchmark_series,
    load_local_market_data,
    run_backtest,
    split_temporal_70_20_10,
)
from quant_ml_b3.plotting import (
    plot_drawdown_curve,
    plot_equity_and_drawdown,
    plot_equity_curve,
    plot_feature_importance,
    plot_oos_vs_walkforward_bars,
)


def main() -> None:
    data_dir = ROOT_DIR / "data" / "processed"
    tables_dir = ROOT_DIR / "outputs" / "tables"
    figures_dir = ROOT_DIR / "outputs" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("1. LOADING B3 IBrX 50 HIGH-LIQUIDITY UNIVERSE")
    print("=" * 78)
    market_full = load_local_market_data(data_dir)
    print(
        f"Universe loaded: {len(market_full.tickers)} stocks x {len(market_full.dates)} trading days "
        f"({market_full.dates[0].date()} to {market_full.dates[-1].date()})\n"
    )

    # ==========================================================================
    # PART A: DATA SCIENCE EVALUATION (CHRONOLOGICAL 70 / 20 / 10 + 21D PURGE)
    # ==========================================================================
    print("=" * 78)
    print("2. CHRONOLOGICAL 70% TRAIN / 20% VAL / 10% TEST SPLIT WITH 21-DAY PURGE")
    print("=" * 78)
    market_15_25 = market_full.slice_dates("2015-01-02", "2025-12-31")
    panel_ml = build_ml_panel(market_15_25, horizon=21, step=5)
    splits = split_temporal_70_20_10(panel_ml, train_ratio=0.70, val_ratio=0.20, purge_days=21)

    tr_p = splits.meta["train_period"]
    va_p = splits.meta["val_period"]
    te_p = splits.meta["test_period"]
    print(f"Total cross-sectional observations (date x stock): {len(panel_ml):,} rows")
    print(
        f" - Train (70%)     : {tr_p[0].date()} to {tr_p[1].date()} "
        f"({splits.meta['n_dates_train']} dates | {len(splits.train):,} samples)"
    )
    print(
        f" - Validation (20%): {va_p[0].date()} to {va_p[1].date()} "
        f"({splits.meta['n_dates_val']} dates | {len(splits.val):,} samples) [21d Purge]"
    )
    print(
        f" - Test (10%)      : {te_p[0].date()} to {te_p[1].date()} "
        f"({splits.meta['n_dates_test']} dates | {len(splits.test):,} samples) [21d Purge]\n"
    )

    fit_ridge = fit_ml_model("ridge", splits, FEATURE_COLS, seed=42)
    fit_xgb = fit_ml_model("xgboost", splits, FEATURE_COLS, seed=42)
    fit_lgb = fit_ml_model("lightgbm", splits, FEATURE_COLS, seed=42)

    print("Hyperparameters selected on Validation (20%):")
    print(f" - Ridge Baseline : optimal lambda = {fit_ridge.best_params['best_lambda']}")
    print(
        f" - XGBoost        : nrounds = {fit_xgb.best_params['best_iter']}, "
        f"max_depth = {fit_xgb.best_params['best_depth']}, eta = {fit_xgb.best_params['best_eta']}"
    )
    print(
        f" - LightGBM       : nrounds = {fit_lgb.best_params['best_iter']}, "
        f"num_leaves = {fit_lgb.best_params['best_leaves']}, lr = {fit_lgb.best_params['best_lr']}\n"
    )

    metrics_ml_df = pd.concat(
        [fit_ridge.metrics_table, fit_xgb.metrics_table, fit_lgb.metrics_table],
        ignore_index=True,
    )
    print("=== MACHINE LEARNING METRICS TABLE (70% TRAIN / 20% VAL / 10% TEST) ===")
    print(metrics_ml_df.to_string(index=False))
    metrics_ml_df.to_csv(tables_dir / "ml_metrics_70_20_10.csv", index=False)

    imp_df = extract_feature_importance(fit_ridge, fit_xgb, fit_lgb)
    print("\n=== CROSS-SECTIONAL FEATURE IMPORTANCE ACROSS MODELS ===")
    print(imp_df.to_string(index=False))
    imp_df.to_csv(tables_dir / "feature_importance.csv", index=False)
    plot_feature_importance(imp_df, figures_dir / "01_feature_importance.png")

    # ==========================================================================
    # PART B: EVENT-DRIVEN BACKTEST (2015-2025 WALK-FORWARD & 10% PURE OOS)
    # ==========================================================================
    print("\n" + "=" * 78)
    print("3. EVENT-DRIVEN WALK-FORWARD BACKTEST (2015-2025 | 15 bps Transaction Cost)")
    print("=" * 78)
    cost_15bps = 0.0015
    initial_capital = 100_000.0
    start_wf = "2015-01-02"
    end_wf = "2025-12-31"

    bt_bh = run_backtest(BuyAndHoldStrategy(), market_full, cost_15bps, initial_capital, start_wf, end_wf)
    bt_ridge = run_backtest(
        CrossSectionalMLStrategy("ridge", top_k=10, retrain_every_months=12),
        market_full,
        cost_15bps,
        initial_capital,
        start_wf,
        end_wf,
    )
    bt_xgb = run_backtest(
        CrossSectionalMLStrategy("xgboost", top_k=10, retrain_every_months=12),
        market_full,
        cost_15bps,
        initial_capital,
        start_wf,
        end_wf,
    )
    bt_lgb = run_backtest(
        CrossSectionalMLStrategy("lightgbm", top_k=10, retrain_every_months=12),
        market_full,
        cost_15bps,
        initial_capital,
        start_wf,
        end_wf,
    )

    wf_results = [bt_bh, bt_ridge, bt_xgb, bt_lgb]
    wf_rows = [{"strategy": r.strategy_name, **r.metrics} for r in wf_results]
    curves_wf = {r.strategy_name: r.equity for r in wf_results}

    # Include passive ETF benchmark BRAX11
    brax_series = load_benchmark_series(data_dir)
    brax_eq = benchmark_equity_curve(brax_series, bt_bh.equity["date"], initial_capital=initial_capital)
    brax_metrics = compute_financial_metrics(brax_eq, trades_df=None, initial_capital=initial_capital)
    wf_rows.append({"strategy": "BRAX11 (Indice Passivo)", **brax_metrics})
    curves_wf["BRAX11 (Indice Passivo)"] = brax_eq

    df_wf = pd.DataFrame(wf_rows)
    print(
        df_wf[
            [
                "strategy",
                "final_equity",
                "total_return",
                "annual_return",
                "annual_vol",
                "sharpe",
                "max_drawdown",
                "calmar",
                "win_rate",
            ]
        ].to_string(index=False)
    )
    df_wf.to_csv(tables_dir / "backtest_walkforward_2015_2025.csv", index=False)
    plot_equity_and_drawdown(curves_wf, figures_dir / "02_walkforward_equity_drawdown.png")
    plot_equity_curve(curves_wf, figures_dir / "02_walkforward_equity.png")
    plot_drawdown_curve(curves_wf, figures_dir / "03_walkforward_drawdown.png")

    print("\n" + "=" * 78)
    print("4. PURE OUT-OF-SAMPLE BACKTEST (10% TEST WINDOW: NOV/2024 - DEC/2025)")
    print("=" * 78)
    start_oos = te_p[0]
    end_oos = "2025-12-31"

    bt_bh_oos = run_backtest(BuyAndHoldStrategy(), market_full, cost_15bps, initial_capital, start_oos, end_oos)
    bt_ridge_oos = run_backtest(
        CrossSectionalMLStrategy("ridge", top_k=10, retrain_every_months=0),
        market_full,
        cost_15bps,
        initial_capital,
        start_oos,
        end_oos,
    )
    bt_xgb_oos = run_backtest(
        CrossSectionalMLStrategy("xgboost", top_k=10, retrain_every_months=0),
        market_full,
        cost_15bps,
        initial_capital,
        start_oos,
        end_oos,
    )
    bt_lgb_oos = run_backtest(
        CrossSectionalMLStrategy("lightgbm", top_k=10, retrain_every_months=0),
        market_full,
        cost_15bps,
        initial_capital,
        start_oos,
        end_oos,
    )

    oos_results = [bt_bh_oos, bt_ridge_oos, bt_xgb_oos, bt_lgb_oos]
    df_oos = pd.DataFrame([{"strategy": r.strategy_name, **r.metrics} for r in oos_results])
    print(
        df_oos[
            [
                "strategy",
                "final_equity",
                "total_return",
                "annual_return",
                "annual_vol",
                "sharpe",
                "max_drawdown",
                "win_rate",
            ]
        ].to_string(index=False)
    )
    df_oos.to_csv(tables_dir / "backtest_oos_test10pct.csv", index=False)

    plot_oos_vs_walkforward_bars(df_oos, df_wf, figures_dir / "03_oos_vs_walkforward_profit.png")
    plot_oos_vs_walkforward_bars(df_oos, df_wf, figures_dir / "04_oos_vs_walkforward_profit.png")
    print(f"\nAll tables saved to {tables_dir} and figures saved to {figures_dir}!")


if __name__ == "__main__":
    main()
