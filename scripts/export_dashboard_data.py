# -*- coding: utf-8 -*-
"""
Generate enriched datasets for Streamlit and Power BI:
  - outputs/tables/walkforward_monthly_curves.csv (Cumulative compound curves 2015-2025)
  - outputs/tables/walkforward_monthly_holdings.csv (Month-by-month predictions vs realized returns)
  - outputs/tables/current_month_prediction.csv (Latest available prediction & MTD performance)
"""

from __future__ import annotations
from pathlib import Path
import sys

# Ensure src is importable
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import numpy as np
import pandas as pd

from quant_ml_b3.data_loader import load_local_market_data
from quant_ml_b3.features import FEATURE_COLS, build_ml_panel, extract_current_features
from quant_ml_b3.splits import split_temporal_70_20_10
from quant_ml_b3.models import fit_ml_model, predict_ml_model
from quant_ml_b3.walk_forward import compute_performance_metrics


def generate_dashboard_data() -> None:
    print("[1/4] Loading B3 market data...")
    market = load_local_market_data(ROOT_DIR / "data" / "processed")
    tables_dir = ROOT_DIR / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    start_date = "2015-01-02"
    end_date = None  # Full period through 2026 (all available months)
    models = ("ridge", "xgboost", "lightgbm")
    retrain_every_months = 12
    train_lookback_days = 756
    top_k_hard = 10
    hysteresis_buffer = 20
    top_k_soft = 15
    seed = 42

    sim_mkt = market.slice_dates(start_date, end_date)
    dt_series = pd.Series(sim_mkt.dates)
    month_year = dt_series.dt.to_period("M")
    first_days_mask = month_year != month_year.shift(1)
    rebal_idx = np.where(first_days_mask)[0]
    n_rebal = len(rebal_idx)

    # Pre-start training
    idx_start_full = int(np.where(market.dates == sim_mkt.dates[rebal_idx[0]])[0][0])
    hist_win = market.slice_iloc(np.arange(max(0, idx_start_full - train_lookback_days), idx_start_full))
    panel_init = build_ml_panel(hist_win, horizon=21, step=5)
    splits_init = split_temporal_70_20_10(panel_init, train_ratio=0.70, val_ratio=0.20, purge_days=21)

    active_models = {m: fit_ml_model(m, splits_init, FEATURE_COLS, seed=seed) for m in models}

    held_hard = {m: [] for m in models}
    prev_w_hard = {m: {} for m in models}
    prev_w_soft = {m: {} for m in models}
    rank_weights = np.arange(top_k_soft, 0, -1, dtype=float)
    rank_weights /= rank_weights.sum()

    # Load Ibovespa (BOVA11) benchmark series
    bova_file = ROOT_DIR / "data" / "processed" / "benchmark_bova11.csv"
    bova_series = None
    if bova_file.exists():
        df_bova = pd.read_csv(bova_file)
        df_bova["date"] = pd.to_datetime(df_bova["date"])
        bova_series = df_bova.set_index("date")["close"]

    monthly_records: list[dict] = []
    holdings_records: list[dict] = []
    turnover_records: list[dict] = []
    months_since_train = 0
    close_mat = sim_mkt.close

    print(f"[2/4] Simulating {n_rebal} monthly rebalances with Walk-Forward (2015 to 2026)...")
    for step in range(n_rebal):
        idx_t = int(rebal_idx[step])
        idx_next = int(rebal_idx[step + 1]) if step < n_rebal - 1 else len(sim_mkt.dates) - 1
        if idx_next <= idx_t:
            continue

        dt = sim_mkt.dates[idx_t]
        dt_end = sim_mkt.dates[idx_next]
        full_idx_t = int(np.where(market.dates == dt)[0][0])

        months_since_train += 1
        if step > 0 and retrain_every_months > 0 and months_since_train >= retrain_every_months:
            win_retrain = market.slice_iloc(np.arange(max(0, full_idx_t - train_lookback_days), full_idx_t + 1))
            p_retrain = build_ml_panel(win_retrain, horizon=21, step=5)
            if len(p_retrain) >= 200:
                s_retrain = split_temporal_70_20_10(p_retrain, train_ratio=0.70, val_ratio=0.20, purge_days=21)
                for m in models:
                    active_models[m] = fit_ml_model(m, s_retrain, FEATURE_COLS, seed=seed)
                months_since_train = 0

        win_today = market.slice_iloc(np.arange(max(0, full_idx_t - 260), full_idx_t + 1))
        df_today = extract_current_features(win_today)
        if df_today.empty:
            continue

        p_start = close_mat.iloc[idx_t]
        p_end = close_mat.iloc[idx_next]
        r_univ = float((p_end / p_start - 1.0).dropna().mean())

        r_ibov = 0.0
        if bova_series is not None:
            try:
                p_b0 = float(bova_series.asof(dt))
                p_b1 = float(bova_series.asof(dt_end))
                if p_b0 > 0:
                    r_ibov = float(p_b1 / p_b0 - 1.0)
            except Exception:
                r_ibov = 0.0

        step_record = {
            "date": dt_end.strftime("%Y-%m-%d"),
            "rebal_date": dt.strftime("%Y-%m-%d"),
            "month_label": dt.strftime("%Y-%m"),
            "Universe": r_univ,
            "Ibovespa": r_ibov,
        }
        to_record = {"date": dt_end.strftime("%Y-%m-%d")}

        for m_type in models:
            preds = predict_ml_model(active_models[m_type], df_today)
            ranked = list(preds.sort_values(ascending=False).index)

            # 1. Hard Cut
            buf_set = set(ranked[: min(hysteresis_buffer, len(ranked))])
            kept = [tk for tk in held_hard[m_type] if tk in buf_set]
            new_picks = [tk for tk in ranked if tk not in kept]
            chosen_hard = kept + new_picks[: top_k_hard - len(kept)]
            held_hard[m_type] = chosen_hard
            r_hard = float((p_end[chosen_hard] / p_start[chosen_hard] - 1.0).dropna().mean())
            step_record[f"{m_type.capitalize()}_HardCut"] = r_hard

            w_hard = {tk: 1.0 / len(chosen_hard) for tk in chosen_hard}
            if prev_w_hard[m_type]:
                all_tk = set(w_hard) | set(prev_w_hard[m_type])
                to_val = 0.5 * sum(abs(w_hard.get(t, 0.0) - prev_w_hard[m_type].get(t, 0.0)) for t in all_tk)
                to_record[f"{m_type.capitalize()}_HardCut_TO"] = to_val
            else:
                to_record[f"{m_type.capitalize()}_HardCut_TO"] = 1.0
            prev_w_hard[m_type] = w_hard

            for rank_pos, tk in enumerate(chosen_hard, 1):
                s_ret = float(p_end[tk] / p_start[tk] - 1.0) if not np.isnan(p_start[tk]) and not np.isnan(p_end[tk]) else 0.0
                w_val = 1.0 / len(chosen_hard)
                holdings_records.append({
                    "date": dt_end.strftime("%Y-%m-%d"),
                    "rebal_date": dt.strftime("%Y-%m-%d"),
                    "month_label": dt.strftime("%Y-%m"),
                    "model": m_type.capitalize(),
                    "scheme": "Hard Cut (Top 10)",
                    "ticker": tk,
                    "rank": rank_pos,
                    "predicted_score": round(float(preds.get(tk, 0.0)), 4),
                    "weight_pct": round(w_val * 100.0, 2),
                    "price_start": round(float(p_start[tk]), 2),
                    "price_end": round(float(p_end[tk]), 2),
                    "stock_return_pct": round(s_ret * 100.0, 2),
                    "contrib_return_pct": round(w_val * s_ret * 100.0, 2),
                    "market_return_pct": round(r_univ * 100.0, 2),
                    "ibov_return_pct": round(r_ibov * 100.0, 2),
                    "beat_market": bool(s_ret > r_univ),
                })

            # 2. Soft Weighting
            chosen_soft = ranked[: min(top_k_soft, len(ranked))]
            m_len = len(chosen_soft)
            w_vec = rank_weights if m_len == top_k_soft else np.arange(m_len, 0, -1, dtype=float) / np.arange(m_len, 0, -1, dtype=float).sum()
            w_soft = {tk: float(w_vec[i]) for i, tk in enumerate(chosen_soft)}
            r_soft = float(sum(w_soft[tk] * (p_end[tk] / p_start[tk] - 1.0) for tk in chosen_soft if not np.isnan(p_start[tk]) and not np.isnan(p_end[tk])))
            step_record[f"{m_type.capitalize()}_SoftWeight"] = r_soft

            if prev_w_soft[m_type]:
                all_tk_sf = set(w_soft) | set(prev_w_soft[m_type])
                to_val_sf = 0.5 * sum(abs(w_soft.get(t, 0.0) - prev_w_soft[m_type].get(t, 0.0)) for t in all_tk_sf)
                to_record[f"{m_type.capitalize()}_SoftWeight_TO"] = to_val_sf
            else:
                to_record[f"{m_type.capitalize()}_SoftWeight_TO"] = 1.0
            prev_w_soft[m_type] = w_soft

            for rank_pos, tk in enumerate(chosen_soft, 1):
                s_ret = float(p_end[tk] / p_start[tk] - 1.0) if not np.isnan(p_start[tk]) and not np.isnan(p_end[tk]) else 0.0
                w_val = w_soft[tk]
                holdings_records.append({
                    "date": dt_end.strftime("%Y-%m-%d"),
                    "rebal_date": dt.strftime("%Y-%m-%d"),
                    "month_label": dt.strftime("%Y-%m"),
                    "model": m_type.capitalize(),
                    "scheme": "Soft Weighting (Top 15)",
                    "ticker": tk,
                    "rank": rank_pos,
                    "predicted_score": round(float(preds.get(tk, 0.0)), 4),
                    "weight_pct": round(w_val * 100.0, 2),
                    "price_start": round(float(p_start[tk]), 2),
                    "price_end": round(float(p_end[tk]), 2),
                    "stock_return_pct": round(s_ret * 100.0, 2),
                    "contrib_return_pct": round(w_val * s_ret * 100.0, 2),
                    "market_return_pct": round(r_univ * 100.0, 2),
                    "ibov_return_pct": round(r_ibov * 100.0, 2),
                    "beat_market": bool(s_ret > r_univ),
                })

        monthly_records.append(step_record)
        turnover_records.append(to_record)

    df_monthly = pd.DataFrame(monthly_records).sort_values("date").reset_index(drop=True)
    df_to = pd.DataFrame(turnover_records).sort_values("date").reset_index(drop=True)
    df_holdings = pd.DataFrame(holdings_records).sort_values(["date", "model", "scheme", "rank"]).reset_index(drop=True)

    # Compute cumulative compound curves
    df_cum = pd.DataFrame({"date": df_monthly["date"], "month_label": df_monthly["month_label"]})
    for col in df_monthly.columns:
        if col not in ("date", "rebal_date", "month_label"):
            df_cum[col] = (np.cumprod(1.0 + df_monthly[col]) - 1.0) * 100.0

    # Compute full summary metrics (2015-2026)
    univ_m = compute_performance_metrics(df_monthly["Universe"])
    ibov_m = compute_performance_metrics(df_monthly["Ibovespa"]) if "Ibovespa" in df_monthly.columns else {}
    summary_rows = [
        {
            "Modelo": "Buy & Hold (Universo 1/N)",
            "Esquema": "Equiponderado (1/N)",
            **univ_m,
            "Turnover_Medio_Mensal_pct": 0.0,
            "Alpha_vs_Universo_pct": 0.0,
        },
        {
            "Modelo": "Ibovespa (BOVA11)",
            "Esquema": "Market-Cap Weighted",
            **ibov_m,
            "Turnover_Medio_Mensal_pct": 0.0,
            "Alpha_vs_Universo_pct": round(ibov_m.get("CAGR_pct", 0.0) - univ_m["CAGR_pct"], 2),
        },
    ]
    for m_type in models:
        c_hard = f"{m_type.capitalize()}_HardCut"
        c_soft = f"{m_type.capitalize()}_SoftWeight"
        to_hd_col = f"{m_type.capitalize()}_HardCut_TO"
        to_sf_col = f"{m_type.capitalize()}_SoftWeight_TO"

        m_hd = compute_performance_metrics(df_monthly[c_hard])
        m_sf = compute_performance_metrics(df_monthly[c_soft])

        avg_to_hd = float(df_to[to_hd_col].iloc[1:].mean() * 100.0) if len(df_to) > 1 else 0.0
        avg_to_sf = float(df_to[to_sf_col].iloc[1:].mean() * 100.0) if len(df_to) > 1 else 0.0

        summary_rows.append(
            {
                "Modelo": f"ML: {m_type.upper()}",
                "Esquema": f"Hard Cut (Top {top_k_hard} 1/N com Histerese)",
                **m_hd,
                "Turnover_Medio_Mensal_pct": round(avg_to_hd, 1),
                "Alpha_vs_Universo_pct": round(m_hd["CAGR_pct"] - univ_m["CAGR_pct"], 2),
            }
        )
        summary_rows.append(
            {
                "Modelo": f"ML: {m_type.upper()}",
                "Esquema": f"Soft Weighting (Top {top_k_soft} Rank Linear)",
                **m_sf,
                "Turnover_Medio_Mensal_pct": round(avg_to_sf, 1),
                "Alpha_vs_Universo_pct": round(m_sf["CAGR_pct"] - univ_m["CAGR_pct"], 2),
            }
        )
    df_summary = pd.DataFrame(summary_rows)

    print("[3/4] Generating Current Month live prediction...")
    # Generate prediction for the latest available date
    latest_win = market.slice_iloc(np.arange(max(0, len(market.dates) - 260), len(market.dates)))
    df_latest = extract_current_features(latest_win)
    latest_date_str = str(market.dates[-1].date())

    current_preds_list: list[dict] = []
    if not df_latest.empty:
        # Determine start of latest month
        latest_month = market.dates[-1].to_period("M")
        month_dates = [d for d in market.dates if d.to_period("M") == latest_month]
        first_day_of_month = month_dates[0] if month_dates else market.dates[-1]
        p_start_m = market.close.loc[market.dates == first_day_of_month].iloc[0]
        p_curr_m = market.close.iloc[-1]
        mkt_mtd = float((p_curr_m / p_start_m - 1.0).dropna().mean())

        ibov_mtd = 0.0
        if bova_series is not None:
            try:
                p_b0 = float(bova_series.asof(first_day_of_month))
                p_b1 = float(bova_series.asof(market.dates[-1]))
                if p_b0 > 0:
                    ibov_mtd = float(p_b1 / p_b0 - 1.0)
            except Exception:
                ibov_mtd = 0.0

        for m_type in models:
            preds = predict_ml_model(active_models[m_type], df_latest)
            ranked = list(preds.sort_values(ascending=False).index)

            # Soft top 15
            chosen_top15 = ranked[:top_k_soft]
            w_vec = rank_weights if len(chosen_top15) == top_k_soft else np.arange(len(chosen_top15), 0, -1, dtype=float) / np.arange(len(chosen_top15), 0, -1, dtype=float).sum()
            for rank_pos, tk in enumerate(chosen_top15, 1):
                p0 = float(p_start_m[tk]) if tk in p_start_m else np.nan
                p1 = float(p_curr_m[tk]) if tk in p_curr_m else np.nan
                ret_mtd = float(p1 / p0 - 1.0) if not np.isnan(p0) and not np.isnan(p1) and p0 > 0 else 0.0
                current_preds_list.append({
                    "as_of_date": latest_date_str,
                    "month_start_date": str(first_day_of_month.date()),
                    "model": m_type.capitalize(),
                    "scheme": "Soft Weighting (Top 15)",
                    "rank": rank_pos,
                    "ticker": tk,
                    "predicted_alpha_score": round(float(preds.get(tk, 0.0)), 4),
                    "suggested_weight_pct": round(float(w_vec[rank_pos - 1]) * 100.0, 2),
                    "price_month_start": round(p0, 2) if not np.isnan(p0) else None,
                    "price_current": round(p1, 2) if not np.isnan(p1) else None,
                    "return_mtd_pct": round(ret_mtd * 100.0, 2),
                    "market_mtd_pct": round(mkt_mtd * 100.0, 2),
                    "ibov_mtd_pct": round(ibov_mtd * 100.0, 2),
                    "beat_market_mtd": bool(ret_mtd > mkt_mtd),
                })

    df_current_preds = pd.DataFrame(current_preds_list)

    print("[4/4] Saving datasets to outputs/tables/...")
    df_monthly.to_csv(tables_dir / "walkforward_monthly_returns.csv", index=False)
    df_cum.to_csv(tables_dir / "walkforward_monthly_curves.csv", index=False)
    df_holdings.to_csv(tables_dir / "walkforward_monthly_holdings.csv", index=False)
    df_current_preds.to_csv(tables_dir / "current_month_prediction.csv", index=False)
    df_summary.to_csv(tables_dir / "comparison_weighting_schemes.csv", index=False)

    print(f"Done! Saved:")
    print(f"  - {tables_dir / 'walkforward_monthly_returns.csv'} ({len(df_monthly)} months)")
    print(f"  - {tables_dir / 'walkforward_monthly_curves.csv'} ({len(df_cum)} months)")
    print(f"  - {tables_dir / 'walkforward_monthly_holdings.csv'} ({len(df_holdings)} stock-month observations)")
    print(f"  - {tables_dir / 'current_month_prediction.csv'} ({len(df_current_preds)} current recommendations)")
    print(f"  - {tables_dir / 'comparison_weighting_schemes.csv'} ({len(df_summary)} strategy summaries)")


if __name__ == "__main__":
    generate_dashboard_data()
