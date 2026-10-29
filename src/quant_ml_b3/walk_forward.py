"""
Walk-Forward Validation and Periodic Retraining Engine.

Provides an automated Walk-Forward simulator to evaluate:
  1. Dynamic Models with periodic retraining (e.g., annual / 12-month expanding lookback)
  2. Static / Frozen Models (trained once on initial historical data and frozen)
  3. Buy & Hold Benchmark (Equal-weighted universe 1/N)

This directly quantifies the impact of Concept Drift and proves why periodic
walk-forward retraining prevents out-of-sample signal degradation.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .data_loader import MarketData
from .features import FEATURE_COLS, build_ml_panel, extract_current_features
from .models import MLModelArtifact, fit_ml_model, predict_ml_model
from .splits import split_temporal_70_20_10


@dataclass
class WalkForwardResult:
    """Container for Walk-Forward simulation output."""

    monthly_returns: pd.DataFrame
    cumulative_returns: pd.DataFrame
    summary_metrics: pd.DataFrame
    retrain_log: list[dict]


@dataclass
class WeightingComparisonResult:
    """Container for Portfolio Weighting Scheme (Hard Cut vs. Soft Weight) comparison."""

    monthly_returns: pd.DataFrame
    cumulative_returns: pd.DataFrame
    summary_metrics: pd.DataFrame
    turnovers: pd.DataFrame


def compute_performance_metrics(
    returns: pd.Series,
    rf: float = 0.0,
) -> dict[str, float]:
    """
    Compute annualized performance metrics from monthly returns:
      - Total Cumulative Return (%)
      - CAGR (% p.a.)
      - Annualized Volatility (%)
      - Sharpe Ratio (Rf = rf)
      - Maximum Drawdown (%)
    """
    n_months = len(returns)
    if n_months == 0:
        return {
            "Retorno_Acumulado_pct": 0.0,
            "CAGR_pct": 0.0,
            "Volatilidade_Anualizada_pct": 0.0,
            "Sharpe_Ratio": 0.0,
            "Max_Drawdown_pct": 0.0,
        }

    cum_series = np.cumprod(1.0 + returns)
    total_ret = float(cum_series.iloc[-1] - 1.0)
    years = n_months / 12.0
    cagr = float((1.0 + total_ret) ** (1.0 / years) - 1.0) if (years > 0 and total_ret > -1.0) else 0.0
    vol_ann = float(returns.std(ddof=1) * np.sqrt(12.0))
    sharpe = float((cagr - rf) / vol_ann) if vol_ann > 1e-6 else 0.0

    peaks = cum_series.cummax()
    drawdowns = (cum_series - peaks) / peaks
    max_dd = float(drawdowns.min())

    return {
        "Retorno_Acumulado_pct": round(total_ret * 100.0, 2),
        "CAGR_pct": round(cagr * 100.0, 2),
        "Volatilidade_Anualizada_pct": round(vol_ann * 100.0, 2),
        "Sharpe_Ratio": round(sharpe, 2),
        "Max_Drawdown_pct": round(max_dd * 100.0, 2),
    }


def run_walk_forward_simulation(
    market: MarketData,
    start_date: str | pd.Timestamp = "2015-01-02",
    end_date: str | pd.Timestamp = "2025-12-31",
    models: tuple[str, ...] = ("ridge", "xgboost", "lightgbm"),
    retrain_every_months: int = 12,
    train_lookback_days: int = 756,
    top_k: int = 10,
    hysteresis_buffer: int = 20,
    seed: int = 42,
    verbose: bool = False,
) -> WalkForwardResult:
    """
    Execute a Walk-Forward simulation from `start_date` to `end_date`:
      - Rebalances on the first trading day of each calendar month.
      - Retrains dynamic models every `retrain_every_months` using the preceding `train_lookback_days`.
      - Simultaneously runs frozen models (fixed at the initial pre-start training state).
      - Tracks realistic monthly portfolio returns with hysteresis buffer (Top 20).
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    sim_mkt = market.slice_dates(start_ts, end_ts)

    months = sim_mkt.dates.to_period("M")
    rebal_idx = np.where(months != np.roll(months, 1))[0]
    n_rebal = len(rebal_idx) - 1
    if n_rebal < 2:
        raise ValueError("Insufficient monthly rebalance dates in specified period.")

    # 1. Fit initial models on pre-start historical window (strictly preceding start_date)
    idx_start_full = int(np.where(market.dates == sim_mkt.dates[rebal_idx[0]])[0][0])
    hist_win = market.slice_iloc(np.arange(max(0, idx_start_full - train_lookback_days), idx_start_full))
    panel_init = build_ml_panel(hist_win, horizon=21, step=5)
    if len(panel_init) < 200:
        raise ValueError(f"Insufficient pre-start panel rows ({len(panel_init)}) for initial fit.")

    splits_init = split_temporal_70_20_10(panel_init, train_ratio=0.70, val_ratio=0.20, purge_days=21)

    active_wf_models: dict[str, MLModelArtifact] = {}
    frozen_models: dict[str, MLModelArtifact] = {}

    for m_type in models:
        fit_obj = fit_ml_model(m_type, splits_init, FEATURE_COLS, seed=seed)
        active_wf_models[m_type] = fit_obj
        frozen_models[m_type] = fit_obj

    retrain_log: list[dict] = [
        {
            "rebal_step": 0,
            "date": sim_mkt.dates[rebal_idx[0]],
            "action": "Initial Training",
            "samples": len(panel_init),
        }
    ]

    # Track portfolio holdings to apply hysteresis buffer
    held_wf: dict[str, list[str]] = {m_type: [] for m_type in models}
    held_frozen: dict[str, list[str]] = {m_type: [] for m_type in models}

    monthly_records: list[dict] = []
    months_since_train = 0

    close_mat = sim_mkt.close

    for step in range(n_rebal):
        idx_t = int(rebal_idx[step])
        idx_next = int(rebal_idx[step + 1])
        dt = sim_mkt.dates[idx_t]
        full_idx_t = int(np.where(market.dates == dt)[0][0])

        # Check periodic Walk-Forward retraining trigger (aligns with December retrain schedule)
        months_since_train += 1
        if step > 0 and retrain_every_months > 0 and months_since_train >= retrain_every_months:
            if verbose:
                print(f"[Walk-Forward] Retraining models at step {step} ({dt.strftime('%Y-%m-%d')})...")
            win_retrain = market.slice_iloc(np.arange(max(0, full_idx_t - train_lookback_days), full_idx_t + 1))
            p_retrain = build_ml_panel(win_retrain, horizon=21, step=5)
            if len(p_retrain) >= 200:
                s_retrain = split_temporal_70_20_10(p_retrain, train_ratio=0.70, val_ratio=0.20, purge_days=21)
                for m_type in models:
                    active_wf_models[m_type] = fit_ml_model(m_type, s_retrain, FEATURE_COLS, seed=seed)
                retrain_log.append(
                    {
                        "rebal_step": step,
                        "date": dt,
                        "action": "Periodic Walk-Forward Retrain",
                        "samples": len(p_retrain),
                    }
                )
                months_since_train = 0

        # Extract causal cross-sectional features for today
        win_today = market.slice_iloc(np.arange(max(0, full_idx_t - 260), full_idx_t + 1))
        df_today = extract_current_features(win_today)
        if df_today.empty:
            continue

        p_start = close_mat.iloc[idx_t]
        p_end = close_mat.iloc[idx_next]

        # Universe (Equal-weighted Buy & Hold baseline)
        r_univ = float((p_end / p_start - 1.0).dropna().mean())

        step_record = {
            "date": sim_mkt.dates[idx_next],
            "Universe": r_univ,
        }

        # Select Top-K with hysteresis for each model
        for m_type in models:
            # 1. Walk-Forward dynamic model
            preds_wf = predict_ml_model(active_wf_models[m_type], df_today)
            ranked_wf = list(preds_wf.sort_values(ascending=False).index)
            buf_set_wf = set(ranked_wf[: min(hysteresis_buffer, len(ranked_wf))])
            kept_wf = [tk for tk in held_wf[m_type] if tk in buf_set_wf]
            new_wf = [tk for tk in ranked_wf if tk not in kept_wf]
            chosen_wf = kept_wf + new_wf[: top_k - len(kept_wf)]
            held_wf[m_type] = chosen_wf
            r_wf = float((p_end[chosen_wf] / p_start[chosen_wf] - 1.0).dropna().mean())
            step_record[f"{m_type.capitalize()}_WF"] = r_wf

            # 2. Frozen static model
            preds_frz = predict_ml_model(frozen_models[m_type], df_today)
            ranked_frz = list(preds_frz.sort_values(ascending=False).index)
            buf_set_frz = set(ranked_frz[: min(hysteresis_buffer, len(ranked_frz))])
            kept_frz = [tk for tk in held_frozen[m_type] if tk in buf_set_frz]
            new_frz = [tk for tk in ranked_frz if tk not in kept_frz]
            chosen_frz = kept_frz + new_frz[: top_k - len(kept_frz)]
            held_frozen[m_type] = chosen_frz
            r_frz = float((p_end[chosen_frz] / p_start[chosen_frz] - 1.0).dropna().mean())
            step_record[f"{m_type.capitalize()}_Frozen"] = r_frz

        monthly_records.append(step_record)

    df_monthly = pd.DataFrame(monthly_records).sort_values("date").reset_index(drop=True)

    # Compute cumulative equity curves
    df_cum = pd.DataFrame({"date": df_monthly["date"]})
    for col in df_monthly.columns:
        if col != "date":
            df_cum[col] = (np.cumprod(1.0 + df_monthly[col]) - 1.0) * 100.0

    # Summary table
    perf_rows: list[dict] = []
    univ_metrics = compute_performance_metrics(df_monthly["Universe"])
    perf_rows.append(
        {
            "Estrategia": "Buy & Hold (Universo 1/N)",
            "Modo": "Benchmark Passivo",
            **univ_metrics,
            "Alpha_vs_Universo_pct": 0.0,
            "Ganho_Retreino_pct": 0.0,
        }
    )

    for m_type in models:
        col_wf = f"{m_type.capitalize()}_WF"
        col_frz = f"{m_type.capitalize()}_Frozen"

        m_wf = compute_performance_metrics(df_monthly[col_wf])
        m_frz = compute_performance_metrics(df_monthly[col_frz])

        alpha_wf = round(m_wf["CAGR_pct"] - univ_metrics["CAGR_pct"], 2)
        alpha_frz = round(m_frz["CAGR_pct"] - univ_metrics["CAGR_pct"], 2)
        gain_retrain = round(m_wf["CAGR_pct"] - m_frz["CAGR_pct"], 2)

        perf_rows.append(
            {
                "Estrategia": f"ML: {m_type.upper()} (Walk-Forward)",
                "Modo": f"Retreino a cada {retrain_every_months}m",
                **m_wf,
                "Alpha_vs_Universo_pct": alpha_wf,
                "Ganho_Retreino_pct": gain_retrain,
            }
        )
        perf_rows.append(
            {
                "Estrategia": f"ML: {m_type.upper()} (Congelado)",
                "Modo": "Estatico (Sem Retreino)",
                **m_frz,
                "Alpha_vs_Universo_pct": alpha_frz,
                "Ganho_Retreino_pct": 0.0,
            }
        )

    df_summary = pd.DataFrame(perf_rows)

    return WalkForwardResult(
        monthly_returns=df_monthly,
        cumulative_returns=df_cum,
        summary_metrics=df_summary,
        retrain_log=retrain_log,
    )


def run_weighting_schemes_comparison(
    market: MarketData,
    start_date: str | pd.Timestamp = "2015-01-02",
    end_date: str | pd.Timestamp = "2025-12-31",
    models: tuple[str, ...] = ("ridge", "xgboost", "lightgbm"),
    retrain_every_months: int = 12,
    train_lookback_days: int = 756,
    top_k_hard: int = 10,
    hysteresis_buffer: int = 20,
    top_k_soft: int = 15,
    seed: int = 42,
    verbose: bool = False,
) -> WeightingComparisonResult:
    """
    Compare portfolio weighting schemes under Walk-Forward annual retraining:
      1. Hard Cut 1/N: Top-K equal-weighted (1/N) with hysteresis maintenance buffer.
      2. Soft Weighting: Top-M with linear rank-decay weighting (w_i proportional to M - i + 1).

    Computes monthly returns, cumulative wealth, Sharpe, Drawdown, and monthly portfolio turnover.
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    sim_mkt = market.slice_dates(start_ts, end_ts)

    months = sim_mkt.dates.to_period("M")
    rebal_idx = np.where(months != np.roll(months, 1))[0]
    n_rebal = len(rebal_idx) - 1
    if n_rebal < 2:
        raise ValueError("Insufficient monthly rebalance dates in specified period.")

    # 1. Fit initial models on pre-start historical window
    idx_start_full = int(np.where(market.dates == sim_mkt.dates[rebal_idx[0]])[0][0])
    hist_win = market.slice_iloc(np.arange(max(0, idx_start_full - train_lookback_days), idx_start_full))
    panel_init = build_ml_panel(hist_win, horizon=21, step=5)
    splits_init = split_temporal_70_20_10(panel_init, train_ratio=0.70, val_ratio=0.20, purge_days=21)

    active_models: dict[str, MLModelArtifact] = {
        m_type: fit_ml_model(m_type, splits_init, FEATURE_COLS, seed=seed) for m_type in models
    }

    # Tracking holdings and weights for turnover
    held_hard: dict[str, list[str]] = {m: [] for m in models}
    prev_w_hard: dict[str, dict[str, float]] = {m: {} for m in models}
    prev_w_soft: dict[str, dict[str, float]] = {m: {} for m in models}

    # Rank-linear weighting vector for top_k_soft
    rank_weights = np.arange(top_k_soft, 0, -1, dtype=float)
    rank_weights /= rank_weights.sum()

    monthly_records: list[dict] = []
    turnover_records: list[dict] = []
    months_since_train = 0
    close_mat = sim_mkt.close

    for step in range(n_rebal):
        idx_t = int(rebal_idx[step])
        idx_next = int(rebal_idx[step + 1])
        dt = sim_mkt.dates[idx_t]
        full_idx_t = int(np.where(market.dates == dt)[0][0])

        months_since_train += 1
        if step > 0 and retrain_every_months > 0 and months_since_train >= retrain_every_months:
            if verbose:
                print(f"[Weighting Comparison] Retraining models at step {step} ({dt.strftime('%Y-%m-%d')})...")
            win_retrain = market.slice_iloc(np.arange(max(0, full_idx_t - train_lookback_days), full_idx_t + 1))
            p_retrain = build_ml_panel(win_retrain, horizon=21, step=5)
            if len(p_retrain) >= 200:
                s_retrain = split_temporal_70_20_10(p_retrain, train_ratio=0.70, val_ratio=0.20, purge_days=21)
                for m_type in models:
                    active_models[m_type] = fit_ml_model(m_type, s_retrain, FEATURE_COLS, seed=seed)
                months_since_train = 0

        win_today = market.slice_iloc(np.arange(max(0, full_idx_t - 260), full_idx_t + 1))
        df_today = extract_current_features(win_today)
        if df_today.empty:
            continue

        p_start = close_mat.iloc[idx_t]
        p_end = close_mat.iloc[idx_next]

        # Universe (Equal-weighted Buy & Hold baseline)
        r_univ = float((p_end / p_start - 1.0).dropna().mean())

        step_record = {
            "date": sim_mkt.dates[idx_next],
            "Universe": r_univ,
        }
        to_record = {"date": sim_mkt.dates[idx_next]}

        for m_type in models:
            preds = predict_ml_model(active_models[m_type], df_today)
            ranked = list(preds.sort_values(ascending=False).index)

            # Scheme 1: Hard Cut 1/N (Top K with hysteresis buffer)
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

            # Scheme 2: Soft Weighting (Top M linear rank decay)
            chosen_soft = ranked[: min(top_k_soft, len(ranked))]
            m_len = len(chosen_soft)
            if m_len == top_k_soft:
                w_vec = rank_weights
            else:
                w_vec = np.arange(m_len, 0, -1, dtype=float)
                w_vec /= w_vec.sum()

            w_soft = {tk: float(w_vec[i]) for i, tk in enumerate(chosen_soft)}
            r_soft = float(
                sum(
                    w_soft[tk] * (p_end[tk] / p_start[tk] - 1.0)
                    for tk in chosen_soft
                    if not np.isnan(p_start[tk]) and not np.isnan(p_end[tk])
                )
            )
            step_record[f"{m_type.capitalize()}_SoftWeight"] = r_soft

            if prev_w_soft[m_type]:
                all_tk_sf = set(w_soft) | set(prev_w_soft[m_type])
                to_val_sf = 0.5 * sum(abs(w_soft.get(t, 0.0) - prev_w_soft[m_type].get(t, 0.0)) for t in all_tk_sf)
                to_record[f"{m_type.capitalize()}_SoftWeight_TO"] = to_val_sf
            else:
                to_record[f"{m_type.capitalize()}_SoftWeight_TO"] = 1.0
            prev_w_soft[m_type] = w_soft

        monthly_records.append(step_record)
        turnover_records.append(to_record)

    df_monthly = pd.DataFrame(monthly_records).sort_values("date").reset_index(drop=True)
    df_to = pd.DataFrame(turnover_records).sort_values("date").reset_index(drop=True)

    # Cumulative wealth
    df_cum = pd.DataFrame({"date": df_monthly["date"]})
    for col in df_monthly.columns:
        if col != "date":
            df_cum[col] = (np.cumprod(1.0 + df_monthly[col]) - 1.0) * 100.0

    # Summary performance metrics
    univ_m = compute_performance_metrics(df_monthly["Universe"])
    summary_rows = [
        {
            "Modelo": "Buy & Hold (Universo)",
            "Esquema": "Equiponderado (1/N)",
            **univ_m,
            "Turnover_Medio_Mensal_pct": 0.0,
            "Alpha_vs_Universo_pct": 0.0,
        }
    ]

    for m_type in models:
        c_hard = f"{m_type.capitalize()}_HardCut"
        c_soft = f"{m_type.capitalize()}_SoftWeight"
        to_hard_col = f"{m_type.capitalize()}_HardCut_TO"
        to_soft_col = f"{m_type.capitalize()}_SoftWeight_TO"

        m_hd = compute_performance_metrics(df_monthly[c_hard])
        m_sf = compute_performance_metrics(df_monthly[c_soft])

        avg_to_hd = float(df_to[to_hard_col].iloc[1:].mean() * 100.0) if len(df_to) > 1 else 0.0
        avg_to_sf = float(df_to[to_soft_col].iloc[1:].mean() * 100.0) if len(df_to) > 1 else 0.0

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

    return WeightingComparisonResult(
        monthly_returns=df_monthly,
        cumulative_returns=df_cum,
        summary_metrics=df_summary,
        turnovers=df_to,
    )
