"""
Quantitative Machine Learning and Portfolio Selection Evaluation Metrics:
  - Regression: RMSE, MAE, Out-of-Sample R^2 (%)
  - Cross-Sectional Ranking: Pearson IC, Spearman Rank IC, Top-K Spread (bps), Top-K Win Periods (%)
  - Directional Classification (Excess Return > 0): Directional Accuracy (%), Exact ROC-AUC, F1-Score
  - Factor Portfolio Performance: Realistic monthly non-overlapping selection of Top-K stocks with hysteresis buffer vs. Buy & Hold baseline
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

from .data_loader import MarketData
from .features import extract_current_features


def compute_ml_metrics(df_part: pd.DataFrame, pred_vec: np.ndarray, top_k: int = 10) -> dict[str, float | int]:
    """
    Compute the 3 families of Quantitative ML evaluation metrics on a partition:
      1. Regression: RMSE, MAE, Out-of-Sample R^2 (%)
      2. Cross-Sectional Ranking: Pearson IC, Spearman Rank IC, Top-K Spread (bps), Top-K Win Periods (%)
      3. Directional Classification (Excess Return > 0): Directional Accuracy (%), Exact ROC-AUC, F1-Score
    """
    y_true = df_part["target_excess"].to_numpy(dtype=float)
    y_raw = df_part["target_raw"].to_numpy(dtype=float)
    preds = np.asarray(pred_vec, dtype=float)
    n = len(y_true)
    if n == 0 or len(preds) != n:
        raise ValueError("Mismatched or empty predictions in compute_ml_metrics.")

    # 1. Regression metrics
    residuals = y_true - preds
    rmse = float(np.sqrt(np.nanmean(residuals**2)))
    mae = float(np.nanmean(np.abs(residuals)))
    ss_res = float(np.nansum(residuals**2))
    ss_tot = float(np.nansum((y_true - np.nanmean(y_true)) ** 2))
    r2_oos = (1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    # 2. Cross-Sectional Ranking (IC & Rank IC)
    if np.nanstd(preds) < 1e-12 or np.nanstd(y_true) < 1e-12:
        ic_pearson = 0.0
        ic_spearman = 0.0
    else:
        ic_pearson = float(pearsonr(preds, y_true).statistic)
        ic_spearman = float(spearmanr(preds, y_true).statistic)

    # Top-K spread per rebalance date: Top-K average return vs. Universe average return
    df_eval = pd.DataFrame({"date": df_part["date"].to_numpy(), "pred": preds, "raw": y_raw})
    spreads: list[float] = []
    for _, grp in df_eval.groupby("date", sort=False):
        p_d = grp["pred"].to_numpy()
        r_d = grp["raw"].to_numpy()
        k = min(top_k, len(p_d))
        top_idx = np.argsort(-p_d, kind="mergesort")[:k]
        top_ret = float(np.nanmean(r_d[top_idx]))
        spreads.append(top_ret - float(np.nanmean(r_d)))

    spreads_arr = np.asarray(spreads, dtype=float)
    mean_spread_bps = float(np.nanmean(spreads_arr) * 10000.0)
    hit_top_k_pct = float(np.nanmean(spreads_arr > 0.0) * 100.0)

    # 3. Directional Classification (above market average: y_true > 0)
    actual_pos = (y_true > 0.0).astype(int)
    pred_pos = (preds > 0.0).astype(int)

    acc_dir = float(np.mean(actual_pos == pred_pos) * 100.0)
    tp = int(np.sum((pred_pos == 1) & (actual_pos == 1)))
    fp = int(np.sum((pred_pos == 1) & (actual_pos == 0)))
    fn = int(np.sum((pred_pos == 0) & (actual_pos == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Exact AUC-ROC via Mann-Whitney-Wilcoxon rank statistic
    pos_scores = preds[actual_pos == 1]
    neg_scores = preds[actual_pos == 0]
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        ranks = rankdata(np.concatenate([pos_scores, neg_scores]))
        n_pos = len(pos_scores)
        n_neg = len(neg_scores)
        auc_roc = float((np.sum(ranks[:n_pos]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    else:
        auc_roc = np.nan

    return {
        "n_obs": int(n),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2_pct": round(r2_oos * 100.0, 2),
        "IC_Pearson": round(ic_pearson, 4),
        "Rank_IC_Spearman": round(ic_spearman, 4),
        "Acuracia_Direcional_pct": round(acc_dir, 2),
        "AUC_ROC": round(auc_roc, 4),
        "F1_Score": round(f1_score, 4),
        "Spread_Top10_bps": round(mean_spread_bps, 1),
        "Win_Periodos_Top10_pct": round(hit_top_k_pct, 1),
    }


def evaluate_monthly_top_k_selection(
    model: "MLModelArtifact",
    market_full: MarketData,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    top_k: int = 10,
    hysteresis_buffer: int = 20,
) -> pd.DataFrame:
    """
    Evaluate realistic monthly non-overlapping selection of Top-K stocks:
      - Rebalance on the 1st trading day of each calendar month
      - Use hysteresis buffer: stocks already held that remain in the Top 20 are kept
      - Calculate exact monthly price returns without overlapping compounding
    """
    from .models import predict_ml_model
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    test_mkt = market_full.slice_dates(start_ts, end_ts)

    months = test_mkt.dates.to_period("M")
    rebal_idx = np.where(months != np.roll(months, 1))[0]
    close_mat = test_mkt.close

    held_tickers: list[str] = []
    records: list[dict] = []

    for i in range(len(rebal_idx) - 1):
        idx_t = int(rebal_idx[i])
        idx_next = int(rebal_idx[i + 1])
        dt = test_mkt.dates[idx_t]

        full_idx_t = int(np.where(market_full.dates == dt)[0][0])
        win = market_full.slice_iloc(np.arange(max(0, full_idx_t - 260), full_idx_t + 1))
        df_today = extract_current_features(win)
        if df_today.empty:
            continue

        preds = predict_ml_model(model, df_today)
        ranked = list(preds.sort_values(ascending=False).index)

        # Hysteresis buffer
        buffer_set = set(ranked[: min(hysteresis_buffer, len(ranked))])
        kept = [tk for tk in held_tickers if tk in buffer_set]
        new_picks = [tk for tk in ranked if tk not in kept]
        chosen = kept + new_picks[: top_k - len(kept)]
        held_tickers = chosen

        p_start = close_mat.iloc[idx_t]
        p_end = close_mat.iloc[idx_next]

        r_all = (p_end / p_start - 1.0).dropna()
        r_top = (p_end[chosen] / p_start[chosen] - 1.0).dropna()

        records.append(
            {
                "date": test_mkt.dates[idx_next],
                "top_k_return": float(r_top.mean()),
                "universe_return": float(r_all.mean()),
            }
        )

    res = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    res["top_k_cumulative"] = np.cumprod(1.0 + res["top_k_return"]) - 1.0
    res["universe_cumulative"] = np.cumprod(1.0 + res["universe_return"]) - 1.0
    res["alpha_cumulative"] = (1.0 + res["top_k_cumulative"]) / (1.0 + res["universe_cumulative"]) - 1.0
    return res
