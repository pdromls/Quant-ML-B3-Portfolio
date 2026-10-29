"""
Quantitative Machine Learning and Portfolio Selection Evaluation Metrics:
  - Regression: RMSE, MAE, Out-of-Sample R^2 (%)
  - Cross-Sectional Ranking: Pearson IC, Spearman Rank IC, Top-K Spread (bps), Top-K Win Periods (%)
  - Directional Classification (Excess Return > 0): Directional Accuracy (%), Exact ROC-AUC, F1-Score
  - Factor Portfolio Performance: Cumulative return of model Top-K selection vs. Equal-Weighted Universe (Buy & Hold baseline)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr


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


def evaluate_top_k_cumulative_performance(
    df_part: pd.DataFrame,
    pred_vec: np.ndarray,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Evaluate the cumulative compound return of selecting the model's Top-K stocks
    at each rebalancing date versus the Equal-Weighted Universe (Buy & Hold baseline).

    Returns a DataFrame with columns:
      ['date', 'top_k_return', 'universe_return', 'top_k_cumulative', 'universe_cumulative', 'alpha_cumulative']
    """
    df_eval = pd.DataFrame(
        {
            "date": df_part["date"].to_numpy(),
            "pred": np.asarray(pred_vec, dtype=float),
            "raw": df_part["target_raw"].to_numpy(dtype=float),
        }
    )

    records: list[dict] = []
    for dt, grp in df_eval.groupby("date", sort=True):
        p_d = grp["pred"].to_numpy()
        r_d = grp["raw"].to_numpy()
        k = min(top_k, len(p_d))
        top_idx = np.argsort(-p_d, kind="mergesort")[:k]
        top_ret = float(np.nanmean(r_d[top_idx]))
        uni_ret = float(np.nanmean(r_d))
        records.append({"date": dt, "top_k_return": top_ret, "universe_return": uni_ret})

    res = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    res["top_k_cumulative"] = np.cumprod(1.0 + res["top_k_return"]) - 1.0
    res["universe_cumulative"] = np.cumprod(1.0 + res["universe_return"]) - 1.0
    res["alpha_cumulative"] = (1.0 + res["top_k_cumulative"]) / (1.0 + res["universe_cumulative"]) - 1.0
    return res
