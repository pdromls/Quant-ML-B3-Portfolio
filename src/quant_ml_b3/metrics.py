"""
Quantitative Machine Learning and Financial Portfolio Performance Metrics.
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

    # 2. Cross-Sectional Ranking / Portfolio Selection (IC & Rank IC)
    if np.nanstd(preds) < 1e-12 or np.nanstd(y_true) < 1e-12:
        ic_pearson = 0.0
        ic_spearman = 0.0
    else:
        ic_pearson = float(pearsonr(preds, y_true).statistic)
        ic_spearman = float(spearmanr(preds, y_true).statistic)

    # Mean Top-K excess spread per rebalance date
    df_eval = pd.DataFrame({"date": df_part["date"].to_numpy(), "pred": preds, "raw": y_raw})
    spreads: list[float] = []
    for _, grp in df_eval.groupby("date", sort=False):
        p_d = grp["pred"].to_numpy()
        r_d = grp["raw"].to_numpy()
        k = min(top_k, len(p_d))
        # Sort descending (stable tie-breaking)
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


def compute_trade_win_rate(
    trades: pd.DataFrame | None,
    from_date: str | pd.Timestamp | None = None,
    to_date: str | pd.Timestamp | None = None,
) -> float:
    """
    Compute realized trade Win Rate: proportion of sell executions with positive net profit
    after accounting for average cost basis (including purchase transaction costs) and sell transaction costs.
    """
    if trades is None or trades.empty:
        return np.nan

    df = trades.copy()
    df["date"] = pd.to_datetime(df["date"])
    if to_date is not None:
        df = df[df["date"] <= pd.Timestamp(to_date)]
    if df.empty:
        return np.nan

    from_ts = pd.Timestamp(from_date) if from_date is not None else None
    pos: dict[str, float] = {}
    avg_cost: dict[str, float] = {}
    wins = 0
    total = 0

    for row in df.itertuples(index=False):
        tk = str(row.ticker)
        side = str(row.side)
        qty = float(row.qty)
        val = float(row.value)
        cost = float(row.cost)
        dt = pd.Timestamp(row.date)

        if not tk or qty <= 0 or not np.isfinite(val) or not np.isfinite(cost):
            continue

        if side == "buy":
            held = pos.get(tk, 0.0)
            base = avg_cost.get(tk, 0.0) * held if held > 0 else 0.0
            new_pos = held + qty
            pos[tk] = new_pos
            avg_cost[tk] = (base + val + cost) / new_pos
        elif side == "sell":
            held = pos.get(tk, 0.0)
            if held <= 0:
                continue
            q = min(qty, held)
            net_proceeds = (val - cost) * (q / qty)
            realized_pnl = net_proceeds - avg_cost.get(tk, 0.0) * q
            if from_ts is None or dt >= from_ts:
                total += 1
                if realized_pnl > 0:
                    wins += 1
            rem = held - q
            if rem <= 1e-12:
                pos[tk] = 0.0
                avg_cost[tk] = 0.0
            else:
                pos[tk] = rem

    return (wins / total) if total > 0 else np.nan


def compute_financial_metrics(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame | None = None,
    initial_capital: float = 100_000.0,
    rf: float = 0.0,
) -> dict[str, float]:
    """
    Compute portfolio performance metrics from a daily equity curve (`['date', 'equity']`) and trade ledger.
    """
    curve = equity_df.dropna(subset=["equity"]).sort_values("date").reset_index(drop=True)
    if len(curve) < 2:
        raise ValueError("At least 2 daily equity observations are required to compute financial metrics.")

    values = curve["equity"].to_numpy(dtype=float)
    returns = values[1:] / values[:-1] - 1.0
    n = len(returns)

    total_return = float(values[-1] / values[0] - 1.0)
    annual_return = float((1.0 + total_return) ** (252.0 / n) - 1.0)
    annual_vol = float(np.std(returns, ddof=1) * np.sqrt(252.0))
    sharpe = float((annual_return - rf) / annual_vol) if annual_vol > 0 else np.nan

    running_max = np.maximum.accumulate(values)
    drawdowns = values / running_max - 1.0
    max_drawdown = float(np.min(drawdowns))
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else np.nan

    start_dt = curve["date"].iloc[0]
    end_dt = curve["date"].iloc[-1]
    win_rate = compute_trade_win_rate(trades_df, from_date=start_dt, to_date=end_dt)

    turnover = np.nan
    total_cost = np.nan
    cost_pct = np.nan
    if trades_df is not None and not trades_df.empty:
        t_sub = trades_df[(trades_df["date"] >= start_dt) & (trades_df["date"] <= end_dt)]
        turnover = float(t_sub["value"].abs().sum() / np.mean(values) * (252.0 / n))
        total_cost = float(t_sub["cost"].sum())
        cost_pct = float(total_cost / initial_capital)

    return {
        "final_equity": round(float(values[-1]), 2),
        "net_profit_brl": round(float(values[-1] - initial_capital), 2),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": win_rate,
        "turnover": turnover,
        "total_cost": total_cost,
        "cost_pct": cost_pct,
    }
