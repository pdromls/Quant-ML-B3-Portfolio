"""
Main CLI entrypoint for the Quantitative Machine Learning Experiment on the B3 IBrX 50 Universe:
  1. Data Loading & Universe Inspection
  2. Causal Feature Engineering & Cross-Sectional Standardization
  3. Chronological 70% Train / 20% Val / 10% Test Split with 21-day Purge (Embargo)
  4. Model Training & Hyperparameter Selection (Ridge L2 Baseline, XGBoost, LightGBM)
  5. Comprehensive ML Evaluation (Regression, Ranking/IC, Directional Classification)
  6. Top-10 Stock Selection Factor Performance vs. Equal-Weighted Universe (Buy & Hold baseline)
  7. Feature Importance Analysis (Gain % and Standardized Coefficients)
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from quant_ml_b3 import (
    FEATURE_COLS,
    build_ml_panel,
    evaluate_top_k_cumulative_performance,
    extract_feature_importance,
    fit_ml_model,
    load_local_market_data,
    predict_ml_model,
    split_temporal_70_20_10,
)
from quant_ml_b3.plotting import (
    plot_feature_importance,
    plot_test_metrics_summary,
    plot_top_k_cumulative_alpha,
)


def main() -> None:
    data_dir = ROOT_DIR / "data" / "processed"
    tables_dir = ROOT_DIR / "outputs" / "tables"
    figures_dir = ROOT_DIR / "outputs" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("1. CARREGAMENTO DO UNIVERSO B3 (IBrX 50 - 41 ACOES DE ALTA LIQUIDEZ)")
    print("=" * 78)
    market_full = load_local_market_data(data_dir)
    print(
        f"Universo carregado: {len(market_full.tickers)} acoes x {len(market_full.dates)} pregoes "
        f"({market_full.dates[0].date()} a {market_full.dates[-1].date()})\n"
    )

    print("=" * 78)
    print("2. ENGENHARIA DE ATRIBUTOS CAUSAIS E DIVISAO CRONOLOGICA COM PURGA (21 DIAS)")
    print("=" * 78)
    market_15_25 = market_full.slice_dates("2015-01-02", "2025-12-31")
    panel_ml = build_ml_panel(market_15_25, horizon=21, step=5)
    splits = split_temporal_70_20_10(panel_ml, train_ratio=0.70, val_ratio=0.20, purge_days=21)

    tr_p = splits.meta["train_period"]
    va_p = splits.meta["val_period"]
    te_p = splits.meta["test_period"]
    print(f"Total de observacoes no painel (data x acao): {len(panel_ml):,} linhas")
    print(
        f" - Treino (70%)     : {tr_p[0].date()} a {tr_p[1].date()} "
        f"({splits.meta['n_dates_train']} datas | {len(splits.train):,} amostras)"
    )
    print(
        f" - Validacao (20%): {va_p[0].date()} a {va_p[1].date()} "
        f"({splits.meta['n_dates_val']} datas | {len(splits.val):,} amostras) [Purga 21d]"
    )
    print(
        f" - Teste Puro (10%) : {te_p[0].date()} a {te_p[1].date()} "
        f"({splits.meta['n_dates_test']} datas | {len(splits.test):,} amostras) [Purga 21d]\n"
    )

    print("=" * 78)
    print("3. TREINAMENTO E SELECAO DE HIPERPARAMETROS NA VALIDACAO (20%)")
    print("=" * 78)
    fit_ridge = fit_ml_model("ridge", splits, FEATURE_COLS, seed=42)
    fit_xgb = fit_ml_model("xgboost", splits, FEATURE_COLS, seed=42)
    fit_lgb = fit_ml_model("lightgbm", splits, FEATURE_COLS, seed=42)

    print("Hiperparametros otimizados via Rank IC na Validacao:")
    print(f" - Ridge Baseline : lambda otimo = {fit_ridge.best_params['best_lambda']}")
    print(
        f" - XGBoost        : nrounds = {fit_xgb.best_params['best_iter']}, "
        f"max_depth = {fit_xgb.best_params['best_depth']}, eta = {fit_xgb.best_params['best_eta']}"
    )
    print(
        f" - LightGBM       : nrounds = {fit_lgb.best_params['best_iter']}, "
        f"num_leaves = {fit_lgb.best_params['best_leaves']}, lr = {fit_lgb.best_params['best_lr']}\n"
    )

    # 4. Tabela de metricas consolidada
    metrics_ml_df = pd.concat(
        [fit_ridge.metrics_table, fit_xgb.metrics_table, fit_lgb.metrics_table],
        ignore_index=True,
    )
    print("=" * 78)
    print("4. TABELA DE METRICAS DE MACHINE LEARNING (TREINO 70% / VAL 20% / TESTE 10%)")
    print("=" * 78)
    print(metrics_ml_df.to_string(index=False))
    metrics_ml_df.to_csv(tables_dir / "ml_metrics_70_20_10.csv", index=False)
    plot_test_metrics_summary(metrics_ml_df, figures_dir / "02_test_metrics_summary.png")

    # 5. Importancia das features
    imp_df = extract_feature_importance(fit_ridge, fit_xgb, fit_lgb)
    print("\n" + "=" * 78)
    print("5. IMPORTANCIA DAS FEATURES NOS MODELOS (GAIN % E BETA PADRONIZADO)")
    print("=" * 78)
    print(imp_df.to_string(index=False))
    imp_df.to_csv(tables_dir / "feature_importance.csv", index=False)
    plot_feature_importance(imp_df, figures_dir / "01_feature_importance.png")

    # 6. Avaliacao Quantitativa de Selecao Top 10 vs. Buy & Hold (Universo Igualitario)
    print("\n" + "=" * 78)
    print("6. AVALIACAO DE SELECAO TOP 10 vs. BUY & HOLD (PARTICAO DE TESTE PURO 10%)")
    print("=" * 78)

    # Predicoes de cada modelo no conjunto de teste puro
    preds_ridge_test = predict_ml_model(fit_ridge, splits.test).to_numpy()
    preds_xgb_test = predict_ml_model(fit_xgb, splits.test).to_numpy()
    preds_lgb_test = predict_ml_model(fit_lgb, splits.test).to_numpy()

    perf_ridge = evaluate_top_k_cumulative_performance(splits.test, preds_ridge_test, top_k=10)
    perf_xgb = evaluate_top_k_cumulative_performance(splits.test, preds_xgb_test, top_k=10)
    perf_lgb = evaluate_top_k_cumulative_performance(splits.test, preds_lgb_test, top_k=10)

    perf_dict = {
        "Ridge": perf_ridge,
        "XGBoost": perf_xgb,
        "LightGBM": perf_lgb,
    }

    summary_rows = [
        {
            "Estrategia / Modelo": "Buy & Hold (Universo 1/N)",
            "Retorno_Acumulado_pct": round(perf_ridge["universe_cumulative"].iloc[-1] * 100.0, 2),
            "Alpha_vs_Mercado_pct": 0.0,
        },
        {
            "Estrategia / Modelo": "ML Baseline: Regressao Ridge",
            "Retorno_Acumulado_pct": round(perf_ridge["top_k_cumulative"].iloc[-1] * 100.0, 2),
            "Alpha_vs_Mercado_pct": round(perf_ridge["alpha_cumulative"].iloc[-1] * 100.0, 2),
        },
        {
            "Estrategia / Modelo": "ML: XGBoost (Top 10)",
            "Retorno_Acumulado_pct": round(perf_xgb["top_k_cumulative"].iloc[-1] * 100.0, 2),
            "Alpha_vs_Mercado_pct": round(perf_xgb["alpha_cumulative"].iloc[-1] * 100.0, 2),
        },
        {
            "Estrategia / Modelo": "ML: LightGBM (Top 10)",
            "Retorno_Acumulado_pct": round(perf_lgb["top_k_cumulative"].iloc[-1] * 100.0, 2),
            "Alpha_vs_Mercado_pct": round(perf_lgb["alpha_cumulative"].iloc[-1] * 100.0, 2),
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(tables_dir / "top10_vs_universe_performance.csv", index=False)

    plot_top_k_cumulative_alpha(perf_dict, figures_dir / "03_top10_cumulative_alpha.png")
    print(f"\nExperimento de ML concluido com sucesso! Figuras em {figures_dir} e tabelas em {tables_dir}.")


if __name__ == "__main__":
    main()
