"""
Unit tests for Machine Learning models (Ridge, XGBoost, LightGBM), feature importance extraction,
and portfolio ranking evaluation metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_ml_b3.features import FEATURE_COLS, build_ml_panel, extract_current_features
from quant_ml_b3.metrics import compute_ml_metrics, evaluate_top_k_cumulative_performance
from quant_ml_b3.models import extract_feature_importance, fit_ml_model, predict_ml_model
from quant_ml_b3.splits import split_temporal_70_20_10
from test_features_and_splits import _make_synthetic_market


def test_all_three_models_train_predict_and_report_importance() -> None:
    mkt = _make_synthetic_market(n_days=340, n_tickers=12, seed=123)
    panel = build_ml_panel(mkt, horizon=21, step=5)
    splits = split_temporal_70_20_10(panel, train_ratio=0.70, val_ratio=0.20, purge_days=21)

    fit_ridge = fit_ml_model("ridge", splits, FEATURE_COLS, seed=42)
    fit_xgb = fit_ml_model("xgboost", splits, FEATURE_COLS, seed=42)
    fit_lgb = fit_ml_model("lightgbm", splits, FEATURE_COLS, seed=42)

    for art in (fit_ridge, fit_xgb, fit_lgb):
        assert len(art.metrics_table) == 3
        assert set(art.metrics_table["Particao"]) == {
            "1. Treino (70%)",
            "2. Validacao (20%)",
            "3. Teste (10%)",
        }
        assert (art.metrics_table["AUC_ROC"] >= 0.0).all()
        assert (art.metrics_table["AUC_ROC"] <= 1.0).all()

    df_today = extract_current_features(mkt)
    for art in (fit_ridge, fit_xgb, fit_lgb):
        preds = predict_ml_model(art, df_today)
        assert len(preds) == len(df_today)
        assert not preds.isna().any()

    imp = extract_feature_importance(fit_ridge, fit_xgb, fit_lgb)
    assert len(imp) == len(FEATURE_COLS)
    assert abs(imp["XGBoost_Gain"].sum() - 1.0) < 1e-3
    assert abs(imp["LightGBM_Gain"].sum() - 1.0) < 1e-3


def test_evaluate_top_k_cumulative_performance() -> None:
    mkt = _make_synthetic_market(n_days=320, n_tickers=10, seed=456)
    panel = build_ml_panel(mkt, horizon=21, step=5)
    splits = split_temporal_70_20_10(panel, train_ratio=0.70, val_ratio=0.20, purge_days=21)

    fit_ridge = fit_ml_model("ridge", splits, FEATURE_COLS, seed=42)
    preds = predict_ml_model(fit_ridge, splits.test).to_numpy()

    perf = evaluate_top_k_cumulative_performance(splits.test, preds, top_k=3)
    assert len(perf) == splits.meta["n_dates_test"]
    assert "top_k_cumulative" in perf.columns
    assert "universe_cumulative" in perf.columns
    assert "alpha_cumulative" in perf.columns
    assert np.isfinite(perf["alpha_cumulative"].iloc[-1])
