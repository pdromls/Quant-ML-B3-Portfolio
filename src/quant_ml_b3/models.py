"""
Cross-Sectional Machine Learning Models for Return Ranking:
  1. Ridge Regression (L2 Baseline)
  2. XGBoost (Depth-wise Gradient Boosted Trees)
  3. LightGBM (Leaf-wise Gradient Boosted Trees)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import xgboost as xgb

from .features import FEATURE_COLS
from .metrics import compute_ml_metrics
from .splits import TemporalSplit


@dataclass
class MLModelArtifact:
    """Trained ML model artifact with validation hyperparameter metadata and evaluation table."""

    model_type: str
    model: Any
    feature_cols: list[str]
    best_params: dict[str, Any]
    metrics_table: pd.DataFrame
    split_meta: dict[str, Any]


def _safe_spearman(pred: np.ndarray, target: np.ndarray) -> float:
    if np.nanstd(pred) < 1e-12 or np.nanstd(target) < 1e-12:
        return -np.inf
    val = float(spearmanr(pred, target).statistic)
    return val if np.isfinite(val) else -np.inf


def fit_ml_model(
    model_type: str,
    splits: TemporalSplit,
    feature_cols: list[str] = FEATURE_COLS,
    seed: int = 42,
) -> MLModelArtifact:
    """
    Train and select hyperparameters on the chronological Validation partition (20%)
    and evaluate across Train (70%), Validation (20%), and Pure Out-of-Sample Test (10%).
    """
    mtype = model_type.strip().lower()
    X_tr = splits.train[feature_cols].to_numpy(dtype=float)
    y_tr = splits.train["target_excess"].to_numpy(dtype=float)
    X_va = splits.val[feature_cols].to_numpy(dtype=float)
    y_va = splits.val["target_excess"].to_numpy(dtype=float)
    X_te = splits.test[feature_cols].to_numpy(dtype=float)

    if mtype == "ridge":
        lambdas = [0.1, 1.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
        p = X_tr.shape[1]
        X_tr_int = np.column_stack([np.ones(len(X_tr)), X_tr])
        X_va_int = np.column_stack([np.ones(len(X_va)), X_va])
        X_te_int = np.column_stack([np.ones(len(X_te)), X_te])

        XtX = X_tr_int.T @ X_tr_int
        Xty = X_tr_int.T @ y_tr

        best_ic = -np.inf
        best_beta = np.zeros(p + 1, dtype=float)
        best_lambda = lambdas[0]

        for lam in lambdas:
            pen = np.diag([0.0] + [lam] * p)  # Do not penalize intercept
            try:
                beta = np.linalg.solve(XtX + pen, Xty)
            except np.linalg.LinAlgError:
                beta = np.zeros(p + 1, dtype=float)

            pred_va_cand = X_va_int @ beta
            ic_va = _safe_spearman(pred_va_cand, y_va)
            if ic_va > best_ic:
                best_ic = ic_va
                best_beta = beta
                best_lambda = lam

        pred_tr = X_tr_int @ best_beta
        pred_va = X_va_int @ best_beta
        pred_te = X_te_int @ best_beta
        model_obj = best_beta
        best_params = {"best_lambda": best_lambda}

    elif mtype == "xgboost":
        dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
        dva = xgb.DMatrix(X_va, label=y_va, feature_names=feature_cols)
        dte = xgb.DMatrix(X_te, feature_names=feature_cols)

        grid_xgb = [
            {"nrounds": nr, "max_depth": md, "eta": eta}
            for eta in (0.02, 0.05)
            for md in (2, 3, 4)
            for nr in (30, 60, 100)
        ]

        best_score = -np.inf
        best_bst = None
        best_cfg = grid_xgb[0]

        for cfg in grid_xgb:
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "max_depth": cfg["max_depth"],
                "learning_rate": cfg["eta"],
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 25,
                "reg_alpha": 0.5,
                "reg_lambda": 2.0,
                "nthread": 2,
                "seed": seed,
            }
            bst_i = xgb.train(params=params, dtrain=dtr, num_boost_round=cfg["nrounds"], verbose_eval=False)
            p_va = np.asarray(bst_i.predict(dva), dtype=float)
            ic_va = _safe_spearman(p_va, y_va)
            rmse_va = float(np.sqrt(np.nanmean((y_va - p_va) ** 2)))
            score_i = (ic_va if np.isfinite(ic_va) else -1.0) - 0.1 * rmse_va
            if score_i > best_score:
                best_score = score_i
                best_bst = bst_i
                best_cfg = cfg

        pred_tr = np.asarray(best_bst.predict(dtr), dtype=float)
        pred_va = np.asarray(best_bst.predict(dva), dtype=float)
        pred_te = np.asarray(best_bst.predict(dte), dtype=float)
        model_obj = best_bst
        best_params = {
            "best_iter": best_cfg["nrounds"],
            "best_depth": best_cfg["max_depth"],
            "best_eta": best_cfg["eta"],
        }

    elif mtype == "lightgbm":
        grid_lgb = [
            {"nrounds": nr, "num_leaves": nl, "lr": lr}
            for lr in (0.02, 0.05)
            for nl in (7, 15)
            for nr in (30, 60, 100)
        ]

        best_score = -np.inf
        best_bst = None
        best_cfg = grid_lgb[0]

        for cfg in grid_lgb:
            ltr = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols, free_raw_data=False)
            params = {
                "objective": "regression",
                "metric": "rmse",
                "max_depth": 4,
                "num_leaves": cfg["num_leaves"],
                "learning_rate": cfg["lr"],
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "min_data_in_leaf": 30,
                "lambda_l1": 0.5,
                "lambda_l2": 2.0,
                "num_threads": 2,
                "verbose": -1,
                "deterministic": True,
                "seed": seed,
                "feature_fraction_seed": seed,
                "bagging_seed": seed,
            }
            bst_i = lgb.train(params=params, train_set=ltr, num_boost_round=cfg["nrounds"])
            p_va = np.asarray(bst_i.predict(X_va), dtype=float)
            ic_va = _safe_spearman(p_va, y_va)
            rmse_va = float(np.sqrt(np.nanmean((y_va - p_va) ** 2)))
            score_i = (ic_va if np.isfinite(ic_va) else -1.0) - 0.1 * rmse_va
            if score_i > best_score:
                best_score = score_i
                best_bst = bst_i
                best_cfg = cfg

        pred_tr = np.asarray(best_bst.predict(X_tr), dtype=float)
        pred_va = np.asarray(best_bst.predict(X_va), dtype=float)
        pred_te = np.asarray(best_bst.predict(X_te), dtype=float)
        model_obj = best_bst
        best_params = {
            "best_iter": best_cfg["nrounds"],
            "best_leaves": best_cfg["num_leaves"],
            "best_lr": best_cfg["lr"],
        }

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    m_tr = compute_ml_metrics(splits.train, pred_tr)
    m_va = compute_ml_metrics(splits.val, pred_va)
    m_te = compute_ml_metrics(splits.test, pred_te)

    rows = [
        {"Modelo": mtype.upper(), "Particao": "1. Treino (70%)", **m_tr},
        {"Modelo": mtype.upper(), "Particao": "2. Validacao (20%)", **m_va},
        {"Modelo": mtype.upper(), "Particao": "3. Teste (10%)", **m_te},
    ]
    metrics_table = pd.DataFrame(rows)

    return MLModelArtifact(
        model_type=mtype,
        model=model_obj,
        feature_cols=list(feature_cols),
        best_params=best_params,
        metrics_table=metrics_table,
        split_meta=splits.meta,
    )


def predict_ml_model(artifact: MLModelArtifact, df_features: pd.DataFrame) -> pd.Series:
    """Predict expected 21-day cross-sectional excess return for each stock in `df_features`."""
    if df_features.empty:
        return pd.Series(dtype=float)

    X = df_features[artifact.feature_cols].to_numpy(dtype=float)
    if artifact.model_type == "ridge":
        X_int = np.column_stack([np.ones(len(X)), X])
        preds = X_int @ artifact.model
    elif artifact.model_type == "xgboost":
        dmat = xgb.DMatrix(X, feature_names=artifact.feature_cols)
        preds = np.asarray(artifact.model.predict(dmat), dtype=float)
    elif artifact.model_type == "lightgbm":
        preds = np.asarray(artifact.model.predict(X), dtype=float)
    else:
        raise ValueError(f"Unsupported model_type: {artifact.model_type}")

    return pd.Series(preds, index=df_features["ticker"].to_numpy(), dtype=float)


def extract_feature_importance(
    fit_ridge: MLModelArtifact,
    fit_xgb: MLModelArtifact,
    fit_lgb: MLModelArtifact,
) -> pd.DataFrame:
    """
    Consolidate feature importances across Ridge (standardized coefficients),
    XGBoost (relative Gain share), and LightGBM (relative Gain share).
    """
    feature_cols = fit_ridge.feature_cols
    beta_no_intercept = fit_ridge.model[1:]

    # XGBoost total gain normalized to 1.0
    xgb_scores = fit_xgb.model.get_score(importance_type="total_gain")
    xgb_vals = np.array([float(xgb_scores.get(f, 0.0)) for f in feature_cols], dtype=float)
    if xgb_vals.sum() > 0:
        xgb_vals = xgb_vals / xgb_vals.sum()

    # LightGBM gain normalized to 1.0
    lgb_vals = np.asarray(fit_lgb.model.feature_importance(importance_type="gain"), dtype=float)
    if lgb_vals.sum() > 0:
        lgb_vals = lgb_vals / lgb_vals.sum()

    df_imp = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Ridge_Abs_Beta": np.round(np.abs(beta_no_intercept), 5),
            "Ridge_Beta": np.round(beta_no_intercept, 5),
            "XGBoost_Gain": np.round(xgb_vals, 5),
            "LightGBM_Gain": np.round(lgb_vals, 5),
        }
    )
    return df_imp.sort_values("XGBoost_Gain", ascending=False).reset_index(drop=True)
