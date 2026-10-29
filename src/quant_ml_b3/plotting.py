"""
Publication-grade visualizations for Quantitative Machine Learning & Model Evaluation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

MODEL_PALETTE = {
    "RIDGE": "#D97706",
    "XGBOOST": "#2563EB",
    "LIGHTGBM": "#059669",
    "BUY & HOLD (UNIVERSO)": "#4B5563",
}


def plot_feature_importance(imp_df: pd.DataFrame, out_path: str | Path) -> None:
    """Generate the 3-panel Feature Importance comparison (XGBoost Gain %, LightGBM Gain %, Ridge Standardized Beta)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), dpi=160)
    fig.patch.set_facecolor("white")

    # 1. XGBoost Gain (%)
    sub_xgb = imp_df.sort_values("XGBoost_Gain", ascending=True)
    axes[0].barh(sub_xgb["Feature"], sub_xgb["XGBoost_Gain"] * 100.0, color="#2563EB", height=0.68)
    axes[0].set_title("XGBoost (Ganho Relativo %)", fontsize=12, fontweight="bold", pad=10)
    axes[0].set_xlabel("Ganho Relativo (%)")
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)
    axes[0].spines[["top", "right"]].set_visible(False)

    # 2. LightGBM Gain (%)
    sub_lgb = imp_df.sort_values("LightGBM_Gain", ascending=True)
    axes[1].barh(sub_lgb["Feature"], sub_lgb["LightGBM_Gain"] * 100.0, color="#059669", height=0.68)
    axes[1].set_title("LightGBM (Ganho Relativo %)", fontsize=12, fontweight="bold", pad=10)
    axes[1].set_xlabel("Ganho Relativo (%)")
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)
    axes[1].spines[["top", "right"]].set_visible(False)

    # 3. Ridge Standardized Beta (bps)
    sub_rdg = imp_df.reindex(imp_df["Ridge_Beta"].abs().sort_values(ascending=True).index)
    colors_rdg = ["#D97706" if v >= 0 else "#DC2626" for v in sub_rdg["Ridge_Beta"]]
    axes[2].barh(sub_rdg["Feature"], sub_rdg["Ridge_Beta"] * 10000.0, color=colors_rdg, height=0.68)
    axes[2].axvline(0, color="#374151", linewidth=1.1)
    axes[2].set_title("Regressão Ridge (Coeficiente Padronizado +/-)", fontsize=12, fontweight="bold", pad=10)
    axes[2].set_xlabel("Coeficiente Padronizado (bps)")
    axes[2].grid(axis="x", linestyle="--", alpha=0.5)
    axes[2].spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Importância Cross-Sectional dos Atributos nos Modelos de ML (Universo IBrX 50)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def plot_test_metrics_summary(metrics_df: pd.DataFrame, out_path: str | Path) -> None:
    """Plot key Out-of-Sample metrics comparison on the Pure Test (10%) partition."""
    test_sub = metrics_df[metrics_df["Particao"].str.contains("Teste")].copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=160)
    fig.patch.set_facecolor("white")

    models = test_sub["Modelo"].tolist()
    colors = [MODEL_PALETTE.get(m, "#4B5563") for m in models]

    # Panel 1: Rank IC (Spearman)
    bars1 = axes[0].bar(models, test_sub["Rank_IC_Spearman"], color=colors, width=0.55)
    axes[0].set_title("Rank IC de Spearman (Teste 10%)", fontsize=11.5, fontweight="bold", pad=10)
    axes[0].set_ylabel("Correlação de Postos")
    axes[0].axhline(0, color="#6B7280", linewidth=0.9)
    axes[0].grid(axis="y", linestyle="--", alpha=0.45)
    axes[0].spines[["top", "right"]].set_visible(False)
    for bar in bars1:
        yval = bar.get_height()
        va = "bottom" if yval >= 0 else "top"
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, yval, f"{yval:+.4f}", ha="center", va=va, fontsize=9.5, fontweight="bold")

    # Panel 2: Directional Accuracy (%)
    bars2 = axes[1].bar(models, test_sub["Acuracia_Direcional_pct"], color=colors, width=0.55)
    axes[1].set_title("Acurácia Direcional % (Acima da Média)", fontsize=11.5, fontweight="bold", pad=10)
    axes[1].set_ylabel("Acurácia (%)")
    axes[1].set_ylim(40.0, 55.0)
    axes[1].axhline(50.0, color="#DC2626", linestyle="--", linewidth=1.0, label="Aleatório (50%)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.45)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(loc="upper right", frameon=True, fontsize=8.8)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.1f}%", ha="center", fontsize=9.5, fontweight="bold")

    # Panel 3: ROC-AUC
    bars3 = axes[2].bar(models, test_sub["AUC_ROC"], color=colors, width=0.55)
    axes[2].set_title("AUC-ROC (Teste Puro 10%)", fontsize=11.5, fontweight="bold", pad=10)
    axes[2].set_ylabel("Área sob a Curva ROC")
    axes[2].set_ylim(0.40, 0.58)
    axes[2].axhline(0.50, color="#DC2626", linestyle="--", linewidth=1.0, label="Aleatório (0.50)")
    axes[2].grid(axis="y", linestyle="--", alpha=0.45)
    axes[2].spines[["top", "right"]].set_visible(False)
    axes[2].legend(loc="upper right", frameon=True, fontsize=8.8)
    for bar in bars3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2.0, yval + 0.004, f"{yval:.4f}", ha="center", fontsize=9.5, fontweight="bold")

    fig.suptitle(
        "Avaliação Estatística dos Modelos de ML na Partição de Teste Puro (10% - 2024/2025)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def plot_top_k_cumulative_alpha(
    perf_dict: dict[str, pd.DataFrame],
    out_path: str | Path,
) -> None:
    """
    Plot the compound cumulative return of selecting Top 10 stocks vs. Equal-Weighted Universe (Buy & Hold baseline)
    on the Pure Test Partition (10%: Nov/2024 to Nov/2025).
    """
    fig, (ax_ret, ax_alp) = plt.subplots(
        2, 1, figsize=(12.5, 8.0), dpi=160, sharex=True, gridspec_kw={"height_ratios": [1.4, 1.0], "hspace": 0.15}
    )
    fig.patch.set_facecolor("white")

    # Plot baseline first
    any_model = next(iter(perf_dict.values()))
    dates = pd.to_datetime(any_model["date"])
    uni_cum = any_model["universe_cumulative"] * 100.0
    ax_ret.plot(dates, uni_cum, label=f"Buy & Hold (Universo Igualitário)  [{uni_cum.iloc[-1]:+.1f}%]", color="#4B5563", linewidth=1.8, linestyle="--")

    for model_name, df_perf in perf_dict.items():
        color = MODEL_PALETTE.get(model_name.upper(), "#111827")
        top_cum = df_perf["top_k_cumulative"] * 100.0
        alp_cum = df_perf["alpha_cumulative"] * 100.0

        lbl_ret = f"Top 10: {model_name}  [{top_cum.iloc[-1]:+.1f}%]"
        lbl_alp = f"Alpha: {model_name}  [{alp_cum.iloc[-1]:+.1f}%]"

        ax_ret.plot(dates, top_cum, label=lbl_ret, color=color, linewidth=2.0)
        ax_alp.plot(dates, alp_cum, label=lbl_alp, color=color, linewidth=1.8)

    ax_ret.set_title("A) Retorno Acumulado Composto: Top 10 Selecionado vs. Universo (Teste Puro 10%)", fontsize=12, fontweight="bold", loc="left", pad=10)
    ax_ret.set_ylabel("Retorno Acumulado (%)", fontsize=10.5)
    ax_ret.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
    ax_ret.grid(True, linestyle="-", color="#E5E7EB", linewidth=0.8)
    ax_ret.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=9.5)
    ax_ret.spines[["top", "right", "left", "bottom"]].set_color("#D1D5DB")

    ax_alp.axhline(0.0, color="#6B7280", linewidth=1.0, linestyle="-")
    ax_alp.set_title("B) Alpha Puro Acumulado (Excesso de Retorno acima do Buy & Hold)", fontsize=12, fontweight="bold", loc="left", pad=10)
    ax_alp.set_ylabel("Alpha Acumulado (%)", fontsize=10.5)
    ax_alp.set_xlabel("Data de Rebalanceamento", fontsize=10.5)
    ax_alp.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
    ax_alp.grid(True, linestyle="-", color="#E5E7EB", linewidth=0.8)
    ax_alp.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=9.5)
    ax_alp.spines[["top", "right", "left", "bottom"]].set_color("#D1D5DB")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)



def plot_walk_forward_comparison(
    df_cum: pd.DataFrame,
    df_summary: pd.DataFrame,
    out_path: str | Path,
) -> None:
    """
    Generate the 2-panel Walk-Forward visual comparison (2015-2025):
      - Panel A: Multi-year cumulative equity curves (Walk-Forward Dynamic vs. Frozen vs. Buy & Hold)
      - Panel B: Bar chart comparing CAGR (% p.a.) between Dynamic Retrained and Static Frozen models
    """
    fig, (ax_curve, ax_bars) = plt.subplots(2, 1, figsize=(15, 12), dpi=160, gridspec_kw={"height_ratios": [1.4, 1.0]})
    fig.patch.set_facecolor("white")

    dates = pd.to_datetime(df_cum["date"])

    # 1. Equity Curves
    if "Universe" in df_cum.columns:
        u_ret = df_cum["Universe"].iloc[-1]
        ax_curve.plot(
            dates,
            df_cum["Universe"],
            label=f"Buy & Hold (Universo 1/N) [{u_ret:+.1f}%]",
            color="#4B5563",
            linewidth=2.2,
            linestyle="--",
        )

    styles = {
        "Xgboost_WF": ("#2563EB", "-", 2.2, "XGBoost (Walk-Forward Anual)"),
        "Ridge_WF": ("#D97706", "-", 2.2, "Ridge (Walk-Forward Anual)"),
        "Lightgbm_WF": ("#059669", "-", 2.0, "LightGBM (Walk-Forward Anual)"),
        "Xgboost_Frozen": ("#93C5FD", ":", 1.6, "XGBoost (Congelado)"),
        "Ridge_Frozen": ("#FCD34D", ":", 1.6, "Ridge (Congelado)"),
        "Lightgbm_Frozen": ("#6EE7B7", ":", 1.6, "LightGBM (Congelado)"),
    }

    for col, (color, lstyle, lwidth, label_prefix) in styles.items():
        if col in df_cum.columns:
            final_val = df_cum[col].iloc[-1]
            lbl = f"{label_prefix} [{final_val:+.1f}%]"
            ax_curve.plot(dates, df_cum[col], label=lbl, color=color, linestyle=lstyle, linewidth=lwidth)

    ax_curve.set_title(
        "A) Curva de Patrimônio Acumulado (2015-2025): Walk-Forward Anual vs. Modelos Congelados vs. Buy & Hold",
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax_curve.set_ylabel("Retorno Acumulado Composto (%)", fontsize=10.5)
    ax_curve.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
    ax_curve.grid(True, linestyle="-", color="#E5E7EB", linewidth=0.8)
    ax_curve.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=9.2, ncol=2)
    ax_curve.spines[["top", "right", "left", "bottom"]].set_color("#D1D5DB")

    # 2. Bar Panel: Retraining Benefit (CAGR % and Total Gain)
    df_models = df_summary[df_summary["Estrategia"].str.contains("ML:")].copy()
    model_types = ["RIDGE", "XGBOOST", "LIGHTGBM"]
    x = np.arange(len(model_types))
    width = 0.32

    wf_cagrs = []
    frz_cagrs = []
    gains = []

    for m in model_types:
        row_wf = df_models[df_models["Estrategia"].str.upper().str.contains(m) & df_models["Estrategia"].str.contains("Walk-Forward")]
        row_frz = df_models[df_models["Estrategia"].str.upper().str.contains(m) & df_models["Estrategia"].str.contains("Congelado")]
        c_wf = float(row_wf["CAGR_pct"].iloc[0]) if len(row_wf) else 0.0
        c_frz = float(row_frz["CAGR_pct"].iloc[0]) if len(row_frz) else 0.0
        wf_cagrs.append(c_wf)
        frz_cagrs.append(c_frz)
        gains.append(c_wf - c_frz)

    rects1 = ax_bars.bar(x - width / 2, wf_cagrs, width, label="Walk-Forward (Retreino Anual)", color="#1E40AF")
    rects2 = ax_bars.bar(x + width / 2, frz_cagrs, width, label="Congelado (Sem Retreino)", color="#94A3B8")

    for i in range(len(model_types)):
        gain_val = gains[i]
        c_wf = wf_cagrs[i]
        c_frz = frz_cagrs[i]
        ax_bars.annotate(
            f"{c_wf:.1f}%",
            (x[i] - width / 2, c_wf),
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            xytext=(0, 2),
            textcoords="offset points",
        )
        ax_bars.annotate(
            f"{c_frz:.1f}%",
            (x[i] + width / 2, c_frz),
            ha="center",
            va="bottom",
            fontsize=9.5,
            xytext=(0, 2),
            textcoords="offset points",
        )
        sign = "+" if gain_val >= 0 else ""
        ax_bars.annotate(
            f"Ganho do Retreino: {sign}{gain_val:.1f}% a.a.",
            (x[i], max(c_wf, c_frz) * 0.5),
            ha="center",
            va="center",
            fontsize=9.0,
            fontweight="bold",
            color="#047857" if gain_val >= 0 else "#DC2626",
            bbox=dict(boxstyle="round,pad=0.25", fc="#F0FDF4" if gain_val >= 0 else "#FEF2F2", ec="#86EFAC" if gain_val >= 0 else "#FECACA", lw=0.8),
        )

    ax_bars.set_title(
        "B) Rentabilidade Anualizada (CAGR % a.a.): Impacto Empírico do Retreino Walk-Forward",
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax_bars.set_ylabel("CAGR (% a.a.)", fontsize=10.5)
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(["Regressão Ridge", "XGBoost", "LightGBM"], fontsize=11, fontweight="bold")
    ax_bars.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax_bars.grid(axis="y", linestyle="-", color="#E5E7EB", linewidth=0.8)
    ax_bars.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=9.5)
    ax_bars.spines[["top", "right", "left", "bottom"]].set_color("#D1D5DB")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def plot_weighting_schemes_comparison(
    df_cum: pd.DataFrame,
    df_summary: pd.DataFrame,
    out_path: str | Path,
) -> None:
    """
    Generate a 2-panel figure comparing Hard Cut 1/N (Top 10) vs Soft Weighting (Top 15 Rank Linear):
      - Panel A: Cumulative compound equity curves (2015-2025).
      - Panel B: Grouped bars comparing Annualized Return (CAGR %) and Sharpe Ratio.
    """
    fig, (ax_curve, ax_bars) = plt.subplots(2, 1, figsize=(15, 12), dpi=160, gridspec_kw={"height_ratios": [1.4, 1.0]})
    fig.patch.set_facecolor("white")

    dates = pd.to_datetime(df_cum["date"])

    # 1. Curve Panel
    if "Universe" in df_cum.columns:
        u_ret = df_cum["Universe"].iloc[-1]
        ax_curve.plot(
            dates,
            df_cum["Universe"],
            label=f"Buy & Hold (Universo 1/N) [{u_ret:+.1f}%]",
            color="#4B5563",
            linewidth=2.2,
            linestyle="--",
        )

    curve_styles = {
        "Ridge_HardCut": ("#D97706", "-", 2.2, "Ridge: Hard Cut Top 10"),
        "Ridge_SoftWeight": ("#F59E0B", "--", 2.0, "Ridge: Soft Weight Top 15"),
        "Xgboost_HardCut": ("#2563EB", "-", 2.2, "XGBoost: Hard Cut Top 10"),
        "Xgboost_SoftWeight": ("#60A5FA", "--", 2.0, "XGBoost: Soft Weight Top 15"),
        "Lightgbm_HardCut": ("#059669", "-", 2.2, "LightGBM: Hard Cut Top 10"),
        "Lightgbm_SoftWeight": ("#34D399", "--", 2.0, "LightGBM: Soft Weight Top 15"),
    }

    for col, (color, lstyle, lwidth, label_prefix) in curve_styles.items():
        if col in df_cum.columns:
            final_val = df_cum[col].iloc[-1]
            lbl = f"{label_prefix} [{final_val:+.1f}%]"
            ax_curve.plot(dates, df_cum[col], label=lbl, color=color, linestyle=lstyle, linewidth=lwidth)

    ax_curve.set_title(
        "A) Curva de Patrimônio Acumulado (2015-2025): Hard Cut (Top 10 1/N) vs. Soft Weighting (Top 15 Rank Linear)",
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax_curve.set_ylabel("Retorno Acumulado Composto (%)", fontsize=10.5)
    ax_curve.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
    ax_curve.grid(True, linestyle="-", color="#E5E7EB", linewidth=0.8)
    ax_curve.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=9.2, ncol=2)
    ax_curve.spines[["top", "right", "left", "bottom"]].set_color("#D1D5DB")

    # 2. Bar Panel: CAGR & Sharpe Comparison
    df_models = df_summary[df_summary["Modelo"].str.contains("ML:")].copy()
    models = ["RIDGE", "XGBOOST", "LIGHTGBM"]
    x = np.arange(len(models))
    width = 0.35

    cagr_hard, cagr_soft = [], []
    sharpe_hard, sharpe_soft = [], []

    for m in models:
        r_hd = df_models[df_models["Modelo"].str.upper().str.contains(m) & df_models["Esquema"].str.contains("Hard Cut")]
        r_sf = df_models[df_models["Modelo"].str.upper().str.contains(m) & df_models["Esquema"].str.contains("Soft Weighting")]

        c_hd = float(r_hd["CAGR_pct"].iloc[0]) if len(r_hd) else 0.0
        c_sf = float(r_sf["CAGR_pct"].iloc[0]) if len(r_sf) else 0.0
        s_hd = float(r_hd["Sharpe_Ratio"].iloc[0]) if len(r_hd) else 0.0
        s_sf = float(r_sf["Sharpe_Ratio"].iloc[0]) if len(r_sf) else 0.0

        cagr_hard.append(c_hd)
        cagr_soft.append(c_sf)
        sharpe_hard.append(s_hd)
        sharpe_soft.append(s_sf)

    bars1 = ax_bars.bar(x - width / 2, cagr_hard, width, label="Hard Cut (Top 10 1/N com Histerese)", color="#1E40AF")
    bars2 = ax_bars.bar(x + width / 2, cagr_soft, width, label="Soft Weighting (Top 15 Rank Linear)", color="#059669")

    for i in range(len(models)):
        ax_bars.annotate(
            f"{cagr_hard[i]:.1f}% a.a.\n(Sharpe {sharpe_hard[i]:.2f})",
            (x[i] - width / 2, cagr_hard[i]),
            ha="center",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
            xytext=(0, 2),
            textcoords="offset points",
        )
        ax_bars.annotate(
            f"{cagr_soft[i]:.1f}% a.a.\n(Sharpe {sharpe_soft[i]:.2f})",
            (x[i] + width / 2, cagr_soft[i]),
            ha="center",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
            xytext=(0, 2),
            textcoords="offset points",
        )

    ax_bars.set_title(
        "B) Desempenho Anualizado (CAGR % a.a. e Sharpe): Comparação entre Políticas de Construção de Portfólio",
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax_bars.set_ylabel("CAGR (% a.a.)", fontsize=10.5)
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(["Regressão Ridge", "XGBoost", "LightGBM"], fontsize=11, fontweight="bold")
    ax_bars.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax_bars.grid(axis="y", linestyle="-", color="#E5E7EB", linewidth=0.8)
    ax_bars.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=9.5)
    ax_bars.spines[["top", "right", "left", "bottom"]].set_color("#D1D5DB")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
