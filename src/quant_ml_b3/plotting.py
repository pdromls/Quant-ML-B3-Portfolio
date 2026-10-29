"""
Publication-grade visualizations for Quantitative Machine Learning & Model Evaluation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
