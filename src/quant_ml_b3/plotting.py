"""
Publication-grade visualizations for Quantitative ML & Portfolio Backtesting.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PALETTE = {
    "ML: LightGBM": "#059669",
    "ML: XGBoost": "#2563EB",
    "ML Baseline: Regressao Ridge": "#D97706",
    "Buy & Hold 1/N (IBrX 50)": "#4B5563",
    "BRAX11 (Indice Passivo)": "#DC2626",
}


def _format_brl_thousands(x: float, _pos: int | None = None) -> str:
    val = int(round(x))
    return f"R$ {val:,}".replace(",", ".")


def plot_feature_importance(imp_df: pd.DataFrame, out_path: str | Path) -> None:
    """Generate the 3-panel Feature Importance comparison (XGBoost Gain, LightGBM Gain, Ridge Standardized Beta)."""
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
        "Importância Cross-Sectional dos Atributos nos 3 Modelos (Universo IBrX 50)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def plot_equity_curve(
    curves: dict[str, pd.DataFrame],
    out_path: str | Path,
    title: str = "Curva de Patrimônio Acumulado — Walk-Forward (2015–2025, Líquido de 15 bps)",
) -> None:
    """Plot standalone full-height cumulative portfolio equity curve (R$) matching ggplot minimal style."""
    fig, ax = plt.subplots(figsize=(12.5, 6.2), dpi=160)
    fig.patch.set_facecolor("white")

    for label, df in curves.items():
        color = PALETTE.get(label, "#111827")
        lw = 2.0 if label.startswith("ML") else 1.5
        ls = "--" if label.startswith("BRAX") else "-"

        dates = pd.to_datetime(df["date"])
        eq = df["equity"].to_numpy(dtype=float)
        final_ret = (eq[-1] / eq[0] - 1.0) * 100.0
        final_str = f"R$ {int(round(eq[-1])):,}".replace(",", ".")
        legend_lbl = f"{label}  ({final_str} | {final_ret:+.1f}%)"
        ax.plot(dates, eq, label=legend_lbl, color=color, linewidth=lw, linestyle=ls, alpha=0.95)

    ax.set_title(title, fontsize=13.5, fontweight="bold", loc="left", pad=14)
    ax.set_ylabel("Patrimônio da Carteira (R$)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_format_brl_thousands))
    ax.grid(True, linestyle="-", color="#E5E7EB", linewidth=0.9)
    ax.spines[["top", "right", "left", "bottom"]].set_color("#D1D5DB")
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#E5E7EB",
        framealpha=0.95,
        fontsize=10,
    )

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def plot_drawdown_curve(
    curves: dict[str, pd.DataFrame],
    out_path: str | Path,
    title: str = "Drawdown (Queda Percentual desde o Pico Histórico — 2015–2025)",
) -> None:
    """Plot standalone full-height drawdown chart (%) without clipping at 0%."""
    fig, ax = plt.subplots(figsize=(12.5, 5.5), dpi=160)
    fig.patch.set_facecolor("white")

    ax.axhline(0.0, color="#9CA3AF", linewidth=1.0, zorder=1)

    for label, df in curves.items():
        color = PALETTE.get(label, "#111827")
        lw = 1.6 if label.startswith("ML") else 1.3
        ls = "--" if label.startswith("BRAX") else "-"

        dates = pd.to_datetime(df["date"])
        eq = df["equity"].to_numpy(dtype=float)
        dd = (eq / np.maximum.accumulate(eq) - 1.0) * 100.0
        max_dd = float(np.min(dd))
        legend_lbl = f"{label}  (Máx DD: {max_dd:.1f}%)"
        ax.plot(dates, dd, label=legend_lbl, color=color, linewidth=lw, linestyle=ls, alpha=0.9)

    ax.set_ylim(-53.0, 2.5)
    ax.set_title(title, fontsize=13.5, fontweight="bold", loc="left", pad=14)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.grid(True, linestyle="-", color="#E5E7EB", linewidth=0.9)
    ax.spines[["top", "right", "left", "bottom"]].set_color("#D1D5DB")
    ax.legend(
        loc="lower left",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#E5E7EB",
        framealpha=0.95,
        fontsize=9.5,
    )

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def plot_equity_and_drawdown(
    curves: dict[str, pd.DataFrame],
    out_path: str | Path,
    title: str = "Walk-Forward Backtest (2015–2025): Patrimônio Acumulado & Drawdown",
) -> None:
    """Plot combined 2-panel figure with proper y-limits and baseline so Drawdown at 0% is never clipped."""
    fig, (ax_eq, ax_dd) = plt.subplots(
        2,
        1,
        figsize=(13.5, 9.2),
        dpi=160,
        sharex=True,
        gridspec_kw={"height_ratios": [1.8, 1.1], "hspace": 0.14},
    )
    fig.patch.set_facecolor("white")

    ax_dd.axhline(0.0, color="#6B7280", linewidth=1.0, linestyle="-", zorder=1)

    for label, df in curves.items():
        color = PALETTE.get(label, "#111827")
        lw = 2.0 if label.startswith("ML") else 1.5
        ls = "--" if label.startswith("BRAX") else "-"

        dates = pd.to_datetime(df["date"])
        eq = df["equity"].to_numpy(dtype=float)
        dd = (eq / np.maximum.accumulate(eq) - 1.0) * 100.0

        final_ret = (eq[-1] / eq[0] - 1.0) * 100.0
        final_str = f"R$ {int(round(eq[-1])):,}".replace(",", ".")
        legend_lbl = f"{label}  ({final_str} | {final_ret:+.1f}%)"
        ax_eq.plot(dates, eq, label=legend_lbl, color=color, linewidth=lw, linestyle=ls, alpha=0.95)
        ax_dd.plot(dates, dd, label=label, color=color, linewidth=1.4, linestyle=ls, alpha=0.9)

    ax_eq.set_title(title, fontsize=13.5, fontweight="bold", loc="left", pad=12)
    ax_eq.set_ylabel("Patrimônio (R$)", fontsize=11)
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(_format_brl_thousands))
    ax_eq.grid(True, linestyle="-", color="#E5E7EB", linewidth=0.9)
    ax_eq.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E5E7EB", framealpha=0.95, fontsize=9.8)
    ax_eq.spines[["top", "right", "left", "bottom"]].set_color("#D1D5DB")

    ax_dd.set_ylim(-53.0, 3.0)
    ax_dd.set_ylabel("Drawdown (%)", fontsize=11)
    ax_dd.set_xlabel("Data do Pregão", fontsize=11)
    ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_dd.grid(True, linestyle="-", color="#E5E7EB", linewidth=0.9)
    ax_dd.spines[["top", "right", "left", "bottom"]].set_color("#D1D5DB")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def plot_oos_vs_walkforward_bars(
    df_oos: pd.DataFrame,
    df_wf: pd.DataFrame,
    out_path: str | Path,
) -> None:
    """
    Plot side-by-side horizontal bar charts comparing Net Profit (above R$ 100k initial capital)
    in the Frozen Out-of-Sample Test (10%: Nov/2024–Dec/2025) vs. Full 11-Year Walk-Forward (2015–2025).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.8), dpi=160)
    fig.patch.set_facecolor("white")

    # Panel A: Pure Out-of-Sample (10%)
    sub_oos = df_oos.sort_values("total_return", ascending=True).reset_index(drop=True)
    profits_oos_k = sub_oos["net_profit_brl"] / 1000.0
    colors_oos = [PALETTE.get(s, "#6B7280") for s in sub_oos["strategy"]]
    bars1 = ax1.barh(sub_oos["strategy"], profits_oos_k, color=colors_oos, height=0.58)
    ax1.set_title(
        "A) Teste Puro Out-of-Sample (10%: Nov/2024 a Dez/2025)\nModelo Congelado (Sem Retreinamento)",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xlabel("Lucro Líquido Acima de R$ 100 mil (em Milhares de Reais)")
    ax1.set_xlim(min(0.0, float(profits_oos_k.min()) * 1.2), float(profits_oos_k.max()) * 1.38)
    ax1.grid(axis="x", linestyle="--", alpha=0.45)
    ax1.spines[["top", "right"]].set_visible(False)

    for bar, (_, row) in zip(bars1, sub_oos.iterrows()):
        val_k = row["net_profit_brl"] / 1000.0
        pct = row["total_return"] * 100.0
        ax1.text(
            max(0.0, val_k) + 0.8,
            bar.get_y() + bar.get_height() / 2.0,
            f"+R$ {val_k:.1f}k ({pct:+.1f}%)",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )

    # Panel B: Full Walk-Forward (2015-2025)
    sub_wf = df_wf.sort_values("total_return", ascending=True).reset_index(drop=True)
    profits_wf_k = sub_wf["net_profit_brl"] / 1000.0
    colors_wf = [PALETTE.get(s, "#6B7280") for s in sub_wf["strategy"]]
    bars2 = ax2.barh(sub_wf["strategy"], profits_wf_k, color=colors_wf, height=0.58)
    ax2.set_title(
        "B) Walk-Forward Completo (11 Anos: 2015 a 2025)\nRetreino Anual a cada 12 Meses (Líquido de 15 bps)",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax2.set_xlabel("Lucro Líquido Acima de R$ 100 mil (em Milhares de Reais)")
    ax2.set_xlim(0.0, float(profits_wf_k.max()) * 1.32)
    ax2.grid(axis="x", linestyle="--", alpha=0.45)
    ax2.spines[["top", "right"]].set_visible(False)

    for bar, (_, row) in zip(bars2, sub_wf.iterrows()):
        val_k = row["net_profit_brl"] / 1000.0
        pct = row["total_return"] * 100.0
        ax2.text(
            val_k + 18.0,
            bar.get_y() + bar.get_height() / 2.0,
            f"+R$ {val_k:,.0f}k ({pct:+.0f}%)".replace(",", "."),
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
