# -*- coding: utf-8 -*-
"""
B3 Quant Alpha Terminal — Machine Learning Portfolio Engine
Interactive Web Application for Stock Selection and Performance Monitoring on the Brazilian Stock Market.
"""

from __future__ import annotations
from pathlib import Path
import sys

# Ensure src is accessible
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant_ml_b3 import ui
from quant_ml_b3.data_loader import load_local_market_data, validate_market_data

# Page configuration
st.set_page_config(
    page_title="B3 Quant Alpha Terminal | ML Stock Selection",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme-aware design system (Light/Dark follow Streamlit's native theme)
ui.inject_css()

# -------------------------------------------------------------
# CACHED DATA LOADERS
# -------------------------------------------------------------
@st.cache_data
def load_dashboard_tables(cache_key: float):
    t_dir = ROOT_DIR / "outputs" / "tables"
    df_curves = pd.read_csv(t_dir / "walkforward_monthly_curves.csv")
    df_monthly = pd.read_csv(t_dir / "walkforward_monthly_returns.csv")
    df_holdings = pd.read_csv(t_dir / "walkforward_monthly_holdings.csv")
    df_current = pd.read_csv(t_dir / "current_month_prediction.csv")
    df_metrics = pd.read_csv(t_dir / "comparison_weighting_schemes.csv")
    df_importance = pd.read_csv(t_dir / "feature_importance.csv")
    return {
        "curves": df_curves,
        "monthly": df_monthly,
        "holdings": df_holdings,
        "current": df_current,
        "metrics": df_metrics,
        "importance": df_importance,
    }

@st.cache_resource
def get_cached_market():
    return load_local_market_data(ROOT_DIR / "data" / "processed")

@st.cache_resource
def get_cached_bova():
    bova_file = ROOT_DIR / "data" / "processed" / "benchmark_bova11.csv"
    if bova_file.exists():
        df_b = pd.read_csv(bova_file)
        df_b["date"] = pd.to_datetime(df_b["date"])
        return df_b.set_index("date")["close"]
    return None

# Load tables with auto-invalidation on file update
_tables_dir = ROOT_DIR / "outputs" / "tables"
_holdings_file = _tables_dir / "walkforward_monthly_holdings.csv"
_mtime = _holdings_file.stat().st_mtime if _holdings_file.exists() else 0.0
tables = load_dashboard_tables(_mtime)
df_curves = tables["curves"]
df_monthly = tables["monthly"]
df_holdings = tables["holdings"]
df_current = tables["current"]
df_metrics = tables["metrics"]
df_importance = tables["importance"]

base_market = get_cached_market()
curve_dates = pd.to_datetime(df_curves["date"])
last_data_date = curve_dates.max()
first_month = pd.to_datetime(df_curves["month_label"].min() + "-01")
last_month = pd.to_datetime(df_curves["month_label"].max() + "-01")
wf_period = f"{first_month:%m/%Y} – {last_month:%m/%Y}"
wf_years = f"{first_month:%Y}–{last_month:%Y}"


def live_market_active() -> bool:
    lm = st.session_state.get("live_market")
    return lm is not None and hasattr(lm, "dates") and len(lm.dates) > 0


# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
with st.sidebar:
    ui.brand("Quant Alpha Terminal", "ML Cross-Sectional · IBrX 50")

    # 1. Data Source Mode
    ui.side_label("Fonte de dados")
    data_mode = st.segmented_control(
        "Fonte de dados",
        ["Base Local Higienizada (2012-2026)", "API Yahoo Finance (Ao Vivo)"],
        default="Base Local Higienizada (2012-2026)",
        required=True,
        format_func=lambda m: "Base local" if m.startswith("Base") else "Yahoo Finance",
        label_visibility="collapsed",
        width="stretch",
        help="Alterne entre a base higienizada pré-carregada ou dados atualizados online via yfinance.",
    )
    base_dates = pd.to_datetime(base_market.dates)
    if data_mode.startswith("Base"):
        ui.hint(
            f"Base higienizada {base_dates.min():%Y}–{base_dates.max():%Y} · "
            f"{len(base_market.tickers)} ativos · {len(base_dates):,} pregões".replace(",", ".")
        )
    else:
        ui.hint(f"A conexão com Yahoo Finance baixa cotações atualizadas das {len(base_market.tickers)} ações do IBrX 50.")
        c_btn1, c_btn2 = st.columns([1.5, 1])
        with c_btn1:
            btn_update = st.button("Atualizar cotações", icon=":material/refresh:", width="stretch")
        with c_btn2:
            btn_reset = st.button("Resetar", icon=":material/restart_alt:", width="stretch")

        if btn_reset:
            if "live_market" in st.session_state:
                del st.session_state["live_market"]
            if "live_updated_at" in st.session_state:
                del st.session_state["live_updated_at"]
            st.success("Sessão da API limpa! Usando base local.")

        if btn_update:
            with st.spinner("Baixando cotações atualizadas da B3 via Yahoo Finance..."):
                from quant_ml_b3.data_loader import download_b3_universe_yfinance
                try:
                    live_mkt = download_b3_universe_yfinance(base_market.tickers, start="2024-01-01", end=None)
                    if hasattr(live_mkt, "dates") and len(live_mkt.dates) > 0:
                        st.session_state["live_market"] = live_mkt
                        st.session_state["live_updated_at"] = pd.Timestamp.now().strftime("%d/%m/%Y %H:%M:%S")
                        st.success(f"Atualizado! {len(live_mkt.dates)} pregões carregados até {live_mkt.dates[-1].strftime('%d/%m/%Y')}.")
                    else:
                        st.error("A API não retornou dados válidos. Tente novamente.")
                except Exception as e:
                    st.error(f"Erro ao conectar com API: {e}")

        if "live_market" in st.session_state:
            if live_market_active():
                lm = st.session_state["live_market"]
                ui.notice(f"<b>API conectada:</b> {len(lm.dates)} pregões carregados até {lm.dates[-1].strftime('%d/%m/%Y')}.", "ok")
            else:
                del st.session_state["live_market"]

    # 2. Model Selection
    ui.side_label("Modelo de Machine Learning")
    model_choice = st.radio(
        "Modelo de Machine Learning",
        [
            "Regressão Ridge (L2 Regularized)",
            "XGBoost (Depth-Wise Tree)",
            "LightGBM (Leaf-Wise Tree)",
            "Comparar Todos os Modelos",
        ],
        index=0,
        format_func=lambda m: {
            "Regressão Ridge (L2 Regularized)": ":blue[●] Regressão Ridge",
            "XGBoost (Depth-Wise Tree)": ":orange[●] XGBoost",
            "LightGBM (Leaf-Wise Tree)": ":violet[●] LightGBM",
            "Comparar Todos os Modelos": "○ Comparar todos os modelos",
        }[m],
        captions=["Regularização L2", "Árvores depth-wise", "Árvores leaf-wise", "Ridge, XGBoost e LightGBM"],
        label_visibility="collapsed",
    )

    # 3. Portfolio Weighting Policy
    ui.side_label("Construção de carteira")
    weighting_choice = st.radio(
        "Política de Construção de Carteira",
        [
            "Soft Weighting (Top 15 Rank Linear)",
            "Hard Cut (Top 10 com Margem de Tolerância)",
        ],
        index=0,
        format_func=lambda w: "Soft Weighting · Top 15" if w.startswith("Soft") else "Hard Cut · Top 10",
        captions=[
            "Pesos lineares decrescentes por rank no Top 15; elimina o efeito falésia.",
            "10% igual por ativo no Top 10, com margem de tolerância anti-giro.",
        ],
        label_visibility="collapsed",
    )

    ui.side_label("Metodologia quant")
    ui.definition_list(
        [
            ("Rebalanceamento", "Mensal · 1º pregão"),
            ("Normalização", "Z-score cross-sectional"),
            ("Purga anti-leakage", '<span class="qa-num">21 dias</span>'),
            ("Custos", '<span class="qa-num">0,15%</span> por transação'),
            ("Alvo", "Excesso de retorno 21d"),
        ]
    )
    ui.side_footer(
        f"Dados até <span class='qa-num'>{last_data_date:%d/%m/%Y}</span> · tabelas recarregadas automaticamente.<br>"
        "Tema claro/escuro: menu ⋮ → Settings → Theme."
    )

# Model Key Mapping
model_map = {
    "Regressão Ridge (L2 Regularized)": "Ridge",
    "XGBoost (Depth-Wise Tree)": "Xgboost",
    "LightGBM (Leaf-Wise Tree)": "Lightgbm",
}
is_soft = weighting_choice.startswith("Soft")
scheme_suffix = "SoftWeight" if is_soft else "HardCut"
scheme_short = "Soft Weighting Top 15" if is_soft else "Hard Cut Top 10"
colors = ui.MODEL_COLORS

# -------------------------------------------------------------
# MAIN APP HEADER
# -------------------------------------------------------------
if live_market_active():
    _lm = st.session_state["live_market"]
    source_chip = (f"Yahoo Finance · {len(_lm.dates)} pregões até {_lm.dates[-1]:%d/%m/%Y}", "live")
else:
    source_chip = (f"Base local · <span class='qa-num'>{len(base_market.dates):,}</span> pregões".replace(",", "."), "ok")
ui.topbar(
    "Painel executivo",
    "B3 Machine Learning Cross-Sectional Alpha Terminal",
    [
        source_chip,
        (f"Walk-forward <span class='qa-num'>{wf_period}</span>", None),
    ],
)

# -------------------------------------------------------------
# APP TABS
# -------------------------------------------------------------
tab_perf, tab_pred_real, tab_explain, tab_audit = st.tabs(
    [
        f"Desempenho histórico ({wf_years})",
        "Previsão vs. realidade",
        "Explicabilidade das features",
        "Auditoria de qualidade de dados",
    ]
)

# =============================================================
# TAB 1: DESEMPENHO HISTÓRICO (WALK-FORWARD)
# =============================================================
with tab_perf:
    row_univ = df_metrics[df_metrics["Modelo"].str.contains("Buy & Hold")]
    row_ibov = df_metrics[df_metrics["Modelo"].str.contains("Ibovespa")]

    tags = [("Política", scheme_short)]
    if model_choice != "Comparar Todos os Modelos":
        selected_key = model_map[model_choice]
        tags.insert(0, ("Modelo", ui.MODEL_LABELS[selected_key]))
    else:
        tags.insert(0, ("Modelo", "Comparação dos 3"))

    row_sel = pd.DataFrame()
    if model_choice != "Comparar Todos os Modelos":
        row_sel = df_metrics[
            (df_metrics["Modelo"].str.contains(selected_key.upper()))
            & (df_metrics["Esquema"].str.startswith("Soft" if is_soft else "Hard"))
        ]
        if not row_sel.empty:
            tags.append(("Turnover médio", f"<span class='qa-num'>{ui.fmt_num(row_sel.iloc[0]['Turnover_Medio_Mensal_pct'], 1)}%</span>"))

    ui.section_header(
        f"Simulação multi-ano fora da amostra · {wf_years}",
        "Walk-forward com rebalanceamento mensal e custos incluídos. Comparação contra Buy &amp; Hold 1/N e Ibovespa.",
        tags,
    )

    # KPI cards for selected model and scheme
    if not row_sel.empty:
        m_data = row_sel.iloc[0]
        u_data = row_univ.iloc[0]
        i_data = row_ibov.iloc[0] if not row_ibov.empty else None
        vol_diff = m_data["Volatilidade_Anualizada_pct"] - u_data["Volatilidade_Anualizada_pct"]
        dd_diff = m_data["Max_Drawdown_pct"] - u_data["Max_Drawdown_pct"]
        sharpe_diff = m_data["Sharpe_Ratio"] - u_data["Sharpe_Ratio"]
        cagr_diff = m_data["CAGR_pct"] - u_data["CAGR_pct"]
        ui.kpi_cards(
            [
                {
                    "label": "Retorno total",
                    "value": ui.fmt_pct(m_data["Retorno_Acumulado_pct"], 1),
                    "foot": f"B&amp;H <span class='qa-num'>{ui.fmt_pct(u_data['Retorno_Acumulado_pct'], 0)}</span>",
                    "help": "Retorno composto de toda a simulação walk-forward, líquido de custos.",
                },
                {
                    "label": "CAGR",
                    "unit": "% a.a.",
                    "value": ui.fmt_pct(m_data["CAGR_pct"], 2, sign=False),
                    "foot": "vs. B&amp;H",
                    "delta": ui.fmt_pp(cagr_diff),
                    "delta_tone": ui.tone_of(cagr_diff),
                },
                {
                    "label": "Índice Sharpe",
                    "value": ui.fmt_num(m_data["Sharpe_Ratio"], 2),
                    "foot": f"B&amp;H <span class='qa-num'>{ui.fmt_num(u_data['Sharpe_Ratio'], 2)}</span>",
                    "delta": ui.fmt_num(sharpe_diff, 2, sign=True),
                    "delta_tone": ui.tone_of(sharpe_diff),
                },
                {
                    "label": "Volatilidade",
                    "unit": "anual",
                    "value": ui.fmt_pct(m_data["Volatilidade_Anualizada_pct"], 1, sign=False),
                    "foot": f"B&amp;H <span class='qa-num'>{ui.fmt_pct(u_data['Volatilidade_Anualizada_pct'], 1, sign=False)}</span>",
                    "delta": ui.fmt_pp(vol_diff, 1),
                    "delta_tone": ui.tone_of(vol_diff, higher_is_better=False),
                },
                {
                    "label": "Max drawdown",
                    "value": ui.fmt_pct(m_data["Max_Drawdown_pct"], 1, sign=False),
                    "tone": "neg",
                    "foot": f"B&amp;H <span class='qa-num'>{ui.fmt_pct(u_data['Max_Drawdown_pct'], 1, sign=False)}</span>",
                    "delta": ui.fmt_pp(dd_diff, 1),
                    "delta_tone": ui.tone_of(dd_diff),
                },
                {
                    "label": "Alpha líquido",
                    "unit": "anual",
                    "value": ui.fmt_pct(m_data["Alpha_vs_Universo_pct"], 2),
                    "tone": ui.tone_of(m_data["Alpha_vs_Universo_pct"]),
                    "foot": "Excedente sobre o universo"
                    if i_data is None
                    else f"vs. Ibov <span class='qa-num'>{ui.fmt_pp(m_data['CAGR_pct'] - i_data['CAGR_pct'])}</span>",
                    "help": "CAGR do modelo menos CAGR do Buy & Hold 1/N (excedente puro).",
                },
            ]
        )
    elif model_choice == "Comparar Todos os Modelos":
        ui.notice("<b>Modo de comparação:</b> as três curvas aparecem juntas abaixo; as métricas de cada modelo estão na tabela resumo comparativa.")

    # Equity curve
    with ui.panel(
        "equity",
        "Curva de patrimônio acumulado (juros compostos)",
        f"Retorno composto desde {curve_dates.min() - pd.offsets.MonthBegin(1):%m/%Y}, líquido de custos",
        info="equity",
        controls=True,
    ) as ctrl:
        with ctrl:
            scale = st.segmented_control(
                "Escala",
                ["Linear", "Log"],
                default="Linear",
                required=True,
                key="equity_scale",
                label_visibility="collapsed",
            )
        log_scale = scale == "Log"

        def curve_y(series: pd.Series) -> pd.Series:
            return 1.0 + series / 100.0 if log_scale else series

        hover_fmt = "R$ %{y:.2f}" if log_scale else "%{y:+.1f}%"
        fig_curve = go.Figure()
        fig_curve.add_trace(
            go.Scatter(
                x=curve_dates,
                y=curve_y(df_curves["Universe"]),
                mode="lines",
                name="Buy & Hold (Universo 1/N)",
                line=dict(color=ui.UNIVERSE_COLOR, width=1.8, dash="dash"),
                hovertemplate=f"Buy & Hold: {hover_fmt}<extra></extra>",
            )
        )
        if "Ibovespa" in df_curves.columns:
            fig_curve.add_trace(
                go.Scatter(
                    x=curve_dates,
                    y=curve_y(df_curves["Ibovespa"]),
                    mode="lines",
                    name="Índice Ibovespa (BOVA11)",
                    line=dict(color=ui.IBOV_COLOR, width=2, dash="dot"),
                    hovertemplate=f"Ibovespa: {hover_fmt}<extra></extra>",
                )
            )

        curve_model_keys = ["Ridge", "Xgboost", "Lightgbm"] if model_choice == "Comparar Todos os Modelos" else [model_map[model_choice]]
        for m_key in curve_model_keys:
            col_name = f"{m_key}_{scheme_suffix}"
            if col_name in df_curves.columns:
                single = len(curve_model_keys) == 1
                fig_curve.add_trace(
                    go.Scatter(
                        x=curve_dates,
                        y=curve_y(df_curves[col_name]),
                        mode="lines",
                        name=f"ML: {ui.MODEL_LABELS[m_key]} ({scheme_short})",
                        line=dict(color=colors[m_key], width=2.6),
                        fill="tonexty" if single else None,
                        fillcolor=ui.rgba(colors[m_key], 0.08) if single else None,
                        hovertemplate=f"{ui.MODEL_LABELS[m_key]}: {hover_fmt}<extra></extra>",
                    )
                )

        ui.style_fig(fig_curve, 440, y_suffix=None if log_scale else "%")
        fig_curve.update_xaxes(hoverformat="%d/%m/%Y")
        if log_scale:
            fig_curve.update_yaxes(type="log", tickprefix="R$ ", title="Patrimônio de R$ 1 (escala log)")
        else:
            fig_curve.update_yaxes(title="Retorno acumulado composto (%)")
        st.plotly_chart(fig_curve, width="stretch")
        ui.explain(
            f"Cada ponto mostra quanto <b>R$ 1 investido no início de {first_month:%m/%Y}</b> teria rendido até aquela data, "
            "com rebalanceamento mensal e custos de 0,15% por transação. A distância vertical entre a linha do modelo e as "
            "tracejadas é o ganho acumulado do modelo sobre o mercado."
        )

    # Drawdown (separate panel, same time axis)
    with ui.panel(
        "drawdown",
        "Drawdown · queda desde o pico anterior",
        "Quanto cada carteira está abaixo do maior valor que já havia atingido · fechamentos mensais",
        info="drawdown",
    ):
        dd_series = [("Buy & Hold 1/N", "Universe", ui.UNIVERSE_COLOR, "dash")]
        if "Ibovespa" in df_curves.columns:
            dd_series.append(("Ibovespa (BOVA11)", "Ibovespa", ui.IBOV_COLOR, "dot"))
        for m_key in curve_model_keys:
            if f"{m_key}_{scheme_suffix}" in df_curves.columns:
                dd_series.append((f"{ui.MODEL_LABELS[m_key]} · {scheme_short}", f"{m_key}_{scheme_suffix}", colors[m_key], "solid"))

        fig_dd = go.Figure()
        single_model_dd = len(curve_model_keys) == 1
        for name, col, color, dash in dd_series:
            is_model = dash == "solid"
            dd_vals = ui.drawdown_series(df_curves[col])
            line_color = ui.NEG_COLOR if (is_model and single_model_dd) else color
            fig_dd.add_trace(
                go.Scatter(
                    x=curve_dates,
                    y=dd_vals,
                    mode="lines",
                    name=name,
                    line=dict(color=line_color, width=2 if is_model else 1.5, dash=dash),
                    fill="tozeroy" if (is_model and single_model_dd) else None,
                    fillcolor=ui.rgba(ui.NEG_COLOR, 0.12) if (is_model and single_model_dd) else None,
                    hovertemplate=f"{name}: %{{y:.1f}}%<extra></extra>",
                )
            )
        ui.style_fig(fig_dd, 250)
        fig_dd.update_xaxes(hoverformat="%d/%m/%Y")
        fig_dd.update_yaxes(title="Drawdown (%)", rangemode="tozero")
        st.plotly_chart(fig_dd, width="stretch")

        dd_rows = []
        for name, col, color, dash in dd_series[::-1]:
            stats = ui.drawdown_stats(df_curves["date"], df_curves[col])
            peak = "início" if stats["peak_date"] is None else f"{stats['peak_date']:%m/%Y}"
            rec = "não recuperado" if stats["recovery_months"] is None else f"{stats['recovery_months']} meses"
            dd_rows.append(
                [
                    ui.dot(color) + name,
                    f"<span class='qa-num qa-neg'>{ui.fmt_pct(stats['worst'], 1, sign=False)}</span>",
                    f"<span class='qa-num'>{peak} → {stats['trough_date']:%m/%Y}</span>",
                    f"<span class='qa-num'>{rec}</span>",
                    f"<span class='qa-num'>{ui.fmt_pct(stats['current'], 1, sign=False)}</span>",
                ]
            )
        ui.stats_table(["Série", "Pior queda", "Pico → fundo", "Recuperação (do fundo ao pico)", "Situação atual"], dd_rows)
        ui.explain(
            "O gráfico nunca passa de 0%: <b>0% significa que a carteira está no seu topo histórico</b>. Cada vale é uma perda a "
            "partir do pico: a <b>profundidade</b> é o tamanho da perda e a <b>largura</b> é o tempo até recuperar. Quando a linha "
            "do modelo fica acima da tracejada, o modelo caiu menos que o mercado naquele período."
        )

    # Performance Comparison Table
    with ui.panel(
        "comparison",
        "Tabela resumo comparativa",
        "Linha destacada = seleção atual · verde e negrito = melhor valor da coluna entre os modelos",
        info="comparison",
    ):
        df_cmp = df_metrics.copy()
        df_cmp["Esquema"] = (
            df_cmp["Esquema"]
            .where(~df_cmp["Esquema"].str.startswith("Hard Cut"), "Hard Cut (Top 10, margem de tolerância)")
            .where(~df_cmp["Esquema"].str.startswith("Soft"), "Soft Weighting (Top 15 rank linear)")
        )
        is_model_row = df_cmp["Modelo"].str.startswith("ML:")
        higher_better = {
            "Retorno_Acumulado_pct": True,
            "CAGR_pct": True,
            "Volatilidade_Anualizada_pct": False,
            "Sharpe_Ratio": True,
            "Max_Drawdown_pct": True,
            "Turnover_Medio_Mensal_pct": False,
            "Alpha_vs_Universo_pct": True,
        }
        if model_choice == "Comparar Todos os Modelos":
            sel_mask = is_model_row & df_cmp["Esquema"].str.startswith("Soft" if is_soft else "Hard")
        else:
            sel_mask = df_cmp["Modelo"].str.contains(model_map[model_choice].upper()) & df_cmp["Esquema"].str.startswith(
                "Soft" if is_soft else "Hard"
            )
        sel_bg = ui.rgba(ui.MODEL_COLORS["Ridge"], 0.10)

        def _style_cmp(frame: pd.DataFrame) -> pd.DataFrame:
            styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
            styles.loc[~is_model_row, :] = f"color: {ui.UNIVERSE_COLOR};"
            for col, hib in higher_better.items():
                vals = frame.loc[is_model_row, col]
                best = vals.max() if hib else vals.min()
                styles.loc[is_model_row & (frame[col] == best), col] = f"color: {ui.POS_COLOR}; font-weight: 700;"
            styles.loc[sel_mask, :] = styles.loc[sel_mask, :] + f"background-color: {sel_bg};"
            return styles

        st.dataframe(
            df_cmp.style.apply(_style_cmp, axis=None).format(
                {
                    "Retorno_Acumulado_pct": lambda v: ui.fmt_pct(v, 2),
                    "CAGR_pct": lambda v: ui.fmt_pct(v, 2, sign=False),
                    "Volatilidade_Anualizada_pct": lambda v: ui.fmt_pct(v, 2, sign=False),
                    "Sharpe_Ratio": lambda v: ui.fmt_num(v, 2),
                    "Max_Drawdown_pct": lambda v: ui.fmt_pct(v, 2, sign=False),
                    "Turnover_Medio_Mensal_pct": lambda v: ui.fmt_pct(v, 1, sign=False) if v else "—",
                    "Alpha_vs_Universo_pct": lambda v: ui.fmt_pct(v, 2),
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Esquema": "Política / esquema",
                "Retorno_Acumulado_pct": "Retorno acumulado",
                "CAGR_pct": "CAGR",
                "Volatilidade_Anualizada_pct": "Vol. anual",
                "Sharpe_Ratio": "Sharpe",
                "Max_Drawdown_pct": "Max drawdown",
                "Turnover_Medio_Mensal_pct": "Turnover mensal",
                "Alpha_vs_Universo_pct": "Alpha vs. universo",
            },
        )


# =============================================================
# TAB 2: PREVISÃO VS. REALIDADE
# =============================================================
MODEL_DISP_TO_KEY = {
    "Regressão Ridge": "Ridge",
    "XGBoost": "Xgboost",
    "LightGBM": "Lightgbm",
}
COMPARE_ALL = "Comparar os 3 Modelos Lado a Lado"


def render_holdings_table(
    sub: pd.DataFrame,
    price_start_col: str,
    price_end_col: str,
    ret_col: str,
    contrib_col: str,
    beat_col: str,
    price_start_label: str,
    price_end_label: str,
    beat_label: str = "Bateu B&H Universo?",
) -> None:
    view = sub[["rank", "ticker", "predicted_score", "weight_pct", price_start_col, price_end_col, ret_col, contrib_col, beat_col]].copy()
    view.columns = [
        "Posição",
        "Ticker",
        "Score Previsto (Alpha)",
        "Peso Carteira (%)",
        price_start_label,
        price_end_label,
        "Retorno da Ação (%)",
        "Contribuição (%)",
        beat_label,
    ]
    styler = (
        view.style.format(
            {
                "Score Previsto (Alpha)": lambda v: ui.fmt_num(v, 4, sign=True),
                "Peso Carteira (%)": lambda v: ui.fmt_pct(v, 2, sign=False),
                price_start_label: lambda v: "R$ " + ui.fmt_num(v, 2),
                price_end_label: lambda v: "R$ " + ui.fmt_num(v, 2),
                "Retorno da Ação (%)": lambda v: ui.fmt_pct(v, 2),
                "Contribuição (%)": lambda v: ui.fmt_pct(v, 2),
                beat_label: lambda b: "Sim" if b else "Não",
            }
        )
        .map(ui.color_signed, subset=["Retorno da Ação (%)", "Contribuição (%)"])
        .map(ui.color_bool, subset=[beat_label])
    )
    st.dataframe(styler, hide_index=True, width="stretch")
    total_contrib = float(sub[contrib_col].sum())
    ui.caption(
        f"Soma dos pesos: <span class='qa-num'>{ui.fmt_pct(float(sub['weight_pct'].sum()), 2, sign=False)}</span> · "
        f"soma das contribuições = retorno da carteira: <span class='qa-num'>{ui.fmt_pct(total_contrib, 2)}</span>"
    )


def render_stock_bars(
    sub_plot: pd.DataFrame,
    ret_col: str,
    beat_col: str,
    universe_ret: float,
    ibov_ret: float,
) -> None:
    fig_bar = go.Figure()
    for beat, color, name in [(True, ui.POS_COLOR, "Superou B&H universo"), (False, ui.NEG_COLOR, "Abaixo do B&H universo")]:
        part = sub_plot[sub_plot[beat_col] == beat]
        fig_bar.add_trace(
            go.Bar(
                x=part["ticker"],
                y=part[ret_col],
                name=name,
                marker_color=color,
                text=[ui.fmt_num(v, 1, sign=True) for v in part[ret_col]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}: %{y:+.2f}%<extra>" + name + "</extra>",
            )
        )
    fig_bar.update_xaxes(categoryorder="array", categoryarray=list(sub_plot["ticker"]))
    ui.add_benchmark_lines(fig_bar, universe_ret, ibov_ret)
    ui.style_fig(fig_bar, 380, hovermode="closest")
    fig_bar.update_layout(barmode="overlay", bargap=0.35)
    fig_bar.update_yaxes(title="Retorno da ação (%)")
    st.plotly_chart(fig_bar, width="stretch")


def render_model_comparison(df_comp: pd.DataFrame, universe_ret: float, ibov_ret: float, period_label: str) -> None:
    with ui.panel(
        f"model_cmp_{period_label}",
        f"Retorno realizado por modelo · {period_label}",
        f"B&amp;H universo {ui.fmt_pct(universe_ret)} · Ibovespa {ui.fmt_pct(ibov_ret)}",
        info="model_comparison",
    ):
        fig_comp = go.Figure(
            go.Bar(
                x=df_comp["Modelo"],
                y=df_comp["Retorno_pct"],
                marker_color=[colors[k] for k in df_comp["key"]],
                text=[ui.fmt_pct(v, 2) for v in df_comp["Retorno_pct"]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}: %{y:+.2f}%<extra></extra>",
            )
        )
        ui.add_benchmark_lines(fig_comp, universe_ret, ibov_ret)
        ui.style_fig(fig_comp, 340, legend=False, hovermode="closest")
        fig_comp.update_layout(bargap=0.55)
        fig_comp.update_yaxes(title="Retorno realizado (%)")
        st.plotly_chart(fig_comp, width="stretch")


def sort_toggle(key: str) -> bool:
    """Returns True when bars should be ordered by realized return."""
    choice = st.segmented_control(
        "Ordenação das ações no gráfico",
        [
            "Ordem de Recomendação (Maior Peso / Rank do Modelo)",
            "Melhor Desempenho para o Pior (Maior Retorno %)",
        ],
        default="Ordem de Recomendação (Maior Peso / Rank do Modelo)",
        required=True,
        format_func=lambda s: "Rank do modelo" if s.startswith("Ordem") else "Maior retorno",
        key=key,
        label_visibility="collapsed",
        help="Ordem de recomendação (maior peso / rank do modelo) ou do melhor para o pior desempenho.",
    )
    return "Melhor Desempenho" in choice


with tab_pred_real:
    ui.section_header(
        "Auditoria de decisões",
        "O que o modelo previu vs. o que realmente aconteceu, mês a mês e pregão a pregão.",
    )

    # Reusable Month Audit Renderer
    def render_month_audit_view(
        df_h: pd.DataFrame,
        target_month: str,
        selected_model_disp: str,
        weighting_scheme_choice: str,
        subtitle_prefix: str = "Auditoria de Carteira",
    ):
        scheme_filter = "Soft Weighting (Top 15)" if "Soft" in weighting_scheme_choice else "Hard Cut (Top 10)"

        # Filter for target month and scheme
        df_m = df_h[(df_h["month_label"] == target_month) & (df_h["scheme"] == scheme_filter)].copy()
        if df_m.empty:
            ui.notice(f"Sem dados detalhados encontrados para o mês {target_month} com o esquema {scheme_filter}.", "warn")
            return

        rebal_dt = pd.to_datetime(df_m["rebal_date"].iloc[0]).strftime("%d/%m/%Y")
        end_dt = pd.to_datetime(df_m["date"].iloc[0]).strftime("%d/%m/%Y")
        month_disp = pd.to_datetime(target_month + "-01").strftime("%m/%Y")
        mkt_ret = df_m["market_return_pct"].iloc[0]
        ibov_ret = df_m["ibov_return_pct"].iloc[0] if "ibov_return_pct" in df_m.columns else 0.0

        ui.section_header(
            f"{subtitle_prefix} · <span class='qa-num'>{month_disp}</span>",
            f"Data base da previsão <b class='qa-num'>{rebal_dt}</b> · fechamento/apuração <b class='qa-num'>{end_dt}</b> · "
            f"política {scheme_filter}",
        )

        is_latest_month = target_month == sorted(df_h["month_label"].unique())[-1]
        if is_latest_month:
            ui.notice(
                f"<b>Carteira recomendada para o mês vigente ({month_disp}).</b> Alocação calculada no 1º pregão do mês "
                f"(<b>{rebal_dt}</b>). Abaixo estão as ações recomendadas para compra, o <b>peso sugerido (%)</b> de cada uma, "
                f"o preço de entrada no início do mês e a evolução até <b>{end_dt}</b>."
            )

        if selected_model_disp == COMPARE_ALL:
            comp_data = []
            for disp_name, k_name in MODEL_DISP_TO_KEY.items():
                sub = df_m[df_m["model"] == k_name]
                if not sub.empty:
                    port_ret = float(sub["contrib_return_pct"].sum())
                    win_pct = float((sub["stock_return_pct"] > mkt_ret).mean() * 100.0)
                    comp_data.append({
                        "Modelo": disp_name,
                        "key": k_name,
                        "Retorno_pct": port_ret,
                        "Alpha_pct": port_ret - mkt_ret,
                        "Win_Rate_pct": win_pct,
                        "N_Acoes": len(sub),
                    })
            df_comp = pd.DataFrame(comp_data)

            if not df_comp.empty:
                cards = []
                for _, r in df_comp.iterrows():
                    cards.append({
                        "label": f"Carteira {ui.MODEL_LABELS[r['key']]}",
                        "value": ui.fmt_pct(r["Retorno_pct"]),
                        "tone": ui.tone_of(r["Retorno_pct"]),
                        "foot": "vs. B&amp;H",
                        "delta": ui.fmt_pp(r["Alpha_pct"]),
                        "delta_tone": ui.tone_of(r["Alpha_pct"]),
                        "help": f"{ui.fmt_num(r['Win_Rate_pct'], 0)}% das {r['N_Acoes']} ações superaram o B&H do universo",
                    })
                cards += [
                    {"label": "Buy & Hold universo (1/N)", "value": ui.fmt_pct(mkt_ret), "tone": ui.tone_of(mkt_ret),
                     "foot": f"{len(base_market.tickers)} ações, equiponderado", "help": "Média equiponderada de todas as ações do universo B3"},
                    {"label": "Ibovespa (BOVA11)", "value": ui.fmt_pct(ibov_ret), "tone": ui.tone_of(ibov_ret),
                     "foot": "Índice oficial da B3", "help": "Índice de mercado ponderado oficial da B3"},
                ]
                ui.kpi_cards(cards, split=True)

                render_model_comparison(df_comp, mkt_ret, ibov_ret, month_disp)

                with ui.panel(f"month_tabs_{target_month}", f"Carteiras recomendadas e retornos em {month_disp} por modelo", info="holdings"):
                    t_ridge, t_xgb, t_lgb = st.tabs(["Regressão Ridge", "XGBoost", "LightGBM"])
                    for t_obj, m_key in [(t_ridge, "Ridge"), (t_xgb, "Xgboost"), (t_lgb, "Lightgbm")]:
                        with t_obj:
                            sub_model = df_m[df_m["model"] == m_key].sort_values("rank").copy()
                            render_holdings_table(
                                sub_model, "price_start", "price_end", "stock_return_pct", "contrib_return_pct", "beat_market",
                                "Preço Início (R$)", "Preço Fim/Atual (R$)",
                            )

        else:
            # Single model view
            target_key = MODEL_DISP_TO_KEY.get(selected_model_disp, "Ridge")
            sub_model = df_m[df_m["model"] == target_key].sort_values("rank").copy()

            if sub_model.empty:
                ui.notice(f"Sem dados para o modelo {selected_model_disp} no mês {month_disp}.", "warn")
                return

            port_ret = float(sub_model["contrib_return_pct"].sum())
            alpha_m = port_ret - mkt_ret
            alpha_ibov = port_ret - ibov_ret
            win_count = int(sub_model["beat_market"].sum())
            tot_count = len(sub_model)
            win_pct = (win_count / tot_count * 100.0) if tot_count > 0 else 0.0

            ui.kpi_cards(
                [
                    {"label": "Data base da previsão", "value": rebal_dt, "foot": f"Apuração até {end_dt}"},
                    {"label": "Retorno da carteira", "value": ui.fmt_pct(port_ret), "tone": ui.tone_of(port_ret),
                     "foot": "vs. B&amp;H", "delta": ui.fmt_pp(alpha_m), "delta_tone": ui.tone_of(alpha_m)},
                    {"label": "Buy & Hold universo (1/N)", "value": ui.fmt_pct(mkt_ret), "tone": ui.tone_of(mkt_ret),
                     "foot": f"{len(base_market.tickers)} ações, equiponderado", "help": "Média equiponderada das ações do universo B3"},
                    {"label": "Ibovespa (BOVA11)", "value": ui.fmt_pct(ibov_ret), "tone": ui.tone_of(ibov_ret),
                     "foot": "vs. Ibov", "delta": ui.fmt_pp(alpha_ibov), "delta_tone": ui.tone_of(alpha_ibov)},
                    {"label": "Taxa de sucesso", "value": f"{win_count} / {tot_count}",
                     "foot": f"{ui.fmt_num(win_pct, 0)}% bateram o B&amp;H",
                     "help": "% de ações que superaram o Buy & Hold do Universo"},
                ],
                split=True,
            )

            with ui.panel(
                f"month_bars_{target_month}",
                f"Desempenho individual das ações · {ui.MODEL_LABELS[target_key]} · {month_disp}",
                f"B&amp;H universo {ui.fmt_pct(mkt_ret)} · Ibovespa {ui.fmt_pct(ibov_ret)}",
                info="stock_bars",
                controls=True,
            ) as ctrl:
                with ctrl:
                    by_return = sort_toggle(f"sort_month_{target_month}_{selected_model_disp}")
                sub_model_plot = (
                    sub_model.sort_values("stock_return_pct", ascending=False)
                    if by_return
                    else sub_model.sort_values("rank", ascending=True)
                )
                render_stock_bars(sub_model_plot, "stock_return_pct", "beat_market", mkt_ret, ibov_ret)

            with ui.panel(f"month_table_{target_month}", "Carteira recomendada", "Score previsto, peso, preços e contribuição de cada ação", info="holdings"):
                render_holdings_table(
                    sub_model, "price_start", "price_end", "stock_return_pct", "contrib_return_pct", "beat_market",
                    "Preço Início (R$)", "Preço Fim/Atual (R$)", beat_label="Bateu Mercado?",
                )

    # Reusable Daily Inspection View (Calendar & Exact Trading Days)
    def render_daily_inspection_view(
        df_h: pd.DataFrame,
        selected_model_disp: str,
        weighting_scheme_choice: str,
    ):
        scheme_filter = "Soft Weighting (Top 15)" if "Soft" in weighting_scheme_choice else "Hard Cut (Top 10)"
        curr_scheme = scheme_filter

        market = st.session_state["live_market"] if live_market_active() else get_cached_market()
        bova_series = get_cached_bova()

        trading_dates = sorted([pd.to_datetime(d).date() for d in market.dates])
        trading_dates_set = set(trading_dates)
        min_date = max(pd.to_datetime("2015-01-02").date(), trading_dates[0])
        max_date = trading_dates[-1]

        dias_semana = {
            0: "Segunda-feira",
            1: "Terça-feira",
            2: "Quarta-feira",
            3: "Quinta-feira",
            4: "Sexta-feira",
            5: "Sábado",
            6: "Domingo",
        }

        ui.section_header(
            "Inspeção pontual · como estava a carteira em um dia específico",
            "Navegue até qualquer pregão da B3. O sistema calcula a rentabilidade acumulada de cada ação desde o 1º pregão do mês "
            "(quando a carteira foi recomendada) até o dia selecionado, comparando com o Buy &amp; Hold do universo e o Ibovespa.",
        )

        if "inspect_current_date" not in st.session_state:
            st.session_state["inspect_current_date"] = max_date

        curr_val = st.session_state["inspect_current_date"]
        if curr_val < min_date:
            curr_val = min_date
        elif curr_val > max_date:
            curr_val = max_date
        st.session_state["inspect_current_date"] = curr_val

        # Quick navigation bar
        c_cal, c_prev, c_next, c_today = st.columns([2.5, 1.2, 1.2, 1.2], vertical_alignment="bottom")

        # Current position in trading dates
        curr_eff = curr_val
        if curr_eff not in trading_dates_set:
            prior_days = [d for d in trading_dates if d <= curr_eff]
            curr_eff = prior_days[-1] if prior_days else trading_dates[0]
        curr_idx = trading_dates.index(curr_eff) if curr_eff in trading_dates_set else len(trading_dates) - 1

        with c_prev:
            if st.button("Pregão anterior", icon=":material/chevron_left:", width="stretch", disabled=(curr_idx <= 0)):
                st.session_state["inspect_current_date"] = trading_dates[max(0, curr_idx - 1)]
                st.rerun()

        with c_next:
            if st.button("Próximo pregão", icon=":material/chevron_right:", icon_position="right", width="stretch",
                         disabled=(curr_idx >= len(trading_dates) - 1)):
                st.session_state["inspect_current_date"] = trading_dates[min(len(trading_dates) - 1, curr_idx + 1)]
                st.rerun()

        with c_today:
            if st.button("Mais recente", icon=":material/last_page:", width="stretch"):
                st.session_state["inspect_current_date"] = max_date
                st.rerun()

        with c_cal:
            picked = st.date_input(
                "Selecione o dia no calendário",
                value=st.session_state["inspect_current_date"],
                min_value=min_date,
                max_value=max_date,
                format="DD/MM/YYYY",
                help="Clique para abrir o calendário mensal/anual. O sistema valida pregões da B3 automaticamente.",
            )
            if picked != st.session_state["inspect_current_date"]:
                st.session_state["inspect_current_date"] = picked
                st.rerun()

        # Validate trading day
        if picked in trading_dates_set:
            eff_date = picked
            ui.notice(f"<b>Pregão válido confirmado:</b> {eff_date.strftime('%d/%m/%Y')} ({dias_semana[eff_date.weekday()]})", "ok")
        else:
            prior_days = [d for d in trading_dates if d <= picked]
            eff_date = prior_days[-1] if prior_days else trading_dates[0]
            orig_day_name = dias_semana[picked.weekday()]
            eff_day_name = dias_semana[eff_date.weekday()]
            ui.notice(
                f"<b>{picked.strftime('%d/%m/%Y')} ({orig_day_name})</b> não foi dia de pregão na B3 (fim de semana ou feriado). "
                f"Ajustado automaticamente para o pregão imediatamente anterior: <b>{eff_date.strftime('%d/%m/%Y')} ({eff_day_name})</b>.",
                "warn",
            )

        year = eff_date.year
        month = eff_date.month
        month_label = f"{year}-{month:02d}"

        # Filter holdings for this month and scheme
        df_m = df_h[(df_h["month_label"] == month_label) & (df_h["scheme"] == curr_scheme)].copy()
        if df_m.empty:
            first_m = pd.to_datetime(df_h["month_label"].min() + "-01")
            last_m = pd.to_datetime(df_h["month_label"].max() + "-01")
            ui.notice(
                f"Sem dados de recomendações do modelo para o mês {month_label}. "
                f"As previsões estão disponíveis de {first_m:%m/%Y} até {last_m:%m/%Y}.",
                "err",
            )
            return

        # Find trading days in month up to eff_date
        m_dates = [d for d in market.dates if d.year == year and d.month == month and pd.to_datetime(d).date() <= eff_date]
        t_0 = m_dates[0]
        t_curr = [d for d in m_dates if pd.to_datetime(d).date() == eff_date][0]

        idx_0 = list(market.dates).index(t_0)
        idx_t = list(market.dates).index(t_curr)
        days_elapsed = len(m_dates) - 1

        t0_str = t_0.strftime("%d/%m/%Y")
        tt_str = eff_date.strftime("%d/%m/%Y")

        # Universe return from t_0 to eff_date
        p0_all = market.close.iloc[idx_0]
        pt_all = market.close.iloc[idx_t]
        mask_univ = (p0_all > 0) & (pt_all > 0) & ~p0_all.isna() & ~pt_all.isna()
        univ_ret = float(((pt_all / p0_all - 1.0) * 100.0)[mask_univ].mean()) if mask_univ.any() else 0.0

        # Ibovespa return from t_0 to eff_date
        if bova_series is not None:
            p0_bova = bova_series.asof(t_0)
            pt_bova = bova_series.asof(t_curr)
            ibov_ret = float((pt_bova / p0_bova - 1.0) * 100.0) if (pd.notna(p0_bova) and pd.notna(pt_bova) and p0_bova > 0) else 0.0
        else:
            ibov_ret = 0.0

        ui.section_header(
            f"Posição da carteira no pregão de <span class='qa-num'>{tt_str}</span>",
            f"Data base da previsão <b class='qa-num'>{t0_str}</b> · <b class='qa-num'>{days_elapsed}</b> pregões decorridos · "
            f"política {curr_scheme}",
        )

        if days_elapsed == 0:
            ui.notice(f"<b>1º pregão do mês selecionado ({t0_str}):</b> esta é a data exata da emissão das recomendações do modelo. O retorno acumulado no primeiro dia é 0,00%.")

        def calc_holdings_for_day(m_key: str):
            sub = df_m[df_m["model"] == m_key].sort_values("rank").copy()
            if sub.empty:
                return sub, 0.0
            sub["price_start_day"] = 0.0
            sub["price_now_day"] = 0.0
            sub["stock_return_day_pct"] = 0.0
            sub["contrib_return_day_pct"] = 0.0

            for idx_row, row in sub.iterrows():
                tk = row["ticker"]
                if tk in market.close.columns:
                    p_s = float(market.close.iloc[idx_0][tk])
                    p_n = float(market.close.iloc[idx_t][tk])
                    r = (p_n / p_s - 1.0) * 100.0 if (p_s > 0 and not np.isnan(p_s) and not np.isnan(p_n)) else 0.0
                    w = float(row["weight_pct"])
                    c = r * (w / 100.0)
                else:
                    p_s = float(row.get("price_start", 0.0))
                    p_n = float(row.get("price_end", 0.0))
                    r = float(row.get("stock_return_pct", 0.0))
                    c = float(row.get("contrib_return_pct", 0.0))
                sub.at[idx_row, "price_start_day"] = p_s
                sub.at[idx_row, "price_now_day"] = p_n
                sub.at[idx_row, "stock_return_day_pct"] = r
                sub.at[idx_row, "contrib_return_day_pct"] = c

            sub["beat_market_day"] = sub["stock_return_day_pct"] > univ_ret
            p_ret = float(sub["contrib_return_day_pct"].sum())
            return sub, p_ret

        price_start_label = f"Preço Entrada {t0_str} (R$)"
        price_now_label = f"Preço em {tt_str} (R$)"

        if selected_model_disp == COMPARE_ALL:
            comp_rows = []
            model_dfs = {}
            for disp_name, k_name in MODEL_DISP_TO_KEY.items():
                sub_m, p_ret = calc_holdings_for_day(k_name)
                model_dfs[k_name] = sub_m
                win_pct = float((sub_m["stock_return_day_pct"] > univ_ret).mean() * 100.0) if not sub_m.empty else 0.0
                comp_rows.append({
                    "Modelo": disp_name,
                    "key": k_name,
                    "Retorno_pct": p_ret,
                    "Alpha_univ_pct": p_ret - univ_ret,
                    "Alpha_ibov_pct": p_ret - ibov_ret,
                    "Win_Rate_pct": win_pct,
                    "N_Acoes": len(sub_m),
                })
            df_comp_day = pd.DataFrame(comp_rows)

            cards = []
            for _, r in df_comp_day.iterrows():
                cards.append({
                    "label": f"{ui.MODEL_LABELS[r['key']]} (até este dia)",
                    "value": ui.fmt_pct(r["Retorno_pct"]),
                    "tone": ui.tone_of(r["Retorno_pct"]),
                    "foot": "vs. B&amp;H",
                    "delta": ui.fmt_pp(r["Alpha_univ_pct"]),
                    "delta_tone": ui.tone_of(r["Alpha_univ_pct"]),
                    "help": f"{ui.fmt_num(r['Win_Rate_pct'], 0)}% das {r['N_Acoes']} ações superaram o B&H do universo",
                })
            cards += [
                {"label": "Buy & Hold universo (1/N)", "value": ui.fmt_pct(univ_ret), "tone": ui.tone_of(univ_ret),
                 "foot": "Até esta data", "help": "Média equiponderada das ações do universo B3 até esta data"},
                {"label": "Ibovespa (BOVA11)", "value": ui.fmt_pct(ibov_ret), "tone": ui.tone_of(ibov_ret),
                 "foot": "Até esta data", "help": "Índice ponderado oficial da B3 até esta data"},
            ]
            ui.kpi_cards(cards, split=True)

            render_model_comparison(df_comp_day, univ_ret, ibov_ret, f"até {tt_str}")

            with ui.panel(f"day_tabs", f"Carteiras recomendadas e posição em {tt_str}", info="holdings"):
                t_ridge, t_xgb, t_lgb = st.tabs(["Regressão Ridge", "XGBoost", "LightGBM"])
                for t_obj, m_key in [(t_ridge, "Ridge"), (t_xgb, "Xgboost"), (t_lgb, "Lightgbm")]:
                    with t_obj:
                        render_holdings_table(
                            model_dfs[m_key], "price_start_day", "price_now_day", "stock_return_day_pct",
                            "contrib_return_day_pct", "beat_market_day", price_start_label, price_now_label,
                        )

        else:
            # Single model view
            target_key = MODEL_DISP_TO_KEY.get(selected_model_disp, "Ridge")
            sub_model, port_ret = calc_holdings_for_day(target_key)

            if sub_model.empty:
                ui.notice(f"Sem dados para o modelo {selected_model_disp} na data {tt_str}.", "warn")
                return

            alpha_univ = port_ret - univ_ret
            alpha_ibov = port_ret - ibov_ret
            win_count = int(sub_model["beat_market_day"].sum())
            tot_count = len(sub_model)
            win_pct = (win_count / tot_count * 100.0) if tot_count > 0 else 0.0

            ui.kpi_cards(
                [
                    {"label": "Data da recomendação", "value": t0_str, "foot": f"{days_elapsed} pregões decorridos"},
                    {"label": "Retorno carteira até a data", "value": ui.fmt_pct(port_ret), "tone": ui.tone_of(port_ret),
                     "foot": "vs. B&amp;H", "delta": ui.fmt_pp(alpha_univ), "delta_tone": ui.tone_of(alpha_univ)},
                    {"label": "Buy & Hold universo (1/N)", "value": ui.fmt_pct(univ_ret), "tone": ui.tone_of(univ_ret),
                     "foot": "Até esta data", "help": "Média equiponderada das ações do universo B3 até esta data"},
                    {"label": "Ibovespa (BOVA11)", "value": ui.fmt_pct(ibov_ret), "tone": ui.tone_of(ibov_ret),
                     "foot": "vs. Ibov", "delta": ui.fmt_pp(alpha_ibov), "delta_tone": ui.tone_of(alpha_ibov)},
                    {"label": "Taxa de sucesso no dia", "value": f"{win_count} / {tot_count}",
                     "foot": f"{ui.fmt_num(win_pct, 0)}% bateram o B&amp;H",
                     "help": "% de ações que superaram o Buy & Hold do Universo até o pregão selecionado"},
                ],
                split=True,
            )

            with ui.panel(
                "day_bars",
                f"Desempenho individual das ações · {ui.MODEL_LABELS[target_key]} · até {tt_str}",
                f"B&amp;H universo {ui.fmt_pct(univ_ret)} · Ibovespa {ui.fmt_pct(ibov_ret)}",
                info="stock_bars",
                controls=True,
            ) as ctrl:
                with ctrl:
                    by_return = sort_toggle(f"sort_day_{tt_str}_{selected_model_disp}")
                sub_model_plot = (
                    sub_model.sort_values("stock_return_day_pct", ascending=False)
                    if by_return
                    else sub_model.sort_values("rank", ascending=True)
                )
                render_stock_bars(sub_model_plot, "stock_return_day_pct", "beat_market_day", univ_ret, ibov_ret)

            with ui.panel("day_table", "Carteira recomendada", f"Posição de cada ação em {tt_str}", info="holdings"):
                render_holdings_table(
                    sub_model, "price_start_day", "price_now_day", "stock_return_day_pct", "contrib_return_day_pct",
                    "beat_market_day", price_start_label, price_now_label,
                )

    # Online API Status banner if live market is active
    if live_market_active():
        lm = st.session_state["live_market"]
        ui.notice(f"<b>Modo online ativo:</b> cotações sincronizadas via API Yahoo Finance ({len(lm.dates)} pregões até {lm.dates[-1].strftime('%d/%m/%Y')}).", "ok")

    # Dual controls: Window Type & Model Selector
    c_ctrl1, c_ctrl2 = st.columns([1.3, 2.0])
    with c_ctrl1:
        view_type = st.segmented_control(
            "Janela de inspeção",
            [
                "Visão Compacta por Ano (Modelo vs. Mercado e Previsões)",
                "Inspecionar por Data Específica (Calendário de Pregões Diários)",
                f"Auditoria Mês a Mês (Histórico Completo {wf_years})",
            ],
            default="Visão Compacta por Ano (Modelo vs. Mercado e Previsões)",
            required=True,
            format_func=lambda v: "Visão anual" if "Compacta" in v else ("Por pregão" if "Data Específica" in v else "Mês a mês"),
            help="Visão anual: modelo vs. mercado e previsões no ano · Por pregão: calendário de pregões diários · "
            f"Mês a mês: histórico completo {wf_years}.",
        )
    with c_ctrl2:
        tab2_model_choice = st.segmented_control(
            "Modelo sob análise",
            ["Regressão Ridge", "XGBoost", "LightGBM", COMPARE_ALL],
            default="Regressão Ridge",
            required=True,
            format_func=lambda m: "Comparar os 3" if m == COMPARE_ALL else m,
            help="Alterne entre os modelos para inspecionar exatamente a carteira e o retorno de cada um.",
        )

    all_months = sorted(df_holdings["month_label"].unique(), reverse=True)
    available_years = sorted(df_monthly["month_label"].str[:4].unique(), reverse=True)

    curr_scheme = "Soft Weighting (Top 15)" if is_soft else "Hard Cut (Top 10)"

    if "Visão Compacta" in view_type:
        c_yr_sel, _ = st.columns([1, 3])
        with c_yr_sel:
            selected_compact_year = st.selectbox(
                "Ano",
                available_years,
                index=0,
                help="Selecione o ano desejado para auditar o desempenho consolidado e as previsões daquele período.",
            )

        df_m_yr = df_monthly[df_monthly["month_label"].str.startswith(selected_compact_year)].sort_values("month_label").reset_index(drop=True)
        is_current_year = selected_compact_year == available_years[0]
        yr_last_date = pd.to_datetime(df_m_yr["date"]).max()
        year_status = (
            f"Year-to-date: 01/01/{selected_compact_year} a {yr_last_date:%d/%m/%Y}"
            if is_current_year
            else f"Ano fechado: 01/01/{selected_compact_year} a 31/12/{selected_compact_year}"
        )
        ui.section_header(
            f"Desempenho consolidado no ano de {selected_compact_year}",
            f"{year_status} · quanto cada modelo previu e entregou vs. o mercado (Buy &amp; Hold 1/N e Ibovespa).",
        )

        # Cumulative compound series for each
        cum_univ_ytd = (np.cumprod(1.0 + df_m_yr["Universe"]) - 1.0) * 100.0
        cum_ibov_ytd = (np.cumprod(1.0 + df_m_yr["Ibovespa"]) - 1.0) * 100.0

        r_univ_ytd = float(cum_univ_ytd.iloc[-1])
        r_ibov_ytd = float(cum_ibov_ytd.iloc[-1])

        # Calculate for each model
        cum_models_ytd = {}
        for m_disp, m_key in MODEL_DISP_TO_KEY.items():
            col = f"{m_key}_{scheme_suffix}"
            if col in df_m_yr.columns:
                cum_models_ytd[m_key] = (np.cumprod(1.0 + df_m_yr[col]) - 1.0) * 100.0

        bench_cards = [
            {"label": "Buy & Hold universo (1/N)", "value": ui.fmt_pct(r_univ_ytd), "tone": ui.tone_of(r_univ_ytd),
             "foot": selected_compact_year, "help": f"Média equiponderada de todas as ações em {selected_compact_year}"},
            {"label": "Ibovespa (BOVA11)", "value": ui.fmt_pct(r_ibov_ytd), "tone": ui.tone_of(r_ibov_ytd),
             "foot": selected_compact_year, "help": f"Índice oficial ponderado da B3 em {selected_compact_year}"},
        ]

        if tab2_model_choice == COMPARE_ALL:
            cards = []
            for m_key in ["Ridge", "Xgboost", "Lightgbm"]:
                r_ytd = float(cum_models_ytd[m_key].iloc[-1]) if m_key in cum_models_ytd else 0.0
                cards.append({
                    "label": f"{ui.MODEL_LABELS[m_key]} ({selected_compact_year})",
                    "value": ui.fmt_pct(r_ytd),
                    "tone": ui.tone_of(r_ytd),
                    "foot": "vs. B&amp;H",
                    "delta": ui.fmt_pp(r_ytd - r_univ_ytd),
                    "delta_tone": ui.tone_of(r_ytd - r_univ_ytd),
                })
            ui.kpi_cards(cards + bench_cards, split=True)
            ytd_keys = ["Ridge", "Xgboost", "Lightgbm"]
            ytd_title = f"Evolução patrimonial acumulada em {selected_compact_year}"
        else:
            target_key = MODEL_DISP_TO_KEY.get(tab2_model_choice, "Ridge")
            r_mod_ytd = float(cum_models_ytd[target_key].iloc[-1]) if target_key in cum_models_ytd else 0.0
            alpha_univ = r_mod_ytd - r_univ_ytd
            alpha_ibov = r_mod_ytd - r_ibov_ytd
            ui.kpi_cards(
                [
                    {"label": f"{tab2_model_choice} ({selected_compact_year})", "value": ui.fmt_pct(r_mod_ytd),
                     "tone": ui.tone_of(r_mod_ytd), "foot": "vs. B&amp;H", "delta": ui.fmt_pp(alpha_univ),
                     "delta_tone": ui.tone_of(alpha_univ)},
                    *bench_cards,
                    {"label": "Alpha vs. universo", "value": ui.fmt_pp(alpha_univ), "tone": ui.tone_of(alpha_univ),
                     "foot": "Modelo − B&amp;H 1/N"},
                    {"label": "Alpha vs. Ibovespa", "value": ui.fmt_pp(alpha_ibov), "tone": ui.tone_of(alpha_ibov),
                     "foot": "Modelo − Ibovespa"},
                ],
                split=True,
            )
            ytd_keys = [target_key]
            ytd_title = f"Evolução patrimonial acumulada em {selected_compact_year}: {tab2_model_choice} vs. mercado"

        # Chart 1: Cumulative Curve
        with ui.panel(f"ytd_curve", ytd_title, "Retorno composto mês a mês dentro do ano (%)", info="ytd_curve"):
            x_ytd = pd.to_datetime(df_m_yr["date"])
            fig_ytd = go.Figure()
            fig_ytd.add_trace(go.Scatter(
                x=x_ytd, y=cum_univ_ytd, mode="lines+markers", name="Buy & Hold Universo (1/N)",
                line=dict(color=ui.UNIVERSE_COLOR, width=1.8, dash="dash"),
                hovertemplate="B&H universo: %{y:+.2f}%<extra></extra>",
            ))
            fig_ytd.add_trace(go.Scatter(
                x=x_ytd, y=cum_ibov_ytd, mode="lines+markers", name="Ibovespa (BOVA11)",
                line=dict(color=ui.IBOV_COLOR, width=2, dash="dot"),
                hovertemplate="Ibovespa: %{y:+.2f}%<extra></extra>",
            ))
            for m_key in ytd_keys:
                if m_key in cum_models_ytd:
                    single = len(ytd_keys) == 1
                    fig_ytd.add_trace(go.Scatter(
                        x=x_ytd, y=cum_models_ytd[m_key], mode="lines+markers", name=f"ML: {ui.MODEL_LABELS[m_key]}",
                        line=dict(color=colors[m_key], width=3),
                        fill="tonexty" if single else None,
                        fillcolor=ui.rgba(colors[m_key], 0.08) if single else None,
                        hovertemplate=f"{ui.MODEL_LABELS[m_key]}: %{{y:+.2f}}%<extra></extra>",
                    ))
            ui.style_fig(fig_ytd, 360)
            fig_ytd.update_xaxes(hoverformat="%d/%m/%Y")
            fig_ytd.update_yaxes(title=f"Retorno acumulado em {selected_compact_year} (%)")
            st.plotly_chart(fig_ytd, width="stretch")

        # Monthly Prediction vs Reality Audit Table & Bar Chart
        target_m_for_table = "Ridge" if tab2_model_choice == COMPARE_ALL else MODEL_DISP_TO_KEY.get(tab2_model_choice, "Ridge")
        sub_h = df_holdings[
            (df_holdings["month_label"].str.startswith(selected_compact_year))
            & (df_holdings["model"] == target_m_for_table)
            & (df_holdings["scheme"] == curr_scheme)
        ].copy()

        ytd_records = []
        for m_lbl, grp in sub_h.groupby("month_label"):
            p_ret = float(grp["contrib_return_pct"].sum())
            u_ret = float(grp["market_return_pct"].iloc[0])
            i_ret = float(grp["ibov_return_pct"].iloc[0]) if "ibov_return_pct" in grp.columns else 0.0
            score_m = float(grp["predicted_score"].mean())
            best_r = grp.sort_values("rank").iloc[0]
            m_br = pd.to_datetime(m_lbl + "-01").strftime("%m/%Y")
            ytd_records.append({
                "Mês": m_br,
                "Score Alpha Previsto": score_m * 100,
                "Retorno Modelo (%)": p_ret,
                "Buy & Hold Universo (%)": u_ret,
                "Ibovespa (%)": i_ret,
                "Alpha vs. B&H (%)": p_ret - u_ret,
                "Bateu B&H?": p_ret > u_ret,
                "Bateu Ibov?": p_ret > i_ret,
                "Top Ação Escolhida": f"{best_r['ticker']} ({ui.fmt_pct(best_r['stock_return_pct'], 1)})",
            })

        df_ytd_table = pd.DataFrame(ytd_records)
        model_lbl = ui.MODEL_LABELS[target_m_for_table]

        # Grouped Bar Chart: Modelo vs B&H vs Ibov
        with ui.panel(
            "ytd_bars",
            f"Retornos mês a mês em {selected_compact_year}: {model_lbl} vs. benchmarks de mercado",
            "Retorno mensal simples de cada série (%)",
            info="ytd_bars",
        ):
            fig_bar_ytd = go.Figure()
            for col_name, name, color in [
                ("Retorno Modelo (%)", f"{model_lbl} (retorno entregue)", colors[target_m_for_table]),
                ("Buy & Hold Universo (%)", "Buy & Hold Universo (1/N)", ui.UNIVERSE_COLOR),
                ("Ibovespa (%)", "Ibovespa (BOVA11)", ui.IBOV_COLOR),
            ]:
                fig_bar_ytd.add_trace(go.Bar(
                    x=df_ytd_table["Mês"], y=df_ytd_table[col_name], name=name, marker_color=color,
                    text=[ui.fmt_num(v, 1, sign=True) for v in df_ytd_table[col_name]], textposition="auto",
                    hovertemplate=f"{name}: %{{y:+.2f}}%<extra></extra>",
                ))
            ui.style_fig(fig_bar_ytd, 360)
            fig_bar_ytd.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
            fig_bar_ytd.update_yaxes(title="Retorno mensal (%)")
            st.plotly_chart(fig_bar_ytd, width="stretch")

        with ui.panel(
            "ytd_table",
            f"O que o modelo previu vs. o que o mercado rendeu mês a mês em {selected_compact_year}",
            f"Modelo de referência: {model_lbl} · {curr_scheme}",
            info="ytd_table",
        ):
            if df_ytd_table.empty:
                ui.notice("Sem carteiras registradas para este ano.", "warn")
            else:
                st.dataframe(
                    df_ytd_table.style.format({
                        "Score Alpha Previsto": lambda v: ui.fmt_pct(v, 2),
                        "Retorno Modelo (%)": lambda v: ui.fmt_pct(v, 2),
                        "Buy & Hold Universo (%)": lambda v: ui.fmt_pct(v, 2),
                        "Ibovespa (%)": lambda v: ui.fmt_pct(v, 2),
                        "Alpha vs. B&H (%)": lambda v: ui.fmt_pct(v, 2),
                        "Bateu B&H?": lambda b: "Sim" if b else "Não",
                        "Bateu Ibov?": lambda b: "Sim" if b else "Não",
                    })
                    .map(ui.color_signed, subset=["Retorno Modelo (%)", "Buy & Hold Universo (%)", "Ibovespa (%)", "Alpha vs. B&H (%)"])
                    .map(ui.color_bool, subset=["Bateu B&H?", "Bateu Ibov?"]),
                    hide_index=True,
                    width="stretch",
                )

    elif "Data Específica" in view_type:
        render_daily_inspection_view(
            df_h=df_holdings,
            selected_model_disp=tab2_model_choice,
            weighting_scheme_choice=weighting_choice,
        )

    else:
        # Auditoria Mês a Mês com navegação rápida
        if "selected_audit_month" not in st.session_state:
            st.session_state["selected_audit_month"] = all_months[0]

        curr_m = st.session_state["selected_audit_month"]
        if curr_m not in all_months:
            curr_m = all_months[0]
            st.session_state["selected_audit_month"] = curr_m
        curr_m_idx = all_months.index(curr_m)

        c_sel_m, c_prev_m, c_next_m, c_today_m = st.columns([2.5, 1.2, 1.2, 1.2], vertical_alignment="bottom")

        with c_prev_m:
            # In descending list, chronologically previous month is at index + 1
            if st.button("Mês anterior", icon=":material/chevron_left:", width="stretch", disabled=(curr_m_idx >= len(all_months) - 1)):
                st.session_state["selected_audit_month"] = all_months[curr_m_idx + 1]
                st.rerun()

        with c_next_m:
            # Chronologically next month is at index - 1
            if st.button("Próximo mês", icon=":material/chevron_right:", icon_position="right", width="stretch", disabled=(curr_m_idx <= 0)):
                st.session_state["selected_audit_month"] = all_months[curr_m_idx - 1]
                st.rerun()

        with c_today_m:
            if st.button("Mais recente", icon=":material/last_page:", width="stretch"):
                st.session_state["selected_audit_month"] = all_months[0]
                st.rerun()

        with c_sel_m:
            picked_m = st.selectbox(
                f"Mês para auditar carteiras e previsões ({wf_years})",
                all_months,
                index=curr_m_idx,
                format_func=lambda m: pd.to_datetime(m + "-01").strftime("%m/%Y") + (" · mês mais recente / em andamento" if m == all_months[0] else ""),
            )
            if picked_m != st.session_state["selected_audit_month"]:
                st.session_state["selected_audit_month"] = picked_m
                st.rerun()

        render_month_audit_view(
            df_h=df_holdings,
            target_month=st.session_state["selected_audit_month"],
            selected_model_disp=tab2_model_choice,
            weighting_scheme_choice=weighting_choice,
            subtitle_prefix="Auditoria de carteira mensal",
        )


# =============================================================
# TAB 3: EXPLICABILIDADE & PESO DAS FEATURES
# =============================================================
with tab_explain:
    ui.section_header(
        "Interpretação econômica e importância dos fatores",
        "Como cada modelo pondera os 12 indicadores técnicos para prever o excesso de retorno de 21 dias.",
    )

    if not df_importance.empty:
        df_imp = df_importance.copy()
        feature_name_map = {
            "vol_21d": "Volatilidade Realizada (21 dias)",
            "bb_pos": "Posição nas Bandas de Bollinger",
            "ret_63d": "Momentum Trimestral (Retorno 63d)",
            "ret_21d": "Retorno Mensal (21 dias)",
            "ret_10d": "Retorno Quinzenal (10 dias)",
            "ret_5d": "Retorno Semanal (5 dias)",
            "ret_1d": "Retorno Diário (1 dia)",
            "rsi_14": "RSI - Força Relativa (14 dias)",
            "sma_ratio": "Inclinação entre Médias (SMA 20/50)",
            "sma50_dist": "Distância da Média Móvel de 50 dias",
            "sma20_dist": "Distância da Média Móvel de 20 dias",
            "vol_ratio": "Aceleração de Volume (5d vs. 21d)",
        }
        df_imp["Nome_Feature"] = df_imp["Feature"].map(feature_name_map).fillna(df_imp["Feature"])
        df_imp["Ridge_Beta_bps"] = (df_imp["Ridge_Beta"] * 10000.0).round(1)
        df_imp["XGBoost_Gain_pct"] = (df_imp["XGBoost_Gain"] * 100.0).round(1)
        df_imp["LightGBM_Gain_pct"] = (df_imp["LightGBM_Gain"] * 100.0).round(1)

        with ui.panel("importance", "Importância dos atributos por modelo", "Selecione o modelo à direita", info="importance", controls=True) as ctrl:
            with ctrl:
                imp_model_choice = st.segmented_control(
                    "Modelo",
                    [
                        "Regressão Ridge (Beta Padronizado em Basis Points)",
                        "XGBoost (Ganho de Informação %)",
                        "LightGBM (Ganho de Informação %)",
                    ],
                    default="Regressão Ridge (Beta Padronizado em Basis Points)",
                    required=True,
                    format_func=lambda m: m.split(" (")[0].replace("Regressão ", ""),
                    label_visibility="collapsed",
                )

            if "Ridge" in imp_model_choice:
                df_sorted = df_imp.sort_values("Ridge_Beta_bps", ascending=True)
                x_col, x_title = "Ridge_Beta_bps", "Coeficiente beta padronizado (bps)"
                bar_colors = [ui.POS_COLOR if v > 0 else ui.NEG_COLOR for v in df_sorted[x_col]]
                chart_sub = "Pesos dos fatores na Regressão Ridge (em basis points) · verde favorece a ação, vermelho penaliza"
                suffix = ""
            elif "XGBoost" in imp_model_choice:
                df_sorted = df_imp.sort_values("XGBoost_Gain_pct", ascending=True)
                x_col, x_title = "XGBoost_Gain_pct", "Ganho de informação (%)"
                bar_colors = colors["Xgboost"]
                chart_sub = "Importância dos fatores no XGBoost (% de ganho das árvores)"
                suffix = "%"
            else:
                df_sorted = df_imp.sort_values("LightGBM_Gain_pct", ascending=True)
                x_col, x_title = "LightGBM_Gain_pct", "Ganho de informação (%)"
                bar_colors = colors["Lightgbm"]
                chart_sub = "Importância dos fatores no LightGBM (% de ganho por folha)"
                suffix = "%"

            ui.caption(chart_sub)
            fig_imp = go.Figure(
                go.Bar(
                    x=df_sorted[x_col],
                    y=df_sorted["Nome_Feature"],
                    orientation="h",
                    marker_color=bar_colors,
                    text=[ui.fmt_num(v, 1, sign=(suffix == "")) + suffix for v in df_sorted[x_col]],
                    textposition="outside",
                    cliponaxis=False,
                    customdata=df_sorted["Feature"],
                    hovertemplate="%{y} (%{customdata}): %{x:.1f}" + suffix + "<extra></extra>",
                )
            )
            ui.style_fig(fig_imp, 460, y_suffix=None, legend=False, hovermode="closest")
            fig_imp.update_xaxes(title=x_title, ticksuffix=suffix, zeroline=True)
            st.plotly_chart(fig_imp, width="stretch")

    with ui.panel("economics", "Por que os pesos fazem sentido econômico?"):
        xgb_vol_gain = df_imp.loc[df_imp["Feature"] == "vol_21d", "XGBoost_Gain"].iloc[0] * 100 if not df_importance.empty else 0.0
        ui.html(
            f"""
<div class="qa-howto">
<span>1</span><div><b><code>ret_63d</code> (Momentum trimestral):</b> ações com tendência sólida e revisões positivas nos últimos 3 meses continuam apresentando fluxo comprador sustentado.</div>
<span>2</span><div><b><code>ret_1d</code> e <code>bb_pos</code> (Filtro de sobrecompra / reversão de curto prazo):</b> coeficientes negativos penalizam ações excessivamente esticadas nas Bandas de Bollinger ou que tiveram repiques parabólicos de 1 dia, evitando compras no topo.</div>
<span>3</span><div><b><code>vol_21d</code> (Volatilidade realizada):</b> principal divisor de águas nos modelos de árvore ({ui.fmt_num(xgb_vol_gain, 1)}% de ganho no XGBoost), calibrando o risco de cauda e o prêmio exigido pelo mercado.</div>
</div>"""
        )


# =============================================================
# TAB 4: AUDITORIA DE QUALIDADE DE DADOS
# =============================================================
with tab_audit:
    ui.section_header(
        "Data quality assurance e validação de integridade",
        "Auditoria formal da base de dados: ausência de valores nulos, infinitos, preços não positivos ou problemas de sincronia temporal.",
    )

    with ui.panel("quality", "Auditoria das matrizes de mercado", "Executa as verificações sobre a base local carregada", info="quality"):
        if st.button("Executar auditoria de dados agora", type="primary", icon=":material/fact_check:"):
            with st.spinner("Auditando matrizes de mercado..."):
                mkt = get_cached_market()
                report = validate_market_data(mkt)

                if report.is_valid:
                    ui.notice("<b>Base de dados aprovada com 100% de integridade.</b>", "ok")
                else:
                    ui.notice("<b>Problemas detectados na auditoria.</b>", "err")

                ui.kpi_cards(
                    [
                        {"label": "Status", "value": "Aprovado" if report.is_valid else "Falha",
                         "tone": "pos" if report.is_valid else "neg"},
                        {"label": "Tickers", "value": str(report.n_tickers)},
                        {"label": "Pregões", "value": f"{report.n_dates:,}".replace(",", "."),
                         "foot": f"{report.date_range[0]} a {report.date_range[1]}"},
                        {"label": "Cobertura média", "value": ui.fmt_pct(report.active_coverage_ratio_pct, 2, sign=False),
                         "foot": "Ações ativas por pregão"},
                    ],
                    split=True,
                )
                checks = [
                    ("Cronologia estrita (monotônica crescente)", report.is_strictly_chronological),
                    ("Sem datas duplicadas", not report.has_duplicate_dates),
                    ("Sem valores infinitos (inf / −inf)", not report.has_infinite_values),
                    ("Sem preços negativos ou zero", not report.has_negative_prices),
                ]
                ui.stats_table(
                    ["Verificação", "Resultado"],
                    [[name, ui.badge(ok, "OK", "Falhou")] for name, ok in checks],
                )
                with st.expander("Relatório completo (texto)"):
                    st.code(report.summary(), language="text")

    soft_ridge = df_metrics[df_metrics["Modelo"].str.contains("RIDGE") & df_metrics["Esquema"].str.startswith("Soft")]
    with ui.panel("pillars", "Pilares metodológicos do projeto"):
        st.markdown(
            r"""
* **Imunidade a data leakage:** purga e embargo temporal estrito de 21 dias nas fronteiras de treino, validação e teste.
* **Z-score cross-sectional:** padronização diária entre ações ativas ($Z = (X - \mu_t)/\sigma_t$), eliminando o viés de euforia ou pânico macroeconômico.
"""
            + (
                f"* **Soft weighting:** ponderação linear proporcional sobre o Top 15, eliminando a sensibilidade a empates e levando a Regressão Ridge a "
                f"**{ui.fmt_pct(soft_ridge.iloc[0]['Retorno_Acumulado_pct'], 2)} com Sharpe {ui.fmt_num(soft_ridge.iloc[0]['Sharpe_Ratio'], 2)}**."
                if not soft_ridge.empty
                else ""
            )
        )
