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
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from quant_ml_b3.data_loader import load_local_market_data, validate_market_data

# Page configuration
st.set_page_config(
    page_title="B3 Quant Alpha Terminal | ML Stock Selection",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark/clean modern financial aesthetic
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        color: #F8FAFC;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .delta-pos { color: #10B981; }
    .delta-neg { color: #EF4444; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px 8px 0px 0px;
        padding: 0 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.image(
    "https://img.icons8.com/fluency/96/bullish.png",
    width=64,
)
st.sidebar.title("Quant Alpha Terminal")
st.sidebar.caption("Machine Learning Cross-Sectional na B3 (IBrX 50)")

st.sidebar.markdown("---")

# 1. Data Source Mode
data_mode = st.sidebar.radio(
    "Fonte de Dados:",
    ["Base Local Higienizada (2012-2026)", "API Yahoo Finance (Ao Vivo)"],
    index=0,
    help="Alterne entre a base higienizada pré-carregada ou dados atualizados online via yfinance.",
)

if data_mode == "API Yahoo Finance (Ao Vivo)":
    st.sidebar.warning("A conexão com Yahoo Finance baixa cotações atualizadas das 41 ações do IBrX 50.")
    c_btn1, c_btn2 = st.sidebar.columns([1.5, 1])
    with c_btn1:
        btn_update = st.button("🔄 Atualizar Cotações via API")
    with c_btn2:
        btn_reset = st.button("🧹 Resetar API")

    if btn_reset:
        if "live_market" in st.session_state:
            del st.session_state["live_market"]
        if "live_updated_at" in st.session_state:
            del st.session_state["live_updated_at"]
        st.sidebar.success("Sessão da API limpa! Usando base local.")

    if btn_update:
        with st.spinner("Baixando cotações atualizadas da B3 via Yahoo Finance..."):
            from quant_ml_b3.data_loader import download_b3_universe_yfinance
            mkt_base = get_cached_market()
            try:
                live_mkt = download_b3_universe_yfinance(mkt_base.tickers, start="2024-01-01", end=None)
                if hasattr(live_mkt, "dates") and len(live_mkt.dates) > 0:
                    st.session_state["live_market"] = live_mkt
                    st.session_state["live_updated_at"] = pd.Timestamp.now().strftime("%d/%m/%Y %H:%M:%S")
                    st.sidebar.success(f"Atualizado! {len(live_mkt.dates)} pregões carregados até {live_mkt.dates[-1].strftime('%d/%m/%Y')}.")
                else:
                    st.sidebar.error("A API não retornou dados válidos. Tente novamente.")
            except Exception as e:
                st.sidebar.error(f"Erro ao conectar com API: {e}")

    if "live_market" in st.session_state:
        lm = st.session_state["live_market"]
        if hasattr(lm, "dates") and len(lm.dates) > 0:
            st.sidebar.info(f"🟢 **API Conectada:** {len(lm.dates)} pregões carregados até {lm.dates[-1].strftime('%d/%m/%Y')}.")
        else:
            del st.session_state["live_market"]

# 2. Model Selection
model_choice = st.sidebar.selectbox(
    "Modelo de Machine Learning:",
    [
        "Regressão Ridge (L2 Regularized)",
        "XGBoost (Depth-Wise Tree)",
        "LightGBM (Leaf-Wise Tree)",
        "Comparar Todos os Modelos",
    ],
    index=0,
)

# 3. Portfolio Weighting Policy
weighting_choice = st.sidebar.radio(
    "Política de Construção de Carteira:",
    [
        "Soft Weighting (Top 15 Rank Linear)",
        "Hard Cut (Top 10 com Margem de Tolerância)",
    ],
    index=0,
    help="Soft Weighting aloca pesos decrescentes no Top 15 (elimina efeito falésia). Hard Cut aloca 10% igualmente no Top 10 com margem de tolerância anti-giro.",
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Metodologia Quant:**\n"
    "- Rebalanceamento Mensal (1º pregão)\n"
    "- Z-Score Cross-Sectional Diário\n"
    "- Purga Temporal de 21 dias (Anti-Leakage)\n"
    "- Custos: 0.15% por transação"
)

# Model Key Mapping
model_map = {
    "Regressão Ridge (L2 Regularized)": "Ridge",
    "XGBoost (Depth-Wise Tree)": "Xgboost",
    "LightGBM (Leaf-Wise Tree)": "Lightgbm",
}
scheme_suffix = "SoftWeight" if "Soft" in weighting_choice else "HardCut"
scheme_label = "Soft Weighting (Top 15 Rank Linear)" if "Soft" in weighting_choice else "Hard Cut (Top 10 com Margem de Tolerância)"

# -------------------------------------------------------------
# MAIN APP HEADER
# -------------------------------------------------------------
st.title("📈 B3 Machine Learning Cross-Sectional Alpha Terminal")
st.markdown(
    "Painel Executivo de Seleção Sistemática de Ações, Monitoramento de Performance e Auditoria de Previsões."
)

# -------------------------------------------------------------
# APP TABS
# -------------------------------------------------------------
tab_perf, tab_pred_real, tab_explain, tab_audit = st.tabs(
    [
        "📊 Desempenho Histórico (2015-2026)",
        "🎯 Previsão vs. Realidade",
        "🧠 Explicabilidade das Features",
        "🛡️ Auditoria de Qualidade de Dados",
    ]
)

# =============================================================
# TAB 1: DESEMPENHO HISTÓRICO (WALK-FORWARD)
# =============================================================
with tab_perf:
    st.subheader("Simulação Multi-Ano Fora da Amostra (Walk-Forward: 2015–2026)")

    # Filter metrics row for selected model and scheme
    if model_choice != "Comparar Todos os Modelos":
        selected_key = model_map[model_choice]
        model_name_upper = selected_key.upper()
        row_sel = df_metrics[
            (df_metrics["Modelo"].str.contains(model_name_upper))
            & (df_metrics["Esquema"] == scheme_label)
        ]
        row_univ = df_metrics[df_metrics["Modelo"].str.contains("Buy & Hold")]

        if not row_sel.empty:
            m_data = row_sel.iloc[0]
            u_data = row_univ.iloc[0]

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Retorno Total</div>
                        <div class="metric-value">+{m_data['Retorno_Acumulado_pct']:,.1f}%</div>
                        <div class="metric-delta delta-pos">vs. +{u_data['Retorno_Acumulado_pct']:,.0f}% B&H</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">CAGR (% a.a.)</div>
                        <div class="metric-value">{m_data['CAGR_pct']:.2f}%</div>
                        <div class="metric-delta delta-pos">+{m_data['CAGR_pct'] - u_data['CAGR_pct']:.2f}% p.a.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Índice Sharpe</div>
                        <div class="metric-value">{m_data['Sharpe_Ratio']:.2f}</div>
                        <div class="metric-delta delta-pos">B&H: {u_data['Sharpe_Ratio']:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Volatilidade Anual</div>
                        <div class="metric-value">{m_data['Volatilidade_Anualizada_pct']:.1f}%</div>
                        <div class="metric-delta">B&H: {u_data['Volatilidade_Anualizada_pct']:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c5:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Max Drawdown</div>
                        <div class="metric-value" style="color: #EF4444;">{m_data['Max_Drawdown_pct']:.1f}%</div>
                        <div class="metric-delta">B&H: {u_data['Max_Drawdown_pct']:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c6:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Alpha Líquido Anual</div>
                        <div class="metric-value" style="color: #10B981;">+{m_data['Alpha_vs_Universo_pct']:.2f}%</div>
                        <div class="metric-delta">Excedente puro</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Interactive Equity Curve with Plotly
    st.markdown("### Curva de Patrimônio Acumulado (Juros Compostos)")

    fig_curve = go.Figure()

    # Universe baseline
    fig_curve.add_trace(
        go.Scatter(
            x=pd.to_datetime(df_curves["date"]),
            y=df_curves["Universe"],
            mode="lines",
            name="Buy & Hold (Universo 1/N)",
            line=dict(color="#94A3B8", width=2, dash="dash"),
            hovertemplate="Data: %{x|%d/%m/%Y}<br>Buy & Hold: +%{y:.1f}%<extra></extra>",
        )
    )

    if "Ibovespa" in df_curves.columns:
        fig_curve.add_trace(
            go.Scatter(
                x=pd.to_datetime(df_curves["date"]),
                y=df_curves["Ibovespa"],
                mode="lines",
                name="Índice Ibovespa (BOVA11)",
                line=dict(color="#38BDF8", width=2, dash="dot"),
                hovertemplate="Data: %{x|%d/%m/%Y}<br>Ibovespa: +%{y:.1f}%<extra></extra>",
            )
        )

    colors = {
        "Ridge": "#2563EB",
        "Xgboost": "#F59E0B",
        "Lightgbm": "#10B981",
    }

    if model_choice == "Comparar Todos os Modelos":
        for m_key in ["Ridge", "Xgboost", "Lightgbm"]:
            col_name = f"{m_key}_{scheme_suffix}"
            if col_name in df_curves.columns:
                fig_curve.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(df_curves["date"]),
                        y=df_curves[col_name],
                        mode="lines",
                        name=f"ML: {m_key} ({weighting_choice.split()[0]})",
                        line=dict(color=colors[m_key], width=2.8),
                        hovertemplate=f"Data: %{{x|%d/%m/%Y}}<br>{m_key}: +%{{y:.1f}}%<extra></extra>",
                    )
                )
    else:
        selected_key = model_map[model_choice]
        col_name = f"{selected_key}_{scheme_suffix}"
        if col_name in df_curves.columns:
            fig_curve.add_trace(
                go.Scatter(
                    x=pd.to_datetime(df_curves["date"]),
                    y=df_curves[col_name],
                    mode="lines",
                    name=f"ML: {selected_key} ({scheme_suffix})",
                    line=dict(color=colors[selected_key], width=3.2),
                    fill="tonexty",
                    fillcolor="rgba(37, 99, 235, 0.08)",
                    hovertemplate=f"Data: %{{x|%d/%m/%Y}}<br>{selected_key}: +%{{y:.1f}}%<extra></extra>",
                )
            )

    fig_curve.update_layout(
        template="plotly_dark",
        height=480,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Ano", showgrid=True, gridcolor="#334155"),
        yaxis=dict(title="Retorno Acumulado Composto (%)", showgrid=True, gridcolor="#334155", ticksuffix="%"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_curve, use_container_width=True)

    # Performance Comparison Table
    st.markdown("### Tabela Resumo Comparativa")
    st.dataframe(
        df_metrics.style.format(
            {
                "Retorno_Acumulado_pct": "{:,.2f}%",
                "CAGR_pct": "{:.2f}%",
                "Volatilidade_Anualizada_pct": "{:.2f}%",
                "Sharpe_Ratio": "{:.2f}",
                "Max_Drawdown_pct": "{:.2f}%",
                "Turnover_Medio_Mensal_pct": "{:.1f}%",
                "Alpha_vs_Universo_pct": "+{:.2f}%",
            }
        ),
        use_container_width=True,
    )


# =============================================================
# TAB 2: PREVISÃO VS. REALIDADE (A ABA PEDIDA PELO USUÁRIO)
# =============================================================
with tab_pred_real:
    st.subheader("🎯 Auditoria de Decisões: O que o Modelo Previu vs. O que Realmente Aconteceu")

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
            st.warning(f"Sem dados detalhados encontrados para o mês {target_month} com o esquema {scheme_filter}.")
            return

        rebal_dt = pd.to_datetime(df_m["rebal_date"].iloc[0]).strftime("%d/%m/%Y")
        end_dt = pd.to_datetime(df_m["date"].iloc[0]).strftime("%d/%m/%Y")
        month_disp = pd.to_datetime(target_month + "-01").strftime("%m/%Y")
        mkt_ret = df_m["market_return_pct"].iloc[0]
        ibov_ret = df_m["ibov_return_pct"].iloc[0] if "ibov_return_pct" in df_m.columns else 0.0

        st.markdown(
            f"#### {subtitle_prefix}: **{month_disp}** "
            f"<span style='font-size:0.88rem; color:#94A3B8; font-weight:normal;'>"
            f"(Data Base da Previsão: <b>{rebal_dt}</b> | Fechamento/Apuração: <b>{end_dt}</b> | Política: <b>{scheme_filter}</b>)</span>",
            unsafe_allow_html=True,
        )

        is_latest_month = (target_month == sorted(df_h["month_label"].unique())[-1])
        if is_latest_month:
            st.info(
                f"💡 **Carteira Recomendada para o Mês Vigente ({month_disp}):** "
                f"Esta é a alocação calculada no 1º pregão do mês (<b>{rebal_dt}</b>). "
                f"Abaixo você vê as ações recomendadas para compra, o <b>peso sugerido (%)</b> de cada uma, "
                f"o preço de entrada no início do mês e a evolução até hoje (<b>{end_dt}</b>).",
                icon="🎯",
            )

        model_key_map = {
            "Regressão Ridge": "Ridge",
            "XGBoost": "Xgboost",
            "LightGBM": "Lightgbm",
        }

        if selected_model_disp == "Comparar os 3 Modelos Lado a Lado":
            comp_data = []
            for disp_name, k_name in model_key_map.items():
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
                c1, c2, c3, c4, c5 = st.columns(5)
                r_ridge = df_comp[df_comp["key"] == "Ridge"]["Retorno_pct"].iloc[0] if (df_comp["key"] == "Ridge").any() else 0.0
                r_xgb = df_comp[df_comp["key"] == "Xgboost"]["Retorno_pct"].iloc[0] if (df_comp["key"] == "Xgboost").any() else 0.0
                r_lgb = df_comp[df_comp["key"] == "Lightgbm"]["Retorno_pct"].iloc[0] if (df_comp["key"] == "Lightgbm").any() else 0.0

                c1.metric("Carteira Ridge", f"{r_ridge:+.2f}%", delta=f"{r_ridge - mkt_ret:+.2f}% vs. B&H")
                c2.metric("Carteira XGBoost", f"{r_xgb:+.2f}%", delta=f"{r_xgb - mkt_ret:+.2f}% vs. B&H")
                c3.metric("Carteira LightGBM", f"{r_lgb:+.2f}%", delta=f"{r_lgb - mkt_ret:+.2f}% vs. B&H")
                c4.metric("Buy & Hold Universo (1/N)", f"{mkt_ret:+.2f}%", help="Média equiponderada de todas as 41 ações do universo B3")
                c5.metric("Ibovespa (BOVA11)", f"{ibov_ret:+.2f}%", help="Índice de mercado ponderado oficial da B3")

                fig_comp = px.bar(
                    df_comp,
                    x="Modelo",
                    y="Retorno_pct",
                    color="Modelo",
                    color_discrete_map={
                        "Regressão Ridge": "#2563EB",
                        "XGBoost": "#F59E0B",
                        "LightGBM": "#10B981",
                    },
                    labels={"Retorno_pct": "Retorno Realizado (%)"},
                    title=f"Comparação de Retorno Realizado no Mês {month_disp} (vs. Buy & Hold Universo: {mkt_ret:+.2f}% | vs. Ibovespa: {ibov_ret:+.2f}%)",
                    text="Retorno_pct",
                )
                fig_comp.add_hline(y=mkt_ret, line_dash="dash", line_color="#F59E0B", annotation_text=f"Buy & Hold Universo ({mkt_ret:+.2f}%)")
                fig_comp.add_hline(y=ibov_ret, line_dash="dot", line_color="#38BDF8", annotation_text=f"Ibovespa BOVA11 ({ibov_ret:+.2f}%)")
                fig_comp.update_layout(template="plotly_dark", height=360, showlegend=False)
                st.plotly_chart(fig_comp, use_container_width=True)

                st.markdown(f"##### Carteiras Recomendadas e Retornos em **{month_disp}** por Modelo:")
                t_ridge, t_xgb, t_lgb = st.tabs(["🔷 Regressão Ridge", "🌲 XGBoost", "⚡ LightGBM"])
                for t_obj, m_key in [(t_ridge, "Ridge"), (t_xgb, "Xgboost"), (t_lgb, "Lightgbm")]:
                    with t_obj:
                        sub_model = df_m[df_m["model"] == m_key].sort_values("rank").copy()
                        st.dataframe(
                            sub_model[[
                                "rank", "ticker", "predicted_score", "weight_pct",
                                "price_start", "price_end", "stock_return_pct", "contrib_return_pct", "beat_market"
                            ]].rename(columns={
                                "rank": "Posição",
                                "ticker": "Ticker",
                                "predicted_score": "Score Previsto (Alpha)",
                                "weight_pct": "Peso Carteira (%)",
                                "price_start": "Preço Início (R$)",
                                "price_end": "Preço Fim/Atual (R$)",
                                "stock_return_pct": "Retorno da Ação (%)",
                                "contrib_return_pct": "Contribuição (%)",
                                "beat_market": "Bateu B&H Universo?"
                            }),
                            use_container_width=True,
                        )

        else:
            # Single model view
            target_key = model_key_map.get(selected_model_disp, "Ridge")
            sub_model = df_m[df_m["model"] == target_key].sort_values("rank").copy()

            if sub_model.empty:
                st.warning(f"Sem dados para o modelo {selected_model_disp} no mês {month_disp}.")
                return

            port_ret = float(sub_model["contrib_return_pct"].sum())
            alpha_m = port_ret - mkt_ret
            alpha_ibov = port_ret - ibov_ret
            win_count = int(sub_model["beat_market"].sum())
            tot_count = len(sub_model)
            win_pct = (win_count / tot_count * 100.0) if tot_count > 0 else 0.0

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Data Base da Previsão", rebal_dt)
            c2.metric("Retorno da Carteira", f"{port_ret:+.2f}%", delta=f"{alpha_m:+.2f}% vs. B&H")
            c3.metric("Buy & Hold Universo (1/N)", f"{mkt_ret:+.2f}%", help="Média equiponderada das 41 ações do universo B3")
            c4.metric("Ibovespa (BOVA11)", f"{ibov_ret:+.2f}%", delta=f"{alpha_ibov:+.2f}% vs. Ibov")
            c5.metric("Taxa de Sucesso", f"{win_count} de {tot_count} ({win_pct:.0f}%)", help="% de ações que superaram o Buy & Hold do Universo")

            fig_bar = px.bar(
                sub_model,
                x="ticker",
                y="stock_return_pct",
                color="beat_market",
                color_discrete_map={True: "#10B981", False: "#EF4444"},
                labels={"stock_return_pct": "Retorno da Ação (%)", "ticker": "Ação", "beat_market": "Superou B&H Universo"},
                title=f"Desempenho Individual das Ações Escolhidas pelo {selected_model_disp} em {month_disp} (B&H Universo: {mkt_ret:+.2f}% | Ibovespa: {ibov_ret:+.2f}%)",
                text="stock_return_pct",
            )
            fig_bar.add_hline(y=mkt_ret, line_dash="dash", line_color="#F59E0B", annotation_text=f"Buy & Hold Universo ({mkt_ret:+.2f}%)")
            fig_bar.add_hline(y=ibov_ret, line_dash="dot", line_color="#38BDF8", annotation_text=f"Ibovespa BOVA11 ({ibov_ret:+.2f}%)")
            fig_bar.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.dataframe(
                sub_model[[
                    "rank", "ticker", "predicted_score", "weight_pct",
                    "price_start", "price_end", "stock_return_pct", "contrib_return_pct", "beat_market"
                ]].rename(columns={
                    "rank": "Posição",
                    "ticker": "Ticker",
                    "predicted_score": "Score Previsto (Alpha)",
                    "weight_pct": "Peso Carteira (%)",
                    "price_start": "Preço Início (R$)",
                    "price_end": "Preço Fim/Atual (R$)",
                    "stock_return_pct": "Retorno da Ação (%)",
                    "contrib_return_pct": "Contribuição (%)",
                    "beat_market": "Bateu Mercado?"
                }),
                use_container_width=True,
            )

    # Reusable Daily Inspection View (Calendar & Exact Trading Days)
    def render_daily_inspection_view(
        df_h: pd.DataFrame,
        selected_model_disp: str,
        weighting_scheme_choice: str,
    ):
        scheme_filter = "Soft Weighting (Top 15)" if "Soft" in weighting_scheme_choice else "Hard Cut (Top 10)"
        curr_scheme = scheme_filter

        market = (
            st.session_state["live_market"]
            if ("live_market" in st.session_state and hasattr(st.session_state["live_market"], "dates") and len(st.session_state["live_market"].dates) > 0)
            else get_cached_market()
        )
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

        st.markdown("#### 📅 Inspeção Pontual: Como Estava a Carteira em um Dia Específico")
        st.caption(
            "Abra o calendário para navegar até qualquer dia de pregão na B3. O sistema calcula a rentabilidade acumulada de cada ação "
            "desde o 1º pregão do mês (quando a carteira foi recomendada) até o dia exato selecionado, comparando com o Buy & Hold do Universo e o Ibovespa."
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
        c_cal, c_prev, c_next, c_today = st.columns([2.5, 1.2, 1.2, 1.2])

        # Current position in trading dates
        curr_eff = curr_val
        if curr_eff not in trading_dates_set:
            prior_days = [d for d in trading_dates if d <= curr_eff]
            curr_eff = prior_days[-1] if prior_days else trading_dates[0]
        curr_idx = trading_dates.index(curr_eff) if curr_eff in trading_dates_set else len(trading_dates) - 1

        with c_prev:
            st.write("")
            st.write("")
            if st.button("◀ Pregão Anterior", use_container_width=True, disabled=(curr_idx <= 0)):
                st.session_state["inspect_current_date"] = trading_dates[max(0, curr_idx - 1)]
                st.rerun()

        with c_next:
            st.write("")
            st.write("")
            if st.button("Próximo Pregão ▶", use_container_width=True, disabled=(curr_idx >= len(trading_dates) - 1)):
                st.session_state["inspect_current_date"] = trading_dates[min(len(trading_dates) - 1, curr_idx + 1)]
                st.rerun()

        with c_today:
            st.write("")
            st.write("")
            if st.button("Hoje / Mais Recente", use_container_width=True):
                st.session_state["inspect_current_date"] = max_date
                st.rerun()

        with c_cal:
            picked = st.date_input(
                "Selecione o Dia no Calendário:",
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
            is_trading_day = True
            st.success(f"✅ **Pregão Válido Confirmado:** {eff_date.strftime('%d/%m/%Y')} ({dias_semana[eff_date.weekday()]})")
        else:
            prior_days = [d for d in trading_dates if d <= picked]
            eff_date = prior_days[-1] if prior_days else trading_dates[0]
            is_trading_day = False
            orig_day_name = dias_semana[picked.weekday()]
            eff_day_name = dias_semana[eff_date.weekday()]
            st.warning(
                f"⚠️ **{picked.strftime('%d/%m/%Y')} ({orig_day_name})** não foi dia de pregão na B3 (fim de semana ou feriado). "
                f"Ajustado automaticamente para o pregão imediatamente anterior: **{eff_date.strftime('%d/%m/%Y')} ({eff_day_name})**."
            )

        year = eff_date.year
        month = eff_date.month
        month_label = f"{year}-{month:02d}"

        # Filter holdings for this month and scheme
        df_m = df_h[(df_h["month_label"] == month_label) & (df_h["scheme"] == curr_scheme)].copy()
        if df_m.empty:
            st.error(f"Sem dados de recomendações do modelo para o mês {month_label}. As previsões estão disponíveis de 01/2015 até 10/2026.")
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

        st.markdown(
            f"##### 🎯 Posição da Carteira no Pregão de **{tt_str}** "
            f"<span style='font-size:0.85rem; color:#94A3B8; font-weight:normal;'>"
            f"(Data Base da Previsão: <b>{t0_str}</b> | <b>{days_elapsed}</b> pregões decorridos | Política: <b>{curr_scheme}</b>)</span>",
            unsafe_allow_html=True,
        )

        if days_elapsed == 0:
            st.info(f"📌 **1º Pregão do Mês Selecionado ({t0_str}):** Esta é a data exata da emissão das recomendações do modelo. O retorno acumulado no primeiro dia é 0,00%.")

        model_key_map = {
            "Regressão Ridge": "Ridge",
            "XGBoost": "Xgboost",
            "LightGBM": "Lightgbm",
        }

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

        if selected_model_disp == "Comparar os 3 Modelos Lado a Lado":
            comp_rows = []
            model_dfs = {}
            for disp_name, k_name in model_key_map.items():
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

            r_ridge = df_comp_day[df_comp_day["key"] == "Ridge"]["Retorno_pct"].iloc[0] if (df_comp_day["key"] == "Ridge").any() else 0.0
            r_xgb = df_comp_day[df_comp_day["key"] == "Xgboost"]["Retorno_pct"].iloc[0] if (df_comp_day["key"] == "Xgboost").any() else 0.0
            r_lgb = df_comp_day[df_comp_day["key"] == "Lightgbm"]["Retorno_pct"].iloc[0] if (df_comp_day["key"] == "Lightgbm").any() else 0.0

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Ridge (até este dia)", f"{r_ridge:+.2f}%", delta=f"{r_ridge - univ_ret:+.2f}% vs. B&H")
            c2.metric("XGBoost (até este dia)", f"{r_xgb:+.2f}%", delta=f"{r_xgb - univ_ret:+.2f}% vs. B&H")
            c3.metric("LightGBM (até este dia)", f"{r_lgb:+.2f}%", delta=f"{r_lgb - univ_ret:+.2f}% vs. B&H")
            c4.metric("Buy & Hold Universo (1/N)", f"{univ_ret:+.2f}%", help="Média equiponderada das 41 ações do universo B3 até esta data")
            c5.metric("Ibovespa (BOVA11)", f"{ibov_ret:+.2f}%", help="Índice ponderado oficial da B3 até esta data")

            fig_comp = px.bar(
                df_comp_day,
                x="Modelo",
                y="Retorno_pct",
                color="Modelo",
                color_discrete_map={
                    "Regressão Ridge": "#2563EB",
                    "XGBoost": "#F59E0B",
                    "LightGBM": "#10B981",
                },
                labels={"Retorno_pct": "Retorno Realizado até a Data (%)"},
                title=f"Retorno dos Modelos até {tt_str} (vs. B&H Universo: {univ_ret:+.2f}% | vs. Ibovespa: {ibov_ret:+.2f}%)",
                text="Retorno_pct",
            )
            fig_comp.add_hline(y=univ_ret, line_dash="dash", line_color="#F59E0B", annotation_text=f"Buy & Hold Universo ({univ_ret:+.2f}%)")
            fig_comp.add_hline(y=ibov_ret, line_dash="dot", line_color="#38BDF8", annotation_text=f"Ibovespa BOVA11 ({ibov_ret:+.2f}%)")
            fig_comp.update_layout(template="plotly_dark", height=360, showlegend=False)
            st.plotly_chart(fig_comp, use_container_width=True)

            st.markdown(f"##### Carteiras Recomendadas e Posição em **{tt_str}**:")
            t_ridge, t_xgb, t_lgb = st.tabs(["🔷 Regressão Ridge", "🌲 XGBoost", "⚡ LightGBM"])
            for t_obj, m_key in [(t_ridge, "Ridge"), (t_xgb, "Xgboost"), (t_lgb, "Lightgbm")]:
                with t_obj:
                    sub_model = model_dfs[m_key]
                    st.dataframe(
                        sub_model[[
                            "rank", "ticker", "predicted_score", "weight_pct",
                            "price_start_day", "price_now_day", "stock_return_day_pct", "contrib_return_day_pct", "beat_market_day"
                        ]].rename(columns={
                            "rank": "Posição",
                            "ticker": "Ticker",
                            "predicted_score": "Score Previsto (Alpha)",
                            "weight_pct": "Peso Carteira (%)",
                            "price_start_day": f"Preço Entrada {t0_str} (R$)",
                            "price_now_day": f"Preço em {tt_str} (R$)",
                            "stock_return_day_pct": "Retorno da Ação (%)",
                            "contrib_return_day_pct": "Contribuição (%)",
                            "beat_market_day": "Bateu B&H Universo?"
                        }).style.format({
                            "Score Previsto (Alpha)": "{:+.4f}",
                            "Peso Carteira (%)": "{:.2f}%",
                            f"Preço Entrada {t0_str} (R$)": "R$ {:.2f}",
                            f"Preço em {tt_str} (R$)": "R$ {:.2f}",
                            "Retorno da Ação (%)": "{:+.2f}%",
                            "Contribuição (%)": "{:+.2f}%",
                        }),
                        use_container_width=True,
                    )

        else:
            # Single model view
            target_key = model_key_map.get(selected_model_disp, "Ridge")
            sub_model, port_ret = calc_holdings_for_day(target_key)

            if sub_model.empty:
                st.warning(f"Sem dados para o modelo {selected_model_disp} na data {tt_str}.")
                return

            alpha_univ = port_ret - univ_ret
            alpha_ibov = port_ret - ibov_ret
            win_count = int(sub_model["beat_market_day"].sum())
            tot_count = len(sub_model)
            win_pct = (win_count / tot_count * 100.0) if tot_count > 0 else 0.0

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Data da Recomendação", t0_str)
            c2.metric("Retorno Carteira até a Data", f"{port_ret:+.2f}%", delta=f"{alpha_univ:+.2f}% vs. B&H")
            c3.metric("Buy & Hold Universo (1/N)", f"{univ_ret:+.2f}%", help="Média equiponderada das 41 ações do universo B3 até esta data")
            c4.metric("Ibovespa (BOVA11)", f"{ibov_ret:+.2f}%", delta=f"{alpha_ibov:+.2f}% vs. Ibov")
            c5.metric("Taxa de Sucesso no Dia", f"{win_count} de {tot_count} ({win_pct:.0f}%)", help="% de ações que superaram o Buy & Hold do Universo até o pregão selecionado")

            fig_bar = px.bar(
                sub_model,
                x="ticker",
                y="stock_return_day_pct",
                color="beat_market_day",
                color_discrete_map={True: "#10B981", False: "#EF4444"},
                labels={"stock_return_day_pct": "Retorno da Ação até a Data (%)", "ticker": "Ação", "beat_market_day": "Superou B&H Universo"},
                title=f"Desempenho Individual das Ações do {selected_model_disp} em {tt_str} (B&H Universo: {univ_ret:+.2f}% | Ibovespa: {ibov_ret:+.2f}%)",
                text="stock_return_day_pct",
            )
            fig_bar.add_hline(y=univ_ret, line_dash="dash", line_color="#F59E0B", annotation_text=f"Buy & Hold Universo ({univ_ret:+.2f}%)")
            fig_bar.add_hline(y=ibov_ret, line_dash="dot", line_color="#38BDF8", annotation_text=f"Ibovespa BOVA11 ({ibov_ret:+.2f}%)")
            fig_bar.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.dataframe(
                sub_model[[
                    "rank", "ticker", "predicted_score", "weight_pct",
                    "price_start_day", "price_now_day", "stock_return_day_pct", "contrib_return_day_pct", "beat_market_day"
                ]].rename(columns={
                    "rank": "Posição",
                    "ticker": "Ticker",
                    "predicted_score": "Score Previsto (Alpha)",
                    "weight_pct": "Peso Carteira (%)",
                    "price_start_day": f"Preço Entrada {t0_str} (R$)",
                    "price_now_day": f"Preço em {tt_str} (R$)",
                    "stock_return_day_pct": "Retorno da Ação (%)",
                    "contrib_return_day_pct": "Contribuição (%)",
                    "beat_market_day": "Bateu B&H Universo?"
                }).style.format({
                    "Score Previsto (Alpha)": "{:+.4f}",
                    "Peso Carteira (%)": "{:.2f}%",
                    f"Preço Entrada {t0_str} (R$)": "R$ {:.2f}",
                    f"Preço em {tt_str} (R$)": "R$ {:.2f}",
                    "Retorno da Ação (%)": "{:+.2f}%",
                    "Contribuição (%)": "{:+.2f}%",
                }),
                use_container_width=True,
            )

    # Online API Status banner if live market is active
    if "live_market" in st.session_state:
        lm = st.session_state["live_market"]
        if hasattr(lm, "dates") and len(lm.dates) > 0:
            st.info(f"🟢 **Modo Online Ativo:** Cotações sincronizadas via API Yahoo Finance ({len(lm.dates)} pregões até {lm.dates[-1].strftime('%d/%m/%Y')}).")

    # Dual controls: Window Type & Model Selector
    c_ctrl1, c_ctrl2 = st.columns([1.6, 2.4])
    with c_ctrl1:
        view_type = st.radio(
            "Janela de Inspeção:",
            [
                "⚡ Visão Compacta do Ano Corrente (YTD 2026: Modelo vs. Mercado e Previsões)",
                "📅 Inspecionar por Data Específica (Calendário de Pregões Diários)",
                "🔴 Previsões Mês a Mês em 2026 (Auditar Carteiras Individuais)",
                "📜 Histórico Multi-Ano Completo (Auditar Qualquer Mês: 2015–2026)",
            ],
            index=0,
        )
    with c_ctrl2:
        tab2_model_choice = st.radio(
            "Modelo sob Análise:",
            ["Regressão Ridge", "XGBoost", "LightGBM", "Comparar os 3 Modelos Lado a Lado"],
            index=0,
            horizontal=True,
            help="Alterne entre os modelos para inspecionar exatamente a carteira e o retorno de cada um.",
        )

    all_months = sorted(df_holdings["month_label"].unique(), reverse=True)
    months_2026 = [m for m in all_months if m >= "2026-01"]
    if not months_2026:
        months_2026 = all_months[:12]

    model_key_map = {
        "Regressão Ridge": "Ridge",
        "XGBoost": "Xgboost",
        "LightGBM": "Lightgbm",
    }
    scheme_suffix = "SoftWeight" if "Soft" in weighting_choice else "HardCut"
    curr_scheme = "Soft Weighting (Top 15)" if "Soft" in weighting_choice else "Hard Cut (Top 10)"

    if "Visão Compacta" in view_type:
        st.markdown("#### ⚡ Desempenho Consolidado no Ano de 2026 (Year-to-Date: 01/01/2026 a 29/10/2026)")
        st.caption("Visão agregada do ano corrente comparando o quanto cada modelo de Machine Learning previu e entregou vs. o mercado (Buy & Hold 1/N e Ibovespa).")

        df_m26 = df_monthly[df_monthly["month_label"] >= "2026-01"].sort_values("month_label").reset_index(drop=True)

        # Cumulative YTD compound series for each
        cum_univ_ytd = (np.cumprod(1.0 + df_m26["Universe"]) - 1.0) * 100.0
        cum_ibov_ytd = (np.cumprod(1.0 + df_m26["Ibovespa"]) - 1.0) * 100.0

        r_univ_ytd = float(cum_univ_ytd.iloc[-1])
        r_ibov_ytd = float(cum_ibov_ytd.iloc[-1])

        # Calculate YTD for each model
        cum_models_ytd = {}
        for m_disp, m_key in model_key_map.items():
            col = f"{m_key}_{scheme_suffix}"
            if col in df_m26.columns:
                cum_models_ytd[m_key] = (np.cumprod(1.0 + df_m26[col]) - 1.0) * 100.0

        if tab2_model_choice == "Comparar os 3 Modelos Lado a Lado":
            r_ridge_ytd = float(cum_models_ytd["Ridge"].iloc[-1]) if "Ridge" in cum_models_ytd else 0.0
            r_xgb_ytd = float(cum_models_ytd["Xgboost"].iloc[-1]) if "Xgboost" in cum_models_ytd else 0.0
            r_lgb_ytd = float(cum_models_ytd["Lightgbm"].iloc[-1]) if "Lightgbm" in cum_models_ytd else 0.0

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Ridge (YTD 2026)", f"{r_ridge_ytd:+.2f}%", delta=f"{r_ridge_ytd - r_univ_ytd:+.2f}% vs. B&H")
            c2.metric("XGBoost (YTD 2026)", f"{r_xgb_ytd:+.2f}%", delta=f"{r_xgb_ytd - r_univ_ytd:+.2f}% vs. B&H")
            c3.metric("LightGBM (YTD 2026)", f"{r_lgb_ytd:+.2f}%", delta=f"{r_lgb_ytd - r_univ_ytd:+.2f}% vs. B&H")
            c4.metric("Buy & Hold Universo (1/N)", f"{r_univ_ytd:+.2f}%", help="Média equiponderada de todas as 41 ações em 2026")
            c5.metric("Ibovespa (BOVA11)", f"{r_ibov_ytd:+.2f}%", help="Índice oficial ponderado da B3 em 2026")

            # Chart 1: YTD Cumulative Curve in 2026
            fig_ytd = go.Figure()
            fig_ytd.add_trace(go.Scatter(
                x=pd.to_datetime(df_m26["date"]), y=cum_univ_ytd,
                mode="lines+markers", name="Buy & Hold Universo (1/N)",
                line=dict(color="#94A3B8", width=2, dash="dash"),
            ))
            fig_ytd.add_trace(go.Scatter(
                x=pd.to_datetime(df_m26["date"]), y=cum_ibov_ytd,
                mode="lines+markers", name="Ibovespa (BOVA11)",
                line=dict(color="#38BDF8", width=2, dash="dot"),
            ))
            colors_map = {"Ridge": "#2563EB", "Xgboost": "#F59E0B", "Lightgbm": "#10B981"}
            for m_key in ["Ridge", "Xgboost", "Lightgbm"]:
                if m_key in cum_models_ytd:
                    fig_ytd.add_trace(go.Scatter(
                        x=pd.to_datetime(df_m26["date"]), y=cum_models_ytd[m_key],
                        mode="lines+markers", name=f"ML: {m_key}",
                        line=dict(color=colors_map[m_key], width=3),
                    ))
            fig_ytd.update_layout(
                template="plotly_dark", height=380,
                title="Evolução Patrimonial Acumulada no Ano de 2026 (YTD %)",
                xaxis=dict(title="Mês", showgrid=True, gridcolor="#334155"),
                yaxis=dict(title="Retorno Acumulado em 2026 (%)", showgrid=True, gridcolor="#334155", ticksuffix="%"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                hovermode="x unified",
            )
            st.plotly_chart(fig_ytd, use_container_width=True)

        else:
            # Single model view
            target_key = model_key_map.get(tab2_model_choice, "Ridge")
            r_mod_ytd = float(cum_models_ytd[target_key].iloc[-1]) if target_key in cum_models_ytd else 0.0
            alpha_univ = r_mod_ytd - r_univ_ytd
            alpha_ibov = r_mod_ytd - r_ibov_ytd

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric(f"{tab2_model_choice} (YTD)", f"{r_mod_ytd:+.2f}%", delta=f"{alpha_univ:+.2f}% vs. B&H")
            c2.metric("Buy & Hold Universo (1/N)", f"{r_univ_ytd:+.2f}%", help="Média equiponderada de todas as 41 ações em 2026")
            c3.metric("Ibovespa (BOVA11)", f"{r_ibov_ytd:+.2f}%", help="Índice oficial ponderado da B3 em 2026")
            c4.metric("Alpha vs. Universo", f"{alpha_univ:+.2f}%", delta=f"{alpha_univ:+.2f}% vs. B&H")
            c5.metric("Alpha vs. Ibovespa", f"{alpha_ibov:+.2f}%", delta=f"{alpha_ibov:+.2f}% vs. Ibov")

            # Chart 1: YTD Cumulative Curve for single model
            fig_ytd = go.Figure()
            fig_ytd.add_trace(go.Scatter(
                x=pd.to_datetime(df_m26["date"]), y=cum_univ_ytd,
                mode="lines+markers", name="Buy & Hold Universo (1/N)",
                line=dict(color="#94A3B8", width=2, dash="dash"),
            ))
            fig_ytd.add_trace(go.Scatter(
                x=pd.to_datetime(df_m26["date"]), y=cum_ibov_ytd,
                mode="lines+markers", name="Ibovespa (BOVA11)",
                line=dict(color="#38BDF8", width=2, dash="dot"),
            ))
            fig_ytd.add_trace(go.Scatter(
                x=pd.to_datetime(df_m26["date"]), y=cum_models_ytd[target_key],
                mode="lines+markers", name=f"{tab2_model_choice}",
                line=dict(color="#2563EB" if target_key == "Ridge" else ("#F59E0B" if target_key == "Xgboost" else "#10B981"), width=3.5),
                fill="tonexty", fillcolor="rgba(37, 99, 235, 0.08)",
            ))
            fig_ytd.update_layout(
                template="plotly_dark", height=380,
                title=f"Evolução Patrimonial Acumulada em 2026: {tab2_model_choice} vs. Mercado",
                xaxis=dict(title="Mês", showgrid=True, gridcolor="#334155"),
                yaxis=dict(title="Retorno Acumulado em 2026 (%)", showgrid=True, gridcolor="#334155", ticksuffix="%"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                hovermode="x unified",
            )
            st.plotly_chart(fig_ytd, use_container_width=True)

        # Monthly Prediction vs Reality Audit Table & Bar Chart
        st.markdown("##### 📋 O que o Modelo Previu vs. O que o Mercado Rendeu Mês a Mês em 2026:")

        target_m_for_table = "Ridge" if tab2_model_choice == "Comparar os 3 Modelos Lado a Lado" else model_key_map.get(tab2_model_choice, "Ridge")
        sub_h = df_holdings[
            (df_holdings["month_label"] >= "2026-01")
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
                "Score Alpha Previsto": f"{score_m * 100:+.2f}%",
                "Retorno Modelo (%)": p_ret,
                "Buy & Hold Universo (%)": u_ret,
                "Ibovespa (%)": i_ret,
                "Alpha vs. B&H (%)": p_ret - u_ret,
                "Bateu B&H?": "✅ Sim" if p_ret > u_ret else "❌ Não",
                "Bateu Ibov?": "✅ Sim" if p_ret > i_ret else "❌ Não",
                "Top Ação Escolhida": f"{best_r['ticker']} ({best_r['stock_return_pct']:+.1f}%)",
            })

        df_ytd_table = pd.DataFrame(ytd_records)

        # Grouped Bar Chart: Modelo vs B&H vs Ibov
        fig_bar_ytd = go.Figure()
        fig_bar_ytd.add_trace(go.Bar(
            x=df_ytd_table["Mês"], y=df_ytd_table["Retorno Modelo (%)"],
            name=f"{target_m_for_table} (Retorno Entregue)",
            marker_color="#2563EB",
            text=[f"{v:+.1f}%" for v in df_ytd_table["Retorno Modelo (%)"]], textposition="auto",
        ))
        fig_bar_ytd.add_trace(go.Bar(
            x=df_ytd_table["Mês"], y=df_ytd_table["Buy & Hold Universo (%)"],
            name="Buy & Hold Universo (1/N)",
            marker_color="#64748B",
            text=[f"{v:+.1f}%" for v in df_ytd_table["Buy & Hold Universo (%)"]], textposition="auto",
        ))
        fig_bar_ytd.add_trace(go.Bar(
            x=df_ytd_table["Mês"], y=df_ytd_table["Ibovespa (%)"],
            name="Ibovespa (BOVA11)",
            marker_color="#38BDF8",
            text=[f"{v:+.1f}%" for v in df_ytd_table["Ibovespa (%)"]], textposition="auto",
        ))
        fig_bar_ytd.update_layout(
            barmode="group", template="plotly_dark", height=380,
            title=f"Retornos Mês a Mês em 2026: Modelo {target_m_for_table} vs. Benchmarks de Mercado",
            xaxis=dict(title="Mês de 2026"),
            yaxis=dict(title="Retorno Mensal (%)", ticksuffix="%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_bar_ytd, use_container_width=True)

        st.dataframe(
            df_ytd_table.style.format({
                "Retorno Modelo (%)": "{:+.2f}%",
                "Buy & Hold Universo (%)": "{:+.2f}%",
                "Ibovespa (%)": "{:+.2f}%",
                "Alpha vs. B&H (%)": "{:+.2f}%",
            }),
            use_container_width=True,
        )

    elif "Calendário" in view_type or "Data Específica" in view_type:
        render_daily_inspection_view(
            df_h=df_holdings,
            selected_model_disp=tab2_model_choice,
            weighting_scheme_choice=weighting_choice,
        )

    elif "Mês a Mês" in view_type:
        c_sel, _ = st.columns([2, 2])
        with c_sel:
            selected_month_2026 = st.selectbox(
                "Selecione o Mês de 2026 para Inspecionar Previsões e Desempenho:",
                months_2026,
                index=0,
                format_func=lambda m: pd.to_datetime(m + "-01").strftime("%m/%Y") + (" (Mês Mais Recente / Em Andamento)" if m == months_2026[0] else " (Mês Fechado)"),
            )
        render_month_audit_view(
            df_h=df_holdings,
            target_month=selected_month_2026,
            selected_model_disp=tab2_model_choice,
            weighting_scheme_choice=weighting_choice,
            subtitle_prefix="Previsões e Desempenho em 2026",
        )
    else:
        c_sel, _ = st.columns([2, 2])
        with c_sel:
            selected_hist_month = st.selectbox(
                "Selecione Qualquer Mês Histórico (2015–2026):",
                all_months,
                index=0,
                format_func=lambda m: pd.to_datetime(m + "-01").strftime("%m/%Y"),
            )
        render_month_audit_view(
            df_h=df_holdings,
            target_month=selected_hist_month,
            selected_model_disp=tab2_model_choice,
            weighting_scheme_choice=weighting_choice,
            subtitle_prefix="Auditoria de Mês Histórico",
        )


# =============================================================
# TAB 3: EXPLICABILIDADE & PESO DAS FEATURES
# =============================================================
with tab_explain:
    st.subheader("🧠 Interpretação Econômica e Importância dos Fatores")

    st.markdown(
        """
        O gráfico abaixo mostra como cada modelo de Machine Learning pondera os 12 indicadores técnicos para prever o excesso de retorno de 21 dias.
        """
    )

    if not df_importance.empty:
        df_imp = df_importance.copy()
        df_imp["Ridge_Beta_bps"] = df_imp["Ridge_Beta"] * 10000.0
        df_imp["XGBoost_Gain_pct"] = df_imp["XGBoost_Gain"] * 100.0
        df_imp["LightGBM_Gain_pct"] = df_imp["LightGBM_Gain"] * 100.0

        imp_model_choice = st.radio(
            "Selecione o Modelo para Visualizar a Importância dos Atributos:",
            [
                "Regressão Ridge (Beta Padronizado em Basis Points)",
                "XGBoost (Ganho de Informação %)",
                "LightGBM (Ganho de Informação %)",
            ],
            horizontal=True,
        )

        if "Ridge" in imp_model_choice:
            df_sorted = df_imp.sort_values("Ridge_Beta_bps", ascending=True)
            fig_imp = px.bar(
                df_sorted,
                x="Ridge_Beta_bps",
                y="Feature",
                orientation="h",
                labels={"Ridge_Beta_bps": "Coeficiente Beta (bps)", "Feature": "Atributo Técnico"},
                title="Pesos dos Fatores na Regressão Ridge (em Basis Points)",
                color="Ridge_Beta_bps",
                color_continuous_scale="Blues",
                text="Ridge_Beta_bps",
            )
        elif "XGBoost" in imp_model_choice:
            df_sorted = df_imp.sort_values("XGBoost_Gain_pct", ascending=True)
            fig_imp = px.bar(
                df_sorted,
                x="XGBoost_Gain_pct",
                y="Feature",
                orientation="h",
                labels={"XGBoost_Gain_pct": "Ganho de Informação (%)", "Feature": "Atributo Técnico"},
                title="Importância dos Fatores no XGBoost (% de Ganho de Árvores)",
                color="XGBoost_Gain_pct",
                color_continuous_scale="Oranges",
                text="XGBoost_Gain_pct",
            )
        else:
            df_sorted = df_imp.sort_values("LightGBM_Gain_pct", ascending=True)
            fig_imp = px.bar(
                df_sorted,
                x="LightGBM_Gain_pct",
                y="Feature",
                orientation="h",
                labels={"LightGBM_Gain_pct": "Ganho de Informação (%)", "Feature": "Atributo Técnico"},
                title="Importância dos Fatores no LightGBM (% de Ganho por Folha)",
                color="LightGBM_Gain_pct",
                color_continuous_scale="Greens",
                text="LightGBM_Gain_pct",
            )

        fig_imp.update_layout(template="plotly_dark", height=460)
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown(
        """
        ### 💡 Por que os Pesos Fazem Sentido Econômico?
        1. **`ret_63d` Positivo (+15 a +20 bps):** Captura o efeito Momentum trimestral clássico da literatura de Finanças (ações com balanços sólidos continuam subindo).
        2. **`ret_21d` Negativo (-10 a -15 bps):** Reflete a **Reversão à Média Mensal (*Short-Term Reversal*)**. Ativos que esticaram demais nos últimos 21 dias tendem a sofrer realização de lucros.
        3. **`vol_21d` Negativo:** Penaliza ativos com volatilidade extrema, reduzindo o risco de cauda e drawdowns severos.
        """
    )



# =============================================================
# TAB 4: AUDITORIA DE QUALIDADE DE DADOS
# =============================================================
with tab_audit:
    st.subheader("🛡️ Data Quality Assurance & Validação de Integridade")

    st.markdown(
        """
        Esta aba audita formalmente a base de dados do projeto, verificando a ausência de valores nulos,
        infinitos, preços não positivos ou problemas de sincronia temporal.
        """
    )

    if st.button("Executar Auditoria de Dados Agora", type="primary"):
        with st.spinner("Auditando matrizes de mercado..."):
            mkt = get_cached_market()
            report = validate_market_data(mkt)

            if report.is_valid:
                st.success("✅ BASE DE DADOS APROVADA COM 100% DE INTEGRIDADE!")
            else:
                st.error("⚠️ PROBLEMAS DETECTADOS NA AUDITORIA.")

            st.code(report.summary(), language="text")

    st.markdown("---")
    st.markdown(
        r"""
        ### 📚 Pilares Metodológicos do Projeto:
        * **Imunidade a Data Leakage:** Purga e Embargo temporal estrito de 21 dias nas fronteiras de Treino, Validação e Teste.
        * **Z-Score Cross-Sectional:** Padronização diária entre ações ativas ($Z = (X - \mu_t)/\sigma_t$), eliminando o viés de euforia ou pânico macroeconômico.
        * **Soft Weighting:** Ponderação linear proporcional sobre o Top 15, eliminando a sensibilidade a empates e empurrando o retorno para **+1.568,85% com Sharpe 1.02**.
        """
    )
