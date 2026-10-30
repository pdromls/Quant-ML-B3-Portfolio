# -*- coding: utf-8 -*-
"""
Visual layer of the Streamlit terminal: design tokens, theme-aware CSS,
KPI cards, panels with an info button, explanatory dialogs and Plotly styling.

Light/Dark mode follows Streamlit's native theme (configured in
.streamlit/config.toml). Streamlit sets ``color-scheme`` on the app root, so
every token below is declared with CSS ``light-dark()`` and resolves to the
active theme without any Python-side detection.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from html import escape
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -------------------------------------------------------------
# COLOR SYSTEM (mid tones readable on both light and dark backgrounds)
# -------------------------------------------------------------
MODEL_COLORS = {
    "Ridge": "#4C6EF5",
    "Xgboost": "#E09A2B",
    "Lightgbm": "#9061F9",
}
MODEL_LABELS = {
    "Ridge": "Ridge",
    "Xgboost": "XGBoost",
    "Lightgbm": "LightGBM",
}
UNIVERSE_COLOR = "#8B93A1"
IBOV_COLOR = "#1FA2C0"
POS_COLOR = "#1F9D63"
NEG_COLOR = "#E5484D"


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


# -------------------------------------------------------------
# NUMBER FORMATTING (pt-BR: 1.691,39)
# -------------------------------------------------------------
def fmt_num(value: float, decimals: int = 2, sign: bool = False) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    rounded = round(float(value), decimals)
    body = f"{abs(rounded):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if rounded < 0:
        return "−" + body
    return ("+" if sign and rounded > 0 else "") + body


def fmt_pct(value: float, decimals: int = 2, sign: bool = True) -> str:
    return fmt_num(value, decimals, sign) + "%"


def fmt_pp(value: float, decimals: int = 2) -> str:
    return fmt_num(value, decimals, sign=True) + " p.p."


def tone_of(value: float, higher_is_better: bool = True) -> str:
    if value == 0:
        return ""
    return "pos" if (value > 0) == higher_is_better else "neg"


# -------------------------------------------------------------
# GLOBAL CSS
# -------------------------------------------------------------
_CSS = """
<style>
:root {
  --qa-ink: light-dark(#15171C, #E7E9EE);
  --qa-muted: light-dark(#5F636C, #98A0AD);
  --qa-faint: light-dark(#8D9098, #6B7381);
  --qa-border: light-dark(#E2DFD7, #252C38);
  --qa-border-strong: light-dark(#D3D0C7, #2E3644);
  --qa-surface: light-dark(#FFFFFF, #131821);
  --qa-surface-2: light-dark(#F9F8F5, #0F131A);
  --qa-pos: light-dark(#1E8656, #3FC58B);
  --qa-pos-bg: light-dark(#E5F3EC, #12291F);
  --qa-neg: light-dark(#C33F38, #F0675F);
  --qa-neg-bg: light-dark(#F9E8E6, #2E1716);
  --qa-warn: light-dark(#9A6A0B, #E5A33F);
  --qa-warn-bg: light-dark(#FBF1DC, #2B2210);
  --qa-accent: light-dark(#3553D6, #7089FF);
  --qa-accent-bg: light-dark(#E9EDFC, #1A2140);
  --qa-shadow: light-dark(0 1px 2px rgba(20,20,20,.04), none);
  --qa-mono: 'IBM Plex Mono', ui-monospace, monospace;
}

[data-testid="stMainBlockContainer"] { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1480px; }
[data-testid="stSidebarContent"] { padding-top: .4rem; }
.qa-num { font-family: var(--qa-mono); font-variant-numeric: tabular-nums; letter-spacing: -.01em; }
.qa-pos { color: var(--qa-pos) !important; }
.qa-neg { color: var(--qa-neg) !important; }

/* Panels (bordered containers created with a "qa_panel_" key) */
div[class*="st-key-qa_panel_"] { background: var(--qa-surface); box-shadow: var(--qa-shadow); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 26px; }
.stTabs [data-baseweb="tab"] { font-weight: 500; padding: 0 0 10px; }

/* Info buttons */
div[class*="st-key-qa_info_"] button { color: var(--qa-faint); min-height: 0; padding: 2px; }
div[class*="st-key-qa_info_"] button:hover { color: var(--qa-accent); }

/* Brand + sidebar */
.qa-brand { display: flex; gap: 12px; align-items: center; margin: 2px 0 6px; }
.qa-mark { width: 36px; height: 36px; border-radius: 8px; background: var(--qa-ink); display: grid; place-items: center; flex: none; }
.qa-mark svg { width: 20px; height: 20px; }
.qa-brand b { display: block; font-size: 15px; font-weight: 700; letter-spacing: -.01em; color: var(--qa-ink); }
.qa-brand small { font-size: 12px; color: var(--qa-muted); }
.qa-side-label { font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--qa-faint); margin: 14px 0 0; }
.qa-hint { font-size: 11.5px; color: var(--qa-faint); margin-top: -6px; }
.qa-dl { display: grid; grid-template-columns: auto 1fr; gap: 7px 12px; font-size: 12px; margin: 4px 0 0; }
.qa-dl dt, .qa-dl dd { font-size: 12px !important; line-height: 1.4; margin: 0 !important; padding: 0 !important; }
.qa-dl dt { color: var(--qa-muted); }
.qa-dl dd { text-align: right; font-weight: 500; color: var(--qa-ink); }
.qa-side-foot { font-size: 11px; color: var(--qa-faint); border-top: 1px solid var(--qa-border); padding-top: 12px; margin-top: 18px; line-height: 1.5; }

/* Top bar */
.qa-topbar { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; padding-bottom: 14px; border-bottom: 1px solid var(--qa-border); margin-bottom: 4px; }
.qa-crumbs { font-size: 13px; color: var(--qa-muted); }
.qa-crumbs b { color: var(--qa-ink); font-weight: 600; }
.qa-chips { display: flex; gap: 8px; flex-wrap: wrap; }
.qa-chip { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; padding: 4px 10px; border: 1px solid var(--qa-border); border-radius: 999px; background: var(--qa-surface); color: var(--qa-muted); }
.qa-chip i { width: 7px; height: 7px; border-radius: 50%; background: var(--qa-pos); display: block; }
.qa-chip i.live { background: var(--qa-accent); }

/* Section header */
.qa-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; flex-wrap: wrap; margin: 10px 0 6px; }
.qa-head h2 { font-size: 22px; font-weight: 600; letter-spacing: -.02em; margin: 0; padding: 0; color: var(--qa-ink); }
.qa-head p { color: var(--qa-muted); font-size: 13px; margin: 4px 0 0; }
.qa-tags { display: flex; gap: 6px; flex-wrap: wrap; }
.qa-tag { font-size: 11.5px; padding: 3px 8px; border-radius: 5px; background: var(--qa-surface); border: 1px solid var(--qa-border); color: var(--qa-muted); }
.qa-tag b { color: var(--qa-ink); font-weight: 600; }

/* Panel header */
.qa-ph h4 { font-size: 14.5px; font-weight: 600; margin: 0; padding: 0; color: var(--qa-ink); }
.qa-ph p { font-size: 12px; color: var(--qa-muted); margin: 2px 0 0; }

/* KPI cards */
.qa-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, var(--qa-min, 150px)), 1fr)); background: var(--qa-surface); border: 1px solid var(--qa-border); border-radius: 12px; box-shadow: var(--qa-shadow); margin: 6px 0 4px; }
.qa-kpis.split { background: none; border: none; box-shadow: none; gap: 12px; }
.qa-kpi { padding: 14px 16px; border-left: 1px solid var(--qa-border); min-width: 0; }
.qa-kpi:first-child { border-left: none; }
.qa-kpis:not(.split) .qa-kpi { border-top: 1px solid var(--qa-border); margin-top: -1px; }
.qa-kpis.split .qa-kpi { border: 1px solid var(--qa-border); border-radius: 10px; background: var(--qa-surface); box-shadow: var(--qa-shadow); }
.qa-kpi .l { font-size: 11.5px; color: var(--qa-muted); font-weight: 500; display: flex; justify-content: space-between; gap: 6px; }
.qa-kpi .l span { color: var(--qa-faint); font-weight: 400; }
.qa-kpi .v { font-family: var(--qa-mono); font-variant-numeric: tabular-nums; font-size: 24px; font-weight: 500; margin-top: 6px; letter-spacing: -.03em; color: var(--qa-ink); white-space: nowrap; }
.qa-kpis.split .qa-kpi .v { font-size: 21px; }
.qa-kpi .c { display: flex; justify-content: space-between; align-items: center; gap: 6px; margin-top: 10px; font-size: 11.5px; color: var(--qa-muted); border-top: 1px dashed var(--qa-border); padding-top: 8px; flex-wrap: wrap; }
.qa-kpi .c .qa-num { color: var(--qa-ink); }
.qa-delta { font-family: var(--qa-mono); font-size: 11.5px; font-weight: 500; padding: 1px 6px; border-radius: 4px; white-space: nowrap; }
.qa-delta.pos { color: var(--qa-pos); background: var(--qa-pos-bg); }
.qa-delta.neg { color: var(--qa-neg); background: var(--qa-neg-bg); }
.qa-delta.neutral { color: var(--qa-muted); background: var(--qa-surface-2); }

/* Explanation, notices, legends */
.qa-explain { display: flex; gap: 12px; padding: 11px 14px; border-radius: 8px; background: var(--qa-surface-2); border: 1px solid var(--qa-border); font-size: 12.5px; color: var(--qa-muted); line-height: 1.55; margin: 2px 0 4px; }
.qa-explain b { color: var(--qa-ink); font-weight: 600; }
.qa-explain .k { flex: none; font-size: 10.5px; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; color: var(--qa-faint); padding-top: 2px; width: 84px; }
.qa-notice { display: flex; gap: 12px; align-items: flex-start; margin: 8px 0; padding: 11px 14px; border-radius: 10px; border: 1px solid var(--qa-border); background: var(--qa-surface-2); font-size: 12.5px; color: var(--qa-muted); line-height: 1.55; }
.qa-notice b { color: var(--qa-ink); }
.qa-notice .ic { width: 20px; height: 20px; border-radius: 50%; display: grid; place-items: center; font-weight: 700; font-size: 11.5px; flex: none; background: var(--qa-accent-bg); color: var(--qa-accent); }
.qa-notice.ok .ic { background: var(--qa-pos-bg); color: var(--qa-pos); }
.qa-notice.warn .ic { background: var(--qa-warn-bg); color: var(--qa-warn); }
.qa-notice.err .ic { background: var(--qa-neg-bg); color: var(--qa-neg); }
.qa-caption { font-size: 11.5px; color: var(--qa-faint); margin: 2px 0 0; }

/* Small stats table (drawdown summary, data-quality checklist) */
.qa-table { width: 100%; border-collapse: collapse; font-size: 12.5px; margin: 2px 0 6px; }
.qa-table th { font-size: 10.5px; font-weight: 500; text-transform: uppercase; letter-spacing: .05em; color: var(--qa-faint); text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--qa-border); white-space: nowrap; }
.qa-table td { padding: 8px 10px; border-bottom: 1px solid var(--qa-border); text-align: right; white-space: nowrap; color: var(--qa-ink); }
.qa-table tr:last-child td { border-bottom: none; }
.qa-table th:first-child, .qa-table td:first-child { text-align: left; }
.qa-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 8px; vertical-align: baseline; }
.qa-badge { font-size: 11px; font-weight: 500; padding: 2px 7px; border-radius: 4px; }
.qa-badge.pos { color: var(--qa-pos); background: var(--qa-pos-bg); }
.qa-badge.neg { color: var(--qa-neg); background: var(--qa-neg-bg); }

/* Dialog content */
.qa-formula { font-family: var(--qa-mono); font-size: 12.5px; background: var(--qa-surface-2); border: 1px solid var(--qa-border); border-radius: 8px; padding: 10px 12px; margin: 4px 0 14px; }
.qa-howto { display: grid; grid-template-columns: auto 1fr; gap: 10px 12px; margin: 4px 0 14px; font-size: 13.5px; }
.qa-howto span { width: 22px; height: 22px; border-radius: 6px; background: var(--qa-accent-bg); color: var(--qa-accent); display: grid; place-items: center; font: 600 11.5px var(--qa-mono); }
.qa-dialog-foot { font-size: 11.5px; color: var(--qa-faint); border-top: 1px solid var(--qa-border); padding-top: 10px; margin-top: 6px; }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# -------------------------------------------------------------
# SIDEBAR PIECES
# -------------------------------------------------------------
def brand(title: str, subtitle: str) -> None:
    html(
        f"""<div class="qa-brand"><div class="qa-mark"><svg viewBox="0 0 20 20" fill="none">
        <path d="M2 15l5-5 3 3 7-8" style="stroke: light-dark(#F6F5F1, #0C0F14)"
        stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
        <div><b>{escape(title)}</b><small>{escape(subtitle)}</small></div></div>"""
    )


def side_label(text: str) -> None:
    html(f'<div class="qa-side-label">{escape(text)}</div>')


def hint(text: str) -> None:
    html(f'<div class="qa-hint">{text}</div>')


def definition_list(items: Sequence[tuple[str, str]]) -> None:
    rows = "".join(f"<dt>{escape(k)}</dt><dd>{v}</dd>" for k, v in items)
    html(f'<dl class="qa-dl">{rows}</dl>')


def side_footer(text: str) -> None:
    html(f'<div class="qa-side-foot">{text}</div>')


# -------------------------------------------------------------
# PAGE PIECES
# -------------------------------------------------------------
def topbar(crumb: str, title: str, chips: Sequence[tuple[str, str | None]]) -> None:
    """chips: (html_text, dot) where dot is None, 'ok' or 'live'."""
    chip_html = "".join(
        '<span class="qa-chip">' + ("" if dot is None else f'<i class="{dot}"></i>') + text + "</span>"
        for text, dot in chips
    )
    html(
        f'<div class="qa-topbar"><div class="qa-crumbs">{escape(crumb)} · <b>{escape(title)}</b></div>'
        f'<div class="qa-chips">{chip_html}</div></div>'
    )


def section_header(title: str, subtitle: str = "", tags: Sequence[tuple[str, str]] = ()) -> None:
    tag_html = "".join(f'<span class="qa-tag">{escape(k)} <b>{v}</b></span>' for k, v in tags)
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    html(f'<div class="qa-head"><div><h2>{title}</h2>{sub}</div><div class="qa-tags">{tag_html}</div></div>')


def kpi_cards(items: Sequence[dict], split: bool = False) -> None:
    """
    Each item: label, value, and optionally unit (small label at the right),
    tone ('pos'/'neg') for the value, foot (html, left side of the footer),
    delta + delta_tone ('pos'/'neg'/'neutral') for the right side, help (tooltip).
    """
    cards = []
    for it in items:
        unit = f"<span>{escape(it['unit'])}</span>" if it.get("unit") else ""
        tone = f" qa-{it['tone']}" if it.get("tone") else ""
        delta = (
            f'<span class="qa-delta {it.get("delta_tone") or "neutral"}">{it["delta"]}</span>'
            if it.get("delta")
            else ""
        )
        foot = it.get("foot", "")
        footer = f'<div class="c"><span>{foot}</span>{delta}</div>' if (foot or delta) else ""
        title = f' title="{escape(it["help"])}"' if it.get("help") else ""
        cards.append(
            f'<div class="qa-kpi"{title}><div class="l">{escape(it["label"])}{unit}</div>'
            f'<div class="v{tone}">{it["value"]}</div>{footer}</div>'
        )
    cls = "qa-kpis split" if split else "qa-kpis"
    html(f'<div class="{cls}" style="--qa-cols:{len(items)}">{"".join(cards)}</div>')


def explain(text: str, label: str = "Como ler") -> None:
    html(f'<div class="qa-explain"><span class="k">{escape(label)}</span><div>{text}</div></div>')


def notice(text: str, tone: str = "info") -> None:
    icon = {"info": "i", "ok": "✓", "warn": "!", "err": "!"}.get(tone, "i")
    html(f'<div class="qa-notice {tone}"><span class="ic">{icon}</span><div>{text}</div></div>')


def caption(text: str) -> None:
    html(f'<div class="qa-caption">{text}</div>')


def dot(color: str) -> str:
    return f'<span class="qa-dot" style="background:{color}"></span>'


def badge(ok: bool, yes: str = "Superou", no: str = "Abaixo") -> str:
    return f'<span class="qa-badge {"pos" if ok else "neg"}">{yes if ok else no}</span>'


def stats_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    html(f'<table class="qa-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>')


@contextmanager
def panel(
    key: str,
    title: str,
    subtitle: str = "",
    info: str | None = None,
    controls: bool = False,
) -> Iterator[st.delta_generator.DeltaGenerator | None]:
    """
    Bordered card with a title, an optional controls slot (yielded) and an
    optional info button at the top-right corner that opens the matching
    explanatory dialog from INFO_TOPICS.
    """
    key = re.sub(r"[^A-Za-z0-9_-]", "_", key)
    with st.container(border=True, key=f"qa_panel_{key}"):
        spec = [6, 3, 0.3] if controls else [9, 0.3]
        cols = st.columns(spec, vertical_alignment="center", gap="small")
        sub = f"<p>{subtitle}</p>" if subtitle else ""
        with cols[0]:
            html(f'<div class="qa-ph"><h4>{title}</h4>{sub}</div>')
        if info:
            with cols[-1]:
                if st.button(
                    "",
                    icon=":material/info:",
                    key=f"qa_info_{key}",
                    type="tertiary",
                    help="Sobre este gráfico",
                ):
                    show_info(info)
        yield cols[1] if controls else None


# -------------------------------------------------------------
# PLOTLY STYLING
# -------------------------------------------------------------
def style_fig(
    fig: go.Figure,
    height: int,
    y_suffix: str | None = "%",
    legend: bool = True,
    hovermode: str | bool = "x unified",
) -> go.Figure:
    """
    Layout shared by every chart. No template is forced: st.plotly_chart's
    native "streamlit" theme paints backgrounds, fonts and grids for the
    active Light/Dark mode, while trace colors stay as defined here.
    """
    fig.update_layout(
        height=height,
        margin=dict(l=4, r=4, t=34 if legend else 10, b=4),
        font=dict(family="IBM Plex Sans, sans-serif", size=12),
        separators=",.",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode=hovermode,
        hoverlabel=dict(font_family="IBM Plex Mono, monospace", font_size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    if y_suffix:
        fig.update_yaxes(ticksuffix=y_suffix)
    fig.update_xaxes(title=None)
    return fig


def add_benchmark_lines(fig: go.Figure, universe: float, ibov: float) -> None:
    fig.add_hline(
        y=universe,
        line_dash="dash",
        line_color=UNIVERSE_COLOR,
        line_width=1.5,
        annotation_text=f"B&H universo {fmt_pct(universe)}",
        annotation_position="bottom right",
        annotation_font_color=UNIVERSE_COLOR,
    )
    fig.add_hline(
        y=ibov,
        line_dash="dot",
        line_color=IBOV_COLOR,
        line_width=2,
        annotation_text=f"Ibovespa {fmt_pct(ibov)}",
        annotation_position="top right",
        annotation_font_color=IBOV_COLOR,
    )


# -------------------------------------------------------------
# DRAWDOWN
# -------------------------------------------------------------
def drawdown_series(cum_return_pct: pd.Series) -> pd.Series:
    """Drawdown (%) from a cumulative-return series (%), with the initial capital as first peak."""
    wealth = 1.0 + cum_return_pct.to_numpy(dtype=float) / 100.0
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return pd.Series((wealth / peak - 1.0) * 100.0, index=cum_return_pct.index)


def drawdown_stats(dates: pd.Series, cum_return_pct: pd.Series) -> dict:
    """Worst drawdown, its peak and trough dates, months from trough to recovery and the current drawdown."""
    wealth = 1.0 + cum_return_pct.to_numpy(dtype=float) / 100.0
    dd = drawdown_series(cum_return_pct).to_numpy()
    d = pd.to_datetime(dates).reset_index(drop=True)
    trough = int(np.argmin(dd))
    pre = np.concatenate([[1.0], wealth[: trough + 1]])
    peak_pos = int(np.argmax(pre)) - 1  # -1 => initial capital
    peak_value = pre[peak_pos + 1]
    rec = next((i for i in range(trough + 1, len(wealth)) if wealth[i] >= peak_value), None)
    return {
        "worst": float(dd[trough]),
        "peak_date": d.iloc[peak_pos] if peak_pos >= 0 else None,
        "trough_date": d.iloc[trough],
        "recovery_months": (rec - trough) if rec is not None else None,
        "current": float(dd[-1]),
    }


# -------------------------------------------------------------
# DATAFRAME STYLING
# -------------------------------------------------------------
def color_signed(value) -> str:
    if isinstance(value, (int, float)) and not pd.isna(value) and value != 0:
        return f"color: {POS_COLOR if value > 0 else NEG_COLOR}"
    return ""


def color_bool(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return f"color: {POS_COLOR if value else NEG_COLOR}; font-weight: 600"
    return ""


# -------------------------------------------------------------
# EXPLANATORY DIALOGS
# -------------------------------------------------------------
INFO_TOPICS: dict[str, dict[str, str]] = {
    "equity": {
        "title": "Curva de patrimônio acumulado",
        "body": """
<p>Mostra quanto <b>R$ 1 investido no início da simulação</b> teria rendido até cada data, já com
rebalanceamento mensal no 1º pregão e custo de <b>0,15% por transação</b>. Todas as decisões
do modelo usam apenas dados disponíveis até a data da previsão (walk-forward fora da amostra).</p>
<div class="qa-formula">Retorno acumulado(t) = Π (1 + r<sub>mês</sub>) − 1</div>
<div class="qa-howto">
<span>↗</span><div><b>Linha do modelo:</b> retorno composto da carteira escolhida pelo modelo.</div>
<span>┄</span><div><b>Buy &amp; Hold 1/N:</b> todas as ações do universo com o mesmo peso. É o benchmark justo, pois usa as mesmas ações disponíveis para o modelo.</div>
<span>⋯</span><div><b>Ibovespa (BOVA11):</b> índice oficial ponderado por valor de mercado.</div>
<span>↕</span><div><b>Distância vertical</b> entre o modelo e os benchmarks = ganho acumulado sobre o mercado.</div>
</div>
<p><b>Escala log:</b> mostra o patrimônio (R$ 1 → R$ x) em escala logarítmica. Nela, a mesma variação
percentual ocupa o mesmo espaço vertical em qualquer época, o que facilita comparar 2016 com 2026.</p>
""",
        "foot": "Fonte: outputs/tables/walkforward_monthly_curves.csv (fechamentos mensais).",
    },
    "drawdown": {
        "title": "Drawdown: queda desde o pico anterior",
        "body": """
<p>O drawdown mede, em cada data, <b>quanto a carteira está abaixo do maior valor que ela já havia
atingido</b> até ali. Ele responde à pergunta "quanto eu teria perdido se tivesse entrado no pior
momento?", algo que a curva de rendimento esconde.</p>
<div class="qa-formula">Drawdown(t) = Patrimônio(t) ÷ máx(Patrimônio até t) − 1</div>
<div class="qa-howto">
<span>0</span><div><b>Linha em 0%:</b> a carteira está no topo histórico. Não existe drawdown positivo.</div>
<span>↓</span><div><b>Profundidade do vale:</b> tamanho da perda. O vale mais fundo é o Max Drawdown dos cartões.</div>
<span>↔</span><div><b>Largura do vale:</b> tempo até recuperar o pico anterior (coluna "Recuperação" da tabela).</div>
<span>≷</span><div><b>Linha do modelo acima da tracejada:</b> nesse período o modelo perdeu menos que o mercado.</div>
</div>
<p>Por que separar da curva de rendimento? Quando a base já está alta, uma queda pequena em % parece
um tombo enorme em pontos percentuais no gráfico de cima. No drawdown, todas as quedas ficam na mesma
régua: <b>% perdido a partir do topo</b>.</p>
""",
        "foot": "Calculado sobre fechamentos mensais da simulação walk-forward; o capital inicial conta como primeiro pico.",
    },
    "comparison": {
        "title": "Tabela resumo comparativa",
        "body": """
<p>Métricas de toda a simulação walk-forward para cada combinação de modelo e política de carteira,
mais os dois benchmarks.</p>
<div class="qa-howto">
<span>%</span><div><b>CAGR:</b> retorno composto anualizado.</div>
<span>σ</span><div><b>Volatilidade anual:</b> desvio-padrão dos retornos, anualizado.</div>
<span>S</span><div><b>Sharpe:</b> retorno por unidade de risco. Acima de 1 é considerado muito bom para ações.</div>
<span>↓</span><div><b>Max drawdown:</b> pior queda desde um pico.</div>
<span>⟳</span><div><b>Turnover médio mensal:</b> fração da carteira trocada a cada rebalanceamento; quanto maior, mais custo.</div>
<span>α</span><div><b>Alpha vs. universo:</b> CAGR do modelo menos CAGR do Buy &amp; Hold 1/N.</div>
</div>
<p>A linha destacada é a combinação selecionada na barra lateral. Valores em <b>verde e negrito</b>
são os melhores de cada coluna entre os modelos.</p>
""",
        "foot": "Fonte: outputs/tables/comparison_weighting_schemes.csv.",
    },
    "stock_bars": {
        "title": "Retorno individual das ações",
        "body": """
<p>Cada barra é o retorno de uma ação recomendada, do preço de entrada (1º pregão do mês, data da
previsão) até a data de apuração.</p>
<div class="qa-howto">
<span>▮</span><div><b>Verde:</b> a ação superou o Buy &amp; Hold do universo no período. <b>Vermelho:</b> ficou abaixo.</div>
<span>┄</span><div><b>Linha tracejada:</b> retorno médio equiponderado do universo (Buy &amp; Hold 1/N).</div>
<span>⋯</span><div><b>Linha pontilhada:</b> retorno do Ibovespa (BOVA11).</div>
</div>
<p>Um mês ruim para a carteira pode ainda ser um bom mês para o modelo, se a maioria das ações
escolhidas caiu menos que o mercado. O que importa é a posição relativa às linhas de referência.</p>
""",
        "foot": "Ordenação por rank = ordem de recomendação do modelo (maior peso primeiro).",
    },
    "holdings": {
        "title": "Carteira recomendada",
        "body": """
<div class="qa-howto">
<span>#</span><div><b>Posição:</b> rank do modelo no dia da previsão.</div>
<span>α</span><div><b>Score previsto:</b> excesso de retorno de 21 dias previsto pelo modelo (unidade decimal; 0,0123 ≈ +1,23%).</div>
<span>w</span><div><b>Peso:</b> fração do capital alocada na ação pela política de carteira.</div>
<span>R$</span><div><b>Preços:</b> fechamento ajustado na entrada e na data de apuração.</div>
<span>Σ</span><div><b>Contribuição:</b> retorno da ação × peso. A soma das contribuições é o retorno da carteira.</div>
</div>
""",
        "foot": "Clique no cabeçalho de uma coluna para reordenar; o ícone de download exporta a tabela em CSV.",
    },
    "model_comparison": {
        "title": "Comparação entre modelos",
        "body": """
<p>Retorno realizado pela carteira de cada modelo no mesmo período, com a mesma política de
construção. As linhas horizontais são os benchmarks no mesmo intervalo.</p>
<p>Use as abas abaixo do gráfico para ver a carteira completa de cada modelo.</p>
""",
        "foot": "Retorno da carteira = soma das contribuições (retorno × peso) das ações recomendadas.",
    },
    "ytd_curve": {
        "title": "Evolução acumulada no ano",
        "body": """
<p>Retorno composto mês a mês dentro do ano selecionado, partindo de 0% em janeiro. Permite ver em
que meses o modelo abriu ou perdeu vantagem sobre os benchmarks.</p>
<div class="qa-formula">Acumulado(mês k) = Π<sub>i ≤ k</sub> (1 + r<sub>i</sub>) − 1</div>
""",
        "foot": "Fonte: outputs/tables/walkforward_monthly_returns.csv.",
    },
    "ytd_bars": {
        "title": "Retornos mês a mês",
        "body": """
<p>Para cada mês, o retorno entregue pelo modelo lado a lado com o Buy &amp; Hold 1/N e o Ibovespa.
Barras do modelo acima das demais indicam meses em que a seleção de ações agregou valor.</p>
<p>No modo de comparação, as barras e a tabela usam a Regressão Ridge como referência.</p>
""",
        "foot": "Retornos mensais simples, sem composição.",
    },
    "ytd_table": {
        "title": "Previsão vs. resultado por mês",
        "body": """
<div class="qa-howto">
<span>α</span><div><b>Score alpha previsto:</b> média do score das ações recomendadas no mês.</div>
<span>%</span><div><b>Retorno do modelo:</b> soma das contribuições da carteira no mês.</div>
<span>✓</span><div><b>Bateu B&amp;H / Ibov:</b> se o retorno da carteira superou cada benchmark.</div>
<span>★</span><div><b>Top ação escolhida:</b> a ação de rank 1 e o retorno que ela entregou.</div>
</div>
""",
        "foot": "Fonte: outputs/tables/walkforward_monthly_holdings.csv.",
    },
    "importance": {
        "title": "Importância das features",
        "body": """
<div class="qa-howto">
<span>β</span><div><b>Ridge:</b> coeficiente padronizado em basis points. Como as features são z-scores
cross-sectionais, o beta mostra quanto o retorno previsto muda para +1 desvio-padrão na feature.
<b>Positivo</b> favorece a ação; <b>negativo</b> penaliza.</div>
<span>G</span><div><b>XGBoost e LightGBM:</b> ganho de informação (%) de cada feature nas divisões das
árvores. Indica relevância, mas não o sinal do efeito.</div>
</div>
<p>As 12 features são indicadores técnicos padronizados diariamente entre as ações ativas; o alvo é o
excesso de retorno de 21 dias.</p>
""",
        "foot": "Fonte: outputs/tables/feature_importance.csv.",
    },
    "quality": {
        "title": "Auditoria de qualidade de dados",
        "body": """
<p>Verificações estruturais aplicadas às matrizes de preços antes de qualquer modelagem:</p>
<div class="qa-howto">
<span>1</span><div>Datas em ordem estritamente crescente e sem duplicatas.</div>
<span>2</span><div>Dimensões consistentes entre abertura, fechamento, volume e datas.</div>
<span>3</span><div>Ausência de valores infinitos.</div>
<span>4</span><div>Ausência de preços negativos ou zero.</div>
<span>5</span><div>Cobertura média de ações ativas por pregão.</div>
</div>
""",
        "foot": "Implementação: quant_ml_b3.data_loader.validate_market_data.",
    },
}


def show_info(topic: str) -> None:
    spec = INFO_TOPICS[topic]

    @st.dialog(spec["title"], width="large")
    def _dialog() -> None:
        html(spec["body"])
        if spec.get("foot"):
            html(f'<div class="qa-dialog-foot">{spec["foot"]}</div>')

    _dialog()
