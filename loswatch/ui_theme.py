"""Shared visual language for the LOSWatch-AI Streamlit dashboard."""
from __future__ import annotations

import html

import streamlit as st


THEME_CSS = """
<style>
:root {
  --los-ink: #122033;
  --los-muted: #64748b;
  --los-border: #e2e8f0;
  --los-panel: #ffffff;
  --los-navy: #0f2747;
  --los-teal: #0f766e;
  --los-blue: #2563eb;
  --los-amber: #b45309;
  --los-bg: #f5f7fb;
}
[data-testid="stAppViewContainer"] { background: var(--los-bg); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: var(--los-navy); }
[data-testid="stSidebar"] * { color: #e5eef8; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(255,255,255,.08); border-radius: 8px; }
.los-brand { padding: .35rem 0 1.1rem; border-bottom: 1px solid rgba(255,255,255,.16); margin-bottom: 1rem; }
.los-brand-name { color: #fff; font-size: 1.18rem; font-weight: 750; letter-spacing: -.02em; }
.los-brand-sub { color: #a9bfd7; font-size: .76rem; margin-top: .2rem; }
.los-kicker { color: var(--los-teal); font-size: .74rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; margin-bottom: .25rem; }
.los-page-title { color: var(--los-ink); font-size: 2rem; font-weight: 760; letter-spacing: -.04em; line-height: 1.1; margin: 0; }
.los-page-description { color: var(--los-muted); font-size: .95rem; margin-top: .55rem; margin-bottom: 1.35rem; }
.los-section-title { color: var(--los-ink); font-size: 1.12rem; font-weight: 720; margin: 1.1rem 0 .55rem; }
.los-section-caption { color: var(--los-muted); font-size: .84rem; margin-bottom: .7rem; }
.los-callout { background: #eef8f7; border: 1px solid #c7ebe5; border-radius: 12px; color: #145b56; padding: .8rem 1rem; }
div[data-testid="stMetric"] { background: var(--los-panel); border: 1px solid var(--los-border); border-radius: 14px; padding: .85rem 1rem; box-shadow: 0 2px 8px rgba(15,39,71,.04); }
div[data-testid="stMetricLabel"] p { color: var(--los-muted); font-size: .78rem; }
div[data-testid="stMetricValue"] { color: var(--los-ink); font-size: 1.6rem; }
div[data-testid="stDataFrame"] { border: 1px solid var(--los-border); border-radius: 12px; overflow: hidden; }
div[data-testid="stExpander"] { border-color: var(--los-border); border-radius: 12px; }
button[kind="primary"] { background: var(--los-teal); border-color: var(--los-teal); }
</style>
"""


def apply_theme() -> None:
    """Apply the dashboard-wide visual system once per Streamlit rerun."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.sidebar.markdown(
        '<div class="los-brand"><div class="los-brand-name">LOSWatch-AI</div>'
        '<div class="los-brand-sub">Hospital operations intelligence</div></div>',
        unsafe_allow_html=True,
    )


def page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="los-kicker">{html.escape(kicker)}</div>'
        f'<div class="los-page-title">{html.escape(title)}</div>'
        f'<div class="los-page-description">{html.escape(description)}</div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, caption: str | None = None) -> None:
    st.markdown(f'<div class="los-section-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="los-section-caption">{html.escape(caption)}</div>', unsafe_allow_html=True)
