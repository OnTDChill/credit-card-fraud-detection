"""Shared Streamlit UI helpers for the fraud dashboard."""

from __future__ import annotations

import streamlit as st


DEFAULT_PAGE_LAYOUT = "wide"
DEFAULT_SIDEBAR_STATE = "expanded"
DEFAULT_PAGE_TITLE = "Fraud Model Detective"


THEME_CSS = """
<style>
:root {
    --fd-bg: #06111f;
    --fd-surface: rgba(11, 23, 40, 0.84);
    --fd-surface-strong: rgba(16, 31, 52, 0.96);
    --fd-border: rgba(148, 163, 184, 0.18);
    --fd-border-strong: rgba(76, 201, 240, 0.28);
    --fd-text: #e5eefb;
    --fd-muted: #9fb0c7;
    --fd-accent: #4cc9f0;
    --fd-accent-soft: rgba(76, 201, 240, 0.15);
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(76, 201, 240, 0.16), transparent 26%),
        radial-gradient(circle at top right, rgba(129, 140, 248, 0.12), transparent 28%),
        linear-gradient(180deg, #04101d 0%, #071320 48%, #0a1626 100%);
    color: var(--fd-text);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(5, 14, 26, 0.98), rgba(10, 21, 37, 0.98));
    border-right: 1px solid var(--fd-border);
}

header[data-testid="stHeader"] {
    background: transparent;
}

div[data-testid="stToolbar"] {
    right: 1rem;
}

div.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2.2rem;
}

.fd-hero {
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    border-radius: 24px;
    border: 1px solid var(--fd-border);
    background: linear-gradient(180deg, rgba(16, 31, 52, 0.92), rgba(10, 21, 37, 0.78));
    box-shadow: 0 24px 64px rgba(2, 8, 20, 0.34);
}

.fd-kicker {
    color: var(--fd-accent);
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}

.fd-hero h1 {
    color: var(--fd-text);
    font-size: 2.1rem;
    line-height: 1.1;
    margin: 0;
}

.fd-hero p {
    color: var(--fd-muted);
    font-size: 0.98rem;
    line-height: 1.6;
    margin: 0.55rem 0 0;
    max-width: 68rem;
}

div[data-testid="stMetric"] {
    background: var(--fd-surface);
    border: 1px solid var(--fd-border);
    border-radius: 18px;
    padding: 1rem 1.05rem;
    box-shadow: 0 16px 36px rgba(2, 8, 20, 0.24);
}

div[data-testid="stMetric"] label {
    color: var(--fd-muted);
}

div[data-testid="stMetricValue"] {
    color: #f8fbff;
    font-weight: 700;
}

div[data-testid="stMetricDelta"] {
    color: #9ae6b4;
}

div[data-baseweb="tab-list"] {
    gap: 0.5rem;
}

button[data-baseweb="tab"] {
    border-radius: 999px;
    border: 1px solid transparent;
    background: transparent;
    color: var(--fd-muted);
    padding: 0.55rem 0.9rem;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: var(--fd-surface-strong);
    border-color: var(--fd-border-strong);
    color: var(--fd-text);
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--fd-border);
    border-radius: 16px;
    overflow: hidden;
}

div[data-testid="stExpander"] {
    border-color: var(--fd-border);
    background: rgba(11, 23, 40, 0.4);
}

button[kind="primary"],
button[kind="secondary"] {
    border-radius: 12px;
}
</style>
"""


def apply_dashboard_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def configure_dashboard_page(
    page_title: str = DEFAULT_PAGE_TITLE,
    *,
    layout: str = DEFAULT_PAGE_LAYOUT,
    sidebar_state: str = DEFAULT_SIDEBAR_STATE,
) -> None:
    st.set_page_config(
        page_title=page_title,
        layout=layout,
        initial_sidebar_state=sidebar_state,
    )
    apply_dashboard_theme()


def render_page_header(title: str, subtitle: str, kicker: str = "Fraud Operations Center") -> None:
    st.markdown(
        f"""
        <div class="fd-hero">
            <div class="fd-kicker">{kicker}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )