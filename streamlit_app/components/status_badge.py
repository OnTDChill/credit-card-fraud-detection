"""Traffic light status badge for system health."""

from __future__ import annotations

import streamlit as st


_TONE_CLASS = {
    "success": "fd-pill--success",
    "warning": "fd-pill--warning",
    "danger": "fd-pill--danger",
}


def render_status_badge(label: str, tone: str, help_text: str = "") -> None:
    safe_help = help_text.replace("\"", "&quot;") if help_text else ""
    title_attr = f" title=\"{safe_help}\"" if safe_help else ""
    css_class = _TONE_CLASS.get(tone, "fd-pill--warning")
    st.markdown(
        f"<span class=\"fd-pill {css_class}\"{title_attr}>{label}</span>",
        unsafe_allow_html=True,
    )
