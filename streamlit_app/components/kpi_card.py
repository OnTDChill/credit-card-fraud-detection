"""KPI card component with tooltip support."""

from __future__ import annotations

import streamlit as st


def render_kpi_card(
    *,
    label: str,
    value: str,
    delta: str | None = None,
    delta_positive: bool | None = None,
    accent_color: str | None = None,
    help_text: str = "",
    subtitle: str | None = None,
) -> None:
    safe_help = help_text.replace("\"", "&quot;") if help_text else ""
    title_attr = f" title=\"{safe_help}\"" if safe_help else ""
    style_parts = ["font-size:2.2rem"]
    if accent_color:
        style_parts.append(f"color:{accent_color}")
    value_style = f" style=\"{' '.join(style_parts)}\"" if style_parts else ""

    delta_html = ""
    if delta is not None:
        tone = "#22c55e" if delta_positive is not False else "#ef4444"
        delta_html = f"<div class=\"fd-kpi-subtitle\" style=\"color:{tone};\">{delta}</div>"

    subtitle_html = ""
    if subtitle:
        subtitle_html = f"<div class=\"fd-kpi-subtitle\" style=\"opacity:0.7;font-size:0.85rem;\">{subtitle}</div>"

    st.markdown(
        f"""
        <div class="fd-kpi-card"{title_attr}>
            <h3>{label}</h3>
            <div class="fd-kpi-value"{value_style}>{value}</div>
            {delta_html}
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
