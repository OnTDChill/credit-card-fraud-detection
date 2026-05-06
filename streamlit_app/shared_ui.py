"""Shared Streamlit UI helpers for the fraud dashboard."""

from __future__ import annotations

import streamlit as st


DEFAULT_PAGE_LAYOUT = "wide" # type: ignore
DEFAULT_SIDEBAR_STATE = "expanded" # type: ignore
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
    --fd-success: #22c55e;
    --fd-warning: #f59e0b;
    --fd-danger: #ef4444;
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

.fd-kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin-top: 0.5rem;
}

.fd-kpi-card {
    border-radius: 20px;
    border: 1px solid var(--fd-border);
    background: var(--fd-surface-strong);
    padding: 1.1rem 1.2rem;
    box-shadow: 0 18px 36px rgba(2, 8, 20, 0.28);
}

.fd-kpi-card h3 {
    margin: 0 0 0.45rem;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--fd-muted);
}

.fd-kpi-card .fd-kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--fd-text);
}

.fd-kpi-card .fd-kpi-subtitle {
    margin-top: 0.4rem;
    font-size: 0.85rem;
    color: var(--fd-muted);
}

.fd-kpi-card .fd-kpi-delta {
    margin-top: 0.35rem;
    font-size: 0.82rem;
    font-weight: 600;
}

.fd-kpi-card .fd-kpi-delta.up {
    color: var(--fd-success);
}

.fd-kpi-card .fd-kpi-delta.down {
    color: var(--fd-danger);
}

.fd-kpi-card .fd-kpi-delta.neutral {
    color: var(--fd-warning);
}

.fd-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.fd-pill--success {
    background: rgba(34, 197, 94, 0.18);
    color: #86efac;
    border: 1px solid rgba(34, 197, 94, 0.4);
}

.fd-pill--warning {
    background: rgba(245, 158, 11, 0.18);
    color: #fde68a;
    border: 1px solid rgba(245, 158, 11, 0.4);
}

.fd-pill--danger {
    background: rgba(239, 68, 68, 0.18);
    color: #fecaca;
    border: 1px solid rgba(239, 68, 68, 0.4);
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


def render_kpi_card(label: str, value: str, subtitle: str = "", help_text: str = "", delta: str | None = None, delta_inverted: bool = False) -> None:
    safe_help = help_text.replace("\"", "&quot;") if help_text else ""
    title_attr = f" title=\"{safe_help}\"" if safe_help else ""
    subtitle_html = f"<div class=\"fd-kpi-subtitle\">{subtitle}</div>" if subtitle else ""
    delta_html = ""
    if delta:
        cls = "up" if delta.startswith("+") else "down" if delta.startswith("-") else "neutral"
        if delta_inverted:
            cls = "down" if cls == "up" else "up" if cls == "down" else cls
        delta_html = f'<div class="fd-kpi-delta {cls}">{delta}</div>'
    st.markdown(
        f"""
        <div class="fd-kpi-card"{title_attr}>
            <h3>{label}</h3>
            <div class="fd-kpi-value">{value}</div>
            {subtitle_html}
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ceo_command_panel(
    tab_key: str,
    commands: list[dict],
) -> None:
    """Render a CEO command panel with a list of predefined commands.

    Args:
        tab_key: Unique key prefix for this tab (e.g. "m1", "m2").
        commands: List of dicts with keys:
            - label: Display name of the command
            - recipient: Who receives the command (e.g. "Giám đốc Marketing")
            - next_steps: List of suggested follow-up actions (strings)
    """
    from datetime import datetime

    session_key = f"ceo_cmd_{tab_key}"
    if session_key not in st.session_state:
        st.session_state[session_key] = []

    st.subheader("Gửi chỉ thị")
    st.caption("Chọn lệnh và gửi yêu cầu cho bộ phận phụ trách")

    labels = [c["label"] for c in commands]
    selected_idx = st.selectbox(
        "Chọn chỉ thị",
        range(len(labels)),
        format_func=lambda i: labels[i],
        key=f"{tab_key}_cmd_select",
    )
    selected_cmd = commands[selected_idx]

    st.markdown(f"**Gửi cho:** {selected_cmd['recipient']}")

    note = st.text_input(
        "Ghi chú thêm (tuỳ chọn)",
        placeholder="Ví dụ: Ưu tiên xử lý trong tuần này...",
        key=f"{tab_key}_cmd_note",
    )

    if st.button("Gửi chỉ thị", key=f"{tab_key}_cmd_send", type="primary"):
        entry = {
            "time": datetime.now().strftime("%H:%M %d/%m/%Y"),
            "action": selected_cmd["label"],
            "recipient": selected_cmd["recipient"],
            "note": note,
        }
        st.session_state[session_key].append(entry)
        st.success(f"Đã gửi **\"{selected_cmd['label']}\"** cho {selected_cmd['recipient']}")

        # Show what happens next in reality
        st.markdown("---")
        st.markdown("##### Quy trình sẽ diễn ra sau khi gửi:")
        for i, step in enumerate(selected_cmd["next_steps"], 1):
            st.markdown(f"{i}. {step}")

    # History
    history = st.session_state[session_key]
    if history:
        with st.expander(f"Lịch sử chỉ thị ({len(history)} lệnh)", expanded=False):
            for d in reversed(history[-10:]):
                note_text = f" — *\"{d['note']}\"*" if d.get("note") else ""
                st.markdown(
                    f"• **{d['time']}** — {d['action']} → _{d['recipient']}_{note_text}"
                )


def render_status_pill(label: str, tone: str, help_text: str = "") -> None:
    safe_help = help_text.replace("\"", "&quot;") if help_text else ""
    title_attr = f" title=\"{safe_help}\"" if safe_help else ""
    st.markdown(
        f"<span class=\"fd-pill fd-pill--{tone}\"{title_attr}>{label}</span>",
        unsafe_allow_html=True,
    )