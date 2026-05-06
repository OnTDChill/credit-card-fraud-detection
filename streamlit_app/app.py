"""Fraud Detection - unified Streamlit entry point."""

import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

# Ensure project root is on sys.path BEFORE any local imports
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

import streamlit_app.config as st_cfg
from streamlit_app.config import REVIEW_DB_PATH
from streamlit_app.components.status_badge import render_status_badge
from streamlit_app.shared_ui import configure_dashboard_page

configure_dashboard_page("Trung tâm vận hành gian lận")


def _load_available_periods() -> list[tuple[int, int]]:
    """Return sorted (year, month) tuples available in DB."""
    if not REVIEW_DB_PATH.exists():
        return []

    try:
        with sqlite3.connect(REVIEW_DB_PATH) as conn:
            rows = conn.execute(
                "SELECT DISTINCT year, month FROM dss_transaction_summary ORDER BY year, month"
            ).fetchall()
        return [(int(r[0]), int(r[1])) for r in rows if r[0] and r[1]]
    except Exception:
        return []


def _load_data_status() -> Tuple[str, str, str]:
    if not REVIEW_DB_PATH.exists():
        return "Chưa có dữ liệu", "warning", "Chưa chạy ETL pipeline"

    try:
        with sqlite3.connect(REVIEW_DB_PATH) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM dss_transaction_summary"
            ).fetchone()
            count = row[0] if row else 0
    except Exception:
        count = 0

    if count > 0:
        return "Sẵn sàng", "success", f"Đã có {count} bản ghi giao dịch"
    return "Chưa có dữ liệu", "warning", "Chưa chạy ETL pipeline"


with st.sidebar:
    st.header("Điều hướng")
    role = st.radio("Vai trò người dùng", ["Kinh doanh", "Kỹ thuật"], horizontal=True)
    st.session_state["user_role"] = role

    st.subheader("Trạng thái hệ thống")
    status_label, status_tone, status_hint = _load_data_status()
    render_status_badge(status_label, status_tone, status_hint)
    st.caption(status_hint)

    st.subheader("Kỳ báo cáo")
    periods = _load_available_periods()
    year_options = sorted({p[0] for p in periods}, reverse=True) if periods else []
    year_choices = ["Tất cả"] + [str(y) for y in year_options]
    sel_year = st.selectbox("Năm", year_choices, index=0, key="_sb_year")

    if sel_year != "Tất cả":
        month_options = sorted({p[1] for p in periods if p[0] == int(sel_year)})
        month_choices = ["Tất cả"] + [f"Tháng {m}" for m in month_options]
    else:
        month_choices = ["Tất cả"]
    sel_month = st.selectbox("Tháng", month_choices, index=0, key="_sb_month")

    st.session_state["filter_year"] = int(sel_year) if sel_year != "Tất cả" else None
    st.session_state["filter_month"] = int(sel_month.replace("Tháng ", "")) if sel_month != "Tất cả" else None


# Define all pages - organized by role
ceo_pages = [
    st.Page("pages/ceo/00_Tong_quan_CEO.py", title="Tổng quan", icon=":material/home:"),
    st.Page("pages/ceo/01_Thu_hut_Kich_hoat.py", title="1. Khách hàng", icon=":material/ads_click:"),
    st.Page("pages/ceo/02_Tin_dung.py", title="2. Tín dụng", icon=":material/credit_card:"),
    st.Page("pages/ceo/03_He_sinh_thai.py", title="3. Dịch vụ", icon=":material/hub:"),
    st.Page("pages/ceo/04_Merchant.py", title="4. Đối tác", icon=":material/store:"),
    st.Page("pages/ceo/05_Giao_dich_An_toan.py", title="5. Giao dịch & An toàn", icon=":material/security:"),
]

tech_pages = [
    st.Page("pages/tech/02_Hang_doi_Xet_duyet.py", title="Xét duyệt", icon=":material/checklist:"),
    st.Page("pages/tech/03_Tinh_chinh_He_thong.py", title="Tinh chỉnh", icon=":material/settings:"),
    st.Page("pages/tech/04_Phan_tich_Ky_thuat.py", title="Kỹ thuật", icon=":material/build:"),
]

# Show pages based on role
if role == "Kỹ thuật":
    all_pages = ceo_pages + tech_pages
else:
    all_pages = ceo_pages

pg = st.navigation(all_pages)
pg.run()