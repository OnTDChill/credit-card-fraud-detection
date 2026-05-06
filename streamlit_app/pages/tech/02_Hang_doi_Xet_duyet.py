"""Hàng đợi xét duyệt thủ công cho giao dịch nghi ngờ."""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st

from streamlit_app.config import (
    CURRENCY_SYMBOL,
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
    REVIEW_DB_PATH,
    REVIEW_TTL_MINUTES,
)

from streamlit_app.components.review_queue import render_review_queue


def _get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(REVIEW_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_database() -> None:
    if REVIEW_DB_PATH.exists():
        return

    from core.retrain_pipeline import init_database

    init_database()


def _load_thresholds() -> tuple[float, float]:
    if not REVIEW_DB_PATH.exists():
        return DEFAULT_LOW_THRESHOLD, DEFAULT_HIGH_THRESHOLD

    with sqlite3.connect(REVIEW_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT config_key, config_value FROM system_config WHERE config_key IN ('low_threshold','high_threshold')"
        ).fetchall()

    values = {row[0]: row[1] for row in rows}
    low_t = float(values.get("low_threshold", DEFAULT_LOW_THRESHOLD))
    high_t = float(values.get("high_threshold", DEFAULT_HIGH_THRESHOLD))
    return low_t, high_t


def _load_review_queue(status_filter: str = "PENDING") -> pd.DataFrame:
    if not REVIEW_DB_PATH.exists():
        return pd.DataFrame()

    with sqlite3.connect(REVIEW_DB_PATH) as conn:
        df = pd.read_sql(
            """
            SELECT * FROM review_queue
            WHERE status = ?
            ORDER BY expires_at ASC
            """,
            conn,
            params=(status_filter,),
        )

    if not df.empty:
        df["expires_at"] = pd.to_datetime(df["expires_at"], errors="coerce")
        df["received_at"] = pd.to_datetime(df["received_at"], errors="coerce")

    return df


def main() -> None:
    st.set_page_config(page_title="Hàng đợi Xét duyệt", layout="wide")
    st.title("Hàng đợi Xét duyệt")
    st.caption("Các giao dịch ở vùng lưỡng lự cần xác nhận thủ công trước khi quyết định.")

    admin_id = st.sidebar.text_input("Mã nhân viên", value="ops_demo")
    low_t, high_t = _load_thresholds()

    # Show threshold info
    st.info(f"**Ngưỡng xét duyệt:** {low_t*100:.1f}% - {high_t*100:.1f}% (xác suất gian lận)")

    # Check database exists
    if not REVIEW_DB_PATH.exists():
        st.warning("Cơ sở dữ liệu xét duyệt chưa được khởi tạo.")
        if st.button("Khởi tạo Database", type="primary"):
            _ensure_database()
            st.success("Đã khởi tạo database. Hãy tải lại trang.")
            st.stop()

    review_df = _load_review_queue()

    if not review_df.empty and "fraud_probability" in review_df.columns:
        review_df["fraud_probability"] = pd.to_numeric(
            review_df["fraud_probability"], errors="coerce"
        ).fillna(0)
        review_df = review_df[review_df["fraud_probability"].between(low_t, high_t)]

    # Render queue with empty state message
    render_review_queue(
        review_df,
        admin_id=admin_id,
        low_threshold=low_t,
        high_threshold=high_t,
    )

    # Show help info when empty
    if review_df.empty:
        st.markdown("---")
        st.subheader("Hướng dẫn")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Giao dịch vào hàng đợi khi:**
            - Xác suất gian lận nằm trong khoảng ngưỡng
            - Hệ thống ML không đủ tự tin để quyết định tự động
            - Cần xác nhận thủ công từ nhân viên vận hành
            """)
        with col2:
            st.markdown("""
            **Các bước xử lý:**
            1. Kiểm tra thông tin giao dịch
            2. Đánh giá các yếu tố rủi ro
            3. Quyết định: Hợp lệ / Gian lận
            4. Giao dịch sẽ được cập nhật trạng thái
            """)


if __name__ == "__main__":
    main()