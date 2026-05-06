"""Tinh chỉnh ngưỡng và cấu hình hệ thống theo thời gian thực."""

from __future__ import annotations

import json
import sqlite3
from typing import Tuple

import pandas as pd
import streamlit as st

from streamlit_app.config import (
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
    REVIEW_DB_PATH,
    TRAINING_REPORT_PATH,
    FRAUD_ARTIFACTS_DIR,
)
from streamlit_app.shared_ui import configure_dashboard_page, render_page_header

configure_dashboard_page("Tinh chỉnh Hệ thống")


def _load_thresholds() -> Tuple[float, float]:
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


def _save_thresholds(low_t: float, high_t: float, admin_id: str) -> None:
    with sqlite3.connect(REVIEW_DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO system_config (config_key, config_value, updated_by) VALUES (?, ?, ?)",
            ("low_threshold", f"{low_t:.4f}", admin_id),
        )
        conn.execute(
            "INSERT OR REPLACE INTO system_config (config_key, config_value, updated_by) VALUES (?, ?, ?)",
            ("high_threshold", f"{high_t:.4f}", admin_id),
        )
        conn.commit()


def _load_scores() -> pd.Series:
    file_name = "predictions_test.csv"
    if TRAINING_REPORT_PATH.exists():
        try:
            report = json.loads(TRAINING_REPORT_PATH.read_text(encoding="utf-8"))
            file_name = (report.get("prediction_files") or {}).get("test", file_name)
        except json.JSONDecodeError:
            pass

    prediction_path = FRAUD_ARTIFACTS_DIR / file_name
    if not prediction_path.exists():
        return pd.Series(dtype=float)

    df = pd.read_csv(prediction_path)
    if "fraud_probability" not in df.columns:
        return pd.Series(dtype=float)

    return pd.to_numeric(df["fraud_probability"], errors="coerce").fillna(0)


def _estimate_distribution(low_t: float, high_t: float) -> Tuple[float, float, float]:
    scores = _load_scores()
    if scores.empty:
        return 0.0, 0.0, 0.0

    allow = (scores < low_t).mean() * 100
    review = ((scores >= low_t) & (scores < high_t)).mean() * 100
    block = (scores >= high_t).mean() * 100
    return allow, review, block


def main() -> None:
    render_page_header(
        "Tinh chỉnh hệ thống",
        "Điều chỉnh mức độ khắt khe để cân bằng giữa bắt gian lận và trải nghiệm khách hàng.",
        kicker="Trung tâm cấu hình",
    )

    admin_id = st.sidebar.text_input("Mã quản trị", value="admin_demo")
    low_default, high_default = _load_thresholds()

    st.info(
        "Thanh trượt này quyết định hệ thống xử lý giao dịch khắt khe đến mức nào. "
        "Khắt khe hơn: bắt nhiều gian lận hơn nhưng có thể chặn nhầm. "
        "Nới lỏng hơn: ít chặn nhầm hơn nhưng gian lận có thể lọt lưới.",
    )

    col1, col2 = st.columns(2)
    with col1:
        low_t = st.slider(
            "Mức độ khắt khe cho vùng cảnh báo",
            min_value=0.10,
            max_value=0.50,
            step=0.05,
            value=float(st.session_state.get("low_threshold", low_default)),
            help="Giao dịch có xác suất gian lận cao hơn mức này sẽ vào hàng đợi xét duyệt.",
        )
        st.caption(f"Hiện tại: > {low_t * 100:.0f}% xác suất vào hàng đợi")

    with col2:
        high_t = st.slider(
            "Mức độ khắt khe cho tự động chặn",
            min_value=0.50,
            max_value=0.95,
            step=0.05,
            value=float(st.session_state.get("high_threshold", high_default)),
            help="Giao dịch có xác suất gian lận cao hơn mức này sẽ bị chặn tự động.",
        )
        st.caption(f"Hiện tại: > {high_t * 100:.0f}% xác suất bị chặn")

    st.markdown("---")
    st.markdown("**Ước tính tác động nếu áp dụng ngưỡng mới (dựa trên dữ liệu mẫu):**")

    allow_pct, review_pct, block_pct = _estimate_distribution(low_t, high_t)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(
            "Tự động duyệt",
            f"{allow_pct:.1f}%",
            help="Giao dịch được hệ thống tự động cho qua.",
        )
    with col_b:
        st.metric(
            "Cần xét duyệt",
            f"{review_pct:.1f}%",
            help="Giao dịch cần nhân viên kiểm tra thủ công.",
        )
    with col_c:
        st.metric(
            "Tự động chặn",
            f"{block_pct:.1f}%",
            help="Giao dịch bị chặn ngay theo ngưỡng mới.",
        )

    if low_t >= high_t:
        st.warning("Ngưỡng cảnh báo phải thấp hơn ngưỡng chặn để hệ thống hoạt động đúng.")
        return

    if st.button("Lưu cấu hình", type="primary"):
        if not REVIEW_DB_PATH.exists():
            from core.retrain_pipeline import init_database

            init_database()
        _save_thresholds(low_t, high_t, admin_id)
        st.session_state["low_threshold"] = low_t
        st.session_state["high_threshold"] = high_t
        st.success("Đã lưu. Hệ thống áp dụng ngưỡng mới cho giao dịch tiếp theo.")


if __name__ == "__main__":
    main()
