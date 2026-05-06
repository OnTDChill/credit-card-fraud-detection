"""Manual review queue components for fraud decisions."""

from __future__ import annotations

import hashlib
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


def _get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(REVIEW_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_database() -> None:
    if REVIEW_DB_PATH.exists():
        return

    from core.retrain_pipeline import init_database

    init_database()


def _insert_audit_log(
    *,
    transaction_id: str,
    user_id: str,
    event_type: str,
    old_value: str,
    new_value: str,
    reason: str,
) -> None:
    raw = f"{event_type}:{transaction_id}:{user_id}:{new_value}:{reason}"
    event_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    with _get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO audit_log (
                event_type, transaction_id, user_id, old_value, new_value, reason, event_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (event_type, transaction_id, user_id, old_value, new_value, reason, event_hash),
        )
        conn.commit()


def _update_review_status(
    *,
    transaction_id: str,
    new_status: str,
    admin_id: str,
    reason: str,
) -> None:
    with _get_db_connection() as conn:
        conn.execute(
            """
            UPDATE review_queue
            SET status = ?, reviewed_by = ?, reviewed_at = CURRENT_TIMESTAMP, review_reason = ?
            WHERE transaction_id = ?
            """,
            (new_status, admin_id, reason, transaction_id),
        )

        conn.execute(
            """
            INSERT OR IGNORE INTO feedback_pool (
                transaction_id, original_decision, original_label, corrected_label, corrected_by, reason
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                transaction_id,
                "REVIEW",
                1 if new_status == "REJECTED" else 0,
                0 if new_status == "APPROVED" else 1,
                admin_id,
                reason,
            ),
        )
        conn.commit()

    _insert_audit_log(
        transaction_id=transaction_id,
        user_id=admin_id,
        event_type="REVIEW_OVERRIDE",
        old_value="PENDING",
        new_value=new_status,
        reason=reason,
    )


def build_risk_reason(row: dict) -> list[str]:
    """
    Chuyển feature values thành lý do dễ hiểu cho người dùng business.
    Thay thế hiển thị mảng dữ liệu thô V1-V28.
    """
    reasons = []
    if row.get('amount', 0) > 1000:
        reasons.append(f"Số tiền lớn bất thường ({row['amount']:,.0f})")
    if row.get('is_night', False):
        reasons.append("Giao dịch lúc đêm khuya (22h-6h)")
    if row.get('location_changed', False):
        reasons.append("Địa điểm khác với lịch sử")
    if row.get('device_changed', False):
        reasons.append("Thiết bị mới chưa từng sử dụng")
    if row.get('txn_count_1h', 0) > 5:
        reasons.append(f"Nhiều giao dịch liên tiếp ({row['txn_count_1h']:.0f} lần/giờ)")
    
    return reasons if reasons else ["Điểm rủi ro cao theo mô hình AI"]


def render_review_queue(
    df_review: pd.DataFrame,
    *,
    admin_id: str,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> None:
    st.subheader("Giao dịch cần xét duyệt thủ công")
    st.caption(
        f"Các giao dịch có xác suất gian lận từ "
        f"{low_threshold * 100:.0f}% đến {high_threshold * 100:.0f}% "
        f"— hệ thống không thể tự quyết định, cần người xác nhận."
    )

    if df_review.empty:
        st.success("Không có giao dịch nào cần xét duyệt. Hàng đợi trống.")
        return

    for idx, row in df_review.iterrows():
        row_dict = row.to_dict()
        created_at = row_dict.get("received_at") or row_dict.get("created_at")
        if created_at and not isinstance(created_at, datetime):
            created_at = pd.to_datetime(created_at, errors="coerce")

        remaining = None
        now = pd.Timestamp(datetime.now())
        
        if row_dict.get("expires_at") is not None:
            expires_at = pd.to_datetime(row_dict.get("expires_at"), errors="coerce")
            if pd.notna(expires_at):
                remaining = max(0, int((expires_at - now).total_seconds() // 60))
        elif created_at is not None and pd.notna(created_at):
            # Ensure created_at is a Timestamp for subtraction
            ts_created_at = pd.Timestamp(created_at)
            elapsed = int((now - ts_created_at).total_seconds() // 60)
            remaining = max(0, 30 - elapsed)

        amount_value = float(row_dict.get("amount") or 0)
        score_value = float(row_dict.get("fraud_probability") or 0)
        
        st.markdown(
            f"""
            <div style="background:#1e2130;border-left:4px solid #f39c12;
                        padding:14px 18px;border-radius:8px;margin-bottom:10px;">
                <div style="font-weight:700;color:#fff;font-size:1rem;">
                    Giao dịch #{row_dict.get('transaction_id', idx)}
                </div>
                <div style="color:#8b9ab0;font-size:0.85rem;margin-top:4px;">
                    Khách hàng: {row_dict.get('customer_id', 'N/A')} &nbsp;|&nbsp;
                    Số tiền: {CURRENCY_SYMBOL}{amount_value:,.2f} &nbsp;|&nbsp;
                    Xác suất gian lận: {score_value * 100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True
        )

        # Risk profile — ly do bi danh dau (thay vi du lieu tho)
        reasons = build_risk_reason(row_dict)
        if reasons:
            st.caption(f"Bị đánh dấu vì: {' + '.join(reasons)}")

        col_a, col_b, col_c = st.columns([1, 1, 4])
        reason_key = f"reason_{row_dict.get('transaction_id', idx)}"
        with col_c:
            note = st.text_input("Lý do xử lý", key=reason_key, label_visibility="collapsed")

        with col_a:
            if st.button("Duyệt", key=f"approve_{idx}",
                         help="Xác nhận đây là giao dịch hợp lệ"):
                _ensure_database()
                _update_review_status(
                    transaction_id=str(row_dict.get("transaction_id")),
                    new_status="APPROVED",
                    admin_id=admin_id,
                    reason=note or "Phê duyệt thủ công",
                )
                st.rerun()
        with col_b:
            if st.button("Chặn", key=f"block_{idx}",
                         help="Xác nhận đây là giao dịch gian lận"):
                _ensure_database()
                _update_review_status(
                    transaction_id=str(row_dict.get("transaction_id")),
                    new_status="REJECTED",
                    admin_id=admin_id,
                    reason=note or "Chặn thủ công",
                )
                st.rerun()
        with col_c:
            # TTL countdown
            if remaining is not None:
                color = "#e74c3c" if remaining < 5 else "#f39c12"
                st.markdown(
                    f'<span style="color:{color};font-size:0.82rem;">'
                    f'Hết hạn sau: {remaining} phút</span>',
                    unsafe_allow_html=True
                )