"""
Review Queue - Admin interface for pending transaction review
"""
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from streamlit_app.shared_ui import configure_dashboard_page, render_page_header

configure_dashboard_page("Review Queue")

from core.decision_engine import DecisionEngine

DB_PATH = os.path.join(os.path.dirname(__file__), "../../artifacts/fraud_system.db")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_review_queue(status_filter: str = 'PENDING'):
    try:
        with get_db_connection() as conn:
            df = pd.read_sql("""
                SELECT * FROM review_queue
                WHERE status = ?
                ORDER BY expires_at ASC
            """, conn, params=(status_filter,))

        if not df.empty:
            df['expires_at'] = pd.to_datetime(df['expires_at'])
            df['remaining_minutes'] = (df['expires_at'] - datetime.utcnow()).dt.total_seconds() / 60
            df['remaining_minutes'] = df['remaining_minutes'].clip(lower=0).round(1)

        return df
    except Exception as e:
        st.warning(f"Database not initialized: {str(e)}")
        st.info("Running first time database setup automatically...")
        
        from core.retrain_pipeline import init_database
        init_database()
        st.success("Database tables created successfully")
        st.rerun()


def update_review_status(transaction_id: str, new_status: str, admin_id: str, reason: str):
    with get_db_connection() as conn:
        conn.execute("""
            UPDATE review_queue
            SET status = ?, reviewed_by = ?, reviewed_at = CURRENT_TIMESTAMP, review_reason = ?
            WHERE transaction_id = ?
        """, (new_status, admin_id, reason, transaction_id))

        conn.execute("""
            INSERT OR IGNORE INTO feedback_pool (
                transaction_id, original_decision, original_label, corrected_label, corrected_by, reason
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            transaction_id,
            'REVIEW',
            1 if new_status == 'REJECT' else 0,
            0 if new_status == 'APPROVED' else 1,
            admin_id,
            reason
        ))

        conn.commit()


def main():
    render_page_header(
        "Review Queue",
        "Transactions pending manual review with TTL, reason capture, and audit-ready actions.",
    )

    admin_id = st.sidebar.text_input("Admin ID", value="admin_demo")

    status_filter = st.sidebar.selectbox(
        "Filter Status",
        options=['PENDING', 'APPROVED', 'REJECTED', 'EXPIRED'],
        index=0
    )

    df = load_review_queue(status_filter)

    if df.empty:
        st.info(f"No {status_filter} transactions in review queue")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Pending Review", len(df[df['status'] == 'PENDING']))
    col2.metric("Total Today", len(df))
    col3.metric("Oldest TTL", f"{df['remaining_minutes'].min():.0f} min")

    st.markdown("### Pending Transactions")

    for idx, row in df.iterrows():
        with st.expander(f"Transaction {row['transaction_id']} | Amount: ${row['amount']:.2f} | TTL: {row['remaining_minutes']} min"):
            col_a, col_b = st.columns(2)

            with col_a:
                st.metric("Fraud Probability", f"{row['fraud_probability']:.1%}")
                st.text(f"Customer ID: {row['customer_id']}")
                st.text(f"Merchant ID: {row['merchant_id']}")
                st.text(f"Reason codes: {row['reason_codes']}")

            with col_b:
                st.progress(row['fraud_probability'])

                reason = st.text_input("Review reason", key=f"reason_{row['transaction_id']}")

                btn_col1, btn_col2 = st.columns(2)

                with btn_col1:
                    if st.button("APPROVE", key=f"approve_{row['transaction_id']}", use_container_width=True, type='primary'):
                        if reason:
                            update_review_status(row['transaction_id'], 'APPROVED', admin_id, reason)
                            st.success(f"Transaction {row['transaction_id']} approved")
                            st.rerun()
                        else:
                            st.warning("Please enter reason")

                with btn_col2:
                    if st.button("REJECT", key=f"reject_{row['transaction_id']}", use_container_width=True):
                        if reason:
                            update_review_status(row['transaction_id'], 'REJECTED', admin_id, reason)
                            st.success(f"Transaction {row['transaction_id']} rejected")
                            st.rerun()
                        else:
                            st.warning("Please enter reason")


if __name__ == "__main__":
    main()
