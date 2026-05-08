"""Seed multi-month data for DSS dashboard.

Extends the existing single-month (2024/01) data to 6 months (2024/01-06)
by applying realistic month-over-month variation. Idempotent: skips
months that already exist.

Usage:
    python -m core.etl.seed_multi_month
"""

from __future__ import annotations

import random
import sqlite3
import sys
from pathlib import Path

# Ensure project root on path
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import config  # noqa: E402

DB_PATH = config.REVIEW_DB_PATH
MONTHS_TO_SEED = [(2024, m) for m in range(2, 7)]  # Feb-Jun 2024

random.seed(42)


def _jitter(base: float, pct: float = 0.05) -> float:
    """Apply random jitter of +/- pct to base value."""
    return base * (1 + random.uniform(-pct, pct))


def _exists(conn: sqlite3.Connection, table: str, year: int, month: int) -> bool:
    row = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE year=? AND month=?", (year, month)
    ).fetchone()
    return row[0] > 0


def seed_transaction_summary(conn: sqlite3.Connection) -> int:
    """Seed dss_transaction_summary for months 2-6 based on month-1 data."""
    base_rows = conn.execute(
        "SELECT * FROM dss_transaction_summary WHERE year=2024 AND month=1"
    ).fetchall()
    if not base_rows:
        return 0
    cols = [d[0] for d in conn.execute(
        "SELECT * FROM dss_transaction_summary LIMIT 1"
    ).description]

    inserted = 0
    for year, month in MONTHS_TO_SEED:
        if _exists(conn, "dss_transaction_summary", year, month):
            continue
        # Monthly growth factor: 3-6% growth with slight seasonality
        growth = 1.0 + 0.03 + (month - 1) * 0.005 + random.uniform(-0.01, 0.01)
        cumulative = growth ** (month - 1)

        for row in base_rows:
            d = dict(zip(cols, row))
            new_count = int(d["total_count"] * cumulative * _jitter(1.0, 0.03))
            new_amount = d["total_amount"] * cumulative * _jitter(1.0, 0.04)
            new_revenue = d["revenue_estimated"] * cumulative * _jitter(1.0, 0.03)
            # Fraud rate fluctuates slightly
            fraud_rate_factor = _jitter(1.0, 0.15)
            new_fraud_count = max(0, int(d["fraud_count"] * cumulative * fraud_rate_factor))
            new_fraud_amount = d["fraud_amount"] * cumulative * fraud_rate_factor if d["fraud_amount"] > 0 else 0
            new_fraud_rate = new_fraud_count / new_count if new_count > 0 else 0

            conn.execute("""
                INSERT OR IGNORE INTO dss_transaction_summary
                (year, month, transaction_type, region, total_count, total_amount,
                 revenue_estimated, fraud_count, fraud_amount, fraud_rate, avg_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (year, month, d["transaction_type"], d["region"],
                  new_count, new_amount, new_revenue,
                  new_fraud_count, new_fraud_amount, new_fraud_rate,
                  _jitter(d["avg_score"] or 1.0, 0.01)))
            inserted += 1
    return inserted


def seed_marketing_monthly(conn: sqlite3.Connection) -> int:
    """Seed dss_marketing_monthly for months 2-6."""
    base_rows = conn.execute(
        "SELECT * FROM dss_marketing_monthly WHERE year=2024 AND month=1"
    ).fetchall()
    if not base_rows:
        return 0
    cols = [d[0] for d in conn.execute(
        "SELECT * FROM dss_marketing_monthly LIMIT 1"
    ).description]

    inserted = 0
    for year, month in MONTHS_TO_SEED:
        if _exists(conn, "dss_marketing_monthly", year, month):
            continue
        # Marketing spend varies +/-8% month-over-month
        spend_factor = _jitter(1.0 + (month - 1) * 0.02, 0.06)
        # Conversion improves slightly over time (learning effect)
        conv_improvement = 1.0 + (month - 1) * 0.005

        for row in base_rows:
            d = dict(zip(cols, row))
            new_spend = d["campaign_spend"] * spend_factor
            new_conv = min(0.15, d["avg_conversion"] * conv_improvement * _jitter(1.0, 0.05))
            new_impressions = int(d["total_impressions"] * _jitter(1.0 + (month - 1) * 0.02, 0.04))
            new_clicks = int(d["total_clicks"] * _jitter(1.0 + (month - 1) * 0.015, 0.04))
            new_cac = new_spend / max(1, new_impressions * new_conv) if new_conv > 0 else d["avg_cac"]
            new_roi = _jitter(d["avg_roi"], 0.08)

            conn.execute("""
                INSERT OR IGNORE INTO dss_marketing_monthly
                (year, month, channel, campaign_type, campaign_spend, avg_cac, avg_roi,
                 avg_conversion, total_clicks, total_impressions, ltv_estimated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (year, month, d["channel"], d["campaign_type"],
                  new_spend, new_cac, new_roi, new_conv,
                  new_clicks, new_impressions, _jitter(d["ltv_estimated"] or 13000, 0.05)))
            inserted += 1
    return inserted


def seed_customer_service(conn: sqlite3.Connection) -> int:
    """Seed dss_customer_service for months 2-6."""
    base = conn.execute(
        "SELECT * FROM dss_customer_service WHERE year=2024 AND month=1"
    ).fetchone()
    if not base:
        return 0
    cols = [d[0] for d in conn.execute(
        "SELECT * FROM dss_customer_service LIMIT 1"
    ).description]
    d = dict(zip(cols, base))

    inserted = 0
    for year, month in MONTHS_TO_SEED:
        if _exists(conn, "dss_customer_service", year, month):
            continue
        # Churn rate gradually improves (fintech maturation)
        churn_decay = 0.95 ** (month - 1)
        new_churn_rate = max(0.02, d["churn_rate"] * churn_decay * _jitter(1.0, 0.1))
        # CSAT improves slightly
        new_csat = min(5.0, d["csat_score"] * (1 + (month - 1) * 0.01) * _jitter(1.0, 0.02))
        new_nps = min(80, d["nps_score"] * (1 + (month - 1) * 0.015) * _jitter(1.0, 0.03))
        # Ticket count grows with user base
        new_tickets = int(d["ticket_count"] * (1 + (month - 1) * 0.03) * _jitter(1.0, 0.05))
        new_churn_count = int(d["churn_count"] * churn_decay * _jitter(1.0, 0.08))
        new_resolution = max(4, d["avg_resolution_hours"] * _jitter(0.97 ** (month - 1), 0.05))

        conn.execute("""
            INSERT OR IGNORE INTO dss_customer_service
            (year, month, ticket_count, avg_resolution_hours, csat_score,
             nps_score, churn_count, churn_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (year, month, new_tickets, new_resolution,
              new_csat, new_nps, new_churn_count, new_churn_rate))
        inserted += 1
    return inserted


def seed_credit_portfolio(conn: sqlite3.Connection) -> int:
    """Seed dss_credit_portfolio for months 2-6."""
    base_rows = conn.execute(
        "SELECT * FROM dss_credit_portfolio WHERE year=2024 AND month=1"
    ).fetchall()
    if not base_rows:
        return 0
    cols = [d[0] for d in conn.execute(
        "SELECT * FROM dss_credit_portfolio LIMIT 1"
    ).description]

    inserted = 0
    for year, month in MONTHS_TO_SEED:
        if _exists(conn, "dss_credit_portfolio", year, month):
            continue
        user_growth = 1.0 + (month - 1) * 0.025  # 2.5% monthly user growth

        for row in base_rows:
            d = dict(zip(cols, row))
            new_users = int(d["total_users"] * user_growth * _jitter(1.0, 0.03))
            new_outstanding = d["total_outstanding"] * user_growth * _jitter(1.0, 0.04)
            # NPL drifts slightly
            npl_drift = _jitter(1.0, 0.08)
            new_npl = max(0.3, min(8.0, d["npl_rate"] * npl_drift))
            new_overdue_30 = d["overdue_30d_amount"] * user_growth * _jitter(1.0, 0.06)
            new_overdue_90 = d["overdue_90d_amount"] * user_growth * _jitter(1.0, 0.06)
            new_revenue = d["revenue_interest"] * user_growth * _jitter(1.0, 0.04)

            conn.execute("""
                INSERT OR IGNORE INTO dss_credit_portfolio
                (year, month, segment, region, total_users, credit_limit,
                 interest_rate, npl_rate, default_probability,
                 total_outstanding, overdue_30d_amount, overdue_90d_amount,
                 revenue_interest)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (year, month, d["segment"], d["region"],
                  new_users, d["credit_limit"],
                  d["interest_rate"], new_npl, _jitter(d["default_probability"], 0.1),
                  new_outstanding, new_overdue_30, new_overdue_90, new_revenue))
            inserted += 1
    return inserted


def seed_service_ecosystem(conn: sqlite3.Connection) -> int:
    """Seed dss_service_ecosystem for months 2-6."""
    base_rows = conn.execute(
        "SELECT * FROM dss_service_ecosystem WHERE year=2024 AND month=1"
    ).fetchall()
    if not base_rows:
        return 0
    cols = [d[0] for d in conn.execute(
        "SELECT * FROM dss_service_ecosystem LIMIT 1"
    ).description]

    inserted = 0
    for year, month in MONTHS_TO_SEED:
        if _exists(conn, "dss_service_ecosystem", year, month):
            continue
        growth = 1.0 + (month - 1) * 0.02

        for row in base_rows:
            d = dict(zip(cols, row))
            new_support = int(d["support_count"] * growth * _jitter(1.0, 0.04))
            new_rev_a = d["revenue_a"] * growth * _jitter(1.0, 0.05)
            new_rev_b = d["revenue_b"] * growth * _jitter(1.0, 0.05)

            conn.execute("""
                INSERT OR IGNORE INTO dss_service_ecosystem
                (year, month, service_a, service_b, support_count, support_pct,
                 confidence, lift, revenue_a, revenue_b,
                 profit_margin_a, profit_margin_b)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (year, month, d["service_a"], d["service_b"],
                  new_support, _jitter(d["support_pct"], 0.03),
                  _jitter(d["confidence"], 0.05), _jitter(d["lift"], 0.03),
                  new_rev_a, new_rev_b,
                  d["profit_margin_a"], d["profit_margin_b"]))
            inserted += 1
    return inserted


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        n1 = seed_transaction_summary(conn)
        n2 = seed_marketing_monthly(conn)
        n3 = seed_customer_service(conn)
        n4 = seed_credit_portfolio(conn)
        n5 = seed_service_ecosystem(conn)
        conn.commit()
        print(f"Seeded: txn={n1}, mkt={n2}, cskh={n3}, credit={n4}, eco={n5}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()