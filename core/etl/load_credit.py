"""ETL pipeline for seed credit portfolio data (Module 2)."""

from __future__ import annotations

import random
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from streamlit_app.components.dss_data_access import ensure_dss_tables, _ensure_db_path, get_available_periods

SEGMENTS = ["GenZ", "Sinh viên", "NV Văn phòng", "Kinh doanh", "Hưu trí", "Tiểu thương"]

# Ensure consistent segment naming
SEGMENT_NORMALIZE = {
    "Sinh vien": "Sinh viên",
    "Sinh viên": "Sinh viên",
}

# Segment parameters: (users_base, credit_limit_mean, interest_mean, npl_mean, default_prob_mean)
# Scaled for MoMo-like fintech: ~2.3M credit users total
# Note: Data aggregated by segment/region/month in DSS tables (~252 rows max)
SEGMENT_PARAMS = {
    "GenZ":          (450_000,  5_000_000,  24.0, 5.5, 0.08),   # 450K users
    "Sinh viên":     (300_000,  3_000_000,  22.0, 4.2, 0.06),   # 300K users
    "NV Văn phòng":  (800_000,  15_000_000, 18.0, 1.5, 0.02),   # 800K users
    "Kinh doanh":    (500_000,  50_000_000, 16.0, 2.8, 0.03),   # 500K users
    "Hưu trí":       (200_000,  8_000_000,  15.0, 0.8, 0.01),   # 200K users
    "Tiểu thương":   (50_000,   80_000_000, 20.0, 3.5, 0.05),   # 50K users (high limit segment)
}


def _generate_credit_seed(year: int = 2024, months: list[int] | None = None) -> pd.DataFrame:
    """Generate realistic credit portfolio seed data."""
    if months is None:
        months = list(range(1, 13))

    rows = []
    for month in months:
        noise = 0.05  # 5% month-to-month variation
        for seg, (users, limit, interest, npl, dprob) in SEGMENT_PARAMS.items():
            np.random.seed(year * 100 + month + hash(seg) % 1000)
            _users = int(users * (1 + np.random.normal(0, noise)))
            _limit = limit * (1 + np.random.normal(0, noise))
            _interest = max(8.0, interest + np.random.normal(0, 1.5))
            _npl = max(0.2, npl + np.random.normal(0, 0.5))
            _dprob = max(0.005, dprob + np.random.normal(0, 0.008))
            _outstanding = _users * _limit * 0.6 * (1 + np.random.normal(0, 0.1))
            _overdue30 = _outstanding * (_npl / 100) * 0.7
            _overdue90 = _outstanding * (_npl / 100) * 0.3
            _rev_interest = _outstanding * (_interest / 100) / 12
            # Assign region based on segment hash
            rh = hash(seg) % 100
            if rh < 35:
                region = "Miền Bắc"
            elif rh < 50:
                region = "Miền Trung"
            else:
                region = "Miền Nam"
            rows.append({
                "year": year,
                "month": month,
                "segment": seg,
                "region": region,
                "total_users": _users,
                "credit_limit": _limit,
                "interest_rate": _interest,
                "npl_rate": _npl,
                "default_probability": _dprob,
                "total_outstanding": _outstanding,
                "overdue_30d_amount": _overdue30,
                "overdue_90d_amount": _overdue90,
                "revenue_interest": _rev_interest,
            })

    df = pd.DataFrame(rows)
    df["credit_limit"] = df["credit_limit"].round(0)
    df["interest_rate"] = df["interest_rate"].round(2)
    df["npl_rate"] = df["npl_rate"].round(2)
    df["default_probability"] = df["default_probability"].round(4)
    df["total_outstanding"] = df["total_outstanding"].round(0)
    df["overdue_30d_amount"] = df["overdue_30d_amount"].round(0)
    df["overdue_90d_amount"] = df["overdue_90d_amount"].round(0)
    df["revenue_interest"] = df["revenue_interest"].round(0)
    return df


def load_credit_to_dss(year: int | None = 2024) -> dict[str, Any]:
    """Seed credit portfolio data into dss_credit_portfolio."""
    ensure_dss_tables()

    periods = get_available_periods()
    if periods:
        years = sorted({y for y, _ in periods})
    else:
        fallback_year = year if year is not None else pd.Timestamp.now().year
        years = [fallback_year]
        periods = [(fallback_year, m) for m in range(1, 13)]

    frames = []
    for y in years:
        months = sorted({m for yy, m in periods if yy == y})
        frames.append(_generate_credit_seed(year=y, months=months))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # Normalize segment names to avoid duplicates
    df["segment"] = df["segment"].replace(SEGMENT_NORMALIZE)

    if df.empty:
        return {"success": False, "rows_loaded": 0, "error": "No data generated"}

    conn = sqlite3.connect(_ensure_db_path())
    try:
        conn.executemany(
            """
            INSERT INTO dss_credit_portfolio
            (year, month, segment, region, total_users, credit_limit, interest_rate,
             npl_rate, default_probability, total_outstanding,
             overdue_30d_amount, overdue_90d_amount, revenue_interest,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(year, month, segment, region) DO UPDATE SET
                total_users = excluded.total_users,
                credit_limit = excluded.credit_limit,
                interest_rate = excluded.interest_rate,
                npl_rate = excluded.npl_rate,
                default_probability = excluded.default_probability,
                total_outstanding = excluded.total_outstanding,
                overdue_30d_amount = excluded.overdue_30d_amount,
                overdue_90d_amount = excluded.overdue_90d_amount,
                revenue_interest = excluded.revenue_interest,
                updated_at = CURRENT_TIMESTAMP
            """,
            df[[
                "year", "month", "segment", "region", "total_users", "credit_limit",
                "interest_rate", "npl_rate", "default_probability",
                "total_outstanding", "overdue_30d_amount", "overdue_90d_amount", "revenue_interest",
            ]].to_records(index=False).tolist(),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "success": True,
        "rows_loaded": len(df),
        "aggregated_records": len(df),
    }


if __name__ == "__main__":
    result = load_credit_to_dss()
    print(f"Credit ETL: {result}")
