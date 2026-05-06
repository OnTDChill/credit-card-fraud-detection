"""ETL pipeline for seed cross-sell ecosystem data (Module 3)."""

from __future__ import annotations

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

SERVICES = [
    "Nạp tiền ĐT", "Thanh toán Hóa đơn", "Nạp tiền viện phí",
    "Thanh toán Viện phí", "Thanh toán Hoàn", "Hành lí",
    "Gọi xe", "Gói xe",
]

# Known profitable cross-sell pairs with realistic metrics
CROSSSELL_PAIRS = [
    # (service_a, service_b, support_pct, confidence, profit_a, profit_b)
    ("Vé CGV", "Highlands Coffee", 35.0, 0.55, 0.15, 0.35),
    ("Nạp tiền ĐT", "Thanh toán Hóa đơn", 48.0, 0.62, 0.08, 0.12),
    ("Vé CGV", "Nạp tiền ĐT", 28.0, 0.40, 0.15, 0.08),
    ("Highlands Coffee", "Thanh toán Viện phí", 15.0, 0.30, 0.35, 0.10),
    ("Gọi xe", "Thanh toán Hóa đơn", 22.0, 0.35, -0.25, 0.12),
    ("Gói xe", "Nạp tiền ĐT", 18.0, 0.28, -0.15, 0.08),
    ("Nạp tiền viện phí", "Thanh toán Hoàn", 12.0, 0.25, 0.05, 0.18),
    ("Hành lí", "Thanh toán Viện phí", 10.0, 0.20, 0.20, 0.10),
    ("Vé CGV", "Gọi xe", 20.0, 0.32, 0.15, -0.25),
    ("Nạp tiền ĐT", "Gói xe", 14.0, 0.22, 0.08, -0.15),
    ("Highlands Coffee", "Nạp tiền viện phí", 16.0, 0.26, 0.35, 0.05),
    ("Thanh toán Hóa đơn", "Hành lí", 9.0, 0.18, 0.12, 0.20),
    ("Vé CGV", "Nạp tiền viện phí", 8.0, 0.15, 0.15, 0.05),
    ("Gọi xe", "Highlands Coffee", 11.0, 0.20, -0.25, 0.35),
    ("Gói xe", "Thanh toán Hoàn", 7.0, 0.14, -0.15, 0.18),
]

# Base revenue for each service (monthly VND)
BASE_REVENUE = {
    "Nạp tiền ĐT": 25_000_000_000,
    "Thanh toán Hóa đơn": 12_000_000_000,
    "Nạp tiền viện phí": 3_000_000_000,
    "Thanh toán Viện phí": 2_000_000_000,
    "Thanh toán Hoàn": 1_500_000_000,
    "Hành lí": 800_000_000,
    "Gọi xe": 5_000_000_000,
    "Gói xe": 1_000_000_000,
    "Vé CGV": 600_000_000,
    "Highlands Coffee": 400_000_000,
}

SEED = 2024
np.random.seed(SEED)


def _generate_ecosystem_seed(year: int = 2024, month: int = 6) -> pd.DataFrame:
    """Generate realistic cross-sell association rules seed data."""
    rows = []
    total_users = 500_000

    for sa, sb, sp, conf, pa, pb in CROSSSELL_PAIRS:
        # Add month-to-month noise
        np.random.seed(SEED + hash(sa + sb) % 10000)
        _support_pct = max(1.0, sp + np.random.normal(0, sp * 0.1))
        _confidence = min(0.95, max(0.05, conf + np.random.normal(0, 0.03)))
        _lift = _confidence / (_support_pct / 100)
        _support_count = int(total_users * (_support_pct / 100))
        _rev_a = BASE_REVENUE.get(sa, 1_000_000_000) * (1 + np.random.normal(0, 0.05))
        _rev_b = BASE_REVENUE.get(sb, 1_000_000_000) * (1 + np.random.normal(0, 0.05))

        rows.append({
            "year": year,
            "month": month,
            "service_a": sa,
            "service_b": sb,
            "support_count": _support_count,
            "support_pct": _support_pct,
            "confidence": _confidence,
            "lift": _lift,
            "revenue_a": _rev_a,
            "revenue_b": _rev_b,
            "profit_margin_a": pa,
            "profit_margin_b": pb,
        })

    df = pd.DataFrame(rows)
    df["support_pct"] = df["support_pct"].round(2)
    df["confidence"] = df["confidence"].round(3)
    df["lift"] = df["lift"].round(3)
    df["revenue_a"] = df["revenue_a"].round(0)
    df["revenue_b"] = df["revenue_b"].round(0)
    df["profit_margin_a"] = df["profit_margin_a"].round(2)
    df["profit_margin_b"] = df["profit_margin_b"].round(2)
    return df


def load_ecosystem_to_dss(year: int = 2024, month: int = 6) -> dict[str, Any]:
    """Seed cross-sell ecosystem data into dss_service_ecosystem."""
    ensure_dss_tables()

    periods = get_available_periods()
    if periods:
        period_list = periods
    else:
        period_list = [(year, month)]

    frames = [_generate_ecosystem_seed(year=y, month=m) for y, m in period_list]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return {"success": False, "rows_loaded": 0, "error": "No data generated"}

    conn = sqlite3.connect(_ensure_db_path())
    try:
        conn.executemany(
            """
            INSERT INTO dss_service_ecosystem
            (year, month, service_a, service_b, support_count, support_pct,
             confidence, lift, revenue_a, revenue_b, profit_margin_a, profit_margin_b,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(year, month, service_a, service_b) DO UPDATE SET
                support_count = excluded.support_count,
                support_pct = excluded.support_pct,
                confidence = excluded.confidence,
                lift = excluded.lift,
                revenue_a = excluded.revenue_a,
                revenue_b = excluded.revenue_b,
                profit_margin_a = excluded.profit_margin_a,
                profit_margin_b = excluded.profit_margin_b,
                updated_at = CURRENT_TIMESTAMP
            """,
            df[[
                "year", "month", "service_a", "service_b", "support_count",
                "support_pct", "confidence", "lift",
                "revenue_a", "revenue_b", "profit_margin_a", "profit_margin_b",
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
    result = load_ecosystem_to_dss()
    print(f"Ecosystem ETL: {result}")
