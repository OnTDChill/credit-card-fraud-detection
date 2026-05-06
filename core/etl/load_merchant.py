"""ETL pipeline for seed merchant anomaly detection data (Module 4)."""

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

SEED = 42
np.random.seed(SEED)

SEGMENTS = ["GenZ", "NV Văn phòng", "Kinh doanh", "Hưu trí"]
# MoMo-like scale: ~50K merchant accounts
# Note: DSS table stores aggregated data (6 segments × 3 regions × 12 months = 216 rows)
N_ACCOUNTS = 50_000


def _generate_merchant_seed(year: int = 2024, month: int = 6) -> pd.DataFrame:
    """Generate realistic merchant anomaly seed data."""
    np.random.seed(SEED + year * 12 + month)
    accounts = []

    for i in range(N_ACCOUNTS):
        account_type = "MERCHANT" if np.random.random() < 0.2 else "PERSONAL"
        segment = np.random.choice(SEGMENTS)

        if account_type == "MERCHANT":
            daily_txn_count = int(np.random.normal(80, 30))
            daily_txn_amount = np.random.normal(150_000_000, 50_000_000)
            monthly_volume = daily_txn_amount * 30
            anomaly_score = np.random.beta(2, 5)  # mostly low
        else:
            # Personal accounts: some are "hidden merchants" with high volume
            daily_txn_count = int(np.random.normal(5, 4))
            daily_txn_amount = np.random.normal(2_000_000, 5_000_000)
            monthly_volume = daily_txn_amount * 30
            # High anomaly if volume is huge for a personal account
            if monthly_volume > 100_000_000:
                anomaly_score = np.random.beta(5, 2)
            else:
                anomaly_score = np.random.beta(1, 5)

        # Clamp
        daily_txn_count = max(1, daily_txn_count)
        daily_txn_amount = max(100_000, daily_txn_amount)
        monthly_volume = max(1_000_000, monthly_volume)

        is_suspected = anomaly_score > 0.6
        est_monthly_revenue = monthly_volume * (0.15 if account_type == "MERCHANT" else 0.02)
        est_tax = est_monthly_revenue * 0.10  # assume 10% VAT

        if anomaly_score > 0.75:
            risk_level = "HIGH"
        elif anomaly_score > 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Assign region: North 35%, Central 15%, South 50%
        region_hash = hash(f"ACC{year}{month:02d}{i+1:04d}") % 100
        if region_hash < 35:
            region = "Miền Bắc"
        elif region_hash < 50:
            region = "Miền Trung"
        else:
            region = "Miền Nam"

        accounts.append({
            "year": year,
            "month": month,
            "account_id": f"ACC{year}{month:02d}{i+1:04d}",
            "account_type": account_type,
            "segment": segment,
            "region": region,
            "daily_txn_count": daily_txn_count,
            "daily_txn_amount": daily_txn_amount,
            "monthly_volume": monthly_volume,
            "anomaly_score": anomaly_score,
            "is_suspected_merchant": is_suspected,
            "est_monthly_revenue": est_monthly_revenue,
            "est_tax_collectable": est_tax,
            "risk_level": risk_level,
        })

    df = pd.DataFrame(accounts)
    df["daily_txn_amount"] = df["daily_txn_amount"].round(0)
    df["monthly_volume"] = df["monthly_volume"].round(0)
    df["anomaly_score"] = df["anomaly_score"].round(3)
    df["est_monthly_revenue"] = df["est_monthly_revenue"].round(0)
    df["est_tax_collectable"] = df["est_tax_collectable"].round(0)
    return df


def load_merchant_to_dss(year: int = 2024, month: int = 6) -> dict[str, Any]:
    """Seed merchant anomaly data into dss_merchant_accounts."""
    ensure_dss_tables()

    periods = get_available_periods()
    if periods:
        period_list = periods
    else:
        period_list = [(year, month)]

    frames = [_generate_merchant_seed(year=y, month=m) for y, m in period_list]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return {"success": False, "rows_loaded": 0, "error": "No data generated"}

    conn = sqlite3.connect(_ensure_db_path())
    try:
        conn.executemany(
            """
            INSERT INTO dss_merchant_accounts
            (year, month, account_id, account_type, segment, region,
             daily_txn_count, daily_txn_amount, monthly_volume,
             anomaly_score, is_suspected_merchant, est_monthly_revenue,
             est_tax_collectable, risk_level, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(year, month, account_id, region) DO UPDATE SET
                account_type = excluded.account_type,
                segment = excluded.segment,
                region = excluded.region,
                daily_txn_count = excluded.daily_txn_count,
                daily_txn_amount = excluded.daily_txn_amount,
                monthly_volume = excluded.monthly_volume,
                anomaly_score = excluded.anomaly_score,
                is_suspected_merchant = excluded.is_suspected_merchant,
                est_monthly_revenue = excluded.est_monthly_revenue,
                est_tax_collectable = excluded.est_tax_collectable,
                risk_level = excluded.risk_level,
                updated_at = CURRENT_TIMESTAMP
            """,
            df[[
                "year", "month", "account_id", "account_type", "segment", "region",
                "daily_txn_count", "daily_txn_amount", "monthly_volume",
                "anomaly_score", "is_suspected_merchant", "est_monthly_revenue",
                "est_tax_collectable", "risk_level",
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
    result = load_merchant_to_dss()
    print(f"Merchant ETL: {result}")
