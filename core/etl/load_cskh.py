"""ETL pipeline for Customer Support Ticket dataset to dss_customer_service."""

from __future__ import annotations

import random
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from config import DATA_DIR
from streamlit_app.components.dss_data_access import ensure_dss_tables, _ensure_db_path, get_available_periods


# MoMo-like scale: ~40M registered users, ~25M MAU
# Support tickets: ~0.1% of MAU per month = ~25K tickets
# Churn rate: 3-5% annually for fintech
CSKH_PARAMS = {
    "base_users": 40_000_000,      # 40M registered
    "mau_rate": 0.625,              # 62.5% MAU (25M active)
    "ticket_rate": 0.001,           # 0.1% of MAU create tickets
    "base_csat": 3.8,               # Avg satisfaction (1-5)
    "base_churn_rate": 0.04,        # 4% monthly churn (annual ~40%)
    "base_resolution_hours": 18,    # Avg resolution time
}


def _generate_cskh_seed(year: int = 2024, month: int = 6) -> pd.DataFrame:
    """Generate realistic MoMo-like customer service seed data."""
    np.random.seed(year * 100 + month)
    
    # Monthly variation
    mau = int(CSKH_PARAMS["base_users"] * CSKH_PARAMS["mau_rate"] * (1 + np.random.uniform(-0.05, 0.05)))
    ticket_count = int(mau * CSKH_PARAMS["ticket_rate"] * (1 + np.random.uniform(-0.1, 0.1)))
    
    # CSAT varies by season (lower during holidays)
    seasonal_factor = 1.0
    if month in [1, 2]:  # Tết
        seasonal_factor = 0.9
    elif month in [11, 12]:  # Year-end
        seasonal_factor = 0.95
    
    csat = max(1.0, min(5.0, CSKH_PARAMS["base_csat"] * seasonal_factor * (1 + np.random.uniform(-0.05, 0.05))))
    
    # NPS derived from CSAT
    nps = (csat - 3) * 50  # -100 to +100 scale
    
    # Churn rate inversely related to CSAT
    churn_rate = max(0.01, min(0.15, (5 - csat) * 0.03 * (1 + np.random.uniform(-0.2, 0.2))))
    churn_count = int(mau * churn_rate)
    
    # Resolution time varies by ticket volume
    resolution_hours = CSKH_PARAMS["base_resolution_hours"] * (1 + (ticket_count / 25000 - 1) * 0.2)
    resolution_hours = max(4, min(48, resolution_hours * (1 + np.random.uniform(-0.1, 0.1))))
    
    return pd.DataFrame([{
        "year": year,
        "month": month,
        "ticket_count": ticket_count,
        "avg_resolution_hours": round(resolution_hours, 1),
        "csat_score": round(csat, 2),
        "nps_score": round(nps, 1),
        "churn_count": churn_count,
        "churn_rate": round(churn_rate, 4),
    }])


CSKH_PATH = DATA_DIR / "support" / "customer_support_tickets.csv"


def _assign_periods(df: pd.DataFrame, periods: list[tuple[int, int]]) -> None:
    if not periods:
        now = pd.Timestamp.now()
        periods = [(now.year, now.month)]

    period_cycle = [periods[i % len(periods)] for i in range(len(df))]
    fallback_years = pd.Series([p[0] for p in period_cycle], index=df.index)
    fallback_months = pd.Series([p[1] for p in period_cycle], index=df.index)

    if "year" not in df.columns or "month" not in df.columns:
        df["year"] = fallback_years
        df["month"] = fallback_months
        return

    year_vals = pd.to_numeric(df["year"], errors="coerce").fillna(-1).astype(int)
    month_vals = pd.to_numeric(df["month"], errors="coerce").fillna(-1).astype(int)
    valid_set = set(periods)
    valid_mask = pd.Series(
        [(y, m) in valid_set for y, m in zip(year_vals, month_vals)],
        index=df.index,
    )
    invalid_mask = ~valid_mask
    if invalid_mask.any():
        df.loc[invalid_mask, "year"] = fallback_years[invalid_mask]
        df.loc[invalid_mask, "month"] = fallback_months[invalid_mask]


def _insert_cskh_to_db(df: pd.DataFrame) -> dict[str, Any]:
    """Insert CSKH DataFrame directly to DSS database."""
    if df.empty:
        return {"success": False, "rows_loaded": 0, "error": "Empty DataFrame"}
    
    conn = sqlite3.connect(_ensure_db_path())
    try:
        for _, row in df.iterrows():
            conn.execute(
                """
                INSERT INTO dss_customer_service
                (year, month, ticket_count, avg_resolution_hours, csat_score, nps_score, 
                 churn_count, churn_rate, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(year, month) DO UPDATE SET
                    ticket_count = excluded.ticket_count,
                    avg_resolution_hours = excluded.avg_resolution_hours,
                    csat_score = excluded.csat_score,
                    nps_score = excluded.nps_score,
                    churn_count = excluded.churn_count,
                    churn_rate = excluded.churn_rate,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    int(row["year"]),
                    int(row["month"]),
                    int(row["ticket_count"]) if pd.notna(row["ticket_count"]) else 0,
                    float(row["avg_resolution_hours"]) if pd.notna(row["avg_resolution_hours"]) else 24.0,
                    float(row["csat_score"]) if pd.notna(row["csat_score"]) else 3.5,
                    float(row["nps_score"]) if pd.notna(row["nps_score"]) else 30.0,
                    int(row["churn_count"]) if pd.notna(row["churn_count"]) else 0,
                    float(row["churn_rate"]) if pd.notna(row["churn_rate"]) else 0.05,
                ),
            )
        conn.commit()
        return {
            "success": True,
            "rows_loaded": len(df),
            "aggregated_records": len(df),
        }
    finally:
        conn.close()


def load_cskh_to_dss(
    file_path: Path | None = None,
) -> dict[str, Any]:
    """Load Customer Support data into dss_customer_service table.
    
    Args:
        file_path: Path to Support Tickets CSV file
    
    Returns:
        Dict with loading statistics
    """
    ensure_dss_tables()

    file_path = file_path or CSKH_PATH
    
    # If CSV doesn't exist, generate seed data for MoMo-like scale
    if not file_path.exists():
        periods = get_available_periods()
        if not periods:
            periods = [(2024, m) for m in range(1, 13)]
        
        frames = [_generate_cskh_seed(year=y, month=m) for y, m in periods]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        
        if df.empty:
            return {"success": False, "rows_loaded": 0, "error": "No CSKH data generated"}
        
        return _insert_cskh_to_db(df)

    try:
        # Read support data with flexible column detection
        df = pd.read_csv(file_path)
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        
        # Flexible column mapping
        col_mapping = {
            'ticket_id': ['ticket_id', 'id', 'ticket_number', 'ticketid'],
            'date': ['date', 'created_date', 'ticket_date', 'timestamp'],
            'resolution_time': ['time_to_resolution', 'resolution_time', 'resolution_hours', 'time_to_resolve'],
            'csat': ['customer_satisfaction_rating', 'csat', 'satisfaction', 'rating', 'csat_score'],
            'nps': ['nps_score', 'net_promoter_score', 'nps'],
            'loyalty': ['customer_loyalty_score', 'loyalty', 'loyalty_score'],
        }
        
        def find_col(col_options):
            for c in col_options:
                if c in df.columns:
                    return c
            return None
        
        date_col = find_col(col_mapping['date'])
        res_col = find_col(col_mapping['resolution_time'])
        csat_col = find_col(col_mapping['csat'])
        nps_col = find_col(col_mapping['nps'])
        loyalty_col = find_col(col_mapping['loyalty'])
        ticket_col = find_col(col_mapping['ticket_id'])
        
        periods = get_available_periods()

        # Parse date
        if date_col:
            df["parsed_date"] = pd.to_datetime(df[date_col], errors="coerce")
            df["year"] = df["parsed_date"].dt.year
            df["month"] = df["parsed_date"].dt.month
        else:
            df["year"] = pd.NA
            df["month"] = pd.NA

        _assign_periods(df, periods)
        df["year"] = df["year"].astype(int)
        df["month"] = df["month"].astype(int)
        
        # Calculate resolution hours
        if res_col:
            df["resolution_hours"] = pd.to_numeric(df[res_col], errors="coerce")
            df["resolution_hours"] = df["resolution_hours"].fillna(24)
        else:
            df["resolution_hours"] = 24
        
        # CSAT score (1-5)
        if csat_col:
            df["csat"] = pd.to_numeric(df[csat_col], errors="coerce")
            df["csat"] = df["csat"].fillna(3.5)
        else:
            df["csat"] = 3.5
        
        # NPS score
        if nps_col:
            df["nps"] = pd.to_numeric(df[nps_col], errors="coerce")
            df["nps"] = df["nps"].fillna(30)
        elif loyalty_col:
            df["loyalty"] = pd.to_numeric(df[loyalty_col], errors="coerce")
            df["nps"] = (df["loyalty"] - 5) * 20
        else:
            df["nps"] = (df["csat"] - 3) * 50
        
        # Churn proxy
        df["churn"] = df["csat"] <= 2
        
        # Use ticket_id or index for counting
        count_col = ticket_col if ticket_col else df.index
        
        # Aggregate by year, month
        agg_dict = {
            "resolution_hours": "mean",
            "csat": "mean",
            "nps": "mean",
            "churn": ["sum", "mean"],
        }
        
        if ticket_col:
            agg_dict[ticket_col] = "count"
        
        grouped = df.groupby(["year", "month"]).agg(agg_dict).reset_index()
        
        # Flatten multi-level columns
        if isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = [' '.join(col).strip() if col[1] not in ['nan', 'NaN'] else col[0] 
                              for col in grouped.columns.values]
        
        # Rename columns
        col_rename = {
            'resolution_hours mean': 'avg_resolution_hours',
            'csat mean': 'csat_score',
            'nps mean': 'nps_score',
            'churn sum': 'churn_count',
            'churn mean': 'churn_rate',
        }
        if ticket_col:
            col_rename[f'{ticket_col} count'] = 'ticket_count'
        
        grouped = grouped.rename(columns=col_rename)
        
        # Ensure required columns exist
        for col, default in [('ticket_count', len(df)), ('avg_resolution_hours', 24), 
                             ('csat_score', 3.5), ('nps_score', 30), ('churn_count', 0), ('churn_rate', 0)]:
            if col not in grouped.columns:
                grouped[col] = default

        # Insert into database
        conn = sqlite3.connect(_ensure_db_path())
        try:
            conn.executemany(
                """
                INSERT INTO dss_customer_service
                (year, month, ticket_count, avg_resolution_hours, csat_score, nps_score, 
                 churn_count, churn_rate, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(year, month) DO UPDATE SET
                    ticket_count = excluded.ticket_count,
                    avg_resolution_hours = excluded.avg_resolution_hours,
                    csat_score = excluded.csat_score,
                    nps_score = excluded.nps_score,
                    churn_count = excluded.churn_count,
                    churn_rate = excluded.churn_rate,
                    updated_at = CURRENT_TIMESTAMP
                """,
                [
                    (
                        int(row["year"]),
                        int(row["month"]),
                        int(row["ticket_count"]) if pd.notna(row["ticket_count"]) else 0,
                        float(row["avg_resolution_hours"]) if pd.notna(row["avg_resolution_hours"]) else 24.0,
                        float(row["csat_score"]) if pd.notna(row["csat_score"]) else 3.5,
                        float(row["nps_score"]) if pd.notna(row["nps_score"]) else 30.0,
                        int(row["churn_count"]) if pd.notna(row["churn_count"]) else 0,
                        float(row["churn_rate"]) if pd.notna(row["churn_rate"]) else 0.05,
                    )
                    for _, row in grouped.iterrows()
                ],
            )
            conn.commit()
        finally:
            conn.close()

        return {
            "success": True,
            "rows_loaded": len(df),
            "aggregated_records": len(grouped),
        }

    except Exception as e:
        return {
            "success": False,
            "rows_loaded": 0,
            "error": str(e),
        }


if __name__ == "__main__":
    result = load_cskh_to_dss()
    print(f"CSKH ETL: {result}")
