"""ETL pipeline for Marketing Campaign dataset to dss_marketing_monthly."""

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


# MoMo-like scale: ~100K monthly acquisitions across all channels
# Source: Industry benchmarks for fintech marketing in Vietnam
CHANNEL_PARAMS = {
    # channel: (impressions, conversion_rate, avg_cac)
    "Mạng xã hội": (50_000_000, 0.025, 45_000),    # 50M impressions, 2.5% conv, 45K CAC
    "TikTok":      (80_000_000, 0.030, 80_000),    # 80M impressions, 3% conv, 80K CAC
    "YouTube":     (30_000_000, 0.020, 55_000),    # 30M impressions, 2% conv, 55K CAC
    "Email":       (5_000_000, 0.008, 25_000),     # 5M impressions, 0.8% conv, 25K CAC
    "SMS":         (2_000_000, 0.012, 35_000),     # 2M impressions, 1.2% conv, 35K CAC
    "Affiliate":   (10_000_000, 0.035, 40_000),    # 10M impressions, 3.5% conv, 40K CAC
    "Referral":    (500_000, 0.25, 15_000),        # 500K impressions, 25% conv, 15K CAC (best)
}


def _generate_marketing_seed(year: int = 2024, month: int = 6) -> pd.DataFrame:
    """Generate realistic MoMo-like marketing campaign seed data."""
    rows = []
    
    for channel, (base_impr, base_conv, base_cac) in CHANNEL_PARAMS.items():
        # Add month-to-month variation (±10%)
        np.random.seed(year * 100 + month + hash(channel) % 1000)
        
        impressions = int(base_impr * (1 + np.random.uniform(-0.1, 0.1)))
        conversion_rate = max(0.005, min(0.5, base_conv * (1 + np.random.uniform(-0.15, 0.15))))
        avg_cac = max(10000, base_cac * (1 + np.random.uniform(-0.1, 0.1)))
        
        # Calculate derived metrics
        acquisitions = int(impressions * conversion_rate)
        clicks = int(impressions * 0.03)  # ~3% CTR industry avg
        campaign_spend = acquisitions * avg_cac
        avg_roi = np.random.uniform(150, 400)  # 150-400% ROI
        ltv_estimated = avg_cac * (1 + avg_roi / 100)
        
        rows.append({
            "year": year,
            "month": month,
            "channel": channel,
            "campaign_type": "General",
            "total_impressions": impressions,
            "total_clicks": clicks,
            "avg_conversion": conversion_rate,
            "avg_cac": avg_cac,
            "campaign_spend": campaign_spend,
            "avg_roi": avg_roi,
            "ltv_estimated": ltv_estimated,
        })
    
    return pd.DataFrame(rows)


MARKETING_PATH = DATA_DIR / "marketing" / "marketing_campaign.csv"


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
        df.loc[invalid_mask, "year"] = fallback_years[invalid_mask].astype("int32")
        df.loc[invalid_mask, "month"] = fallback_months[invalid_mask].astype("int32")


def _insert_marketing_to_db(df: pd.DataFrame) -> dict[str, Any]:
    """Insert marketing DataFrame directly to DSS database."""
    if df.empty:
        return {"success": False, "rows_loaded": 0, "error": "Empty DataFrame"}
    
    conn = sqlite3.connect(_ensure_db_path())
    try:
        for _, row in df.iterrows():
            conn.execute(
                """
                INSERT INTO dss_marketing_monthly
                (year, month, channel, campaign_type, campaign_spend, avg_cac, avg_roi, 
                 avg_conversion, total_clicks, total_impressions, ltv_estimated, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(year, month, channel) DO UPDATE SET
                    campaign_spend = excluded.campaign_spend,
                    avg_cac = excluded.avg_cac,
                    avg_roi = excluded.avg_roi,
                    avg_conversion = excluded.avg_conversion,
                    total_clicks = excluded.total_clicks,
                    total_impressions = excluded.total_impressions,
                    ltv_estimated = excluded.ltv_estimated,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    int(row["year"]),
                    int(row["month"]),
                    str(row["channel"]),
                    str(row.get("campaign_type", "General")),
                    float(row["campaign_spend"]) if pd.notna(row["campaign_spend"]) else 0.0,
                    float(row["avg_cac"]) if pd.notna(row["avg_cac"]) else 0.0,
                    float(row["avg_roi"]) if pd.notna(row["avg_roi"]) else 0.0,
                    float(row["avg_conversion"]) if pd.notna(row["avg_conversion"]) else 0.0,
                    int(row["total_clicks"]) if pd.notna(row["total_clicks"]) else 0,
                    int(row["total_impressions"]) if pd.notna(row["total_impressions"]) else 0,
                    float(row["ltv_estimated"]) if pd.notna(row["ltv_estimated"]) else 0.0,
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


def load_marketing_to_dss(
    file_path: Path | None = None,
) -> dict[str, Any]:
    """Load Marketing Campaign data into dss_marketing_monthly table.
    
    Args:
        file_path: Path to Marketing CSV file
    
    Returns:
        Dict with loading statistics
    """
    ensure_dss_tables()

    file_path = file_path or MARKETING_PATH
    
    # If CSV doesn't exist, generate seed data for MoMo-like scale
    if not file_path.exists():
        periods = get_available_periods()
        if not periods:
            periods = [(2024, m) for m in range(1, 13)]
        
        frames = [_generate_marketing_seed(year=y, month=m) for y, m in periods]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        
        if df.empty:
            return {"success": False, "rows_loaded": 0, "error": "No marketing data generated"}
        
        return _insert_marketing_to_db(df)

    try:
        # Read marketing data - sample first to detect columns
        df_sample = pd.read_csv(file_path, nrows=5)
        cols = [c.strip().lower().replace(' ', '_') for c in df_sample.columns]
        
        # Read full data with detected columns
        df = pd.read_csv(file_path)
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        
        # Flexible column mapping
        col_mapping = {
            'acquisition_cost': ['acquisition_cost', 'cac', 'cost', 'acquisition_cost_usd'],
            'roi': ['roi', 'return_on_investment', 'roi_percentage'],
            'conversion_rate': ['conversion_rate', 'conversion', 'conv_rate'],
            'clicks': ['clicks', 'total_clicks'],
            'impressions': ['impressions', 'total_impressions', 'impr'],
            'channel': ['channel', 'channel_used', 'channels_used', 'channels', 'marketing_channel'],
            'date': ['date', 'campaign_date', 'date_campaign'],
        }
        
        def find_col(col_options):
            for c in col_options:
                if c in df.columns:
                    return c
            return None
        
        acq_col = find_col(col_mapping['acquisition_cost'])
        roi_col = find_col(col_mapping['roi'])
        conv_col = find_col(col_mapping['conversion_rate'])
        click_col = find_col(col_mapping['clicks'])
        impr_col = find_col(col_mapping['impressions'])
        ch_col = find_col(col_mapping['channel'])
        date_col = find_col(col_mapping['date'])
        
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
        
        # Map channel
        if ch_col:
            df["channel"] = df[ch_col].astype(str).str.title()
        else:
            df["channel"] = "Marketing"
        
        # Convert numeric columns
        def to_numeric(col):
            if col and col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('$', '', regex=False).str.replace('%', '', regex=False).str.replace(',', '', regex=False), errors='coerce')
            return col
        
        acq_col = to_numeric(acq_col)
        roi_col = to_numeric(roi_col)
        conv_col = to_numeric(conv_col)
        click_col = to_numeric(click_col)
        impr_col = to_numeric(impr_col)
        
        # Aggregate by year/month/channel
        agg_dict = {}
        rename_map = {}
        if acq_col:
            agg_dict[acq_col] = 'mean'
            rename_map[acq_col] = 'avg_cac'
        if roi_col:
            agg_dict[roi_col] = 'mean'
            rename_map[roi_col] = 'avg_roi'
        if conv_col:
            agg_dict[conv_col] = 'mean'
            rename_map[conv_col] = 'avg_conversion'
        if click_col:
            agg_dict[click_col] = 'sum'
            rename_map[click_col] = 'total_clicks'
        if impr_col:
            agg_dict[impr_col] = 'sum'
            rename_map[impr_col] = 'total_impressions'
        
        if not agg_dict:
            return {"success": False, "error": "No numeric columns found for aggregation"}
        
        grouped = df.groupby(["year", "month", "channel"]).agg(agg_dict).reset_index()
        
        # Rename source columns to schema names
        grouped = grouped.rename(columns=rename_map)
        
        # Ensure all required columns exist with sensible defaults
        for col, default in [("avg_cac", 0), ("avg_roi", 0), ("avg_conversion", 0), 
                             ("total_clicks", 1), ("total_impressions", 1)]:
            if col not in grouped.columns:
                grouped[col] = default
        
        # Normalize conversion_rate: if values > 1 treat as percentage, else as ratio
        if grouped["avg_conversion"].mean() > 1:
            grouped["avg_conversion"] = grouped["avg_conversion"] / 100.0
        
        # Calculate derived metrics
        grouped["campaign_spend"] = grouped["avg_cac"] * grouped["total_impressions"] * grouped["avg_conversion"]
        grouped["ltv_estimated"] = grouped["avg_cac"] * (1 + grouped["avg_roi"] / 100)

        # Insert into database
        conn = sqlite3.connect(_ensure_db_path())
        try:
            conn.executemany(
                """
                INSERT INTO dss_marketing_monthly
                (year, month, channel, campaign_type, campaign_spend, avg_cac, avg_roi, 
                 avg_conversion, total_clicks, total_impressions, ltv_estimated, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(year, month, channel) DO UPDATE SET
                    campaign_spend = excluded.campaign_spend,
                    avg_cac = excluded.avg_cac,
                    avg_roi = excluded.avg_roi,
                    avg_conversion = excluded.avg_conversion,
                    total_clicks = excluded.total_clicks,
                    total_impressions = excluded.total_impressions,
                    ltv_estimated = excluded.ltv_estimated,
                    updated_at = CURRENT_TIMESTAMP
                """,
                [
                    (
                        int(row["year"]),
                        int(row["month"]),
                        str(row["channel"]),
                        "General",  # campaign_type
                        float(row["campaign_spend"]) if pd.notna(row["campaign_spend"]) else 0.0,
                        float(row["avg_cac"]) if pd.notna(row["avg_cac"]) else 0.0,
                        float(row["avg_roi"]) if pd.notna(row["avg_roi"]) else 0.0,
                        float(row["avg_conversion"]) if pd.notna(row["avg_conversion"]) else 0.0,
                        int(row["total_clicks"]) if pd.notna(row["total_clicks"]) else 0,
                        int(row["total_impressions"]) if pd.notna(row["total_impressions"]) else 0,
                        float(row["ltv_estimated"]) if pd.notna(row["ltv_estimated"]) else 0.0,
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
    result = load_marketing_to_dss()
    print(f"Marketing ETL: {result}")
