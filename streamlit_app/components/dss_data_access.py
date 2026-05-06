"""Data access layer for DSS CEO Dashboard tables."""

from __future__ import annotations

import importlib.util
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Load root config by absolute path to avoid circular import
_root_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.py")
_spec = importlib.util.spec_from_file_location("_root_config_dss", _root_config_path)
if _spec is not None and _spec.loader is not None:
    _rcfg = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_rcfg)
else:
    raise ImportError(f"Could not load root config from {_root_config_path}")
REVIEW_DB_PATH = _rcfg.REVIEW_DB_PATH


DB_PATH = Path(REVIEW_DB_PATH)


def _ensure_db_path() -> Path:
    """Return the database path, creating parent directories if needed."""
    if not DB_PATH.parent.exists():
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DB_PATH


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, col_def: str) -> None:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")


def _unique_constraint_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check whether any unique index on table includes the given column."""
    for idx in conn.execute(f"PRAGMA index_list({table})").fetchall():
        if idx[2]:  # unique flag
            cols = [r[2] for r in conn.execute(f"PRAGMA index_info({idx[1]})")]
            if column in cols:
                return True
    return False


def _rebuild_table_if_constraint_mismatch(conn: sqlite3.Connection, table: str, column: str) -> None:
    """Drop table if its unique constraint does not include column so schema script recreates it."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    if cur.fetchone() and not _unique_constraint_has_column(conn, table, column):
        conn.execute(f"DROP TABLE {table}")


def ensure_dss_tables() -> None:
    """Create DSS tables if they don't exist (idempotent)."""
    schema_path = Path(__file__).resolve().parent.parent.parent / "core" / "database_schema.sql"
    if not schema_path.exists():
        return

    conn = sqlite3.connect(_ensure_db_path())
    try:
        # Rebuild tables if unique constraint is outdated (missing region)
        _rebuild_table_if_constraint_mismatch(conn, "dss_transaction_summary", "region")
        _rebuild_table_if_constraint_mismatch(conn, "dss_merchant_accounts", "region")
        _rebuild_table_if_constraint_mismatch(conn, "dss_credit_portfolio", "region")
        with open(schema_path, "r", encoding="utf-8") as f:
            sql = f.read()
            conn.executescript(sql)
        # Migrate: add region columns if missing (for tables not rebuilt above)
        _add_column_if_missing(conn, "dss_transaction_summary", "region", "TEXT")
        _add_column_if_missing(conn, "dss_merchant_accounts", "region", "TEXT")
        _add_column_if_missing(conn, "dss_credit_portfolio", "region", "TEXT")
        conn.commit()
    finally:
        conn.close()


def get_transaction_summary(
    year: int | None = None, month: int | None = None
) -> pd.DataFrame:
    """Get transaction summary data for Revenue and Risk tabs (aggregated across regions)."""
    conn = sqlite3.connect(_ensure_db_path())
    try:
        query = """
            SELECT year, month, transaction_type,
                   SUM(total_count) as total_count,
                   SUM(total_amount) as total_amount,
                   SUM(revenue_estimated) as revenue_estimated,
                   SUM(fraud_count) as fraud_count,
                   SUM(fraud_amount) as fraud_amount,
                   AVG(fraud_rate) as fraud_rate,
                   AVG(avg_score) as avg_score
            FROM dss_transaction_summary
            WHERE 1=1
        """
        params: list[Any] = []
        if year:
            query += " AND year = ?"
            params.append(year)
        if month:
            query += " AND month = ?"
            params.append(month)
        query += " GROUP BY year, month, transaction_type ORDER BY year DESC, month DESC, transaction_type"
        return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def get_marketing_summary(
    year: int | None = None, month: int | None = None
) -> pd.DataFrame:
    """Get marketing campaign summary for Services tab (Acquisition)."""
    conn = sqlite3.connect(_ensure_db_path())
    try:
        query = "SELECT * FROM dss_marketing_monthly WHERE 1=1"
        params: list[Any] = []
        if year:
            query += " AND year = ?"
            params.append(year)
        if month:
            query += " AND month = ?"
            params.append(month)
        query += " ORDER BY year DESC, month DESC, channel"
        return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def get_cskh_summary(
    year: int | None = None, month: int | None = None
) -> pd.DataFrame:
    """Get customer service summary for Services tab (Retention)."""
    conn = sqlite3.connect(_ensure_db_path())
    try:
        query = "SELECT * FROM dss_customer_service WHERE 1=1"
        params: list[Any] = []
        if year:
            query += " AND year = ?"
            params.append(year)
        if month:
            query += " AND month = ?"
            params.append(month)
        query += " ORDER BY year DESC, month DESC"
        return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def get_strategy_targets(target_year: int | None = None) -> pd.DataFrame:
    """Get CEO strategy targets and actuals for Gap Analysis."""
    conn = sqlite3.connect(_ensure_db_path())
    try:
        query = "SELECT * FROM dss_strategy_targets WHERE 1=1"
        params: list[Any] = []
        if target_year:
            query += " AND target_year = ?"
            params.append(target_year)
        else:
            query += " AND target_year = ?"
            params.append(datetime.now().year)
        query += " ORDER BY domain, kpi_name"
        return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def save_strategy_target(
    target_year: int,
    domain: str,
    kpi_name: str,
    target_value: float,
    unit: str = "percent",
    strategy_note: str = "",
    created_by: str = "CEO",
) -> bool:
    """Save or update a CEO strategy target."""
    conn = sqlite3.connect(_ensure_db_path())
    try:
        conn.execute(
            """
            INSERT INTO dss_strategy_targets
            (target_year, domain, kpi_name, target_value, unit, strategy_note, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(target_year, domain, kpi_name) DO UPDATE SET
                target_value = excluded.target_value,
                unit = excluded.unit,
                strategy_note = excluded.strategy_note,
                updated_at = CURRENT_TIMESTAMP
            """,
            (target_year, domain, kpi_name, target_value, unit, strategy_note, created_by),
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def get_product_catalog() -> pd.DataFrame:
    """Get the product dimension table."""
    conn = sqlite3.connect(_ensure_db_path())
    try:
        return pd.read_sql_query("SELECT * FROM dim_product", conn)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def get_transaction_with_product(
    year: int | None = None, month: int | None = None
) -> pd.DataFrame:
    """Get transaction summary JOINed with dim_product for multi-dimensional analysis."""
    conn = sqlite3.connect(_ensure_db_path())
    try:
        query = """
            SELECT t.*, p.product_code, p.service_type, p.channel, p.segment,
                   p.use_case, p.display_name, p.risk_level
            FROM dss_transaction_summary t
            LEFT JOIN dim_product p ON t.transaction_type = p.transaction_type
            WHERE 1=1
        """
        params: list[Any] = []
        if year:
            query += " AND t.year = ?"
            params.append(year)
        if month:
            query += " AND t.month = ?"
            params.append(month)
        query += " ORDER BY t.year DESC, t.month DESC, t.transaction_type"
        return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def get_ceo_scorecard() -> dict[str, Any]:
    """Aggregate scorecard from all 3 DSS tables for Overview tab."""
    conn = sqlite3.connect(_ensure_db_path())
    current_year = datetime.now().year

    scorecard: dict[str, Any] = {
        "revenue": {"status": "unknown", "value": 0, "target": 0, "trend": 0},
        "risk": {"status": "unknown", "value": 0, "target": 0, "trend": 0},
        "service": {"status": "unknown", "value": 0, "target": 0, "trend": 0},
        "growth": {"status": "unknown", "value": 0, "target": 0, "trend": 0},
        "alerts": [],
    }

    try:
        # Revenue metrics
        row = conn.execute(
            """
            SELECT SUM(revenue_estimated) as total_revenue,
                   SUM(total_count) as total_txns
            FROM dss_transaction_summary
            WHERE year = ?
            """,
            (current_year,),
        ).fetchone()
        if row and row[0]:
            scorecard["revenue"]["value"] = row[0]

        # Risk metrics
        row = conn.execute(
            """
            SELECT SUM(fraud_amount) as total_loss,
                   AVG(fraud_rate) as avg_fraud_rate
            FROM dss_transaction_summary
            WHERE year = ?
            """,
            (current_year,),
        ).fetchone()
        if row:
            scorecard["risk"]["value"] = row[0] or 0
            scorecard["risk"]["fraud_rate"] = row[1] or 0

        # Service metrics
        row = conn.execute(
            """
            SELECT AVG(csat_score) as avg_csat,
                   AVG(nps_score) as avg_nps,
                   AVG(churn_rate) as avg_churn
            FROM dss_customer_service
            WHERE year = ?
            """,
            (current_year,),
        ).fetchone()
        if row:
            scorecard["service"]["csat"] = row[0] or 0
            scorecard["service"]["nps"] = row[1] or 0
            scorecard["service"]["churn"] = row[2] or 0

        # Load targets and calculate gaps
        targets = pd.read_sql_query(
            "SELECT domain, kpi_name, target_value, actual_value FROM dss_strategy_targets WHERE target_year = ?",
            conn,
            params=(current_year,),
        )
        for _, t in targets.iterrows():
            domain = t["domain"].lower()
            if domain in scorecard:
                scorecard[domain]["target"] = t["target_value"]

    except Exception:
        pass
    finally:
        conn.close()

    return scorecard


def get_credit_portfolio(
    year: int | None = None, month: int | None = None
) -> pd.DataFrame:
    """Get credit portfolio data for Module 2 (Tín dụng), aggregated across regions."""
    conn = sqlite3.connect(_ensure_db_path())
    try:
        query = """
            SELECT year, month, segment,
                   SUM(total_users) as total_users,
                   AVG(credit_limit) as credit_limit,
                   AVG(interest_rate) as interest_rate,
                   AVG(npl_rate) as npl_rate,
                   AVG(default_probability) as default_probability,
                   SUM(total_outstanding) as total_outstanding,
                   SUM(overdue_30d_amount) as overdue_30d_amount,
                   SUM(overdue_90d_amount) as overdue_90d_amount,
                   SUM(revenue_interest) as revenue_interest
            FROM dss_credit_portfolio
            WHERE 1=1
        """
        params: list[Any] = []
        if year:
            query += " AND year = ?"
            params.append(year)
        if month:
            query += " AND month = ?"
            params.append(month)
        query += " GROUP BY year, month, segment ORDER BY year DESC, month DESC, segment"
        return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def get_merchant_accounts(
    year: int | None = None, month: int | None = None,
    risk_level: str | None = None, min_anomaly: float = 0.0,
) -> pd.DataFrame:
    """Get merchant anomaly data for Module 4."""
    conn = sqlite3.connect(_ensure_db_path())
    try:
        query = "SELECT * FROM dss_merchant_accounts WHERE 1=1"
        params: list[Any] = []
        if year:
            query += " AND year = ?"
            params.append(year)
        if month:
            query += " AND month = ?"
            params.append(month)
        if risk_level:
            query += " AND risk_level = ?"
            params.append(risk_level)
        if min_anomaly > 0:
            query += " AND anomaly_score >= ?"
            params.append(min_anomaly)
        query += " ORDER BY anomaly_score DESC, monthly_volume DESC"
        return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def get_service_ecosystem(
    year: int | None = None, month: int | None = None
) -> pd.DataFrame:
    """Get cross-sell ecosystem data for Module 3."""
    conn = sqlite3.connect(_ensure_db_path())
    try:
        query = "SELECT * FROM dss_service_ecosystem WHERE 1=1"
        params: list[Any] = []
        if year:
            query += " AND year = ?"
            params.append(year)
        if month:
            query += " AND month = ?"
            params.append(month)
        query += " ORDER BY lift DESC, support_pct DESC"
        return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def get_transaction_by_region(
    year: int | None = None, month: int | None = None
) -> pd.DataFrame:
    """Get transaction summary aggregated by region."""
    conn = sqlite3.connect(_ensure_db_path())
    try:
        query = """
            SELECT region,
                   SUM(total_count) as total_count,
                   SUM(total_amount) as total_amount,
                   SUM(revenue_estimated) as revenue_estimated,
                   SUM(fraud_count) as fraud_count,
                   SUM(fraud_amount) as fraud_amount,
                   AVG(fraud_rate) as fraud_rate
            FROM dss_transaction_summary
            WHERE 1=1
        """
        params: list[Any] = []
        if year:
            query += " AND year = ?"
            params.append(year)
        if month:
            query += " AND month = ?"
            params.append(month)
        query += " GROUP BY region ORDER BY total_amount DESC"
        return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def get_merchant_by_region(
    year: int | None = None, month: int | None = None
) -> pd.DataFrame:
    """Get merchant summary aggregated by region."""
    conn = sqlite3.connect(_ensure_db_path())
    try:
        query = """
            SELECT region,
                   COUNT(*) as total_merchants,
                   SUM(monthly_volume) as total_volume,
                   SUM(est_monthly_revenue) as total_revenue,
                   SUM(est_tax_collectable) as total_tax,
                   AVG(anomaly_score) as avg_anomaly,
                   SUM(CASE WHEN is_suspected_merchant = 1 THEN 1 ELSE 0 END) as suspected_count
            FROM dss_merchant_accounts
            WHERE 1=1
        """
        params: list[Any] = []
        if year:
            query += " AND year = ?"
            params.append(year)
        if month:
            query += " AND month = ?"
            params.append(month)
        query += " GROUP BY region ORDER BY total_volume DESC"
        return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def get_latest_data_date() -> datetime | None:
    """Get the latest date of available data across all DSS tables."""
    conn = sqlite3.connect(_ensure_db_path())
    try:
        row = conn.execute(
            """
            SELECT MAX(year) as y, MAX(month) as m
            FROM dss_transaction_summary
            """
        ).fetchone()
        if row and row[0] and row[1]:
            return datetime(row[0], row[1], 1)
        return None
    except Exception:
        return None
    finally:
        conn.close()


def get_available_periods() -> list[tuple[int, int]]:
    """Return sorted (year, month) tuples available in transaction summary."""
    conn = sqlite3.connect(_ensure_db_path())
    try:
        rows = conn.execute(
            "SELECT DISTINCT year, month FROM dss_transaction_summary ORDER BY year, month"
        ).fetchall()
        return [(int(r[0]), int(r[1])) for r in rows if r[0] and r[1]]
    except Exception:
        return []
    finally:
        conn.close()


def get_kpi_trend_history(year: int | None = None, month: int | None = None, months: int = 12) -> pd.DataFrame:
    """Get monthly aggregated KPI history for trend/sparkline analysis.
    Returns columns: year, month, total_amount, total_count, revenue_estimated,
    fraud_rate, npl_rate, churn_rate, csat_score, active_users.
    """
    conn = sqlite3.connect(_ensure_db_path())
    try:
        query = """
            SELECT
                t.year,
                t.month,
                SUM(t.total_amount) as total_amount,
                SUM(t.total_count) as total_count,
                SUM(t.revenue_estimated) as revenue_estimated,
                AVG(t.fraud_rate) as fraud_rate,
                AVG(c.npl_rate) as npl_rate,
                AVG(cs.churn_rate) as churn_rate,
                AVG(cs.csat_score) as csat_score,
                SUM(c.total_users) as active_users
            FROM dss_transaction_summary t
            LEFT JOIN (
                SELECT year, month, AVG(npl_rate) as npl_rate, SUM(total_users) as total_users
                FROM dss_credit_portfolio GROUP BY year, month
            ) c ON t.year = c.year AND t.month = c.month
            LEFT JOIN dss_customer_service cs ON t.year = cs.year AND t.month = cs.month
            WHERE 1=1
        """
        params: list[Any] = []
        # When both year and month are specified, get last N months up to that point.
        # When only year is specified, get all months in that year.
        if year is not None and month is not None:
            query += " AND (t.year < ? OR (t.year = ? AND t.month <= ?))"
            params.extend([year, year, month])
        elif year is not None:
            query += " AND t.year = ?"
            params.append(year)

        query += """ GROUP BY t.year, t.month ORDER BY t.year DESC, t.month DESC LIMIT ?"""
        params.append(months)

        df = pd.read_sql_query(query, conn, params=params)
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def get_funnel_history(months: int = 6) -> pd.DataFrame:
    """Get historical marketing funnel conversion rates.
    
    Returns DataFrame with columns:
    - year, month: time period
    - click_rate: clicks / impressions (CTR)
    - signup_rate: signups / clicks (conversion)
    - acquisition_count: total new signups
    
    Used for calculating realistic funnel projections based on historical performance.
    """
    conn = sqlite3.connect(_ensure_db_path())
    try:
        query = """
            SELECT 
                year,
                month,
                total_impressions,
                total_clicks,
                avg_conversion,
                campaign_spend,
                avg_cac
            FROM dss_marketing_monthly
            WHERE total_impressions > 0 AND total_clicks > 0
            ORDER BY year DESC, month DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(months,))
        
        if df.empty or len(df) < 2:
            return pd.DataFrame()
        
        df["acquisition_count"] = (df["total_impressions"] * df["avg_conversion"]).astype(int)
        df["click_rate"] = df["total_clicks"] / df["total_impressions"]
        df["signup_rate"] = df["acquisition_count"] / df["total_clicks"]
        df = df.dropna(subset=["click_rate", "signup_rate"])

        return df[["year", "month", "click_rate", "signup_rate", "acquisition_count"]]
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()
