"""ETL pipeline to seed review_queue with sample transactions for manual review."""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from streamlit_app.config import REVIEW_DB_PATH
from streamlit_app.components.dss_data_access import _ensure_db_path


def _generate_review_queue_sample() -> dict[str, Any]:
    """Generate sample transactions for review queue."""
    np.random.seed(42)
    
    conn = sqlite3.connect(_ensure_db_path())
    try:
        # Generate 20 sample transactions in the gray zone (0.3 - 0.7 fraud probability)
        sample_transactions = []
        for i in range(20):
            fraud_prob = np.random.uniform(0.3, 0.7)
            amount = np.random.uniform(1_000_000, 50_000_000)  # 1M - 50M VND
            customer_id = f"CUST{np.random.randint(10000, 99999)}"
            merchant_id = f"MERC{np.random.randint(1000, 9999)}"
            transaction_id = f"TXN{datetime.now().strftime('%Y%m%d%H%M%S')}{i:04d}"
            
            # Reason codes based on fraud probability
            reason_codes = []
            if fraud_prob > 0.5:
                reason_codes.append("HIGH_AMOUNT")
            if fraud_prob > 0.4:
                reason_codes.append("NEW_CUSTOMER")
            if fraud_prob > 0.45:
                reason_codes.append("UNUSUAL_LOCATION")
            if not reason_codes:
                reason_codes.append("ROUTINE_CHECK")
            
            received_at = datetime.now() - timedelta(minutes=np.random.randint(0, 60))
            expires_at = received_at + timedelta(hours=24)
            
            sample_transactions.append({
                "transaction_id": transaction_id,
                "fraud_probability": fraud_prob,
                "amount": amount,
                "customer_id": customer_id,
                "merchant_id": merchant_id,
                "received_at": received_at.strftime("%Y-%m-%d %H:%M:%S"),
                "expires_at": expires_at.strftime("%Y-%m-%d %H:%M:%S"),
                "reason_codes": ",".join(reason_codes),
                "status": "PENDING",
            })
        
        # Insert into review_queue
        for txn in sample_transactions:
            conn.execute(
                """
                INSERT OR REPLACE INTO review_queue
                (transaction_id, fraud_probability, amount, customer_id, merchant_id, 
                 received_at, expires_at, reason_codes, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    txn["transaction_id"],
                    txn["fraud_probability"],
                    txn["amount"],
                    txn["customer_id"],
                    txn["merchant_id"],
                    txn["received_at"],
                    txn["expires_at"],
                    txn["reason_codes"],
                    txn["status"],
                ),
            )
        
        conn.commit()
        
        return {
            "success": True,
            "rows_added": len(sample_transactions),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
    finally:
        conn.close()


def load_review_queue_to_dss() -> dict[str, Any]:
    """Load sample transactions into review_queue table."""
    if not REVIEW_DB_PATH.exists():
        return {"success": False, "error": "Database does not exist"}
    
    return _generate_review_queue_sample()


if __name__ == "__main__":
    result = load_review_queue_to_dss()
    print(f"Review Queue ETL: {result}")
