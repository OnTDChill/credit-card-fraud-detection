import sqlite3
from config import REVIEW_DB_PATH

def verify():
    conn = sqlite3.connect(REVIEW_DB_PATH)
    tables = ['dss_transaction_summary', 'dss_marketing_monthly', 'dss_customer_service', 'dss_credit_portfolio']
    for t in tables:
        count = conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        print(f"{t}: {count} rows")
    conn.close()

if __name__ == "__main__":
    verify()