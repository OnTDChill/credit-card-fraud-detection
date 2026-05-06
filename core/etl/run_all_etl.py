"""Master ETL runner for all DSS CEO Dashboard datasets."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from load_paysim import load_paysim_to_dss
from load_marketing import load_marketing_to_dss
from load_cskh import load_cskh_to_dss
from load_credit import load_credit_to_dss
from load_merchant import load_merchant_to_dss
from load_ecosystem import load_ecosystem_to_dss


def run_all_etl() -> dict:
    """Run all ETL pipelines in sequence.
    
    Returns:
        Dict with results from each pipeline
    """
    start_time = datetime.now()
    
    print("=" * 60)
    print("DSS CEO Dashboard ETL Pipeline")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results = {}
    
    # 1. PaySim (Doanh thu + Rủi ro)
    print("\n[1/3] Loading PaySim dataset...")
    results["paysim"] = load_paysim_to_dss()
    if results["paysim"]["success"]:
        print(f"  ✓ Loaded {results['paysim'].get('rows_processed', 0):,} rows")
        print(f"  ✓ Created {results['paysim'].get('aggregated_records', 0)} aggregated records")
    else:
        print(f"  ✗ Error: {results['paysim'].get('error', 'Unknown error')}")
    
    # 2. Marketing (Dịch vụ - Acquisition)
    print("\n[2/3] Loading Marketing dataset...")
    results["marketing"] = load_marketing_to_dss()
    if results["marketing"]["success"]:
        print(f"  ✓ Loaded {results['marketing'].get('rows_loaded', 0):,} rows")
        print(f"  ✓ Created {results['marketing'].get('aggregated_records', 0)} aggregated records")
    else:
        print(f"  ✗ Error: {results['marketing'].get('error', 'Unknown error')}")
    
    # 3. CSKH (Dịch vụ - Retention)
    print("\n[3/6] Loading Customer Support dataset...")
    results["cskh"] = load_cskh_to_dss()
    if results["cskh"]["success"]:
        print(f"  ✓ Loaded {results['cskh'].get('rows_loaded', 0):,} rows")
        print(f"  ✓ Created {results['cskh'].get('aggregated_records', 0)} aggregated records")
    else:
        print(f"  ✗ Error: {results['cskh'].get('error', 'Unknown error')}")

    # 4. Credit Portfolio (Module 2)
    print("\n[4/6] Loading Credit Portfolio seed data...")
    results["credit"] = load_credit_to_dss()
    if results["credit"]["success"]:
        print(f"  ✓ Loaded {results['credit'].get('rows_loaded', 0):,} rows")
    else:
        print(f"  ✗ Error: {results['credit'].get('error', 'Unknown error')}")

    # 5. Merchant Accounts (Module 4)
    print("\n[5/6] Loading Merchant Anomaly seed data...")
    results["merchant"] = load_merchant_to_dss()
    if results["merchant"]["success"]:
        print(f"  ✓ Loaded {results['merchant'].get('rows_loaded', 0):,} rows")
    else:
        print(f"  ✗ Error: {results['merchant'].get('error', 'Unknown error')}")

    # 6. Service Ecosystem (Module 3)
    print("\n[6/6] Loading Cross-sell Ecosystem seed data...")
    results["ecosystem"] = load_ecosystem_to_dss()
    if results["ecosystem"]["success"]:
        print(f"  ✓ Loaded {results['ecosystem'].get('rows_loaded', 0):,} rows")
    else:
        print(f"  ✗ Error: {results['ecosystem'].get('error', 'Unknown error')}")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 60)
    print(f"ETL Completed in {duration:.1f} seconds")
    print("=" * 60)

    # Summary
    success_count = sum(1 for r in results.values() if r.get("success", False))
    print(f"\nSummary: {success_count}/6 pipelines succeeded")
    
    if success_count < 3:
        print("\nNote: Some datasets may not be available yet.")
        print("Please download datasets from Kaggle and place them in:")
        print("  - data/paysim/PS_20174392719_1491204439457_log.csv")
        print("  - data/marketing/marketing_campaign.csv")
        print("  - data/support/customer_support_tickets.csv")
    
    return {
        "success": success_count >= 3,
        "results": results,
        "duration_seconds": duration,
        "timestamp": end_time.isoformat(),
    }


if __name__ == "__main__":
    result = run_all_etl()
    sys.exit(0 if result["success"] else 1)
