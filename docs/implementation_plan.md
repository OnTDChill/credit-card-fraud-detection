# Fix Marketing Budget Slider & Ensure Real Computed Data on CEO Dashboard

## Problem Summary

Two issues on the Customer tab ("Khach hang"):

1. **Marketing Budget slider does not work** -- The slider key uses `id(mkt)` (Python object memory address), which changes on every Streamlit rerun, causing the slider to reset to 0 each time.
2. **Data is insufficient for meaningful output** -- The database has only **1 month** of data (2024/01), so trend computations (MoM deltas, growth projections, cohort analysis) produce empty or misleading results.

Additionally, the user requires that **all numbers, charts, and tables are computed from real data** (not hardcoded decoration), and that generated data produces reasonable output for a CEO DSS of a financial intermediary company.

## Root Cause Analysis

### Bug 1: Slider key instability

In [01_Thu_hut_Kich_hoat.py](file:///g:/DataMN/credit-card-fraud-detection-main/credit-card-fraud-detection-main/streamlit_app/pages/ceo/01_Thu_hut_Kich_hoat.py#L132):
```python
key=f"m1_budget_short_term_{id(mkt)}"  # id(mkt) changes every rerun!
```
Streamlit uses widget keys to preserve state across reruns. When the key changes, it creates a new widget initialized at default (0), so the slider always resets.

### Bug 2: Insufficient historical data

Current DB state:
- `dss_marketing_monthly`: **6 rows** (all month=1, year=2024, one per channel)
- `dss_customer_service`: **1 row** (month=1, year=2024)
- `dss_transaction_summary`: **15 rows** (1 month, 5 types x 3 regions)
- `dss_credit_portfolio`: **12 rows** (1 month, segments x regions)

This means:
- MoM trend: cannot compute (needs >=2 months)
- Cohort retention: only 1 data point
- Growth projection: falls back to synthetic rates
- Overview health score: no MoM comparison possible

## Proposed Changes

### Component 1: Fix Slider Key (Bug fix)

#### [MODIFY] [01_Thu_hut_Kich_hoat.py](file:///g:/DataMN/credit-card-fraud-detection-main/credit-card-fraud-detection-main/streamlit_app/pages/ceo/01_Thu_hut_Kich_hoat.py)

**Line 132**: Change the slider key from `f"m1_budget_short_term_{id(mkt)}"` to a stable key like `"m1_budget_short_term"`. The `id(mkt)` was likely added to avoid a Streamlit `DuplicateWidgetID` error, but the correct fix is a static unique key.

---

### Component 2: Seed Multi-Month Data

#### [NEW] [core/etl/seed_multi_month.py](file:///g:/DataMN/credit-card-fraud-detection-main/credit-card-fraud-detection-main/core/etl/seed_multi_month.py)

Create a script that extends the existing single-month data to 6 months (2024/01 through 2024/06) by applying realistic month-over-month variation to the existing data. This ensures:

- **Transaction summary**: 6 months x 5 types x 3 regions = 90 rows (currently 15)
- **Marketing monthly**: 6 months x 6 channels = 36 rows (currently 6)
- **Customer service**: 6 months = 6 rows (currently 1)
- **Credit portfolio**: 6 months x segments x regions (currently 12 for 1 month)

Data generation approach:
- Use existing month-1 data as the base
- Apply 3-8% monthly growth with small random noise for transaction volumes
- Apply slight seasonal variation for marketing spend/conversion
- Gradually improve churn rate (realistic for a growing fintech)
- Keep NPL rate within realistic bounds (1-7% depending on segment)

> [!IMPORTANT]
> The script will INSERT new rows for months 2-6. It will NOT modify existing month-1 data. It is idempotent (skips months that already exist).

---

### Component 3: Audit All Pages for Decorative vs Computed Values

After reviewing all 6 CEO pages, here is the audit result:

| Page | Decorative? | Issue |
|------|-------------|-------|
| 00_Tong_quan | No | All computed from DB queries. MoM deltas need multi-month data (fixed by seed). |
| 01_Thu_hut | **Bug** | Slider key resets. Cohort only has 1 month. Growth projection falls back to synthetic rates. All fixed by Component 1 + 2. |
| 02_Tin_dung | No | Gauge/pie all computed from `credit_portfolio`. NPL trend chart needs multi-month (fixed by seed). |
| 03_He_sinh_thai | No | BCG matrix, combo simulation all computed from `service_ecosystem`. |
| 04_Merchant | No | Computed from `merchant_accounts`. |
| 05_Giao_dich | No | All computed from `transaction_summary`. Fraud forecast uses actual avg_fraud_rate as base. |

No pages have hardcoded decorative numbers. The main issue is insufficient data making computations produce empty/fallback results.

## Verification Plan

### Automated Tests
1. Run `python core/etl/seed_multi_month.py` and verify DB row counts increase
2. Run `streamlit run streamlit_app/app.py` and verify:
   - Marketing Budget slider holds its value when adjusted
   - KPI cards show computed values
   - Trend charts show 6 data points
   - Cohort heatmap renders with 6 months
   - Growth projection uses historical quantiles (not fallback rates)

### Manual Verification (Browser)
- Navigate to "Khach hang" tab
- Adjust Marketing Budget slider -- metrics should update reactively
- Check all expander sections render charts with real data
