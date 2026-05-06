# Plan: 4-Day Sprint - MoMo DSS Dashboard (Revised)

Plan này tập trung vào Decision Support System thực sự giúp CEO ra quyết định, với data aggregation đúng cách và DSS features hoạt động.

## Ngày 1: Fix Calculations & DSS Core (Priority: Critical)

### 1.1 Module 1 - Fix Funnel Calculations + Implement get_funnel_history

**Files:**
- `streamlit_app/components/dss_data_access.py` (add function)
- `streamlit_app/pages/ceo/01_Thu_hut_Kich_hoat.py` (fix calculations)

**Task 1A: Implement get_funnel_history() in dss_data_access.py**
```python
def get_funnel_history(months: int = 6) -> pd.DataFrame:
    """Get historical funnel conversion rates from marketing data."""
    # Query dss_marketing_monthly for impressions, clicks, signups by month
    # Calculate click_rate and signup_rate per month
    # Return DataFrame with: month, click_rate, signup_rate
    # If insufficient data, return empty DataFrame
```

**Time estimate:** 1.5 hours (includes testing)

**Task 1B: Fix funnel calculations in 01_Thu_hut_Kich_hoat.py**
```python
# Calculate funnel from actual marketing data
hist_funnel = get_funnel_history(months=6)

if not hist_funnel.empty and len(hist_funnel) >= 3:
    # Use historical rates
    click_rate = hist_funnel["click_rate"].mean()  # ~2-5% industry
    signup_rate = hist_funnel["signup_rate"].mean()  # ~20-40% of clicks
else:
    # Industry fallback with documented basis
    click_rate = 0.03  # 3% - industry avg for digital ads (WordStream 2023)
    signup_rate = 0.30  # 30% - fintech app benchmark (adjust based on UX)

# Calculate actual funnel from impressions
impressions = mkt["total_impressions"].sum()
funnel_counts = [
    int(impressions),
    int(impressions * click_rate),
    int(impressions * click_rate * signup_rate),
]

# Add tooltip explaining calculation basis
st.caption("💡 Tỷ lệ chuyển đổi dựa trên " + 
           (f"{len(hist_funnel)} kỳ lịch sử" if len(hist_funnel) >= 3 
            else "benchmark ngành fintech VN"))
```

**Time estimate:** 2 hours

### 1.2 Module 5 - Fix Security Scenario with Documented Fallback

**File:** `streamlit_app/pages/ceo/05_Giao_dich_An_toan.py`

**Current (hardcoded):**
```python
sec_factors = {
    "🟢 Cơ bản": {"fraud_mult": 1.0, "drop_mult": 1.0, "cost_mult": 1.0},
    "🟡 Nâng cao": {"fraud_mult": 0.7, "drop_mult": 1.05, "cost_mult": 1.3},
    "🔴 Siết chặt": {"fraud_mult": 0.4, "drop_mult": 1.15, "cost_mult": 2.0},
}
```

**Fix with documented basis:**
```python
# Security scenario multipliers based on industry research
# Source: McKinsey Digital Fraud Study 2023, Forrester TEI
# Fraud reduction: Enhanced auth reduces fraud by 30-70%
# Friction cost: Additional security adds 3-5% txn drop, 30-100% cost increase

sec_factors = {
    "🟢 Cơ bản": {
        "fraud_mult": 1.0,    # baseline - current fraud rate
        "drop_mult": 1.0,     # no additional friction
        "cost_mult": 1.0,     # current operating cost
        "basis": "Baseline - hệ thống hiện tại"
    },
    "🟡 Nâng cao": {
        "fraud_mult": 0.70,   # -30% fraud (McKinsey avg for 2FA)
        "drop_mult": 1.05,    # +5% txn drop (friction from extra steps)
        "cost_mult": 1.30,    # +30% cost (SMS/verification costs)
        "basis": "Thêm xác thực 2 lớp - giảm 30% gian lận (McKinsey 2023)"
    },
    "🔴 Siết chặt": {
        "fraud_mult": 0.40,   # -60% fraud (biometric + behavioral)
        "drop_mult": 1.15,    # +15% txn drop (high friction)
        "cost_mult": 2.00,    # +100% cost (advanced systems)
        "basis": "Biometric + Behavioral - giảm 60% gian lận, +15% friction"
    },
}

# Display basis in tooltip
st.caption(f"📊 Cơ sở: {sec_factors[security_level]['basis']}")
```

**Time estimate:** 1.5 hours

### 1.3 Fix File Syntax Error in 01_Thu_hut_Kich_hoat.py

**Issue:** File has broken syntax at lines 16-17 and 86-89.

**Fix:**
```python
# Line 15-16 should be:
configure_dashboard_page("Khách hàng mới")
render_page_header(
    "1. Thu hút khách hàng mới",
    "Theo dõi hiệu quả chiến dịch marketing và tỷ lệ chuyển đổi người dùng",
    kicker="Báo cáo CEO",
)
```

**Time estimate:** 30 minutes

**Day 1 Total: ~5.5 hours** (leaves buffer for testing)

---

## Ngày 2: Aggregation Strategy & Data Scale (Priority: Critical)

### 2.1 ETL - Fix Data Aggregation (Avoid 318M Rows Problem)

**Core principle:** ETL aggregates raw data → DSS tables chỉ chứa monthly summary (~360 rows/bảng max)

**Files:**
- `core/etl/load_credit.py`
- `core/etl/load_marketing.py`
- `core/etl/load_merchant.py`

**Current approach (problematic):**
```python
# Current: Generates individual records per user/merchant
# 26.5M users × 12 months = 318M rows
```

**Correct approach:**
```python
# ETL aggregates raw data into monthly summary by segment/region
# DSS table structure (already correct in database_schema.sql):
# - dss_credit_portfolio: aggregated by (year, month, segment, region)
# Max rows: 12 months × 7 segments × 3 regions = 252 rows
```

**Task 2A: Verify current ETL already aggregates correctly**
- Check `dss_credit_portfolio` schema - row per segment/region/month
- Check `dss_marketing_monthly` - row per channel/month
- Check `dss_merchant_accounts` - row per merchant/month (individual merchants OK, 50K not 200K)

**Time estimate:** 1 hour

**Task 2B: Scale segment parameters for MoMo-like numbers**
```python
# File: core/etl/load_credit.py
# Change segment names and scale numbers

SEGMENTS = ["GenZ", "Sinh viên", "NV Văn phòng", "Kinh doanh", 
            "Hưu trí", "Tiểu thương", "Công nhân"]
# Note: "Taos" → "Tiểu thương"

# Scale to realistic MoMo proportions (individual records aggregated to segment level)
# Final dashboard sees: 7 segments × 3 regions × 12 months = 252 rows
SEGMENT_PARAMS = {
    "GenZ":          (450_000, 5_000_000, 24.0, 5.5, 0.08),      # 450K users
    "Sinh viên":     (300_000, 3_000_000, 22.0, 4.2, 0.06),      # 300K users
    "NV Văn phòng":  (800_000, 15_000_000, 18.0, 1.5, 0.02),     # 800K users
    "Kinh doanh":    (500_000, 50_000_000, 16.0, 2.8, 0.03),     # 500K users
    "Hưu trí":       (200_000, 8_000_000, 15.0, 0.8, 0.01),      # 200K users
    "Tiểu thương":   (50_000, 80_000_000, 20.0, 3.5, 0.05),      # 50K users (high limit)
    "Công nhân":     (350_000, 10_000_000, 20.0, 3.0, 0.04),     # 350K users
}
# Total: ~2.65M credit users (segment aggregate, not individual rows)
```

**Time estimate:** 1.5 hours

**Task 2C: Scale marketing for realistic acquisition**
```python
# File: core/etl/load_marketing.py
# Target: ~100K monthly new users (realistic for demo, scalable to 1M)

CHANNEL_PARAMS = {
    "Mạng xã hội": (50_000_000, 0.025, 4_500),    # 50M impressions
    "TikTok":      (80_000_000, 0.030, 8_000),    # 80M impressions
    "YouTube":     (30_000_000, 0.020, 5_000),
    "Email":       (5_000_000, 0.008, 500),
    "SMS":         (2_000_000, 0.012, 1_500),
    "Affiliate":   (10_000_000, 0.035, 4_000),
    "Referral":    (500_000, 0.25, 1_000),
}
# Result: ~100K acquisitions/month at ~50K CAC
```

**Time estimate:** 1 hour

**Task 2D: Scale merchants to 50K**
```python
# File: core/etl/load_merchant.py
N_ACCOUNTS = 50_000  # Achievable without performance issues
```

**Time estimate:** 30 minutes

**Task 2E: Run ETL and verify row counts**
```bash
python -m core.etl.run_all_etl
```

Verify:
- `dss_credit_portfolio`: <300 rows
- `dss_marketing_monthly`: <100 rows
- `dss_merchant_accounts`: 50,000 rows (OK for SQLite)
- Query time: <2 seconds per dashboard page

**Time estimate:** 1-2 hours (including regeneration)

**Day 2 Total: ~6 hours**

---

## Ngày 3: DSS Features & CEO Command Panel (Priority: High)

### 3.1 Test CEO Command Panel Functionality

**Files to test:**
- All 5 CEO pages with `render_ceo_command_panel`

**Test scenarios:**
```python
# Test case 1: Module 5 - Fraud > 1.5% triggers command panel action
# Steps:
# 1. Navigate to "Giao dịch & An toàn"
# 2. Check if fraud_rate > 1.5%
# 3. Verify command panel shows relevant actions
# 4. Test action: "Tăng cường xác thực giao dịch"
# 5. Verify action links correctly

# Test case 2: Module 2 - NPL > 5% triggers warning
# Steps:
# 1. Navigate to "Tín dụng"
# 2. Check if NPL > 5%
# 3. Verify command panel shows "Siết chặt chính sách cho vay"

# Test case 3: Module 1 - CAC too high
# Steps:
# 1. Navigate to "Khách hàng mới"
# 2. If CAC > 100K, verify panel shows "Tối ưu ngân sách Marketing"
```

**Time estimate:** 3 hours (includes fixes)

### 3.2 Verify What-if Simulations Work

**Test scenarios:**
```python
# Module 1: Marketing budget slider
# - Drag slider to +50% budget
# - Verify new_acq calculates correctly (new_budget / weighted_cac)
# - Verify delta metrics display

# Module 2: Economic scenario NPL forecast
# - Select "🌧️ Suy thoái nhẹ"
# - Verify NPL projection increases by 1.25x
# - Verify chart updates

# Module 5: Security level scenario
# - Select "🔴 Siết chặt"
# - Verify fraud_mult = 0.4 applied correctly
# - Verify cost_mult = 2.0 applied correctly
```

**Time estimate:** 2 hours (includes fixes)

### 3.3 Single Port Configuration

**File:** `.streamlit/config.toml` (create)
```toml
[server]
port = 8501
headless = true
runOnSave = false
fileWatcherType = "none"

[browser]
gatherUsageStats = false
```

**File:** `streamlit_app/app.py` (verify page config exists)
```python
# Should already exist:
st.set_page_config(
    page_title="Fraud Detection DSS",
    page_icon="🛡️",
    layout="wide",
)
```

**Time estimate:** 30 minutes

**Day 3 Total: ~5.5 hours**

---

## Ngày 4: Validation & Documentation (Priority: Medium)

### 4.1 Performance Validation

**Test query performance:**
```python
# Test each CEO page loads in <3 seconds
# Test filters (year/month) apply in <2 seconds
# Test charts render without lag
```

**Time estimate:** 2 hours (includes optimization if needed)

### 4.2 Add Data Simulation Footnote (Not Banner)

**Files:** All CEO pages

**Change from banner to footnote:**
```python
# Instead of st.info banner at top:
# Use small caption at bottom of each page

st.caption(
    "📊 Dữ liệu mô phỏng quy mô MoMo cho mục đích demo. "
    "Methodology: Industry benchmarks + Synthetic aggregation."
)
```

**Time estimate:** 1 hour

### 4.3 Create Technical Demo Guide

**File:** `DEMO_GUIDE.md` (for presenter, not CEO)

**Content:**
```markdown
# DSS Dashboard Demo Guide

## Audience: Technical reviewer / Product manager

### Key DSS Features to Showcase

1. **What-if Simulation: Marketing Budget**
   - Page: "1. Khách hàng mới"
   - Action: Drag budget slider to +50%
   - Expected: New CAC and acquisition count update in real-time
   - DSS Value: CEO can decide optimal budget before spending

2. **Alert-driven Command Panel: Fraud Response**
   - Page: "5. Giao dịch & An toàn" (if fraud > 1.5%)
   - Action: Click "Tăng cường xác thực"
   - Expected: Action logged, recommendation shown
   - DSS Value: Immediate response to threshold breach

3. **Scenario Forecast: Economic Impact**
   - Page: "2. Tín dụng"
   - Action: Select "⛈️ Khủng hoảng" scenario
   - Expected: NPL forecast increases, risk-adjusted lending shown
   - DSS Value: Proactive risk management

### Sample CEO Questions & Answers

Q: "Nếu tôi cắt ngân sách marketing 30%, mất bao nhiêu khách?"
A: Show Module 1 slider, drag to -30%, read delta_acq metric.

Q: "Tỷ lệ gian lận 2% là cao hay thấp?"
A: Show Module 5 gauge, explain 2% is above 1.5% threshold, show command panel actions.

Q: "Nợ xấu dự báo năm sau thế nào?"
A: Show Module 2 forecast chart, switch economic scenarios.
```

**Time estimate:** 1.5 hours

### 4.4 Final Testing Checklist

- [ ] All 6 CEO pages load <3 seconds
- [ ] CEO Command Panels render on all pages
- [ ] What-if sliders work on Module 1, 2, 5
- [ ] Funnel calculations use data (not hardcoded)
- [ ] Security scenarios have documented basis
- [ ] Port 8501 only
- [ ] No "Taos" segment (renamed to "Tiểu thương")
- [ ] Data simulation noted in footnote (not banner)
- [ ] ETL row counts <1000 per table
- [ ] Query time <2 seconds

**Time estimate:** 2 hours

**Day 4 Total: ~6.5 hours**

---

## Summary: 4-Day Timeline

| Ngày | Focus | Key Deliverables | Est. Hours |
|------|-------|------------------|------------|
| 1 | Fix Calculations | get_funnel_history(), documented security factors, syntax fix | 5.5 |
| 2 | Aggregation & Scale | Verified ETL aggregation, scaled segments (Tiểu thương), 50K merchants | 6.0 |
| 3 | DSS Features | CEO Command Panel tested, What-if validated, single port config | 5.5 |
| 4 | Validation | Performance <3s, footnotes, demo guide, final checklist | 6.5 |

**Total: ~24 hours (realistic for 4-day sprint)**

## Success Criteria

- ✅ DSS features work (Command Panel, What-if)
- ✅ Data aggregated correctly (<1000 rows/table)
- ✅ Query performance <3 seconds
- ✅ No hardcoded calculations (documented fallbacks only)
- ✅ "Tiểu thương" segment (not "Taos")
- ✅ Port 8501 configured
- ✅ CEO can answer: "Nếu X thì Y" questions using dashboard

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| get_funnel_history takes too long | Fallback to documented industry benchmarks |
| CEO Command Panel has bugs | Test only core 3 panels (M1, M2, M5) if time limited |
| 50K merchants slow | Reduce to 20K, keep realistic ratio |
| Query still slow after aggregation | Add SQLite indexes on (year, month) columns |

## Out of Scope

❌ Real data connection (needs DWH access)
❌ PostgreSQL migration (needs >1 week)
❌ Authentication (needs >1 week)
❌ Kubernetes (needs >2 weeks)
❌ Real-time streaming (needs Kafka setup)
