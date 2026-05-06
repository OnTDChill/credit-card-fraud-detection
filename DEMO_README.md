# MoMo DSS Dashboard - CEO Decision Support System

## 🎯 Mục đích Demo

Dashboard này là **Decision Support System (DSS)** mô phỏng cho một fintech trung gian thanh toán tại Việt Nam (quy mô tương đương MoMo - 40M+ users). Mục tiêu là cung cấp cho CEO và Ban Lãnh đạo công cụ:

- **Tổng quan KPI** thời gian thực qua 5 trụ cột: Thu hút, Tín dụng, Hệ sinh thái, Merchant, An toàn
- **What-if scenarios** cho các quyết định chiến lược (economic stress testing, security scenarios)
- **CEO Command Panel** - Các "mệnh lệnh" có thể thực thi cho từng module

## 📊 Methodology - Dữ liệu mô phỏng

### Scale dữ liệu (MoMo-like)

| Module | Scale | Nguồn |
|--------|-------|-------|
| **Thu hút & Kích hoạt** | ~100K acquisitions/tháng | Industry benchmarks: 3% CTR, 30% conversion |
| **Tín dụng** | 2.3M credit users | 6 segments × 3 regions × 12 months |
| **Hệ sinh thái** | 25M MAU | Cross-sell từ combo dịch vụ |
| **Merchant** | 50K accounts | Hidden merchant detection (anomaly score) |
| **An toàn** | 25M transactions/tháng | Fraud rate 0.1-0.5% |

### DSS Aggregation Strategy

Thay vì lưu 318M raw transactions, chúng tôi aggregate vào DSS tables:
- `dss_credit_portfolio`: ~252 rows (7 segments × 3 regions × 12 months)
- `dss_marketing_monthly`: ~84 rows (7 channels × 12 months)
- `dss_merchant_accounts`: ~600K rows (50K accounts × 12 months - vẫn manageable)
- `dss_transaction_summary`: ~180 rows (5 regions × 3 months × 12 txn types)

### Calculation Sources

| Metric | Fallback Value | Source |
|--------|---------------|--------|
| CTR (Click-Through Rate) | 3% | Industry benchmark (Google Ads avg) |
| Signup conversion | 30% | MoMo app benchmark |
| GDP→NPL elasticity | -0.5 to -0.8 | IMF Working Paper 2023 |
| Fraud reduction (2FA) | -30% | McKinsey Digital Fraud Study 2023 |
| Merchant fee premium | 2% | MoMo/VietQR rates (1.5-2.5%) |
| VAT rate | 10% | Vietnam tax law |

## 🏗️ Kiến trúc

```
streamlit_app/
├── app.py                    # Entry point
├── .streamlit/config.toml    # Single port 8501
├── components/
│   ├── dss_data_access.py    # Data layer (SQLite DSS)
│   ├── dss_engine.py         # What-if simulation engine
│   └── ui_components.py        # Shared UI components
├── pages/ceo/
│   ├── 00_Tong_quan_CEO.py   # CEO Overview
│   ├── 01_Thu_hut_Kich_hoat.py
│   ├── 02_Tin_dung.py
│   ├── 03_He_sinh_thai.py
│   ├── 04_Merchant.py
│   └── 05_Giao_dich_An_toan.py

core/etl/
├── load_credit.py            # ~2.3M users, 6 segments
├── load_marketing.py         # ~177M impressions, 7 channels
├── load_merchant.py          # 50K accounts
├── load_cskh.py              # 40M users, ~25K tickets/month
└── load_ecosystem.py         # Cross-sell combos
```

## 🚀 Chạy Dashboard

### Yêu cầu
- Python 3.10+
- Dependencies: `pip install -r requirements.txt`

### Chạy
```bash
cd streamlit_app
streamlit run app.py
```

Hoặc dùng launcher:
```bash
python run_dashboard.py
```

Dashboard chạy tại: **http://localhost:8501**

## 📈 Tính năng CEO DSS

### 1. Thu hút & Kích hoạt (M1)
- Funnel visualization: Impressions → Clicks → Signups → Activations
- CAC và ROI theo channel
- What-if: Điều chỉnh ngân sách marketing

### 2. Tín dụng (M2)
- Portfolio by segment (GenZ, NVVP, Kinh doanh...)
- NPL tracking và economic stress testing
- What-if: GDP scenarios (tăng trưởng → suy thoái)

### 3. Hệ sinh thái (M3)
- Cross-sell combos (Ví + Credit + Tiết kiệm)
- LTV analysis
- What-if: Bundle discount scenarios

### 4. Merchant (M4)
- Hidden merchant detection (anomaly score)
- Tax collectable estimation
- What-if: Conversion rate impact

### 5. An toàn (M5)
- Fraud detection metrics
- Security level scenarios (Cơ bản → Siết chặt)
- What-if: Friction vs Fraud trade-off

## ⚠️ Limitations

1. **Dữ liệu mô phỏng**: Không phải dữ liệu MoMo thực, dựa trên industry benchmarks
2. **SQLite**: Phù hợp demo, production cần PostgreSQL
3. **Single user**: Chưa có authentication/authorization
4. **Static ETL**: Real-time updates cần streaming pipeline

## 📚 References

- McKinsey & Company: "The State of Fraud 2023"
- IMF Working Paper: "Credit Cycles and Economic Growth"
- Vietnam Fintech Report 2024
- MoMo Official Statistics (public domain)

---

**Version**: 4-Day Sprint v2  
**Last Updated**: May 2026  
**Author**: AI Assistant + User Collaboration
