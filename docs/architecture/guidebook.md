# Hệ Thống Phát Hiện Gian Lận Thẻ Tín Dụng
## Tài Liệu Kỹ Thuật Chi Tiết Dành Cho Đối Tác

> **Mục tiêu tài liệu**: Cung cấp đủ thông tin để đối tác có thể **tái lập hoàn toàn** (reproduce) hệ thống — từ ETL pipeline, thiết kế database, đến từng component hiển thị trên Streamlit Dashboard — mà không cần xem mã nguồn gốc.

---

## Mục Lục

1. [Tổng Quan Kiến Trúc & Workflow](#1-tổng-quan-kiến-trúc--workflow)
2. [Cấu Trúc Thư Mục Dự Án](#2-cấu-trúc-thư-mục-dự-án)
3. [ETL Pipeline — Hướng Dẫn Triển Khai Chi Tiết](#3-etl-pipeline--hướng-dẫn-triển-khai-chi-tiết)
4. [Database Schema](#4-database-schema)
5. [Data Access Layer (DAL)](#5-data-access-layer-dal)
6. [Dashboard — Triển Khai Từng Module](#6-dashboard--triển-khai-từng-module)
7. [Shared UI Components](#7-shared-ui-components)
8. [Hướng Dẫn Vận Hành](#8-hướng-dẫn-vận-hành)
9. [Lộ Trình Mở Rộng & Scaling](#9-lộ-trình-mở-rộng--scaling)

---

## 1. Tổng Quan Kiến Trúc & Workflow

### 1.1 Mô Hình Tổng Thể

Hệ thống được xây dựng theo mô hình **Modern Data Stack rút gọn**, ưu tiên tính tương tác cao, khả năng phản hồi gần real-time, và dễ bàn giao cho đối tác vận hành.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                │
│   CSV Files  /  External DB  /  API Streams  /  Synthetic Generator │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ETL LAYER  (core/etl/)                         │
│  load_paysim.py │ load_credit.py │ load_marketing.py                │
│  load_cskh.py   │ load_merchant.py │ load_review_queue.py           │
│                                                                     │
│  Nhiệm vụ: Validate → Transform → Aggregate → Write DB             │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STORAGE LAYER  (artifacts/fraud_system.db)             │
│  SQLite — Các bảng aggregate tối ưu cho query theo year/month       │
│  Index trên (year, month, region, segment)                          │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│       DATA ACCESS LAYER  (streamlit_app/components/dss_data_access) │
│  Abstraction layer — Tách SQL khỏi UI logic                         │
│  Trả về pandas DataFrame chuẩn hóa cho mọi component               │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│          PRESENTATION LAYER  (streamlit_app/pages/)                 │
│  Streamlit + Plotly + Custom HTML/CSS                               │
│  Module 1: Marketing & Growth  │  Module 2: Credit & NPL           │
│  Module 3: Merchant Analysis   │  Module 4: Executive Summary       │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Nguyên Tắc Thiết Kế

| Nguyên tắc | Lý do áp dụng |
|---|---|
| **Aggregate-first** | Dashboard không query dữ liệu thô — mọi số liệu đã được tính sẵn trong ETL. Query chỉ là `SELECT` đơn giản. |
| **DAL pattern** | Khi chuyển từ SQLite → PostgreSQL/BigQuery, chỉ cần sửa `dss_data_access.py`, không đụng đến UI. |
| **Stateless UI** | Streamlit re-render toàn bộ page mỗi khi có tương tác. Dùng `st.session_state` để persist state giữa các lần render. |
| **Plotly for all charts** | Plotly cho phép hover tooltip, zoom, export PNG — đáp ứng nhu cầu phân tích của CEO và kỹ thuật viên. |

---

## 2. Cấu Trúc Thư Mục Dự Án

```
fraud_detection_system/
│
├── core/
│   └── etl/
│       ├── load_paysim.py          # ETL giao dịch & gian lận
│       ├── load_credit.py          # ETL tín dụng & NPL
│       ├── load_marketing.py       # ETL marketing funnel
│       ├── load_cskh.py            # ETL chăm sóc khách hàng
│       ├── load_merchant.py        # ETL đối tác & anomaly
│       └── load_review_queue.py    # ETL hàng đợi review
│
├── streamlit_app/
│   ├── pages/
│   │   ├── 1_Marketing_Growth.py
│   │   ├── 2_Credit_NPL.py
│   │   ├── 3_Merchant_Analysis.py
│   │   └── 4_Executive_Summary.py
│   └── components/
│       ├── dss_data_access.py      # Data Access Layer
│       └── shared_ui.py            # Shared UI Components
│
├── artifacts/
│   └── fraud_system.db             # SQLite database (auto-generated)
│
├── run_all_etl.py                  # Chạy toàn bộ ETL pipeline
├── run_dashboard.py                # Khởi chạy Streamlit
└── requirements.txt
```

---

## 3. ETL Pipeline — Hướng Dẫn Triển Khai Chi Tiết

Mỗi module ETL đảm nhận một nghiệp vụ độc lập. Dưới đây là hướng dẫn tái lập từng module với đầy đủ logic, tham số, và ví dụ code.

### 3.1 Khung ETL Chung (Base Pattern)

Tất cả module ETL đều tuân theo cấu trúc sau:

```python
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "artifacts/fraud_system.db"

def run_etl():
    """Điểm vào chính của mỗi ETL module."""
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Bước 1: Tạo bảng nếu chưa có
        _create_tables(conn)
        
        # Bước 2: Sinh hoặc đọc dữ liệu
        df = _generate_or_load_data()
        
        # Bước 3: Transform & Aggregate
        df_agg = _aggregate(df)
        
        # Bước 4: Ghi vào database (replace toàn bộ)
        df_agg.to_sql("table_name", conn, if_exists="replace", index=False)
        
        conn.commit()
        print(f"[ETL] Loaded {len(df_agg)} rows.")
    finally:
        conn.close()

if __name__ == "__main__":
    run_etl()
```

> **Lưu ý**: Dùng `if_exists="replace"` để ETL idempotent — có thể chạy lại nhiều lần mà không gây duplicate dữ liệu.

---

### 3.2 Module Giao Dịch & Gian Lận (`load_paysim.py`)

**Mục tiêu**: Sinh dữ liệu mô phỏng luồng tiền (GMV) và tỷ lệ gian lận có tính mùa vụ, tăng trưởng theo năm, và phân bổ vùng miền giống môi trường Fintech thực tế (scale MoMo).

#### 3.2.1 Cấu Trúc Bảng Output

```sql
CREATE TABLE IF NOT EXISTS dss_transaction_summary (
    year        INTEGER,
    month       INTEGER,
    region      TEXT,       -- 'Miền Nam', 'Miền Bắc', 'Miền Trung'
    gmv         REAL,       -- Tổng giá trị giao dịch (VND)
    txn_count   INTEGER,    -- Số lượng giao dịch
    fraud_count INTEGER,    -- Số giao dịch gian lận
    fraud_rate  REAL        -- fraud_count / txn_count
);
```

#### 3.2.2 Logic Sinh Dữ Liệu

```python
import numpy as np
import pandas as pd

# --- Tham số cố định ---
YEARS = [2023, 2024, 2025]
MONTHS = list(range(1, 13))
REGIONS = ["Miền Nam", "Miền Bắc", "Miền Trung"]

# Phân bổ volume theo vùng (tổng = 1.0)
REGION_WEIGHTS = {
    "Miền Nam":   0.50,   # Trung tâm kinh tế → chiếm 50% GMV
    "Miền Bắc":  0.30,
    "Miền Trung": 0.20,
}

# Hệ số khuếch đại GMV tuyệt đối theo vùng
REGION_VOLUME_MULTIPLIERS = {
    "Miền Nam":   2.5,
    "Miền Bắc":  1.8,
    "Miền Trung": 1.0,    # Baseline
}

# Tỷ lệ gian lận cơ bản toàn quốc = 2%
BASE_FRAUD_RATE = 0.02

# Miền Trung có rủi ro cao hơn 20%
REGION_FRAUD_MULTIPLIERS = {
    "Miền Nam":   1.0,
    "Miền Bắc":  0.9,
    "Miền Trung": 1.2,
}

BASE_GMV_PER_MONTH = 500_000_000_000  # 500 tỷ VND/tháng (toàn quốc)


def generate_transaction_data() -> pd.DataFrame:
    records = []

    for year in YEARS:
        # Tăng trưởng 15%/năm so với năm 2023
        year_factor = 1.0 + (year - 2023) * 0.15

        for month in MONTHS:
            # Tính mùa vụ: dao động ±10% dùng hàm Sine
            # Đỉnh vào tháng 7 (giữa năm), đáy tháng 1 và 12
            seasonality = 1.0 + 0.1 * np.sin((month - 1) * np.pi / 6)

            for region in REGIONS:
                weight = REGION_WEIGHTS[region]
                vol_multiplier = REGION_VOLUME_MULTIPLIERS[region]
                fraud_multiplier = REGION_FRAUD_MULTIPLIERS[region]

                # GMV = GMV_baseline × tăng_trưởng × mùa_vụ × trọng_số_vùng × khuếch_đại_vùng
                gmv = (BASE_GMV_PER_MONTH
                       * year_factor
                       * seasonality
                       * weight
                       * vol_multiplier)

                # Số giao dịch: ước tính trung bình 200K VND/txn
                txn_count = int(gmv / 200_000)

                # Tỷ lệ gian lận theo vùng
                fraud_rate = BASE_FRAUD_RATE * fraud_multiplier

                # Thêm nhiễu ngẫu nhiên nhỏ ±5% để dữ liệu tự nhiên hơn
                fraud_rate *= np.random.uniform(0.95, 1.05)
                fraud_count = int(txn_count * fraud_rate)

                records.append({
                    "year": year,
                    "month": month,
                    "region": region,
                    "gmv": round(gmv, 0),
                    "txn_count": txn_count,
                    "fraud_count": fraud_count,
                    "fraud_rate": round(fraud_rate, 4),
                })

    return pd.DataFrame(records)
```

> **Giải thích công thức mùa vụ**: `sin((month-1) * π/6)` tạo chu kỳ 12 tháng với biên độ [-1, 1]. Nhân thêm 0.1 → dao động ±10%. Tháng 7 (index=6) → `sin(6π/6) = sin(π) = 0`, tháng 4 → `sin(3π/6) = sin(π/2) = 1` (đỉnh). Chỉnh `np.pi / 6` để dịch phase nếu cần peak vào dịp Tết (tháng 1-2).

---

### 3.3 Module Tín Dụng & NPL (`load_credit.py`)

**Mục tiêu**: Quản lý danh mục tín dụng, tính toán dư nợ và tỷ lệ nợ xấu (NPL) theo từng phân khúc khách hàng.

#### 3.3.1 Định Nghĩa Phân Khúc

```python
SEGMENTS = {
    "GenZ": {
        "users": 120_000,
        "credit_limit": 5_000_000,     # 5 triệu VND
        "npl_rate": 0.030,             # 3.0%
        "utilization_rate": 0.55,      # Sử dụng 55% hạn mức
    },
    "Sinh viên": {
        "users": 80_000,
        "credit_limit": 3_000_000,     # 3 triệu VND — hạn mức thấp nhất
        "npl_rate": 0.042,             # 4.2% — rủi ro cao nhất
        "utilization_rate": 0.70,      # Sử dụng nhiều hơn vì thu nhập thấp
    },
    "NV Văn phòng": {
        "users": 200_000,
        "credit_limit": 15_000_000,
        "npl_rate": 0.015,             # 1.5% — rủi ro thấp nhất (thu nhập ổn định)
        "utilization_rate": 0.50,
    },
    "Kinh doanh": {
        "users": 50_000,
        "credit_limit": 30_000_000,
        "npl_rate": 0.025,
        "utilization_rate": 0.65,
    },
    "Hưu trí": {
        "users": 30_000,
        "credit_limit": 10_000_000,
        "npl_rate": 0.018,
        "utilization_rate": 0.40,
    },
    "Tiểu thương": {
        "users": 40_000,
        "credit_limit": 80_000_000,    # 80 triệu VND — hạn mức cao nhất
        "npl_rate": 0.032,
        "utilization_rate": 0.60,
    },
}
```

#### 3.3.2 Tính Toán Dư Nợ & NPL

```python
def compute_credit_portfolio(segments: dict) -> pd.DataFrame:
    records = []

    for segment_name, params in segments.items():
        users = params["users"]
        limit = params["credit_limit"]
        npl_rate = params["npl_rate"]
        util = params["utilization_rate"]

        # Dư nợ = số users × hạn mức × tỷ lệ sử dụng
        # Công thức gốc trong tài liệu: users * limit * 0.6
        # Đây là phiên bản mở rộng với utilization_rate theo segment
        total_outstanding = users * limit * util

        # Nợ xấu tuyệt đối
        npl_amount = total_outstanding * npl_rate

        records.append({
            "segment": segment_name,
            "users": users,
            "credit_limit": limit,
            "utilization_rate": util,
            "total_outstanding": round(total_outstanding, 0),
            "npl_rate": npl_rate,
            "npl_amount": round(npl_amount, 0),
        })

    return pd.DataFrame(records)
```

---

### 3.4 Module Marketing Funnel (`load_marketing.py`)

**Mục tiêu**: Theo dõi hiệu quả chuyển đổi từng kênh marketing qua 3 tầng phễu: Impressions → Clicks → Registrations.

#### 3.4.1 Tham Số Kênh

```python
CHANNEL_PARAMS = {
    "TikTok": {
        "monthly_impressions": 5_000_000,   # Volume lớn nhất
        "click_rate":         0.03,          # CTR 3%
        "conversion_rate":    0.08,          # CVR thấp — awareness channel
        "cac":                80_000,        # CAC cao nhất (VND)
        "roi":                120,           # ROI 120%
    },
    "Facebook": {
        "monthly_impressions": 3_500_000,
        "click_rate":          0.04,
        "conversion_rate":     0.12,
        "cac":                 65_000,
        "roi":                 150,
    },
    "Google": {
        "monthly_impressions": 2_000_000,
        "click_rate":          0.06,         # CTR cao — intent-based
        "conversion_rate":     0.18,
        "cac":                 55_000,
        "roi":                 180,
    },
    "Referral": {
        "monthly_impressions": 500_000,      # Volume thấp nhất
        "click_rate":          0.40,         # CTR rất cao — người quen giới thiệu
        "conversion_rate":     0.25,         # CVR cao nhất
        "cac":                 30_000,       # CAC thấp nhất
        "roi":                 250,
    },
    "Email": {
        "monthly_impressions": 800_000,
        "click_rate":          0.15,
        "conversion_rate":     0.20,
        "cac":                 20_000,
        "roi":                 300,
    },
}
```

#### 3.4.2 Tính Toán Funnel & Chi Phí

```python
def compute_marketing_funnel(channel_params: dict) -> pd.DataFrame:
    records = []

    for channel, p in channel_params.items():
        impressions = p["monthly_impressions"]
        clicks      = int(impressions * p["click_rate"])
        acquisitions = int(clicks * p["conversion_rate"])

        # Chi phí chiến dịch = số lượng acquisition × CAC
        campaign_spend = acquisitions * p["cac"]

        # LTV ước tính = CAC × (1 + ROI/100)
        # Ví dụ: CAC=80K, ROI=120% → LTV = 80K × 2.2 = 176K
        ltv_estimated = p["cac"] * (1 + p["roi"] / 100)

        records.append({
            "channel":        channel,
            "impressions":    impressions,
            "clicks":         clicks,
            "acquisitions":   acquisitions,
            "cac":            p["cac"],
            "campaign_spend": campaign_spend,
            "ltv_estimated":  round(ltv_estimated, 0),
            "roi":            p["roi"],
        })

    return pd.DataFrame(records)
```

---

### 3.5 Module Đối Tác & Anomaly Detection (`load_merchant.py`)

**Mục tiêu**: Phát hiện các tài khoản cá nhân (Personal) có hành vi giống merchant — giao dịch volume lớn bất thường, nghi ngờ là "Merchant ẩn" chưa đăng ký.

#### 3.5.1 Thuật Toán Anomaly Score

Hệ thống dùng phân phối **Beta** để mô phỏng anomaly score. Đây là phân phối liên tục trên [0, 1] — phù hợp để biểu diễn "mức độ bất thường".

```python
from scipy.stats import beta as beta_dist
import numpy as np

def compute_anomaly_score(account_type: str, monthly_volume_vnd: float) -> float:
    """
    Tính anomaly score dựa trên loại tài khoản và volume giao dịch.
    
    Returns:
        float: Score trong [0, 1]. Score > 0.7 → cảnh báo rủi ro.
    """
    THRESHOLD_PERSONAL_VOLUME = 100_000_000  # 100 triệu VND/tháng

    if account_type == "Personal" and monthly_volume_vnd > THRESHOLD_PERSONAL_VOLUME:
        # Beta(5, 2): Phân phối lệch phải → score cao (0.6–0.95)
        # α=5 (shape kéo về 1), β=2 (shape kéo về 0 nhẹ)
        # Mean = α/(α+β) = 5/7 ≈ 0.71
        score = beta_dist.rvs(a=5, b=2)
    else:
        # Beta(1, 5): Phân phối lệch trái → score thấp (0.05–0.4)
        # Mean = 1/6 ≈ 0.17
        score = beta_dist.rvs(a=1, b=5)

    return round(float(score), 4)


# --- Ví dụ áp dụng cho DataFrame ---
def flag_anomalous_merchants(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["anomaly_score"] = df.apply(
        lambda row: compute_anomaly_score(row["account_type"], row["monthly_volume"]),
        axis=1
    )
    df["is_suspicious"] = df["anomaly_score"] > 0.70
    df["risk_label"] = df["anomaly_score"].apply(
        lambda s: "🔴 Cao" if s > 0.70 else ("🟡 Trung bình" if s > 0.40 else "🟢 Thấp")
    )
    return df
```

> **Giải thích tham số Beta**:
> - `Beta(α, β)` với α >> β → phân phối lệch về phía 1 (score cao).
> - `Beta(α, β)` với β >> α → phân phối lệch về phía 0 (score thấp).
> - Tăng α lên 8 nếu muốn score của tài khoản nghi ngờ tập trung hơn ở vùng 0.8–0.95.

---

## 4. Database Schema

### 4.1 Toàn Bộ Các Bảng

```sql
-- Bảng tổng hợp giao dịch theo tháng/vùng
CREATE TABLE dss_transaction_summary (
    year INTEGER, month INTEGER, region TEXT,
    gmv REAL, txn_count INTEGER,
    fraud_count INTEGER, fraud_rate REAL
);

-- Bảng danh mục tín dụng theo phân khúc
CREATE TABLE dss_credit_portfolio (
    segment TEXT, users INTEGER, credit_limit REAL,
    utilization_rate REAL, total_outstanding REAL,
    npl_rate REAL, npl_amount REAL
);

-- Bảng funnel marketing theo kênh
CREATE TABLE dss_marketing_funnel (
    channel TEXT, impressions INTEGER, clicks INTEGER,
    acquisitions INTEGER, cac REAL,
    campaign_spend REAL, ltv_estimated REAL, roi REAL
);

-- Bảng merchant với anomaly score
CREATE TABLE dss_merchant_analysis (
    merchant_id TEXT, account_type TEXT, merchant_name TEXT,
    region TEXT, monthly_volume REAL, support_count INTEGER,
    revenue REAL, anomaly_score REAL,
    is_suspicious BOOLEAN, risk_label TEXT
);

-- Bảng CSKH
CREATE TABLE dss_cskh_summary (
    year INTEGER, month INTEGER,
    total_tickets INTEGER, resolved_tickets INTEGER,
    avg_resolution_hours REAL, csat_score REAL
);
```

### 4.2 Index Tối Ưu Query

```sql
-- Index cho query filter theo thời gian (dùng nhiều nhất trên Dashboard)
CREATE INDEX idx_txn_year_month  ON dss_transaction_summary(year, month);
CREATE INDEX idx_txn_region      ON dss_transaction_summary(region);

-- Index cho merchant anomaly lookup
CREATE INDEX idx_merchant_suspicious ON dss_merchant_analysis(is_suspicious);
CREATE INDEX idx_merchant_score      ON dss_merchant_analysis(anomaly_score DESC);
```

---

## 5. Data Access Layer (DAL)

File: `streamlit_app/components/dss_data_access.py`

DAL là lớp trung gian duy nhất được phép thực thi SQL. Mọi page Streamlit chỉ được import từ DAL — không bao giờ viết SQL trực tiếp trong file page.

```python
import sqlite3
import pandas as pd
from typing import Optional, List

DB_PATH = "artifacts/fraud_system.db"


def _get_connection() -> sqlite3.Connection:
    """Trả về connection SQLite. Trong môi trường production, thay bằng SQLAlchemy engine."""
    return sqlite3.connect(DB_PATH)


def get_transaction_summary(
    years: Optional[List[int]] = None,
    regions: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Lấy dữ liệu giao dịch tổng hợp, có thể filter theo năm và vùng.
    
    Args:
        years:   Danh sách năm cần lấy, ví dụ [2023, 2024]. None = lấy tất cả.
        regions: Danh sách vùng, ví dụ ["Miền Nam"]. None = lấy tất cả.
    
    Returns:
        DataFrame với các cột: year, month, region, gmv, txn_count, fraud_count, fraud_rate
    """
    conn = _get_connection()
    query = "SELECT * FROM dss_transaction_summary WHERE 1=1"
    params = []

    if years:
        placeholders = ",".join("?" * len(years))
        query += f" AND year IN ({placeholders})"
        params.extend(years)

    if regions:
        placeholders = ",".join("?" * len(regions))
        query += f" AND region IN ({placeholders})"
        params.extend(regions)

    query += " ORDER BY year, month, region"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_credit_portfolio() -> pd.DataFrame:
    conn = _get_connection()
    df = pd.read_sql_query("SELECT * FROM dss_credit_portfolio", conn)
    conn.close()
    return df


def get_marketing_funnel() -> pd.DataFrame:
    conn = _get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM dss_marketing_funnel ORDER BY acquisitions DESC", conn
    )
    conn.close()
    return df


def get_suspicious_merchants(min_score: float = 0.70) -> pd.DataFrame:
    conn = _get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM dss_merchant_analysis WHERE anomaly_score >= ? ORDER BY anomaly_score DESC",
        conn, params=[min_score]
    )
    conn.close()
    return df


def get_merchant_bcg_data() -> pd.DataFrame:
    """Lấy dữ liệu để vẽ BCG Matrix: revenue, support_count, merchant_name."""
    conn = _get_connection()
    df = pd.read_sql_query(
        "SELECT merchant_name, revenue, support_count, account_type, region FROM dss_merchant_analysis",
        conn
    )
    conn.close()
    return df
```

> **Hướng dẫn chuyển sang PostgreSQL/BigQuery**: Thay `sqlite3.connect(DB_PATH)` bằng `sqlalchemy.create_engine("postgresql://user:pass@host/db")` và `pd.read_sql_query(query, engine)`. Phần còn lại của codebase **không thay đổi gì**.

---

## 6. Dashboard — Triển Khai Từng Module

### 6.1 Cấu Hình Toàn Cục Streamlit

Đặt ở `run_dashboard.py` hoặc đầu file `app.py`:

```python
import streamlit as st

st.set_page_config(
    page_title="Fraud Detection DSS",
    page_icon="🛡️",
    layout="wide",            # Bắt buộc để các biểu đồ có đủ không gian
    initial_sidebar_state="expanded",
)

# Inject Dark Mode + Glassmorphism CSS toàn cục
st.markdown("""
<style>
    /* Dark background tổng thể */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Glassmorphism cho các container */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    
    /* Ẩn Streamlit footer mặc định */
    footer { visibility: hidden; }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #161b22;
    }
</style>
""", unsafe_allow_html=True)
```

---

### 6.2 Module 1: Marketing & Growth (`1_Marketing_Growth.py`)

#### A. Phễu Chuyển Đổi (Conversion Funnel)

**Mục tiêu hiển thị**: CEO thấy ngay tầng nào có tỷ lệ rơi rụng (drop-off) lớn nhất.

```python
import streamlit as st
import plotly.graph_objects as go
from components.dss_data_access import get_marketing_funnel

def render_funnel_chart():
    df = get_marketing_funnel()
    
    # Chọn kênh để so sánh (sidebar filter)
    selected_channel = st.sidebar.selectbox(
        "Chọn kênh",
        options=df["channel"].tolist(),
        index=0,
    )
    
    row = df[df["channel"] == selected_channel].iloc[0]
    
    # Dữ liệu 3 tầng phễu
    stages = ["Hiển thị (Impressions)", "Nhấp (Clicks)", "Đăng ký (Registrations)"]
    values = [row["impressions"], row["clicks"], row["acquisitions"]]
    
    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        
        # Hiển thị số tuyệt đối và % so với bước đầu tiên bên trong thanh
        textposition="inside",
        textinfo="value+percent initial",
        
        # Màu gradient từ xanh lá → vàng → đỏ cam
        marker=dict(
            color=["#00C49F", "#FFBB28", "#FF8042"],
            line=dict(width=2, color="rgba(255,255,255,0.2)")
        ),
        
        # Tooltip khi hover
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Số lượng: %{x:,.0f}<br>"
            "Tỷ lệ so với đầu phễu: %{percentInitial:.1%}<br>"
            "<extra></extra>"    # Ẩn legend phụ trong tooltip
        ),
        
        connector=dict(
            line=dict(color="rgba(255,255,255,0.1)", width=1)
        ),
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Phễu Chuyển Đổi — {selected_channel}",
            font=dict(size=18, color="#ffffff"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",   # Nền trong suốt (dark mode)
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Insight tự động bên dưới biểu đồ
    click_rate = row["clicks"] / row["impressions"]
    conv_rate  = row["acquisitions"] / row["clicks"]
    
    col1, col2 = st.columns(2)
    col1.metric("CTR (Click-through Rate)", f"{click_rate:.1%}", 
                delta=f"{click_rate - 0.05:.1%} so với benchmark")
    col2.metric("CVR (Conversion Rate)",    f"{conv_rate:.1%}",
                delta=f"{conv_rate - 0.15:.1%} so với benchmark")
```

> **Tinh chỉnh key parameters của `go.Funnel`**:
> - `textinfo="value+percent initial"` → hiển thị cả số tuyệt đối lẫn % so với đỉnh phễu
> - `textinfo="value+percent previous"` → nếu muốn % so với tầng liền trước (tỷ lệ rơi rụng từng bước)
> - `connector` → đường nối giữa các tầng, đặt `fillcolor` để tô màu vùng rơi rụng

---

#### B. Dự Báo Tăng Trưởng (Growth Forecasting)

**Mục tiêu**: Trình bày 3 kịch bản (Conservative / Baseline / Optimistic) dựa trên tốc độ tăng trưởng lịch sử.

```python
import numpy as np
import plotly.graph_objects as go

def render_growth_forecast():
    # Lấy dữ liệu lịch sử
    df = get_transaction_summary()
    
    # Tính tổng customers (acquisitions) theo năm
    df_annual = (df.groupby("year")["acquisitions"].sum().reset_index()
                   if "acquisitions" in df.columns
                   else _compute_from_marketing())
    
    # --- Tính tốc độ tăng trưởng lịch sử ---
    annual_values = df_annual["acquisitions"].values
    growth_rates = np.diff(annual_values) / annual_values[:-1]
    
    # 3 kịch bản dựa trên quantile của growth rate lịch sử
    # Nếu chỉ có 1-2 data points, dùng giá trị cứng
    rate_conservative = np.percentile(growth_rates, 25) if len(growth_rates) > 1 else 0.10
    rate_baseline     = np.percentile(growth_rates, 50) if len(growth_rates) > 1 else 0.15
    rate_optimistic   = np.percentile(growth_rates, 75) if len(growth_rates) > 1 else 0.22
    
    # --- Dự báo 3 năm tới ---
    last_year  = int(df_annual["year"].max())
    last_value = float(df_annual["acquisitions"].iloc[-1])
    forecast_years = [last_year + i for i in range(1, 4)]
    
    def forecast(rate, n_years):
        # Compound growth: V_n = V_0 × (1 + rate)^n
        return [round(last_value * (1 + rate) ** n, 0) for n in range(1, n_years + 1)]
    
    fig = go.Figure()
    
    # Thanh lịch sử (màu trung tính)
    fig.add_trace(go.Bar(
        x=df_annual["year"].tolist(),
        y=df_annual["acquisitions"].tolist(),
        name="Thực tế",
        marker_color="#4A90D9",
        opacity=0.9,
    ))
    
    # Thanh dự báo 3 kịch bản (cùng group, màu khác biệt)
    scenarios = [
        ("Thận trọng", rate_conservative, "#FF6B6B"),
        ("Cơ bản",    rate_baseline,     "#FFD93D"),
        ("Tích cực",  rate_optimistic,   "#6BCB77"),
    ]
    
    for name, rate, color in scenarios:
        fig.add_trace(go.Bar(
            x=forecast_years,
            y=forecast(rate, 3),
            name=f"Dự báo — {name} ({rate:.0%}/năm)",
            marker_color=color,
            opacity=0.75,
        ))
    
    fig.update_layout(
        title="Dự Báo Tăng Trưởng Khách Hàng (3 Kịch Bản)",
        barmode="group",            # Thanh đứng cạnh nhau thay vì chồng lên
        xaxis=dict(
            title="Năm",
            tickmode="linear",
            dtick=1,
        ),
        yaxis=dict(
            title="Số Khách Hàng",
            tickformat=",",         # Format dấu phẩy ngàn
        ),
        legend=dict(
            orientation="h",        # Legend nằm ngang bên dưới
            yanchor="bottom", y=-0.3,
            xanchor="center", x=0.5,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"),
        height=450,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Bảng tóm tắt số liệu dự báo
    with st.expander("📊 Chi tiết số liệu dự báo"):
        summary_data = {}
        for name, rate, _ in scenarios:
            summary_data[name] = forecast(rate, 3)
        st.dataframe(
            pd.DataFrame(summary_data, index=forecast_years),
            use_container_width=True,
        )
```

---

### 6.3 Module 2: Tín Dụng & NPL (`2_Credit_NPL.py`)

#### A. Đồng Hồ NPL (NPL Gauge Chart)

**Mục tiêu**: Hiển thị tỷ lệ nợ xấu tổng thể với hệ thống cảnh báo màu sắc tức thì.

```python
import plotly.graph_objects as go

def render_npl_gauge(npl_rate: float):
    """
    Args:
        npl_rate: Tỷ lệ NPL tổng danh mục, ví dụ 0.028 = 2.8%
    """
    npl_pct = npl_rate * 100   # Chuyển sang dạng %

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=npl_pct,
        number=dict(
            suffix="%",
            font=dict(size=40, color="#ffffff"),
            valueformat=".2f",
        ),
        delta=dict(
            reference=3.0,      # Ngưỡng cảnh báo 3%
            increasing=dict(color="#FF4B4B"),  # Tăng so với ngưỡng → đỏ
            decreasing=dict(color="#00C49F"),  # Giảm → xanh
            valueformat=".2f",
            suffix="%",
        ),
        gauge=dict(
            axis=dict(
                range=[0, 10],   # Dải 0–10%
                tickwidth=1,
                tickcolor="#ffffff",
                tickformat=".0f",
                ticksuffix="%",
                nticks=11,
            ),
            bar=dict(
                color="#4A90D9",
                thickness=0.3,
            ),
            # Dải màu cảnh báo
            steps=[
                dict(range=[0, 3], color="rgba(0, 196, 159, 0.2)"),    # Xanh lá: An toàn
                dict(range=[3, 5], color="rgba(255, 187, 40, 0.2)"),   # Vàng: Chú ý
                dict(range=[5, 10], color="rgba(255, 75, 75, 0.2)"),   # Đỏ: Nguy hiểm
            ],
            # Đường ngưỡng cảnh báo sớm tại 3%
            threshold=dict(
                line=dict(color="#FFD93D", width=3),
                thickness=0.75,
                value=3.0,
            ),
        ),
        title=dict(
            text="Tỷ Lệ Nợ Xấu (NPL)<br><span style='font-size:14px;color:#aaa'>Ngưỡng cảnh báo: 3%</span>",
            font=dict(size=20, color="#ffffff"),
        ),
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"),
        height=300,
        margin=dict(l=30, r=30, t=80, b=30),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cảnh báo ngữ cảnh bên dưới
    if npl_pct >= 5:
        st.error(f"⚠️ NPL {npl_pct:.2f}% vượt ngưỡng nguy hiểm (5%). Cần xem xét cắt giảm hạn mức.")
    elif npl_pct >= 3:
        st.warning(f"🟡 NPL {npl_pct:.2f}% vượt ngưỡng cảnh báo (3%). Theo dõi chặt.")
    else:
        st.success(f"✅ NPL {npl_pct:.2f}% trong vùng an toàn (< 3%).")
```

> **Tinh chỉnh**: Đổi `range=[0, 10]` thành `[0, 5]` nếu danh mục rủi ro thấp để tăng độ phân giải hiển thị. Thêm `threshold` thứ hai tại 5% nếu muốn 2 ngưỡng cảnh báo.

---

#### B. Biểu Đồ Phân Tầng Rủi Ro Tín Dụng

```python
def render_credit_risk_breakdown():
    df = get_credit_portfolio()
    
    # Sắp xếp theo NPL rate giảm dần để segment rủi ro nhất nằm trên cùng
    df = df.sort_values("npl_rate", ascending=True)
    
    fig = go.Figure()
    
    # Thanh ngang: Dư nợ theo segment
    fig.add_trace(go.Bar(
        y=df["segment"],
        x=df["total_outstanding"] / 1e9,   # Đổi sang tỷ VND
        orientation="h",
        name="Dư nợ (tỷ VND)",
        marker=dict(
            color=df["npl_rate"],           # Màu gradient theo NPL rate
            colorscale="RdYlGn_r",          # Đỏ = NPL cao, Xanh = NPL thấp
            colorbar=dict(
                title="NPL Rate",
                tickformat=".1%",
            ),
            showscale=True,
        ),
        text=[f"{v:.1f} tỷ" for v in df["total_outstanding"] / 1e9],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Dư nợ: %{x:.1f} tỷ VND<br>"
            "NPL: %{marker.color:.1%}<br>"
            "<extra></extra>"
        ),
    ))
    
    fig.update_layout(
        title="Phân Tích Danh Mục Tín Dụng Theo Phân Khúc",
        xaxis=dict(title="Dư Nợ (tỷ VND)", tickformat=",.0f"),
        yaxis=dict(title=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"),
        height=400,
    )
    
    st.plotly_chart(fig, use_container_width=True)
```

---

### 6.4 Module 3: Phân Tích Đối Tác (`3_Merchant_Analysis.py`)

#### A. Ma Trận BCG (BCG Matrix)

**Mục tiêu**: Phân loại đối tác vào 4 nhóm chiến lược dựa trên Revenue và User Base.

```python
import plotly.express as px

def render_bcg_matrix():
    df = get_merchant_bcg_data()
    
    # --- Tính ngưỡng phân loại (dùng median) ---
    median_revenue = df["revenue"].median()
    median_users   = df["support_count"].median()
    
    # Phân loại BCG
    def classify_bcg(row):
        high_rev   = row["revenue"]       >= median_revenue
        high_users = row["support_count"] >= median_users
        if high_rev and high_users:
            return "⭐ Star"
        elif high_rev and not high_users:
            return "🐄 Cash Cow"
        elif not high_rev and high_users:
            return "❓ Question Mark"
        else:
            return "🐕 Dog"
    
    df["bcg_category"] = df.apply(classify_bcg, axis=1)
    
    # Màu cho từng quadrant
    color_map = {
        "⭐ Star":          "#FFD700",
        "🐄 Cash Cow":      "#00C49F",
        "❓ Question Mark": "#4A90D9",
        "🐕 Dog":           "#FF6B6B",
    }
    
    fig = px.scatter(
        df,
        x="support_count",                      # Trục X: User base
        y="revenue",                             # Trục Y: Revenue
        size="revenue",                          # Kích thước bong bóng ~ Revenue
        color="bcg_category",                    # Màu ~ BCG category
        color_discrete_map=color_map,
        hover_name="merchant_name",              # Tên hiển thị khi hover
        hover_data={
            "support_count": ":,",
            "revenue": ":,.0f",
            "bcg_category": True,
            "account_type": True,
        },
        labels={
            "support_count": "Số Lượng Người Dùng",
            "revenue": "Doanh Thu (VND)",
        },
        title="Ma Trận BCG — Phân Loại Đối Tác",
        size_max=60,     # Bong bóng lớn nhất = 60px — điều chỉnh nếu có outlier
    )
    
    # Vẽ đường kẻ median để chia 4 quadrant
    fig.add_vline(
        x=median_users,
        line_dash="dash",
        line_color="rgba(255,255,255,0.4)",
        annotation_text="Median Users",
        annotation_position="top right",
    )
    fig.add_hline(
        y=median_revenue,
        line_dash="dash",
        line_color="rgba(255,255,255,0.4)",
        annotation_text="Median Revenue",
        annotation_position="right",
    )
    
    # Label 4 góc
    annotations_cfg = dict(showarrow=False, font=dict(size=11, color="rgba(255,255,255,0.5)"))
    x_max = df["support_count"].max() * 0.95
    y_max = df["revenue"].max() * 0.95
    
    for text, x, y in [
        ("STAR", x_max, y_max),
        ("CASH COW", median_users * 0.1, y_max),
        ("QUESTION MARK", x_max, median_revenue * 0.1),
        ("DOG", median_users * 0.1, median_revenue * 0.1),
    ]:
        fig.add_annotation(text=text, x=x, y=y, **annotations_cfg)
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
        font=dict(color="#ffffff"),
        height=550,
        legend=dict(
            title="Phân Loại BCG",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
        ),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Bảng tóm tắt theo category
    summary = (df.groupby("bcg_category")
                 .agg(count=("merchant_name", "count"),
                      total_revenue=("revenue", "sum"),
                      avg_users=("support_count", "mean"))
                 .reset_index())
    
    st.dataframe(
        summary.style.format({
            "total_revenue": "{:,.0f}",
            "avg_users": "{:,.0f}",
        }),
        use_container_width=True,
    )
```

> **Tinh chỉnh key parameters**:
> - `size_max=60` → Tăng lên 80-100 nếu biểu đồ rộng và muốn bong bóng nổi bật hơn
> - Thay `size="revenue"` bằng `size="support_count"` nếu muốn kích thước bong bóng thể hiện quy mô user
> - `color_discrete_map` → Đổi màu theo brand guideline của đối tác

---

#### B. Danh Sách Merchant Nghi Ngờ

```python
def render_suspicious_merchant_table():
    st.subheader("🔴 Tài Khoản Nghi Ngờ Là Merchant Ẩn")
    
    # Thanh trượt điều chỉnh ngưỡng anomaly score
    threshold = st.slider(
        "Ngưỡng Anomaly Score",
        min_value=0.0, max_value=1.0,
        value=0.70, step=0.05,
        help="Giá trị cao hơn = lọc chặt hơn, chỉ hiện những trường hợp rõ ràng nhất",
    )
    
    df = get_suspicious_merchants(min_score=threshold)
    
    if df.empty:
        st.info("✅ Không có tài khoản nào vượt ngưỡng đã chọn.")
        return
    
    # Tô màu cột anomaly_score theo mức độ
    def color_score(val):
        if val >= 0.85:
            return "background-color: rgba(255,75,75,0.3); color: #FF4B4B; font-weight: bold"
        elif val >= 0.70:
            return "background-color: rgba(255,187,40,0.3); color: #FFB800"
        return ""
    
    styled_df = (df[["merchant_name", "account_type", "region",
                      "monthly_volume", "anomaly_score", "risk_label"]]
                   .rename(columns={
                       "merchant_name": "Tên Tài Khoản",
                       "account_type": "Loại TK",
                       "region": "Vùng",
                       "monthly_volume": "Volume/Tháng (VND)",
                       "anomaly_score": "Điểm Bất Thường",
                       "risk_label": "Mức Độ Rủi Ro",
                   })
                   .style
                   .applymap(color_score, subset=["Điểm Bất Thường"])
                   .format({"Volume/Tháng (VND)": "{:,.0f}", "Điểm Bất Thường": "{:.3f}"}))
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    st.caption(f"Hiển thị {len(df)} tài khoản có anomaly score ≥ {threshold:.2f}")
```

---

### 6.5 Module 4: Executive Summary (`4_Executive_Summary.py`)

#### CEO Command Panel

Panel cho phép CEO nhập chỉ thị văn bản và nhận phản hồi phân tích tức thì, sử dụng `st.session_state` để duy trì lịch sử.

```python
def render_ceo_command_panel():
    st.subheader("🎯 Bảng Điều Khiển CEO")
    
    # Khởi tạo session state nếu chưa có
    if "command_history" not in st.session_state:
        st.session_state.command_history = []
    
    # Các chỉ thị mẫu nhanh
    quick_commands = [
        "Tóm tắt tình hình NPL tuần này",
        "Top 5 merchant nghi ngờ gian lận",
        "So sánh hiệu quả kênh marketing",
        "Dự báo doanh thu quý tới",
    ]
    
    st.write("**Chỉ thị nhanh:**")
    cols = st.columns(len(quick_commands))
    for i, cmd in enumerate(quick_commands):
        if cols[i].button(cmd, key=f"quick_cmd_{i}"):
            st.session_state.pending_command = cmd
    
    # Input thủ công
    user_input = st.text_input(
        "Hoặc nhập chỉ thị:",
        value=st.session_state.get("pending_command", ""),
        placeholder="Ví dụ: 'Các kênh nào đang có ROI dưới 100%?'",
        key="command_input",
    )
    
    if st.button("▶ Thực hiện", type="primary") and user_input:
        # Tạo phản hồi dựa trên logic phân tích (không dùng LLM)
        response = _generate_command_response(user_input)
        
        # Lưu vào lịch sử
        st.session_state.command_history.insert(0, {
            "command": user_input,
            "response": response,
            "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
        })
        
        # Xóa pending command
        if "pending_command" in st.session_state:
            del st.session_state["pending_command"]
    
    # Hiển thị lịch sử
    for entry in st.session_state.command_history[:5]:  # Chỉ hiện 5 gần nhất
        with st.expander(f"[{entry['timestamp']}] {entry['command'][:50]}...", expanded=False):
            st.markdown(entry["response"])


def _generate_command_response(command: str) -> str:
    """Sinh phản hồi dựa trên keyword matching và dữ liệu thực từ DB."""
    command_lower = command.lower()
    
    if "npl" in command_lower or "nợ xấu" in command_lower:
        df = get_credit_portfolio()
        total_npl = df["npl_amount"].sum()
        worst_segment = df.loc[df["npl_rate"].idxmax(), "segment"]
        return (f"**NPL Tổng Danh Mục**: {total_npl/1e9:.1f} tỷ VND\n\n"
                f"**Phân khúc rủi ro nhất**: {worst_segment} "
                f"(NPL: {df['npl_rate'].max():.1%})\n\n"
                f"*Khuyến nghị*: Xem xét siết chặt điều kiện vay với nhóm này.")
    
    elif "roi" in command_lower or "marketing" in command_lower:
        df = get_marketing_funnel()
        low_roi = df[df["roi"] < 100][["channel", "roi"]]
        if low_roi.empty:
            return "✅ Tất cả kênh marketing đang có ROI > 100%."
        return f"**Kênh ROI thấp (<100%)**:\n{low_roi.to_markdown(index=False)}"
    
    else:
        return f"Đã nhận chỉ thị: *{command}*. Vui lòng xem các module liên quan để biết thêm chi tiết."
```

---

## 7. Shared UI Components

File: `streamlit_app/components/shared_ui.py`

### 7.1 KPI Card Component

```python
def render_kpi_card(
    title: str,
    value: str,
    delta: str = None,
    delta_type: str = "normal",   # "normal" | "inverse" | "off"
    icon: str = "📊",
    subtitle: str = None,
) -> None:
    """
    Hiển thị một KPI card với glassmorphism style.
    
    Args:
        title:      Tiêu đề KPI, ví dụ "Tổng GMV"
        value:      Giá trị hiển thị, ví dụ "1,234 tỷ VND"
        delta:      Thay đổi so với kỳ trước, ví dụ "+12.3%"
        delta_type: "normal" = tăng là tốt; "inverse" = tăng là xấu (dùng cho NPL, fraud)
        icon:       Emoji icon
        subtitle:   Dòng chú thích nhỏ bên dưới
    """
    # Xác định màu delta
    if delta:
        is_positive = delta.startswith("+")
        if delta_type == "inverse":
            delta_color = "#FF4B4B" if is_positive else "#00C49F"
        elif delta_type == "off":
            delta_color = "#888888"
        else:
            delta_color = "#00C49F" if is_positive else "#FF4B4B"
        delta_html = f'<span style="color:{delta_color};font-size:14px;font-weight:bold">{delta}</span>'
    else:
        delta_html = ""
    
    subtitle_html = f'<p style="color:#888;font-size:11px;margin:4px 0 0 0">{subtitle}</p>' if subtitle else ""
    
    card_html = f"""
    <div class="glass-card" style="
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 20px 24px;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    ">
        <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <p style="color:#aaa;font-size:13px;margin:0;text-transform:uppercase;
                      letter-spacing:0.5px">{icon} {title}</p>
            {delta_html}
        </div>
        <div>
            <p style="color:#ffffff;font-size:28px;font-weight:700;margin:0;
                      line-height:1.1">{value}</p>
            {subtitle_html}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


# --- Ví dụ sử dụng ---
# col1, col2, col3, col4 = st.columns(4)
# 
# with col1:
#     render_kpi_card("Tổng GMV", "2,340 tỷ", "+8.2%", icon="💰",
#                     subtitle="So với tháng trước")
# with col2:
#     render_kpi_card("Tỷ lệ Gian lận", "1.87%", "-0.15%", delta_type="inverse",
#                     icon="🛡️", subtitle="YoY giảm 0.15pp")
# with col3:
#     render_kpi_card("Tỷ lệ NPL", "2.3%", "+0.2%", delta_type="inverse",
#                     icon="⚠️", subtitle="Ngưỡng: 3%")
# with col4:
#     render_kpi_card("Tổng Khách Hàng", "521,000", "+15.2%", icon="👥")
```

### 7.2 Section Header Component

```python
def render_section_header(title: str, description: str = None):
    """Tiêu đề section có đường kẻ gradient bên dưới."""
    desc_html = f'<p style="color:#888;font-size:14px;margin:4px 0 0 0">{description}</p>' if description else ""
    st.markdown(f"""
    <div style="margin: 32px 0 20px 0">
        <h2 style="color:#ffffff;margin:0;font-weight:700">{title}</h2>
        {desc_html}
        <div style="height:2px;background:linear-gradient(90deg,#4A90D9,transparent);
                    margin-top:8px;border-radius:2px"></div>
    </div>
    """, unsafe_allow_html=True)
```

---

## 8. Hướng Dẫn Vận Hành

### 8.1 Yêu Cầu Hệ Thống

| Thành phần | Phiên bản tối thiểu |
|---|---|
| Python | 3.9+ |
| SQLite | 3.35+ (bundled với Python) |
| RAM | 2GB (đủ cho SQLite + Streamlit) |
| Disk | 500MB cho database |

### 8.2 Cài Đặt Môi Trường

```powershell
# 1. Clone repository
git clone <repo_url>
cd fraud_detection_system

# 2. Tạo và kích hoạt môi trường ảo
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 3. Cài dependencies
pip install -r requirements.txt
```

**`requirements.txt` cần có:**

```
streamlit>=1.28.0
plotly>=5.17.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0          # Cho Beta distribution trong anomaly detection
sqlite3                # Bundled với Python
```

### 8.3 Quy Trình Chạy Hệ Thống

```
Lần đầu / Cập nhật dữ liệu:       Hàng ngày (production):
─────────────────────────          ─────────────────────────
python run_all_etl.py              Cron: 0 6 * * * python run_all_etl.py
        │
        ▼
python run_dashboard.py
        │
        ▼
Truy cập: http://localhost:8501
```

**Chạy từng ETL module riêng lẻ** (debug hoặc update một phần):

```powershell
python core/etl/load_paysim.py        # Cập nhật dữ liệu giao dịch
python core/etl/load_credit.py        # Cập nhật danh mục tín dụng
python core/etl/load_marketing.py     # Cập nhật funnel marketing
python core/etl/load_cskh.py          # Cập nhật dữ liệu CSKH
python core/etl/load_merchant.py      # Cập nhật phân tích đối tác
python core/etl/load_review_queue.py  # Cập nhật hàng đợi review
```

**`run_all_etl.py`** — Wrapper chạy tuần tự, có log:

```python
import subprocess
import sys
from datetime import datetime

ETL_MODULES = [
    "core/etl/load_paysim.py",
    "core/etl/load_credit.py",
    "core/etl/load_marketing.py",
    "core/etl/load_cskh.py",
    "core/etl/load_merchant.py",
    "core/etl/load_review_queue.py",
]

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"ETL Pipeline Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")
    
    for module in ETL_MODULES:
        print(f"▶ Running {module}...")
        result = subprocess.run([sys.executable, module], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  ❌ FAILED: {result.stderr}")
            sys.exit(1)
        else:
            print(f"  ✅ OK: {result.stdout.strip()}")
    
    print(f"\n{'='*50}")
    print(f"ETL Pipeline Complete: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*50}\n")
```

### 8.4 Kiểm Tra Sức Khỏe Hệ Thống

```python
# health_check.py — Chạy để verify DB sau ETL
import sqlite3
import pandas as pd

DB_PATH = "artifacts/fraud_system.db"

EXPECTED_TABLES = [
    "dss_transaction_summary",
    "dss_credit_portfolio",
    "dss_marketing_funnel",
    "dss_merchant_analysis",
]

conn = sqlite3.connect(DB_PATH)

for table in EXPECTED_TABLES:
    df = pd.read_sql_query(f"SELECT COUNT(*) as cnt FROM {table}", conn)
    cnt = df.iloc[0]["cnt"]
    status = "✅" if cnt > 0 else "❌ EMPTY"
    print(f"{status} {table}: {cnt} rows")

conn.close()
```

---

## 9. Lộ Trình Mở Rộng & Scaling

### 9.1 Chuyển Đổi Từ SQLite Sang PostgreSQL

```python
# dss_data_access.py — Chỉ cần thay đổi 1 chỗ:

# BEFORE (SQLite):
import sqlite3
def _get_connection():
    return sqlite3.connect("artifacts/fraud_system.db")

# AFTER (PostgreSQL):
from sqlalchemy import create_engine
_engine = create_engine("postgresql://user:password@host:5432/fraud_db")
def _get_connection():
    return _engine.connect()
```

### 9.2 Tích Hợp Dữ Liệu Real-time

Khi cần cập nhật Dashboard gần real-time (< 5 phút):

1. Thêm column `updated_at TIMESTAMP` vào mỗi bảng
2. ETL chuyển sang **incremental mode** (chỉ load dữ liệu mới thay vì replace toàn bộ)
3. Dùng `@st.cache_data(ttl=300)` để Streamlit tự reload data mỗi 5 phút:

```python
@st.cache_data(ttl=300)   # TTL = 300 giây = 5 phút
def get_transaction_summary(years=None, regions=None):
    # ... query code ...
```

### 9.3 Thêm Module Mới

Để thêm một module nghiệp vụ mới (ví dụ: Phân tích Rủi ro Địa lý):

1. Tạo `core/etl/load_geo_risk.py` theo khung ETL ở Mục 3.1
2. Thêm bảng mới vào DB schema (Mục 4.1)
3. Thêm hàm query vào `dss_data_access.py` (Mục 5)
4. Tạo page mới `streamlit_app/pages/5_Geo_Risk.py`
5. Streamlit tự detect file mới — không cần sửa config

---

