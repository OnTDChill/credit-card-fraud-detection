# 📊 Kịch bản Thuyết trình: Hệ thống BI & DSS Hỗ trợ Ra Quyết Định Chống Gian Lận

Tài liệu này cung cấp kịch bản thuyết trình chi tiết, tập trung vào góc độ **Business Intelligence (BI)** và hệ thống **Decision Support System (DSS)**. Cấu trúc đi từ "Bếp" (Dữ liệu đầu vào, Xử lý, Mô hình) ra tới "Bàn ăn" (Dashboard thực tế cho CEO).

---

## 1. Dữ liệu Đầu vào (Input Datasets)
Hệ thống kết hợp dữ liệu từ Kaggle và dữ liệu được sinh ra có chủ đích (Seeded) để tạo thành hệ sinh thái hoàn chỉnh cho công ty tài chính.

### 1.1 IEEE-CIS Fraud Detection (Nguồn: Kaggle)
- **Nguồn gốc**: Dữ liệu giao dịch thương mại điện tử thực tế từ Vesta Corporation (đã ẩn danh).
- **Quy mô**: ~590,000 dòng dữ liệu huấn luyện (Train) và ~500,000 dòng dữ liệu Test.
- **Label quan trọng**: `isFraud` (0 = Hợp lệ, 1 = Gian lận).
- **Feature Selection**:
  - **Giữ lại**: Thời gian (`TransactionDT`), số tiền (`TransactionAmt`), mã sản phẩm (`ProductCD`), thông tin thẻ (`card1-6`), thông tin thiết bị (`DeviceInfo`, `id_01-38`). *Lý do: Đây là những trường cốt lõi phản ánh hành vi người dùng và "chữ ký" của thiết bị.*
  - **Loại bỏ**: Các cột V (Vesta features) có tỷ lệ thiếu dữ liệu (missing rate) > 50% hoặc các cột định danh đơn thuần như `TransactionID`. *Lý do: Tránh làm mô hình bị nhiễu (noise) và bị học vẹt (overfitting).*

### 1.2 PaySim Mobile Money (Nguồn: Kaggle)
- **Nguồn gốc**: Dữ liệu mô phỏng giao dịch ví điện tử/mobile money.
- **Quy mô**: ~6.3 triệu dòng.
- **Giữ lại & Biến đổi**: Dùng cột `step` để map ra chuỗi thời gian (tháng/năm), giữ `type`, `amount`, và tạo trường `region` (vùng miền). Dữ liệu này dùng để tính toán GMV, biến động doanh thu.
- **Loại bỏ**: Số dư trước/sau (`oldbalance`, `newbalance`). *Lý do: Trong thực tế, mô hình dễ bị Data Leakage khi học các quy luật trừ tiền cơ học, làm mất đi khả năng phát hiện hành vi tinh vi.*

### 1.3 Dữ liệu Sinh thái Nội bộ (Data Seeding)
- **Nguồn gốc**: Được tạo ra thông qua các ETL Scripts nội bộ (`seed_multi_month.py`).
- **Quy mô & Cách tạo**: Hệ thống không dùng dữ liệu tĩnh 1 tháng. Chúng tôi viết script sinh dữ liệu cho **6 tháng liên tục**, áp dụng các nguyên lý thực tế:
  - *YoY/MoM Growth*: Mô phỏng tăng trưởng 3-8%/tháng của ngành Fintech.
  - *Seasonality & Jitter*: Thêm nhiễu ngẫu nhiên (random noise) và tính mùa vụ để biểu đồ không bị "thẳng tắp" một cách phi lý.
- Dữ liệu này bơm vào các mảng: Marketing (chi phí, CAC), Tín dụng (Nợ xấu NPL), Hệ sinh thái (Bán chéo), và Đối tác.

---

## 2. Các Kỹ thuật, Thuật toán & Lý thuyết áp dụng

Với mỗi kỹ thuật, chúng ta đi qua **What** (Nó là gì?), **Why** (Tại sao phải dùng?), **How** (Đã dùng thế nào?).

### 2.1 ETL Pipeline (Trích xuất & Tổng hợp)
- **What**: Xây dựng đường ống biến đổi dữ liệu (Data Pipeline).
- **Why**: Streamlit không thể (và không nên) load trực tiếp 6 triệu dòng giao dịch mỗi khi CEO đổi bộ lọc. Nếu làm vậy, dashboard sẽ sập.
- **How**: Python scripts gom nhóm (Group By) dữ liệu thô theo Tháng, Vùng miền, Phân khúc để lưu thành các bảng tổng hợp (Aggregated Tables) siêu nhẹ trong SQLite. 

### 2.2 Xử lý Imbalanced Data & Chống Overfitting
- **What**: Dữ liệu gian lận cực hiếm (chỉ < 1-3%).
- **Why**: Nếu không xử lý, mô hình AI sẽ "chơi lầy" bằng cách dự đoán 100% giao dịch là hợp lệ. Độ chính xác (Accuracy) vẫn cao 99% nhưng vô dụng trong thực tế.
- **How**: 
  - Áp dụng `Class Weighting` (Phạt thật nặng mô hình nếu nó bỏ lọt giao dịch gian lận).
  - Sử dụng **Time-based Split** (chia dữ liệu train/test theo mốc thời gian) thay vì Random Split, nhằm mô phỏng thực tế "dùng quá khứ đoán tương lai".

### 2.3 Machine Learning (Decision Engine)
- **What**: Lớp AI trí tuệ nhân tạo đứng sau hệ thống.
- **Why**: Luật quy tắc (Rule-based) do con người viết ra không thể bắt kịp các mánh khóe tội phạm ngày càng tinh vi. Cần AI tìm ra các pattern ẩn.
- **How**:
  - Dùng **Random Forest/LightGBM** phân tích xác suất gian lận (`fraud_probability`) cho từng giao dịch.
  - Dùng **Isolation Forest** (Unsupervised Learning) chấm điểm bất thường (`anomaly_score`) để tìm ra các đối tác Merchant đáng ngờ, ngay cả khi ta chưa có nhãn dữ liệu của họ.

### 2.4 Đánh giá Mô hình theo góc nhìn Kinh doanh (BI & Technical)
- **Baseline (Logistic Regression)**: Chúng tôi sử dụng mô hình Logistic Regression làm "vạch xuất phát" để so sánh. Điều này giúp CEO thấy được giá trị tăng thêm của các mô hình AI phức tạp so với các phương pháp thống kê truyền thống.
- **Cost of Error (Chi phí sai lầm)**: Thay vì chỉ nhìn vào độ chính xác (Accuracy) khô khan, hệ thống tính toán thiệt hại thực tế bằng tiền (VND):
  - *Thiệt hại từ FN*: Giá trị giao dịch bị mất khi bỏ lọt gian lận.
  - *Thiệt hại từ FP*: Chi phí để thu hút lại một khách hàng mới nếu chúng ta chặn nhầm và làm họ rời bỏ nền tảng (Churn).
- **Mục tiêu**: Chọn mô hình có **Tổng chi phí sai lầm thấp nhất**, không chỉ là mô hình có điểm số cao nhất.

---

## 3. Sử dụng Dashboard Trong Thực Tế (BI & DSS)

Mọi con số, biểu đồ trên Dashboard đều được **tính toán từ dữ liệu thực tế** trong DB (Data-driven), không phải số liệu vẽ tĩnh (decoration).

### 3.1 Tab: Tổng quan (Executive Summary)
- **What**: Bức tranh 360 độ về Doanh thu, An toàn và Khách hàng.
- **Why**: CEO chỉ có 1-2 phút mỗi ngày để lướt xem sức khỏe doanh nghiệp, họ cần thông tin đã được cô đọng nhất.
- **How**: KPI Cards so sánh tăng trưởng MoM, hệ thống cảnh báo tự động thông minh (Ví dụ: báo động đỏ nếu Tỷ lệ gian lận vượt ngưỡng an toàn).

### 3.2 Tab: Thu hút & Kích hoạt (Marketing)
- **What**: Phễu khách hàng, theo dõi CAC (Chi phí có 1 khách hàng) và kịch bản tăng trưởng.
- **Why**: Đảm bảo công ty không "đốt tiền" marketing vô ích.
- **How (DSS)**: CEO có thể kéo thanh trượt "Thay đổi ngân sách". Ngay lập tức hệ thống dùng hệ số 탄 hồi (elasticity) để dự báo khách hàng mới và thay đổi CAC. Các dự báo 3 năm được gắn liền với **giả định bối cảnh vĩ mô thực tế** (Kinh tế giảm tốc, Lãi suất tăng...) để CEO không bị ảo tưởng bởi các mức tăng trưởng cao.

### 3.3 Tab: Tín dụng & Dòng tiền
- **What**: Bảng điều khiển rủi ro cho vay.
- **Why**: Nợ xấu (NPL - Non-performing loan) là căn bệnh ung thư của ngành tài chính, cần phát hiện sớm.
- **How**: Dashboard thay thế các bảng kỹ thuật khô khan bằng định dạng CEO-friendly (cảnh báo Rủi ro cao/An toàn). Biểu đồ xu hướng nợ xấu được giải thích bằng các yếu tố thực tế (VD: nợ tăng do tình hình kinh tế chung). Kèm theo Command Panel để CEO ra chỉ thị trực tiếp.

### 3.4 Tab: Hệ sinh thái Dịch vụ
- **What**: Phân tích hành vi bán chéo (Cross-sell).
- **Why**: Tăng LTV (Life-time Value) thay vì chỉ đi tìm khách mới.
- **How**: Sử dụng thuật toán Association Rule (Support, Lift). Trực quan hóa bằng ma trận BCG (Bò sữa, Ngôi sao, Chó mực) giúp CEO quyết định "Bơm tiền" cho dịch vụ nào và "Cắt bỏ" dịch vụ nào.

### 3.5 Tab: Đối tác (Merchant) & Tab Giao dịch
- **What**: Quản lý Merchant và luồng tiền GMV.
- **Why**: Chống gian lận rửa tiền hoặc trục lợi khuyến mãi từ đối tác.
- **How**: Drill-down từ tổng quan GMV đến các khu vực địa lý có tỷ lệ Fraud cao. Cung cấp dự báo rủi ro gian lận 4 quý tới dựa trên các giả định thực tế (Đầu tư AI phòng chống, Tội phạm mạng deepfake gia tăng).

### 3.6 Các Tab Kỹ thuật (Ops & Tech)
- **Review Queue**: Khẳng định nguyên lý "AI trợ lý, Con người quyết định" (Human-in-the-loop). AI đẩy các ca nghi ngờ 50-50 vào hàng đợi để Ops duyệt thủ công.
- **Threshold Config**: Giao diện để CEO điều chỉnh thanh gạt rủi ro (Nới lỏng để tăng doanh thu, hay Siết chặt để đảm bảo an toàn). Mọi quyết định kéo thanh gạt lập tức hiện ra tác động: Tỷ lệ chặn nhầm (False Positive) là bao nhiêu.

---

## 4. Đánh giá: Product đã Production-Ready chưa?

**Kết luận**: Sản phẩm hiện tại là một **End-to-End POC (Proof of Concept) ở mức độ xuất sắc (Pre-Production Ready)**. Nó minh họa hoàn hảo luồng tư duy và công nghệ của một công ty Fintech thực tế.

**Những điểm đã "Ready" (Sẵn sàng):**
1. **Hoàn toàn Data-driven**: Mọi UI đều render từ data thật trong DB. Đổi data, UI sẽ tự đổi.
2. **Kiến trúc phân lớp chuẩn**: Tách bạch rõ rệt lớp ETL (Pipeline), lớp Lưu trữ (SQLite DB), Lớp truy xuất (Data Access), Lớp logic kinh doanh (Engine) và Lớp UI (Streamlit).
3. **Luồng nghiệp vụ trơn tru**: Có đầy đủ vòng lặp Machine Learning -> Rule Engine -> Review Queue -> Feedback.

**Future Work (Những thứ cần bổ sung để thành Production 100% thực tế):**
1. **Hạ tầng Dữ liệu (Infrastructure)**: Chuyển từ SQLite sang các Cloud Data Warehouse thực thụ (như PostgreSQL, Google BigQuery, Snowflake).
2. **Real-time Inference**: Hiện tại là Batch processing. Thực tế cần thiết lập hệ thống streaming bằng Kafka và API bằng FastAPI để chốt giao dịch trong dưới 50ms.
3. **MLOps**: Cần tích hợp MLflow để tự động phát hiện Data Drift và kích hoạt luồng Retrain mô hình tự động.
4. **Security & RBAC**: Phân quyền Role-based Access Control rõ ràng (CEO chỉ xem Dashboard, Ops chỉ xem Review Queue). Thêm cơ chế SSO và lưu Audit Log chặt chẽ cho mọi tác vụ phê duyệt.