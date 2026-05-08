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

### 2.5 Chi tiết Thuật toán & Tham số tinh chỉnh (DSS Engine)
Hệ thống Decision Support System (DSS) không chỉ dự báo mà còn cung cấp cơ chế "Tối ưu hóa đa mục tiêu". Dưới đây là các tham số "linh hồn" được cấu hình trong `dss_engine.py`:

1.  **Mô hình Độ co giãn Phí (Fee Elasticity - tham số `-0.5`)**:
    *   **Lý thuyết**: Trong kinh tế học, độ co giãn đo lường mức độ phản ứng của nhu cầu khi giá thay đổi. 
    *   **Logic tính toán**: $Volume_{new} = Volume_{old} \times (1 + \% \Delta Fee \times -0.5)$.
    *   **Ý nghĩa thực tế**: Giúp CEO hiểu rằng không thể tăng phí vô tội vạ. Nếu tăng phí quá cao, khách hàng sẽ bỏ sang đối thủ (VNPay, MoMo), dẫn đến tổng doanh thu thực tế có thể giảm thay vì tăng.

2.  **Nguyên lý Pareto & Phân phối đuôi nặng (Heavy-tail Distribution)**:
    *   **Logic**: Gian lận thường tập trung vào các giao dịch giá trị lớn để tối ưu "công sức" của tội phạm.
    *   **Tham số**: Hệ thống giả định tỷ lệ **1.5x**. Nghĩa là cứ giảm 10% hạn mức giao dịch ở nhóm cao nhất, ta chặn được 15% tổng giá trị thiệt hại.
    *   **Ý nghĩa**: Tìm ra "điểm ngọt" (Sweet spot) - nơi ta chặn được nhiều gian lận nhất mà chỉ gây ảnh hưởng đến ít hơn 5% khách hàng VIP thực sự.

3.  **Mô hình Tương quan Lãi suất - Nợ xấu (tham số `0.1`)**:
    *   **Logic**: Khi lãi suất vay tăng (Interest Rate), áp lực trả nợ của khách hàng tăng, dẫn đến xác suất vỡ nợ tăng.
    *   **Công thức**: $NPL_{new} = NPL_{old} + (Rate_{new} - Rate_{old}) \times 0.1$.
    *   **Ý nghĩa**: Cảnh báo CEO về hiện tượng **Adverse Selection (Lựa chọn ngược)**: Lãi suất quá cao sẽ chỉ thu hút những người vay liều lĩnh, làm "nát" bảng cân đối kế toán.

4.  **Thuật toán Association Rules (Apriori/FP-Growth)**:
    *   **Tham số**: Tập trung vào **Lift (Độ nâng)**. Nếu $Lift(A \rightarrow B) > 1.5$, có sự cộng hưởng hành vi mạnh.
    *   **Logic**: Hệ thống quét lịch sử giao dịch để tìm các cặp dịch vụ "mua kèm". 
    *   **Ý nghĩa**: Chuyển dịch từ việc "bán cái mình có" sang "bán cái khách hàng cần tiếp theo".

5.  **Isolation Forest cho Merchant Anomaly**:
    *   **Tham số**: `Contamination=0.05` (Giả định 5% đối tác là bất thường).
    *   **Logic**: Tính toán độ cô lập của một Merchant dựa trên: Tần suất giao dịch đêm, tỷ lệ hoàn tiền, và sự lệch chuẩn GMV so với các Merchant cùng ngành.
    *   **Ý nghĩa**: Tự động phát hiện các Merchant "rửa tiền" hoặc dùng tài khoản cá nhân để kinh doanh trốn thuế.

---

## 3. Phân tích Chi tiết Dashboard Streamlit (Business Intelligence View)

Dưới đây là mô tả toàn bộ các thành phần hiển thị, cách tính và ý nghĩa của chúng đối với người điều hành.

### 3.1 Tab: Tổng quan CEO (Executive Summary)
*   **Chỉ số Sức khỏe (Health Score)**:
    *   *Cách tính*: Tổng trọng số (Weighted Score) của 3 trụ cột: An toàn (40%), Doanh thu (40%), Dịch vụ (20%).
    *   *Logic*: Nếu tỷ lệ gian lận tăng vọt, điểm Safety sẽ kéo tụt toàn bộ Health Score ngay cả khi doanh thu đang tăng trưởng xanh.
    *   *Ý nghĩa*: Giúp CEO không bị "mờ mắt" bởi doanh thu mà bỏ qua các rủi ro hệ thống.
*   **Hệ thống Cảnh báo Thông minh (Smart Alerts)**:
    *   *Logic*: Sử dụng câu lệnh điều kiện lồng nhau (Nested Logic) để so sánh thực tế với Ngưỡng an toàn (Thresholds) và Mục tiêu chiến lược (Targets).
    *   *Ví dụ*: Nếu NPL > 5% $\rightarrow$ Hiện thông báo Đỏ. Nếu MoM Revenue < -10% $\rightarrow$ Hiện thông báo Vàng.
*   **Bản đồ Vùng miền & Khu vực**:
    *   *Logic*: Aggregation theo tọa độ và nhóm địa lý (Bắc, Trung, Nam).
    *   *Ý nghĩa*: Nhận diện các "điểm nóng" (hotspots). Ví dụ: Gian lận thường tăng cao ở các vùng biên giới hoặc các thành phố lớn nơi tội phạm công nghệ tập trung.

### 3.2 Tab: Thu hút & Kích hoạt (Marketing Funnel)
*   **Biểu đồ Phễu (Funnel Chart)**:
    *   *Các bước*: Impressions (Ấn tượng) $\rightarrow$ Clicks (Truy cập) $\rightarrow$ KYC (Định danh) $\rightarrow$ Activation (Giao dịch đầu tiên).
    *   *Chỉ số*: Tỷ lệ chuyển đổi qua từng bước (Conversion Rate).
    *   *Ý nghĩa*: Tìm ra "nút thắt cổ chai". Nếu Click nhiều nhưng KYC ít, nghĩa là quy trình đăng ký đang quá phức tạp, làm khách hàng nản lòng.
*   **Mô phỏng Tái phân bổ ngân sách (Budget Reallocation)**:
    *   *Logic*: Lấy tổng ngân sách $\times$ Trọng số kênh $\div$ CAC lịch sử của kênh đó.
    *   *Ý nghĩa*: CEO có thể "chơi thử" các kịch bản đầu tư. Ví dụ: Thay vì đổ tiền vào Facebook Ads (CAC cao, người dùng ảo nhiều), hãy chuyển sang In-App Game (CAC thấp hơn, tính gắn kết cao).

### 3.3 Tab: Tín dụng & Dòng tiền (Credit Portfolio)
*   **Thước đo NPL (Gauge Chart)**:
    *   *Cách tính*: Dư nợ nhóm 3-5 / Tổng dư nợ.
    *   *Ý nghĩa*: Đây là "huyết áp" của công ty tài chính. NPL tăng đồng nghĩa với việc lợi nhuận trong tương lai đang bị "ăn mòn".
*   **Mô phỏng Chính sách Tín dụng**:
    *   *Biến số*: CEO thay đổi Lãi suất và Hạn mức.
    *   *Kết quả*: Hệ thống tính toán "Biên lợi nhuận ròng" (Net Margin) sau khi trừ đi Dự phòng rủi ro (Provisioning).
    *   *Ý nghĩa*: Giúp CEO tìm thấy "Điểm hòa vốn rủi ro" - nơi lãi suất đủ bù đắp cho tỷ lệ nợ xấu dự kiến.

### 3.4 Tab: Hệ sinh thái Dịch vụ (Product Ecosystem)
*   **Ma trận BCG (Growth-Share Matrix)**:
    *   *Cách tính*: Trục X là Thị phần/Doanh thu, Trục Y là Tốc độ tăng trưởng.
    *   *Phân loại*: Bò sữa (Revenue cao, Growth thấp), Ngôi sao (Cả hai cao), Chó mực (Cả hai thấp).
    *   *Ý nghĩa*: Quyết định chiến lược: Dùng tiền từ "Bò sữa" để nuôi các "Ngôi sao" mới nhú.
*   **Mô phỏng Combo Giảm giá**:
    *   *Logic*: Sử dụng chỉ số Lift từ thuật toán Association Rules để dự báo mức độ "mua kèm".
    *   *Ý nghĩa*: Thiết kế các gói combo (Ví dụ: Thanh toán điện + Nạp điện thoại) để tăng tính "stickiness" (khả năng bám rễ) của khách hàng.

### 3.5 Tab: Đối tác & Giao dịch (Merchant & Ops)
*   **Bảng xếp hạng Rủi ro Merchant**:
    *   *Chỉ số*: Anomaly Score (0-1).
    *   *Logic*: Merchant có điểm cao sẽ bị đẩy lên đầu danh sách Review Queue.
    *   *Ý nghĩa*: Ưu tiên nguồn lực kiểm tra. Thay vì kiểm tra ngẫu nhiên, ta tập trung vào 5% những đối tác có hành vi mờ ám nhất.
*   **Dự báo Gian lận theo Kịch bản (Scenario Analysis)**:
    *   *Logic*: Sử dụng kịch bản cơ sở (Base case) $\pm$ các yếu tố ngoại biên (Sự phát triển của công nghệ bảo mật vs Sự tinh vi của hacker).
    *   *Ý nghĩa*: Giúp CEO chuẩn bị tâm thế và nguồn lực dự phòng cho các quý tiếp theo.


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