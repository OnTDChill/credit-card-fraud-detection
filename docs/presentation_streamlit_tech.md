# Hướng dẫn Thuyết trình: Cấu trúc Kỹ thuật và Phân tích Mô hình trên Dashboard

Tài liệu này trình bày chi tiết cơ chế vận hành kỹ thuật của Dashboard: Từ luồng dữ liệu, cách tính toán chỉ số Machine Learning, đến việc trực quan hóa các đặc trưng mô hình nhằm phục vụ công tác giám sát và tinh chỉnh hệ thống.

---

## 1. Luồng vận hành Kỹ thuật (Technical Workflow)

Hệ thống được xây dựng theo mô hình phân lớp để đảm bảo tính tách biệt giữa dữ liệu và hiển thị:
1. **Data Layer**: Dữ liệu giao dịch và chỉ số kinh doanh được lưu trữ trong SQLite, được cập nhật thông qua các ETL pipelines (`core/etl/`).
2. **Model Artifacts**: Các mô hình đã huấn luyện và tệp dự đoán (`predictions_test.csv`) được lưu trữ tại `FRAUD_ARTIFACTS_DIR`. 
3. **Logic Layer**: Sử dụng `scikit-learn` để tính toán metrics động dựa trên xác suất dự đoán (`fraud_probability`) và nhãn thực tế (`y_true`).
4. **Presentation Layer**: Streamlit render các biểu đồ tương tác thông qua Plotly, cho phép phân tích sâu (drill-down) vào các lỗi của mô hình.

---

## 2. Phân tích Hiệu năng Mô hình (ML Evaluation)

Thay vì chỉ cung cấp một con số chính xác (Accuracy) dễ gây hiểu lầm trên dữ liệu mất cân bằng, hệ thống tập trung vào bộ chỉ số đo lường khả năng phân tách lớp:

- **ROC-AUC**: Đo lường khả năng phân loại tổng quát của mô hình trên mọi ngưỡng quyết định.
- **Precision & Recall**: 
  - **Precision**: Tỷ lệ chính xác khi hệ thống báo động gian lận (giảm False Positives).
  - **Recall**: Khả năng "quét" sạch gian lận trong tập dữ liệu (giảm False Negatives).
- **F1-Score**: Điểm cân bằng giữa Precision và Recall, là chỉ số chính để đánh giá chất lượng mô hình.
- **Confusion Matrix**: Trực quan hóa số lượng giao dịch Bị chặn đúng (TP), Cho qua đúng (TN), Chặn nhầm (FP) và Bỏ lọt (FN).

---

## 3. Cơ chế Ngưỡng Quyết định (Thresholding)

Hệ thống không trả về kết quả nhị phân cứng nhắc mà trả về **Fraud Probability** (0.0 đến 1.0). 
- **Ngưỡng (Threshold)**: Được lấy từ báo cáo huấn luyện (`training_report.json`) hoặc cấu hình hệ thống.
- **Logic Phân loại**: 
  - `fraud_probability >= threshold` $\rightarrow$ Gán nhãn **Gian lận**.
  - `fraud_probability < threshold` $\rightarrow$ Gán nhãn **Hợp lệ**.
- **Ý nghĩa**: Việc điều chỉnh ngưỡng cho phép doanh nghiệp linh hoạt thay đổi "khẩu vị rủi ro" mà không cần huấn luyện lại mô hình.

---

## 4. Giải thích Mô hình & Phân tích Lỗi (Interpretability & Error Analysis)

Để mô hình không còn là "hộp đen", hệ thống cung cấp hai công cụ phân tích sâu:

### A. Mức độ quan trọng đặc trưng (Feature Importance)
Hệ thống trích xuất trực tiếp từ mô hình (ví dụ: `feature_importances_` của Random Forest hoặc `coef_` của Logistic Regression) để xác định 15 yếu tố ảnh hưởng nhất đến quyết định chặn giao dịch. Điều này giúp chuyên viên dữ liệu hiểu rõ "tại sao mô hình cho rằng giao dịch này là gian lận".

### B. Phân tích lỗi chi tiết (Error Analysis)
Hệ thống tách riêng hai nhóm lỗi để tối ưu:
- **False Positives (Báo động giả)**: Phân tích các giao dịch sạch bị chặn nhầm để nới lỏng luật.
- **False Negatives (Bỏ sót)**: Truy vết các giao dịch gian lận lọt lưới để bổ sung đặc trưng (features) mới cho mô hình.

---

## 5. Hệ thống Benchmarking và Nhật ký (Monitoring)

- **Model Benchmarking**: Dashboard hiển thị bảng so sánh hiệu năng giữa các thuật toán khác nhau (Random Forest, XGBoost, Logistic Regression, v.v.) dựa trên tập Test, giúp lựa chọn mô hình tối ưu nhất cho triển khai.
- **Audit Logs**: Ghi lại toàn bộ lịch sử thay đổi cấu hình, thay đổi ngưỡng và các sự kiện hệ thống, đảm bảo tính minh bạch và khả năng truy vết trong vận hành.

---

## Kết luận: Giá trị Kỹ thuật

Hệ thống không chỉ là một công cụ hiển thị, mà là một **môi trường phân tích kỹ thuật** cho phép:
- **Đo lường chính xác**: Sử dụng các metrics chuyên biệt cho bài toán imbalanced data.
- **Tinh chỉnh linh hoạt**: Thay đổi ngưỡng quyết định để cân bằng giữa rủi ro và tăng trưởng.
- **Cải tiến liên tục**: Thông qua phân tích lỗi và so sánh benchmark để nâng cấp mô hình theo thời gian.