"""Phân tích kỹ thuật dành cho BI Engineer và Data Analyst."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.config import (
    BENCHMARK_PATH,
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
    FRAUD_ARTIFACTS_DIR,
    REVIEW_DB_PATH,
    TRAINING_REPORT_PATH,
    resolve_model_path,
)
from streamlit_app.shared_ui import configure_dashboard_page, render_page_header

configure_dashboard_page("Phân tích Kỹ thuật")


@st.cache_data(show_spinner=False)
def _load_training_report() -> Dict[str, Any]:
    if not TRAINING_REPORT_PATH.exists():
        return {}
    try:
        return json.loads(TRAINING_REPORT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_thresholds() -> Tuple[float, float]:
    if not REVIEW_DB_PATH.exists():
        return DEFAULT_LOW_THRESHOLD, DEFAULT_HIGH_THRESHOLD

    with sqlite3.connect(REVIEW_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT config_key, config_value FROM system_config WHERE config_key IN ('low_threshold','high_threshold')"
        ).fetchall()

    values = {row[0]: row[1] for row in rows}
    low_t = float(values.get("low_threshold", DEFAULT_LOW_THRESHOLD))
    high_t = float(values.get("high_threshold", DEFAULT_HIGH_THRESHOLD))
    return low_t, high_t


@st.cache_data(show_spinner=False)
def _load_prediction_frames() -> Dict[str, pd.DataFrame]:
    report = _load_training_report()
    file_map = report.get("prediction_files") or {
        "train": "predictions_train.csv",
        "validation": "predictions_validation.csv",
        "test": "predictions_test.csv",
    }

    frames: Dict[str, pd.DataFrame] = {}
    for split_name in ["train", "validation", "test"]:
        file_name = file_map.get(split_name, f"predictions_{split_name}.csv")
        prediction_path = FRAUD_ARTIFACTS_DIR / file_name
        if not prediction_path.exists():
            continue
        frame = pd.read_csv(prediction_path)
        if {"y_true", "fraud_probability"}.issubset(frame.columns):
            frames[split_name] = frame

    return frames


@st.cache_data(show_spinner=False)
def _load_benchmark_results() -> pd.DataFrame:
    if not BENCHMARK_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(BENCHMARK_PATH)


@st.cache_resource(show_spinner=False)
def _load_feature_importance() -> pd.DataFrame:
    model_path = resolve_model_path()
    if not model_path.exists():
        return pd.DataFrame()

    try:
        import joblib
    except ImportError:
        return pd.DataFrame()

    try:
        model = joblib.load(model_path)
    except Exception:
        return pd.DataFrame()

    estimator = model
    feature_names: list[str] | None = None

    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.values())
        if steps:
            estimator = steps[-1]
        preprocessor = model.named_steps.get("preprocessor")
        if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
            feature_names = list(preprocessor.feature_names_in_)

    if feature_names is None and hasattr(estimator, "feature_names_in_"):
        feature_names = list(estimator.feature_names_in_)

    if hasattr(estimator, "feature_importances_"):
        importances = np.array(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        importances = np.abs(np.array(estimator.coef_).reshape(-1))
    else:
        return pd.DataFrame()

    if feature_names is None:
        feature_names = [f"feature_{idx}" for idx in range(len(importances))]

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(15)
    return df


@st.cache_data(show_spinner=False)
def _load_audit_logs(limit: int = 200) -> pd.DataFrame:
    if not REVIEW_DB_PATH.exists():
        return pd.DataFrame()

    with sqlite3.connect(REVIEW_DB_PATH) as conn:
        df = pd.read_sql(
            """
            SELECT event_time, event_type, transaction_id, user_id, old_value, new_value, reason
            FROM audit_log
            ORDER BY event_time DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )

    return df


def _compute_metrics(frame: pd.DataFrame, threshold: float) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    if frame.empty:
        return {}, {}

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_curve,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        roc_curve,
    )

    y_true = frame["y_true"].astype(int).to_numpy()
    y_score = frame["fraud_probability"].astype(float).to_numpy()
    y_pred = (y_score >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
    except ValueError:
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])

    try:
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_score)
    except ValueError:
        precision_curve, recall_curve, thresholds = (
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([]),
        )

    fp = float(cm[0, 1]) if cm.size else 0.0
    fn = float(cm[1, 0]) if cm.size else 0.0
    tn = float(cm[0, 0]) if cm.size else 0.0
    tp = float(cm[1, 1]) if cm.size else 0.0

    fpr_rate = fp / (fp + tn) if (fp + tn) else 0.0
    fnr_rate = fn / (fn + tp) if (fn + tp) else 0.0

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "TN": int(tn),
        "FPR": fpr_rate,
        "FNR": fnr_rate,
    }

    curves = {
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision_curve,
        "recall": recall_curve,
        "thresholds": thresholds,
        "confusion": cm,
    }

    return metrics, curves


def _plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_score: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC-AUC = {auc_score:.3f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
    fig.update_layout(
        title="Đường ROC-AUC",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=320,
        template="plotly_dark",
    )
    return fig


def _plot_pr_curve(precision: np.ndarray, recall: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR Curve"))
    fig.update_layout(
        title="Đường Precision-Recall",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=320,
        template="plotly_dark",
    )
    return fig


def _plot_confusion_matrix(cm: np.ndarray) -> go.Figure:
    if cm.size == 0:
        cm = np.array([[0, 0], [0, 0]])

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Dự đoán 0", "Dự đoán 1"],
            y=["Thực tế 0", "Thực tế 1"],
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(title="Ma trận nhầm lẫn", height=320, template="plotly_dark")
    return fig


def _plot_feature_importance(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["importance"],
            y=df["feature"],
            orientation="h",
            marker_color="#22c55e",
        )
    )
    fig.update_layout(title="Mức độ quan trọng đặc trưng", height=320, template="plotly_dark")
    return fig


def _render_error_analysis(frame: pd.DataFrame, threshold: float) -> None:
    if frame.empty:
        st.info("Chưa có dữ liệu lỗi để phân tích.")
        return

    probs = frame["fraud_probability"].astype(float)
    truth = frame["y_true"].astype(int)
    pred = (probs >= threshold).astype(int)

    fp_mask = (truth == 0) & (pred == 1)
    fn_mask = (truth == 1) & (pred == 0)

    counts = pd.DataFrame({
        "Loại lỗi": ["Báo động giả (FP)", "Bỏ sót gian lận (FN)"],
        "Số lượng": [int(fp_mask.sum()), int(fn_mask.sum())],
    })

    st.bar_chart(counts.set_index("Loại lỗi"))

    sample_cols = [col for col in ["fraud_probability", "y_true"] if col in frame.columns]
    if sample_cols:
        st.dataframe(
            frame.loc[fp_mask | fn_mask, sample_cols].head(30),
            width='stretch',
        )


def main() -> None:
    render_page_header(
        "Phân tích kỹ thuật",
        "Dành cho chuyên viên dữ liệu: theo dõi đầy đủ chỉ số, mô hình và nhật ký hệ thống.",
        kicker="Trung tâm phân tích dữ liệu",
    )

    # Add reporting period filter to technical tab for consistency
    f_year = st.session_state.get("filter_year")
    f_month = st.session_state.get("filter_month")
    if f_year:
        st.caption(f"Đang xem dữ liệu cho năm: {f_year}" + (f" - Tháng {f_month}" if f_month else ""))

    frames = _load_prediction_frames()
    test_frame = frames.get("test", pd.DataFrame())

    if test_frame.empty:
        st.warning("Không tìm thấy dữ liệu dự đoán cho tập Test. Vui lòng chạy lại pipeline huấn luyện.")
        st.stop()

    report = _load_training_report()
    threshold = (
        (report.get("test") or {}).get("threshold")
        or (report.get("validation") or {}).get("threshold")
        or (report.get("train") or {}).get("threshold")
        or 0.5
    )
    metrics, curves = _compute_metrics(test_frame, float(threshold))

    st.subheader("Chỉ số mô hình")
    if not metrics:
        st.info("Chưa có dữ liệu dự đoán để tính toán chỉ số.")
    else:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}", help="(TP + TN) / tổng số")
        col2.metric("Precision", f"{metrics['Precision']:.4f}", help="TP / (TP + FP)")
        col3.metric("Recall", f"{metrics['Recall']:.4f}", help="TP / (TP + FN)")
        col4.metric("F1-Score", f"{metrics['F1']:.4f}", help="Trung bình điều hòa Precision và Recall")
        col5.metric("ROC-AUC", f"{metrics['ROC_AUC']:.4f}", help="Diện tích dưới đường ROC")

    tab_a, tab_b, tab_c, tab_d = st.tabs(
        ["Đường ROC-AUC", "Ma trận nhầm lẫn", "Đường Precision-Recall", "Mức độ quan trọng đặc trưng"]
    )

    with tab_a:
        if metrics:
            st.plotly_chart(
                _plot_roc_curve(curves["fpr"], curves["tpr"], metrics["ROC_AUC"]),
                width='stretch',
            )
    with tab_b:
        if metrics:
            st.plotly_chart(_plot_confusion_matrix(curves["confusion"]), width='stretch')
            with st.expander("Giải thích Confusion Matrix"):
                st.markdown(
                    "- **TP (True Positive):** Hệ thống nhận đúng là gian lận, đã chặn.\n"
                    "- **TN (True Negative):** Hệ thống nhận đúng là bình thường, đã cho qua.\n"
                    "- **FP (False Positive):** Hệ thống nhận sai là gian lận, chặn nhầm khách thật.\n"
                    "- **FN (False Negative):** Hệ thống nhận sai là bình thường, gian lận lọt lưới."
                )
    with tab_c:
        if metrics:
            st.plotly_chart(_plot_pr_curve(curves["precision"], curves["recall"]), width='stretch')
    with tab_d:
        feature_df = _load_feature_importance()
        if feature_df.empty:
            st.info("Chưa có dữ liệu mức độ quan trọng đặc trưng để hiển thị.")
        else:
            st.plotly_chart(_plot_feature_importance(feature_df), width='stretch')

    st.markdown("---")
    st.subheader("Bảng so sánh toàn bộ mô hình đã thử nghiệm")
    benchmark_df = _load_benchmark_results()
    if benchmark_df.empty:
        st.info("Chưa có dữ liệu benchmark để hiển thị.")
    else:
        st.dataframe(benchmark_df.sort_values("f1", ascending=False), width='stretch')

    with st.expander("Phân tích lỗi (False Negative & False Positive)"):
        _render_error_analysis(test_frame, float(threshold))

    with st.expander("Nhật ký hệ thống"):
        logs_df = _load_audit_logs()
        if logs_df.empty:
            st.info("Chưa có nhật ký hệ thống để hiển thị.")
        else:
            st.dataframe(logs_df, width='stretch')


if __name__ == "__main__":
    main()
