"""
Realtime Monitor - Transaction scoring and monitoring
"""
from typing import Any, Dict, List
from datetime import datetime, timezone
import os
import random

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from streamlit_app.shared_ui import configure_dashboard_page, render_page_header

configure_dashboard_page("Realtime Monitor")


def _safe_get_json(url: str) -> Dict[str, Any]:
    try:
        response = requests.get(url, timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "BACKEND_OFFLINE", "message": "Backend API is not running. Use Local Model scoring instead."}
    except Exception as error:
        return {"error": str(error)}


def _safe_post_json(url: str, payload: Dict[str, Any], api_key: str, timeout_seconds: int = 8) -> Dict[str, Any]:
    try:
        response = requests.post(
            url,
            json=payload,
            headers={
                "X-Fraud-Api-Key": api_key,
                "Content-Type": "application/json",
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "BACKEND_OFFLINE", "message": "Backend API is not running. Use Local Model scoring instead."}
    except Exception as error:
        return {"error": str(error)}


def _safe_get_json_with_key(url: str, api_key: str) -> Dict[str, Any]:
    try:
        response = requests.get(url, headers={"X-Fraud-Api-Key": api_key}, timeout=8)
        response.raise_for_status()
        return response.json()
    except Exception as error:
        return {"error": str(error)}


def _mask_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) <= 4:
        return "*" * len(text)
    return f"{text[:2]}***{text[-2:]}"


def _safe_runtime_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    runtime = payload.get("runtime") or {}
    if not isinstance(runtime, dict):
        return {}
    return runtime


def _default_artifacts_dir() -> str:
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(os.path.dirname(current_dir))
    return os.path.join(project_root, "artifacts", "fraud")


def _default_model_path(artifacts_dir: str) -> str:
    return os.path.join(artifacts_dir, "fraud_model.joblib")


@st.cache_resource(show_spinner=False)
def _load_local_pipeline(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def _build_model_input(upload_df: pd.DataFrame, pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    expected_columns = list(preprocessor.feature_names_in_)

    model_input = pd.DataFrame(np.nan, index=upload_df.index, columns=expected_columns)

    for column_name in upload_df.columns:
        if column_name in model_input.columns:
            model_input[column_name] = upload_df[column_name]

    if "TransactionAmt" in model_input.columns and "amount" in upload_df.columns:
        model_input["TransactionAmt"] = model_input["TransactionAmt"].fillna(upload_df["amount"])
    if "TransactionDT" in model_input.columns and "transaction_dt" in upload_df.columns:
        model_input["TransactionDT"] = model_input["TransactionDT"].fillna(upload_df["transaction_dt"])
    if "DeviceInfo" in model_input.columns and "device_info" in upload_df.columns:
        model_input["DeviceInfo"] = model_input["DeviceInfo"].fillna(upload_df["device_info"])

    feature_columns = [name for name in upload_df.columns if name.startswith("feature_")]
    for feature_column in feature_columns:
        target_name = feature_column.replace("feature_", "", 1)
        if target_name in model_input.columns:
            model_input[target_name] = model_input[target_name].fillna(upload_df[feature_column])

    if "TransactionDT" in model_input.columns:
        transaction_dt = pd.to_numeric(model_input["TransactionDT"], errors="coerce")
        if "transaction_hour" in model_input.columns:
            model_input["transaction_hour"] = ((transaction_dt // 3600) % 24).astype(float)
        if "transaction_day" in model_input.columns:
            model_input["transaction_day"] = (transaction_dt // (3600 * 24)).astype(float)
        if "transaction_week" in model_input.columns:
            model_input["transaction_week"] = (transaction_dt // (3600 * 24 * 7)).astype(float)

    if "TransactionAmt" in model_input.columns and "transaction_amt_log1p" in model_input.columns:
        safe_amount = pd.to_numeric(model_input["TransactionAmt"], errors="coerce").fillna(0).clip(lower=0)
        model_input["transaction_amt_log1p"] = np.log1p(safe_amount)

    if "entity_key" in model_input.columns:
        key_columns = [name for name in ["card1", "card2", "addr1", "DeviceInfo"] if name in model_input.columns]
        if key_columns:
            model_input["entity_key"] = model_input[key_columns].fillna("na").astype(str).agg("|".join, axis=1)
        else:
            model_input["entity_key"] = "unknown"

    numeric_columns = preprocessor.transformers_[0][2]
    for numeric_column in numeric_columns:
        if numeric_column in model_input.columns:
            model_input[numeric_column] = pd.to_numeric(model_input[numeric_column], errors="coerce")

    return model_input[expected_columns]


def _compute_binary_summary(result_df: pd.DataFrame, threshold: float) -> Dict[str, Any]:
    summary = {
        "total": int(len(result_df)),
        "predicted_fraud": int((result_df["predicted_label"] == "fraud").sum()),
        "predicted_normal": int((result_df["predicted_label"] == "normal").sum()),
        "threshold": float(threshold),
    }

    if "truth_label" not in result_df.columns or result_df["truth_label"].isna().all():
        return summary

    valid_rows = result_df[result_df["truth_label"].notna()].copy()
    if valid_rows.empty:
        return summary

    truth = pd.to_numeric(valid_rows["truth_label"], errors="coerce").fillna(0).astype(int)
    pred = (valid_rows["predicted_label"] == "fraud").astype(int)

    tp = int(((truth == 1) & (pred == 1)).sum())
    tn = int(((truth == 0) & (pred == 0)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(valid_rows) if len(valid_rows) else 0.0

    summary.update({
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "accuracy": float(accuracy),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    })
    return summary


def _score_uploaded_csv(upload_df: pd.DataFrame, pipeline, threshold: float) -> Dict[str, Any]:
    model_input = _build_model_input(upload_df, pipeline)
    probabilities = pipeline.predict_proba(model_input)[:, 1]
    predicted_labels = np.where(probabilities >= threshold, "fraud", "normal")
    confidence = np.maximum(probabilities, 1 - probabilities)

    result_df = upload_df.copy()
    result_df["fraud_probability"] = probabilities
    result_df["prediction_confidence"] = confidence
    result_df["predicted_label"] = predicted_labels

    truth_column = None
    for candidate in ["is_fraud_truth", "isFraud", "truth_label", "label", "target"]:
        if candidate in result_df.columns:
            truth_column = candidate
            break
    if truth_column is not None:
        result_df["truth_label"] = pd.to_numeric(result_df[truth_column], errors="coerce").fillna(0).astype(int)

    if "external_transaction_id" in result_df.columns:
        result_df["external_transaction_id"] = result_df["external_transaction_id"].map(_mask_text)
    if "external_user_id" in result_df.columns:
        result_df["external_user_id"] = result_df["external_user_id"].map(_mask_text)

    summary = _compute_binary_summary(result_df, threshold=threshold)
    return {"result_df": result_df, "summary": summary}


def _prepare_backend_transactions(upload_df: pd.DataFrame) -> List[Dict[str, Any]]:
    transactions: List[Dict[str, Any]] = []

    truth_column = None
    for candidate in ["is_fraud_truth", "isFraud", "truth_label", "label", "target"]:
        if candidate in upload_df.columns:
            truth_column = candidate
            break

    feature_columns = [name for name in upload_df.columns if name.startswith("feature_")]

    for row_index, row in upload_df.reset_index(drop=True).iterrows():
        amount = row.get("amount") if pd.notna(row.get("amount")) else row.get("TransactionAmt")
        transaction_dt = row.get("transaction_dt") if pd.notna(row.get("transaction_dt")) else row.get("TransactionDT")

        feature_payload: Dict[str, Any] = {"TransactionAmt": float(amount) if pd.notna(amount) else 0.0}
        if pd.notna(transaction_dt):
            feature_payload["TransactionDT"] = float(transaction_dt)

        for column_name in ["card1", "card2", "addr1", "C1", "D1", "DeviceInfo"]:
            if column_name in upload_df.columns and pd.notna(row.get(column_name)):
                feature_payload[column_name] = row.get(column_name)

        for feature_column in feature_columns:
            if pd.notna(row.get(feature_column)):
                target_name = feature_column.replace("feature_", "", 1)
                feature_payload[target_name] = row.get(feature_column)

        payload = {
            "input_index": int(row_index),
            "external_transaction_id": str(row.get("external_transaction_id") or f"batch-{row_index}"),
            "external_user_id": str(row.get("external_user_id") or f"demo-user-{row_index:05d}"),
            "card_fingerprint": str(row.get("card_fingerprint") or f"demo-card-{row_index:05d}"),
            "amount": float(amount) if pd.notna(amount) else 0.0,
            "currency": str(row.get("currency") or "usd"),
            "transaction_dt": int(float(transaction_dt)) if pd.notna(transaction_dt) else None,
            "features": feature_payload,
        }

        if truth_column is not None and pd.notna(row.get(truth_column)):
            payload["is_fraud_truth"] = int(float(row.get(truth_column)))

        transactions.append(payload)

    return transactions


def _score_uploaded_csv_via_api(upload_df: pd.DataFrame, threshold: float, api_base_url: str, api_key: str, chunk_size: int = 2000) -> Dict[str, Any]:
    transactions = _prepare_backend_transactions(upload_df)
    batch_url = f"{api_base_url.rstrip('/')}/batch-score/"

    all_results: List[Dict[str, Any]] = []
    progress_bar = st.progress(0, text="Sending batch scoring to backend...")

    for start_index in range(0, len(transactions), chunk_size):
        end_index = min(start_index + chunk_size, len(transactions))
        chunk_transactions = transactions[start_index:end_index]
        response_payload = _safe_post_json(
            batch_url,
            {"threshold": threshold, "transactions": chunk_transactions},
            api_key=api_key,
            timeout_seconds=120,
        )

        if "error" in response_payload:
            raise RuntimeError(response_payload["error"])

        all_results.extend(response_payload.get("results", []))
        progress_bar.progress(end_index / max(len(transactions), 1), text=f"Scored {end_index}/{len(transactions)} transactions")

    progress_bar.empty()

    result_df = pd.DataFrame(all_results)
    if result_df.empty:
        raise RuntimeError("Backend returned no scoring results.")

    if "row_index" in result_df.columns:
        result_df = result_df.sort_values("row_index").reset_index(drop=True)

    if "external_transaction_id" in result_df.columns:
        result_df["external_transaction_id"] = result_df["external_transaction_id"].map(_mask_text)
    if "external_user_id" in result_df.columns:
        result_df["external_user_id"] = result_df["external_user_id"].map(_mask_text)

    summary = _compute_binary_summary(result_df, threshold=threshold)
    return {"result_df": result_df, "summary": summary}


def _df_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _prepare_transactions_df(stream_payload: Dict[str, Any]) -> pd.DataFrame:
    transactions = stream_payload.get("transactions", [])
    if not transactions:
        return pd.DataFrame()

    tx_df = pd.DataFrame(transactions).copy()
    if "created_at" in tx_df.columns:
        tx_df["created_at"] = pd.to_datetime(tx_df["created_at"], errors="coerce")
    if "external_transaction_id" in tx_df.columns:
        tx_df["external_transaction_id"] = tx_df["external_transaction_id"].map(_mask_text)
    if "external_user_id" in tx_df.columns:
        tx_df["external_user_id"] = tx_df["external_user_id"].map(_mask_text)
    return tx_df


def _prepare_alerts_df(stream_payload: Dict[str, Any]) -> pd.DataFrame:
    alerts = stream_payload.get("alerts", [])
    if not alerts:
        return pd.DataFrame()

    alert_df = pd.DataFrame(alerts).copy()
    if "created_at" in alert_df.columns:
        alert_df["created_at"] = pd.to_datetime(alert_df["created_at"], errors="coerce")
    return alert_df


def _render_runtime_kpis(runtime: Dict[str, Any]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total", int(runtime.get("total_transactions", 0) or 0))
    with c2:
        st.metric("24h", int(runtime.get("transactions_last_24h", 0) or 0))
    with c3:
        st.metric("Blocked", int(runtime.get("blocked_transactions", 0) or 0))
    with c4:
        st.metric("Review", int(runtime.get("review_transactions", 0) or 0))
    with c5:
        st.metric("Open Alerts", int(runtime.get("open_alerts", 0) or 0))


def _render_transaction_charts(tx_df: pd.DataFrame) -> None:
    if tx_df.empty:
        st.info("No transactions in stream yet.")
        return

    if "decision" in tx_df.columns:
        decision_counts = tx_df["decision"].value_counts().rename_axis("decision").to_frame("count")
        st.markdown("#### Decision Distribution")
        st.bar_chart(decision_counts)

    if "created_at" in tx_df.columns and "fraud_score" in tx_df.columns:
        score_df = tx_df[["created_at", "fraud_score"]].dropna().sort_values("created_at")
        if not score_df.empty:
            st.markdown("#### Fraud Score Trend")
            st.line_chart(score_df.set_index("created_at"))

    st.markdown("#### Recent Transactions (PII masked)")
    display_cols = [
        "created_at", "external_transaction_id", "external_user_id",
        "amount", "currency", "fraud_score", "decision", "is_fraud_prediction"
    ]
    available_cols = [col for col in display_cols if col in tx_df.columns]
    st.dataframe(tx_df[available_cols], width="stretch", hide_index=True)


def _render_alert_charts(alert_df: pd.DataFrame) -> None:
    if alert_df.empty:
        st.info("No alerts in stream yet.")
        return

    if "severity" in alert_df.columns:
        severity_counts = alert_df["severity"].value_counts().rename_axis("severity").to_frame("count")
        st.markdown("#### Alerts by Severity")
        st.bar_chart(severity_counts)

    if "status" in alert_df.columns:
        status_counts = alert_df["status"].value_counts().rename_axis("status").to_frame("count")
        st.markdown("#### Alerts by Status")
        st.bar_chart(status_counts)

    st.markdown("#### Alert List")
    display_cols = ["created_at", "severity", "status", "message"]
    available_cols = [col for col in display_cols if col in alert_df.columns]
    st.dataframe(alert_df[available_cols], width="stretch", hide_index=True)


def _initialize_state() -> None:
    if "fraud_metrics_payload" not in st.session_state:
        st.session_state.fraud_metrics_payload = {}
    if "fraud_stream_payload" not in st.session_state:
        st.session_state.fraud_stream_payload = {}
    if "batch_result_df" not in st.session_state:
        st.session_state.batch_result_df = pd.DataFrame()
    if "batch_summary" not in st.session_state:
        st.session_state.batch_summary = {}
    if "batch_filter" not in st.session_state:
        st.session_state.batch_filter = "all"


def _resolve_api_key() -> str:
    env_key = os.getenv("FRAUD_API_KEY")
    if env_key:
        return env_key

    try:
        return str(st.secrets["fraud_api_key"])
    except (StreamlitSecretNotFoundError, KeyError, AttributeError, TypeError):
        return "local-fraud-api-key"


def _render_batch_results() -> None:
    result_df = st.session_state.batch_result_df
    summary = st.session_state.batch_summary or {}

    if result_df.empty:
        return

    st.markdown("### Batch Model Scoring Results")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Transactions", int(summary.get("total", len(result_df))))
    with c2:
        if st.button(f"Fraud: {int(summary.get('predicted_fraud', 0))}", use_container_width=True):
            st.session_state.batch_filter = "fraud"
    with c3:
        if st.button(f"Normal: {int(summary.get('predicted_normal', 0))}", use_container_width=True):
            st.session_state.batch_filter = "normal"
    with c4:
        if st.button("View All", use_container_width=True):
            st.session_state.batch_filter = "all"

    if "precision" in summary:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Precision", f"{float(summary.get('precision', 0.0)):.2%}")
        with m2:
            st.metric("Recall", f"{float(summary.get('recall', 0.0)):.2%}")
        with m3:
            st.metric("F1", f"{float(summary.get('f1_score', 0.0)):.2%}")
        with m4:
            st.metric("Accuracy", f"{float(summary.get('accuracy', 0.0)):.2%}")

        cm = summary.get("confusion_matrix", {})
        cm_df = pd.DataFrame(
            {"value": [int(cm.get("tp", 0)), int(cm.get("fp", 0)), int(cm.get("fn", 0)), int(cm.get("tn", 0))]},
            index=["TP", "FP", "FN", "TN"],
        )
        st.bar_chart(cm_df)

    probability_view = result_df[["fraud_probability"]].copy()
    probability_view["bucket"] = pd.cut(
        probability_view["fraud_probability"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        include_lowest=True,
    )
    bucket_counts = probability_view.groupby("bucket", observed=False).size().to_frame("count").reset_index()
    bucket_counts["bucket"] = bucket_counts["bucket"].astype(str)
    bucket_counts = bucket_counts.set_index("bucket")

    st.markdown("#### Fraud Probability Distribution")
    st.bar_chart(bucket_counts)

    filter_mode = st.session_state.batch_filter
    filtered_df = result_df
    if filter_mode == "fraud":
        filtered_df = result_df[result_df["predicted_label"] == "fraud"]
    elif filter_mode == "normal":
        filtered_df = result_df[result_df["predicted_label"] == "normal"]

    st.markdown(f"#### Transaction List ({filter_mode})")
    display_columns = [
        "external_transaction_id", "external_user_id", "amount",
        "fraud_probability", "prediction_confidence", "predicted_label"
    ]
    if "truth_label" in filtered_df.columns:
        display_columns.append("truth_label")
    available_columns = [col for col in display_columns if col in filtered_df.columns]
    st.dataframe(filtered_df[available_columns], width="stretch", hide_index=True)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Export All Results (CSV)",
            data=_df_to_csv_bytes(result_df),
            file_name="batch_scoring_all.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Export Filtered Results (CSV)",
            data=_df_to_csv_bytes(filtered_df),
            file_name=f"batch_scoring_{filter_mode}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_realtime_monitor_page(api_base_url: str, model_threshold: float, artifacts_dir: str) -> None:
    render_page_header(
        "Realtime Monitor",
        "Live transaction scoring with secure API calls, local fallback scoring, and review policy controls.",
    )

    _initialize_state()

    left_col, right_col = st.columns(2)
    api_key = _resolve_api_key()

    with left_col:
        st.markdown("### Policy Configuration")
        threshold = st.slider("Fraud Threshold", min_value=0.10, max_value=0.99, value=0.70, step=0.01)
        block_transaction = st.toggle("Block Transaction", value=False)
        ban_user = st.toggle("Ban User ID", value=False)
        block_card = st.toggle("Block Card Fingerprint", value=False)
        manual_review = st.toggle("Manual Review", value=True)
        send_email_alert = st.toggle("Email Alert", value=False)

        policy_payload = {
            "fraud_threshold": threshold,
            "block_transaction": block_transaction,
            "ban_user": ban_user,
            "block_card_fingerprint": block_card,
            "manual_review": manual_review,
            "send_email_alert": send_email_alert,
        }

        policy_url = f"{api_base_url.rstrip('/')}/policy/"
        if st.button("Save Policy", type="primary"):
            response_payload = _safe_post_json(policy_url, policy_payload, api_key=api_key)
            if "error" in response_payload:
                st.error(response_payload["error"])
            else:
                st.success("Policy updated successfully.")

        st.markdown("### Send Test Transaction")
        scenario = st.selectbox("Scenario", options=["normal", "fraud"], index=0)
        amount = st.number_input("Amount", min_value=1.0, max_value=10000.0, value=120.0, step=10.0)
        st.caption("Do not enter real identification information. Dashboard uses demo data only.")

        ingest_url = f"{api_base_url.rstrip('/')}/ingest/"
        if st.button("Send Test Transaction"):
            adjusted_amount = amount
            if scenario == "fraud":
                adjusted_amount = max(amount, 1500.0)

            random_user = f"demo-user-{random.randint(1, 999):03d}"
            random_card = f"demo-card-{random.randint(1, 999):03d}"

            payload = {
                "external_transaction_id": f"streamlit-{int(datetime.now(timezone.utc).timestamp())}",
                "external_user_id": random_user,
                "card_fingerprint": random_card,
                "amount": adjusted_amount,
                "currency": "usd",
                "ip_address": "10.10.0.11",
                "device_info": random.choice(["streamlit-chrome", "streamlit-mobile"]),
                "event_time": datetime.now(timezone.utc).isoformat(),
                "transaction_dt": int(datetime.now(timezone.utc).timestamp()),
                "features": {
                    "card1": random.randint(1000, 20000),
                    "card2": random.randint(100, 600),
                    "addr1": random.randint(100, 500),
                    "C1": random.randint(0, 40),
                    "D1": random.randint(0, 20),
                },
            }

            response_payload = _safe_post_json(ingest_url, payload, api_key=api_key)
            if "error" in response_payload:
                st.error(response_payload["error"])
            else:
                score = float(response_payload.get("fraud_score", 0.0))
                model_label = "FRAUD" if score >= model_threshold else "NORMAL"

                st.success("Test transaction sent successfully.")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Fraud Score", round(score, 4))
                with m2:
                    st.metric("Model Label", model_label)
                with m3:
                    st.metric("Policy Decision", str(response_payload.get("decision", "unknown")).upper())
                with m4:
                    latency = response_payload.get("latency_ms")
                    latency_text = f"{float(latency):.2f} ms" if isinstance(latency, (int, float)) else "N/A"
                    st.metric("Latency", latency_text)

        st.divider()
        st.markdown("### Batch CSV Scoring")
        scoring_engine = st.radio(
            "Scoring Engine",
            options=["Backend API", "Local Model"],
            horizontal=True,
            index=1,
            help="Backend API requires Django server running on port 8000",
        )
        uploaded_file = st.file_uploader(
            "Upload Transaction CSV",
            type=["csv"],
            help="CSV may contain is_fraud_truth/isFraud column for truth-vs-predict comparison.",
        )

        if st.button("Score Entire CSV File"):
            if uploaded_file is None:
                st.warning("Please select a CSV file first.")
            else:
                try:
                    upload_df = pd.read_csv(uploaded_file)

                    if scoring_engine == "Backend API":
                        scoring_output = _score_uploaded_csv_via_api(
                            upload_df=upload_df,
                            threshold=model_threshold,
                            api_base_url=api_base_url,
                            api_key=api_key,
                        )
                    else:
                        model_path = _default_model_path(artifacts_dir)
                        pipeline = _load_local_pipeline(model_path)
                        scoring_output = _score_uploaded_csv(upload_df, pipeline, threshold=model_threshold)

                    st.session_state.batch_result_df = scoring_output["result_df"]
                    st.session_state.batch_summary = scoring_output["summary"]
                    st.session_state.batch_filter = "all"
                    st.success("Batch scoring completed for all transactions.")
                except Exception as error:
                    st.error(f"Cannot score CSV: {error}")

    with right_col:
        st.markdown("### Runtime Dashboard")
        metrics_url = f"{api_base_url.rstrip('/')}/metrics/"
        stream_url = f"{api_base_url.rstrip('/')}/stream/"

        if st.button("Refresh Data", type="primary"):
            metrics_payload = _safe_get_json_with_key(metrics_url, api_key=api_key)
            stream_payload = _safe_get_json_with_key(stream_url, api_key=api_key)

            if "error" in metrics_payload:
                st.error(metrics_payload["error"])
            else:
                st.session_state.fraud_metrics_payload = metrics_payload

            if "error" in stream_payload:
                st.error(stream_payload["error"])
            else:
                st.session_state.fraud_stream_payload = stream_payload

        metrics_payload = st.session_state.fraud_metrics_payload
        stream_payload = st.session_state.fraud_stream_payload

        if metrics_payload or stream_payload:
            runtime = _safe_runtime_metrics(metrics_payload)
            if runtime:
                _render_runtime_kpis(runtime)

            tx_df = _prepare_transactions_df(stream_payload)
            alert_df = _prepare_alerts_df(stream_payload)

            st.divider()
            _render_transaction_charts(tx_df)
            st.divider()
            _render_alert_charts(alert_df)

        _render_batch_results()

        with st.expander("Technical Details (hidden by default)"):
            st.write("API base URL:", api_base_url)
            st.write("Model threshold:", model_threshold)
            st.write("Artifacts dir:", artifacts_dir)
            st.write("Batch rows loaded:", len(st.session_state.batch_result_df))


def main() -> None:
    with st.sidebar:
        st.header("Configuration")
        advanced_mode = st.toggle("Show Advanced Configuration", value=False)

        model_threshold = st.slider(
            "Model Classification Threshold (fraud_probability)",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.01,
        )

        if advanced_mode:
            api_base_url = st.text_input(
                "Fraud API Base URL",
                value="http://127.0.0.1:8000/api/fraud",
                help="Endpoint base for ingest/stream/metrics/policy.",
            )
            artifacts_dir = st.text_input(
                "Artifacts Directory",
                value=_default_artifacts_dir(),
                help="Directory containing fraud_model.joblib.",
            )
        else:
            api_base_url = "http://127.0.0.1:8000/api/fraud"
            artifacts_dir = _default_artifacts_dir()

    render_realtime_monitor_page(
        api_base_url=api_base_url,
        model_threshold=model_threshold,
        artifacts_dir=artifacts_dir,
    )


if __name__ == "__main__":
    main()
