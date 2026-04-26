"""
Truth vs Predict Analysis - Gap analysis and improvement recommendations
"""
import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

from streamlit_app.shared_ui import configure_dashboard_page, render_page_header

configure_dashboard_page("Truth vs Predict Analysis")


def _load_benchmark_results():
    benchmark_path = os.path.join(
        os.path.dirname(__file__), "../../artifacts/benchmark_results.csv"
    )
    if os.path.exists(benchmark_path):
        return pd.read_csv(benchmark_path)
    return pd.DataFrame()


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _analyze_truth_vs_predict(truth, pred):
    """Calculate confusion matrix and metrics"""
    tp = int(((truth == 1) & (pred == 1)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    tn = int(((truth == 0) & (pred == 0)).sum())

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    fpr = _safe_divide(fp, fp + tn)
    fnr = _safe_divide(fn, fn + tp)
    tpr = recall
    tnr = _safe_divide(tn, tn + fp)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "fpr": fpr, "fnr": fnr, "tpr": tpr, "tnr": tnr,
    }


def _render_gap_analysis(metrics: Dict[str, Any], benchmark_df: pd.DataFrame):
    """Analyze gap between current and target metrics"""
    st.markdown("## Gap Analysis")

    targets = {
        "recall": 0.92,
        "precision": 0.75,
        "f1": 0.82,
        "fpr": 0.10,
        "latency_p95_ms": 120.0,
    }

    col1, col2, col3 = st.columns(3)

    with col1:
        recall_gap = metrics["recall"] - targets["recall"]
        recall_color = "normal" if recall_gap >= 0 else "inverse"
        st.metric(
            "Recall (Target: 92%)",
            f"{metrics['recall']:.2%}",
            f"{recall_gap:+.2%}",
            delta_color=recall_color,
        )

    with col2:
        prec_gap = metrics["precision"] - targets["precision"]
        prec_color = "normal" if prec_gap >= 0 else "inverse"
        st.metric(
            "Precision (Target: 75%)",
            f"{metrics['precision']:.2%}",
            f"{prec_gap:+.2%}",
            delta_color=prec_color,
        )

    with col3:
        f1_gap = metrics["f1"] - targets["f1"]
        f1_color = "normal" if f1_gap >= 0 else "inverse"
        st.metric(
            "F1 (Target: 0.82)",
            f"{metrics['f1']:.4f}",
            f"{f1_gap:+.4f}",
            delta_color=f1_color,
        )

    st.markdown("### Comparison with Target")

    gap_data = []
    for metric_name, target_val in targets.items():
        if metric_name == "latency_p95_ms":
            current = benchmark_df["latency_p95_ms"].min() if not benchmark_df.empty else 0
            unit = "ms"
        elif metric_name in ["fpr"]:
            current = metrics.get(metric_name, 0)
            unit = "%"
        else:
            current = metrics.get(metric_name, 0)
            unit = "%" if metric_name != "f1" else ""

        gap = current - target_val
        status = "PASS" if gap >= 0 else "FAIL"

        if metric_name == "fpr":
            gap = target_val - current
            status = "PASS" if gap >= 0 else "FAIL"

        gap_data.append({
            "Metric": metric_name.upper(),
            "Current": f"{current:.2%}" if unit == "%" else f"{current:.2f}{unit}",
            "Target": f"{target_val:.2%}" if unit == "%" else f"{target_val:.2f}{unit}",
            "Gap": f"{gap:+.2%}" if unit == "%" else f"{gap:+.2f}{unit}",
            "Status": status,
        })

    gap_df = pd.DataFrame(gap_data)
    st.dataframe(gap_df, width='stretch', hide_index=True)


def _render_confusion_visualization(metrics: Dict[str, Any]):
    """Display confusion matrix with labels"""
    st.markdown("## Confusion Matrix")

    tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]

    cm_data = {
        "": ["Actual Fraud (Truth=1)", "Actual Normal (Truth=0)"],
        "Predict Fraud (1)": [tp, fp],
        "Predict Normal (0)": [fn, tn],
    }
    cm_df = pd.DataFrame(cm_data).set_index("")
    st.dataframe(cm_df, width='stretch')

    confusion_df = pd.DataFrame({
        "count": [tp, fp, fn, tn],
        "type": ["TP (Correct)", "FP (False Alarm)", "FN (Missed)", "TN (Correct)"],
    }).set_index("type")

    st.bar_chart(confusion_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("TP (Fraud Caught)", tp, f"{metrics['recall']:.1%} Recall")
    with c2:
        st.metric("FP (Normal Blocked)", fp, f"{metrics['fpr']:.1%} FPR")
    with c3:
        st.metric("FN (Fraud Missed)", fn, f"{metrics['fnr']:.1%} FNR")
    with c4:
        st.metric("TN (Normal Passed)", tn, f"{metrics['tnr']:.1%} TNR")


def _render_root_cause_analysis(metrics: Dict[str, Any], benchmark_df: pd.DataFrame):
    """Root cause analysis of prediction gaps"""
    st.markdown("## Root Cause Analysis")

    issues = []

    if metrics["precision"] < 0.30:
        issues.append({
            "issue": "Precision too low",
            "value": f"{metrics['precision']:.2%}",
            "cause": "Threshold too low or model cannot learn clear separation patterns",
            "impact": f"{metrics['fp']} normal transactions blocked incorrectly, affecting customer UX",
            "severity": "CRITICAL",
        })

    if metrics["recall"] < 0.60:
        issues.append({
            "issue": "Recall below requirement",
            "value": f"{metrics['recall']:.2%}",
            "cause": "Model not sensitive enough to detect fraud, possibly due to undersampling or insufficient features",
            "impact": f"{metrics['fn']} fraud transactions missed, financial loss",
            "severity": "CRITICAL",
        })

    if metrics["fpr"] > 0.30:
        issues.append({
            "issue": "False Positive Rate high",
            "value": f"{metrics['fpr']:.2%}",
            "cause": "Model too sensitive or severely imbalanced dataset",
            "impact": "Customers blocked incorrectly, reduced trust score",
            "severity": "WARNING",
        })

    if not benchmark_df.empty:
        best_f1 = benchmark_df["f1"].max()
        if best_f1 < 0.40:
            issues.append({
                "issue": "Model F1 Score low",
                "value": f"{best_f1:.4f}",
                "cause": "IEEE-CIS dataset has many missing values, feature engineering insufficient",
                "impact": "Overall performance poor, need retrain with better features",
                "severity": "WARNING",
            })

    if not issues:
        st.success("No critical issues detected!")
    else:
        for issue in issues:
            with st.container(border=True):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"### {issue['severity']}")
                    st.markdown(f"**{issue['issue']}**")
                    st.metric("Value", issue["value"])
                with col2:
                    st.markdown(f"**Cause:** {issue['cause']}")
                    st.markdown(f"**Impact:** {issue['impact']}")

    st.markdown("### Ratio Analysis")

    total = metrics["tp"] + metrics["fp"] + metrics["fn"] + metrics["tn"]
    fraud_total = metrics["tp"] + metrics["fn"]
    normal_total = metrics["fp"] + metrics["tn"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", total)
    with col2:
        st.metric("Actual Fraud", fraud_total, f"{fraud_total/total:.1%}" if total else "0%")
    with col3:
        st.metric("Actual Normal", normal_total, f"{normal_total/total:.1%}" if total else "0%")

    catch_rate = _safe_divide(metrics["tp"], fraud_total)
    miss_rate = _safe_divide(metrics["fn"], fraud_total)

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Fraud Catch Rate", f"{catch_rate:.2%}")
    with col_b:
        st.metric("Fraud Miss Rate", f"{miss_rate:.2%}")

    st.warning(f"""
    **Quick Summary:**
    - Out of {fraud_total} fraud transactions: caught {metrics['tp']} ({catch_rate:.1%}), missed {metrics['fn']} ({miss_rate:.1%})
    - Out of {normal_total} normal transactions: incorrectly blocked {metrics['fp']} ({metrics['fpr']:.1%})
    - **Main Issue:** Low precision -> Too many false positives
    - **Secondary Issue:** Recall may not meet 92% target
    """)


def _render_recommendations(metrics: Dict[str, Any], benchmark_df: pd.DataFrame):
    """Improvement recommendations"""
    st.markdown("## Recommendations")

    if not benchmark_df.empty:
        best_model = benchmark_df.loc[benchmark_df["f1"].idxmax()]
        best_recall_model = benchmark_df.loc[benchmark_df["recall"].idxmax()]
    else:
        best_model = None
        best_recall_model = None

    st.markdown("### 1. Model Selection")

    if best_model is not None and best_recall_model is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""
            **Best F1 Model:**
            - **{best_model['model']}** (F1 = {best_model['f1']:.4f})
            - Precision: {best_model['precision']:.2%}
            - Recall: {best_model['recall']:.2%}
            - ROC-AUC: {best_model['roc_auc']:.4f}
            - Latency p95: {best_model['latency_p95_ms']:.3f}ms
            """)

        with col2:
            st.info(f"""
            **Best Recall Model (Fraud Detection):**
            - **{best_recall_model['model']}** (Recall = {best_recall_model['recall']:.2%})
            - Precision: {best_recall_model['precision']:.2%}
            - F1: {best_recall_model['f1']:.4f}
            - ROC-AUC: {best_recall_model['roc_auc']:.4f}
            """)
    else:
        st.warning("No benchmark data available for model comparison.")

    st.markdown("### 2. Threshold Adjustment")

    st.markdown("""
    | Threshold | Precision | Recall | Use Case |
    |-----------|-----------|--------|----------|
    | **0.30** | Lower | Higher (>80%) | High-risk season, prioritize fraud detection |
    | **0.50** (current) | Medium | Medium | Balanced (but currently not optimal) |
    | **0.70** | Higher | Lower | Low-risk season, reduce false alarms |
    | **0.85** | Very High | Very Low | Only block obvious fraud |
    """)

    st.markdown("### 3. Three-Zone Decision Recommendation")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("""
        **ALLOW**
        - Score < 0.30
        - Auto-approve
        - No review needed
        """)

    with col2:
        st.warning("""
        **REVIEW**
        - Score 0.30 - 0.70
        - Queue for manual review
        - Requires human-in-the-loop
        """)

    with col3:
        st.error("""
        **BLOCK**
        - Score > 0.70
        - Auto-block transaction
        - Send alert to risk team
        """)

    st.markdown("### 4. Business Rules")

    st.markdown("""
    To reduce false positives and increase true positives, add these rules:

    1. **Auto REVIEW if:**
       - Amount > $1000
       - New device never seen before
       - Suspicious email domain
       - Transaction at 2-5 AM
       - Velocity: >3 transactions/hour/same card

    2. **Auto BLOCK if:**
       - Amount > $5000 + new device
       - Card previously reported fraud
       - IP from blacklist/VPN/Tor
       - 3 failed transactions + sudden success

    3. **Auto ALLOW if:**
       - User > 1 year old + trusted device
       - Amount < $50 + trusted merchant
       - Recurring transaction (subscription, utility)
    """)

    st.markdown("### 5. Model Improvement")

    st.markdown("""
    **Feature Engineering:**
    - Add velocity features (transactions/hour/day per card/user)
    - Add historical aggregates (30-day avg amount, std deviation)
    - Add time-based features (hour of day, day of week)
    - Add device/network features (ISP, country risk score)

    **Training Strategy:**
    - Use BalancedRandomForest (highest recall: 74.7%)
    - Oversample fraud with SMOTE or ADASYN
    - Ensemble: combine RandomForest + BalancedRandomForest
    - Calibrate probability with Isotonic or Platt scaling

    **Validation:**
    - Time-split CV instead of random split
    - Group-aware split by card/user
    - Out-of-time test set (last 30 days)
    """)


def _render_cost_benefit_analysis(metrics: Dict[str, Any]):
    """Cost-benefit analysis of false positives vs false negatives"""
    st.markdown("## Cost-Benefit Analysis")

    tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]

    cost_fn = 500
    cost_fp = 50
    benefit_tp = 500
    benefit_tn = 0

    total_cost_fn = fn * cost_fn
    total_cost_fp = fp * cost_fp
    total_benefit_tp = tp * benefit_tp

    net_benefit = total_benefit_tp - total_cost_fn - total_cost_fp

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("TP Benefit", f"${total_benefit_tp:,}")
    with col2:
        st.metric("FN Cost", f"-${total_cost_fn:,}")
    with col3:
        st.metric("FP Cost", f"-${total_cost_fp:,}")
    with col4:
        delta_color = "normal" if net_benefit >= 0 else "inverse"
        st.metric("Net Benefit", f"${net_benefit:,}", delta_color=delta_color)

    st.markdown(f"""
    **Assumptions:**
    - Each missed fraud (FN) costs: ${cost_fn}
    - Each false positive (FP) costs: ${cost_fp}
    - Each caught fraud (TP) saves: ${benefit_tp}

    **Result:**
    - Current net benefit: **${net_benefit:,}**
    - {'System is profitable' if net_benefit >= 0 else 'System is losing money (cost > benefit)'}
    """)

    st.markdown("### Optimization by Objective")

    scenarios = [
        {"name": "Minimize FP (Increase Precision)", "prec_target": 0.75, "rec_target": 0.50, "fpr_target": 0.05},
        {"name": "Balanced (Recommended Target)", "prec_target": 0.75, "rec_target": 0.92, "fpr_target": 0.10},
        {"name": "Maximize Recall (Maximum Protection)", "prec_target": 0.30, "rec_target": 0.95, "fpr_target": 0.20},
    ]

    for scenario in scenarios:
        est_tp = int(metrics["tp"] * (scenario["rec_target"] / max(metrics["recall"], 0.001)))
        est_fn = int((metrics["tp"] + metrics["fn"]) - est_tp)
        est_fp = int(metrics["fp"] * (scenario["fpr_target"] / max(metrics["fpr"], 0.001)))
        est_tn = int((metrics["fp"] + metrics["tn"]) - est_fp)

        cost = est_fn * cost_fn + est_fp * cost_fp
        benefit = est_tp * benefit_tp
        net = benefit - cost

        with st.container(border=True):
            st.markdown(f"**{scenario['name']}**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                - Precision target: {scenario['prec_target']:.0%}
                - Recall target: {scenario['rec_target']:.0%}
                - FPR target: {scenario['fpr_target']:.0%}
                """)
            with col_b:
                st.markdown(f"""
                - Estimated TP: {est_tp}, FN: {est_fn}
                - Estimated FP: {est_fp}, TN: {est_tn}
                - **Estimated Net Benefit: ${net:,}**
                """)


def main():
    render_page_header(
        "Truth vs Predict Analysis",
        "Gap analysis between ground truth and model predictions, with root-cause findings and remediation guidance.",
    )

    benchmark_df = _load_benchmark_results()

    with st.sidebar:
        st.header("Analysis Data")

        input_mode = st.radio(
            "Data Source",
            ["Auto from benchmark", "Manual input (from image)"],
        )

        if input_mode == "Manual input (from image)":
            tp = st.number_input("TP (Fraud Caught)", value=9, min_value=0)
            fp = st.number_input("FP (Normal Blocked)", value=271, min_value=0)
            fn = st.number_input("FN (Fraud Missed)", value=11, min_value=0)
            tn = st.number_input("TN (Normal Passed)", value=459, min_value=0)
        else:
            if not benchmark_df.empty:
                best = benchmark_df.loc[benchmark_df["f1"].idxmax()]
                tp = int(best["true_positives"])
                fp = int(best["false_positives"])
                fn = int(best["false_negatives"])
                tn = int(best["true_negatives"])
                st.info(f"Using model: {best['model']}")
            else:
                tp, fp, fn, tn = 0, 0, 0, 0

        st.divider()
        st.markdown("**Current Metrics:**")
        st.write(f"TP: {tp}, FP: {fp}")
        st.write(f"FN: {fn}, TN: {tn}")

    truth = np.array([1]*tp + [1]*fn + [0]*fp + [0]*tn)
    pred = np.array([1]*tp + [0]*fn + [1]*fp + [0]*tn)

    metrics = _analyze_truth_vs_predict(truth, pred)

    _render_gap_analysis(metrics, benchmark_df)
    _render_confusion_visualization(metrics)
    _render_root_cause_analysis(metrics, benchmark_df)
    _render_recommendations(metrics, benchmark_df)
    _render_cost_benefit_analysis(metrics)

    st.divider()
    st.caption("""
    Analysis auto-generated from benchmark results.
    Recommendations based on fraud detection best practices and current system metrics.
    """)


if __name__ == "__main__":
    main()
