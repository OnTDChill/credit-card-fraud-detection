"""
Overview Dashboard - Main landing page
Displays system status, KPIs, and quick navigation
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from streamlit_app.shared_ui import configure_dashboard_page, render_page_header

configure_dashboard_page("Overview")

from core.data_pipeline import DataPipeline


def load_pipeline_stats():
    pipeline = DataPipeline()
    train_df, test_df = pipeline.run_ingestion()
    stats = pipeline.get_validation_stats()
    return stats, train_df


def main():
    render_page_header(
        "Fraud Model Detective",
        "Production dashboard for fraud scoring, manual review, and model monitoring.",
    )
    
    # Load data
    try:
        stats, train_df = load_pipeline_stats()
    except Exception as e:
        st.error(f"Failed to load data pipeline: {e}")
        stats = {}
        train_df = pd.DataFrame()
    
    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Transactions",
            value=f"{stats.get('train_records', 0):,}",
            delta="IEEE-CIS Dataset"
        )
    
    with col2:
        fraud_count = stats.get('fraud_count', 0)
        fraud_pct = stats.get('fraud_percent', 0)
        st.metric(
            label="Fraud Cases",
            value=f"{fraud_count:,}",
            delta=f"{fraud_pct}% of total"
        )
    
    with col3:
        join_rate = stats.get('join_rate_percent', 'N/A')
        st.metric(
            label="Identity Match Rate",
            value=f"{join_rate}%" if join_rate != 'N/A' else "N/A",
            delta="Transaction + Identity"
        )
    
    with col4:
        st.metric(
            label="Models Trained",
            value="4",
            delta="RF, BRF, DT, IF"
        )
    
    with col5:
        st.metric(
            label="Best F1 Score",
            value="0.3693",
            delta="RandomForest Champion"
        )
    
    st.divider()
    
    # Two column layout
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        st.markdown("### System Architecture")
        
        st.markdown("""
        **Three-Tier Decision Engine:**
        
        1. **REVIEW ZONE** - Dynamic thresholds based on amount, customer tier, and merchant risk
           - Low amount (< $50): Relaxed thresholds, fewer reviews
           - High amount (> $1000): Stricter thresholds, more reviews
           - VIP customers: Whitelist bypass for trusted users
        
        2. **MANUAL OVERRIDE** - Admin correction with full audit trail
           - False Positive (FP): Whitelist customer, adjust threshold
           - False Negative (FN): Blacklist device/merchant, trigger retrain
        
        3. **CONTINUOUS LEARNING** - Feedback loop for model improvement
           - Weekly retrain on feedback pool
           - Champion/Challenger comparison before deployment
           - Drift monitoring with automatic alerts
        """)
        
        st.markdown("### Model Benchmark Summary")
        
        benchmark_path = os.path.join(os.path.dirname(__file__), "../../artifacts/benchmark_results.csv")
        if os.path.exists(benchmark_path):
            benchmark_df = pd.read_csv(benchmark_path)
            
            # Show champion
            best_model = benchmark_df.loc[benchmark_df['f1'].idxmax()]
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Champion Model", best_model['model'])
            with col_b:
                st.metric("F1 Score", f"{best_model['f1']:.4f}")
            with col_c:
                st.metric("Latency p95", f"{best_model['latency_p95_ms']:.3f}ms")
            
            st.dataframe(
                benchmark_df[['model', 'precision', 'recall', 'f1', 'roc_auc', 'latency_p95_ms']]
                .sort_values('f1', ascending=False)
                .round(4),
                width="stretch",
                hide_index=True
            )
        else:
            st.info("Run model benchmark to see results")
    
    with right_col:
        st.markdown("### Quick Actions")
        
        st.markdown("""
        - [Realtime Monitor](/realtime_monitor) - Score transactions and monitor alerts
        - [Model Metrics](/model_metrics) - Detailed model performance analysis
        - [Review Queue](/review_queue) - Approve or reject pending transactions
        - [Truth vs Predict](/truth_predict_analysis) - Gap analysis and recommendations
        """)
        
        st.markdown("### System Status")
        
        status_data = {
            "Component": ["Django Backend", "Streamlit UI", "SQLite Database", "ML Model"],
            "Status": ["Online", "Online", "Ready", "Loaded"],
            "Version": ["v1.2.0", "v1.0.0", "v1.0.0", "v1.2.0"]
        }
        st.dataframe(pd.DataFrame(status_data), width="stretch", hide_index=True)
        
        st.markdown("### Recommended Next Steps")
        st.info("""
        1. Review model performance in Model Metrics tab
        2. Check threshold configuration for optimal precision/recall balance
        3. Monitor review queue for false positives
        4. Trigger retrain when feedback pool reaches 100+ samples
        """)


if __name__ == "__main__":
    main()
