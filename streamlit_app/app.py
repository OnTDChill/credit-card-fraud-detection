"""
Fraud Model Detective - Unified Streamlit Application
Single entry point for all fraud detection dashboards
"""
import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from streamlit_app.shared_ui import configure_dashboard_page

configure_dashboard_page()

# Main navigation pages
pg = st.navigation([
    st.Page("pages/overview.py", title="Overview"),
    st.Page("pages/realtime_monitor.py", title="Realtime Monitor"),
    st.Page("pages/model_metrics.py", title="Model Metrics"),
    st.Page("pages/review_queue.py", title="Review Queue"),
    st.Page("pages/truth_predict_analysis.py", title="Truth vs Predict"),
])

pg.run()
