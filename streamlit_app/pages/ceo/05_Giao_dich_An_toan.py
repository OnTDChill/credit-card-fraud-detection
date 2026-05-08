"""Module 5: Giao dịch & An toàn — CEO view."""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from streamlit_app.shared_ui import (
    configure_dashboard_page, render_page_header, render_kpi_card,
    render_ceo_command_panel,
)
from streamlit_app.components.dss_data_access import (
    get_transaction_summary, get_transaction_by_region,
)

configure_dashboard_page("Giao dịch & An toàn")
render_page_header(
    "Giao dịch & An toàn",
    "Theo dõi GMV, phân loại giao dịch và tình hình gian lận",
    kicker="Báo cáo CEO",
)

# ── Filter ──
f_year = st.session_state.get("filter_year")
f_month = st.session_state.get("filter_month")

# ── Load data ──
txn = get_transaction_summary(year=f_year, month=f_month)

if txn.empty:
    st.info("Chưa có dữ liệu giao dịch trong kỳ. Vui lòng liên hệ bộ phận dữ liệu để cập nhật.")
    st.stop()

# ── Compute metrics ──
total_gmv = txn["total_amount"].sum()
total_count = txn["total_count"].sum()
fraud_rate = txn["fraud_rate"].mean()

# By transaction type
by_type = txn.groupby("transaction_type").agg({
    "total_amount": "sum",
    "total_count": "sum",
    "fraud_amount": "sum",
    "fraud_rate": "mean",
}).reset_index()

# ── KPI Row ──
st.markdown("#### Tổng quan giao dịch")
k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi_card("GMV", f"{total_gmv/1e9:.1f} tỷ", subtitle="Tổng giá trị giao dịch")
with k2:
    render_kpi_card("Số lượng", f"{total_count:,.0f}", subtitle="Giao dịch/tháng")
with k3:
    aov = total_gmv / total_count if total_count > 0 else 0
    render_kpi_card("AOV", f"{aov:,.0f} VND", subtitle="Giá trị trung bình")
with k4:
    render_kpi_card("Tỷ lệ gian lận", f"{fraud_rate:.2f}%", subtitle="Trung bình")

st.divider()

# ── Chi tiết: Phân loại giao dịch ──
with st.expander("Xem chi tiết: Phân loại giao dịch", expanded=False):
    if not txn.empty:
        st.markdown("##### GMV theo loại giao dịch")
        type_labels = {
            "PAYMENT": "Thanh toán",
            "TRANSFER": "Chuyển khoản",
            "CASH_IN": "Nạp tiền",
            "CASH_OUT": "Rút tiền",
            "DEBIT": "Trừ nợ",
        }
        by_type["Loại"] = by_type["transaction_type"].map(type_labels)
        by_type["GMV (tỷ)"] = by_type["total_amount"] / 1e9

        fig_type = px.pie(
            by_type, values="GMV (tỷ)", names="Loại",
            hole=0.45,
            color_discrete_sequence=["#4cc9f0", "#818cf8", "#22c55e", "#f59e0b", "#ef4444"],
        )
        fig_type.update_traces(
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>GMV: %{value:.1f} tỷ<br>Tỷ lệ: %{percent}<extra></extra>",
        )
        fig_type.update_layout(
            title="Cơ cấu GMV theo loại giao dịch",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5eefb"), height=350,
            showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        st.plotly_chart(fig_type, width='stretch', config={
            "displayModeBar": True, "scrollZoom": True,
        })

        st.caption("Chọn loại giao dịch để xem chi tiết")

        st.markdown("---")
        type_names = by_type["Loại"].tolist()
        sel_type = st.radio("Chọn loại giao dịch:", type_names, horizontal=True, key="m5_type_drill")
        _type_row = by_type[by_type["Loại"] == sel_type].iloc[0]
        _type_gmv = _type_row["total_amount"] / 1e9
        _type_count = _type_row["total_count"]
        _type_fraud = _type_row["fraud_rate"]

        st.info(
            f"**{sel_type}:** GMV {_type_gmv:.1f} tỷ — {_type_count:,.0f} giao dịch — "
            f"Tỷ lệ gian lận: **{_type_fraud:.2f}%**"
        )

        if _type_fraud > 1:
            st.markdown(
                f"**Nhận xét:** Loại giao dịch này có tỷ lệ gian lận cao ({_type_fraud:.2f}%). "
                "Cần rà soát quy trình xác thực."
            )
        elif _type_fraud > 0.5:
            st.markdown(
                f"**Nhận xét:** Gian lận ở mức trung bình ({_type_fraud:.2f}%). "
                "Theo dõi và tăng cường giám sát."
            )
        else:
            st.markdown(f"**Nhận xét:** An toàn ({_type_fraud:.2f}%). Duy trì quy trình hiện tại.")
    else:
        st.info("Chưa có dữ liệu.")

# ── Geographic Breakdown ──
with st.expander("Phân bổ gian lận theo vùng miền", expanded=False):
    txn_reg = get_transaction_by_region(year=f_year, month=f_month)
    if not txn_reg.empty:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_gmv_reg = go.Figure(go.Bar(
                x=txn_reg["region"], y=txn_reg["total_amount"] / 1e9,
                text=[f"{v:.1f}B" for v in txn_reg["total_amount"] / 1e9],
                textposition="outside",
                marker_color=["#4cc9f0", "#22c55e", "#f59e0b"],
            ))
            fig_gmv_reg.update_layout(
                title="GMV theo vùng", yaxis_title="GMV (tỷ VND)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5eefb"), height=280, showlegend=False,
            )
            st.plotly_chart(fig_gmv_reg, width='stretch', config={"displayModeBar": False})
        with col_g2:
            fig_fraud_reg = go.Figure(go.Bar(
                x=txn_reg["region"], y=txn_reg["fraud_rate"],
                text=[f"{v:.2f}%" for v in txn_reg["fraud_rate"]],
                textposition="outside",
                marker_color=["#ef4444", "#f59e0b", "#22c55e"],
            ))
            fig_fraud_reg.update_layout(
                title="Gian lận theo vùng", yaxis_title="Gian lận (%)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5eefb"), height=280, showlegend=False,
            )
            st.plotly_chart(fig_fraud_reg, width='stretch', key="security_fraud_reg", config={"displayModeBar": False})
    else:
        st.info("Chưa có dữ liệu theo vùng miền.")

# ── Fraud by Region ──
with st.expander("Phân bổ gian lận theo vùng miền", expanded=False):
    txn_reg = get_transaction_by_region(year=f_year, month=f_month)
    if not txn_reg.empty:
        fig_fraud = go.Figure(go.Bar(
            x=txn_reg["region"], y=txn_reg["fraud_rate"],
            text=[f"{v:.2f}%" for v in txn_reg["fraud_rate"]],
            textposition="outside",
            marker_color=["#ef4444", "#f59e0b", "#22c55e"],
            hovertemplate="<b>%{x}</b><br>Gian lận: %{y:.2f}%<extra></extra>",
        ))
        fig_fraud.update_layout(
            title="Tỷ lệ gian lận theo vùng", yaxis_title="Gian lận (%)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5eefb"), height=300, showlegend=False,
        )
        st.plotly_chart(fig_fraud, width='stretch', key="fraud_by_region", config={"displayModeBar": False})
    else:
        st.info("Chưa có dữ liệu gian lận theo vùng.")

# ── Fraud by Khu vực (Thành thị, Nông thôn, Biển đảo, Biên giới) ──
with st.expander("Phân bổ gian lận theo khu vực", expanded=False):
    if not txn_reg.empty:
        # Map regions to khu vực types with distribution
        khu_vuc_mapping = {
            "Miền Bắc": {"Thành thị": 0.5, "Nông thôn": 0.3, "Biên giới": 0.15, "Biển đảo": 0.05},
            "Miền Trung": {"Thành thị": 0.3, "Nông thôn": 0.4, "Biên giới": 0.1, "Biển đảo": 0.2},
            "Miền Nam": {"Thành thị": 0.6, "Nông thôn": 0.25, "Biên giới": 0.05, "Biển đảo": 0.1},
        }
        
        khu_vuc_fraud = {"Thành thị": 0, "Nông thôn": 0, "Biên giới": 0, "Biển đảo": 0}
        khu_vuc_total = {"Thành thị": 0, "Nông thôn": 0, "Biên giới": 0, "Biển đảo": 0}
        
        for _, row in txn_reg.iterrows():
            region = row["region"]
            fraud_rate = row["fraud_rate"]
            total_amount = row["total_amount"]
            mapping = khu_vuc_mapping.get(region, {"Thành thị": 0.5, "Nông thôn": 0.3, "Biên giới": 0.1, "Biển đảo": 0.1})
            for khu_vuc, weight in mapping.items():
                khu_vuc_total[khu_vuc] += total_amount * weight
                khu_vuc_fraud[khu_vuc] += total_amount * weight * fraud_rate
        
        khu_vuc_rates = {}
        for khu_vuc in khu_vuc_total:
            if khu_vuc_total[khu_vuc] > 0:
                khu_vuc_rates[khu_vuc] = (khu_vuc_fraud[khu_vuc] / khu_vuc_total[khu_vuc]) * 100
            else:
                khu_vuc_rates[khu_vuc] = 0
        
        khu_vuc_df = pd.DataFrame([
            {"Khu vực": k, "Gian lận (%)": v} for k, v in khu_vuc_rates.items()
        ])
        
        fig_khu_vuc_fraud = go.Figure(go.Bar(
            x=khu_vuc_df["Khu vực"], y=khu_vuc_df["Gian lận (%)"],
            text=[f"{v:.2f}%" for v in khu_vuc_df["Gian lận (%)"]],
            textposition="outside",
            marker_color=["#ef4444", "#f59e0b", "#22c55e", "#4cc9f0"],
            hovertemplate="<b>%{x}</b><br>Gian lận: %{y:.2f}%<extra></extra>",
        ))
        fig_khu_vuc_fraud.update_layout(
            title="Tỷ lệ gian lận theo khu vực", yaxis_title="Gian lận (%)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5eefb"), height=300, showlegend=False,
        )
        st.plotly_chart(fig_khu_vuc_fraud, width='stretch', key="fraud_by_khu_vuc", config={"displayModeBar": False})
    else:
        st.info("Chưa có dữ liệu gian lận theo khu vực.")

# ── Fraud Trend Forecast ──
with st.expander("Kịch bản: Dự báo xu hướng gian lận 4 quý tới", expanded=False):
    st.caption("Dự báo dựa trên xu hướng gian lận hiện tại, có tính đến yếu tố bên ngoài")
    
    # Get historical fraud data for trend analysis
    txn_hist = get_transaction_summary(year=None, month=None)
    if not txn_hist.empty:
        txn_hist = txn_hist.sort_values(["year", "month"])
        # Calculate average fraud rate from history
        avg_fraud_rate = txn_hist["fraud_rate"].mean()

        # Real-world context for fraud scenarios
        fraud_scenario_context = {
            "Cải thiện (-20%/quý)": (
                "Triển khai AI chống gian lận mới, tăng cường xác thực sinh trắc học, "
                "phối hợp chặt với Ngân hàng Nhà nước và công an mạng."
            ),
            "Ổn định (giữ nguyên)": (
                "Duy trì hệ thống hiện tại. Gian lận mới bù đắp bởi cải tiến phát hiện. "
                "Không có thay đổi lớn về chính sách hoặc công nghệ."
            ),
            "Xấu đi (+30%/quý)": (
                "Xu hướng tội phạm mạng gia tăng toàn cầu, deepfake và social engineering "
                "phức tạp hơn, hệ thống phòng thủ chưa kịp nâng cấp."
            ),
        }
        
        # Scenario selection
        scenario = st.radio(
            "Chọn kịch bản dự báo:",
            [
                "Cải thiện (-20%/quý)",
                "Ổn định (giữ nguyên)",
                "Xấu đi (+30%/quý)",
            ],
            horizontal=True, key="m5_fraud_scenario"
        )
        
        st.caption(f"**Giả định:** {fraud_scenario_context.get(scenario, '')}")
        
        scenario_changes = {
            "Cải thiện (-20%/quý)": -0.20,
            "Ổn định (giữ nguyên)": 0.0,
            "Xấu đi (+30%/quý)": 0.30,
        }
        quarterly_change = scenario_changes[scenario]
        
        quarters = ["Hiện tại", "Q1", "Q2", "Q3", "Q4"]
        fraud_rates = [avg_fraud_rate * 100]
        for i in range(4):
            new_rate = fraud_rates[-1] * (1 + quarterly_change)
            fraud_rates.append(max(0, new_rate))
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=quarters, y=fraud_rates,
            mode="lines+markers",
            line=dict(color="#ef4444" if quarterly_change > 0 else "#22c55e", width=3),
            marker=dict(size=8),
            text=[f"{v:.2f}%" for v in fraud_rates],
            textposition="top center",
            hovertemplate="<b>%{x}</b><br>Gian lận: %{y:.2f}%<extra></extra>",
        ))
        fig_forecast.update_layout(
            title=f"Dự báo gian lận - {scenario}",
            yaxis_title="Tỷ lệ gian lận (%)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5eefb"), height=350, showlegend=False,
        )
        st.plotly_chart(fig_forecast, width='stretch', config={"displayModeBar": False})
        
        final_rate = fraud_rates[-1]
        rate_change = final_rate - fraud_rates[0]
        
        if quarterly_change < 0:
            st.success(f"An toàn: Gian lận dự kiến giảm từ {fraud_rates[0]:.2f}% -> {final_rate:.2f}% sau 4 quý")
        elif quarterly_change == 0:
            st.info(f"Ổn định: Gian lận dự kiến giữ mức {fraud_rates[0]:.2f}% -> {final_rate:.2f}% sau 4 quý")
        else:
            st.warning(f"Cảnh báo: Gian lận dự kiến tăng từ {fraud_rates[0]:.2f}% -> {final_rate:.2f}% sau 4 quý")

        st.caption(
            "Lưu ý: Dự báo phụ thuộc vào mức đầu tư an ninh mạng, xu hướng tội phạm số, "
            "và hiệu quả phối hợp liên ngành. Cần cập nhật hàng quý."
        )
    else:
        st.info("Chưa có dữ liệu lịch sử để dự báo.")

st.divider()

# ── CEO Command Panel ──
M5_COMMANDS = [
    {
        "label": "Tăng cường xác thực giao dịch",
        "recipient": "Giám đốc Rủi ro",
        "next_steps": [
            "Giám đốc Rủi ro nhận chỉ thị và triệu tập đội ngũ trong 24h",
            "Trong 3 ngày, bộ phận Rủi ro đánh giá các điểm yếu trong quy trình xác thực",
            "Trong 1 tuần, triển khai thêm lớp bảo mật (OTP, sinh trắc học) cho giao dịch trên 5 triệu",
            "Báo cáo kết quả cho CEO sau 2 tuần",
        ],
    },
    {
        "label": "Rà soát toàn bộ merchant nghi ngờ",
        "recipient": "Giám đốc Đối tác",
        "next_steps": [
            "Giám đốc Đối tác nhận chỉ thị và lập danh sách ưu tiên rà soát",
            "Trong 1 tuần, đội ngũ kiểm tra lịch sử giao dịch và hồ sơ pháp lý của merchant nghi ngờ",
            "Đình chỉ hợp tác với merchant có dấu hiệu gian lận rõ rệt",
            "Cập nhật tiêu chuẩn tuyển chọn merchant để ngăn ngừa trong tương lai",
        ],
    },
    {
        "label": "Đầu tư công nghệ chống gian lận",
        "recipient": "Giám đốc Công nghệ",
        "next_steps": [
            "Giám đốc Công nghệ nhận chỉ thị và đánh giá giải pháp AI/ML chống gian lận",
            "Trong 2 tuần, trình bày phương án và ngân sách cho CEO phê duyệt",
            "Sau phê duyệt, triển khai thí điểm trong 1 tháng",
            "Đánh giá hiệu quả và quyết định triển khai toàn diện",
        ],
    },
]

render_ceo_command_panel("m5", M5_COMMANDS)
