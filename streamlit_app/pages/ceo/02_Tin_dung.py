"""Module 2: Tín dụng & Dòng tiền — CEO view."""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from streamlit_app.shared_ui import (
    configure_dashboard_page, render_page_header, render_kpi_card,
    render_ceo_command_panel,
)
from streamlit_app.components.dss_data_access import get_credit_portfolio, get_kpi_trend_history

configure_dashboard_page("Tín dụng")
render_page_header(
    "Tín dụng & Dòng tiền",
    "Theo dõi tình hình cho vay, nợ xấu, thanh khoản và dòng tiền",
    kicker="Báo cáo CEO",
)

f_year = st.session_state.get("filter_year")
f_month = st.session_state.get("filter_month")
credit = get_credit_portfolio(year=f_year, month=f_month)

# Normalize segment names to avoid duplicates
SEGMENT_NORMALIZE = {
    "Sinh vien": "Sinh viên",
    "Sinh viên": "Sinh viên",
}
if not credit.empty:
    credit["segment"] = credit["segment"].replace(SEGMENT_NORMALIZE)

# ── Tính toán số liệu ──
if not credit.empty:
    npl = credit["npl_rate"].mean()
    total_outstanding = credit["total_outstanding"].sum()
    total_interest = credit["revenue_interest"].sum()
    total_users = int(credit["total_users"].sum())
else:
    npl = total_outstanding = total_interest = 0
    total_users = 0

# ── KPI Cards ──
c1, c2, c3, c4 = st.columns(4)
with c1:
    status = "An toàn" if npl < 3 else "Cần theo dõi" if npl < 5 else "Rủi ro cao"
    render_kpi_card("Nợ khó đòi", status, f"{npl:.1f}% tổng dư nợ")
with c2:
    render_kpi_card("Tổng dư nợ", f"{total_outstanding/1e9:.1f} tỷ", "Đang cho vay")
with c3:
    render_kpi_card("Lãi thu được", f"{total_interest/1e6:.0f} triệu", "Tháng này")
with c4:
    render_kpi_card("Khách vay", f"{total_users:,} người", "Đang có dư nợ")

st.divider()

# ── Nhận xét tóm tắt (luôn hiện) ──
if not credit.empty:
    if npl < 3:
        st.success("Tình hình tốt. Nợ xấu trong tầm kiểm soát.")
    elif npl < 5:
        st.warning("Cần theo dõi. Nợ xấu đang tăng, cần siết chặt xét duyệt.")
    else:
        st.error("Cần hành động ngay. Nợ xấu vượt ngưỡng an toàn.")
else:
    st.info("Chưa có dữ liệu tín dụng trong kỳ. Vui lòng liên hệ bộ phận dữ liệu để cập nhật.")

# ── Chi tiết: Đồng hồ nợ xấu + Phân bổ dư nợ (xổ ra) ──
with st.expander("Xem chi tiết: Tình trạng nợ & Phân bổ dư nợ", expanded=False):
    if not credit.empty:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.caption("Tỷ lệ khách hàng không trả nợ đúng hạn (quá 90 ngày)")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=npl,
                number={"suffix": "%", "font": {"size": 48, "color": "#e5eefb"}},
                title={"text": "Tỷ lệ nợ xấu", "font": {"size": 16, "color": "#9fb0c7"}},
                gauge={
                    "axis": {"range": [0, 10], "tickwidth": 1, "tickcolor": "#e5eefb",
                             "tickvals": [0, 3, 5, 10], "ticktext": ["0%", "3%\nAn toàn", "5%\nCảnh báo", "10%"]},
                    "bar": {"color": "#ef4444" if npl > 3 else "#22c55e"},
                    "bgcolor": "rgba(11,23,40,0.84)",
                    "steps": [
                        {"range": [0, 3], "color": "rgba(34,197,94,0.3)"},
                        {"range": [3, 5], "color": "rgba(245,158,11,0.3)"},
                        {"range": [5, 10], "color": "rgba(239,68,68,0.3)"},
                    ],
                    "threshold": {"line": {"color": "#f59e0b", "width": 4}, "thickness": 0.75, "value": 3.0},
                },
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5eefb"),
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.markdown("##### Giải thích")
            st.markdown("""
            - **Xanh (0-3%):** An toàn
            - **Vàng (3-5%):** Cần theo dõi chặt
            - **Đỏ (>5%):** Rủi ro cao
            """)

        st.divider()
        st.markdown("##### Phân bổ dư nợ theo nhóm khách hàng")
        segment_data = credit.groupby("segment").agg({
            "total_outstanding": "sum",
            "npl_rate": "mean",
            "total_users": "sum",
        }).reset_index()
        segment_data["total_outstanding"] = segment_data["total_outstanding"]
        segment_data["total_users"] = segment_data["total_users"]

        segment_names = {
            "GenZ": "Trẻ (18-25)",
            "Sinh viên": "Sinh viên",
            "NV Văn phòng": "NV Văn phòng",
            "Kinh doanh": "Kinh doanh",
            "Hưu trí": "Hưu trí",
            "Tiểu thương": "Tiểu thương",
        }
        segment_data["Nhóm"] = segment_data["segment"].map(lambda x: segment_names.get(x, x))

        fig = px.pie(
            segment_data, values="total_outstanding", names="Nhóm",
            hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e5eefb"),
            height=350, legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        fig.update_traces(
            textposition="inside", textinfo="percent",
            hovertemplate="<b>%{label}</b><br>Dư nợ: %{value:,.0f}<br>Tỷ trọng: %{percent}<extra></extra>",
        )
        st.plotly_chart(fig, width='stretch')
        st.caption("Chọn nhóm khách hàng bên dưới để xem nhận xét chi tiết")

        # Drill-down: CEO chọn nhóm khách để xem chi tiết
        st.markdown("---")
        group_names = segment_data["Nhóm"].tolist()
        sel_group = st.radio("Chọn nhóm khách hàng để xem chi tiết:", group_names, horizontal=True, key="m2_segment")
        row = segment_data[segment_data["Nhóm"] == sel_group].iloc[0]
        _outstanding_b = row['total_outstanding'] / 1e9
        _npl_g = row['npl_rate']
        _users_g = int(row['total_users'])
        st.info(f"**{sel_group}:** Dư nợ **{_outstanding_b:.2f} tỷ** — Nợ xấu **{_npl_g:.1f}%** — **{_users_g:,}** khách vay")
        if _npl_g > 5:
            st.markdown(f"**Nhận xét:** Nhóm này có nợ xấu cao ({_npl_g:.1f}%). Nên xem xét siết điều kiện cho vay hoặc tăng yêu cầu bảo lãnh.")
        elif _npl_g > 3:
            st.markdown(f"**Nhận xét:** Nợ xấu ở mức trung bình ({_npl_g:.1f}%). Theo dõi sát và cân nhắc giảm hạn mức cho nhóm có lịch sử trả chậm.")
        else:
            st.markdown(f"**Nhận xét:** Nhóm này trả nợ tốt ({_npl_g:.1f}%). Có thể xem xét tăng hạn mức để tăng doanh thu lãi.")
    else:
        st.info("Chưa có dữ liệu.")

# ── Chính sách cho vay hiện tại ──
with st.expander("Chính sách cho vay hiện tại", expanded=False):
    if not credit.empty:
        st.caption("Tóm tắt chính sách theo dữ liệu kỳ báo cáo.")
        policy_df = credit.groupby("segment").agg(
            credit_limit=("credit_limit", "mean"),
            interest_rate=("interest_rate", "mean"),
            npl_rate=("npl_rate", "mean"),
            total_users=("total_users", "sum"),
            total_outstanding=("total_outstanding", "sum"),
        ).reset_index()
        st.dataframe(policy_df, width='stretch')
    else:
        st.info("Chưa có dữ liệu.")

# ── Xu hướng nợ xấu theo lịch sử ──
with st.expander("Xu hướng nợ xấu theo lịch sử", expanded=False):
    hist_npl = get_kpi_trend_history(months=12)
    if not hist_npl.empty:
        hist_npl = hist_npl.sort_values(["year", "month"]).reset_index(drop=True)
        fig_npl = go.Figure()
        hist_labels = [f"T{i+1}" for i in range(len(hist_npl))]
        fig_npl.add_trace(go.Scatter(
            x=hist_labels, y=hist_npl["npl_rate"],
            mode="lines+markers", name="Lịch sử",
            line=dict(color="#9fb0c7", width=2),
            marker=dict(size=6),
            hovertemplate="%{x}<br>NPL: %{y:.1f}%<extra></extra>",
        ))
        fig_npl.update_layout(
            title="Xu hướng nợ xấu theo lịch sử",
            yaxis_title="Tỷ lệ nợ xấu (%)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5eefb"), height=350,
            showlegend=False,
        )
        st.plotly_chart(fig_npl, width='stretch', config={
            "displayModeBar": True, "scrollZoom": True,
        })
    else:
        st.info("Chưa có dữ liệu.")

st.divider()

# ── CEO Command Panel ──
M2_COMMANDS = [
    {
        "label": "Siết chặt chính sách cho vay",
        "recipient": "Giám đốc Rủi ro",
        "next_steps": [
            "Giám đốc Rủi ro nhận chỉ thị và tổ chức họp nội bộ bộ phận trong 24h",
            "Trong 3 ngày, bộ phận Rủi ro xây dựng bảng hạn mức mới theo từng phân khúc khách hàng",
            "Chính sách mới được áp dụng sau khi CEO duyệt; bộ phận Rủi ro báo cáo tác động hàng tuần trong 1 tháng đầu",
        ],
    },
    {
        "label": "Đẩy mạnh thu hồi nợ xấu",
        "recipient": "Bộ phận Thu hồi nợ",
        "next_steps": [
            "Bộ phận Thu hồi nợ lập danh sách khách hàng quá hạn >90 ngày trong 2 ngày",
            "Đội thu hồi liên hệ trực tiếp nhóm nợ lớn nhất trước, ưu tiên thương lượng trả góp",
            "Các khoản không thể thu hồi sau 30 ngày sẽ được chuyển sang xem xét bán nợ hoặc xoá nợ",
        ],
    },
    {
        "label": "Tăng hạn mức cho khách VIP",
        "recipient": "Giám đốc Kinh doanh",
        "next_steps": [
            "Bộ phận Kinh doanh phối hợp Rủi ro lọc danh sách khách VIP (trả nợ tốt, thu nhập ổn định)",
            "Trong 5 ngày, đề xuất gói hạn mức mới kèm điều kiện cụ thể cho từng nhóm VIP",
            "Triển khai thí điểm với 500 khách VIP trong 2 tuần, sau đó báo cáo kết quả cho CEO",
        ],
    },
]
render_ceo_command_panel("m2", M2_COMMANDS)
