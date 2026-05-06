"""Module 3: Hệ sinh thái dịch vụ."""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import pandas as pd

from streamlit_app.shared_ui import (
    configure_dashboard_page, render_page_header, render_kpi_card,
    render_ceo_command_panel,
)
from streamlit_app.components.dss_data_access import get_service_ecosystem
from streamlit_app.components.dss_engine import simulate_combo_discount

configure_dashboard_page("Dịch vụ")
render_page_header(
    "3. Hệ sinh thái dịch vụ",
    "Theo dõi các dịch vụ và cơ hội bán chéo cho khách hàng",
    kicker="Báo cáo CEO",
)

f_year = st.session_state.get("filter_year")
f_month = st.session_state.get("filter_month")
eco = get_service_ecosystem(year=f_year, month=f_month)

# ── Tính toán số liệu ──
if not eco.empty:
    total_services = len(eco)
    total_rev = (eco["revenue_a"].sum() + eco["revenue_b"].sum()) / 1000
    top_idx = eco["support_count"].idxmax()
    top_combo = f"{eco.loc[top_idx, 'service_a']} + {eco.loc[top_idx, 'service_b']}"
    top_users = eco.loc[top_idx, "support_count"]
    cross_sell_rate = eco["support_pct"].mean() if "support_pct" in eco.columns else None
else:
    total_services = 0
    total_rev = 0
    top_combo = "Chưa có"
    top_users = 0
    cross_sell_rate = None

# ── KPI Cards ──
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_kpi_card("Số combo dịch vụ", f"{total_services}", "Đang hoạt động")
with c2:
    render_kpi_card("Doanh thu dịch vụ", f"{total_rev/1e9:.1f} tỷ", "Tháng này")
with c3:
    render_kpi_card("Combo phổ biến nhất", top_combo[:20], f"{top_users:,} khách")
with c4:
    cross_sell_value = f"{cross_sell_rate:.1f}%" if cross_sell_rate is not None else "Chưa có dữ liệu"
    render_kpi_card("Tỷ lệ bán chéo", cross_sell_value, "Khách dùng >1 dịch vụ")

st.divider()

# ── Nhận xét tóm tắt (luôn hiện) ──
if not eco.empty:
    st.success(f"Có {total_services} combo đang hoạt động. "
               f"Combo **{top_combo}** phổ biến nhất với {top_users:,} khách sử dụng.")
else:
    st.info("Chưa có dữ liệu dịch vụ trong kỳ. Vui lòng liên hệ bộ phận dữ liệu để cập nhật.")

# ── Chi tiết: Doanh thu + Top combo (xổ ra) ──
with st.expander("Xem chi tiết: Doanh thu & Combo phổ biến", expanded=False):
    if not eco.empty:
        st.markdown("##### Doanh thu theo dịch vụ")
        st.caption("Top 10 dịch vụ có doanh thu cao nhất")

        services_a = eco[["service_a", "revenue_a"]].rename(columns={"service_a": "Dịch vụ", "revenue_a": "Doanh thu"})
        services_b = eco[["service_b", "revenue_b"]].rename(columns={"service_b": "Dịch vụ", "revenue_b": "Doanh thu"})
        all_services = pd.concat([services_a, services_b])
        service_rev = all_services.groupby("Dịch vụ")["Doanh thu"].sum().reset_index()
        service_rev["Doanh thu"] = service_rev["Doanh thu"] / 1e9
        service_rev = service_rev.sort_values("Doanh thu", ascending=True).tail(10)

        fig = px.bar(service_rev, x="Doanh thu", y="Dịch vụ", orientation="h",
                     color="Doanh thu", color_continuous_scale="Blues")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5eefb"), height=400, showlegend=False,
            xaxis_title="Doanh thu (tỷ VND)", yaxis_title="",
            coloraxis_showscale=False,
        )
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>Doanh thu: %{x:.1f} tỷ VND<extra></extra>",
        )
        st.plotly_chart(fig, width='stretch')

        st.divider()
        st.markdown("##### Combo dịch vụ phổ biến")
        st.caption("Khách hàng thường sử dụng các dịch vụ này cùng nhau")
        top_combos = eco.nlargest(5, "support_count")
        combo_labels = []
        for _, row in top_combos.iterrows():
            combo_name = f"{row['service_a']} + {row['service_b']}"
            users = row["support_count"]
            rev = (row["revenue_a"] + row["revenue_b"]) / 1e9
            st.markdown(f"**{combo_name}** — {users:,} khách — Doanh thu: {rev:.1f} tỷ")
            combo_labels.append(combo_name)
    else:
        st.info("Chưa có dữ liệu.")

# ── Kịch bản 1: Tung combo khuyến mãi ──
with st.expander("Kịch bản: Tung combo khuyến mãi", expanded=False):
    if not eco.empty:
        st.caption(f"Dự báo dựa trên {len(eco)} combo dịch vụ thực tế từ data hiện tại (doanh thu, lift, margin)")

        pairs = [(r["service_a"], r["service_b"]) for _, r in eco.iterrows()]
        pair_labels = [f"{a} + {b}" for a, b in pairs]
        sel_label = st.selectbox("Chọn combo dịch vụ", pair_labels)
        sel_pair = pairs[pair_labels.index(sel_label)]
        discount = st.slider("Mức giảm giá (%)", 5, 30, 10, 5, key="m3_discount")

        sim = simulate_combo_discount(sel_pair, discount, eco)
        c1, c2, c3 = st.columns(3)
        with c1:
            uplift_billion = sim['total_uplift'] / 1e9
            st.metric("Tăng doanh thu", f"+{uplift_billion:.3f} tỷ")
        with c2:
            st.metric("Khách hàng mới", f"+{int(sim['total_uplift']/1e6*10):,}")
        with c3:
            roi = sim.get("roi", 1.0)
            st.metric("Hiệu quả", "Tốt" if roi > 1 else "Cần xem xét")

        if roi > 1.5:
            st.success(f"Combo này có tiềm năng tốt. Giảm {discount}% có thể tăng doanh thu đáng kể.")
        elif roi > 1:
            st.info("Combo khả thi. Cân nhắc thử nghiệm với nhóm khách hàng nhỏ trước.")
        else:
            st.warning("Mức giảm giá này có thể không hiệu quả. Cân nhắc giảm ít hơn.")
    else:
        st.info("Chưa có dữ liệu.")

# ── Kịch bản 2: Tối ưu danh mục dịch vụ ──
with st.expander("Kịch bản: Tối ưu danh mục dịch vụ 1-2 năm tới", expanded=False):
    if not eco.empty:
        st.caption(f"Phân loại BCG dựa trên doanh thu và số khách thực tế từ {len(eco)} combo trong data hiện tại")
        
        # Tính toán doanh thu và số combo cho mỗi dịch vụ
        services_a = eco[["service_a", "revenue_a", "support_count"]].rename(
            columns={"service_a": "service", "revenue_a": "revenue"})
        services_b = eco[["service_b", "revenue_b", "support_count"]].rename(
            columns={"service_b": "service", "revenue_b": "revenue"})
        all_svc = pd.concat([services_a, services_b])
        svc_summary = all_svc.groupby("service").agg({
            "revenue": "sum", "support_count": "sum"
        }).reset_index()
        svc_summary["revenue_b"] = svc_summary["revenue"] / 1e9
        
        # Phân loại theo ma trận BCG đơn giản
        rev_median = svc_summary["revenue_b"].median()
        user_median = svc_summary["support_count"].median()
        
        def classify(row):
            if row["revenue_b"] >= rev_median and row["support_count"] >= user_median:
                return "Ngôi sao (Đầu tư mạnh)"
            elif row["revenue_b"] >= rev_median:
                return "Bò sữa (Duy trì)"
            elif row["support_count"] >= user_median:
                return "Dấu hỏi (Cân nhắc)"
            else:
                return "Chó (Xem xét cắt)"
        
        svc_summary["Phân loại"] = svc_summary.apply(classify, axis=1)
        
        # Biểu đồ scatter tương tác
        fig_portfolio = px.scatter(
            svc_summary, x="support_count", y="revenue_b",
            text="service", color="Phân loại",
            color_discrete_map={
                "Ngôi sao (Đầu tư mạnh)": "#22c55e",
                "Bò sữa (Duy trì)": "#4cc9f0",
                "Dấu hỏi (Cân nhắc)": "#f59e0b",
                "Chó (Xem xét cắt)": "#ef4444",
            },
            size="revenue_b", size_max=40,
        )
        fig_portfolio.update_traces(
            textposition="top center",
            hovertemplate="<b>%{text}</b><br>Doanh thu: %{y:.1f} tỷ<br>Khách hàng: %{x:,}<extra></extra>",
        )
        fig_portfolio.add_hline(y=rev_median, line_dash="dash", line_color="#9fb0c7", opacity=0.5)
        fig_portfolio.add_vline(x=user_median, line_dash="dash", line_color="#9fb0c7", opacity=0.5)
        fig_portfolio.update_layout(
            title="Ma trận danh mục dịch vụ",
            xaxis_title="Số khách hàng sử dụng",
            yaxis_title="Doanh thu (tỷ VND)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5eefb"), height=400,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        st.plotly_chart(fig_portfolio, width='stretch', config={
            "displayModeBar": True, "scrollZoom": True,
        })
        
        # Tóm tắt định hướng
        stars = svc_summary[svc_summary["Phân loại"].str.contains("Ngôi sao")]
        dogs = svc_summary[svc_summary["Phân loại"].str.contains("Chó")]
        
        st.markdown("##### Định hướng chiến lược:")
        if len(stars) > 0:
            st.success(f"**Đầu tư mạnh:** {', '.join(stars['service'].tolist())} — Đây là dịch vụ trụ cột, cần ưu tiên nguồn lực.")
        if len(dogs) > 0:
            st.warning(f"**Xem xét cắt giảm:** {', '.join(dogs['service'].tolist())} — Doanh thu thấp, ít khách. Cân nhắc dừng hoặc tái cấu trúc.")
    else:
        st.info("Chưa có dữ liệu.")

st.divider()

# ── CEO Command Panel ──
M3_COMMANDS = [
    {
        "label": "Tung combo khuyến mãi",
        "recipient": "Giám đốc Marketing",
        "next_steps": [
            "Giám đốc Marketing nhận chỉ thị và giao đội thiết kế chương trình trong 3 ngày (điều kiện, thời gian, mức giảm)",
            "Bộ phận Marketing gửi thông báo tới khách hàng mục tiêu qua app và email",
            "Bộ phận Marketing theo dõi tỷ lệ đăng ký hàng ngày và gửi báo cáo tuần cho CEO",
        ],
    },
    {
        "label": "Đẩy mạnh bán chéo",
        "recipient": "Giám đốc Kinh doanh",
        "next_steps": [
            "Giám đốc Kinh doanh phân bổ chỉ tiêu bán chéo cho từng nhóm và chuẩn bị tài liệu hướng dẫn",
            "Đội Sales bắt đầu giới thiệu combo cho khách hiện tại khi giao dịch tại quầy/qua app",
            "Sau 2 tuần, Giám đốc Kinh doanh gửi báo cáo kết quả kèm đề xuất điều chỉnh cho CEO",
        ],
    },
    {
        "label": "Dừng dịch vụ kém hiệu quả",
        "recipient": "Giám đốc Sản phẩm",
        "next_steps": [
            "Giám đốc Sản phẩm rà soát danh sách dịch vụ có doanh thu thấp nhất và tỷ lệ sử dụng <5%",
            "Bộ phận CSKH thông báo cho khách hàng đang sử dụng trước 30 ngày và hướng dẫn chuyển sang dịch vụ thay thế",
            "Nguồn lực được chuyển sang phát triển dịch vụ có tiềm năng; Giám đốc Sản phẩm báo cáo tiến độ hàng tháng",
        ],
    },
]
render_ceo_command_panel("m3", M3_COMMANDS)
