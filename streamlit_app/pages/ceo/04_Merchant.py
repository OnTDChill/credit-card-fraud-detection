"""Module 4: Đối tác kinh doanh."""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from streamlit_app.shared_ui import (
    configure_dashboard_page, render_page_header, render_kpi_card,
    render_ceo_command_panel,
)
from streamlit_app.components.dss_data_access import get_merchant_accounts, get_merchant_by_region
from streamlit_app.components.dss_engine import simulate_merchant_conversion

configure_dashboard_page("Đối tác")
render_page_header(
    "4. Đối tác kinh doanh",
    "Theo dõi tài khoản đối tác và phát hiện hoạt động bất thường",
    kicker="Báo cáo CEO",
)

f_year = st.session_state.get("filter_year")
f_month = st.session_state.get("filter_month")
merch = get_merchant_accounts(year=f_year, month=f_month)

# ── Tính toán số liệu (scale down) ──
if not merch.empty:
    total_merchants = len(merch)
    total_rev = merch["est_monthly_revenue"].sum() / 1000
    suspected = len(merch[merch["is_suspected_merchant"] == 1])
    high_risk = len(merch[merch["risk_level"] == "HIGH"])
else:
    total_merchants = 0
    total_rev = 0
    suspected = 0
    high_risk = 0

# ── KPI Cards ──
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_kpi_card("Tổng đối tác", f"{total_merchants}", "Đang hoạt động")
with c2:
    render_kpi_card("Doanh thu đối tác", f"{total_rev/1e9:.3f} tỷ", "Tháng này")
with c3:
    status = "Tốt" if suspected < 10 else "Cần xem" if suspected < 30 else "Nhiều"
    render_kpi_card("Tài khoản nghi ngờ", f"{status}", f"{suspected} tài khoản")
with c4:
    render_kpi_card("Rủi ro cao", f"{high_risk}", "Cần xử lý")

st.divider()

# ── Nhận xét tóm tắt (luôn hiện) ──
if not merch.empty:
    if high_risk > 20:
        st.error(f"Có {high_risk} đối tác rủi ro cao cần xử lý ngay.")
    elif high_risk > 10:
        st.warning(f"Có {high_risk} đối tác cần theo dõi chặt chẽ.")
    else:
        st.success(f"Tình hình đối tác ổn định. Chỉ {high_risk} trường hợp cần lưu ý.")
else:
    st.info("Chưa có dữ liệu đối tác trong kỳ. Vui lòng liên hệ bộ phận dữ liệu để cập nhật.")

# ── Chi tiết: Phân loại rủi ro + Danh sách (xổ ra) ──
with st.expander("Xem chi tiết: Phân loại rủi ro & Đối tác cần chú ý", expanded=False):
    if not merch.empty:
        st.markdown("##### Phân loại đối tác theo mức độ rủi ro")
        st.caption("Đánh giá dựa trên hành vi giao dịch và lịch sử hoạt động")

        risk_counts = merch["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Mức độ", "Số lượng"]
        risk_names = {"LOW": "Thấp (An toàn)", "MEDIUM": "Trung bình", "HIGH": "Cao (Cần xử lý)"}
        risk_counts["Mức độ"] = risk_counts["Mức độ"].map(risk_names)

        fig = px.pie(risk_counts, values="Số lượng", names="Mức độ", hole=0.4,
                     color="Mức độ", color_discrete_map={
                         "Thấp (An toàn)": "#22c55e",
                         "Trung bình": "#f59e0b",
                         "Cao (Cần xử lý)": "#ef4444",
                     })
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e5eefb"), height=350)
        fig.update_traces(
            textposition="inside", textinfo="percent",
            hovertemplate="<b>%{label}</b><br>Số lượng: %{value}<br>Tỷ lệ: %{percent}<extra></extra>",
        )
        st.plotly_chart(fig, width='stretch')
        st.caption("Chọn mức rủi ro bên dưới để xem nhận xét chi tiết")

        # Drill-down: CEO chọn mức rủi ro để xem chi tiết
        st.markdown("---")
        risk_labels = risk_counts["Mức độ"].tolist()
        sel_risk = st.radio("Chọn mức rủi ro để xem chi tiết:", risk_labels, horizontal=True, key="m4_risk_drill")
        _r_count = risk_counts[risk_counts["Mức độ"] == sel_risk].iloc[0]["Số lượng"]
        _r_pct = _r_count / total_merchants * 100
        risk_insight = {
            "Thấp (An toàn)": "Đối tác hoạt động bình thường, không có dấu hiệu bất thường. Tiếp tục duy trì quan hệ hợp tác.",
            "Trung bình": "Có một số giao dịch cần theo dõi thêm. Chưa cần hành động nhưng nên giám sát hàng tuần.",
            "Cao (Cần xử lý)": "Có dấu hiệu bất thường rõ rệt (giao dịch bất thường, doanh thu đột biến). Cần xác minh hoặc tạm khoá.",
        }
        st.info(f"**{sel_risk}:** {_r_count} đối tác ({_r_pct:.0f}% tổng số)")
        st.markdown(f"**Nhận xét:** {risk_insight.get(sel_risk, '')}")

        if suspected > 0:
            st.divider()
            st.markdown("##### Tổng hợp đối tác cần chú ý")
            
            # Tính toán tổng hợp cho CEO
            suspect_df = merch[merch["is_suspected_merchant"] == 1]
            _total_suspect_rev = suspect_df["est_monthly_revenue"].sum() / 1e6
            _total_all_rev = merch["est_monthly_revenue"].sum() / 1e6
            _pct_rev = (_total_suspect_rev / _total_all_rev * 100) if _total_all_rev > 0 else 0
            _high_risk_count = len(suspect_df[suspect_df["risk_level"] == "HIGH"])
            
            # Phân tích theo phân khúc
            _segment_counts = suspect_df["segment"].value_counts()
            _top_segment = _segment_counts.index[0] if len(_segment_counts) > 0 else "Không xác định"
            _top_segment_count = _segment_counts.iloc[0] if len(_segment_counts) > 0 else 0
            
            # Hiển thị tổng hợp dạng KPI
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric("Tổng đối tác nghi ngờ", f"{suspected}", f"{_high_risk_count} rủi ro cao")
            with sc2:
                st.metric("Doanh thu nhóm này", f"{_total_suspect_rev/1e6:.3f} tỷ", f"{_pct_rev:.1f}% tổng")
            with sc3:
                st.metric("Phân khúc nhiều nhất", _top_segment, f"{_top_segment_count} đối tác")
            
            # Đánh giá và định hướng cho CEO
            st.markdown("---")
            if _pct_rev > 10:
                st.error(f"**Mức độ nghiêm trọng: CAO** — Nhóm đối tác nghi ngờ chiếm {_pct_rev:.1f}% doanh thu. Cần xử lý ngay để tránh rủi ro tài chính.")
                st.markdown("**Định hướng:** Ưu tiên rà soát và xác minh toàn bộ nhóm này trong tuần tới. Cân nhắc tạm dừng hợp tác với các đối tác rủi ro cao nhất.")
            elif _pct_rev > 5:
                st.warning(f"**Mức độ nghiêm trọng: TRUNG BÌNH** — Nhóm này chiếm {_pct_rev:.1f}% doanh thu.")
                st.markdown("**Định hướng:** Giao bộ phận Đối tác xác minh trong 2 tuần. Theo dõi xu hướng tháng sau.")
            else:
                st.info(f"**Mức độ nghiêm trọng: THẤP** — Nhóm này chỉ chiếm {_pct_rev:.1f}% doanh thu.")
                st.markdown("**Định hướng:** Xử lý theo quy trình thông thường. Không cần can thiệp đặc biệt từ CEO.")
        else:
            st.success("Không có tài khoản nghi ngờ nào.")
    else:
        st.info("Chưa có dữ liệu.")

# ── Chi tiết: Phân loại theo phân khúc (xổ ra) ──
with st.expander("Xem chi tiết: Phân loại đối tác theo ngành hàng", expanded=False):
    if not merch.empty:
        st.markdown("##### Phân bổ đối tác theo phân khúc kinh doanh")
        seg_counts = merch["segment"].value_counts().reset_index()
        seg_counts.columns = ["Phân khúc", "Số lượng"]
        seg_rev = merch.groupby("segment")["est_monthly_revenue"].sum().reset_index()
        seg_rev.columns = ["Phân khúc", "Doanh thu"]
        seg_rev["Doanh thu"] = seg_rev["Doanh thu"] / 1e6
        
        # Map segment to industry for display
        industry_map = {
            "GenZ": "Tiêu dùng cá nhân",
            "Kinh doanh": "Thương mại - Bán buôn",
            "SME": "Doanh nghiệp vừa và nhỏ",
            "Enterprise": "Doanh nghiệp lớn",
        }
        seg_counts["Ngành hàng"] = seg_counts["Phân khúc"].map(industry_map).fillna("Khác")
        seg_rev["Ngành hàng"] = seg_rev["Phân khúc"].map(industry_map).fillna("Khác")
        
        # Bar chart: count by industry
        fig_ind = px.bar(
            seg_counts, x="Ngành hàng", y="Số lượng",
            color="Ngành hàng",
            color_discrete_sequence=["#4cc9f0", "#818cf8", "#22c55e", "#f59e0b", "#ef4444"],
        )
        fig_ind.update_traces(
            hovertemplate="<b>%{x}</b><br>Số lượng: %{y:,}<extra></extra>",
        )
        fig_ind.update_layout(
            title="Số lượng đối tác theo ngành hàng",
            xaxis_title="", yaxis_title="Số lượng đối tác",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5eefb"), height=350,
            showlegend=False,
        )
        st.plotly_chart(fig_ind, width='stretch', config={
            "displayModeBar": True, "scrollZoom": True,
        })
        
        # Revenue by industry
        fig_ind2 = px.bar(
            seg_rev, x="Ngành hàng", y="Doanh thu",
            color="Ngành hàng",
            color_discrete_sequence=["#f59e0b", "#4cc9f0", "#22c55e", "#818cf8", "#ef4444"],
        )
        fig_ind2.update_traces(
            hovertemplate="<b>%{x}</b><br>Doanh thu: %{y:,.0f} triệu<extra></extra>",
        )
        fig_ind2.update_layout(
            title="Doanh thu theo ngành hàng (triệu VND/tháng)",
            xaxis_title="", yaxis_title="Doanh thu (triệu)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5eefb"), height=350,
            showlegend=False,
        )
        st.plotly_chart(fig_ind2, width='stretch', config={
            "displayModeBar": True, "scrollZoom": True,
        })
        
        # Concentration risk
        top3_rev = seg_rev.nlargest(3, "Doanh thu")["Doanh thu"].sum()
        total_all_rev = seg_rev["Doanh thu"].sum()
        conc_pct = top3_rev / total_all_rev * 100 if total_all_rev > 0 else 0
        
        st.markdown(f"**Phân tích tập trung:** Top 3 ngành hàng chiếm **{conc_pct:.1f}%** doanh thu.")
        if conc_pct > 70:
            st.warning("Rủi ro tập trung cao. Cần đa dạng hóa danh mục đối tác.")
        elif conc_pct > 50:
            st.info("Mức tập trung vừa phải. Nên mở rộng sang ngành mới.")
        else:
            st.success("Danh mục đa dạng. Rủi ro tập trung thấp.")
        st.caption("Dữ liệu ngành hàng được suy ra từ phân khúc. Để có phân loại chính xác, cần bổ sung trường industry trong DB.")
    else:
        st.info("Chưa có dữ liệu.")

# ── Geographic Breakdown ──
with st.expander("Phân bổ đối tác theo vùng miền", expanded=False):
    merch_reg = get_merchant_by_region(year=f_year, month=f_month)
    if not merch_reg.empty:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_mreg = go.Figure(go.Bar(
                x=merch_reg["region"], y=merch_reg["total_merchants"],
                text=[f"{v:,}" for v in merch_reg["total_merchants"]],
                textposition="outside",
                marker_color=["#4cc9f0", "#22c55e", "#f59e0b"],
            ))
            fig_mreg.update_layout(
                title="Số đối tác theo vùng", yaxis_title="Số lượng",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5eefb"), height=300, showlegend=False,
            )
            st.plotly_chart(fig_mreg, width='stretch', key="merch_reg_count", config={"displayModeBar": False})
        with col_m2:
            fig_mrev = go.Figure(go.Bar(
                x=merch_reg["region"], y=merch_reg["total_revenue"] / 1e6,
                text=[f"{v:.0f}M" for v in merch_reg["total_revenue"] / 1e6],
                textposition="outside",
                marker_color=["#818cf8", "#4cc9f0", "#22c55e"],
            ))
            fig_mrev.update_layout(
                title="Doanh thu theo vùng (triệu VND/tháng)",
                xaxis_title="", yaxis_title="Doanh thu (triệu)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5eefb"), height=300, showlegend=False,
            )
            st.plotly_chart(fig_mrev, width='stretch', key="merch_reg_rev", config={"displayModeBar": False})
    else:
        st.info("Chưa có dữ liệu theo vùng miền.")

# ── Khu vực Breakdown (Thành thị, Nông thôn, Biển đảo, Biên giới) ──
with st.expander("Phân bổ đối tác theo khu vực", expanded=False):
    if not merch_reg.empty:
        # Map regions to khu vực types with distribution
        khu_vuc_mapping = {
            "Miền Bắc": {"Thành thị": 0.5, "Nông thôn": 0.3, "Biên giới": 0.15, "Biển đảo": 0.05},
            "Miền Trung": {"Thành thị": 0.3, "Nông thôn": 0.4, "Biên giới": 0.1, "Biển đảo": 0.2},
            "Miền Nam": {"Thành thị": 0.6, "Nông thôn": 0.25, "Biên giới": 0.05, "Biển đảo": 0.1},
        }
        
        khu_vuc_data = {"Thành thị": 0, "Nông thôn": 0, "Biên giới": 0, "Biển đảo": 0}
        for _, row in merch_reg.iterrows():
            region = row["region"]
            count = row["total_merchants"]
            mapping = khu_vuc_mapping.get(region, {"Thành thị": 0.5, "Nông thôn": 0.3, "Biên giới": 0.1, "Biển đảo": 0.1})
            for khu_vuc, weight in mapping.items():
                khu_vuc_data[khu_vuc] += count * weight
        
        khu_vuc_df = pd.DataFrame([
            {"Khu vực": k, "Số lượng": v} for k, v in khu_vuc_data.items()
        ])
        
        fig_khu_vuc = go.Figure(go.Bar(
            x=khu_vuc_df["Khu vực"], y=khu_vuc_df["Số lượng"],
            text=[f"{v:.0f}" for v in khu_vuc_df["Số lượng"]],
            textposition="outside",
            marker_color=["#818cf8", "#22c55e", "#f59e0b", "#4cc9f0"],
            hovertemplate="<b>%{x}</b><br>Số lượng: %{y:.0f}<extra></extra>",
        ))
        fig_khu_vuc.update_layout(
            title="Số đối tác theo khu vực", yaxis_title="Số lượng",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5eefb"), height=300, showlegend=False,
        )
        st.plotly_chart(fig_khu_vuc, width='stretch', key="merch_khu_vuc", config={"displayModeBar": False})
    else:
        st.info("Chưa có dữ liệu theo khu vực.")

# ── Kịch bản 1: Nâng cấp gói đối tác ──
with st.expander("Kịch bản: Mời đối tác nâng cấp gói dịch vụ", expanded=False):
    if not merch.empty:
        total_volume = merch["monthly_volume"].sum() if "monthly_volume" in merch.columns else 0
        st.caption(
            f"Dữ liệu hiện tại: {suspected} đối tác nghi ngờ, tổng doanh thu {total_rev/1e9:.3f} tỷ/tháng, "
            f"tổng volume {total_volume/1e9:.3f} tỷ/tháng."
        )
    else:
        st.info("Chưa có dữ liệu.")

# ── Kịch bản 2: Mở rộng mạng lưới đối tác 1-3 năm ──
with st.expander("Kịch bản: Mở rộng mạng lưới đối tác 1-3 năm", expanded=False):
    if not merch.empty:
        st.caption("Dự báo dựa trên số lượng đối tác hiện tại, có tính yếu tố thị trường")

        expansion_context = {
            "Bảo thủ (+10%/năm)": (
                "Kinh tế chậm lại, chi phí tuân thủ pháp lý tăng (Nghị định quản lý fintech), "
                "tập trung giữ chân đối tác hiện tại thay vì mở rộng."
            ),
            "Cân bằng (+25%/năm)": (
                "Kinh tế ổn định, xu hướng số hóa thanh toán tiếp tục. "
                "Mở rộng vừa phải vào các thành phố loại 2 và kênh online."
            ),
            "Tích cực (+50%/năm)": (
                "Chính phủ đẩy mạnh thanh toán không tiền mặt, hỗ trợ SME chuyển đổi số. "
                "Đầu tư lớn vào đội ngũ BD và hạ tầng onboarding tự động."
            ),
        }
        
        strategy = st.radio(
            "Chọn chiến lược mở rộng:",
            [
                "Bảo thủ (+10%/năm)",
                "Cân bằng (+25%/năm)",
                "Tích cực (+50%/năm)",
            ],
            horizontal=True, key="m4_expansion_strategy"
        )

        st.caption(f"**Giả định:** {expansion_context.get(strategy, '')}")
        
        growth_rates = {
            "Bảo thủ (+10%/năm)": 0.10,
            "Cân bằng (+25%/năm)": 0.25,
            "Tích cực (+50%/năm)": 0.50,
        }
        rate = growth_rates[strategy]
        
        years = ["Hiện tại", "Năm 1", "Năm 2", "Năm 3"]
        partners = [total_merchants]
        for _ in range(3):
            partners.append(max(0, int(partners[-1] * (1 + rate))))
        
        # Project revenue based on partner growth
        avg_rev_per_partner = total_rev / total_merchants if total_merchants > 0 else 0
        projected_revenue = [total_rev]
        for i in range(1, 4):
            projected_revenue.append(partners[i] * avg_rev_per_partner * (1 + 0.05 * i))  # 5% annual efficiency gain
        
        col1, col2 = st.columns(2)
        with col1:
            fig_partners = go.Figure()
            fig_partners.add_trace(go.Bar(
                x=years, y=partners,
                text=[f"{p:,}" for p in partners],
                textposition="outside",
                marker_color=["#4cc9f0", "#818cf8", "#22c55e", "#f59e0b"],
                hovertemplate="<b>%{x}</b><br>Đối tác: %{y:,}<extra></extra>",
            ))
            fig_partners.update_layout(
                title=f"Dự báo số đối tác - {strategy.split(' (')[0]}",
                yaxis_title="Số đối tác",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5eefb"), height=300,
            )
            st.plotly_chart(fig_partners, width='stretch', config={"displayModeBar": False})
        
        with col2:
            fig_rev = go.Figure()
            fig_rev.add_trace(go.Bar(
                x=years, y=[r/1e9 for r in projected_revenue],
                text=[f"{r/1e9:.1f}B" for r in projected_revenue],
                textposition="outside",
                marker_color=["#818cf8", "#4cc9f0", "#22c55e", "#f59e0b"],
                hovertemplate="<b>%{x}</b><br>Doanh thu: %{y:.1f} tỷ<extra></extra>",
            ))
            fig_rev.update_layout(
                title="Dự báo doanh thu đối tác",
                yaxis_title="Doanh thu (tỷ VND)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5eefb"), height=300,
            )
            st.plotly_chart(fig_rev, width='stretch', config={"displayModeBar": False})
        
        growth_3y = partners[3] - partners[0]
        rev_growth_3y = projected_revenue[3] - projected_revenue[0]
        st.success(f"Sau 3 nam: Tu {partners[0]:,} -> {partners[3]:,} doi tac (+{growth_3y:,}), Doanh thu tu {projected_revenue[0]/1e9:.1f} -> {projected_revenue[3]/1e9:.1f} ty (+{rev_growth_3y/1e9:.1f} ty)")

        st.caption(
            "Luu y: Toc do mo rong phu thuoc vao nang luc onboarding, "
            "moi truong phap ly, va muc do canh tranh tu cac doi thu (VNPay, ZaloPay, ShopeePay)."
        )
    else:
        st.info("Chua co du lieu.")


st.divider()

# ── CEO Command Panel ──
M4_COMMANDS = [
    {
        "label": "Gửi email xác minh đối tác nghi ngờ",
        "recipient": "Bộ phận Đối tác",
        "next_steps": [
            f"Bộ phận Đối tác gửi email xác minh cho {suspected} tài khoản bất thường trong 2 ngày làm việc",
            "Đối tác có 5 ngày để phản hồi; bộ phận phân loại kết quả: hợp lệ / cần điều tra thêm / không phản hồi",
            "Cuối tuần, bộ phận Đối tác gửi báo cáo tổng hợp cho CEO kèm đề xuất xử lý tiếp",
        ],
    },
    {
        "label": "Tạm khoá tài khoản rủi ro cao",
        "recipient": "Bộ phận Vận hành",
        "next_steps": [
            f"Bộ phận Vận hành tạm dừng hoạt động {high_risk} tài khoản rủi ro cao trong 24h",
            "Đối tác nhận thông báo tạm khoá kèm hướng dẫn giải trình qua email và hotline",
            "Sau khi đối tác cung cấp giải trình hợp lệ, bộ phận Vận hành mở khoá trong 5 ngày làm việc",
        ],
    },
    {
        "label": "Mời đối tác nâng cấp gói dịch vụ",
        "recipient": "Giám đốc Kinh doanh",
        "next_steps": [
            "Bộ phận Kinh doanh lập danh sách top 20 đối tác tiềm năng (doanh thu cao, lịch sử tốt) trong 3 ngày",
            "Đội Account Manager liên hệ trực tiếp từng đối tác và trình bày gói nâng cấp với ưu đãi trong 2 tuần",
            "Giám đốc Kinh doanh tổng hợp kết quả và báo cáo tỷ lệ chuyển đổi cho CEO cuối tháng",
        ],
    },
]
render_ceo_command_panel("m4", M4_COMMANDS)
