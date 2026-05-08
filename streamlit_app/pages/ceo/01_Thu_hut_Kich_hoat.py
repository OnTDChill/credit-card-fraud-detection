"""Module 1: Thu hút & Kích hoạt người dùng."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from streamlit_app.shared_ui import (
    configure_dashboard_page, render_page_header, render_kpi_card,
    render_ceo_command_panel,
)
from streamlit_app.components.dss_data_access import (
    get_marketing_summary, get_transaction_summary,
    get_cskh_summary, get_credit_portfolio
)

configure_dashboard_page("Khách hàng mới")
render_page_header(
    "1. Thu hút khách hàng mới",
    "Theo dõi hiệu quả chiến dịch marketing và tỷ lệ chuyển đổi người dùng",
    kicker="Báo cáo CEO",
)

# ── Filter ──
f_year = st.session_state.get("filter_year")
f_month = st.session_state.get("filter_month")

# ── Load data ──
mkt = get_marketing_summary(year=f_year, month=f_month)
txn = get_transaction_summary(year=f_year, month=f_month)
cskh = get_cskh_summary(year=f_year, month=f_month)

# ── Tính toán số liệu từ data thực ──
total_spend = mkt["campaign_spend"].sum() if not mkt.empty else 0
impressions = mkt["total_impressions"].sum() if not mkt.empty else 0
acq = int((mkt["total_impressions"] * mkt["avg_conversion"]).sum()) if not mkt.empty else 0
avg_conv = (acq / impressions) if impressions > 0 else 0
cac = total_spend / acq if acq > 0 else 0

# ── KPI Cards ──
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_kpi_card("Chi phí Marketing", f"{total_spend/1e9:,.1f} tỷ", "Tháng này")
with col2:
    render_kpi_card("Khách hàng mới", f"{acq:,} người", "Đăng ký thành công")
with col3:
    cost_per_user = f"{cac/1e3:,.0f}K" if cac > 0 else "Chưa có"
    render_kpi_card("Chi phí/khách hàng", cost_per_user, "Trung bình")
with col4:
    efficiency = "Tốt" if cac > 0 and cac < 100000 else "Cần cải thiện" if cac > 0 else "Chưa có dữ liệu"
    render_kpi_card("Hiệu quả", efficiency, "So với ngân sách")

st.divider()

# ── Chi tiết: Phễu chuyển đổi ──
with st.expander("Xem chi tiết: Hành trình khách hàng", expanded=False):
    if not mkt.empty and acq > 0:
        st.caption("Từ lúc biết đến app cho đến khi trở thành khách hàng thực sự")

        total_clicks = mkt["total_clicks"].sum() if "total_clicks" in mkt.columns else 0
        if impressions <= 0 or total_clicks <= 0:
            st.info("Chưa đủ dữ liệu click để vẽ phễu chuyển đổi.")
        else:
            funnel_steps = ["Hiển thị", "Click", "Đăng ký"]
            funnel_counts = [
                int(impressions),
                int(total_clicks),
                int(acq),
            ]
        
            df_funnel = pd.DataFrame({"Bước": funnel_steps, "Số người": funnel_counts})

            _funnel_text = [f"{c:,} người" for c in funnel_counts]
            fig_funnel = go.Figure(go.Funnel(
                y=df_funnel["Bước"],
                x=df_funnel["Số người"],
                text=_funnel_text,
                textposition="inside",
                textinfo="text",
                hovertemplate="<b>%{y}</b><br>Số người: %{x:,}<br>%{percentInitial:.0%} so với bước đầu<extra></extra>",
                marker=dict(color=["#4cc9f0", "#818cf8", "#22c55e"]),
            ))
            fig_funnel.update_layout(
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5eefb"),
                height=350,
            )
            st.plotly_chart(fig_funnel, width='stretch', key="m1_funnel_chart")
            st.caption("Chọn bước bên dưới để xem nhận xét chi tiết")

            # Drill-down: CEO chọn bước để xem chi tiết
            st.markdown("---")
            sel_step = st.radio("Chọn bước để xem chi tiết:", funnel_steps, horizontal=True, key="m1_funnel_step")
            step_idx = funnel_steps.index(sel_step)
            drop_rate = (1 - funnel_counts[step_idx] / funnel_counts[step_idx - 1]) * 100 if step_idx > 0 else 0
            step_detail = {
                "Hiển thị": {
                    "desc": f"**{funnel_counts[0]:,}** lượt hiển thị từ các chiến dịch marketing.",
                    "insight": "Đây là độ phủ ban đầu. Con số này phụ thuộc vào ngân sách và phân phối kênh.",
                    "direction": "Tối ưu phân bổ ngân sách để tăng reach với chi phí thấp nhất.",
                },
                "Click": {
                    "desc": f"**{funnel_counts[1]:,}** lượt click (mất **{drop_rate:.0f}%** so với hiển thị).",
                    "insight": "Tỷ lệ click phản ánh chất lượng thông điệp và nhắm mục tiêu.",
                    "direction": "A/B test nội dung quảng cáo và tối ưu phân khúc mục tiêu.",
                },
                "Đăng ký": {
                    "desc": f"**{funnel_counts[2]:,}** lượt đăng ký (mất **{drop_rate:.0f}%** so với click).",
                    "insight": "Tỷ lệ đăng ký phản ánh mức độ thuyết phục của landing/app.",
                    "direction": "Tối ưu onboarding và rút ngắn luồng đăng ký.",
                },
            }
            detail = step_detail[sel_step]
            st.info(detail["desc"])
            st.markdown(f"**Nhận xét:** {detail['insight']}")
            st.markdown(f"**Định hướng:** {detail['direction']}")
    else:
        st.info("Chưa đủ dữ liệu phễu (cần impressions, clicks, đăng ký).")

# ── Kịch bản 1: Thay đổi ngân sách ngắn hạn ──
with st.expander("Kịch bản: Thay đổi ngân sách Marketing", expanded=False):
    if not mkt.empty and total_spend > 0:
        st.caption("Kéo thanh trượt để xem kết quả dự kiến nếu tăng/giảm ngân sách")

        budget_change = st.slider(
            "Thay đổi ngân sách (%)",
            -50, 100, 0, 10,
            help="Âm = giảm ngân sách, Dương = tăng ngân sách",
            key="m1_budget_short_term"
        )

        new_budget = total_spend * (1 + budget_change / 100)
        weighted_cac = (
            (mkt["avg_cac"] * mkt["campaign_spend"]).sum() / total_spend
            if total_spend > 0 else mkt["avg_cac"].mean()
        )
        new_acq = int(new_budget / weighted_cac) if weighted_cac and weighted_cac > 0 else 0
        new_cac = new_budget / new_acq if new_acq > 0 else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            delta = new_budget - total_spend
            st.metric("Ngân sách mới", f"{new_budget/1e9:,.1f} tỷ", f"{delta/1e9:+,.1f} tỷ")
        with c2:
            delta_acq = new_acq - acq
            st.metric("Khách hàng dự kiến", f"{new_acq:,}", f"{delta_acq:+,} người")
        with c3:
            delta_cac = new_cac - cac if cac > 0 else 0
            st.metric("Chi phí/khách", f"{new_cac/1e3:,.0f}K", f"{delta_cac/1e3:+,.0f}K")

        if budget_change > 0:
            st.info(f"Tăng {budget_change}% ngân sách có thể mang về thêm {delta_acq:,} khách hàng mới.")
        elif budget_change < 0:
            st.warning(f"Giảm {abs(budget_change)}% ngân sách sẽ mất khoảng {abs(delta_acq):,} khách hàng tiềm năng.")
    else:
        st.info("Chưa có dữ liệu ngân sách.")

# ── Kịch bản 2: Dự báo tăng trưởng 3 năm ──
with st.expander("Kịch bản: Dự báo tăng trưởng khách hàng 1-3 năm", expanded=False):
    if not mkt.empty and acq > 0:
        hist_mkt = get_marketing_summary(year=None, month=None)
        hist_mkt = hist_mkt if not hist_mkt.empty else mkt
        monthly = (
            hist_mkt.assign(acq_est=hist_mkt["total_impressions"] * hist_mkt["avg_conversion"])
            .groupby(["year", "month"])
            ["acq_est"].sum()
            .reset_index()
            .sort_values(["year", "month"])
        )
        growth_series = monthly["acq_est"].pct_change().dropna()

        if len(growth_series) < 3:
            st.info("Dự báo dựa trên giả định thị trường fintech Việt Nam (do thiếu dữ liệu lịch sử).")
            rate_low, rate_mid, rate_high = -0.05, 0.15, 0.30
        else:
            st.info(f"Dự báo dựa trên tăng trưởng thực tế {len(growth_series)} kỳ gần nhất.")
            q_low, q_mid, q_high = growth_series.quantile([0.25, 0.5, 0.75]).tolist()

            def _annualize(monthly_rate: float) -> float:
                capped = max(-0.9, min(monthly_rate, 2.0))
                return (1 + capped) ** 12 - 1

            rate_low = _annualize(q_low)
            rate_mid = _annualize(q_mid)
            rate_high = _annualize(q_high)

        # Real-world context for each scenario
        scenario_context = {
            "Thận trọng": (
                "Kinh tế giảm tốc, lãi suất tăng, cạnh tranh gay gắt từ ngân hàng số. "
                "Người dùng thận trọng hơn trong chi tiêu và đăng ký dịch vụ mới."
            ),
            "Cơ bản": (
                "Kinh tế ổn định, thị trường fintech tiếp tục tăng trưởng theo xu hướng. "
                "Chính sách thanh toán không tiền mặt của Chính phủ hỗ trợ tích cực."
            ),
            "Tích cực": (
                "Kinh tế phục hồi mạnh, thu nhập khả dụng tăng, hạ tầng số phát triển. "
                "Mở rộng thành công sang thị trường nông thôn và nhóm khách trung niên."
            ),
        }

        scenario = st.radio(
            "Chọn kịch bản:",
            [
                f"Thận trọng ({rate_low*100:.0f}%/năm)",
                f"Cơ bản ({rate_mid*100:.0f}%/năm)",
                f"Tích cực ({rate_high*100:.0f}%/năm)",
            ],
            horizontal=True, key="m1_growth_scenario"
        )
        growth_rates = {
            f"Thận trọng ({rate_low*100:.0f}%/năm)": rate_low,
            f"Cơ bản ({rate_mid*100:.0f}%/năm)": rate_mid,
            f"Tích cực ({rate_high*100:.0f}%/năm)": rate_high,
        }
        rate = growth_rates[scenario]

        # Show context for selected scenario
        scenario_key = scenario.split(" (")[0]
        st.caption(f"**Giả định:** {scenario_context.get(scenario_key, '')}")

        years = ["Hiện tại", "Năm 1", "Năm 2", "Năm 3"]
        customers = [acq]
        for _ in range(3):
            customers.append(max(0, int(customers[-1] * (1 + rate))))

        fig_growth = go.Figure()
        fig_growth.add_trace(go.Bar(
            x=years, y=customers,
            text=[f"{c:,}" for c in customers],
            textposition="outside",
            marker_color=["#4cc9f0", "#818cf8", "#22c55e", "#f59e0b"],
            hovertemplate="<b>%{x}</b><br>Khách hàng: %{y:,}<extra></extra>",
        ))
        fig_growth.update_layout(
            title=f"Dự báo khách hàng - Kịch bản {scenario_key}",
            yaxis_title="Số khách hàng",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5eefb"), height=350,
        )
        st.plotly_chart(fig_growth, width='stretch', config={
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["downloadSVG", "downloadPNG"],
            "scrollZoom": True,
        })

        growth_3y = customers[3] - customers[0]
        growth_pct = (customers[3] / customers[0] - 1) * 100 if customers[0] > 0 else 0
        st.success(f"Sau 3 năm: Từ {customers[0]:,} -> {customers[3]:,} khach (+{growth_3y:,}, tang {growth_pct:.0f}%)")

        # Risk caveat
        st.caption(
            "Luu y: Du bao mang tinh tham khao, phu thuoc nhieu yeu to ben ngoai "
            "(chinh sach Nha nuoc, bien dong kinh te, hanh vi nguoi tieu dung). "
            "Can cap nhat kich ban hang quy."
        )
    else:
        st.info("Chua du du lieu marketing de du bao tang truong.")

# ── Cohort Retention Analysis ──
with st.expander("Phân tích Cohort: Giữ chân khách hàng", expanded=False):
    cskh_hist = get_cskh_summary(year=None, month=None)
    credit_hist = get_credit_portfolio(year=None, month=None)

    if cskh_hist.empty:
        st.info("Chưa đủ dữ liệu lịch sử cho phân tích cohort.")
    else:
        cskh_hist = cskh_hist.sort_values(["year", "month"]).reset_index(drop=True)
        # Build a churn series from available data
        churn_df = cskh_hist.copy()
        if not credit_hist.empty:
            users_by_period = (
                credit_hist.groupby(["year", "month"])["total_users"].sum().reset_index()
            )
            churn_df = churn_df.merge(users_by_period, on=["year", "month"], how="left")
        else:
            churn_df["total_users"] = None

        churn_df["churn_rate"] = pd.to_numeric(churn_df["churn_rate"], errors="coerce")
        churn_df["churn_count"] = pd.to_numeric(churn_df["churn_count"], errors="coerce")
        churn_df["total_users"] = pd.to_numeric(churn_df["total_users"], errors="coerce")

        needs_rate = churn_df["churn_rate"].isna() | (churn_df["churn_rate"] <= 0)
        can_calc = churn_df["churn_count"].gt(0) & churn_df["total_users"].gt(0)
        churn_df.loc[needs_rate & can_calc, "churn_rate"] = (
            churn_df.loc[needs_rate & can_calc, "churn_count"]
            / churn_df.loc[needs_rate & can_calc, "total_users"]
        )

        churn_df = churn_df.dropna(subset=["churn_rate"])
        
        # Build cohort matrix from available churn data
        n_months = len(churn_df)
        if n_months > 0:
            if n_months < 3:
                st.info(f"Hiển thị dựa trên {n_months} tháng dữ liệu có sẵn. Để có phân tích chi tiết hơn, cần thêm dữ liệu lịch sử.")
            
            ret = [100.0]
            for _, row in churn_df.iterrows():
                churn = float(row["churn_rate"])
                ret.append(ret[-1] * (1 - churn))
            ret = ret[1:]

            cohort_matrix = []
            for i in range(n_months):
                row_vals = [0.0] * n_months
                for j in range(i, n_months):
                    cum = 100.0
                    for k in range(i + 1, j + 1):
                        churn_k = float(churn_df.iloc[k]["churn_rate"])
                        cum *= (1 - churn_k)
                    row_vals[j] = float(cum)
                cohort_matrix.append(row_vals)

            labels = [f"T{i+1}" for i in range(n_months)]
            fig_cohort = go.Figure(data=go.Heatmap(
                z=cohort_matrix,
                x=labels,
                y=labels,
                colorscale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#22c55e"]],
                hovertemplate="Cohort %{y}<br>Tháng %{x}<br>Giữ chân: %{z:.1f}%<extra></extra>",
                colorbar=dict(title="Giữ chân %", tickfont=dict(color="#e5eefb")),
            ))
            fig_cohort.update_layout(
                title="Ma trận giữ chân khách hàng (Cohort)",
                xaxis_title="Tháng kể từ đăng ký",
                yaxis_title="Cohort (tháng đăng ký)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5eefb"), height=400,
            )
            st.plotly_chart(fig_cohort, width='stretch', config={"displayModeBar": False})
            st.caption("Dựa trên tỷ lệ rời bỏ thực tế từng tháng. Màu xanh = giữ chân tốt, đỏ = nhiều khách rời bỏ.")
        else:
            st.info("Chưa có dữ liệu để phân tích cohort.")

# ── Business Efficiency: ARPU, LTV, CAC ──
with st.expander("Hiệu quả kinh doanh khách hàng", expanded=False):
    total_rev = txn["revenue_estimated"].sum() if not txn.empty else 0
    credit_period = get_credit_portfolio(year=f_year, month=f_month)
    if credit_period.empty and f_year is not None and f_month is None:
        credit_period = get_credit_portfolio(year=f_year, month=None)
        if not credit_period.empty:
            latest_month = credit_period["month"].max()
            credit_period = credit_period[credit_period["month"] == latest_month]
    if credit_period.empty and f_year is None and f_month is None:
        credit_period = get_credit_portfolio(year=None, month=None)
        if not credit_period.empty:
            latest_year = credit_period["year"].max()
            latest_month = credit_period[credit_period["year"] == latest_year]["month"].max()
            credit_period = credit_period[
                (credit_period["year"] == latest_year) & (credit_period["month"] == latest_month)
            ]

    active_users = int(credit_period["total_users"].sum()) if not credit_period.empty else 0
    total_mkt_spend = mkt["campaign_spend"].sum() if not mkt.empty else 0
    acq_eff = int((mkt["total_impressions"] * mkt["avg_conversion"]).sum()) if not mkt.empty else 0
    cac_eff = total_mkt_spend / acq_eff if acq_eff > 0 else 0
    arpu_eff = total_rev / active_users if active_users > 0 else None
    churn_rate_eff = cskh["churn_rate"].mean() if not cskh.empty else None
    ltv_eff = (arpu_eff * 12 / churn_rate_eff) if arpu_eff is not None and churn_rate_eff and churn_rate_eff > 0 else None
    ltv_cac_eff = ltv_eff / cac_eff if ltv_eff is not None and cac_eff > 0 else None

    be1, be2, be3, be4 = st.columns(4)
    with be1:
        arpu_value = f"{arpu_eff/1e3:,.0f}K" if arpu_eff is not None else "Chưa có dữ liệu"
        render_kpi_card("ARPU", arpu_value, subtitle="Doanh thu/khách/tháng")
    with be2:
        ltv_value = f"{ltv_eff/1e6:,.1f} triệu" if ltv_eff is not None else "Chưa có dữ liệu"
        render_kpi_card("LTV", ltv_value, subtitle="Giá trị vòng đời")
    with be3:
        _cac_disp = f"{cac_eff/1e3:,.0f}K" if cac_eff > 0 else "Chưa có"
        render_kpi_card("CAC", _cac_disp, subtitle="Chi phí thu hút")
    with be4:
        if ltv_cac_eff is not None:
            eff = "Hiệu quả" if ltv_cac_eff >= 3 else "Tạm được" if ltv_cac_eff >= 1 else "Lỗ"
            render_kpi_card("LTV/CAC", eff, subtitle=f"{ltv_cac_eff:.1f}x")
        else:
            render_kpi_card("LTV/CAC", "Chưa có dữ liệu", subtitle="Cần dữ liệu đầy đủ")
    st.caption("LTV/CAC ≥ 3 là ngưỡng kinh doanh bền vững theo chuẩn fintech.")

st.divider()

# ── CEO Command Panel ──
M1_COMMANDS = [
    {
        "label": "Tối ưu ngân sách Marketing",
        "recipient": "Giám đốc Marketing",
        "next_steps": [
            "Giám đốc Marketing nhận chỉ thị và triệu tập đội ngũ trong 24h",
            "Trong 3 ngày, bộ phận Marketing phân tích hiệu quả từng kênh (Facebook, Google, Email) và lập báo cáo",
            "Cuối tuần, Giám đốc Marketing trình phương án phân bổ lại ngân sách để CEO duyệt",
        ],
    },
    {
        "label": "Cải thiện quy trình đăng ký",
        "recipient": "Giám đốc Sản phẩm",
        "next_steps": [
            "Giám đốc Sản phẩm nhận yêu cầu và giao đội UX khảo sát quy trình hiện tại",
            "Trong 5 ngày, đội Product đo lường tỷ lệ bỏ cuộc từng bước và xác định điểm nghẽn",
            "Đội Product đề xuất 2-3 phương án rút gọn quy trình và ước tính thời gian triển khai",
        ],
    },
    {
        "label": "Tạm dừng kênh quảng cáo kém hiệu quả",
        "recipient": "Giám đốc Marketing",
        "next_steps": [
            "Bộ phận Marketing rà soát chi phí/khách hàng của từng kênh trong 2 ngày",
            "Kênh có chi phí cao nhất sẽ bị tạm dừng, ngân sách được chuyển sang kênh hiệu quả hơn",
            "Sau 2 tuần, bộ phận Marketing gửi báo cáo so sánh để CEO quyết định mở lại hoặc cắt hẳn",
        ],
    },
]
render_ceo_command_panel("m1", M1_COMMANDS)

# Data simulation footnote
st.caption(
    "Dữ liệu mô phỏng quy mô MoMo cho mục đích demo. "
    "Methodology: Industry benchmarks + Synthetic aggregation."
)
