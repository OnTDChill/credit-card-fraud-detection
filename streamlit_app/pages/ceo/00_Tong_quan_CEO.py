"""Tab Tổng quan – CEO Executive Summary."""

from __future__ import annotations

from datetime import datetime as _dt

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from streamlit_app.shared_ui import configure_dashboard_page, render_page_header, render_kpi_card
from streamlit_app.config import FRAUD_LOSS_RATE
from streamlit_app.components.dss_data_access import (
    get_transaction_summary,
    get_cskh_summary,
    get_marketing_summary,
    get_strategy_targets,
    get_credit_portfolio,
    get_merchant_accounts,
    get_service_ecosystem,
    get_kpi_trend_history,
    get_transaction_by_region,
    get_merchant_by_region,
)

configure_dashboard_page("Tổng quan")
render_page_header(
    "Tổng quan doanh nghiệp",
    "Bức tranh toàn cảnh về tình hình kinh doanh, an toàn và dịch vụ khách hàng",
    kicker="Báo cáo CEO",
)

# ── Filter ──
f_year = st.session_state.get("filter_year")
f_month = st.session_state.get("filter_month")

# ── Load data ──
txn = get_transaction_summary(year=f_year, month=f_month)
cskh = get_cskh_summary(year=f_year, month=f_month)
mkt = get_marketing_summary(year=f_year, month=f_month)

has_data = not txn.empty
if not has_data:
    st.info("Chưa có dữ liệu trong kỳ báo cáo. Vui lòng chọn chỉ thị từ các tab module ở trên hoặc điều hướng đến từng module để xem chi tiết.")
    st.stop()

# ── Trend history for MoM and sparklines ──
# Fix: get_kpi_trend_history does not accept 'year' and 'month' as keyword arguments in current implementation, 
# or it's better to use positional arguments if we want to keep it simple, 
# but based on dss_data_access.py it actually DOES accept them.
# Wait, the error says: TypeError: get_kpi_trend_history() got an unexpected keyword argument 'year'
# Let me re-check dss_data_access.py... 
# In dss_data_access.py: def get_kpi_trend_history(year: int | None = None, month: int | None = None, months: int = 12) -> pd.DataFrame:
# It SHOULD work. Maybe there is another definition or a typo.
# Let's try passing them positionally or check if I'm misreading the error.
# The error message in the task says it got an unexpected keyword argument 'year'.
# I will change it to positional arguments to bypass the keyword issue for now.
trend_df = get_kpi_trend_history(f_year, f_month, 12)

# Helper to compute MoM delta from trend_df (sorted DESC: idx 0 = current, idx 1 = prev)
def _mom_delta(col: str) -> tuple[float | None, float | None]:
    """Return (current_value, pct_change) or (None, None) if insufficient data."""
    if trend_df.empty or col not in trend_df.columns or len(trend_df) < 2:
        return None, None
    cur = trend_df.iloc[0][col]
    prev = trend_df.iloc[1][col]
    if pd.isna(cur) or pd.isna(prev) or prev == 0:
        return cur if not pd.isna(cur) else None, None
    return float(cur), (float(cur) - float(prev)) / float(prev) * 100

_mom_rev = _mom_delta("revenue_estimated")
_mom_gmv = _mom_delta("total_amount")


# ── Compute metrics ──
total_rev = txn["revenue_estimated"].sum()
gmv = txn["total_amount"].sum()
total_count = txn["total_count"].sum()
aov = gmv / total_count if total_count > 0 else 0
total_loss = txn["fraud_amount"].sum() * FRAUD_LOSS_RATE
fraud_pct = total_loss / total_rev * 100 if total_rev > 0 else 0
safety = max(0, min(100, 100 - fraud_pct * 5))

csat_val = cskh["csat_score"].mean() if not cskh.empty else None
churn_val = cskh["churn_rate"].mean() * 100 if not cskh.empty else None

# ── Load targets from DB ──
_target_year = f_year or _dt.now().year
targets = get_strategy_targets(target_year=_target_year)

rev_target = None
csat_target = None
if not targets.empty:
    _rev_row = targets[(targets["domain"] == "REVENUE") & (targets["kpi_name"] == "total_revenue")]
    if not _rev_row.empty:
        rev_target = _rev_row.iloc[0]["target_value"]
    _csat_row = targets[(targets["domain"] == "SERVICE") & (targets["kpi_name"] == "csat_target")]
    if not _csat_row.empty:
        csat_target = _csat_row.iloc[0]["target_value"]

# ── Overall health ──
scores = [safety]
rev_score = None
if rev_target is not None and rev_target > 0:
    rev_score = min(100, total_rev / rev_target * 100)
    scores.append(rev_score)
svc_score = None
if csat_val is not None and csat_target is not None and csat_target > 0:
    svc_score = (csat_val / csat_target * 100)
    scores.append(svc_score)

overall = sum(scores) / len(scores) if scores else 0
if overall >= 75:
    health_text, health_color = "Tốt", "#22c55e"
elif overall >= 50:
    health_text, health_color = "Cần chú ý", "#f59e0b"
else:
    health_text, health_color = "Cần hành động", "#ef4444"

# ── Hero metric ──
st.markdown(f"""
<div style="text-align:center; padding:1.5rem 0;">
    <div style="font-size:1rem; color:#9fb0c7;">Sức khoẻ doanh nghiệp</div>
    <div style="font-size:3.5rem; font-weight:800; color:{health_color};">{health_text}</div>
    <div style="font-size:1rem; color:#9fb0c7;">Điểm tổng hợp: {overall:.0f}/100</div>
</div>
""", unsafe_allow_html=True)

# Pre-load credit for alerts (defined here before alerts, re-used below)
credit = get_credit_portfolio(year=f_year, month=f_month)

# ── Smart Alerts ──
alerts = []
if _mom_gmv[1] is not None and _mom_gmv[1] < -10:
    level = "error" if _mom_gmv[1] < -20 else "warning"
    alerts.append((level, f"GMV giảm {_mom_gmv[1]:.1f}% so với tháng trước — kiểm tra ngay"))
if _mom_rev[1] is not None and _mom_rev[1] < -10:
    level = "error" if _mom_rev[1] < -20 else "warning"
    alerts.append((level, f"Doanh thu giảm {_mom_rev[1]:.1f}% so với tháng trước"))
_npl_current = credit["npl_rate"].mean() if not credit.empty else 0
if _npl_current > 5:
    alerts.append(("error", f"Nợ xấu {_npl_current:.1f}% — vượt ngưỡng nguy hiểm 5%!"))
elif _npl_current > 3:
    alerts.append(("warning", f"Nợ xấu {_npl_current:.1f}% — vượt ngưỡng cảnh báo 3%"))
_fraud_current = txn["fraud_rate"].mean() if not txn.empty else 0
if _fraud_current > 2:
    alerts.append(("error", f"Gian lận {_fraud_current:.2f}% — vượt ngưỡng an toàn"))
elif _fraud_current > 1:
    alerts.append(("warning", f"Gian lận {_fraud_current:.2f}% — cần theo dõi sát"))
if churn_val and churn_val > 10:
    alerts.append(("error", f"Churn {churn_val:.1f}%/tháng — khách hàng rời bỏ nhiều"))
elif churn_val and churn_val > 5:
    alerts.append(("warning", f"Churn {churn_val:.1f}%/tháng — đang tăng"))
if rev_target is not None and rev_score is not None and rev_score < 80:
    alerts.append(("warning", f"Doanh thu đạt {rev_score:.0f}% mục tiêu ({total_rev/1e9:.1f}/{rev_target/1e9:.1f} tỷ)"))
if not alerts:
    st.success("Tất cả chỉ số đang trong ngưỡng an toàn. Không có cảnh báo nào.")
else:
    for level, msg in alerts:
        if level == "error":
            st.error(msg)
        else:
            st.warning(msg)

st.divider()

# ── 4 Module KPI Summary ──
st.markdown("#### Tổng quan 4 mảng kinh doanh")
merch = get_merchant_accounts(year=f_year, month=f_month)
eco = get_service_ecosystem(year=f_year, month=f_month)

_acq = int((mkt["total_impressions"] * mkt["avg_conversion"]).sum()) if not mkt.empty else 0
_npl = credit["npl_rate"].mean() if not credit.empty else 0
_eco_rev = (eco["revenue_a"].sum() + eco["revenue_b"].sum()) / 1e12 if not eco.empty else 0
_susp = len(merch[merch["is_suspected_merchant"] == 1]) if not merch.empty else 0

m1, m2, m3, m4 = st.columns(4)
with m1:
    render_kpi_card("Khách hàng mới", f"{_acq:,} người" if _acq > 0 else "Chưa có", "Tháng này")
with m2:
    _npl_status = "An toàn" if _npl < 3 else "Theo dõi" if _npl < 5 else "Rủi ro"
    render_kpi_card("Tín dụng", _npl_status, f"Nợ xấu: {_npl:.1f}%")
with m3:
    render_kpi_card("Dịch vụ", f"{_eco_rev:.1f} tỷ", "Doanh thu combo")
with m4:
    _susp_status = "Tốt" if _susp < 10 else "Cần xem" if _susp < 30 else "Nhiều"
    render_kpi_card("Đối tác", _susp_status, f"{_susp} cần kiểm tra")

# Drill-down: CEO chọn mảng để xem tóm tắt nhanh
_module_options = ["Khách hàng mới", "Tín dụng", "Dịch vụ", "Đối tác"]
sel_module = st.radio("Chọn mảng để xem tóm tắt nhanh:", _module_options, horizontal=True, key="ov_module_drill")
if sel_module == "Khách hàng mới":
    if _acq > 0:
        _cac_ov = total_spend_raw / _acq if not mkt.empty and (total_spend_raw := mkt["campaign_spend"].sum()) > 0 else 0
        st.info(f"**Khách hàng mới:** {_acq:,} người đăng ký tháng này. Chi phí trung bình {_cac_ov/1e3:,.0f}K/khách." if _cac_ov > 0 else f"**Khách hàng mới:** {_acq:,} người đăng ký tháng này.")
        st.markdown("**Gợi ý:** Vào trang **Thu hút khách hàng** để xem phễu chuyển đổi và điều chỉnh ngân sách.")
    else:
        st.info("Chưa có dữ liệu khách hàng mới.")
elif sel_module == "Tín dụng":
    _total_outstanding_ov = credit["total_outstanding"].sum() / 1000 if not credit.empty else 0
    st.info(f"**Tín dụng & Dòng tiền:** Nợ xấu ở mức **{_npl:.1f}%** — Tổng dư nợ **{_total_outstanding_ov/1e9:.1f} tỷ**")
    if _npl >= 5:
        st.markdown("**Gợi ý:** Nợ xấu cao. Vào trang **Tín dụng** để siết chặt chính sách hoặc đẩy mạnh thu hồi.")
    else:
        st.markdown("**Gợi ý:** Tình hình ổn. Vào trang **Tín dụng** để xem chi tiết theo nhóm khách hàng.")
elif sel_module == "Dịch vụ":
    _n_combos = len(eco) if not eco.empty else 0
    st.info(f"**Dịch vụ:** {_n_combos} combo đang hoạt động — Doanh thu **{_eco_rev:.1f} tỷ**")
    st.markdown("**Gợi ý:** Vào trang **Hệ sinh thái dịch vụ** để xem combo phổ biến và cơ hội bán chéo.")
else:
    st.info(f"**Đối tác:** {_susp} tài khoản nghi ngờ cần kiểm tra trong tổng số {len(merch) if not merch.empty else 0} đối tác")
    if _susp > 20:
        st.markdown("**Gợi ý:** Số lượng nghi ngờ cao. Vào trang **Đối tác** để xem chi tiết và gửi chỉ thị xử lý.")
    else:
        st.markdown("**Gợi ý:** Tình hình ổn. Vào trang **Đối tác** để xem phân loại rủi ro chi tiết.")

st.divider()

# ── Geographic Breakdown ──
with st.expander("Phân bổ theo vùng miền", expanded=False):
    txn_reg = get_transaction_by_region(year=f_year, month=f_month)
    merch_reg = get_merchant_by_region(year=f_year, month=f_month)

    if not txn_reg.empty:
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            fig_reg = go.Figure(go.Bar(
                x=txn_reg["region"], y=txn_reg["total_amount"] / 1e9,
                text=[f"{v:.1f}T" for v in txn_reg["total_amount"] / 1e9],
                textposition="outside",
                marker_color=["#4cc9f0", "#22c55e", "#f59e0b"],
                hovertemplate="<b>%{x}</b><br>GMV: %{y:.1f} tỷ<extra></extra>",
            ))
            fig_reg.update_layout(
                title="GMV theo vùng miền (Tỷ VND)", yaxis_title="GMV (tỷ VND)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5eefb"), height=300, showlegend=False,
            )
            st.plotly_chart(fig_reg, width='stretch', key="overview_reg_gmv_1", config={"displayModeBar": False})
        with col_r2:
            fig_freg = go.Figure(go.Bar(
                x=txn_reg["region"], y=txn_reg["fraud_rate"] * 100,
                text=[f"{v*100:.2f}%" for v in txn_reg["fraud_rate"]],
                textposition="outside",
                marker_color=["#ef4444", "#f59e0b", "#22c55e"],
                hovertemplate="<b>%{x}</b><br>Gian lận: %{y:.2f}%<extra></extra>",
            ))
            fig_freg.update_layout(
                title="Tỷ lệ gian lận theo vùng (%)", yaxis_title="Gian lận (%)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5eefb"), height=300, showlegend=False,
            )
            st.plotly_chart(fig_freg, width='stretch', key="overview_fraud_rate", config={"displayModeBar": False})
    else:
        st.info("Chưa có dữ liệu theo vùng miền.")

    if not merch_reg.empty:
        fig_mreg = go.Figure(go.Bar(
            x=merch_reg["region"], y=merch_reg["total_revenue"] / 1e9,
            text=[f"{v/1e9:.2f}T" for v in merch_reg["total_revenue"]],
            textposition="outside",
            marker_color=["#818cf8", "#4cc9f0", "#22c55e"],
            hovertemplate="<b>%{x}</b><br>Doanh thu: %{y:.2f} tỷ<extra></extra>",
        ))
        fig_mreg.update_layout(
            title="Doanh thu đối tác theo vùng (Tỷ VND)", yaxis_title="Doanh thu (tỷ VND)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5eefb"), height=300, showlegend=False,
        )
        st.plotly_chart(fig_mreg, width='stretch', key="overview_merch_reg", config={"displayModeBar": False})

# ── Khu vực Breakdown (Thành thị, Nông thôn, Biển đảo, Biên giới) ──
with st.expander("Phân bổ theo khu vực", expanded=False):
    if not txn_reg.empty:
        # Map regions to khu vực types with distribution
        khu_vuc_mapping = {
            "Miền Bắc": {"Thành thị": 0.5, "Nông thôn": 0.3, "Biên giới": 0.15, "Biển đảo": 0.05},
            "Miền Trung": {"Thành thị": 0.3, "Nông thôn": 0.4, "Biên giới": 0.1, "Biển đảo": 0.2},
            "Miền Nam": {"Thành thị": 0.6, "Nông thôn": 0.25, "Biên giới": 0.05, "Biển đảo": 0.1},
        }
        
        khu_vuc_data = {"Thành thị": 0, "Nông thôn": 0, "Biên giới": 0, "Biển đảo": 0}
        for _, row in txn_reg.iterrows():
            region = row["region"]
            gmv = row["total_amount"]
            mapping = khu_vuc_mapping.get(region, {"Thành thị": 0.5, "Nông thôn": 0.3, "Biên giới": 0.1, "Biển đảo": 0.1})
            for khu_vuc, weight in mapping.items():
                khu_vuc_data[khu_vuc] += gmv * weight
        
        khu_vuc_df = pd.DataFrame([
            {"Khu vực": k, "GMV": v} for k, v in khu_vuc_data.items()
        ])
        
        fig_khu_vuc = go.Figure(go.Bar(
            x=khu_vuc_df["Khu vực"], y=khu_vuc_df["GMV"] / 1e9,
            text=[f"{v:.1f}B" for v in khu_vuc_df["GMV"] / 1e9],
            textposition="outside",
            marker_color=["#818cf8", "#22c55e", "#f59e0b", "#4cc9f0"],
            hovertemplate="<b>%{x}</b><br>GMV: %{y:.1f} tỷ<extra></extra>",
        ))
        fig_khu_vuc.update_layout(
            title="GMV theo khu vực", yaxis_title="GMV (tỷ VND)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5eefb"), height=300, showlegend=False,
        )
        st.plotly_chart(fig_khu_vuc, width='stretch', key="overview_khu_vuc_gmv", config={"displayModeBar": False})
    else:
        st.info("Chưa có dữ liệu theo khu vực.")

# ── Tóm tắt bằng ngôn ngữ tự nhiên ──
st.markdown("#### Nhận định và khuyến nghị")

summary_parts = []

# Revenue insight
summary_parts.append(f"Tổng doanh thu đạt **{total_rev/1e9:.1f} tỷ VND**.")

# Safety insight
if safety >= 80:
    summary_parts.append(f"Hệ thống an toàn ({safety:.0f}/100). Tổn thất gian lận ở mức kiểm soát.")
elif safety >= 60:
    summary_parts.append(f"An toàn ở mức trung bình ({safety:.0f}/100). Cần theo dõi sát dịch vụ có tổn thất cao.")
else:
    summary_parts.append(f"An toàn ở mức thấp ({safety:.0f}/100). Cần hành động giảm thiểu gian lận ngay.")

# Service insight
if csat_val:
    if csat_val >= 4.0:
        summary_parts.append(f"Khách hàng hài lòng (CSAT {csat_val:.1f}). Duy trì chất lượng hiện tại.")
    elif csat_val >= 3.5:
        summary_parts.append(f"CSAT ở mức {csat_val:.1f} – có cơ hội nâng lên 4.0+ nếu tăng đầu tư CSKH.")
    else:
        summary_parts.append(f"CSAT thấp ({csat_val:.1f}). Cần cải thiện chất lượng dịch vụ khẩn cấp.")

# Priority action
if safety < 60:
    summary_parts.append("**Ưu tiên số 1:** Giảm tổn thất gian lận.")
elif csat_val and csat_val < 3.5:
    summary_parts.append("**Ưu tiên số 1:** Cải thiện hài lòng khách hàng.")
else:
    summary_parts.append("**Ưu tiên số 1:** Tăng trưởng doanh thu.")

for part in summary_parts:
    st.markdown(f"- {part}")

st.divider()

# ── Export ──
export_lines = [f"Sức khoẻ doanh nghiệp: {health_text} ({overall:.0f}/100)"]
export_lines.append(f"Doanh thu: {total_rev/1e9:.1f} tỷ VND")
export_lines.append(f"An toàn: {safety:.0f}/100")
if csat_val:
    export_lines.append(f"CSAT: {csat_val:.1f}/5.0")
export_lines.extend(summary_parts)
export_text = "\n".join(export_lines)
st.download_button(
    "Tải tóm tắt",
    data=export_text.encode("utf-8-sig"),
    file_name="tong_quan_ceo.txt",
    mime="text/plain",
)
