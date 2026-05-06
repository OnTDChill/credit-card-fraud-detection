"""DSS Scenario Engine for CEO What-If simulations."""

from __future__ import annotations

import importlib.util
import os
from typing import Any

import pandas as pd

# Load root config by absolute path to avoid circular import
_root_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.py")
_spec = importlib.util.spec_from_file_location("_root_config_engine", _root_config_path)
_rcfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rcfg)
FEE_RATES = _rcfg.FEE_RATES
TRANSACTION_TYPE_NAMES = _rcfg.TRANSACTION_TYPE_NAMES
DEFAULT_TRANSACTION_LIMIT = _rcfg.DEFAULT_TRANSACTION_LIMIT
DEFAULT_CHANNEL_BUDGET = _rcfg.DEFAULT_CHANNEL_BUDGET
from .dss_data_access import get_transaction_summary, get_marketing_summary, get_cskh_summary


def simulate_fee_change(
    transaction_type: str,
    new_rate: float,
    current_data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Simulate revenue impact of changing fee rate for a transaction type."""
    if current_data is None:
        current_data = get_transaction_summary()

    if current_data.empty:
        return {
            "original_revenue": 0,
            "new_revenue": 0,
            "difference": 0,
            "percent_change": 0,
            "explanation": "Chưa có dữ liệu để mô phỏng.",
        }

    # Filter for the specific transaction type
    type_data = current_data[current_data["transaction_type"] == transaction_type]
    if type_data.empty:
        return {
            "original_revenue": 0,
            "new_revenue": 0,
            "difference": 0,
            "percent_change": 0,
            "explanation": f"Không có dữ liệu cho loại {TRANSACTION_TYPE_NAMES.get(transaction_type, transaction_type)}.",
        }

    total_amount = type_data["total_amount"].sum()
    original_rate = FEE_RATES.get(transaction_type, 0.015)
    original_revenue = total_amount * original_rate
    new_revenue = total_amount * new_rate
    difference = new_revenue - original_revenue
    percent_change = (difference / original_revenue * 100) if original_revenue > 0 else 0

    # Estimate volume impact (simplified elasticity model)
    # Assume 10% fee increase → 5% volume decrease (elasticity = -0.5)
    elasticity = -0.5
    rate_change_pct = (new_rate - original_rate) / original_rate if original_rate > 0 else 0
    volume_change_pct = rate_change_pct * elasticity
    adjusted_new_revenue = new_revenue * (1 + volume_change_pct)
    adjusted_difference = adjusted_new_revenue - original_revenue

    type_name = TRANSACTION_TYPE_NAMES.get(transaction_type, transaction_type)

    explanation = (
        f"Nếu tăng phí {type_name} từ {original_rate*100:.1f}% lên {new_rate*100:.1f}%:\n"
        f"- Doanh thu trước điều chỉnh: {original_revenue:,.0f} VND\n"
        f"- Doanh thu sau điều chỉnh: {adjusted_new_revenue:,.0f} VND\n"
        f"- Thay đổi: {adjusted_difference:+,.0f} VND ({percent_change*volume_change_pct:+.1f}%)\n"
    )
    if volume_change_pct < -0.01:
        explanation += f"- Lưu ý: Số lượng giao dịch có thể giảm {abs(volume_change_pct)*100:.0f}%"

    return {
        "original_revenue": original_revenue,
        "new_revenue": adjusted_new_revenue,
        "difference": adjusted_difference,
        "percent_change": percent_change * volume_change_pct,
        "volume_impact_pct": volume_change_pct,
        "explanation": explanation,
    }


def simulate_limit_change(
    transaction_type: str,
    new_limit: float,
    current_data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Simulate fraud loss reduction by setting transaction limits."""
    if current_data is None:
        current_data = get_transaction_summary()

    if current_data.empty:
        return {
            "original_loss": 0,
            "new_loss": 0,
            "loss_reduction": 0,
            "percent_reduction": 0,
            "explanation": "Chưa có dữ liệu để mô phỏng.",
    }

    # Estimate based on historical fraud distribution
    # Assume fraud amount follows Pareto: 80% of fraud from 20% of large transactions
    type_data = current_data[current_data["transaction_type"] == transaction_type]
    if type_data.empty:
        return {
            "original_loss": 0,
            "new_loss": 0,
            "loss_reduction": 0,
            "percent_reduction": 0,
            "explanation": f"Không có dữ liệu cho loại {TRANSACTION_TYPE_NAMES.get(transaction_type, transaction_type)}.",
        }

    original_loss = type_data["fraud_amount"].sum()
    original_count = type_data["fraud_count"].sum()

    # Pareto estimate: assume larger transactions have higher fraud rate
    # Limiting to X blocks Y% of fraud based on amount distribution
    default_limit = DEFAULT_TRANSACTION_LIMIT.get(transaction_type, 100_000_000)
    if new_limit < default_limit:
        # More restrictive → more fraud prevented
        # Rough estimate: each 10% reduction in limit → 15% reduction in fraud loss
        reduction_ratio = (default_limit - new_limit) / default_limit
        loss_reduction_pct = min(0.8, reduction_ratio * 1.5)  # Cap at 80%
    else:
        # Less restrictive → less fraud prevented (inverse)
        increase_ratio = (new_limit - default_limit) / default_limit
        loss_reduction_pct = -min(0.3, increase_ratio * 0.3)  # Cap increase at 30%

    new_loss = original_loss * (1 - loss_reduction_pct)
    loss_reduction = original_loss - new_loss
    percent_reduction = loss_reduction_pct * 100

    # Estimate legitimate transactions affected
    # Assume normal distribution: ~5% of legit transactions > new_limit
    total_txns = type_data["total_count"].sum()
    affected_pct = max(0.02, min(0.15, (new_limit / default_limit - 1) * -0.1 + 0.05))
    affected_count = int(total_txns * affected_pct)

    type_name = TRANSACTION_TYPE_NAMES.get(transaction_type, transaction_type)

    explanation = (
        f"Nếu giới hạn {type_name} ở {new_limit:,.0f} VND:\n"
        f"- Tổn thất hiện tại: {original_loss:,.0f} VND ({original_count} vụ)\n"
        f"- Tổn thất ước tính sau giới hạn: {new_loss:,.0f} VND\n"
        f"- Giảm tổn thất: {loss_reduction:,.0f} VND ({percent_reduction:.0f}%)\n"
        f"- Giao dịch hợp lệ bị ảnh hưởng: ~{affected_count:,} ({affected_pct*100:.1f}%)"
    )

    return {
        "original_loss": original_loss,
        "new_loss": new_loss,
        "loss_reduction": loss_reduction,
        "percent_reduction": percent_reduction,
        "affected_transactions": affected_count,
        "explanation": explanation,
    }


def simulate_budget_realloc(
    channel_weights: dict[str, float],
    total_budget: float | None = None,
    current_data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Simulate CAC and new user acquisition with different budget allocation."""
    if current_data is None:
        current_data = get_marketing_summary()

    if current_data.empty:
        return {
            "projected_cac": 0,
            "projected_new_users": 0,
            "explanation": "Chưa có dữ liệu marketing để mô phỏng.",
        }

    # Calculate weighted average ROI and CAC
    total_weight = sum(channel_weights.values())
    normalized_weights = {k: v / total_weight for k, v in channel_weights.items()}

    # Channel efficiency (ROI) from historical data
    channel_roi = current_data.groupby("channel")["avg_roi"].mean().to_dict()
    channel_cac = current_data.groupby("channel")["avg_cac"].mean().to_dict()

    # Weighted projections
    weighted_roi = sum(
        normalized_weights.get(ch, 0) * channel_roi.get(ch, 2.0)
        for ch in normalized_weights
    )
    weighted_cac = sum(
        normalized_weights.get(ch, 0) * channel_cac.get(ch, 500000)
        for ch in normalized_weights
    )

    # Estimate new users: budget / cac
    if total_budget is None:
        total_budget = current_data["campaign_spend"].sum()

    projected_new_users = int(total_budget / weighted_cac) if weighted_cac > 0 else 0

    # Build channel breakdown
    channel_breakdown = []
    for ch, weight in normalized_weights.items():
        ch_budget = total_budget * weight
        ch_users = int(ch_budget / channel_cac.get(ch, 500000)) if channel_cac.get(ch, 0) > 0 else 0
        channel_breakdown.append(f"- {ch}: {ch_budget:,.0f} VND → ~{ch_users} users (ROI: {channel_roi.get(ch, 2.0):.1f}x)")

    explanation = (
        f"Với phân bổ ngân sách mới (tổng {total_budget:,.0f} VND):\n"
        + "\n".join(channel_breakdown)
        + f"\n\nDự báo tổng:\n"
        f"- Chi phí thu hút/khách: {weighted_cac:,.0f} VND\n"
        f"- Users mới ước tính: {projected_new_users:,}\n"
        f"- Hiệu quả đầu tư TB: {weighted_roi:.1f}x"
    )

    return {
        "projected_cac": weighted_cac,
        "projected_new_users": projected_new_users,
        "weighted_roi": weighted_roi,
        "total_budget": total_budget,
        "channel_breakdown": channel_breakdown,
        "explanation": explanation,
    }


def simulate_cskh_budget_increase(
    increase_pct: float,
    current_data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Simulate CSAT improvement from increased CSKH budget."""
    if current_data is None:
        current_data = get_cskh_summary()

    if current_data.empty:
        return {
            "current_csat": 0,
            "projected_csat": 0,
            "explanation": "Chưa có dữ liệu CSKH để mô phỏng.",
        }

    current_csat = current_data["csat_score"].mean()
    current_nps = current_data["nps_score"].mean()

    # Assume correlation: 10% budget increase → 0.3 point CSAT improvement (diminishing returns)
    # But cap at 5.0
    csat_improvement = min(0.5, increase_pct * 0.03)
    projected_csat = min(5.0, current_csat + csat_improvement)

    # NPS follows similar pattern but amplified
    nps_improvement = csat_improvement * 10
    projected_nps = min(100, current_nps + nps_improvement)

    # Churn rate reduction: 10% budget increase → 15% churn reduction
    current_churn = current_data["churn_rate"].mean()
    churn_reduction = current_churn * (increase_pct * 0.015)
    projected_churn = max(0, current_churn - churn_reduction)

    explanation = (
        f"Nếu tăng ngân sách CSKH {increase_pct:.0f}%:\n"
        f"- CSAT hiện tại: {current_csat:.1f} → Dự báo: {projected_csat:.1f}\n"
        f"- NPS hiện tại: {current_nps:.0f} → Dự báo: {projected_nps:.0f}\n"
        f"- Tỷ lệ mất khách: {current_churn*100:.1f}% → {projected_churn*100:.1f}%\n"
        f"\nLưu ý: Hiệu quả giảm dần khi CSAT đã cao (>4.5)."
    )

    return {
        "current_csat": current_csat,
        "projected_csat": projected_csat,
        "current_nps": current_nps,
        "projected_nps": projected_nps,
        "current_churn": current_churn,
        "projected_churn": projected_churn,
        "explanation": explanation,
    }


def calculate_gap_analysis(
    targets: dict[str, float],
    actuals: dict[str, float],
) -> dict[str, Any]:
    """Calculate gap between targets and actuals for all 4 pillars."""
    gaps = {}

    for domain in ["revenue", "risk", "service", "growth"]:
        target = targets.get(domain, 0)
        actual = actuals.get(domain, 0)

        if domain == "risk":
            # For risk, lower is better (reverse the gap)
            gap = target - actual  # Positive = good (actual < target)
            gap_pct = (gap / target * 100) if target != 0 else 0
            status = "good" if gap > 0 else "warning" if gap > -0.01 else "danger"
        else:
            gap = actual - target
            gap_pct = (gap / target * 100) if target != 0 else 0
            status = "good" if gap >= 0 else "warning" if gap_pct > -10 else "danger"

        gaps[domain] = {
            "target": target,
            "actual": actual,
            "gap": gap,
            "gap_pct": gap_pct,
            "status": status,
        }

    # Overall assessment
    good_count = sum(1 for g in gaps.values() if g["status"] == "good")
    total = len(gaps)

    if good_count == total:
        overall_status = "excellent"
        overall_message = "Tất cả chỉ tiêu đều đạt. Công ty vận hành xuất sắc!"
    elif good_count >= total * 0.75:
        overall_status = "good"
        overall_message = f"{good_count}/{total} chỉ tiêu đạt. Cần cải thiện {total - good_count} mục."
    elif good_count >= total * 0.5:
        overall_status = "warning"
        overall_message = f"{good_count}/{total} chỉ tiêu đạt. Cần hành động khẩn cấp."
    else:
        overall_status = "danger"
        overall_message = f"Chỉ {good_count}/{total} chỉ tiêu đạt. Nguy cơ lớn cho công ty!"

    return {
        "gaps": gaps,
        "overall_status": overall_status,
        "overall_message": overall_message,
        "good_count": good_count,
        "total": total,
    }


def generate_recommendation(
    data: pd.DataFrame,
    domain: str,
) -> list[str]:
    """Generate Vietnamese natural language recommendations for CEO."""
    recommendations = []

    if domain == "revenue":
        if data.empty:
            recommendations.append("Chưa có dữ liệu doanh thu. Vui lòng chạy ETL để nhập dữ liệu.")
            return recommendations

        top_type = data.loc[data["revenue_estimated"].idxmax(), "transaction_type"]
        bottom_type = data.loc[data["revenue_estimated"].idxmin(), "transaction_type"]
        growth_types = data[data["total_count"].pct_change() > 0.1]["transaction_type"].tolist()

        recommendations.append(
            f"Dịch vụ {TRANSACTION_TYPE_NAMES.get(top_type, top_type)} đang mang lại doanh thu cao nhất."
        )
        if growth_types:
            growth_names = [TRANSACTION_TYPE_NAMES.get(t, t) for t in growth_types]
            recommendations.append(
                f"Dịch vụ {', '.join(growth_names)} đang tăng trưởng tốt. Cân nhắc đầu tư thêm."
            )
        recommendations.append(
            f"Dịch vụ {TRANSACTION_TYPE_NAMES.get(bottom_type, bottom_type)} có doanh thu thấp nhất. Đánh giá lại chiến lược."
        )

    elif domain == "risk":
        if data.empty:
            recommendations.append("Chưa có dữ liệu rủi ro. Vui lòng chạy ETL để nhập dữ liệu.")
            return recommendations

        if "fraud_rate" in data.columns:
            high_risk = data[data["fraud_rate"] > 0.03]
        else:
            _fr = data["fraud_count"] / data["total_count"]
            high_risk = data[_fr > 0.03]
        if not high_risk.empty:
            risk_names = [TRANSACTION_TYPE_NAMES.get(t, t) for t in high_risk["transaction_type"].tolist()]
            recommendations.append(
                f"Cảnh báo: {', '.join(risk_names)} có tỷ lệ rủi ro cao (>3%). Cần tăng kiểm soát."
            )

        loss_trend = data.groupby("month")["fraud_amount"].sum().pct_change().iloc[-1]
        if loss_trend > 0.1:
            recommendations.append(
                "Tổn thất đang tăng nhanh hơn 10%/tháng. Đề xuất họp khẩn với bộ phận An toàn."
            )

        recommendations.append(
            "Đề xuất: Giới hạn mức giao dịch cho các loại có rủi ro cao."
        )

    elif domain == "service":
        if data.empty:
            recommendations.append("Chưa có dữ liệu dịch vụ. Vui lòng chạy ETL để nhập dữ liệu.")
            return recommendations

        avg_csat = data["csat_score"].mean()
        if avg_csat < 3.5:
            recommendations.append(
                f"CSAT trung bình {avg_csat:.1f}/5.0 thấp. Khách hàng không hài lòng. Cần tăng ngân sách CSKH."
            )
        elif avg_csat >= 4.5:
            recommendations.append(
                f"CSAT xuất sắc ({avg_csat:.1f}/5.0). Có thể chuyển một phần ngân sách CSKH sang Marketing."
            )

        churn = data["churn_rate"].mean()
        if churn > 0.05:
            recommendations.append(
                f"Tỷ lệ mất khách cao ({churn*100:.1f}%). Đề xuất chương trình giữ chân khách hàng."
            )

    elif domain == "overview":
        recommendations.append("Xem báo cáo chi tiết ở các tab chuyên biệt.")
        recommendations.append("Thiết lập mục tiêu năm để so sánh với thực tế.")

    return recommendations


# =============================================
# MODULE 1: Acquisition & Activation What-If
# =============================================

def simulate_acquisition_funnel(
    budget_weights: dict[str, int],
    total_budget: float,
    current_data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Simulate user acquisition funnel based on marketing budget reallocation."""
    if current_data is None:
        current_data = get_marketing_summary()

    if current_data.empty or total_budget <= 0:
        return {
            "acquisition": 0, "kyc_rate": 0, "activation_rate": 0,
            "cac": 0, "projected_cac": 0, "explanation": "Chưa có dữ liệu.",
        }

    channels = current_data["channel"].unique().tolist()
    total_weight = sum(budget_weights.values())
    if total_weight <= 0:
        return {"acquisition": 0, "kyc_rate": 0, "activation_rate": 0,
                "cac": 0, "projected_cac": 0, "explanation": "Chưa phân bổ ngân sách."}

    # Weighted CAC and conversion by new budget allocation
    total_new_users = 0
    for ch in channels:
        ch_data = current_data[current_data["channel"] == ch]
        if ch_data.empty:
            continue
        weight = budget_weights.get(ch, 0) / total_weight
        budget = total_budget * weight
        avg_cac = ch_data["avg_cac"].mean()
        conversion = ch_data["avg_conversion"].mean()
        if avg_cac > 0:
            new_users = budget / avg_cac
            total_new_users += new_users

    # Funnel stages (simplified from marketing → app install → KYC → deposit → first txn)
    base_acquisition = int(total_new_users)
    # KYC rate improves with In-App Game budget (vs Facebook Ads)
    inapp_weight = budget_weights.get("In-App Game", 0) / total_weight if "In-App Game" in budget_weights else 0
    kyc_rate = 0.50 + inapp_weight * 0.20  # 50% → 70% as inapp increases
    activation_rate = kyc_rate * 0.75  # 75% of KYC users activate
    projected_cac = total_budget / max(base_acquisition, 1)

    explanation = (
        f"Phân bổ ngân sách: {budget_weights}\n"
        f"- Users mới dự báo: {base_acquisition:,}\n"
        f"- Tỷ lệ KYC thành công: {kyc_rate*100:.0f}%\n"
        f"- Tỷ lệ Activation: {activation_rate*100:.0f}%\n"
        f"- CAC dự báo: {projected_cac/1e3:,.0f}K VND\n"
    )

    return {
        "acquisition": base_acquisition,
        "kyc_rate": kyc_rate,
        "activation_rate": activation_rate,
        "cac": projected_cac,
        "projected_cac": projected_cac,
        "funnel_stages": [base_acquisition, int(base_acquisition * kyc_rate),
                          int(base_acquisition * activation_rate)],
        "explanation": explanation,
    }


# =============================================
# MODULE 2: Credit Portfolio What-If
# =============================================

def simulate_credit_policy(
    segment: str,
    new_limit: float,
    new_interest_rate: float,
    current_data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Simulate NPL and profit impact of credit policy changes."""
    if current_data is None:
        from .dss_data_access import get_credit_portfolio
        current_data = get_credit_portfolio()

    if current_data.empty:
        return {
            "npl_before": 0, "npl_after": 0, "profit_before": 0, "profit_after": 0,
            "net_margin_before": 0, "net_margin_after": 0, "explanation": "Chưa có dữ liệu.",
        }

    seg_data = current_data[current_data["segment"] == segment]
    if seg_data.empty:
        return {
            "npl_before": 0, "npl_after": 0, "profit_before": 0, "profit_after": 0,
            "net_margin_before": 0, "net_margin_after": 0,
            "explanation": f"Không có dữ liệu phân khúc {segment}.",
        }

    row = seg_data.iloc[0]
    old_limit = float(row["credit_limit"])
    old_rate = float(row["interest_rate"])
    old_npl = float(row["npl_rate"])
    outstanding = float(row["total_outstanding"])
    revenue_interest = float(row["revenue_interest"])

    # Limit impact: lower limit → lower NPL (less exposure)
    limit_change = (new_limit - old_limit) / old_limit if old_limit > 0 else 0
    npl_change = -limit_change * 15  # Each 10% limit reduction → 1.5% NPL reduction
    new_npl = max(0.1, old_npl + npl_change)

    # Interest impact: higher rate → higher profit but slightly higher NPL
    rate_change = new_interest_rate - old_rate
    profit_change_pct = (new_interest_rate / old_rate - 1) * 100 if old_rate > 0 else 0
    npl_from_rate = rate_change * 0.1  # each +1% rate → +0.1% NPL
    new_npl += npl_from_rate

    # Revenue
    new_revenue = revenue_interest * (1 + profit_change_pct / 100)
    # Loss provision (NPL * outstanding)
    old_loss = outstanding * (old_npl / 100)
    new_loss = outstanding * (new_npl / 100)
    net_margin_before = revenue_interest - old_loss
    net_margin_after = new_revenue - new_loss

    explanation = (
        f"Phân khúc {segment}:\n"
        f"- NPL: {old_npl:.1f}% → {new_npl:.1f}%\n"
        f"- Lợi nhuận lãi: {revenue_interest/1e6:.0f}M → {new_revenue/1e6:.0f}M VND\n"
        f"- Dự phòng rủi ro: {old_loss/1e6:.0f}M → {new_loss/1e6:.0f}M VND\n"
        f"- Biên lợi nhuận ròng: {net_margin_before/1e6:.0f}M → {net_margin_after/1e6:.0f}M VND\n"
    )

    return {
        "npl_before": old_npl,
        "npl_after": new_npl,
        "profit_before": revenue_interest,
        "profit_after": new_revenue,
        "net_margin_before": net_margin_before,
        "net_margin_after": net_margin_after,
        "explanation": explanation,
    }


# =============================================
# MODULE 3: Cross-sell Combo What-If
# =============================================

def simulate_combo_discount(
    service_pair: tuple[str, str],
    discount_pct: float,
    current_data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Simulate revenue impact of cross-sell combo discounts."""
    if current_data is None:
        from .dss_data_access import get_service_ecosystem
        current_data = get_service_ecosystem()

    if current_data.empty:
        return {
            "revenue_a_before": 0, "revenue_b_before": 0,
            "revenue_a_after": 0, "revenue_b_after": 0,
            "total_uplift": 0, "roi": 0, "explanation": "Chưa có dữ liệu.",
        }

    sa, sb = service_pair
    pair_data = current_data[
        (current_data["service_a"] == sa) & (current_data["service_b"] == sb)
    ]
    if pair_data.empty:
        pair_data = current_data[
            (current_data["service_a"] == sb) & (current_data["service_b"] == sa)
        ]
    if pair_data.empty:
        return {
            "revenue_a_before": 0, "revenue_b_before": 0,
            "revenue_a_after": 0, "revenue_b_after": 0,
            "total_uplift": 0, "roi": 0,
            "explanation": f"Không có dữ liệu cặp {sa} + {sb}.",
        }

    row = pair_data.iloc[0]
    rev_a = float(row["revenue_a"])
    rev_b = float(row["revenue_b"])
    lift = float(row["lift"])
    conf = float(row["confidence"])
    margin_a = float(row["profit_margin_a"])
    margin_b = float(row["profit_margin_b"])

    # Discount stimulates cross-sell: each 1% discount → confidence increase by 0.3%
    conf_boost = conf * (1 + discount_pct * 0.03)
    # Revenue uplift for B driven by cross-sell
    uplift_b = rev_b * (lift - 1) * (discount_pct / 10) * conf_boost
    # Revenue loss from discount on A
    loss_a = rev_a * (discount_pct / 100)

    new_rev_a = rev_a - loss_a
    new_rev_b = rev_b + uplift_b
    total_uplift = (new_rev_a + new_rev_b) - (rev_a + rev_b)

    # Net profit change (considering margins)
    profit_before = rev_a * margin_a + rev_b * margin_b
    profit_after = new_rev_a * margin_a + new_rev_b * margin_b
    roi = (profit_after - profit_before) / max(loss_a * margin_a, 1)

    explanation = (
        f"Combo {sa} + {sb} giảm {discount_pct:.0f}%:\n"
        f"- Doanh thu {sa}: {rev_a/1e9:.1f}B → {new_rev_a/1e9:.1f}B VND\n"
        f"- Doanh thu {sb}: {rev_b/1e9:.1f}B → {new_rev_b/1e9:.1f}B VND\n"
        f"- Tổng tăng trưởng: {total_uplift/1e9:+.1f}B VND\n"
        f"- ROI: {roi:.1f}x\n"
    )

    return {
        "revenue_a_before": rev_a,
        "revenue_b_before": rev_b,
        "revenue_a_after": new_rev_a,
        "revenue_b_after": new_rev_b,
        "total_uplift": total_uplift,
        "roi": roi,
        "explanation": explanation,
    }


# =============================================
# MODULE 4: Merchant Conversion What-If
# =============================================

def simulate_merchant_conversion(
    conversion_rate: float,
    current_data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Simulate revenue/tax impact of converting suspected merchants."""
    if current_data is None:
        from .dss_data_access import get_merchant_accounts
        current_data = get_merchant_accounts()

    if current_data.empty:
        return {
            "suspected_count": 0, "converted_count": 0,
            "extra_fee_revenue": 0, "extra_tax_collectable": 0,
            "explanation": "Chưa có dữ liệu.",
        }

    suspected = current_data[current_data["is_suspected_merchant"] == 1]
    if suspected.empty:
        return {
            "suspected_count": 0, "converted_count": 0,
            "extra_fee_revenue": 0, "extra_tax_collectable": 0,
            "explanation": "Không phát hiện tài khoản nghi ngờ.",
        }

    n_suspected = len(suspected)
    n_converted = int(n_suspected * conversion_rate)

    # Merchant conversion financial impact calculation
    # Source: Industry benchmarks for Vietnamese fintech
    # - Merchant transaction fees: 1.5-2.5% vs personal: 0.5-1% (MoMo/VietQR rates)
    # - VAT: 10% standard rate per Vietnam tax law
    avg_monthly_volume = suspected["monthly_volume"].mean()
    extra_fee_pct = 0.02  # 2% merchant fee premium (industry avg: 1.5-2.5%)
    extra_fee_revenue = n_converted * avg_monthly_volume * extra_fee_pct * 12  # annual
    # Tax calculation from seed data (est_tax_collectable already includes 10% VAT assumption)
    extra_tax = n_converted * suspected["est_tax_collectable"].sum() * 12

    explanation = (
        f"Chuyển đổi {conversion_rate*100:.0f}% ({n_converted:,}/{n_suspected:,}) tài khoản nghi ngờ:\n"
        f"- Thu thêm phí giao dịch/năm: +{extra_fee_revenue/1e9:.1f} tỷ VND\n"
        f"- Thu thêm thuế thu hộ/năm: +{extra_tax/1e9:.1f} tỷ VND\n"
        f"- Tổng tác động tài chính: +{(extra_fee_revenue + extra_tax)/1e9:.1f} tỷ VND\n"
    )

    return {
        "suspected_count": n_suspected,
        "converted_count": n_converted,
        "extra_fee_revenue": extra_fee_revenue,
        "extra_tax_collectable": extra_tax,
        "explanation": explanation,
    }
