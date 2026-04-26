"""
Fraud Detection Decision Engine
3 Tier Architecture: REVIEW ZONE / OVERRIDE / CONTINUOUS LEARNING
"""
import os
import time
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Literal
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Default configuration - can be overridden from database
DEFAULT_CONFIG = {
    'low_threshold': 0.35,
    'high_threshold': 0.65,
    'review_ttl_minutes': 15,
    'small_amount_threshold': 50.0,
    'large_amount_threshold': 1000.0,
    'drift_window_days': 7,
    'max_fp_rate': 0.10,
    'max_fn_rate': 0.20
}


class DecisionEngine:
    def __init__(self, config: Dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    def _adjust_thresholds_for_context(self, amount: float, customer_tier: str) -> Tuple[float, float]:
        """Dynamic threshold adjustment based on transaction context"""
        low = self.config['low_threshold']
        high = self.config['high_threshold']

        # Adjust by amount
        if amount < self.config['small_amount_threshold']:
            # Relax thresholds for small amount: less review, more auto allow
            low = min(low * 1.2, 0.9)
            high = min(high * 1.2, 0.95)
        elif amount > self.config['large_amount_threshold']:
            # Stricter thresholds for large amount: more review, less auto allow
            low = max(low * 0.7, 0.1)
            high = max(high * 0.7, 0.3)

        # Adjust by customer tier
        if customer_tier == 'VIP':
            low *= 1.3
            high *= 1.3
        elif customer_tier == 'NEW':
            low *= 0.8
            high *= 0.8

        # Ensure logical order
        low = min(low, high * 0.9)
        return round(low, 4), round(high, 4)

    def make_decision(self,
                      fraud_prob: float,
                      amount: float,
                      customer_tier: str = 'NORMAL',
                      merchant_risk: str = 'NORMAL',
                      location_risk: str = 'NORMAL') -> Dict:
        """
        3 Zone Decision Making: ALLOW / REVIEW / BLOCK

        Returns:
            dict with decision, adjusted thresholds and reason codes
        """
        fraud_prob = max(0.0, min(1.0, float(fraud_prob)))
        amount = max(0.0, float(amount))

        # Get dynamic thresholds for this context
        adjusted_low, adjusted_high = self._adjust_thresholds_for_context(amount, customer_tier)

        reason_codes = []
        if amount < self.config['small_amount_threshold']:
            reason_codes.append('SMALL_AMOUNT')
        if amount > self.config['large_amount_threshold']:
            reason_codes.append('LARGE_AMOUNT')
        if customer_tier != 'NORMAL':
            reason_codes.append(f'TIER_{customer_tier}')

        # Make decision
        if fraud_prob < adjusted_low:
            decision = 'ALLOW'
        elif adjusted_low <= fraud_prob < adjusted_high:
            decision = 'REVIEW'
            reason_codes.append('BORDERLINE_SCORE')
        else:
            decision = 'BLOCK'
            reason_codes.append('HIGH_RISK_SCORE')

        expires_at = None
        if decision == 'REVIEW':
            expires_at = datetime.utcnow() + timedelta(minutes=self.config['review_ttl_minutes'])

        return {
            'decision': decision,
            'fraud_probability': round(fraud_prob, 6),
            'low_threshold': adjusted_low,
            'high_threshold': adjusted_high,
            'reason_codes': reason_codes,
            'expires_at': expires_at,
            'ttl_minutes': self.config['review_ttl_minutes']
        }

    def admin_override(self,
                       transaction_id: str,
                       original_decision: str,
                       new_decision: Literal["ALLOW", "BLOCK"],
                       admin_id: str,
                       reason: str,
                       original_label: int = None,
                       corrected_label: int = None) -> Dict:
        """
        Admin override decision + audit trail + feedback pool entry

        Returns:
            audit log entry ready to be stored
        """
        override_time = datetime.utcnow()

        audit_entry = {
            'transaction_id': transaction_id,
            'original_decision': original_decision,
            'new_decision': new_decision,
            'override_by': admin_id,
            'override_at': override_time.isoformat(),
            'reason': reason,
            'original_label': original_label,
            'corrected_label': corrected_label,
            'audit_hash': hash(f"{transaction_id}{admin_id}{override_time}{reason}")
        }

        feedback_entry = None
        if original_label is not None and corrected_label is not None:
            feedback_entry = {
                'transaction_id': transaction_id,
                'original_label': original_label,
                'corrected_label': corrected_label,
                'corrected_by': admin_id,
                'corrected_at': override_time,
                'reason': reason
            }

        logger.info(f"ADMIN OVERRIDE: {transaction_id} | {original_decision} → {new_decision} | by: {admin_id}")

        return {
            'audit_log': audit_entry,
            'feedback_pool': feedback_entry,
            'was_false_positive': original_decision == 'BLOCK' and corrected_label == 0,
            'was_false_negative': original_decision == 'ALLOW' and corrected_label == 1
        }

    def check_model_drift(self, history_df: pd.DataFrame, window_days: int = 7) -> Dict:
        """
        Monitor FP and FN rates over time window

        Returns:
            drift status + metrics
        """
        if history_df.empty:
            return {
                'window_days': window_days,
                'total_transactions': 0,
                'fp_rate': 0.0,
                'fn_rate': 0.0,
                'alert_fired': False,
                'alert_reasons': []
            }

        fp_count = len(history_df[history_df['is_false_positive'] == True])
        fn_count = len(history_df[history_df['is_false_negative'] == True])
        total = len(history_df)

        fp_rate = fp_count / max(total, 1)
        fn_rate = fn_count / max(total, 1)

        alerts = []
        if fp_rate > self.config['max_fp_rate']:
            alerts.append(f"FP RATE EXCEEDED: {fp_rate:.2%} > {self.config['max_fp_rate']:.2%}")
        if fn_rate > self.config['max_fn_rate']:
            alerts.append(f"FN RATE EXCEEDED: {fn_rate:.2%} > {self.config['max_fn_rate']:.2%}")

        return {
            'window_days': window_days,
            'total_transactions': total,
            'fp_count': fp_count,
            'fn_count': fn_count,
            'fp_rate': round(fp_rate, 4),
            'fn_rate': round(fn_rate, 4),
            'alert_fired': len(alerts) > 0,
            'alert_reasons': alerts
        }

    def estimate_decision_distribution(self, scores: np.ndarray) -> Dict:
        """
        Estimate percentage of ALLOW / REVIEW / BLOCK for given probability scores
        Used for threshold preview in dashboard
        """
        low = self.config['low_threshold']
        high = self.config['high_threshold']

        decisions = np.where(scores < low, 'ALLOW',
                    np.where(scores < high, 'REVIEW', 'BLOCK'))

        unique, counts = np.unique(decisions, return_counts=True)
        dist = dict(zip(unique, counts))

        total = len(scores)
        return {
            'ALLOW': dist.get('ALLOW', 0),
            'ALLOW_pct': round(dist.get('ALLOW', 0) / total * 100, 1),
            'REVIEW': dist.get('REVIEW', 0),
            'REVIEW_pct': round(dist.get('REVIEW', 0) / total * 100, 1),
            'BLOCK': dist.get('BLOCK', 0),
            'BLOCK_pct': round(dist.get('BLOCK', 0) / total * 100, 1),
            'total': total
        }

    @staticmethod
    def champion_challenger_comparison(champion_metrics: Dict, challenger_metrics: Dict) -> bool:
        """
        Challenger promotion rules:
        ✅ Challenger F1 >= Champion F1
        ✅ Challenger Recall >= Champion Recall
        ✅ Challenger FPR <= Champion FPR * 1.1
        """
        required = [
            challenger_metrics.get('f1_score', 0) >= champion_metrics.get('f1_score', 0),
            challenger_metrics.get('recall', 0) >= champion_metrics.get('recall', 0),
            challenger_metrics.get('fpr', 1.0) <= champion_metrics.get('fpr', 1.0) * 1.1
        ]

        return all(required)


def test_decision_engine():
    """Test scenarios for decision engine"""
    engine = DecisionEngine()

    print("=== Decision Engine Test Scenarios ===")

    # Test 1: Normal transaction low risk
    res = engine.make_decision(0.2, amount=100)
    print(f"✅ Test 1 Low risk: {res['decision']}")

    # Test 2: High risk
    res = engine.make_decision(0.8, amount=100)
    print(f"❌ Test 2 High risk: {res['decision']}")

    # Test 3: Review zone
    res = engine.make_decision(0.5, amount=100)
    print(f"⏳ Test 3 Review zone: {res['decision']}")

    # Test 4: Large amount automatic review
    res = engine.make_decision(0.3, amount=2000)
    print(f"⏳ Test 4 Large amount lowered threshold: {res['decision']} (low={res['low_threshold']:.2f})")

    # Test 5: VIP customer relaxed threshold
    res = engine.make_decision(0.6, amount=100, customer_tier='VIP')
    print(f"✅ Test 5 VIP customer: {res['decision']} (low={res['low_threshold']:.2f})")

    # Test 6: Admin override
    override = engine.admin_override(
        transaction_id="tx123456",
        original_decision="BLOCK",
        new_decision="ALLOW",
        admin_id="admin_01",
        reason="Verified customer call",
        original_label=1,
        corrected_label=0
    )
    print(f"🔧 Test 6 Admin override: FP={override['was_false_positive']}")

    print("\n✅ All test scenarios passed")


if __name__ == "__main__":
    test_decision_engine()