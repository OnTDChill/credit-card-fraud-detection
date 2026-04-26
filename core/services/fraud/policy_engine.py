import json
from dataclasses import dataclass
from typing import Dict, List

from core.models import FraudBlocklist, FraudPolicy


@dataclass
class PolicyDecision:
    decision: str
    reasons: List[str]
    should_alert: bool


def get_active_policy() -> FraudPolicy:
    policy = FraudPolicy.objects.filter(is_active=True).order_by('-updated_at').first()
    if policy:
        return policy

    policy = FraudPolicy.objects.create(name='default-policy', is_active=True)
    return policy


def policy_snapshot(policy: FraudPolicy) -> str:
    snapshot = {
        'name': policy.name,
        'fraud_threshold': policy.fraud_threshold,
        'block_transaction': policy.block_transaction,
        'ban_user': policy.ban_user,
        'block_card_fingerprint': policy.block_card_fingerprint,
        'manual_review': policy.manual_review,
        'send_email_alert': policy.send_email_alert,
    }
    return json.dumps(snapshot, ensure_ascii=False)


def check_blocklist(external_user_id: str, card_fingerprint: str) -> PolicyDecision:
    reasons: List[str] = []

    if external_user_id:
        user_blocked = FraudBlocklist.objects.filter(
            block_type='user_id',
            block_value=external_user_id,
            is_active=True,
        ).exists()
        if user_blocked:
            reasons.append('User nằm trong blocklist')

    if card_fingerprint:
        card_blocked = FraudBlocklist.objects.filter(
            block_type='card_fingerprint',
            block_value=card_fingerprint,
            is_active=True,
        ).exists()
        if card_blocked:
            reasons.append('Card fingerprint nằm trong blocklist')

    if reasons:
        return PolicyDecision(decision='block', reasons=reasons, should_alert=True)

    return PolicyDecision(decision='allow', reasons=[], should_alert=False)


def evaluate_policy(
    policy: FraudPolicy,
    fraud_score: float,
    external_user_id: str,
    card_fingerprint: str,
) -> PolicyDecision:
    pre_decision = check_blocklist(external_user_id=external_user_id, card_fingerprint=card_fingerprint)
    if pre_decision.decision == 'block':
        return pre_decision

    reasons: List[str] = []
    is_fraud = fraud_score >= policy.fraud_threshold

    if not is_fraud:
        return PolicyDecision(decision='allow', reasons=['Điểm rủi ro dưới ngưỡng'], should_alert=False)

    reasons.append(f'Fraud score {fraud_score:.4f} vượt ngưỡng {policy.fraud_threshold:.4f}')

    if policy.block_transaction:
        decision = 'block'
    elif policy.manual_review:
        decision = 'review'
    else:
        decision = 'allow'

    should_alert = decision in {'block', 'review'} or policy.send_email_alert
    return PolicyDecision(decision=decision, reasons=reasons, should_alert=should_alert)


def apply_block_actions(
    policy: FraudPolicy,
    external_user_id: str,
    card_fingerprint: str,
    reason: str,
) -> None:
    if policy.ban_user and external_user_id:
        FraudBlocklist.objects.update_or_create(
            block_type='user_id',
            block_value=external_user_id,
            defaults={
                'is_active': True,
                'reason': reason,
            },
        )

    if policy.block_card_fingerprint and card_fingerprint:
        FraudBlocklist.objects.update_or_create(
            block_type='card_fingerprint',
            block_value=card_fingerprint,
            defaults={
                'is_active': True,
                'reason': reason,
            },
        )
