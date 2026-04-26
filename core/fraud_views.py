import json
import os
from time import perf_counter
from typing import Any, Dict, List

from django.conf import settings
from django.http import JsonResponse
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from core.models import FraudAlert, FraudModelReport, FraudPolicyAuditLog, FraudTransaction
from core.services.fraud.inference import load_champion_pipeline, predict_fraud_score
from core.services.fraud.policy_engine import (
    apply_block_actions,
    evaluate_policy,
    get_active_policy,
    policy_snapshot,
)


def _parse_json_body(request) -> Dict[str, Any]:
    if not request.body:
        return {}
    try:
        return json.loads(request.body.decode('utf-8'))
    except json.JSONDecodeError:
        return {}


def _expected_api_key() -> str:
    return getattr(settings, 'FRAUD_CLIENT_API_KEY', 'local-fraud-api-key')


def _has_valid_api_key(request) -> bool:
    provided = request.headers.get('X-Fraud-Api-Key') or request.GET.get('api_key')
    return provided == _expected_api_key()


def _serialize_transaction(transaction: FraudTransaction) -> Dict[str, Any]:
    return {
        'id': transaction.id,
        'external_transaction_id': transaction.external_transaction_id,
        'external_user_id': transaction.external_user_id,
        'amount': transaction.amount,
        'currency': transaction.currency,
        'fraud_score': transaction.fraud_score,
        'is_fraud_prediction': transaction.is_fraud_prediction,
        'decision': transaction.decision,
        'created_at': transaction.created_at.isoformat(),
    }


def _serialize_alert(alert: FraudAlert) -> Dict[str, Any]:
    return {
        'id': alert.id,
        'transaction_id': alert.transaction_id,
        'severity': alert.severity,
        'status': alert.status,
        'message': alert.message,
        'created_at': alert.created_at.isoformat(),
    }


def _read_training_report() -> Dict[str, Any]:
    artifacts_dir = getattr(
        settings,
        'FRAUD_ARTIFACTS_DIR',
        os.path.join(settings.BASE_DIR, 'artifacts', 'fraud')
    )
    report_path = os.path.join(artifacts_dir, 'training_report.json')
    if not os.path.exists(report_path):
        return {}

    with open(report_path, 'r', encoding='utf-8') as report_file:
        return json.load(report_file)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _compute_batch_summary(results: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    predicted_fraud = sum(1 for item in results if item.get('predicted_label') == 'fraud')
    predicted_normal = len(results) - predicted_fraud

    summary: Dict[str, Any] = {
        'total': len(results),
        'predicted_fraud': predicted_fraud,
        'predicted_normal': predicted_normal,
        'threshold': float(threshold),
    }

    truth_available = all(item.get('truth_label') is not None for item in results) and bool(results)
    if not truth_available:
        return summary

    tp = sum(1 for item in results if item.get('truth_label') == 1 and item.get('predicted_label') == 'fraud')
    tn = sum(1 for item in results if item.get('truth_label') == 0 and item.get('predicted_label') == 'normal')
    fp = sum(1 for item in results if item.get('truth_label') == 0 and item.get('predicted_label') == 'fraud')
    fn = sum(1 for item in results if item.get('truth_label') == 1 and item.get('predicted_label') == 'normal')

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(results) if results else 0.0

    summary.update(
        {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'accuracy': float(accuracy),
            'confusion_matrix': {
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
            },
        }
    )
    return summary


@csrf_exempt
@require_http_methods(['POST'])
def fraud_ingest_view(request):
    if not _has_valid_api_key(request):
        return JsonResponse({'detail': 'Unauthorized'}, status=401)

    payload = _parse_json_body(request)
    if not payload:
        return JsonResponse({'detail': 'Payload không hợp lệ.'}, status=400)

    policy = get_active_policy()

    external_transaction_id = str(payload.get('external_transaction_id') or '')
    external_user_id = str(payload.get('external_user_id') or '')
    card_fingerprint = str(payload.get('card_fingerprint') or '')
    amount = float(payload.get('amount') or 0)
    currency = str(payload.get('currency') or 'usd')
    ip_address = str(payload.get('ip_address') or '')
    device_info = str(payload.get('device_info') or '')
    event_time = payload.get('event_time')
    event_time_value = parse_datetime(event_time) if isinstance(event_time, str) else None

    feature_payload = payload.get('features') or {}
    feature_payload.update(
        {
            'TransactionAmt': amount,
        }
    )

    if isinstance(payload.get('transaction_dt'), (int, float)):
        feature_payload['TransactionDT'] = payload['transaction_dt']

    model_name = 'rule-fallback'
    model_version = timezone.now().strftime('%Y%m%d%H%M%S')

    start_time = perf_counter()
    try:
        pipeline, model_name, model_version = load_champion_pipeline()
        fraud_score = predict_fraud_score(pipeline, feature_payload)
    except FileNotFoundError:
        # {'mục_đích': 'Fallback an toàn khi model chưa được train để hệ thống demo vẫn hoạt động', 'đầu_vào': 'số tiền giao dịch', 'đầu_ra': 'fraud_score heuristic'}
        fraud_score = min(1.0, amount / 5000.0)

    decision = evaluate_policy(
        policy=policy,
        fraud_score=fraud_score,
        external_user_id=external_user_id,
        card_fingerprint=card_fingerprint,
    )

    if decision.decision == 'block':
        apply_block_actions(
            policy=policy,
            external_user_id=external_user_id,
            card_fingerprint=card_fingerprint,
            reason='; '.join(decision.reasons),
        )

    elapsed_ms = (perf_counter() - start_time) * 1000

    fraud_transaction = FraudTransaction.objects.create(
        external_transaction_id=external_transaction_id or None,
        external_user_id=external_user_id or None,
        card_fingerprint=card_fingerprint or None,
        amount=amount,
        currency=currency,
        ip_address=ip_address or None,
        device_info=device_info or None,
        event_time=event_time_value,
        model_name=model_name,
        model_version=model_version,
        fraud_score=fraud_score,
        is_fraud_prediction=fraud_score >= policy.fraud_threshold,
        decision=decision.decision,
        decision_reasons='; '.join(decision.reasons),
        policy_snapshot=policy_snapshot(policy),
        raw_payload=json.dumps(payload, ensure_ascii=False),
        processing_latency_ms=elapsed_ms,
    )

    if decision.should_alert:
        severity = 'high' if decision.decision == 'block' else 'medium'
        FraudAlert.objects.create(
            transaction=fraud_transaction,
            severity=severity,
            status='open',
            message='; '.join(decision.reasons) or 'Fraud detection alert',
        )

    return JsonResponse(
        {
            'transaction_id': fraud_transaction.id,
            'external_transaction_id': external_transaction_id,
            'fraud_score': fraud_score,
            'fraud_threshold': policy.fraud_threshold,
            'decision': decision.decision,
            'reasons': decision.reasons,
            'latency_ms': elapsed_ms,
            'model_name': model_name,
            'model_version': model_version,
        },
        status=200,
    )


@csrf_exempt
@require_http_methods(['POST'])
def fraud_batch_score_view(request):
    if not _has_valid_api_key(request):
        return JsonResponse({'detail': 'Unauthorized'}, status=401)

    payload = _parse_json_body(request)
    transactions = payload.get('transactions') if isinstance(payload, dict) else None
    if not isinstance(transactions, list) or not transactions:
        return JsonResponse({'detail': 'Payload phải chứa danh sách transactions.'}, status=400)

    if len(transactions) > 5000:
        return JsonResponse({'detail': 'Số lượng transaction mỗi batch không vượt quá 5000.'}, status=400)

    threshold = _safe_float(payload.get('threshold', 0.5), default=0.5)
    threshold = min(max(threshold, 0.01), 0.99)

    model_name = 'rule-fallback'
    model_version = timezone.now().strftime('%Y%m%d%H%M%S')

    pipeline = None
    try:
        pipeline, model_name, model_version = load_champion_pipeline()
    except FileNotFoundError:
        pipeline = None

    start_time = perf_counter()
    results: List[Dict[str, Any]] = []

    for index, transaction in enumerate(transactions):
        if not isinstance(transaction, dict):
            continue

        global_index = _safe_int(transaction.get('input_index')) if transaction.get('input_index') is not None else index

        amount = _safe_float(transaction.get('amount', 0.0), default=0.0)
        feature_payload = transaction.get('features') if isinstance(transaction.get('features'), dict) else {}

        feature_payload['TransactionAmt'] = amount
        if transaction.get('transaction_dt') is not None:
            feature_payload['TransactionDT'] = _safe_float(transaction.get('transaction_dt'))

        if pipeline is None:
            fraud_score = min(1.0, amount / 5000.0)
        else:
            fraud_score = predict_fraud_score(pipeline, feature_payload)

        predicted_label = 'fraud' if fraud_score >= threshold else 'normal'
        confidence = max(fraud_score, 1 - fraud_score)

        truth_label = None
        for truth_key in ['is_fraud_truth', 'isFraud', 'truth_label', 'label', 'target']:
            if transaction.get(truth_key) is not None:
                truth_label = _safe_int(transaction.get(truth_key))
                break

        results.append(
            {
                'row_index': global_index,
                'external_transaction_id': str(transaction.get('external_transaction_id') or ''),
                'external_user_id': str(transaction.get('external_user_id') or ''),
                'amount': amount,
                'fraud_probability': float(fraud_score),
                'prediction_confidence': float(confidence),
                'predicted_label': predicted_label,
                'truth_label': truth_label,
            }
        )

    total_latency_ms = (perf_counter() - start_time) * 1000
    summary = _compute_batch_summary(results=results, threshold=threshold)

    return JsonResponse(
        {
            'detail': 'Batch scoring thành công.',
            'model_name': model_name,
            'model_version': model_version,
            'summary': summary,
            'results': results,
            'latency_ms': total_latency_ms,
        },
        status=200,
    )


@require_http_methods(['GET'])
def fraud_stream_view(request):
    if not _has_valid_api_key(request):
        return JsonResponse({'detail': 'Unauthorized'}, status=401)

    limit = int(request.GET.get('limit', 25))
    limit = max(1, min(200, limit))

    transactions = FraudTransaction.objects.order_by('-created_at')[:limit]
    alerts = FraudAlert.objects.order_by('-created_at')[:limit]

    return JsonResponse(
        {
            'transactions': [_serialize_transaction(tx) for tx in transactions],
            'alerts': [_serialize_alert(alert) for alert in alerts],
        },
        status=200,
    )


@require_http_methods(['GET'])
def fraud_metrics_view(request):
    if not _has_valid_api_key(request):
        return JsonResponse({'detail': 'Unauthorized'}, status=401)

    now = timezone.now()
    window_start = now - timezone.timedelta(hours=24)

    total_count = FraudTransaction.objects.count()
    recent_count = FraudTransaction.objects.filter(created_at__gte=window_start).count()
    block_count = FraudTransaction.objects.filter(decision='block').count()
    review_count = FraudTransaction.objects.filter(decision='review').count()
    alert_open_count = FraudAlert.objects.filter(status='open').count()

    latest_report = _read_training_report()
    latest_model_report = FraudModelReport.objects.order_by('-created_at').first()

    champion_summary: Dict[str, Any] = {}
    if latest_model_report:
        try:
            champion_summary = json.loads(latest_model_report.report_json)
        except json.JSONDecodeError:
            champion_summary = {'raw': latest_model_report.report_json}

    return JsonResponse(
        {
            'runtime': {
                'total_transactions': total_count,
                'transactions_last_24h': recent_count,
                'blocked_transactions': block_count,
                'review_transactions': review_count,
                'open_alerts': alert_open_count,
            },
            'training_report': latest_report,
            'champion_report_db': champion_summary,
        },
        status=200,
    )


@csrf_exempt
@require_http_methods(['GET', 'POST'])
def fraud_policy_view(request):
    if not _has_valid_api_key(request):
        return JsonResponse({'detail': 'Unauthorized'}, status=401)

    policy = get_active_policy()

    if request.method == 'GET':
        return JsonResponse(
            {
                'name': policy.name,
                'fraud_threshold': policy.fraud_threshold,
                'block_transaction': policy.block_transaction,
                'ban_user': policy.ban_user,
                'block_card_fingerprint': policy.block_card_fingerprint,
                'manual_review': policy.manual_review,
                'send_email_alert': policy.send_email_alert,
                'updated_at': policy.updated_at.isoformat(),
            },
            status=200,
        )

    payload = _parse_json_body(request)
    if not payload:
        return JsonResponse({'detail': 'Payload không hợp lệ.'}, status=400)

    if 'fraud_threshold' in payload:
        threshold_value = float(payload['fraud_threshold'])
        if threshold_value <= 0 or threshold_value >= 1:
            return JsonResponse({'detail': 'fraud_threshold phải nằm trong khoảng (0, 1).'}, status=400)
        policy.fraud_threshold = threshold_value

    for bool_field in [
        'block_transaction',
        'ban_user',
        'block_card_fingerprint',
        'manual_review',
        'send_email_alert',
    ]:
        if bool_field in payload:
            policy.__setattr__(bool_field, bool(payload[bool_field]))

    policy.updated_by = request.user if getattr(request, 'user', None) and request.user.is_authenticated else None
    policy.save()

    FraudPolicyAuditLog.objects.create(
        policy=policy,
        changed_by=policy.updated_by,
        change_source='api',
        payload=json.dumps(payload, ensure_ascii=False),
    )

    return JsonResponse(
        {
            'detail': 'Cập nhật policy thành công.',
            'policy': {
                'name': policy.name,
                'fraud_threshold': policy.fraud_threshold,
                'block_transaction': policy.block_transaction,
                'ban_user': policy.ban_user,
                'block_card_fingerprint': policy.block_card_fingerprint,
                'manual_review': policy.manual_review,
                'send_email_alert': policy.send_email_alert,
                'updated_at': policy.updated_at.isoformat(),
            },
        },
        status=200,
    )
