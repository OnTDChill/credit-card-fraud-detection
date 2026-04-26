import random
import time
from datetime import datetime, timezone

import requests
from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Gửi giao dịch giả lập tới Fraud API để kiểm thử realtime detection.'

    def add_arguments(self, parser):
        parser.add_argument('--total', type=int, default=50, help='Tổng số giao dịch gửi.')
        parser.add_argument('--fraud-ratio', type=float, default=0.2, help='Tỷ lệ giao dịch fraud giả lập (0-1).')
        parser.add_argument('--interval-ms', type=int, default=300, help='Khoảng nghỉ giữa 2 request (ms).')
        parser.add_argument('--api-base-url', type=str, default='http://127.0.0.1:8000/api/fraud', help='Base URL Fraud API.')
        parser.add_argument(
            '--api-key',
            type=str,
            default=getattr(settings, 'FRAUD_CLIENT_API_KEY', 'local-fraud-api-key'),
            help='API key gửi qua header X-Fraud-Api-Key.',
        )

    def handle(self, *args, **options):
        total = max(1, options['total'])
        fraud_ratio = min(1.0, max(0.0, options['fraud_ratio']))
        interval_ms = max(0, options['interval_ms'])
        api_base_url = options['api_base_url'].rstrip('/')
        api_key = options['api_key']

        ingest_url = f'{api_base_url}/ingest/'
        headers = {
            'X-Fraud-Api-Key': api_key,
            'Content-Type': 'application/json',
        }

        blocked_count = 0
        review_count = 0
        allow_count = 0

        for index in range(total):
            is_fraud_scenario = random.random() < fraud_ratio
            base_amount = random.uniform(10, 400)
            if is_fraud_scenario:
                amount = base_amount + random.uniform(800, 3500)
                card_fingerprint = f'card-risk-{index % 5}'
                user_id = f'user-risk-{index % 7}'
            else:
                amount = base_amount
                card_fingerprint = f'card-safe-{index % 50}'
                user_id = f'user-safe-{index % 80}'

            payload = {
                'external_transaction_id': f'sim-tx-{int(time.time())}-{index}',
                'external_user_id': user_id,
                'card_fingerprint': card_fingerprint,
                'amount': round(amount, 2),
                'currency': 'usd',
                'ip_address': f'10.0.0.{(index % 240) + 10}',
                'device_info': random.choice(['chrome-win', 'mobile-safari', 'android-app']),
                'event_time': datetime.now(timezone.utc).isoformat(),
                'transaction_dt': int(time.time()),
                'features': {
                    'card1': random.randint(1000, 20000),
                    'card2': random.randint(100, 600),
                    'addr1': random.randint(100, 500),
                    'C1': random.randint(0, 50),
                    'C2': random.randint(0, 40),
                    'D1': random.randint(0, 30),
                },
                'scenario_tag': 'fraud' if is_fraud_scenario else 'normal',
            }

            try:
                response = requests.post(ingest_url, json=payload, headers=headers, timeout=10)
                response.raise_for_status()
                response_json = response.json()
                decision = response_json.get('decision', 'unknown')
            except Exception as error:  # noqa: BLE001
                self.stdout.write(self.style.WARNING(f'#{index + 1}/{total} lỗi gửi request: {error}'))
                time.sleep(interval_ms / 1000)
                continue

            if decision == 'block':
                blocked_count += 1
            elif decision == 'review':
                review_count += 1
            elif decision == 'allow':
                allow_count += 1

            self.stdout.write(
                f"#{index + 1}/{total} scenario={'fraud' if is_fraud_scenario else 'normal'} "
                f"amount={payload['amount']} decision={decision} score={response_json.get('fraud_score')}"
            )

            if interval_ms:
                time.sleep(interval_ms / 1000)

        self.stdout.write(self.style.SUCCESS('Hoàn tất giả lập giao dịch.'))
        self.stdout.write(
            self.style.SUCCESS(
                f'Tổng kết: allow={allow_count}, review={review_count}, block={blocked_count}, total={total}'
            )
        )
