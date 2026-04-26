import json

from django.test import Client, TestCase, override_settings

from core.models import (
	FraudAlert,
	FraudBlocklist,
	FraudPolicy,
	FraudPolicyAuditLog,
	FraudTransaction,
)


@override_settings(FRAUD_CLIENT_API_KEY='test-fraud-key')
class FraudPolicyApiTests(TestCase):
	def setUp(self):
		self.client = Client()
		self.headers = {'HTTP_X_FRAUD_API_KEY': 'test-fraud-key'}

	def test_policy_get_requires_api_key(self):
		response = self.client.get('/api/fraud/policy/')
		self.assertEqual(response.status_code, 401)

	def test_policy_get_and_update(self):
		response = self.client.get('/api/fraud/policy/', **self.headers)
		self.assertEqual(response.status_code, 200)

		update_payload = {
			'fraud_threshold': 0.61,
			'manual_review': True,
			'block_transaction': True,
			'ban_user': True,
		}
		update_response = self.client.post(
			'/api/fraud/policy/',
			data=json.dumps(update_payload),
			content_type='application/json',
			**self.headers,
		)
		self.assertEqual(update_response.status_code, 200)

		policy = FraudPolicy.objects.get(name='default-policy')
		self.assertAlmostEqual(policy.fraud_threshold, 0.61)
		self.assertTrue(policy.manual_review)
		self.assertTrue(policy.block_transaction)
		self.assertTrue(policy.ban_user)
		self.assertEqual(FraudPolicyAuditLog.objects.count(), 1)


@override_settings(FRAUD_CLIENT_API_KEY='test-fraud-key')
class FraudRealtimeFlowTests(TestCase):
	def setUp(self):
		self.client = Client()
		self.headers = {'HTTP_X_FRAUD_API_KEY': 'test-fraud-key'}

	def test_ingest_stream_metrics_flow(self):
		payload = {
			'external_transaction_id': 'test-tx-001',
			'external_user_id': 'user-001',
			'card_fingerprint': 'card-001',
			'amount': 950.0,
			'currency': 'usd',
			'transaction_dt': 123456,
			'features': {
				'card1': 1234,
				'addr1': 320,
				'C1': 8,
			},
		}

		ingest_response = self.client.post(
			'/api/fraud/ingest/',
			data=json.dumps(payload),
			content_type='application/json',
			**self.headers,
		)
		self.assertEqual(ingest_response.status_code, 200)
		self.assertEqual(FraudTransaction.objects.count(), 1)

		stream_response = self.client.get('/api/fraud/stream/?limit=5', **self.headers)
		self.assertEqual(stream_response.status_code, 200)
		stream_json = stream_response.json()
		self.assertGreaterEqual(len(stream_json.get('transactions', [])), 1)

		metrics_response = self.client.get('/api/fraud/metrics/', **self.headers)
		self.assertEqual(metrics_response.status_code, 200)
		metrics_json = metrics_response.json()
		self.assertEqual(metrics_json.get('runtime', {}).get('total_transactions'), 1)

	def test_batch_score_requires_api_key(self):
		response = self.client.post(
			'/api/fraud/batch-score/',
			data=json.dumps({'transactions': []}),
			content_type='application/json',
		)
		self.assertEqual(response.status_code, 401)

	def test_batch_score_returns_predictions(self):
		payload = {
			'threshold': 0.5,
			'transactions': [
				{
					'input_index': 0,
					'external_transaction_id': 'batch-001',
					'external_user_id': 'user-001',
					'amount': 250.0,
					'transaction_dt': 123456,
					'is_fraud_truth': 0,
					'features': {
						'card1': 1234,
						'addr1': 200,
					},
				},
				{
					'input_index': 1,
					'external_transaction_id': 'batch-002',
					'external_user_id': 'user-002',
					'amount': 4800.0,
					'transaction_dt': 123556,
					'is_fraud_truth': 1,
				},
			],
		}

		response = self.client.post(
			'/api/fraud/batch-score/',
			data=json.dumps(payload),
			content_type='application/json',
			**self.headers,
		)
		self.assertEqual(response.status_code, 200)

		response_json = response.json()
		self.assertIn('summary', response_json)
		self.assertIn('results', response_json)
		self.assertEqual(response_json['summary']['total'], 2)
		self.assertEqual(len(response_json['results']), 2)
		self.assertIn('predicted_label', response_json['results'][0])


@override_settings(FRAUD_CLIENT_API_KEY='test-fraud-key')
class FraudEndToEndLifecycleTests(TestCase):
	def setUp(self):
		self.client = Client()
		self.headers = {'HTTP_X_FRAUD_API_KEY': 'test-fraud-key'}

	def test_policy_to_blocklist_lifecycle(self):
		policy_payload = {
			'fraud_threshold': 0.30,
			'manual_review': True,
			'block_transaction': True,
			'ban_user': True,
			'block_card_fingerprint': True,
		}

		policy_response = self.client.post(
			'/api/fraud/policy/',
			data=json.dumps(policy_payload),
			content_type='application/json',
			**self.headers,
		)
		self.assertEqual(policy_response.status_code, 200)

		high_risk_payload = {
			'external_transaction_id': 'e2e-high-risk-001',
			'external_user_id': 'e2e-user-001',
			'card_fingerprint': 'e2e-card-001',
			'amount': 5000.0,
			'currency': 'usd',
			'transaction_dt': 123456,
			'features': {
				'card1': 4556,
				'addr1': 101,
				'C1': 12,
			},
		}

		ingest_response = self.client.post(
			'/api/fraud/ingest/',
			data=json.dumps(high_risk_payload),
			content_type='application/json',
			**self.headers,
		)
		self.assertEqual(ingest_response.status_code, 200)
		ingest_json = ingest_response.json()
		self.assertEqual(ingest_json['decision'], 'block')

		self.assertTrue(
			FraudBlocklist.objects.filter(
				block_type='user_id',
				block_value='e2e-user-001',
				is_active=True,
			).exists()
		)
		self.assertTrue(
			FraudBlocklist.objects.filter(
				block_type='card_fingerprint',
				block_value='e2e-card-001',
				is_active=True,
			).exists()
		)

		follow_up_payload = {
			'external_transaction_id': 'e2e-follow-up-001',
			'external_user_id': 'e2e-user-001',
			'card_fingerprint': 'e2e-card-001',
			'amount': 15.0,
			'currency': 'usd',
			'transaction_dt': 123556,
			'features': {
				'card1': 4556,
				'addr1': 101,
			},
		}

		follow_up_response = self.client.post(
			'/api/fraud/ingest/',
			data=json.dumps(follow_up_payload),
			content_type='application/json',
			**self.headers,
		)
		self.assertEqual(follow_up_response.status_code, 200)
		follow_up_json = follow_up_response.json()
		self.assertEqual(follow_up_json['decision'], 'block')

		self.assertEqual(FraudTransaction.objects.count(), 2)
		self.assertGreaterEqual(FraudAlert.objects.count(), 2)

		stream_response = self.client.get('/api/fraud/stream/?limit=10', **self.headers)
		self.assertEqual(stream_response.status_code, 200)
		stream_json = stream_response.json()
		self.assertEqual(len(stream_json.get('transactions', [])), 2)
		self.assertGreaterEqual(len(stream_json.get('alerts', [])), 2)

		metrics_response = self.client.get('/api/fraud/metrics/', **self.headers)
		self.assertEqual(metrics_response.status_code, 200)
		runtime = metrics_response.json().get('runtime', {})
		self.assertEqual(runtime.get('total_transactions'), 2)
		self.assertEqual(runtime.get('blocked_transactions'), 2)
		self.assertGreaterEqual(runtime.get('open_alerts', 0), 2)
