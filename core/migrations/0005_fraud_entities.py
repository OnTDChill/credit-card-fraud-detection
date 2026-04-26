# Generated manually for fraud entities

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('core', '0004_auto_20190630_1408'),
    ]

    operations = [
        migrations.CreateModel(
            name='FraudBlocklist',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('block_type', models.CharField(choices=[('user_id', 'User ID'), ('card_fingerprint', 'Card Fingerprint')], max_length=32)),
                ('block_value', models.CharField(max_length=255)),
                ('reason', models.CharField(blank=True, max_length=255, null=True)),
                ('is_active', models.BooleanField(default=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'unique_together': {('block_type', 'block_value')},
            },
        ),
        migrations.CreateModel(
            name='FraudModelReport',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model_name', models.CharField(max_length=128)),
                ('model_version', models.CharField(max_length=64)),
                ('is_champion', models.BooleanField(default=False)),
                ('report_json', models.TextField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='FraudPolicy',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(default='default-policy', max_length=64, unique=True)),
                ('is_active', models.BooleanField(default=True)),
                ('fraud_threshold', models.FloatField(default=0.5)),
                ('block_transaction', models.BooleanField(default=True)),
                ('ban_user', models.BooleanField(default=False)),
                ('block_card_fingerprint', models.BooleanField(default=True)),
                ('manual_review', models.BooleanField(default=True)),
                ('send_email_alert', models.BooleanField(default=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('updated_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='updated_fraud_policies', to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='FraudTransaction',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('external_transaction_id', models.CharField(blank=True, max_length=128, null=True)),
                ('external_user_id', models.CharField(blank=True, max_length=128, null=True)),
                ('card_fingerprint', models.CharField(blank=True, max_length=255, null=True)),
                ('amount', models.FloatField(default=0)),
                ('currency', models.CharField(default='usd', max_length=12)),
                ('ip_address', models.CharField(blank=True, max_length=64, null=True)),
                ('device_info', models.CharField(blank=True, max_length=255, null=True)),
                ('event_time', models.DateTimeField(blank=True, null=True)),
                ('model_name', models.CharField(blank=True, max_length=128, null=True)),
                ('model_version', models.CharField(blank=True, max_length=64, null=True)),
                ('fraud_score', models.FloatField(default=0)),
                ('is_fraud_prediction', models.BooleanField(default=False)),
                ('decision', models.CharField(choices=[('allow', 'Allow'), ('review', 'Manual Review'), ('block', 'Block')], default='allow', max_length=16)),
                ('decision_reasons', models.TextField(blank=True, null=True)),
                ('policy_snapshot', models.TextField(blank=True, null=True)),
                ('raw_payload', models.TextField(blank=True, null=True)),
                ('processing_latency_ms', models.FloatField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='FraudPolicyAuditLog',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('change_source', models.CharField(default='api', max_length=64)),
                ('payload', models.TextField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('changed_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='fraud_policy_audit_logs', to=settings.AUTH_USER_MODEL)),
                ('policy', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='audit_logs', to='core.fraudpolicy')),
            ],
        ),
        migrations.CreateModel(
            name='FraudAlert',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('severity', models.CharField(choices=[('low', 'Low'), ('medium', 'Medium'), ('high', 'High')], default='medium', max_length=16)),
                ('status', models.CharField(choices=[('open', 'Open'), ('closed', 'Closed')], default='open', max_length=16)),
                ('message', models.TextField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('transaction', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='alerts', to='core.fraudtransaction')),
            ],
        ),
    ]
