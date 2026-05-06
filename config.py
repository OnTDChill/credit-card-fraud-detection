"""Centralized configuration for fraud detection paths and thresholds."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
FRAUD_ARTIFACTS_DIR = ARTIFACTS_DIR / "fraud"

# File paths
MODEL_PATH = MODEL_DIR / "champion.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
RESULTS_PATH = MODEL_DIR / "fraud_detection_results.pkl"
BEST_MODEL_PATH = MODEL_DIR / "best_model_info.pkl"
DATA_PATH = DATA_DIR / "creditcard.csv"
BENCHMARK_PATH = ARTIFACTS_DIR / "benchmark_results.csv"
TRAINING_REPORT_PATH = FRAUD_ARTIFACTS_DIR / "training_report.json"
REVIEW_DB_PATH = ARTIFACTS_DIR / "fraud_system.db"
UNIFIED_PAYMENTS_PATH = DATA_DIR / "unified_payments.csv"

# Existing artifact fallback
FRAUD_MODEL_FALLBACK_PATH = FRAUD_ARTIFACTS_DIR / "fraud_model.joblib"

# Business parameters
AVG_FRAUD_AMOUNT = 122.21
INVESTIGATION_COST = 50.00
CURRENCY_SYMBOL = "$"

# Threshold defaults
DEFAULT_LOW_THRESHOLD = 0.35
DEFAULT_HIGH_THRESHOLD = 0.65
REVIEW_TTL_MINUTES = 30

# Alert thresholds
FP_RATE_ALERT = 0.20
FN_RATE_ALERT = 0.20
REVIEW_QUEUE_WARN = 50

# DSS / Executive parameters
MONTHLY_RISK_BUDGET = 500_000_000.0  # VND — ngân sách rủi ro tháng
FIXED_SYSTEM_COST = 50_000_000.0  # VND — chi phí vận hành hệ thống cố định/tháng
COST_PER_REVIEW = 25_000.0  # VND — chi phí nhân sự xét duyệt 1 giao dịch
CUSTOMER_CHURN_COST = 200_000.0  # VND — chi phí mất 1 khách hàng do FP
MERCHANT_RISK_THRESHOLD = 0.05  # 5% — ngưỡng merchant cần cảnh báo
SLA_MINUTES = 30  # SLA xét duyệt
UPGRADE_INVESTMENT = 2_000_000_000.0  # VND — đầu tư nâng cấp AI


def resolve_model_path() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    if FRAUD_MODEL_FALLBACK_PATH.exists():
        return FRAUD_MODEL_FALLBACK_PATH
    return MODEL_PATH


def get_api_base_url() -> str:
    return os.getenv("FRAUD_API_BASE_URL", "http://127.0.0.1:8000/api/fraud")


def get_benchmark_path() -> Path:
    return BENCHMARK_PATH


def get_fraud_artifacts_dir() -> Path:
    return FRAUD_ARTIFACTS_DIR


def get_review_db_path() -> Path:
    return REVIEW_DB_PATH


# =============================================
# DSS CEO Dashboard Configuration
# =============================================

# Fee rates by transaction type (%)
FEE_RATES = {
    "PAYMENT": 0.015,    # Thanh toán: 1.5%
    "TRANSFER": 0.005,   # Chuyển tiền: 0.5%
    "CASH_IN": 0.003,    # Nạp ví: 0.3%
    "CASH_OUT": 0.01,    # Rút tiền: 1%
    "DEBIT": 0.008,      # Thanh toán hóa đơn: 0.8%
}

# Vietnamese names for transaction types
TRANSACTION_TYPE_NAMES = {
    "PAYMENT": "Thanh toán",
    "TRANSFER": "Chuyển tiền",
    "CASH_IN": "Nạp ví",
    "CASH_OUT": "Rút tiền",
    "DEBIT": "Thanh toán hóa đơn",
}

# Risk thresholds
FRAUD_LOSS_RATE = 0.05  # Actual company loss = 5% of fraud transaction value (chargeback, investigation, liability)
DEFAULT_RISK_TOLERANCE = 0.02  # 2% fraud rate acceptable
DEFAULT_TRANSACTION_LIMIT = {
    "PAYMENT": 100_000_000,    # 100M VND
    "TRANSFER": 200_000_000,   # 200M VND
    "CASH_IN": 50_000_000,     # 50M VND
    "CASH_OUT": 30_000_000,    # 30M VND
    "DEBIT": 20_000_000,       # 20M VND
}

# Marketing channel defaults
MARKETING_CHANNELS = ["Mạng xã hội", "Email", "Display", "SMS", "Affiliate"]
DEFAULT_CHANNEL_BUDGET = {
    "Mạng xã hội": 0.40,
    "Email": 0.20,
    "Display": 0.15,
    "SMS": 0.15,
    "Affiliate": 0.10,
}

# CSAT score thresholds
CSAT_EXCELLENT = 4.5
CSAT_GOOD = 4.0
CSAT_WARNING = 3.5

# NPS thresholds
NPS_PROMOTER = 50
NPS_PASSIVE = 30
NPS_DETRACTOR = 0

# =============================================
# Product Code System
# Format: {SERVICE}-{CHANNEL}-{SEGMENT}-{USECASE}
# =============================================

PRODUCT_CODE_MAP = {
    "PAYMENT":  "PAY-QR-B2C-SHOP",
    "TRANSFER": "TRF-APP-B2C-P2P",
    "CASH_IN":  "WAL-APP-B2C-TOP",
    "CASH_OUT": "WAL-APP-B2C-WDR",
    "DEBIT":    "BILL-API-B2C-UTIL",
}

PRODUCT_DISPLAY_NAMES = {
    "PAY-QR-B2C-SHOP":  "Thanh toán mua sắm",
    "TRF-APP-B2C-P2P":  "Chuyển tiền cá nhân",
    "WAL-APP-B2C-TOP":  "Nạp ví",
    "WAL-APP-B2C-WDR":  "Rút tiền",
    "BILL-API-B2C-UTIL": "Thanh toán hóa đơn",
}

PRODUCT_CATALOG = {
    "PAY-QR-B2C-SHOP":  {"service_type": "PAY",  "channel": "QR",  "segment": "B2C", "use_case": "SHOP", "fee_rate": 0.015, "txn_limit": 100_000_000},
    "TRF-APP-B2C-P2P":  {"service_type": "TRF",  "channel": "APP", "segment": "B2C", "use_case": "P2P",  "fee_rate": 0.005, "txn_limit": 200_000_000},
    "WAL-APP-B2C-TOP":  {"service_type": "WAL",  "channel": "APP", "segment": "B2C", "use_case": "TOP",  "fee_rate": 0.003, "txn_limit": 50_000_000},
    "WAL-APP-B2C-WDR":  {"service_type": "WAL",  "channel": "APP", "segment": "B2C", "use_case": "WDR",  "fee_rate": 0.01,  "txn_limit": 30_000_000},
    "BILL-API-B2C-UTIL": {"service_type": "BILL", "channel": "API", "segment": "B2C", "use_case": "UTIL", "fee_rate": 0.008, "txn_limit": 20_000_000},
}
