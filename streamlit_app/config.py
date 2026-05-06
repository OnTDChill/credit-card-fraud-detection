"""Streamlit configuration shim for the centralized config module."""

from __future__ import annotations

import importlib.util
import os
import sys

# Load root config.py by absolute file path to avoid circular import.
# Streamlit adds streamlit_app/ to sys.path, so `import config` would
# resolve to this very file instead of the project-root config.py.
_root_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py")
_spec = importlib.util.spec_from_file_location("_root_config", _root_config_path)
_root_config = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_root_config", _root_config)
_spec.loader.exec_module(_root_config)

# Re-export everything from root config
ARTIFACTS_DIR = _root_config.ARTIFACTS_DIR
AVG_FRAUD_AMOUNT = _root_config.AVG_FRAUD_AMOUNT
BENCHMARK_PATH = _root_config.BENCHMARK_PATH
BEST_MODEL_PATH = _root_config.BEST_MODEL_PATH
COST_PER_REVIEW = _root_config.COST_PER_REVIEW
CURRENCY_SYMBOL = _root_config.CURRENCY_SYMBOL
CUSTOMER_CHURN_COST = _root_config.CUSTOMER_CHURN_COST
DATA_DIR = _root_config.DATA_DIR
DATA_PATH = _root_config.DATA_PATH
DEFAULT_HIGH_THRESHOLD = _root_config.DEFAULT_HIGH_THRESHOLD
DEFAULT_LOW_THRESHOLD = _root_config.DEFAULT_LOW_THRESHOLD
FIXED_SYSTEM_COST = _root_config.FIXED_SYSTEM_COST
FN_RATE_ALERT = _root_config.FN_RATE_ALERT
FP_RATE_ALERT = _root_config.FP_RATE_ALERT
FRAUD_ARTIFACTS_DIR = _root_config.FRAUD_ARTIFACTS_DIR
FRAUD_LOSS_RATE = _root_config.FRAUD_LOSS_RATE
FRAUD_MODEL_FALLBACK_PATH = _root_config.FRAUD_MODEL_FALLBACK_PATH
INVESTIGATION_COST = _root_config.INVESTIGATION_COST
MERCHANT_RISK_THRESHOLD = _root_config.MERCHANT_RISK_THRESHOLD
MODEL_DIR = _root_config.MODEL_DIR
MODEL_PATH = _root_config.MODEL_PATH
MONTHLY_RISK_BUDGET = _root_config.MONTHLY_RISK_BUDGET
OUTPUT_DIR = _root_config.OUTPUT_DIR
RESULTS_PATH = _root_config.RESULTS_PATH
REVIEW_DB_PATH = _root_config.REVIEW_DB_PATH
REVIEW_QUEUE_WARN = _root_config.REVIEW_QUEUE_WARN
REVIEW_TTL_MINUTES = _root_config.REVIEW_TTL_MINUTES
ROOT_DIR = _root_config.ROOT_DIR
SCALER_PATH = _root_config.SCALER_PATH
SLA_MINUTES = _root_config.SLA_MINUTES
TRAINING_REPORT_PATH = _root_config.TRAINING_REPORT_PATH
UNIFIED_PAYMENTS_PATH = _root_config.UNIFIED_PAYMENTS_PATH
UPGRADE_INVESTMENT = _root_config.UPGRADE_INVESTMENT
get_api_base_url = _root_config.get_api_base_url
get_benchmark_path = _root_config.get_benchmark_path
get_fraud_artifacts_dir = _root_config.get_fraud_artifacts_dir
get_review_db_path = _root_config.get_review_db_path
resolve_model_path = _root_config.resolve_model_path

PRODUCT_CODE_MAP = _root_config.PRODUCT_CODE_MAP
PRODUCT_DISPLAY_NAMES = _root_config.PRODUCT_DISPLAY_NAMES
PRODUCT_CATALOG = _root_config.PRODUCT_CATALOG
