-- =============================================
-- FRAUD DETECTION PRODUCTION DATABASE SCHEMA
-- 3 Tier Architecture: Review Zone / Override / Continuous Learning
-- =============================================

-- =============================================
-- TABLE: feedback_pool
-- Stores ground truth after admin corrections for model retraining
-- =============================================
CREATE TABLE IF NOT EXISTS feedback_pool (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id      TEXT NOT NULL UNIQUE,
    original_decision   TEXT NOT NULL,          -- ALLOW / REVIEW / BLOCK
    original_label      INTEGER NOT NULL,       -- 0 = normal, 1 = fraud (model prediction)
    corrected_label     INTEGER NOT NULL,       -- 0 = normal, 1 = fraud (ground truth)
    corrected_by        TEXT NOT NULL,          -- admin_id / system
    corrected_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reason              TEXT NOT NULL,
    is_false_positive   BOOLEAN GENERATED ALWAYS AS (original_label = 1 AND corrected_label = 0) STORED,
    is_false_negative   BOOLEAN GENERATED ALWAYS AS (original_label = 0 AND corrected_label = 1) STORED,
    used_for_training   BOOLEAN DEFAULT FALSE,
    training_batch_id   TEXT
);

-- =============================================
-- TABLE: model_versions
-- Model version history for champion/challenger and rollback
-- =============================================
CREATE TABLE IF NOT EXISTS model_versions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    version             TEXT NOT NULL UNIQUE,
    trained_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    training_samples    INTEGER NOT NULL,
    feedback_samples    INTEGER NOT NULL DEFAULT 0,
    f1_score            REAL NOT NULL,
    recall              REAL NOT NULL,
    precision           REAL NOT NULL,
    pr_auc              REAL NOT NULL,
    fpr                 REAL NOT NULL,
    latency_p95_ms      REAL NOT NULL,
    is_champion         BOOLEAN DEFAULT FALSE,
    is_active           BOOLEAN DEFAULT TRUE,
    model_path          TEXT NOT NULL,
    training_log_path   TEXT,
    deployed_at         TIMESTAMP,
    deployed_by         TEXT,
    rollback_reason     TEXT
);

-- =============================================
-- TABLE: drift_logs
-- Model performance drift monitoring logs
-- =============================================
CREATE TABLE IF NOT EXISTS drift_logs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    window_days         INTEGER NOT NULL DEFAULT 7,
    total_transactions  INTEGER NOT NULL,
    fp_count            INTEGER NOT NULL,
    fn_count            INTEGER NOT NULL,
    fp_rate             REAL NOT NULL,
    fn_rate             REAL NOT NULL,
    alert_fired         BOOLEAN DEFAULT FALSE,
    alert_reasons       TEXT
);

-- =============================================
-- TABLE: review_queue
-- Transactions waiting for manual review
-- =============================================
CREATE TABLE IF NOT EXISTS review_queue (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id      TEXT NOT NULL UNIQUE,
    fraud_probability   REAL NOT NULL,
    amount              REAL NOT NULL,
    customer_id         TEXT,
    merchant_id         TEXT,
    received_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at          TIMESTAMP NOT NULL,
    reason_codes        TEXT,
    status              TEXT NOT NULL DEFAULT 'PENDING',  -- PENDING / APPROVED / REJECTED / EXPIRED
    reviewed_by         TEXT,
    reviewed_at         TIMESTAMP,
    review_reason       TEXT
);

-- =============================================
-- TABLE: audit_log
-- Immutable log of all admin actions and system decisions
-- =============================================
CREATE TABLE IF NOT EXISTS audit_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type          TEXT NOT NULL,
    transaction_id      TEXT,
    user_id             TEXT,
    event_time          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    old_value           TEXT,
    new_value           TEXT,
    reason              TEXT,
    event_hash          TEXT NOT NULL
);

-- =============================================
-- TABLE: system_config
-- Runtime configuration stored in database
-- =============================================
CREATE TABLE IF NOT EXISTS system_config (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    config_key          TEXT NOT NULL UNIQUE,
    config_value        TEXT NOT NULL,
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_by          TEXT
);

-- =============================================
-- INSERT DEFAULT CONFIGURATION
-- =============================================
INSERT OR IGNORE INTO system_config (config_key, config_value) VALUES
    ('low_threshold', '0.35'),
    ('high_threshold', '0.65'),
    ('review_ttl_minutes', '15'),
    ('small_amount_threshold', '50.0'),
    ('large_amount_threshold', '1000.0'),
    ('drift_window_days', '7'),
    ('max_fp_rate', '0.10'),
    ('max_fn_rate', '0.20'),
    ('auto_retrain_enabled', 'true'),
    ('auto_retrain_days', '7'),
    ('minimum_feedback_samples', '100');

-- =============================================
-- INSERT INITIAL MODEL VERSION
-- =============================================
INSERT OR IGNORE INTO model_versions (
    version, trained_at, training_samples, f1_score, recall, precision, pr_auc, fpr, latency_p95_ms,
    is_champion, model_path
) VALUES (
    'v1.0.0-initial',
    CURRENT_TIMESTAMP,
    590540,
    0.3693,
    0.5920,
    0.2684,
    0.4621,
    0.05,
    0.04,
    TRUE,
    'artifacts/fraud_model_v1.0.0.joblib'
);

-- =============================================
-- TABLE: dss_transaction_summary
-- Aggregated transaction data for Revenue and Risk tabs
-- =============================================
CREATE TABLE IF NOT EXISTS dss_transaction_summary (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    year                INTEGER NOT NULL,
    month               INTEGER NOT NULL,
    transaction_type    TEXT NOT NULL,           -- PAYMENT, TRANSFER, CASH_IN, CASH_OUT, DEBIT
    region              TEXT,                   -- Miền Bắc, Miền Trung, Miền Nam
    total_count         INTEGER NOT NULL DEFAULT 0,
    total_amount        REAL NOT NULL DEFAULT 0.0,
    revenue_estimated   REAL NOT NULL DEFAULT 0.0,
    fraud_count         INTEGER NOT NULL DEFAULT 0,
    fraud_amount        REAL NOT NULL DEFAULT 0.0,
    fraud_rate          REAL NOT NULL DEFAULT 0.0,
    avg_score           REAL DEFAULT 0.0,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(year, month, transaction_type, region)
);

-- =============================================
-- TABLE: dss_marketing_monthly
-- Marketing campaign aggregated data for Services tab
-- =============================================
CREATE TABLE IF NOT EXISTS dss_marketing_monthly (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    year                INTEGER NOT NULL,
    month               INTEGER NOT NULL,
    channel             TEXT NOT NULL,
    campaign_type       TEXT,
    campaign_spend      REAL NOT NULL DEFAULT 0.0,
    avg_cac             REAL NOT NULL DEFAULT 0.0,
    avg_roi             REAL NOT NULL DEFAULT 0.0,
    avg_conversion      REAL NOT NULL DEFAULT 0.0,
    total_clicks        INTEGER DEFAULT 0,
    total_impressions   INTEGER DEFAULT 0,
    ltv_estimated       REAL DEFAULT 0.0,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(year, month, channel)
);

-- =============================================
-- TABLE: dss_customer_service
-- Customer support aggregated data for Services tab
-- =============================================
CREATE TABLE IF NOT EXISTS dss_customer_service (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    year                    INTEGER NOT NULL,
    month                   INTEGER NOT NULL,
    ticket_count            INTEGER NOT NULL DEFAULT 0,
    avg_resolution_hours    REAL DEFAULT 0.0,
    csat_score              REAL DEFAULT 0.0,
    nps_score               REAL DEFAULT 0.0,
    churn_count             INTEGER DEFAULT 0,
    churn_rate              REAL DEFAULT 0.0,
    created_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(year, month)
);

-- =============================================
-- TABLE: dss_strategy_targets
-- CEO strategy targets and gap analysis
-- =============================================
CREATE TABLE IF NOT EXISTS dss_strategy_targets (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    target_year         INTEGER NOT NULL,
    domain              TEXT NOT NULL,           -- REVENUE, RISK, SERVICE, GROWTH
    kpi_name            TEXT NOT NULL,
    target_value        REAL NOT NULL,
    actual_value        REAL DEFAULT 0.0,
    unit                TEXT DEFAULT 'percent', -- percent, amount, count
    strategy_note       TEXT,
    created_by          TEXT,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(target_year, domain, kpi_name)
);

-- =============================================
-- TABLE: dim_product
-- Product code dimension table for multi-dimensional BI analysis
-- =============================================
CREATE TABLE IF NOT EXISTS dim_product (
    product_code    TEXT PRIMARY KEY,
    transaction_type TEXT NOT NULL,       -- PaySim original type
    service_type    TEXT NOT NULL,        -- PAY, TRF, WAL, BILL
    channel         TEXT NOT NULL,        -- APP, QR, POS, API, WEB
    segment         TEXT NOT NULL,        -- B2C, B2B, MER
    use_case        TEXT NOT NULL,        -- SHOP, P2P, TOP, WDR, UTIL
    display_name    TEXT NOT NULL,        -- Vietnamese display name
    risk_level      TEXT DEFAULT 'LOW',   -- LOW, MEDIUM, HIGH
    fee_rate        REAL NOT NULL,
    txn_limit       REAL DEFAULT 0
);

INSERT OR IGNORE INTO dim_product (product_code, transaction_type, service_type, channel, segment, use_case, display_name, risk_level, fee_rate, txn_limit) VALUES
    ('PAY-QR-B2C-SHOP',  'PAYMENT',  'PAY',  'QR',  'B2C', 'SHOP', 'Thanh toán mua sắm',  'LOW',    0.015, 100000000),
    ('TRF-APP-B2C-P2P',  'TRANSFER', 'TRF',  'APP', 'B2C', 'P2P',  'Chuyển tiền cá nhân', 'MEDIUM', 0.005, 200000000),
    ('WAL-APP-B2C-TOP',  'CASH_IN',  'WAL',  'APP', 'B2C', 'TOP',  'Nạp ví',              'LOW',    0.003, 50000000),
    ('WAL-APP-B2C-WDR',  'CASH_OUT', 'WAL',  'APP', 'B2C', 'WDR',  'Rút tiền',            'HIGH',   0.01,  30000000),
    ('BILL-API-B2C-UTIL','DEBIT',    'BILL', 'API', 'B2C', 'UTIL', 'Thanh toán hóa đơn',  'LOW',    0.008, 20000000);

-- =============================================
-- TABLE: dss_credit_portfolio
-- Credit portfolio summary by segment for Module 2
-- =============================================
CREATE TABLE IF NOT EXISTS dss_credit_portfolio (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    year                INTEGER NOT NULL,
    month               INTEGER NOT NULL,
    segment             TEXT NOT NULL,           -- GenZ, Kinh doanh, NV Van phong, Taos, Huu tri, Sinh vien
    region              TEXT,                   -- Miền Bắc, Miền Trung, Miền Nam
    total_users         INTEGER NOT NULL DEFAULT 0,
    credit_limit        REAL NOT NULL DEFAULT 0.0,    -- average credit limit VND
    interest_rate       REAL NOT NULL DEFAULT 0.0,     -- %
    npl_rate            REAL NOT NULL DEFAULT 0.0,    -- non-performing loan rate %
    default_probability REAL NOT NULL DEFAULT 0.0,     -- model predicted
    total_outstanding   REAL NOT NULL DEFAULT 0.0,     -- total outstanding amount VND
    overdue_30d_amount  REAL NOT NULL DEFAULT 0.0,
    overdue_90d_amount  REAL NOT NULL DEFAULT 0.0,
    revenue_interest    REAL NOT NULL DEFAULT 0.0,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(year, month, segment, region)
);

-- =============================================
-- TABLE: dss_merchant_accounts
-- Merchant / individual account anomaly detection for Module 4
-- =============================================
CREATE TABLE IF NOT EXISTS dss_merchant_accounts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    year                INTEGER NOT NULL,
    month               INTEGER NOT NULL,
    account_id          TEXT NOT NULL,
    account_type        TEXT NOT NULL DEFAULT 'PERSONAL', -- PERSONAL, MERCHANT
    segment             TEXT,                          -- GenZ, Kinh doanh, ...
    region              TEXT,                          -- Miền Bắc, Miền Trung, Miền Nam
    daily_txn_count     INTEGER NOT NULL DEFAULT 0,
    daily_txn_amount    REAL NOT NULL DEFAULT 0.0,
    monthly_volume      REAL NOT NULL DEFAULT 0.0,
    anomaly_score       REAL NOT NULL DEFAULT 0.0,     -- Isolation Forest score
    is_suspected_merchant BOOLEAN DEFAULT FALSE,
    est_monthly_revenue REAL NOT NULL DEFAULT 0.0,
    est_tax_collectable REAL NOT NULL DEFAULT 0.0,
    risk_level          TEXT DEFAULT 'LOW',
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(year, month, account_id, region)
);

-- =============================================
-- TABLE: dss_service_ecosystem
-- Cross-sell association rules between services for Module 3
-- =============================================
CREATE TABLE IF NOT EXISTS dss_service_ecosystem (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    year                INTEGER NOT NULL,
    month               INTEGER NOT NULL,
    service_a           TEXT NOT NULL,
    service_b           TEXT NOT NULL,
    support_count       INTEGER NOT NULL DEFAULT 0,
    support_pct         REAL NOT NULL DEFAULT 0.0,     -- % users bought both
    confidence          REAL NOT NULL DEFAULT 0.0,     -- P(B|A)
    lift                REAL NOT NULL DEFAULT 0.0,
    revenue_a           REAL NOT NULL DEFAULT 0.0,     -- monthly revenue from A
    revenue_b           REAL NOT NULL DEFAULT 0.0,
    profit_margin_a     REAL NOT NULL DEFAULT 0.0,     -- profit margin %
    profit_margin_b     REAL NOT NULL DEFAULT 0.0,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(year, month, service_a, service_b)
);

-- =============================================
-- Database schema created successfully