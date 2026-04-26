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
    training_batch_id   TEXT,

    INDEX idx_transaction_id (transaction_id),
    INDEX idx_corrected_at (corrected_at),
    INDEX idx_false_positive (is_false_positive),
    INDEX idx_false_negative (is_false_negative)
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
    rollback_reason     TEXT,

    INDEX idx_is_champion (is_champion),
    INDEX idx_trained_at (trained_at)
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
    alert_reasons       TEXT,

    INDEX idx_logged_at (logged_at)
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
    review_reason       TEXT,

    INDEX idx_status (status),
    INDEX idx_expires_at (expires_at)
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
    event_hash          TEXT NOT NULL,

    INDEX idx_event_type (event_type),
    INDEX idx_transaction_id (transaction_id),
    INDEX idx_event_time (event_time)
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
    updated_by          TEXT,

    INDEX idx_config_key (config_key)
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
print("✅ Database schema created successfully");