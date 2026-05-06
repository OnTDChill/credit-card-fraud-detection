# Production Upgrade Recommendations

This document lists recommended production improvements and migration steps from the current PoC/local setup.

## Goals
- Replace local/PoC components with production-grade services
- Improve reliability, scalability, and security

## 1) Replace SQLite with Managed DB (Postgres)
- Provision managed Postgres (RDS / Azure Database for PostgreSQL)
- Export SQLite schema/data and import to Postgres
  - Use `pgloader` or export CSVs then `COPY`
- Update `DATABASE_URL` and verify Django migrations
- Test transactions, admin overrides, and retrain writes

## 2) Move model artifacts to Object Storage
- Use S3 / Azure Blob for `.joblib` and cached parquet
- On training completion, upload model to bucket and write `model_path` as an object URL in DB
- API service should fetch model from object store into local cache on startup

## 3) Deploy a lightweight inference service (FastAPI)
- Reason: separate concerns, faster startup, easier autoscaling
- Suggested structure:
  - `inference/` (FastAPI app) with endpoint `/score` that loads champion model and returns `fraud_score`
  - Dockerfile + healthcheck
  - Model hot-swap: poll `model_versions` or subscribe to webhook for promotions
- Example: containerize and deploy to Kubernetes / Container Apps / App Service

## 4) CI/CD for Retrain & Promote
- Pipeline stages:
  1. Data validation check
  2. Train challenger in isolated runner
  3. Evaluate metrics and store artifacts
  4. If metrics pass, run promotion job (manual approval step recommended)
  5. Deploy new model to staging inference service and run smoke tests
  6. Promote to production inference service
- Use GitHub Actions / Azure Pipelines / GitLab CI

## 5) Security & Secrets
- Store secrets in Key Vault / Secrets Manager
- Use IAM roles for object storage access (no embedded credentials)
- Use TLS and authenticate API calls (API key or OAuth)

## 6) Observability
- Export metrics (FP/FN rates, decision distribution, latency) to Prometheus/Grafana
- Send alerts to PagerDuty/Teams when thresholds exceeded
- Centralize logs (ELK / Azure Monitor)

## 7) Backups & Rollback
- Keep previous model versions for immediate rollback
- Backup DB daily and keep point-in-time restore where possible

---
Generated on 2026-04-29
