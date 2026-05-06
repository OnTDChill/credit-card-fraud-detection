# Production Readiness Checklist

This checklist defines the minimum release gate for the current product.

## 1. Automated Quality Gate

- CI workflow exists at `.github/workflows/ci.yml`.
- Backend regression tests pass: `python manage.py test core --verbosity 2`.
- Smoke + e2e gate passes:
  - `python manage.py test core.test_streamlit_smoke`
    `core.tests.FraudEndToEndLifecycleTests --verbosity 2`

## 2. Runtime Consistency

- Backend stack is Django API (`/api/fraud/*`).
- Dashboard stack is Streamlit (`streamlit_app/app.py`).
- Shared dashboard shell is applied across pages via `streamlit_app/shared_ui.py`.

## 3. Documentation Consistency

- `README.md` describes the current runtime architecture and run commands.
- Prompt docs are language-consistent and aligned to Django + Streamlit.
- Markdown lint configuration is defined in `.markdownlint.json`.

## 4. Fraud Lifecycle Reliability

- Policy update API is protected by API key checks.
- Ingest endpoint produces decision and creates alerts when required.
- Blocklist actions are applied when configured.
- Metrics endpoint reflects runtime counters for transactions, decisions, and alerts.

## 5. Release Commands

Run before deployment:

```powershell
python manage.py test core --verbosity 2
python manage.py test core.test_streamlit_smoke \
  core.tests.FraudEndToEndLifecycleTests --verbosity 2
```

If both commands pass, the current codebase satisfies the baseline production gate.
