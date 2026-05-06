# Fraud Model Detective

Unified credit-card fraud detection product with a Django API backend, model training and evaluation pipelines, and a multi-page Streamlit CEO Dashboard.

## Current Product Stack

- **Backend API**: Django (`manage.py`, `core/`)
- **CEO Dashboard**: Streamlit (`streamlit_app/app.py`)
- **Data store**: SQLite (`db.sqlite3`, `artifacts/fraud_system.db`)
- **ML stack**: scikit-learn + imbalanced-learn pipelines (`core/`)
- **Dataset**: PaySim and generated demo batches (`data/`)

## Repository Layout

```
├── config.py                 # Centralized configuration
├── core/                     # Django app: API, ML pipelines, ETL
│   ├── etl/                  # Data loading scripts (PaySim, Marketing, CSKH)
│   ├── services/fraud/       # ML training, inference, policy engine
│   └── ...
├── streamlit_app/            # CEO Dashboard
│   ├── app.py                # Main entry point
│   ├── pages/
│   │   ├── ceo/              # Business pages (Tổng quan, Doanh thu, An toàn, Dịch vụ, Báo cáo)
│   │   └── tech/             # Technical pages (Xét duyệt, Tinh chỉnh, Kỹ thuật)
│   ├── components/           # Reusable UI components & DSS engine
│   └── shared_ui.py          # Common styling & layout
├── docs/                     # Documentation
│   ├── architecture/         # System design docs
│   ├── operations/           # Runbooks & deployment guides
│   └── guides/               # User guides
├── data/                     # Raw datasets
├── artifacts/                # Model artifacts & outputs
├── djecommerce/              # Django settings
├── templates/                # Django HTML templates
└── static_in_env/            # Static assets
```

## Requirements

- Windows PowerShell
- Python 3.10.x
- A local virtual environment

## Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If the environment was created earlier and you only need to reactivate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Run Backend API

```powershell
python manage.py migrate --noinput
python manage.py runserver
```

Backend base URL: `http://127.0.0.1:8000/api/fraud`

## Run CEO Dashboard

```powershell
python -m streamlit run streamlit_app/app.py
```

Dashboard will be available at `http://localhost:8501`

## Test Gate

Run the full backend regression suite:

```powershell
python manage.py test core --verbosity 2
```

Run the production smoke gate:

```powershell
python manage.py test core.test_streamlit_smoke core.tests.FraudEndToEndLifecycleTests --verbosity 2
```

Expected current baseline: both commands pass.

## Operational Notes

- Use the unified Streamlit entrypoint (`streamlit_app/app.py`) for day-to-day usage.
- Shared page configuration and theme are centralized in `streamlit_app/shared_ui.py`.
- Dashboard batch scoring uses the backend API by default, so the Django server must be running before uploading CSV files in Streamlit.
- Product language is standardized in English across UI and docs.
