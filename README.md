# Fraud Model Detective

Unified credit-card fraud detection product with a Django API backend, model training and evaluation pipelines, and a multi-page Streamlit governance dashboard.

## Current Product Stack

- Backend API: Django (`manage.py`, `core/`)
- Dashboard: Streamlit (`streamlit_app/app.py`)
- Data store: SQLite (`db.sqlite3`, `artifacts/fraud_system.db`)
- ML stack: scikit-learn + imbalanced-learn pipelines (`core/`)
- Dataset: IEEE-CIS and generated demo batches (`data/`)

## Repository Layout

- `core/`: API, decision engine, training and retrain pipelines, benchmarks
- `streamlit_app/`: Unified BI-style dashboard with page navigation
- `artifacts/`: model artifacts, benchmark outputs, predictions
- `data/`: raw and demo datasets
- `djecommerce/settings/`: Django settings profiles

## Requirements

- Windows PowerShell
- Python 3.10.x
- A local virtual environment

## Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install "pip<24" "setuptools<58" "wheel<0.41"
python -m pip install --no-build-isolation -r requirements.txt
python -m pip install -r requirements-ml.txt
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

Backend base URL used by the dashboard:

- `http://127.0.0.1:8000/api/fraud`

Default local API key:

- `local-fraud-api-key`

## Run Unified Streamlit Dashboard

Start the backend first, then run the dashboard:

```powershell
Set-Location .\streamlit_app
python -m streamlit run app.py
```

The legacy single-file dashboard remains available for debugging:

```powershell
Set-Location .\streamlit_app
python -m streamlit run fraud_dashboard.py
```

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
