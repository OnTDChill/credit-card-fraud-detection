# Architecture Diagram (Mermaid)

This file shows the high-level service boundaries and data flow for the fraud detection product.

```mermaid
flowchart LR
  subgraph Ingest
    A[Raw Data: IEEE CSVs / Demo Batches] -->|CSV| DP[Data Pipeline]
  end

  subgraph Training
    DP -->|master tables (parquet)| TR[Retrain Pipeline]
    TR --> ART[Artifacts/Model Storage]
    ART --> DB[Model Metadata (SQLite)]
  end

  subgraph Serving
    DB --> API[Django API (/api/fraud)]
    ART -->|load joblib| API
    API --> DE[DecisionEngine]
    DE -->|decision| TX[FraudTransaction (DB)]
    TX --> Alerts[FraudAlert / Review Queue]
  end

  subgraph Dashboard
    UI[Streamlit Dashboard] -->|REST| API
    UI -->|upload CSV| API
  end

  subgraph Operators
    Admin[Operator / Admin UI] -->|override| API
    Admin --> DB
  end

  subgraph External
    PaymentGateway((Payment Service)) -->|transactions| API
  end

  %% Notes
  classDef infra fill:#f8f9fa,stroke:#333,stroke-width:1
  ART:::infra
  DB:::infra
  API:::infra

  click ART "./artifacts" "Open artifacts folder"
```
