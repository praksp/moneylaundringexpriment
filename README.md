# AML Transaction Risk Engine

An Anti-Money Laundering (AML) detection system built on a **Neo4j graph database** with a multi-model ML risk scoring engine, mule account detection, model versioning, and a full React + TypeScript frontend.

---

## Screenshots

### Login
![Login](screenshots/login.png)

### Dashboard
![Dashboard](screenshots/dashboard.png)

### Dashboard — World Transaction Heatmap
![Dashboard Map](screenshots/dashboard-map.png)

### Global Overview
![Global Overview](screenshots/global-overview.png)

### Submit Transaction
![Submit Transaction](screenshots/submit-transaction.png)

### Customer Profile List
![Customer Profile](screenshots/customer-profile.png)

### Customer Detail with Transaction History
![Customer Detail](screenshots/customer-detail.png)

### Transaction Detail & Risk Breakdown
![Transaction Detail](screenshots/transaction-detail.png)

### Model Monitor & Drift Detection
![Model Monitor](screenshots/model-monitor.png)

### Model Monitor — Charts
![Model Monitor Charts](screenshots/model-monitor-charts.png)

### Feature Store — High-Risk Account Analysis
![Feature Store](screenshots/feature-store.png)

### KNN Anomaly Detection
![KNN Anomaly](screenshots/knn-anomaly.png)

### GraphSAGE Mule Account Detection
![GraphSAGE Detection](screenshots/graphsage-detection.png)

### GraphSAGE — Account Detail (Mule Pattern Badges)
![GraphSAGE Drawer](screenshots/graphsage-drawer.png)

### GraphSAGE — Suspicious Transactions
![GraphSAGE Drawer Transactions](screenshots/graphsage-drawer-txns.png)

### GraphSAGE — How It Works (Mule Pattern Glossary)
![GraphSAGE Patterns](screenshots/graphsage-patterns.png)

---

## System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         React + TypeScript Frontend                          ║
║  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐   ║
║  │Dashboard │ │ Submit   │ │Customer  │ │ Feature  │ │ Model Monitor  │   ║
║  │+ HeatMap │ │Transaction│ │ Profiles │ │  Store   │ │ + Versioning   │   ║
║  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────────────┘   ║
║  ┌────────────────────────────────────────────────────────────────────────┐ ║
║  │  KNN Anomaly Detection  │  GraphSAGE Mule Detection  │  Global Overview│ ║
║  └────────────────────────────────────────────────────────────────────────┘ ║
╚══════════════════════════════╤═══════════════════════════════════════════════╝
                               │ REST (Axios)  JWT Bearer
                               │
╔══════════════════════════════▼═══════════════════════════════════════════════╗
║                    FastAPI Backend  (api/main.py)                            ║
║                                                                              ║
║  Auth & RBAC         ┌──────────────────────────────────────────────────┐   ║
║  ┌────────────────┐  │                   API Routes                     │   ║
║  │ JWT + bcrypt   │  │  /transactions    /evaluate    /submit            │   ║
║  │ admin / viewer │  │  /profiles        /monitoring  /anomaly           │   ║
║  │ roles in Neo4j │  │  /graphsage       /models      /auth              │   ║
║  └────────────────┘  └──────────┬───────────────────────────────────────┘   ║
║                                 │                                            ║
║  TTL In-Memory Cache            │          Background Tasks                  ║
║  ┌────────────────┐             │        ┌────────────────────┐              ║
║  │ api/cache.py   │◄────────────┤        │ Incremental Train  │              ║
║  │ world-map, stats│            │        │ (threading)        │              ║
║  └────────────────┘             │        └────────────────────┘              ║
╚═════════════════════════════════╤════════════════════════════════════════════╝
                                  │
          ┌───────────────────────┼────────────────────────┐
          │                       │                        │
╔═════════▼══════════╗   ╔════════▼═══════════╗  ╔════════▼══════════════╗
║   Risk Engine      ║   ║  Anomaly Detection ║  ║  Graph Neural Network ║
║  (risk/engine.py)  ║   ║  (ml/anomaly.py)   ║  ║  (ml/graphsage.py)    ║
║                    ║   ║                    ║  ║                       ║
║ ┌────────────────┐ ║   ║ FAISS IndexFlatL2  ║  ║ GraphSAGE from-scratch║
║ │Bayesian Engine │ ║   ║ 8 account features ║  ║ NumPy + SciPy sparse  ║
║ │ log-odds NB    │ ║   ║ Mule-account       ║  ║ Node embeddings via   ║
║ │ 55% weight     │ ║   ║ pattern scoring    ║  ║ 2-layer aggregation   ║
║ └────────┬───────┘ ║   ║ Two-step Neo4j     ║  ║                       ║
║          │         ║   ║ (ID fetch → stats) ║  ║ Mule patterns:        ║
║ ┌────────▼───────┐ ║   ╚════════════════════╝  ║ • Pass-through        ║
║ │  ML Ensemble   │ ║                           ║ • Funnel/aggregation  ║
║ │                │ ║                           ║ • Smurfing            ║
║ │ XGBoost 45%  ──┼─╫──► XGB warm-start        ║ • Structuring         ║
║ │ (hist quant.)  │ ║   incremental trains      ║ • Rapid cycling       ║
║ │                │ ║                           ╚═══════════════════════╝
║ │ SGD/SVM  10% ──┼─╫──► partial_fit()
║ │ (calibrated)   │ ║
║ └────────────────┘ ║
║                    ║
║  44 graph features ║
║  (risk/features.py)║
╚══════════════╤═════╝
               │
╔══════════════▼══════════════════════════════════════════════════════════════╗
║                    ML Training Pipeline  (ml/train.py)                       ║
║                                                                              ║
║  Full Retrain (scripts/retrain_models.sh)    Incremental (ml/incremental.py)║
║  ┌──────────────────────────────────────┐   ┌──────────────────────────────┐║
║  │ 1. Load all transactions (Neo4j)     │   │ 1. Delta query: WHERE        ║
║  │ 2. Extract 44 features per row       │   │    timestamp > checkpoint    ║
║  │ 3. Train XGBoost (300 trees, hist)   │   │ 2. XGB warm-start (+100 trees║
║  │ 4. Train SGD/SVM (modified_huber)    │   │ 3. SGD partial_fit()         ║
║  │ 5. Train KNN anomaly (FAISS)         │   │ 4. KNN re-index              ║
║  │ 6. Train GraphSAGE (60 epochs)       │   │ 5. GraphSAGE fine-tune       ║
║  │ 7. Export ONNX (INT8 quantized)      │   │    (20 epochs)               ║
║  │ 8. Register new baseline version     │   │ 6. Evaluate on hold-out 20%  ║
║  └──────────────────────────────────────┘   │ 7. Auto-promote if AUC ≥     ║
║                                             │    baseline × 0.99           ║
║                                             └──────────────────────────────┘║
╚══════════════╤═════════════════════════╤═══════════════════════════════════╝
               │                         │
╔══════════════▼═══════════╗   ╔═════════▼════════════════════════════════════╗
║  Model Version Registry   ║   ║            Neo4j Graph Database               ║
║  (ml/version.py)          ║   ║                                               ║
║                           ║   ║  Nodes:  Customer · Account · Transaction     ║
║  models_saved/            ║   ║          Device · IPAddress · Merchant        ║
║  ├── versions.json        ║   ║          BeneficiaryAccount · Country         ║
║  │   (manifest)           ║   ║          User · Role                          ║
║  └── versions/            ║   ║                                               ║
║      ├── v1/ (baseline)   ║   ║  Relationships:                               ║
║      │   ├── aml_model.   ║   ║  OWNS · INITIATED · CREDITED_TO              ║
║      │   │   joblib        ║   ║  PAID_TO · ORIGINATED_FROM                   ║
║      │   └── version_     ║   ║  SOURCED_FROM · SENT_TO_EXTERNAL             ║
║      │       meta.json     ║   ║  RESIDENT_OF · BASED_IN                      ║
║      └── v2/ (exper.)     ║   ║                                               ║
║          └── ...          ║   ║  Also stores:                                 ║
║                           ║   ║  PredictionLog · DriftReport                  ║
║  Lifecycle:               ║   ║  FeatureSnapshot · AnomalyScore               ║
║  full-retrain → baseline  ║   ║  GraphSAGEScore · UserRole                    ║
║  incremental  → exper.    ║   ╚═══════════════════════════════════════════════╝
║  promote      → baseline  ║
║  superseded   → retired   ║   ╔═══════════════════════════════════════════════╗
╚═══════════════════════════╝   ║         Monitoring & Drift (monitoring/)       ║
                                ║                                               ║
                                ║  PSI (Population Stability Index) on:         ║
                                ║   • Score distribution (0–999 buckets)        ║
                                ║   • Per-feature distributions (44 features)   ║
                                ║  Drift causes: score shift, feature drift,    ║
                                ║  fraud-rate change, volume anomaly            ║
                                ║  DriftReport stored in Neo4j with causes      ║
                                ╚═══════════════════════════════════════════════╝
```

---

## Data Flow — Transaction Evaluation

```
Submit transaction
       │
       ▼
Extract 44 graph features ──► Feature Store snapshot (Neo4j)
       │
       ├──► Bayesian Engine (log-odds NB) ──────────────────────┐
       │        LR priors for: sanctions, TOR, structuring,     │
       │        smurfing, dormancy, FATF corridor, velocity      │
       │                                                         │
       ├──► XGBoost (hist-quantized, 300 trees) ────────────────┤ Weighted
       │        ONNX INT8 inference when available               │ Ensemble
       │                                                         │
       └──► SGD/SVM (modified_huber, calibrated) ───────────────┘
                                                                 │
                                              0–999 Risk Score ◄─┘
                                                     │
                              ┌──────────────────────┼───────────────────────┐
                              ▼                      ▼                       ▼
                         0–399 ALLOW         400–699 CHALLENGE        700–999 DECLINE
                         Auto-approve        Present OTP /           Block + alert
                                             challenge question       compliance

       Shadow scoring (if experimental model exists):
       XGB v2 score stored alongside v1 decision for A/B tracking
```

---

## ML Model Suite

| Model | Algorithm | Role | Training |
|-------|-----------|------|----------|
| **XGBoost** | Gradient boosting, hist-quantized, 300 trees | Primary risk score (45%) | Full + warm-start incremental |
| **SGD/SVM** | `modified_huber` loss, calibrated | Secondary risk score (10%) | Full + `partial_fit()` |
| **Bayesian** | Log-odds Naive Bayes, calibrated LRs | Rule-based risk (55%) | Heuristic, no training |
| **KNN Anomaly** | FAISS `IndexFlatL2` on 8 account features | Mule account detection | Full + re-index |
| **GraphSAGE** | 2-layer GNN, NumPy + SciPy sparse | Mule account detection | Full + fine-tune (20 epochs) |

### Quantization
- XGBoost exported to ONNX → INT8 dynamic quantization via ONNX Runtime
- SGD/SVM exported to FP32 ONNX for portable inference
- ~4× model size reduction, ~2–3× inference speedup

---

## Model Versioning

Every trained model is tracked as a versioned artifact:

```
models_saved/
├── aml_model.joblib          ← active baseline models (used by API)
├── aml_scaler.joblib
├── svm_model.joblib
├── anomaly_index.faiss
├── graphsage_model.npz
├── training_metadata.json    ← drift detection baseline
├── versions.json             ← version manifest
└── versions/
    ├── v1/                   ← baseline (full retrain)
    │   ├── aml_model.joblib
    │   ├── version_meta.json
    │   └── ...
    └── v2/                   ← experimental (incremental)
        └── ...
```

### Version Lifecycle
```
bash scripts/retrain_models.sh   →  v1 created (status: baseline)
POST /models/train/incremental   →  v2 created (status: experimental)
  auto-promote if AUC ≥ v1 × 0.99 →  v2 promoted, v1 retired
  POST /models/versions/v2/promote →  manual promotion + hot model reload
```

### Incremental Training Optimization
Full retrain loads all 961k transactions (~10 min). Incremental training loads only new transactions since the version checkpoint:

```
MATCH (t:Transaction) WHERE t.timestamp > $last_trained_ts
```

For 1,000 new daily transactions this reduces data loading from ~10 min to < 5 s.

---

## Graph Data Model

Follows ACAMS / FATF standard financial crime graph patterns.

### Node Labels
| Node | Key Properties |
|------|----------------|
| `Customer` | id, name, nationality, kyc_level, pep_flag, sanctions_flag, risk_tier |
| `Account` | id, account_number, type, currency, balance, bank, status, anomaly_score |
| `Transaction` | id, amount, currency, type, channel, timestamp, is_fraud, fraud_type, outcome |
| `Device` | id, fingerprint, device_type |
| `IPAddress` | ip, country, is_vpn, is_tor |
| `Merchant` | id, name, mcc_code, category, country |
| `BeneficiaryAccount` | id, account_number, bank_swift, country |
| `Country` | code, name, fatf_risk, is_sanctioned, is_tax_haven |
| `User` | id, username, hashed_password, role |
| `PredictionLog` | txn_id, score, outcome, model_scores, latency_ms, timestamp |
| `DriftReport` | computed_at, score_psi, alert_level, drift_causes, top_features |

### Relationships
```
(Customer)-[:OWNS]->(Account)
(Account)-[:INITIATED]->(Transaction)
(Transaction)-[:CREDITED_TO]->(Account)
(Transaction)-[:PAID_TO]->(Merchant)
(Transaction)-[:ORIGINATED_FROM]->(Device)
(Transaction)-[:SOURCED_FROM]->(IPAddress)
(Transaction)-[:SENT_TO_EXTERNAL]->(BeneficiaryAccount)
(Customer)-[:RESIDENT_OF]->(Country)
(Account)-[:BASED_IN]->(Country)
(User)-[:HAS_ROLE]->(Role)
```

---

## Risk Scoring

### Score Range: 0–999
| Score | Outcome | Action |
|-------|---------|--------|
| 0–399 | **ALLOW** | Transaction proceeds automatically |
| 400–699 | **CHALLENGE** | Present challenge question / OTP |
| 700–999 | **DECLINE** | Transaction blocked, compliance alerted |

### Bayesian Risk Engine (55% weight)
Log-odds Naive Bayes with calibrated likelihood ratios:
- Prior: P(fraud) = 2%
- Key LRs: Sanctions (×50), TOR (×20), Structuring (×18–20), Rapid velocity (×12), Dormant account (×6), FATF corridor (×8)

### XGBoost (45% weight)
- Histogram-quantized gradient boosting (`tree_method="hist"`, `max_bin=256`)
- 300 trees, depth 6, `scale_pos_weight=5` for class imbalance
- INT8 ONNX quantized inference when available

### SGD/SVM (10% weight)
- `SGDClassifier(loss="modified_huber")` — O(n) training, O(1) inference
- Wrapped in `CalibratedClassifierCV` for probability calibration

---

## Mule Account Detection

### KNN Anomaly Detection (ml/anomaly.py)
- 8 account-level features: out_volume, in_volume, avg_out, avg_in, unique_senders, structuring, out_30d, avg_monthly_vol
- FAISS `IndexFlatL2` for fast nearest-neighbour distance
- Accounts with L2 distance > threshold flagged as anomalies
- Two-step Neo4j query: cheap ID fetch → targeted aggregation

### GraphSAGE (ml/graphsage.py)
- 2-layer Graph Sample & Aggregate implemented from scratch (NumPy + SciPy)
- Node features: 7 per account (out_volume, in_volume, structuring, etc.)
- Identifies 5 mule patterns:

| Pattern | Description |
|---------|-------------|
| **Pass-through** | >70% of received funds forwarded immediately |
| **Funnel / aggregation** | Many senders → one account |
| **Smurfing** | Multiple small deposits (CTR avoidance) |
| **Structuring** | Transactions near $9,000–$10,000 |
| **Rapid cycling** | High volume with low net retention |

---

## RBAC Security

| Role | Access |
|------|--------|
| **admin** | Full access: customer profiles, model training, all data |
| **viewer** | Aggregate views only; customer PII obfuscated/hidden |

- JWT authentication via `python-jose`
- Passwords hashed with `bcrypt`
- Users and roles stored in Neo4j

---

## Monitoring & Drift Detection

- **PSI** (Population Stability Index) computed on score distribution and all 44 features
- Drift causes classified as: score shift, feature drift, fraud-rate change, volume anomaly
- Every `DriftReport` stored in Neo4j with timestamped causes and top drifted features
- API: `POST /monitoring/drift` — compute on demand; `GET /monitoring/drift/history` — history

---

## Quick Start

### 1. Start Neo4j
```bash
docker-compose up -d
# Wait ~30s for Neo4j to initialise
```

### 2. Install Python dependencies
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run setup (seed data + train models)
```bash
python scripts/setup.py
```

### 4. Start the backend
```bash
uvicorn api.main:app --reload --port 8001
```

### 5. Start the frontend
```bash
cd frontend && npm install && npm run dev
```

### 6. Open the app
- **Frontend**: http://localhost:5174  (admin / password)
- **API Swagger**: http://localhost:8001/docs
- **Neo4j Browser**: http://localhost:7474 (neo4j / amlpassword123)

### Restart everything
```bash
bash scripts/restart.sh
```

### Retrain all models (full retrain, creates new baseline version)
```bash
bash scripts/retrain_models.sh
# Options:
#   --skip-sage     skip GraphSAGE (faster)
#   --skip-anomaly  skip KNN anomaly
#   --clean-only    wipe models without retraining
```

### Incremental train (via API — fast delta only)
```bash
# Trigger from the Model Monitor page, or:
curl -X POST http://localhost:8001/models/train/incremental \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"trigger":"manual","force":false,"auto_promote":true}'
```

---

## API Endpoints

### Authentication
```
POST /auth/token                    → obtain JWT
```

### Transactions
```
GET  /transactions/                 → list (paginated)
GET  /transactions/stats/summary    → aggregate stats
GET  /transactions/{id}             → full transaction + graph context
GET  /transactions/world-map        → heatmap data (cached)
```

### Evaluation
```
POST /evaluate/{txn_id}             → evaluate stored transaction
POST /evaluate/inline/evaluate      → evaluate inline (pre-storage)
POST /evaluate/batch/evaluate       → batch evaluation
POST /submit/transaction            → submit + evaluate new transaction
```

### Customer Profiles  *(admin only)*
```
GET  /profiles/customers            → list customers (paginated)
GET  /profiles/customers/{id}       → customer + transaction history
```

### Model Versioning  *(admin only)*
```
GET  /models/versions               → list all versions
GET  /models/versions/current       → baseline + experimental + comparison
GET  /models/versions/{vid}         → version detail
GET  /models/versions/{vid}/compare → compare vs baseline
POST /models/train/incremental      → trigger incremental training
GET  /models/train/status           → poll training job
POST /models/versions/{vid}/promote → promote to baseline + hot-reload
POST /models/versions/{vid}/retire  → retire a version
```

### Monitoring
```
GET  /monitoring/summary            → KPIs, outcome rates, drift alert
GET  /monitoring/drift/history      → drift reports + causes
POST /monitoring/drift              → compute drift now
GET  /monitoring/outcome-trend      → day-by-day outcome breakdown
GET  /monitoring/score-distribution → rolling score percentiles
```

### Anomaly Detection
```
POST /anomaly/train                 → train KNN anomaly detector
POST /anomaly/scan                  → scan accounts for mule suspects
GET  /anomaly/suspects              → list flagged accounts
GET  /anomaly/accounts/{id}         → account detail + anomaly score
```

### GraphSAGE
```
POST /graphsage/train               → train GraphSAGE
GET  /graphsage/summary             → model stats
GET  /graphsage/suspects            → mule suspects (paginated)
GET  /graphsage/accounts/{id}       → account detail + mule patterns
GET  /graphsage/comparison          → KNN vs GraphSAGE agreement
```

### Feature Store
```
GET  /monitoring/feature-store      → high-risk account features
```

---

## Project Structure

```
moneylaundringexpriment/
├── docker-compose.yml              # Neo4j 5.15 with APOC + GDS
├── requirements.txt
├── .env                            # Neo4j credentials, thresholds
│
├── config/
│   └── settings.py                 # Pydantic settings + feature flags
│
├── db/
│   ├── client.py                   # Neo4j driver + session context manager
│   ├── schema.py                   # Cypher constraints, indexes
│   └── models.py                   # Pydantic models for graph entities
│
├── auth/
│   ├── security.py                 # JWT creation/validation, bcrypt
│   ├── dependencies.py             # FastAPI deps: require_admin / require_viewer
│   └── models.py                   # User + Role Pydantic models
│
├── risk/
│   ├── features.py                 # 44-feature extractor from graph context
│   ├── bayesian.py                 # Bayesian log-odds risk engine
│   └── engine.py                   # Orchestrator: extract → score → decide
│
├── ml/
│   ├── model.py                    # XGBoost + SGD/SVM model wrappers + ONNX export
│   ├── train.py                    # Full training pipeline; registers baseline version
│   ├── anomaly.py                  # FAISS KNN mule-account anomaly detector
│   ├── graphsage.py                # GraphSAGE GNN (NumPy + SciPy, from scratch)
│   ├── incremental.py              # Incremental training (delta load + warm-start)
│   └── version.py                  # Model version registry (versions.json manifest)
│
├── monitoring/
│   ├── logger.py                   # PredictionLog writer to Neo4j
│   ├── drift.py                    # PSI computation + DriftReport writer
│   └── performance.py              # Latency + outcome KPI aggregation
│
├── store/
│   └── feature_store.py            # High-risk account feature snapshots
│
├── profiles/
│   └── customer_profile.py         # Customer + transaction history queries
│
├── api/
│   ├── main.py                     # FastAPI app + lifespan + CORS
│   ├── cache.py                    # TTL in-memory cache (world-map, stats)
│   └── routes/
│       ├── auth.py                 # /auth/token
│       ├── transactions.py         # /transactions
│       ├── evaluate.py             # /evaluate
│       ├── submit.py               # /submit/transaction
│       ├── profiles.py             # /profiles (admin only)
│       ├── monitoring.py           # /monitoring
│       ├── anomaly.py              # /anomaly
│       ├── graphsage.py            # /graphsage
│       └── models.py               # /models (versioning + incremental train)
│
├── data/
│   └── generator.py                # Synthetic transaction generator
│
├── scripts/
│   ├── setup.py                    # One-shot: schema + data + train
│   ├── retrain_models.sh           # Full retrain + version registration
│   ├── restart.sh                  # Stop/start backend + frontend
│   ├── expand_data.py              # Add more synthetic transactions
│   └── expand_customers.py         # Add more synthetic customers
│
├── models_saved/
│   ├── aml_model.joblib            # Active XGBoost model
│   ├── aml_scaler.joblib           # Feature scaler
│   ├── svm_model.joblib            # Active SGD/SVM model
│   ├── svm_model_fp32.onnx         # ONNX FP32 export
│   ├── xgb_model_fp32.onnx         # ONNX INT8 export
│   ├── anomaly_index.faiss         # KNN anomaly FAISS index
│   ├── anomaly_meta.pkl            # KNN metadata (scaler, threshold)
│   ├── graphsage_model.npz         # GraphSAGE weights
│   ├── graphsage_meta.pkl          # GraphSAGE metadata
│   ├── training_metadata.json      # Drift detection baseline
│   ├── versions.json               # Version manifest
│   └── versions/                   # Archived model artifacts per version
│       ├── v1/                     # Baseline
│       └── v2/                     # Experimental (if exists)
│
└── frontend/
    └── src/
        ├── api/client.ts           # Axios API client + TypeScript types
        └── pages/
            ├── Login.tsx
            ├── Dashboard.tsx
            ├── Overview.tsx
            ├── SubmitTransaction.tsx
            ├── CustomerProfiles.tsx
            ├── FeatureStore.tsx
            ├── ModelMonitor.tsx    # Drift + versioning + incremental train UI
            ├── AnomalyDetection.tsx
            └── GraphSAGEDetection.tsx
```

---

## Fraud Patterns in Dataset (~961k transactions)

| Pattern | Description |
|---------|-------------|
| **Structuring** | Multiple transactions just below $10,000 CTR threshold |
| **Smurfing** | Multiple sources aggregating funds to one account |
| **Layering** | Chain A→B→C→D→E rapid fund movement across accounts |
| **Round-tripping** | Money leaves and returns to origin within 48 hours |
| **Dormant Burst** | Long-dormant account suddenly makes high-volume transfers |
| **High-Risk Corridor** | Transfers to FATF grey/blacklisted countries |
| **Rapid Velocity** | 10+ transactions from one account within 1 hour |

---

## Useful Neo4j Queries

```cypher
// All fraud transactions by type
MATCH ()-[:INITIATED]->(t:Transaction {is_fraud: true})
RETURN t.fraud_type, count(*) ORDER BY count(*) DESC

// Fund flow path for a transaction
MATCH path = (c:Customer)-[:OWNS]->(a:Account)-[:INITIATED]->(t:Transaction)-[:CREDITED_TO]->(b:Account)
WHERE t.id = 'your-txn-id'
RETURN path

// Mule accounts flagged by both models
MATCH (a:Account)
WHERE a.anomaly_score IS NOT NULL AND a.graphsage_score IS NOT NULL
  AND a.anomaly_score > 0.7 AND a.graphsage_score > 0.7
RETURN a.id, a.anomaly_score, a.graphsage_score ORDER BY a.graphsage_score DESC

// Structuring pattern
MATCH (a:Account)-[:INITIATED]->(t:Transaction)
WHERE t.amount >= 9000 AND t.amount < 10000
WITH a, count(t) AS structuring_count WHERE structuring_count >= 2
RETURN a.id, a.account_number, structuring_count ORDER BY structuring_count DESC

// Recent drift reports
MATCH (d:DriftReport) RETURN d ORDER BY d.computed_at DESC LIMIT 5

// Model version history
MATCH (v:ModelVersion) RETURN v ORDER BY v.trained_at DESC
```
