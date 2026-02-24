"""
AML Risk Engine – FastAPI Application
=======================================
Endpoints:
  GET  /health                         → service health + model status
  GET  /transactions/                  → list transactions (paginated)
  GET  /transactions/stats/summary     → aggregate statistics
  GET  /transactions/{id}              → full transaction + graph context
  POST /evaluate/{txn_id}              → evaluate stored transaction
  POST /evaluate/inline/evaluate       → evaluate transaction inline (pre-storage)
  POST /evaluate/batch/evaluate        → batch evaluation
"""
import time
from contextlib import asynccontextmanager

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from api.routes.transactions import router as txn_router
from api.routes.evaluate import router as eval_router
from api.routes.profiles import router as profiles_router
from api.routes.monitoring import router as monitoring_router
from api.routes.submit import router as submit_router
from api.routes.auth import router as auth_router
from api.routes.anomaly import router as anomaly_router
from api.routes.graphsage import router as graphsage_router
from db.client import get_driver, close_driver
from ml.model import get_model, get_registry
from ml.anomaly import get_detector
from config.settings import settings
from monitoring.logger import ensure_schema
from auth.security import seed_default_users
from config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    get_driver()            # Verify Neo4j connectivity
    get_model()             # Pre-load XGBoost (legacy)
    get_registry()          # Pre-load risk models (XGBoost + SVM)
    if settings.enable_knn_anomaly:
        get_detector()      # Pre-load KNN anomaly detector (feature-flagged)
    if settings.enable_graphsage:
        from ml.graphsage import get_sage
        get_sage()          # Pre-load GraphSAGE weights if saved
    ensure_schema()         # Prediction log constraints
    seed_default_users()    # Create admin/viewer users if not present
    yield
    # Shutdown
    close_driver()


app = FastAPI(
    title="AML Transaction Risk Engine",
    description=(
        "Anti-Money Laundering detection system using Neo4j graph database "
        "and a Bayesian + ML risk scoring engine. "
        "Risk scores: 0–399 ALLOW | 400–699 CHALLENGE | 700–999 DECLINE"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(submit_router)
app.include_router(txn_router)
app.include_router(eval_router)
app.include_router(profiles_router)
app.include_router(monitoring_router)
app.include_router(anomaly_router)
app.include_router(graphsage_router)


@app.get("/health", tags=["System"])
async def health_check():
    """Service health check including Neo4j connectivity and all ML model status."""
    neo4j_ok = False
    neo4j_error = None
    try:
        driver = get_driver()
        driver.verify_connectivity()
        neo4j_ok = True
    except Exception as e:
        neo4j_error = str(e)

    registry = get_registry()
    detector = get_detector()
    from ml.graphsage import get_sage
    sage_model = get_sage()
    models_status = {
        "xgboost":           registry.xgb.is_trained,
        "svm":               registry.svm.is_trained,
        "knn_anomaly":       detector.is_trained,
        "graphsage":         sage_model.is_trained,
    }
    all_models_ok = any([registry.xgb.is_trained, registry.svm.is_trained])

    return {
        "status": "healthy" if (neo4j_ok and all_models_ok) else "degraded",
        "neo4j": {"connected": neo4j_ok, "error": neo4j_error},
        "ml_models": models_status,
        "ensemble_weights": registry.ENSEMBLE_WEIGHTS,
        "risk_thresholds": {
            "ALLOW": f"0–{settings.risk_allow_max}",
            "CHALLENGE": f"{settings.risk_allow_max + 1}–{settings.risk_challenge_max}",
            "DECLINE": f"{settings.risk_challenge_max + 1}–999",
        },
    }


@app.post("/retrain", tags=["System"])
async def retrain_models():
    """Retrain all three ML models (XGBoost, SVM, KNN) from the current Neo4j dataset."""
    import asyncio
    from ml.train import train_and_save_all
    loop = asyncio.get_event_loop()
    metrics = await loop.run_in_executor(None, train_and_save_all)
    return {
        "status": "success",
        "message": "All models retrained and saved",
        "metrics": metrics,
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "service": "AML Transaction Risk Engine",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0",
    }

# Serve React frontend (must be last)
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(_frontend_dist / "assets")), name="assets")

    @app.get("/app/{full_path:path}", include_in_schema=False)
    async def serve_frontend(_: str):
        return FileResponse(str(_frontend_dist / "index.html"))
