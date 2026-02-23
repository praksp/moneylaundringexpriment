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
from db.client import get_driver, close_driver
from ml.model import get_model
from monitoring.logger import ensure_schema
from config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    get_driver()        # Verify Neo4j connectivity
    get_model()         # Pre-load ML model
    ensure_schema()     # Prediction log constraints
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

app.include_router(submit_router)
app.include_router(txn_router)
app.include_router(eval_router)
app.include_router(profiles_router)
app.include_router(monitoring_router)


@app.get("/health", tags=["System"])
async def health_check():
    """Service health check including Neo4j connectivity and ML model status."""
    neo4j_ok = False
    neo4j_error = None
    try:
        driver = get_driver()
        driver.verify_connectivity()
        neo4j_ok = True
    except Exception as e:
        neo4j_error = str(e)

    model = get_model()
    model_ok = model.is_trained

    return {
        "status": "healthy" if (neo4j_ok and model_ok) else "degraded",
        "neo4j": {"connected": neo4j_ok, "error": neo4j_error},
        "ml_model": {"loaded": model_ok},
        "risk_thresholds": {
            "ALLOW": f"0–{settings.risk_allow_max}",
            "CHALLENGE": f"{settings.risk_allow_max + 1}–{settings.risk_challenge_max}",
            "DECLINE": f"{settings.risk_challenge_max + 1}–999",
        },
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
