"""
api/routes/graphsage.py
=======================
REST endpoints for GraphSAGE mule-account detection.
All endpoints are admin-only.
"""
from fastapi import APIRouter, Depends, Query, BackgroundTasks, HTTPException
from pydantic import BaseModel

from auth.dependencies import require_admin
from db.client import neo4j_session
from config.settings import settings

router = APIRouter(prefix="/graphsage", tags=["GraphSAGE"])


# ── Response schemas ───────────────────────────────────────────────────────────

class GraphSAGESuspect(BaseModel):
    account_id: str
    graphsage_score: float          # 0–100
    is_suspect: bool
    customer_id: str | None = None
    customer_name: str | None = None
    customer_country: str | None = None
    account_number: str | None = None
    bank_name: str | None = None
    account_type: str | None = None
    scored_at: str | None = None
    knn_anomaly_score: float | None = None   # side-by-side comparison


class SuspectPageResponse(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    suspects: list[GraphSAGESuspect]
    model_trained: bool
    training_stats: dict


class TrainingStatus(BaseModel):
    status: str
    message: str
    model_trained: bool


# ── Cypher helpers ─────────────────────────────────────────────────────────────

SUSPECT_QUERY = """
MATCH (a:Account)
WHERE a.graphsage_suspect = true
OPTIONAL MATCH (c:Customer)-[:OWNS]->(a)
RETURN
  a.id                   AS account_id,
  a.account_number       AS account_number,
  a.bank_name            AS bank_name,
  a.account_type         AS account_type,
  a.graphsage_score      AS graphsage_score,
  a.graphsage_scored_at  AS scored_at,
  a.anomaly_score        AS knn_score,
  c.id                   AS customer_id,
  c.name                 AS customer_name,
  c.country_of_residence AS customer_country
ORDER BY a.graphsage_score DESC
SKIP $skip LIMIT $limit
"""

SUSPECT_COUNT_QUERY = """
MATCH (a:Account) WHERE a.graphsage_suspect = true RETURN count(a) AS n
"""


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/suspects", response_model=SuspectPageResponse,
            dependencies=[Depends(require_admin)])
async def list_graphsage_suspects(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
):
    """Paginated list of accounts flagged by GraphSAGE, ranked by score."""
    from ml.graphsage import get_sage
    model = get_sage()
    skip  = (page - 1) * page_size

    with neo4j_session() as s:
        total = s.run(SUSPECT_COUNT_QUERY).single()["n"]
        rows  = s.run(SUSPECT_QUERY, skip=skip, limit=page_size).data()

    suspects = [
        GraphSAGESuspect(
            account_id       = r["account_id"],
            graphsage_score  = float(r.get("graphsage_score") or 0),
            is_suspect       = True,
            customer_id      = r.get("customer_id"),
            customer_name    = r.get("customer_name"),
            customer_country = r.get("customer_country"),
            account_number   = r.get("account_number"),
            bank_name        = r.get("bank_name"),
            account_type     = r.get("account_type"),
            scored_at        = r.get("scored_at"),
            knn_anomaly_score= float(r["knn_score"]) if r.get("knn_score") is not None else None,
        )
        for r in rows
    ]

    total_pages = max(1, -(-total // page_size))
    return SuspectPageResponse(
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        suspects=suspects,
        model_trained=model.is_trained,
        training_stats=model._training_stats if model.is_trained else {},
    )


@router.get("/summary", dependencies=[Depends(require_admin)])
async def graphsage_summary():
    """Training statistics and detection summary."""
    from ml.graphsage import get_sage
    model = get_sage()

    with neo4j_session() as s:
        total_accts  = s.run("MATCH (a:Account) RETURN count(a) AS n").single()["n"]
        scored       = s.run(
            "MATCH (a:Account) WHERE a.graphsage_score IS NOT NULL RETURN count(a) AS n"
        ).single()["n"]
        suspects     = s.run(
            "MATCH (a:Account) WHERE a.graphsage_suspect=true RETURN count(a) AS n"
        ).single()["n"]
        high_conf    = s.run(
            "MATCH (a:Account) WHERE a.graphsage_score >= 80 RETURN count(a) AS n"
        ).single()["n"]
        # Accounts flagged by BOTH GraphSAGE and KNN
        both_flagged = s.run("""
            MATCH (a:Account)
            WHERE a.graphsage_suspect=true AND a.mule_suspect=true
            RETURN count(a) AS n
        """).single()["n"]

    return {
        "model_trained":    model.is_trained,
        "training_stats":   model._training_stats if model.is_trained else {},
        "total_accounts":   total_accts,
        "scored_accounts":  scored,
        "graphsage_suspects": suspects,
        "high_confidence":  high_conf,
        "flagged_by_both":  both_flagged,
        "coverage_pct":     round(scored / max(total_accts, 1) * 100, 1),
        "feature_flag_on":  settings.enable_graphsage,
    }


@router.get("/accounts/{account_id}", dependencies=[Depends(require_admin)])
async def get_account_graphsage(account_id: str):
    """GraphSAGE score for a single account."""
    from ml.graphsage import get_sage, get_graph_cache, build_account_graph, set_graph_cache
    model = get_sage()
    if not model.is_trained:
        raise HTTPException(503, "GraphSAGE model not trained. POST /graphsage/train first.")

    # Use cached graph if available, otherwise score from Neo4j properties
    with neo4j_session() as s:
        row = s.run("""
            MATCH (a:Account {id: $id})
            OPTIONAL MATCH (c:Customer)-[:OWNS]->(a)
            RETURN
              a.graphsage_score     AS score,
              a.graphsage_suspect   AS suspect,
              a.graphsage_scored_at AS scored_at,
              a.anomaly_score       AS knn_score,
              a.mule_suspect        AS knn_suspect,
              a.account_number      AS account_number,
              a.bank_name           AS bank_name,
              a.account_type        AS account_type,
              c.id   AS customer_id,
              c.name AS customer_name,
              c.country_of_residence AS country
        """, id=account_id).single()

    if row is None:
        raise HTTPException(404, "Account not found")

    return {
        "account_id":         account_id,
        "graphsage_score":    float(row.get("score") or 0),
        "is_suspect":         bool(row.get("suspect") or False),
        "scored_at":          row.get("scored_at"),
        "knn_anomaly_score":  float(row["knn_score"]) if row.get("knn_score") is not None else None,
        "knn_suspect":        bool(row.get("knn_suspect") or False),
        "flagged_by_both":    bool(row.get("suspect") and row.get("knn_suspect")),
        "account_number":     row.get("account_number"),
        "bank_name":          row.get("bank_name"),
        "account_type":       row.get("account_type"),
        "customer_id":        row.get("customer_id"),
        "customer_name":      row.get("customer_name"),
        "customer_country":   row.get("country"),
    }


@router.get("/comparison", dependencies=[Depends(require_admin)])
async def model_comparison(limit: int = Query(default=200, le=1000)):
    """Compare GraphSAGE vs KNN anomaly scores for top suspects."""
    with neo4j_session() as s:
        rows = s.run("""
            MATCH (a:Account)
            WHERE a.graphsage_score IS NOT NULL OR a.anomaly_score IS NOT NULL
            OPTIONAL MATCH (c:Customer)-[:OWNS]->(a)
            RETURN
              a.id                   AS account_id,
              a.account_number       AS account_number,
              coalesce(a.graphsage_score, 0) AS sage_score,
              coalesce(a.anomaly_score, 0)   AS knn_score,
              a.graphsage_suspect            AS sage_suspect,
              a.mule_suspect                 AS knn_suspect,
              c.name                         AS customer_name,
              c.country_of_residence         AS country
            ORDER BY (coalesce(a.graphsage_score,0) + coalesce(a.anomaly_score,0)) DESC
            LIMIT $limit
        """, limit=limit).data()

    agreement = sum(
        1 for r in rows
        if bool(r.get("sage_suspect")) == bool(r.get("knn_suspect"))
    )
    return {
        "accounts":  rows,
        "total":     len(rows),
        "agreement_pct": round(agreement / max(len(rows), 1) * 100, 1),
    }


# ── Training endpoints ─────────────────────────────────────────────────────────

@router.post("/train", response_model=TrainingStatus,
             dependencies=[Depends(require_admin)])
async def train_graphsage_model(
    background_tasks: BackgroundTasks,
    max_nodes: int = Query(default=50_000, le=200_000),
    max_edges: int = Query(default=500_000, le=2_000_000),
    epochs:    int = Query(default=60,     le=200),
):
    """
    Train (or retrain) the GraphSAGE mule-account detector.
    Runs in the background (~2–5 min depending on dataset size).
    """
    if not settings.enable_graphsage:
        raise HTTPException(
            400,
            "GraphSAGE is disabled (ENABLE_GRAPHSAGE=false). "
            "Set ENABLE_GRAPHSAGE=true to enable.",
        )
    from ml.graphsage import get_sage
    background_tasks.add_task(_bg_train, max_nodes, max_edges, epochs)
    model = get_sage()
    return TrainingStatus(
        status="started",
        message=f"GraphSAGE training started — up to {max_nodes:,} nodes × {epochs} epochs. "
                "Poll /graphsage/summary for progress.",
        model_trained=model.is_trained,
    )


def _bg_train(max_nodes: int, max_edges: int, epochs: int) -> None:
    from ml.graphsage import train_graphsage, reset_sage
    try:
        train_graphsage(max_nodes=max_nodes, max_edges=max_edges, epochs=epochs)
        reset_sage()   # force reload on next request
        print("[GraphSAGE] Background training complete.")
    except Exception as e:
        print(f"[GraphSAGE] Training failed: {e}")
        raise
