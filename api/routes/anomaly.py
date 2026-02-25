"""
api/routes/anomaly.py
=====================
Endpoints for KNN-based mule-account anomaly detection.

All write/scan endpoints require admin role.
Read endpoints require admin role (they expose PII-adjacent data).
"""
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query, BackgroundTasks, HTTPException
from pydantic import BaseModel

from auth.dependencies import require_admin
from db.client import neo4j_session
from ml.anomaly import get_detector, reset_detector
from ml.anomaly import MuleAccountDetector as _MAD   # re-export for _train_and_scan

router = APIRouter(prefix="/anomaly", tags=["Anomaly Detection"])


# ── Response schemas ───────────────────────────────────────────────────────────

class AnomalyAccountResult(BaseModel):
    account_id: str
    anomaly_score: float
    knn_distance_score: float = 0.0
    rule_score: float = 0.0
    is_mule_suspect: bool
    indicators: list[str]
    pass_through_ratio: float = 0.0
    unique_senders_30d: int = 0
    structuring_30d: int = 0
    in_volume: float = 0.0
    out_volume: float = 0.0
    # enriched from graph
    customer_id: str | None = None
    customer_name: str | None = None
    customer_country: str | None = None
    account_number: str | None = None
    bank_name: str | None = None
    account_type: str | None = None
    scored_at: str | None = None


class SuspectListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    suspects: list[AnomalyAccountResult]
    detector_trained: bool


class ScanStatusResponse(BaseModel):
    status: str
    message: str
    detector_trained: bool


# ── Pre-scored suspects from Neo4j ────────────────────────────────────────────

SUSPECT_QUERY = """
MATCH (a:Account)
WHERE a.mule_suspect = true
OPTIONAL MATCH (c:Customer)-[:OWNS]->(a)
RETURN
  a.id              AS account_id,
  a.account_number  AS account_number,
  a.bank_name       AS bank_name,
  a.account_type    AS account_type,
  a.anomaly_score   AS anomaly_score,
  a.mule_indicators AS indicators,
  a.anomaly_scored_at AS scored_at,
  c.id              AS customer_id,
  c.name            AS customer_name,
  c.country_of_residence AS customer_country
ORDER BY a.anomaly_score DESC
SKIP $skip LIMIT $limit
"""

SUSPECT_COUNT_QUERY = "MATCH (a:Account) WHERE a.mule_suspect = true RETURN count(a) AS n"


@router.get("/suspects", response_model=SuspectListResponse, dependencies=[Depends(require_admin)])
async def list_mule_suspects(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
):
    """List all accounts flagged as mule suspects, ranked by anomaly score."""
    detector = get_detector()
    skip = (page - 1) * page_size

    with neo4j_session() as s:
        total = s.run(SUSPECT_COUNT_QUERY).single()["n"]
        rows  = s.run(SUSPECT_QUERY, skip=skip, limit=page_size).data()

    suspects = []
    for r in rows:
        indicators = r.get("indicators") or []
        if isinstance(indicators, str):
            indicators = [indicators]
        suspects.append(AnomalyAccountResult(
            account_id         = r["account_id"],
            anomaly_score      = float(r.get("anomaly_score") or 0),
            is_mule_suspect    = True,
            indicators         = list(indicators),
            customer_id        = r.get("customer_id"),
            customer_name      = r.get("customer_name"),
            customer_country   = r.get("customer_country"),
            account_number     = r.get("account_number"),
            bank_name          = r.get("bank_name"),
            account_type       = r.get("account_type"),
            scored_at          = r.get("scored_at"),
        ))

    total_pages = max(1, -(-total // page_size))
    return SuspectListResponse(
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        suspects=suspects,
        detector_trained=detector.is_trained,
    )


@router.get("/accounts/{account_id}", dependencies=[Depends(require_admin)])
async def get_account_anomaly(account_id: str):
    """Compute live anomaly score for a single account."""
    detector = get_detector()
    if not detector.is_trained:
        raise HTTPException(503, "Anomaly detector not trained yet. POST /anomaly/train first.")
    result = detector.score_account(account_id)
    if "error" in result:
        raise HTTPException(404, result["error"])

    # Enrich with graph data
    with neo4j_session() as s:
        meta = s.run("""
            MATCH (a:Account {id: $id})
            OPTIONAL MATCH (c:Customer)-[:OWNS]->(a)
            RETURN a.account_number AS account_number,
                   a.bank_name AS bank_name,
                   a.account_type AS account_type,
                   c.id AS customer_id, c.name AS customer_name,
                   c.country_of_residence AS customer_country
        """, id=account_id).single()

    if meta:
        result.update({
            "account_number":  meta["account_number"],
            "bank_name":       meta["bank_name"],
            "account_type":    meta["account_type"],
            "customer_id":     meta["customer_id"],
            "customer_name":   meta["customer_name"],
            "customer_country":meta["customer_country"],
        })
    return result


@router.get("/customers/{customer_id}", dependencies=[Depends(require_admin)])
async def get_customer_anomaly(customer_id: str):
    """Aggregate mule-account anomaly summary for a customer."""
    detector = get_detector()

    with neo4j_session() as s:
        accounts = [dict(r) for r in s.run("""
            MATCH (c:Customer {id: $cid})-[:OWNS]->(a:Account)
            RETURN a.id AS account_id,
                   a.account_number AS account_number,
                   a.anomaly_score AS anomaly_score,
                   a.mule_suspect AS mule_suspect,
                   a.mule_indicators AS indicators
        """, cid=customer_id)]

    if not accounts:
        raise HTTPException(404, "Customer not found or has no accounts")

    scored = [a for a in accounts if a.get("anomaly_score") is not None]
    max_score = max((a["anomaly_score"] for a in scored), default=0)
    any_suspect = any(a.get("mule_suspect") for a in accounts)

    # Live score if detector is trained and no pre-scored results
    live_results = []
    if detector.is_trained and not scored:
        for a in accounts:
            r = detector.score_account(a["account_id"])
            live_results.append(r)

    return {
        "customer_id": customer_id,
        "accounts": accounts,
        "live_scores": live_results,
        "max_anomaly_score": round(max_score, 1),
        "is_mule_suspect": any_suspect,
        "detector_trained": detector.is_trained,
    }


@router.get("/summary", dependencies=[Depends(require_admin)])
async def anomaly_summary():
    """High-level anomaly detection statistics."""
    detector = get_detector()
    with neo4j_session() as s:
        total_accounts = s.run("MATCH (a:Account) RETURN count(a) AS n").single()["n"]
        scored = s.run(
            "MATCH (a:Account) WHERE a.anomaly_score IS NOT NULL RETURN count(a) AS n"
        ).single()["n"]
        suspects = s.run(
            "MATCH (a:Account) WHERE a.mule_suspect = true RETURN count(a) AS n"
        ).single()["n"]
        high_risk = s.run(
            "MATCH (a:Account) WHERE a.anomaly_score >= 70 RETURN count(a) AS n"
        ).single()["n"]
        indicator_dist = s.run("""
            MATCH (a:Account) WHERE a.mule_indicators IS NOT NULL
            UNWIND a.mule_indicators AS ind
            RETURN ind AS indicator, count(*) AS freq ORDER BY freq DESC
        """).data()

    return {
        "total_accounts": total_accounts,
        "scored_accounts": scored,
        "mule_suspects": suspects,
        "high_risk_accounts": high_risk,
        "suspect_rate_pct": round(suspects / max(scored, 1) * 100, 1),
        "indicator_distribution": indicator_dist,
        "detector_trained": detector.is_trained,
        "coverage_pct": round(scored / max(total_accounts, 1) * 100, 1),
    }


# ── Training and scanning endpoints ───────────────────────────────────────────

@router.post("/train", dependencies=[Depends(require_admin)])
async def train_anomaly_detector(
    background_tasks: BackgroundTasks,
    max_normal: int = Query(
        default=5_000,
        ge=500,
        le=50_000,
        description="Max normal transactions to index (500–50 000). "
                    "Lower = faster; 5 000 is recommended.",
    ),
    max_accounts: int = Query(
        default=5_000,
        ge=100,
        le=50_000,
        description="Max accounts to scan after training.",
    ),
):
    """
    (Re)train the KNN anomaly detector on a sample of normal transactions,
    then scan a sample of accounts.  Runs in background.
    """
    background_tasks.add_task(_train_and_scan, max_normal, max_accounts)
    return {
        "status": "started",
        "message": (
            f"Anomaly detector training started "
            f"(max_normal={max_normal:,}, max_accounts={max_accounts:,}). "
            "Check /anomaly/summary for progress."
        ),
    }


@router.post("/scan", dependencies=[Depends(require_admin)])
async def scan_accounts(
    background_tasks: BackgroundTasks,
    force: bool = Query(default=False, description="Re-score already-scored accounts"),
    max_accounts: int = Query(
        default=5_000,
        ge=100,
        le=50_000,
        description="Max accounts to score in this run.",
    ),
):
    """
    Run anomaly scoring on accounts using the trained detector.
    Use force=true to re-score accounts that already have scores.
    """
    detector = get_detector()
    if not detector.is_trained:
        raise HTTPException(
            503,
            "Anomaly detector not trained yet. POST /anomaly/train first.",
        )
    background_tasks.add_task(_scan_accounts, force, max_accounts)
    return {
        "status": "started",
        "message": (
            f"Account scan started (force={force}, max={max_accounts:,}). "
            "Results available at /anomaly/suspects."
        ),
    }


def _train_and_scan(max_normal: int = 5_000, max_accounts: int = 5_000):
    """Background task: train detector on a sample, then scan a sample of accounts."""
    from ml.anomaly import MuleAccountDetector

    print(f"[Anomaly] Training detector on up to {max_normal:,} normal transactions…")
    detector = MuleAccountDetector()
    metrics  = detector.fit(max_normal=max_normal)
    detector.save()
    reset_detector()
    print(f"[Anomaly] Detector trained: {metrics}")

    print(f"[Anomaly] Scanning up to {max_accounts:,} accounts…")
    fresh = get_detector()
    suspects = fresh.scan_all_accounts(batch_size=100, force=True, max_accounts=max_accounts)
    print(f"[Anomaly] Scan complete. {len(suspects)} mule suspects found.")


def _scan_accounts(force: bool, max_accounts: int = 5_000):
    detector = get_detector()
    suspects = detector.scan_all_accounts(batch_size=100, force=force, max_accounts=max_accounts)
    print(f"[Anomaly] Scan complete. {len(suspects)} mule suspects found.")
