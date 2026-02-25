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
    """
    GraphSAGE score for a single account with full feature explanation
    and list of fraud / high-risk transactions that triggered the flag.
    """
    from ml.graphsage import get_sage
    model = get_sage()
    if not model.is_trained:
        raise HTTPException(503, "GraphSAGE model not trained. POST /graphsage/train first.")

    with neo4j_session() as s:
        # ── Core account + score row ──────────────────────────────────────────
        row = s.run("""
            MATCH (a:Account {id: $id})
            OPTIONAL MATCH (c:Customer)-[:OWNS]->(a)
            RETURN
              a.graphsage_score        AS score,
              a.graphsage_suspect      AS suspect,
              a.graphsage_scored_at    AS scored_at,
              a.anomaly_score          AS knn_score,
              a.mule_suspect           AS knn_suspect,
              a.account_number         AS account_number,
              a.bank_name              AS bank_name,
              a.account_type           AS account_type,
              a.country                AS account_country,
              c.id                     AS customer_id,
              c.name                   AS customer_name,
              c.country_of_residence   AS country,
              c.pep_flag               AS pep_flag,
              c.sanctions_flag         AS sanctions_flag,
              c.kyc_level              AS kyc_level,
              c.risk_tier              AS risk_tier
        """, id=account_id).single()

        if row is None:
            raise HTTPException(404, "Account not found")

        # ── Feature vector used by GraphSAGE (all 20 features) ───────────────
        feat_row = s.run("""
            MATCH (a:Account {id: $id})
            OPTIONAL MATCH (a)-[:INITIATED]->(t_out:Transaction)-[:CREDITED_TO]->(recv:Account)
            WITH a,
              count(DISTINCT t_out)                                              AS out_count,
              coalesce(sum(t_out.amount_usd), 0)                                AS out_vol,
              coalesce(avg(t_out.amount_usd), 0)                                AS avg_out,
              coalesce(sum(CASE WHEN t_out.is_fraud THEN 1 ELSE 0 END), 0)     AS fraud_count,
              coalesce(sum(CASE WHEN t_out.fraud_type IN
                ['SMURFING','LAYERING','STRUCTURING','ROUND_TRIP',
                 'DORMANT_BURST','HIGH_RISK_CORRIDOR','RAPID_VELOCITY']
                THEN 1 ELSE 0 END), 0)                                          AS pattern_count,
              count(DISTINCT recv)                                               AS unique_receivers
            OPTIONAL MATCH (sender:Account)-[:INITIATED]->(t_in:Transaction)-[:CREDITED_TO]->(a)
            WITH a, out_count, out_vol, avg_out, fraud_count, pattern_count, unique_receivers,
              count(DISTINCT t_in)                                              AS in_count,
              coalesce(sum(t_in.amount_usd), 0)                                AS in_vol,
              count(DISTINCT sender)                                            AS unique_senders
            RETURN out_count, out_vol, in_count, in_vol, avg_out,
                   fraud_count, pattern_count, unique_senders, unique_receivers,
                   CASE WHEN out_count > 0 THEN toFloat(fraud_count) / out_count ELSE 0 END
                       AS fraud_ratio,
                   CASE WHEN in_vol > 100 THEN out_vol / in_vol ELSE 0 END
                       AS pass_through_ratio,
                   // Fan-in diversity: near 1 = every sender is new (smurfing signal)
                   CASE WHEN in_count > 0 THEN toFloat(unique_senders) / in_count ELSE 0 END
                       AS sender_diversity_ratio,
                   // Funnel concentration: 1 - recv/out = high when sending to few recipients
                   CASE WHEN out_count > 0 THEN 1.0 - toFloat(unique_receivers) / out_count ELSE 0 END
                       AS funnel_concentration,
                   // Net retention: how much of inflow stays (mule keeps near 0)
                   CASE WHEN in_vol > 0 THEN (in_vol - out_vol) / in_vol ELSE 1 END
                       AS net_retention_ratio
        """, id=account_id).single()

        # ── Fraud & high-risk transactions ────────────────────────────────────
        txn_rows = s.run("""
            MATCH (a:Account {id: $id})-[:INITIATED]->(t:Transaction)-[:CREDITED_TO]->(recv:Account)
            OPTIONAL MATCH (recv_c:Customer)-[:OWNS]->(recv)
            WHERE t.is_fraud = true
               OR t.fraud_type IS NOT NULL
            RETURN
              t.id          AS txn_id,
              t.amount_usd  AS amount,
              t.currency    AS currency,
              t.timestamp   AS ts,
              t.is_fraud    AS is_fraud,
              t.fraud_type  AS fraud_type,
              t.outcome     AS outcome,
              recv.account_number AS recv_account,
              recv.country        AS recv_country,
              recv_c.name         AS recv_name
            ORDER BY t.timestamp DESC
            LIMIT 50
        """, id=account_id).data()

        # ── Incoming (received) suspicious transactions ────────────────────────
        in_txn_rows = s.run("""
            MATCH (sender:Account)-[:INITIATED]->(t:Transaction)-[:CREDITED_TO]->(a:Account {id: $id})
            OPTIONAL MATCH (sender_c:Customer)-[:OWNS]->(sender)
            WHERE t.is_fraud = true OR t.fraud_type IS NOT NULL
            RETURN
              t.id          AS txn_id,
              t.amount_usd  AS amount,
              t.currency    AS currency,
              t.timestamp   AS ts,
              t.is_fraud    AS is_fraud,
              t.fraud_type  AS fraud_type,
              sender.account_number AS sender_account,
              sender.country        AS sender_country,
              sender_c.name         AS sender_name,
              'INBOUND'             AS direction
            ORDER BY t.timestamp DESC
            LIMIT 20
        """, id=account_id).data()

    # ── Build feature explanation ──────────────────────────────────────────────
    fv = feat_row or {}
    out_count        = int(fv.get("out_count")  or 0)
    in_count         = int(fv.get("in_count")   or 0)
    out_vol          = float(fv.get("out_vol")  or 0)
    in_vol           = float(fv.get("in_vol")   or 0)
    fraud_count      = int(fv.get("fraud_count") or 0)
    pattern_count    = int(fv.get("pattern_count") or 0)
    fraud_ratio      = float(fv.get("fraud_ratio") or 0)
    pass_thru        = float(fv.get("pass_through_ratio") or 0)
    avg_out          = float(fv.get("avg_out") or 0)
    unique_senders   = int(fv.get("unique_senders") or 0)
    unique_receivers = int(fv.get("unique_receivers") or 0)
    sender_div       = float(fv.get("sender_diversity_ratio") or 0)
    funnel_conc      = float(fv.get("funnel_concentration") or 0)
    net_retention    = float(fv.get("net_retention_ratio") or 1.0)

    _COUNTRY_RISK = {
        "IR": 1.0, "KP": 1.0, "SY": 1.0, "RU": 0.8, "MM": 0.8, "VE": 0.8,
        "NG": 0.7, "PK": 0.7, "CN": 0.3, "IN": 0.3, "AE": 0.4, "SA": 0.4,
        "TR": 0.4, "ZA": 0.4, "BR": 0.4, "MX": 0.4, "KY": 0.6, "PA": 0.5,
    }
    country_risk = _COUNTRY_RISK.get(str(row.get("account_country") or ""), 0.0)

    # Structural mule check (mirrors the training label)
    structural_mule = (unique_senders >= 5 and 0.7 <= pass_thru <= 1.5 and in_vol > 10_000)

    # Pattern names — for UI badge display
    triggered_patterns: list[str] = []
    if fraud_ratio >= 0.30:         triggered_patterns.append("HIGH_FRAUD_RATIO")
    if pattern_count >= 2:          triggered_patterns.append("FRAUD_PATTERN_MATCH")
    if structural_mule:             triggered_patterns.append("STRUCTURAL_RELAY")
    if unique_senders >= 5:         triggered_patterns.append("FAN_IN_COLLECTION")
    if 0.7 <= pass_thru <= 1.5 and in_vol > 1000:
                                    triggered_patterns.append("PASS_THROUGH_RELAY")
    if funnel_conc >= 0.7 and unique_senders >= 5:
                                    triggered_patterns.append("FUNNEL_AGGREGATION")
    if net_retention < 0.15 and in_vol > 5000:
                                    triggered_patterns.append("ZERO_RETENTION")

    # Each feature: name, value, threshold, triggered (bool), weight %, description
    # Ordered by GraphSAGE feature importance (empirical from training weights norm)
    features = [
        # ── Explicit fraud signals ──────────────────────────────────────────────
        {
            "name": "Fraud Transaction Ratio",
            "pattern": "HIGH_FRAUD_RATIO",
            "value": round(fraud_ratio * 100, 1),
            "unit": "%",
            "threshold": 30.0,
            "triggered": fraud_ratio >= 0.30,
            "weight_pct": 22,
            "description": (
                f"{fraud_count} of {out_count} outbound transactions are explicitly "
                f"marked fraud ({fraud_ratio*100:.1f}%). Threshold: ≥30%."
            ),
        },
        {
            "name": "Fraud Pattern Count",
            "pattern": "FRAUD_PATTERN_MATCH",
            "value": pattern_count,
            "unit": "txns",
            "threshold": 2,
            "triggered": pattern_count >= 2,
            "weight_pct": 18,
            "description": (
                "Transactions tagged with a fraud typology: SMURFING, LAYERING, "
                "STRUCTURING, ROUND_TRIP, DORMANT_BURST, HIGH_RISK_CORRIDOR, or "
                "RAPID_VELOCITY. Two or more confirms pattern membership."
            ),
        },
        # ── NEW mule-structural signals ─────────────────────────────────────────
        {
            "name": "Fan-in Sender Diversity",
            "pattern": "FAN_IN_COLLECTION",
            "value": unique_senders,
            "unit": " senders",
            "threshold": 5,
            "triggered": unique_senders >= 5,
            "weight_pct": 18,
            "description": (
                f"{unique_senders} distinct accounts have sent money INTO this account. "
                "A legitimate account (salary, vendor payments) has 1–3 known senders. "
                "≥5 unrelated senders is the hallmark of a SMURFING COLLECTION POINT — "
                "many participants each deposit small amounts to avoid detection thresholds."
            ),
        },
        {
            "name": "Pass-Through Relay Ratio",
            "pattern": "PASS_THROUGH_RELAY",
            "value": round(pass_thru, 2),
            "unit": "×",
            "threshold": "0.7–1.5",
            "triggered": 0.7 <= pass_thru <= 1.5 and in_vol > 1_000,
            "weight_pct": 16,
            "description": (
                f"Outflow ÷ Inflow = {pass_thru:.2f}×. A ratio near 1.0 means the account "
                "receives money and forwards almost all of it — it keeps no funds. "
                "This is the defining characteristic of a RELAY/MULE ACCOUNT. "
                "Normal spending accounts retain 60–80% of inflows."
            ),
        },
        {
            "name": "Funnel Concentration",
            "pattern": "FUNNEL_AGGREGATION",
            "value": round(funnel_conc * 100, 1),
            "unit": "%",
            "threshold": 70.0,
            "triggered": funnel_conc >= 0.7 and unique_senders >= 5,
            "weight_pct": 12,
            "description": (
                f"Sends to only {unique_receivers} distinct recipients across {out_count} "
                f"outbound transactions ({round(funnel_conc*100,1)}% concentration). "
                "Many-in + few-out is the FUNNEL AGGREGATION pattern: the mule collects "
                "scattered criminal proceeds and channels them to a controller account."
            ),
        },
        {
            "name": "Net Balance Retention",
            "pattern": "ZERO_RETENTION",
            "value": round(net_retention * 100, 1),
            "unit": "%",
            "threshold": 15.0,
            "triggered": net_retention < 0.15 and in_vol > 5_000,
            "weight_pct": 8,
            "description": (
                f"Only {round(net_retention*100,1)}% of inbound funds are retained. "
                "A mule account acts as a financial wire — money flows straight through "
                "with negligible balance accumulation. Legitimate accounts accumulate savings."
            ),
        },
        # ── Identity / jurisdiction signals ─────────────────────────────────────
        {
            "name": "Outbound Volume",
            "pattern": "HIGH_VOLUME",
            "value": round(out_vol, 0),
            "unit": " USD",
            "threshold": 100_000,
            "triggered": out_vol > 100_000,
            "weight_pct": 4,
            "description": f"${out_vol:,.0f} total outbound  |  ${avg_out:,.0f} avg per transaction.",
        },
        {
            "name": "Country Risk",
            "pattern": "HIGH_RISK_JURISDICTION",
            "value": round(country_risk * 100, 0),
            "unit": "/100",
            "threshold": 30.0,
            "triggered": country_risk >= 0.3,
            "weight_pct": 1,
            "description": (
                f"Account registered in {row.get('account_country') or 'unknown'}. "
                "Jurisdictions with weak AML controls, FATF grey/blacklist status, "
                "or known financial secrecy laws increase the risk score."
            ),
        },
        {
            "name": "PEP / Sanctions Flag",
            "pattern": "PEP_SANCTIONS",
            "value": "YES" if (row.get("pep_flag") or row.get("sanctions_flag")) else "NO",
            "unit": "",
            "threshold": "YES",
            "triggered": bool(row.get("pep_flag") or row.get("sanctions_flag")),
            "weight_pct": 1,
            "description": "Politically Exposed Person or sanctions watchlist match on the account owner.",
        },
    ]

    # Combine outbound + inbound fraud txns
    all_txns = [
        {
            "txn_id":       t["txn_id"],
            "direction":    "OUTBOUND",
            "amount":       float(t.get("amount") or 0),
            "currency":     t.get("currency") or "USD",
            "timestamp":    t.get("ts"),
            "is_fraud":     bool(t.get("is_fraud")),
            "fraud_type":   t.get("fraud_type"),
            "outcome":      t.get("outcome"),
            "counterparty": t.get("recv_name") or t.get("recv_account") or "Unknown",
            "country":      t.get("recv_country"),
        }
        for t in txn_rows
    ] + [
        {
            "txn_id":       t["txn_id"],
            "direction":    "INBOUND",
            "amount":       float(t.get("amount") or 0),
            "currency":     t.get("currency") or "USD",
            "timestamp":    t.get("ts"),
            "is_fraud":     bool(t.get("is_fraud")),
            "fraud_type":   t.get("fraud_type"),
            "outcome":      None,
            "counterparty": t.get("sender_name") or t.get("sender_account") or "Unknown",
            "country":      t.get("sender_country"),
        }
        for t in in_txn_rows
    ]
    all_txns.sort(key=lambda x: str(x.get("timestamp") or ""), reverse=True)

    triggered_features = [f for f in features if f["triggered"]]

    # Explain WHY this account was labelled a mule during training
    if fraud_ratio >= 0.30:
        mule_label_reason = f"≥30% fraud transaction ratio ({fraud_ratio*100:.1f}%)"
    elif pattern_count >= 2:
        mule_label_reason = f"≥2 fraud pattern transactions ({pattern_count} detected)"
    elif structural_mule:
        mule_label_reason = (
            f"Structural relay signature: {unique_senders} senders, "
            f"{pass_thru:.2f}× pass-through, ${in_vol:,.0f} inflow"
        )
    else:
        mule_label_reason = "GraphSAGE neighbourhood propagation (neighbours flagged)"

    sage_score = float(row.get("score") or 0)
    return {
        "account_id":         account_id,
        "graphsage_score":    sage_score,
        "is_suspect":         bool(row.get("suspect") or False),
        "scored_at":          row.get("scored_at"),
        "knn_anomaly_score":  float(row["knn_score"]) if row.get("knn_score") is not None else None,
        "knn_suspect":        bool(row.get("knn_suspect") or False),
        "flagged_by_both":    bool(row.get("suspect") and row.get("knn_suspect")),
        "mule_label_reason":  mule_label_reason,
        # Account info
        "account_number":     row.get("account_number"),
        "bank_name":          row.get("bank_name"),
        "account_type":       row.get("account_type"),
        "account_country":    row.get("account_country"),
        "customer_id":        row.get("customer_id"),
        "customer_name":      row.get("customer_name"),
        "customer_country":   row.get("country"),
        "pep_flag":           bool(row.get("pep_flag") or False),
        "sanctions_flag":     bool(row.get("sanctions_flag") or False),
        "kyc_level":          row.get("kyc_level"),
        "risk_tier":          row.get("risk_tier"),
        # Feature explanation
        "features":           features,
        "triggered_count":    len(triggered_features),
        "feature_summary": {
            "fraud_ratio_pct":      round(fraud_ratio * 100, 1),
            "fraud_count":          fraud_count,
            "pattern_count":        pattern_count,
            "pass_through":         round(pass_thru, 2),
            "unique_senders":       unique_senders,
            "unique_receivers":     unique_receivers,
            "sender_diversity":     round(sender_div, 2),
            "funnel_concentration": round(funnel_conc, 2),
            "net_retention_pct":    round(net_retention * 100, 1),
            "out_volume_usd":       round(out_vol, 0),
            "in_volume_usd":        round(in_vol, 0),
            "out_txn_count":        out_count,
            "in_txn_count":         in_count,
            "triggered_patterns":   triggered_patterns,
            "is_structural_mule":   structural_mule,
        },
        # Suspicious transactions
        "fraud_transactions":      all_txns,
        "fraud_txn_count":         len(all_txns),
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
