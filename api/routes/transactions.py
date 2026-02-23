"""Transaction management routes."""
from fastapi import APIRouter, HTTPException, Query, Depends
from db.client import neo4j_session
from auth.dependencies import require_viewer_or_admin

router = APIRouter(prefix="/transactions", tags=["Transactions"])

LIST_QUERY = """
MATCH (sender:Account)-[:INITIATED]->(t:Transaction)
OPTIONAL MATCH (t)-[:CREDITED_TO]->(receiver:Account)
RETURN t, sender.id AS sender_id, receiver.id AS receiver_id
ORDER BY t.timestamp DESC
SKIP $skip LIMIT $limit
"""

COUNT_QUERY = "MATCH ()-[:INITIATED]->(t:Transaction) RETURN count(t) AS total"

GET_QUERY = """
MATCH (sender:Account)-[:INITIATED]->(t:Transaction {id: $txn_id})
OPTIONAL MATCH (t)-[:CREDITED_TO]->(receiver:Account)
OPTIONAL MATCH (sender)<-[:OWNS]-(sc:Customer)
OPTIONAL MATCH (receiver)<-[:OWNS]-(rc:Customer)
OPTIONAL MATCH (t)-[:ORIGINATED_FROM]->(device:Device)
OPTIONAL MATCH (t)-[:SOURCED_FROM]->(ip:IPAddress)
OPTIONAL MATCH (t)-[:PAID_TO]->(merchant:Merchant)
OPTIONAL MATCH (t)-[:SENT_TO_EXTERNAL]->(beneficiary:BeneficiaryAccount)
RETURN t, sender, receiver, sc AS sender_customer, rc AS receiver_customer,
       device, ip, merchant, beneficiary
"""


@router.get("/stats/summary")
async def transaction_stats():
    """Aggregate statistics about the transaction dataset."""
    with neo4j_session() as session:
        stats = {}

        r = session.run("MATCH (t:Transaction) RETURN count(t) AS total, "
                        "sum(t.amount) AS total_volume, avg(t.amount) AS avg_amount, "
                        "max(t.amount) AS max_amount").single()
        stats["total_transactions"] = r["total"]
        stats["total_volume_usd"] = round(float(r["total_volume"] or 0), 2)
        stats["avg_amount_usd"] = round(float(r["avg_amount"] or 0), 2)
        stats["max_amount_usd"] = round(float(r["max_amount"] or 0), 2)

        r2 = session.run("MATCH (t:Transaction) WHERE t.is_fraud = true RETURN count(t) AS c").single()
        stats["fraud_count"] = r2["c"] if r2 else 0
        stats["fraud_rate_pct"] = round(
            stats["fraud_count"] / max(stats["total_transactions"], 1) * 100, 2
        )

        r3 = session.run("MATCH (t:Transaction) WHERE t.is_fraud = true "
                         "RETURN t.fraud_type AS type, count(*) AS c").data()
        stats["fraud_by_type"] = {row["type"]: row["c"] for row in r3}

        r4 = session.run("MATCH (n:Customer) RETURN count(n) AS c").single()
        stats["total_customers"] = r4["c"] if r4 else 0

        r5 = session.run("MATCH (n:Account) RETURN count(n) AS c").single()
        stats["total_accounts"] = r5["c"] if r5 else 0

    return stats


@router.get("/")
async def list_transactions(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    fraud_only: bool = Query(default=False),
):
    """List transactions stored in the graph (paginated)."""
    query = LIST_QUERY
    if fraud_only:
        query = query.replace("MATCH (sender:Account)-[:INITIATED]->(t:Transaction)",
                              "MATCH (sender:Account)-[:INITIATED]->(t:Transaction {is_fraud: true})")

    with neo4j_session() as session:
        result = session.run(query, skip=skip, limit=limit)
        records = [dict(r) for r in result]
        count_res = session.run(COUNT_QUERY).single()
        total = count_res["total"] if count_res else 0

    transactions = []
    for r in records:
        txn = dict(r["t"])
        txn["sender_id"] = r.get("sender_id")
        txn["receiver_id"] = r.get("receiver_id")
        transactions.append(txn)

    return {"total": total, "skip": skip, "limit": limit, "transactions": transactions}


@router.get("/{txn_id}")
async def get_transaction(txn_id: str):
    """Get full transaction details including graph context."""
    with neo4j_session() as session:
        result = session.run(GET_QUERY, txn_id=txn_id)
        record = result.single()

    if record is None:
        raise HTTPException(status_code=404, detail=f"Transaction {txn_id} not found")

    def _n(node):
        return dict(node) if node else None

    return {
        "transaction": _n(record["t"]),
        "sender_account": _n(record["sender"]),
        "receiver_account": _n(record["receiver"]),
        "sender_customer": _n(record["sender_customer"]),
        "receiver_customer": _n(record["receiver_customer"]),
        "device": _n(record["device"]),
        "ip_address": _n(record["ip"]),
        "merchant": _n(record["merchant"]),
        "beneficiary": _n(record["beneficiary"]),
    }


@router.get("/aggregate/world-map", dependencies=[Depends(require_viewer_or_admin)])
async def aggregate_world_map():
    """
    Aggregated transaction heatmap by country — no customer PII.
    Available to all authenticated users (admin + viewer).
    Returns per-country totals, fraud counts, risk distribution,
    and transaction volume for the global heatmap.
    """
    with neo4j_session() as session:
        rows = session.run("""
            MATCH (a:Account)-[:INITIATED]->(t:Transaction)
            OPTIONAL MATCH (a)-[:BASED_IN]->(sc:Country)
            OPTIONAL MATCH (t)-[:CREDITED_TO]->(ra:Account)-[:BASED_IN]->(rc:Country)
            OPTIONAL MATCH (t)-[:SENT_TO_EXTERNAL]->(b:BeneficiaryAccount)
            OPTIONAL MATCH (t)-[:HAS_PREDICTION]->(pl:PredictionLog)
            RETURN
                sc.code AS sender_cc, sc.name AS sender_name,
                sc.fatf_risk AS sender_fatf,
                rc.code AS recv_cc, rc.name AS recv_name,
                rc.fatf_risk AS recv_fatf,
                b.country AS bene_cc,
                t.is_fraud AS is_fraud, t.fraud_type AS fraud_type,
                t.amount AS amount, t.type AS txn_type,
                COALESCE(pl.final_score,
                    CASE t.is_fraud WHEN true THEN 720 ELSE null END) AS score,
                COALESCE(pl.outcome,
                    CASE t.is_fraud WHEN true THEN 'DECLINE' ELSE null END) AS outcome
        """).data()

    country_stats: dict[str, dict] = {}

    def _add(code, name, fatf, amount, score, is_fraud, fraud_type, txn_type, direction):
        if not code:
            return
        if code not in country_stats:
            country_stats[code] = {
                "code": code, "name": name or code,
                "fatf_risk": fatf or "LOW",
                "txn_count": 0, "fraud_count": 0,
                "total_amount": 0.0, "scores": [],
                "directions": set(), "fraud_types": set(),
                "txn_types": set(),
            }
        s = country_stats[code]
        s["txn_count"] += 1
        s["total_amount"] += float(amount or 0)
        if is_fraud:
            s["fraud_count"] += 1
            if fraud_type:
                s["fraud_types"].add(fraud_type)
        if score is not None:
            s["scores"].append(int(score))
        s["directions"].add(direction)
        if txn_type:
            s["txn_types"].add(txn_type)

    for r in rows:
        amt = r.get("amount", 0) or 0
        fraud = bool(r.get("is_fraud", False))
        ftype = r.get("fraud_type")
        score = r.get("score")
        ttype = r.get("txn_type")
        _add(r.get("sender_cc"), r.get("sender_name"), r.get("sender_fatf"),
             amt, score, fraud, ftype, ttype, "sender")
        _add(r.get("recv_cc"), r.get("recv_name"), r.get("recv_fatf"),
             amt, score, fraud, ftype, ttype, "receiver")
        if r.get("bene_cc"):
            _add(r["bene_cc"], r["bene_cc"], "HIGH",
                 amt, score, fraud, ftype, ttype, "beneficiary")

    result = []
    for s in country_stats.values():
        scores = s["scores"]
        avg_score = round(sum(scores) / len(scores)) if scores else None
        max_score = max(scores) if scores else None
        fraud_pct = s["fraud_count"] / s["txn_count"] if s["txn_count"] else 0
        if max_score is not None and max_score >= 700:
            risk_level = "CRITICAL"
        elif max_score is not None and max_score >= 400:
            risk_level = "HIGH"
        elif fraud_pct > 0:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        result.append({
            "code": s["code"], "name": s["name"],
            "fatf_risk": s["fatf_risk"],
            "txn_count": s["txn_count"],
            "fraud_count": s["fraud_count"],
            "total_amount": round(s["total_amount"], 2),
            "avg_score": avg_score, "max_score": max_score,
            "risk_level": risk_level,
            "directions": list(s["directions"]),
            "fraud_types": list(s["fraud_types"]),
            "txn_types": list(s["txn_types"]),
        })

    result.sort(key=lambda x: x["txn_count"], reverse=True)

    # Summary stats (no PII)
    total_txns = sum(r["txn_count"] for r in result)
    total_fraud = sum(r["fraud_count"] for r in result)
    return {
        "countries": result,
        "total_countries": len(result),
        "summary": {
            "total_transactions": total_txns,
            "total_fraud": total_fraud,
            "fraud_rate_pct": round(total_fraud / max(total_txns, 1) * 100, 1),
            "high_risk_countries": sum(1 for r in result if r["risk_level"] in ("HIGH", "CRITICAL")),
        },
    }


