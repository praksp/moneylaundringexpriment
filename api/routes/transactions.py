"""Transaction management routes."""
from fastapi import APIRouter, HTTPException, Query, Depends
from db.client import neo4j_session, async_neo4j_session
from auth.dependencies import require_viewer_or_admin

router = APIRouter(prefix="/transactions", tags=["Transactions"])

LIST_QUERY = """
MATCH (sender:Account)-[:INITIATED]->(t:Transaction)
WHERE ($cursor IS NULL OR t.timestamp < $cursor)
OPTIONAL MATCH (t)-[:CREDITED_TO]->(receiver:Account)
RETURN t, sender.id AS sender_id, receiver.id AS receiver_id
ORDER BY t.timestamp DESC
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
    """Aggregate statistics about the transaction dataset. Cached for 2 minutes."""
    from api.cache import get_cached, set_cached
    cached = get_cached("txn_stats_summary")
    if cached is not None:
        return cached

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

    set_cached("txn_stats_summary", stats, ttl=120)
    return stats


@router.get("/")
async def list_transactions(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str = Query(default=None, description="Cursor timestamp for faster pagination"),
    fraud_only: bool = Query(default=False),
):
    """List transactions stored in the graph (paginated)."""
    query = LIST_QUERY
    if fraud_only:
        query = query.replace("MATCH (sender:Account)-[:INITIATED]->(t:Transaction)",
                              "MATCH (sender:Account)-[:INITIATED]->(t:Transaction {is_fraud: true})")

    # If using legacy 'skip', append it. Always append limit.
    if cursor is None and skip > 0:
        query += " SKIP $skip"
    query += " LIMIT $limit"

    async with async_neo4j_session() as session:
        result = await session.run(query, skip=skip, limit=limit, cursor=cursor)
        records = [dict(r) async for r in result]
        count_res_raw = await session.run(COUNT_QUERY)
        count_res = await count_res_raw.single()
        total = count_res["total"] if count_res else 0

    transactions = []
    next_cursor = None
    for r in records:
        txn = dict(r["t"])
        txn["sender_id"] = r.get("sender_id")
        txn["receiver_id"] = r.get("receiver_id")
        transactions.append(txn)
        next_cursor = txn.get("timestamp")

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "cursor": cursor,
        "next_cursor": next_cursor,
        "transactions": transactions
    }


@router.get("/{txn_id}")
async def get_transaction(txn_id: str):
    """Get full transaction details including graph context."""
    async with async_neo4j_session() as session:
        result = await session.run(GET_QUERY, txn_id=txn_id)
        record = await result.single()

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


def _build_world_map() -> dict:
    """
    Aggregate transaction data by country entirely in Neo4j.
    Returns ~71 rows (one per country) instead of streaming 961k rows to Python.
    Result is cached for 5 minutes.
    """
    from api.cache import get_cached, set_cached
    cached = get_cached("world_map")
    if cached is not None:
        return cached

    with neo4j_session() as session:
        # Sender-country aggregation
        sender_rows = session.run("""
            MATCH (a:Account)-[:BASED_IN]->(c:Country)
            MATCH (a)-[:INITIATED]->(t:Transaction)
            WITH c,
                 count(t)                                              AS txn_count,
                 sum(t.amount)                                         AS total_amount,
                 sum(CASE WHEN t.is_fraud = true THEN 1 ELSE 0 END)   AS fraud_count,
                 collect(DISTINCT t.fraud_type)                        AS fraud_types,
                 collect(DISTINCT t.transaction_type)                  AS txn_types
            RETURN c.code AS cc, c.name AS name, c.fatf_risk AS fatf,
                   txn_count, total_amount, fraud_count,
                   fraud_types, txn_types,
                   'sender' AS direction
        """).data()

        # Receiver-country aggregation
        recv_rows = session.run("""
            MATCH (ra:Account)-[:BASED_IN]->(c:Country)
            MATCH (t:Transaction)-[:CREDITED_TO]->(ra)
            WITH c,
                 count(t)                                              AS txn_count,
                 sum(t.amount)                                         AS total_amount,
                 sum(CASE WHEN t.is_fraud = true THEN 1 ELSE 0 END)   AS fraud_count,
                 collect(DISTINCT t.fraud_type)                        AS fraud_types,
                 collect(DISTINCT t.transaction_type)                  AS txn_types
            RETURN c.code AS cc, c.name AS name, c.fatf_risk AS fatf,
                   txn_count, total_amount, fraud_count,
                   fraud_types, txn_types,
                   'receiver' AS direction
        """).data()

        # Overall transaction stats (fast with indexed fields)
        stats_row = session.run("""
            MATCH (t:Transaction)
            RETURN count(t) AS total,
                   sum(CASE WHEN t.is_fraud = true THEN 1 ELSE 0 END) AS fraud_total
        """).single()

    # Merge into per-country buckets
    country_stats: dict[str, dict] = {}
    for row in sender_rows + recv_rows:
        cc = row.get("cc")
        if not cc:
            continue
        if cc not in country_stats:
            country_stats[cc] = {
                "code": cc,
                "name": row.get("name") or cc,
                "fatf_risk": row.get("fatf") or "LOW",
                "txn_count": 0, "fraud_count": 0,
                "total_amount": 0.0,
                "directions": set(),
                "fraud_types": set(),
                "txn_types": set(),
            }
        s = country_stats[cc]
        s["txn_count"]   += int(row.get("txn_count") or 0)
        s["fraud_count"] += int(row.get("fraud_count") or 0)
        s["total_amount"] += float(row.get("total_amount") or 0)
        s["directions"].add(row.get("direction", ""))
        for ft in (row.get("fraud_types") or []):
            if ft:
                s["fraud_types"].add(ft)
        for tt in (row.get("txn_types") or []):
            if tt:
                s["txn_types"].add(tt)

    result = []
    for s in country_stats.values():
        fraud_pct = s["fraud_count"] / max(s["txn_count"], 1)
        # Derive risk level from fraud % and FATF tier
        fatf = s.get("fatf_risk", "LOW")
        if fraud_pct >= 0.25 or fatf == "BLACKLIST":
            risk_level = "CRITICAL"
        elif fraud_pct >= 0.10 or fatf == "HIGH":
            risk_level = "HIGH"
        elif fraud_pct > 0 or fatf == "MEDIUM":
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        result.append({
            "code": s["code"],
            "name": s["name"],
            "fatf_risk": s["fatf_risk"],
            "txn_count": s["txn_count"],
            "fraud_count": s["fraud_count"],
            "total_amount": round(s["total_amount"], 2),
            "risk_level": risk_level,
            "directions": list(s["directions"]),
            "fraud_types": list(s["fraud_types"]),
            "txn_types": list(s["txn_types"]),
        })

    result.sort(key=lambda x: x["txn_count"], reverse=True)

    total_txns  = int(stats_row["total"] or 0) if stats_row else 0
    total_fraud = int(stats_row["fraud_total"] or 0) if stats_row else 0

    payload = {
        "countries": result,
        "total_countries": len(result),
        "summary": {
            "total_transactions": total_txns,
            "total_fraud": total_fraud,
            "fraud_rate_pct": round(total_fraud / max(total_txns, 1) * 100, 1),
            "high_risk_countries": sum(1 for r in result if r["risk_level"] in ("HIGH", "CRITICAL")),
        },
    }
    set_cached("world_map", payload, ttl=300)   # 5-minute cache
    return payload


@router.get("/aggregate/world-map", dependencies=[Depends(require_viewer_or_admin)])
async def aggregate_world_map():
    """
    Aggregated transaction heatmap by country — no customer PII.
    Available to all authenticated users (admin + viewer).
    Aggregation is done inside Neo4j (returns ~71 rows, not 961k).
    Result is cached for 5 minutes.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _build_world_map)


