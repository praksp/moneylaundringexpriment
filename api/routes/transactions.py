"""Transaction management routes."""
from fastapi import APIRouter, HTTPException, Query
from db.client import neo4j_session

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


