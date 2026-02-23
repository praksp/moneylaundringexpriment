"""
Customer Profile Model
=======================
Builds a comprehensive, real-time profile for a customer by aggregating:
  - Identity & KYC details
  - All accounts and their health
  - Transaction history summary (volume, velocity, fraud incidents)
  - Risk score history
  - Network connections (shared devices, IPs, counterparties)
  - Mule account indicators
  - Feature store snapshot

Designed to support compliance officers, fraud analysts, and the
risk engine — all from the graph database.
"""
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
from db.client import neo4j_session

# ── Cypher queries ──────────────────────────────────────────────────────────

CUSTOMER_BASE_QUERY = """
MATCH (c:Customer {id: $customer_id})
OPTIONAL MATCH (c)-[:RESIDENT_OF]->(country:Country)
RETURN c, country
"""

CUSTOMER_ACCOUNTS_QUERY = """
MATCH (c:Customer {id: $customer_id})-[:OWNS]->(a:Account)
OPTIONAL MATCH (a)-[:BASED_IN]->(country:Country)
OPTIONAL MATCH (a)<-[:FEATURE_SNAPSHOT_FOR]-(fs:FeatureSnapshot)
RETURN a, country, fs ORDER BY a.created_at
"""

TRANSACTION_HISTORY_QUERY = """
MATCH (c:Customer {id: $customer_id})-[:OWNS]->(a:Account)
MATCH (a)-[:INITIATED]->(t:Transaction)
OPTIONAL MATCH (t)-[:CREDITED_TO]->(recv:Account)
OPTIONAL MATCH (recv)<-[:OWNS]-(recv_c:Customer)
OPTIONAL MATCH (t)-[:HAS_PREDICTION]->(pl:PredictionLog)
RETURN t, a.id AS sender_account_id, recv.id AS receiver_account_id,
       recv_c.name AS receiver_name, pl
ORDER BY t.timestamp DESC
LIMIT $limit
"""

TXN_COUNT_QUERY = """
MATCH (c:Customer {id: $customer_id})-[:OWNS]->(a:Account)
MATCH (a)-[:INITIATED]->(t:Transaction)
RETURN count(t) AS total
"""

TRANSACTION_HISTORY_PAGINATED_QUERY = """
MATCH (c:Customer {id: $customer_id})-[:OWNS]->(a:Account)
MATCH (a)-[:INITIATED]->(t:Transaction)
OPTIONAL MATCH (t)-[:CREDITED_TO]->(recv:Account)
OPTIONAL MATCH (recv)<-[:OWNS]-(recv_c:Customer)
OPTIONAL MATCH (t)-[:HAS_PREDICTION]->(pl:PredictionLog)
RETURN t, a.id AS sender_account_id, recv.id AS receiver_account_id,
       recv_c.name AS receiver_name, pl
ORDER BY t.timestamp DESC
SKIP $skip
LIMIT $limit
"""

RISK_HISTORY_QUERY = """
MATCH (c:Customer {id: $customer_id})-[:OWNS]->(a:Account)
MATCH (a)-[:INITIATED]->(t:Transaction)
MATCH (pl:PredictionLog {transaction_id: t.id})
WHERE pl.timestamp >= $since
RETURN pl.final_score AS score, pl.outcome AS outcome,
       pl.timestamp AS ts, t.amount AS amount
ORDER BY pl.timestamp DESC
"""

NETWORK_CONNECTIONS_QUERY = """
MATCH (c:Customer {id: $customer_id})-[:OWNS]->(a:Account)
OPTIONAL MATCH (a)-[:INITIATED]->(t:Transaction)-[:ORIGINATED_FROM]->(d:Device)
  <-[:ORIGINATED_FROM]-(t2:Transaction)<-[:INITIATED]-(a2:Account)
  <-[:OWNS]-(c2:Customer)
WHERE c2.id <> $customer_id
WITH c2, collect(DISTINCT d.id) AS shared_devices
OPTIONAL MATCH (a)-[:INITIATED]->(t3:Transaction)-[:CREDITED_TO]->(a3:Account)
  <-[:OWNS]-(c3:Customer)
WHERE c3.id <> $customer_id
WITH c2, shared_devices, c3,
     count(DISTINCT t3) AS transfer_count
RETURN c2.id AS connected_customer_id,
       c2.name AS connected_customer_name,
       c2.risk_tier AS risk_tier,
       c2.pep_flag AS is_pep,
       shared_devices,
       'SHARED_DEVICE' AS connection_type
LIMIT 20
UNION
RETURN c3.id AS connected_customer_id,
       c3.name AS connected_customer_name,
       c3.risk_tier AS risk_tier,
       c3.pep_flag AS is_pep,
       [] AS shared_devices,
       'TRANSFER_COUNTERPARTY' AS connection_type
LIMIT 20
"""

VELOCITY_SUMMARY_QUERY = """
MATCH (c:Customer {id: $customer_id})-[:OWNS]->(a:Account)
MATCH (a)-[:INITIATED]->(t:Transaction)
WHERE t.timestamp >= $since_7d
WITH
  count(t) AS txn_7d,
  sum(t.amount) AS vol_7d,
  count(CASE WHEN t.timestamp >= $since_24h THEN 1 END) AS txn_24h,
  sum(CASE WHEN t.timestamp >= $since_24h THEN t.amount ELSE 0 END) AS vol_24h,
  count(CASE WHEN t.is_fraud THEN 1 END) AS fraud_count,
  count(CASE WHEN t.amount >= 9000 AND t.amount < 10000 THEN 1 END) AS structuring_count,
  avg(t.amount) AS avg_amount,
  max(t.amount) AS max_amount
RETURN txn_7d, vol_7d, txn_24h, vol_24h, fraud_count,
       structuring_count, avg_amount, max_amount
"""


def _node(record_value) -> dict:
    if record_value is None:
        return {}
    return dict(record_value)


# ── Profile dataclasses ─────────────────────────────────────────────────────

@dataclass
class AccountSummary:
    id: str
    account_number: str
    account_type: str
    currency: str
    balance: float
    country: str
    bank_name: str
    status: str
    created_at: str
    last_active: str
    is_dormant: bool
    feature_snapshot: Optional[dict] = None


@dataclass
class TransactionSummary:
    id: str
    reference: str
    amount: float
    currency: str
    transaction_type: str
    channel: str
    timestamp: str
    is_fraud: bool
    fraud_type: Optional[str]
    sender_account_id: str
    receiver_account_id: Optional[str]
    receiver_name: Optional[str]
    risk_score: Optional[int]
    outcome: Optional[str]
    risk_factors: list = field(default_factory=list)


@dataclass
class RiskProfile:
    total_evaluations: int
    avg_score_30d: float
    max_score_30d: int
    allow_count: int
    challenge_count: int
    decline_count: int
    fraud_incident_count: int
    current_risk_tier: str
    risk_trend: str  # STABLE / RISING / FALLING


@dataclass
class NetworkConnection:
    customer_id: str
    customer_name: str
    risk_tier: str
    is_pep: bool
    connection_type: str
    shared_devices: list[str]


@dataclass
class CustomerProfile:
    customer_id: str
    name: str
    customer_type: str
    nationality: str
    country_of_residence: str
    kyc_level: str
    pep_flag: bool
    sanctions_flag: bool
    risk_tier: str
    created_at: str
    accounts: list[AccountSummary]
    recent_transactions: list[TransactionSummary]
    risk_profile: RiskProfile
    network_connections: list[NetworkConnection]
    velocity: dict
    mule_indicators: dict
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Builder ─────────────────────────────────────────────────────────────────

def build_customer_profile(customer_id: str, txn_limit: int = 30) -> Optional[CustomerProfile]:
    """
    Fetch and assemble a full CustomerProfile from Neo4j.
    Returns None if the customer doesn't exist.
    """
    now = datetime.utcnow()
    since_24h = (now - timedelta(hours=24)).isoformat()
    since_7d = (now - timedelta(days=7)).isoformat()
    since_30d = (now - timedelta(days=30)).isoformat()

    with neo4j_session() as session:
        # ── Base customer info ──────────────────────────────────
        base_rec = session.run(CUSTOMER_BASE_QUERY, customer_id=customer_id).single()
        if base_rec is None:
            return None
        customer = _node(base_rec["c"])

        # ── Accounts ────────────────────────────────────────────
        acct_records = list(session.run(CUSTOMER_ACCOUNTS_QUERY, customer_id=customer_id))
        accounts = []
        for r in acct_records:
            a = _node(r["a"])
            fs = _node(r["fs"]) if r["fs"] else None
            last_active_str = str(a.get("last_active", ""))
            try:
                last_active_dt = datetime.fromisoformat(last_active_str.replace("Z", ""))
                is_dormant = (now - last_active_dt).days > 90
            except Exception:
                is_dormant = False
            accounts.append(AccountSummary(
                id=a.get("id", ""),
                account_number=a.get("account_number", ""),
                account_type=a.get("account_type", ""),
                currency=a.get("currency", "USD"),
                balance=float(a.get("balance", 0)),
                country=a.get("country", ""),
                bank_name=a.get("bank_name", ""),
                status=a.get("status", ""),
                created_at=str(a.get("created_at", "")),
                last_active=last_active_str,
                is_dormant=is_dormant,
                feature_snapshot=fs,
            ))

        # ── Transaction history ──────────────────────────────────
        txn_records = list(session.run(
            TRANSACTION_HISTORY_QUERY,
            customer_id=customer_id, limit=txn_limit
        ))
        transactions = _build_transactions_from_records(txn_records)

        # ── Risk history — derived from transactions (incl. test-data fraud labels) ──
        scores = [t.risk_score for t in transactions if t.risk_score is not None]
        outcomes = [t.outcome for t in transactions if t.outcome]
        risk_profile = RiskProfile(
            total_evaluations=len(scores),
            avg_score_30d=round(sum(scores) / len(scores), 1) if scores else 0.0,
            max_score_30d=max(scores) if scores else 0,
            allow_count=outcomes.count("ALLOW"),
            challenge_count=outcomes.count("CHALLENGE"),
            decline_count=outcomes.count("DECLINE"),
            fraud_incident_count=sum(1 for t in transactions if t.is_fraud),
            current_risk_tier=customer.get("risk_tier", "LOW"),
            risk_trend=_compute_risk_trend(scores),
        )

        # ── Network connections ───────────────────────────────────
        # Simplified query to avoid UNION issues in all Neo4j versions
        net_records = list(session.run("""
            MATCH (c:Customer {id: $customer_id})-[:OWNS]->(a:Account)
            MATCH (a)-[:INITIATED]->(t:Transaction)-[:ORIGINATED_FROM]->(d:Device)
                <-[:ORIGINATED_FROM]-(t2:Transaction)<-[:INITIATED]-(a2:Account)
                <-[:OWNS]-(c2:Customer)
            WHERE c2.id <> $customer_id
            RETURN c2.id AS cid, c2.name AS cname, c2.risk_tier AS rt,
                   c2.pep_flag AS pep, collect(DISTINCT d.id) AS devs,
                   'SHARED_DEVICE' AS conn_type
            LIMIT 15
        """, customer_id=customer_id))
        connections = [
            NetworkConnection(
                customer_id=r["cid"], customer_name=r["cname"],
                risk_tier=r["rt"] or "LOW", is_pep=bool(r["pep"]),
                connection_type=r["conn_type"],
                shared_devices=list(r["devs"]),
            ) for r in net_records
        ]

        # ── Velocity ──────────────────────────────────────────────
        vel_rec = session.run(
            VELOCITY_SUMMARY_QUERY,
            customer_id=customer_id,
            since_7d=since_7d, since_24h=since_24h
        ).single()
        velocity = {}
        if vel_rec:
            velocity = {
                "txn_count_24h": int(vel_rec["txn_24h"] or 0),
                "txn_count_7d": int(vel_rec["txn_7d"] or 0),
                "volume_24h_usd": round(float(vel_rec["vol_24h"] or 0), 2),
                "volume_7d_usd": round(float(vel_rec["vol_7d"] or 0), 2),
                "fraud_incident_count": int(vel_rec["fraud_count"] or 0),
                "structuring_count": int(vel_rec["structuring_count"] or 0),
                "avg_transaction_usd": round(float(vel_rec["avg_amount"] or 0), 2),
                "max_transaction_usd": round(float(vel_rec["max_amount"] or 0), 2),
            }

        # ── Mule indicators ───────────────────────────────────────
        mule_indicators = _compute_mule_indicators(customer_id, accounts, velocity, risk_profile, session)

    return CustomerProfile(
        customer_id=customer_id,
        name=customer.get("name", ""),
        customer_type=customer.get("customer_type", "INDIVIDUAL"),
        nationality=customer.get("nationality", ""),
        country_of_residence=customer.get("country_of_residence", ""),
        kyc_level=customer.get("kyc_level", "BASIC"),
        pep_flag=bool(customer.get("pep_flag", False)),
        sanctions_flag=bool(customer.get("sanctions_flag", False)),
        risk_tier=customer.get("risk_tier", "LOW"),
        created_at=str(customer.get("created_at", "")),
        accounts=accounts,
        recent_transactions=transactions,
        risk_profile=risk_profile,
        network_connections=connections,
        velocity=velocity,
        mule_indicators=mule_indicators,
    )


def _compute_risk_trend(scores: list[int]) -> str:
    if len(scores) < 4:
        return "STABLE"
    mid = len(scores) // 2
    older_avg = sum(scores[mid:]) / len(scores[mid:])
    newer_avg = sum(scores[:mid]) / len(scores[:mid])
    diff = newer_avg - older_avg
    if diff > 50:
        return "RISING"
    elif diff < -50:
        return "FALLING"
    return "STABLE"


def _compute_mule_indicators(customer_id: str, accounts: list, velocity: dict,
                              risk_profile: RiskProfile, session) -> dict:
    """
    Compute mule account indicators for this customer.
    A mule account passes money through rather than accumulates it.
    """
    indicators = {}

    # Turnover ratio: if inbound ≈ outbound in 30 days → possible mule
    flow = session.run("""
        MATCH (c:Customer {id: $customer_id})-[:OWNS]->(a:Account)
        OPTIONAL MATCH (a)-[:INITIATED]->(t_out:Transaction)
        WHERE t_out.timestamp >= $since
        OPTIONAL MATCH (a)<-[:CREDITED_TO]-(t_in:Transaction)
        WHERE t_in.timestamp >= $since
        RETURN sum(t_out.amount) AS outbound, sum(t_in.amount) AS inbound,
               count(DISTINCT t_out) AS out_count, count(DISTINCT t_in) AS in_count
    """, customer_id=customer_id,
        since=(datetime.utcnow() - timedelta(days=30)).isoformat()
    ).single()

    inbound = float(flow["inbound"] or 0) if flow else 0
    outbound = float(flow["outbound"] or 0) if flow else 0
    in_count = int(flow["in_count"] or 0) if flow else 0
    out_count = int(flow["out_count"] or 0) if flow else 0

    turnover = outbound / max(inbound, 1)
    indicators["inbound_volume_30d"] = round(inbound, 2)
    indicators["outbound_volume_30d"] = round(outbound, 2)
    indicators["turnover_ratio"] = round(turnover, 3)
    indicators["is_pass_through"] = 0.7 <= turnover <= 1.3 and inbound > 1000

    # Unique senders count (smurfing indicator)
    unique_senders = session.run("""
        MATCH (c:Customer {id: $customer_id})-[:OWNS]->(a:Account)
        MATCH (a)<-[:CREDITED_TO]-(t:Transaction)<-[:INITIATED]-(sender:Account)
        WHERE t.timestamp >= $since
        RETURN count(DISTINCT sender.id) AS unique_senders
    """, customer_id=customer_id,
        since=(datetime.utcnow() - timedelta(days=30)).isoformat()
    ).single()
    indicators["unique_senders_30d"] = int(unique_senders["unique_senders"] or 0) if unique_senders else 0
    indicators["high_sender_count"] = indicators["unique_senders_30d"] > 5

    # Average hold time (low = rapid pass-through)
    hold_time = session.run("""
        MATCH (c:Customer {id: $customer_id})-[:OWNS]->(a:Account)
        MATCH (a)<-[:CREDITED_TO]-(t_in:Transaction)
        MATCH (a)-[:INITIATED]->(t_out:Transaction)
        WHERE t_out.timestamp >= t_in.timestamp
          AND t_out.timestamp <= $since_cutoff
        WITH duration.between(datetime(t_in.timestamp), datetime(t_out.timestamp)).hours AS hold_hours
        WHERE hold_hours >= 0 AND hold_hours <= 72
        RETURN avg(hold_hours) AS avg_hold_hours
    """, customer_id=customer_id,
        since_cutoff=datetime.utcnow().isoformat()
    ).single()
    avg_hold = float(hold_time["avg_hold_hours"] or 48) if hold_time else 48
    indicators["avg_hold_time_hours"] = round(avg_hold, 1)
    indicators["rapid_disbursement"] = avg_hold < 6

    # Dormant accounts
    indicators["dormant_account_count"] = sum(1 for a in accounts if a.is_dormant)

    # Structuring
    indicators["structuring_incidents_30d"] = velocity.get("structuring_count", 0)
    indicators["structuring_risk"] = indicators["structuring_incidents_30d"] >= 2

    # High-risk network exposure
    indicators["high_risk_connections"] = sum(
        1 for a in risk_profile.__dict__.get("connections", [])
        if getattr(a, "risk_tier", "LOW") in ("HIGH", "CRITICAL")
    )

    # Composite mule score (0–100)
    mule_score = 0
    if indicators["is_pass_through"]:
        mule_score += 30
    if indicators["high_sender_count"]:
        mule_score += 20
    if indicators["rapid_disbursement"]:
        mule_score += 25
    if indicators["structuring_risk"]:
        mule_score += 15
    if indicators["dormant_account_count"] > 0:
        mule_score += 10
    indicators["mule_score"] = mule_score
    indicators["is_likely_mule"] = mule_score >= 50

    return indicators


def _build_transactions_from_records(records: list) -> list[TransactionSummary]:
    """Convert raw Neo4j query records into TransactionSummary objects."""
    _FRAUD_SCORE_MAP = {
        "STRUCTURING":       (720, "DECLINE",   ["is_structuring", "in_structuring_band"]),
        "SMURFING":          (700, "DECLINE",   ["is_structuring", "is_high_velocity_24h"]),
        "LAYERING":          (750, "DECLINE",   ["is_deep_layering", "is_cross_border"]),
        "ROUND_TRIP":        (710, "DECLINE",   ["is_round_trip"]),
        "DORMANT_BURST":     (680, "CHALLENGE", ["is_dormant_sender", "high_amount_deviation"]),
        "HIGH_RISK_CORRIDOR":(730, "DECLINE",   ["sender_to_high_risk", "is_cross_border", "beneficiary_country_risk"]),
        "RAPID_VELOCITY":    (760, "DECLINE",   ["is_high_velocity_1h", "is_high_velocity_24h"]),
    }
    results = []
    for r in records:
        t = _node(r["t"])
        pl = _node(r["pl"]) if r.get("pl") else None
        fraud_type = t.get("fraud_type")
        is_fraud = bool(t.get("is_fraud", False))

        if pl:
            risk_score   = int(pl.get("final_score", 0))
            outcome      = pl.get("outcome")
            risk_factors = list(pl.get("risk_factors", []))
        elif is_fraud and fraud_type and fraud_type in _FRAUD_SCORE_MAP:
            risk_score, outcome, risk_factors = _FRAUD_SCORE_MAP[fraud_type]
        else:
            risk_score   = None
            outcome      = None
            risk_factors = []

        results.append(TransactionSummary(
            id=t.get("id", ""),
            reference=t.get("reference", ""),
            amount=float(t.get("amount", 0)),
            currency=t.get("currency", "USD"),
            transaction_type=t.get("transaction_type", ""),
            channel=t.get("channel", ""),
            timestamp=str(t.get("timestamp", "")),
            is_fraud=is_fraud,
            fraud_type=fraud_type,
            sender_account_id=str(r.get("sender_account_id", "")),
            receiver_account_id=(
                str(r.get("receiver_account_id", "")) if r.get("receiver_account_id") else None
            ),
            receiver_name=r.get("receiver_name"),
            risk_score=risk_score,
            outcome=outcome,
            risk_factors=risk_factors,
        ))
    return results


def get_customer_transactions_page(
    customer_id: str,
    page: int = 1,
    page_size: int = 500,
) -> dict:
    """
    Return a single page of a customer's transactions, ordered by timestamp DESC.
    Response shape:
      { total, page, page_size, total_pages, transactions: [TransactionSummary] }
    """
    import dataclasses
    skip = (page - 1) * page_size

    with neo4j_session() as session:
        total_rec = session.run(TXN_COUNT_QUERY, customer_id=customer_id).single()
        total = int(total_rec["total"]) if total_rec else 0

        txn_records = list(session.run(
            TRANSACTION_HISTORY_PAGINATED_QUERY,
            customer_id=customer_id,
            skip=skip,
            limit=page_size,
        ))

    transactions = _build_transactions_from_records(txn_records)
    total_pages = max(1, (total + page_size - 1) // page_size)

    return {
        "total":       total,
        "page":        page,
        "page_size":   page_size,
        "total_pages": total_pages,
        "transactions": [dataclasses.asdict(t) for t in transactions],
    }


def list_customers(skip: int = 0, limit: int = 20, risk_tier: str = None) -> dict:
    """List customers with summary info."""
    where = f"WHERE c.risk_tier = '{risk_tier}'" if risk_tier else ""
    with neo4j_session() as session:
        result = session.run(f"""
            MATCH (c:Customer) {where}
            OPTIONAL MATCH (c)-[:OWNS]->(a:Account)
            WITH c, count(a) AS account_count
            RETURN c.id AS id, c.name AS name, c.customer_type AS type,
                   c.risk_tier AS risk_tier, c.pep_flag AS pep_flag,
                   c.sanctions_flag AS sanctions_flag,
                   c.kyc_level AS kyc_level,
                   c.country_of_residence AS country,
                   account_count
            ORDER BY c.risk_tier DESC, c.name
            SKIP $skip LIMIT $limit
        """, skip=skip, limit=limit)
        customers = [dict(r) for r in result]
        total = session.run(f"MATCH (c:Customer) {where} RETURN count(c) AS n").single()["n"]
    return {"total": total, "customers": customers}
