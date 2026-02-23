"""
AML Risk Engine
================
Orchestrates the full evaluation pipeline for a single transaction:

  1. Pull transaction context from Neo4j graph
  2. Extract FeatureVector (graph + behavioural signals)
  3. Compute Bayesian risk score (0–999)
  4. Compute ML model risk score (0–999)
  5. Fuse scores using weighted ensemble
  6. Determine outcome: ALLOW / CHALLENGE / DECLINE
  7. Generate challenge question if needed

Risk Score: 0–999
  0–399   → ALLOW      (low risk)
  400–699 → CHALLENGE  (medium risk – present challenge question)
  700–999 → DECLINE    (high risk – block transaction)
"""
import time
import uuid
import random
from typing import Optional

from db.client import neo4j_session
from db.models import TransactionOutcome, RiskScore, ChallengeQuestion
from risk.features import extract_features, FeatureVector
from risk.bayesian import compute_bayesian_score
from ml.model import get_model
from config.settings import settings

# Weights for score fusion
BAYESIAN_WEIGHT = 0.55
ML_WEIGHT = 0.45

CHALLENGE_QUESTIONS = [
    "Please confirm this transaction by entering the OTP sent to your registered mobile number.",
    "What is the primary purpose of this transfer? (e.g. Business payment, Invoice, Personal support)",
    "Please confirm: do you authorise this transfer to {beneficiary} for {currency} {amount}?",
    "Enter the one-time passcode sent to your registered email address to proceed.",
    "Can you verify your identity? Please provide your registered mobile number's last 4 digits.",
    "This transaction has been flagged for review. Please confirm the destination account name.",
    "Please state the source of funds for this transfer.",
]

# Cypher: full transaction context for risk evaluation
TXN_CONTEXT_CYPHER = """
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

VELOCITY_CYPHER = """
MATCH (a:Account {id: $account_id})-[:INITIATED]->(tx:Transaction)
WHERE tx.timestamp < $ts AND tx.id <> $txn_id
WITH tx, $ts AS ts_ref
RETURN
  size([x IN collect(tx) WHERE x.timestamp >= $ts_1h]) AS count_1h,
  size([x IN collect(tx) WHERE x.timestamp >= $ts_24h]) AS count_24h,
  size([x IN collect(tx) WHERE x.timestamp >= $ts_7d]) AS count_7d,
  reduce(s=0.0, x IN [x IN collect(tx) WHERE x.timestamp >= $ts_24h] | s + x.amount) AS total_24h,
  reduce(s=0.0, x IN [x IN collect(tx) WHERE x.timestamp >= $ts_7d] | s + x.amount) AS total_7d,
  size([x IN collect(tx) WHERE x.amount >= 9000 AND x.amount < 10000
        AND x.timestamp >= $ts_24h]) AS structuring_24h
"""

NETWORK_CYPHER = """
MATCH path = (a:Account {id: $account_id})-[:INITIATED|CREDITED_TO*1..5]-(other:Account)
RETURN count(DISTINCT other) AS connected_accounts, max(length(path)) AS max_hops
"""

ROUND_TRIP_CYPHER = """
MATCH (a:Account {id: $account_id})-[:INITIATED]->(t_out:Transaction)-[:CREDITED_TO]->(b:Account)
MATCH (b)-[:INITIATED]->(t_in:Transaction)-[:CREDITED_TO]->(a)
WHERE t_out.timestamp >= $since AND t_in.timestamp >= $since
RETURN count(*) AS round_trip_count
"""

SHARED_DEVICE_CYPHER = """
MATCH (t:Transaction {id: $txn_id})-[:ORIGINATED_FROM]->(d:Device)
MATCH (other_t:Transaction)-[:ORIGINATED_FROM]->(d)
MATCH (a2:Account)-[:INITIATED]->(other_t)
MATCH (c2:Customer)-[:OWNS]->(a2)
RETURN count(DISTINCT c2) AS device_user_count
"""


def _node_to_dict(node) -> dict:
    if node is None:
        return {}
    return dict(node)


def _fetch_graph_context(txn_id: str) -> Optional[dict]:
    """Pull full graph context for a transaction ID."""
    with neo4j_session() as session:
        result = session.run(TXN_CONTEXT_CYPHER, txn_id=txn_id)
        record = result.single()
        if record is None:
            return None

        txn = _node_to_dict(record["t"])
        sender = _node_to_dict(record["sender"])

        ts = str(txn.get("timestamp", ""))

        # Velocity
        vel = {"count_1h": 0, "count_24h": 0, "count_7d": 0,
               "total_24h": 0.0, "total_7d": 0.0, "structuring_24h": 0}
        if sender.get("id"):
            from datetime import datetime, timedelta
            try:
                now = datetime.fromisoformat(ts.replace("Z", ""))
            except Exception:
                now = datetime.utcnow()
            vel_res = session.run(
                VELOCITY_CYPHER,
                account_id=sender["id"],
                txn_id=txn_id,
                ts=ts,
                ts_1h=(now - timedelta(hours=1)).isoformat(),
                ts_24h=(now - timedelta(hours=24)).isoformat(),
                ts_7d=(now - timedelta(days=7)).isoformat(),
            ).single()
            if vel_res:
                vel = dict(vel_res)

        # Network hops
        hop_count = 1
        if sender.get("id"):
            hop_res = session.run(NETWORK_CYPHER, account_id=sender["id"]).single()
            if hop_res:
                hop_count = int(hop_res.get("max_hops") or 1)

        # Shared device
        device_users = 1
        device_res = session.run(SHARED_DEVICE_CYPHER, txn_id=txn_id).single()
        if device_res:
            device_users = int(device_res.get("device_user_count") or 1)

        # Round-trip
        rt_count = 0
        if sender.get("id"):
            from datetime import timedelta
            try:
                since_48h = (datetime.fromisoformat(ts.replace("Z", "")) - timedelta(hours=48)).isoformat()
            except Exception:
                from datetime import datetime
                since_48h = (datetime.utcnow() - timedelta(hours=48)).isoformat()
            rt_res = session.run(ROUND_TRIP_CYPHER, account_id=sender["id"], since=since_48h).single()
            if rt_res:
                rt_count = int(rt_res.get("round_trip_count") or 0)

        return {
            "txn": txn,
            "sender": sender,
            "receiver": _node_to_dict(record["receiver"]),
            "sender_customer": _node_to_dict(record["sender_customer"]),
            "receiver_customer": _node_to_dict(record["receiver_customer"]),
            "device": _node_to_dict(record["device"]),
            "ip": _node_to_dict(record["ip"]),
            "merchant": _node_to_dict(record["merchant"]),
            "beneficiary": _node_to_dict(record["beneficiary"]),
            "txn_count_1h": int(vel.get("count_1h", 0)),
            "txn_count_24h": int(vel.get("count_24h", 0)),
            "txn_count_7d": int(vel.get("count_7d", 0)),
            "total_amount_24h": float(vel.get("total_24h", 0.0)),
            "total_amount_7d": float(vel.get("total_7d", 0.0)),
            "structuring_count_24h": int(vel.get("structuring_24h", 0)),
            "round_trip_count": rt_count,
            "shared_device_user_count": device_users,
            "network_hop_count": hop_count,
        }


def _fuse_scores(bayesian_score: int, ml_score: int) -> int:
    """Weighted linear fusion of Bayesian and ML scores."""
    fused = (bayesian_score * BAYESIAN_WEIGHT) + (ml_score * ML_WEIGHT)
    return max(0, min(999, round(fused)))


def _determine_outcome(score: int) -> TransactionOutcome:
    if score <= settings.risk_allow_max:
        return TransactionOutcome.ALLOW
    elif score <= settings.risk_challenge_max:
        return TransactionOutcome.CHALLENGE
    else:
        return TransactionOutcome.DECLINE


def _build_explanation(outcome: TransactionOutcome, score: int, factors: list[str]) -> str:
    if outcome == TransactionOutcome.ALLOW:
        return f"Transaction approved. Risk score {score}/999 is within acceptable limits."
    elif outcome == TransactionOutcome.CHALLENGE:
        top = ", ".join(factors[:3]) if factors else "velocity patterns"
        return (f"Transaction requires additional verification (score {score}/999). "
                f"Elevated risk detected: {top}.")
    else:
        top = ", ".join(factors[:3]) if factors else "multiple high-risk signals"
        return (f"Transaction declined (score {score}/999). "
                f"High-risk indicators detected: {top}. Transaction blocked.")


def _generate_challenge(txn_id: str, txn: dict) -> ChallengeQuestion:
    template = random.choice(CHALLENGE_QUESTIONS)
    amount = txn.get("amount", "")
    currency = txn.get("currency", "USD")
    beneficiary = txn.get("description", "the beneficiary")
    question = (template
                .replace("{amount}", f"{amount:,.2f}" if isinstance(amount, (int, float)) else str(amount))
                .replace("{currency}", currency)
                .replace("{beneficiary}", str(beneficiary)))
    return ChallengeQuestion(
        question=question,
        question_id=str(uuid.uuid4()),
        transaction_id=txn_id,
    )


def evaluate_transaction_by_id(txn_id: str) -> dict:
    """
    Full evaluation pipeline for a transaction already stored in Neo4j.
    Returns a dict suitable for TransactionEvaluationResponse.
    """
    t0 = time.perf_counter()

    ctx = _fetch_graph_context(txn_id)
    if ctx is None:
        raise ValueError(f"Transaction {txn_id} not found in graph database.")

    txn = ctx["txn"]
    graph_data = {k: v for k, v in ctx.items() if k != "txn"}

    # ── Feature extraction ───────────────────────────────────────
    fv: FeatureVector = extract_features(txn, graph_data)

    # ── Bayesian score ───────────────────────────────────────────
    bayes_result = compute_bayesian_score(fv)
    bayesian_score = bayes_result.score

    # ── ML score ─────────────────────────────────────────────────
    ml_score = 0
    try:
        model = get_model()
        if model.is_trained:
            ml_prob = model.predict_proba_single(fv)
            ml_score = max(0, min(999, round(ml_prob * 999)))
    except Exception:
        ml_score = bayesian_score  # fallback if model not loaded

    # ── Fused score ───────────────────────────────────────────────
    final_score = _fuse_scores(bayesian_score, ml_score)

    # ── Outcome decision ─────────────────────────────────────────
    outcome = _determine_outcome(final_score)
    explanation = _build_explanation(outcome, final_score, bayes_result.triggered_factors)

    confidence = min(1.0, abs(bayes_result.probability - 0.5) * 2 + 0.5)

    risk_score = RiskScore(
        score=final_score,
        bayesian_score=bayesian_score,
        ml_score=ml_score,
        outcome=outcome,
        risk_factors=bayes_result.triggered_factors,
        confidence=round(confidence, 4),
        explanation=explanation,
    )

    challenge_question = None
    if outcome == TransactionOutcome.CHALLENGE:
        challenge_question = _generate_challenge(txn_id, txn)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "transaction_id": txn_id,
        "risk_score": risk_score,
        "challenge_question": challenge_question,
        "processing_time_ms": round(elapsed_ms, 2),
    }


def evaluate_transaction_inline(txn_data: dict, graph_data: dict) -> dict:
    """
    Evaluate a transaction provided inline (not yet stored in Neo4j).
    Used for real-time pre-submission evaluation.
    """
    t0 = time.perf_counter()

    txn_id = txn_data.get("id", str(uuid.uuid4()))
    fv: FeatureVector = extract_features(txn_data, graph_data)

    bayes_result = compute_bayesian_score(fv)
    bayesian_score = bayes_result.score

    ml_score = bayesian_score
    try:
        model = get_model()
        if model.is_trained:
            ml_prob = model.predict_proba_single(fv)
            ml_score = max(0, min(999, round(ml_prob * 999)))
    except Exception:
        pass

    final_score = _fuse_scores(bayesian_score, ml_score)
    outcome = _determine_outcome(final_score)
    explanation = _build_explanation(outcome, final_score, bayes_result.triggered_factors)
    confidence = min(1.0, abs(bayes_result.probability - 0.5) * 2 + 0.5)

    risk_score = RiskScore(
        score=final_score,
        bayesian_score=bayesian_score,
        ml_score=ml_score,
        outcome=outcome,
        risk_factors=bayes_result.triggered_factors,
        confidence=round(confidence, 4),
        explanation=explanation,
    )

    challenge_question = None
    if outcome == TransactionOutcome.CHALLENGE:
        challenge_question = _generate_challenge(txn_id, txn_data)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "transaction_id": txn_id,
        "risk_score": risk_score,
        "challenge_question": challenge_question,
        "processing_time_ms": round(elapsed_ms, 2),
    }
