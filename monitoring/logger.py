"""
Prediction Logger
==================
Logs every risk evaluation to a PredictionLog node in Neo4j.
These logs are the raw material for:
  - Model drift detection
  - Performance monitoring
  - Ground-truth label collection
  - Audit trail
"""
import json
import uuid
from datetime import datetime
from db.client import neo4j_session

SCHEMA_QUERY = """
CREATE CONSTRAINT prediction_log_id IF NOT EXISTS
  FOR (pl:PredictionLog) REQUIRE pl.id IS UNIQUE
"""

INDEX_QUERY = """
CREATE INDEX prediction_log_ts IF NOT EXISTS
  FOR (pl:PredictionLog) ON (pl.timestamp)
"""


def ensure_schema():
    with neo4j_session() as session:
        session.run(SCHEMA_QUERY)
        session.run(INDEX_QUERY)


def log_prediction(
    transaction_id: str,
    bayesian_score: int,
    ml_score: int,
    final_score: int,
    outcome: str,
    risk_factors: list[str],
    feature_vector: list[float],
    confidence: float,
    processing_time_ms: float,
    is_fraud_ground_truth: bool | None = None,
) -> str:
    """
    Persist a prediction log entry. Returns the log ID.
    """
    log_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    with neo4j_session() as session:
        session.run("""
            MERGE (pl:PredictionLog {id: $id})
            SET pl += {
                transaction_id: $transaction_id,
                timestamp: $timestamp,
                bayesian_score: $bayesian_score,
                ml_score: $ml_score,
                final_score: $final_score,
                outcome: $outcome,
                risk_factors: $risk_factors,
                feature_vector_json: $feature_vector_json,
                confidence: $confidence,
                processing_time_ms: $processing_time_ms,
                is_fraud_ground_truth: $is_fraud_ground_truth
            }
            WITH pl
            OPTIONAL MATCH (t:Transaction {id: $transaction_id})
            FOREACH (_ IN CASE WHEN t IS NOT NULL THEN [1] ELSE [] END |
                MERGE (t)-[:HAS_PREDICTION]->(pl)
            )
        """,
            id=log_id,
            transaction_id=transaction_id,
            timestamp=now,
            bayesian_score=bayesian_score,
            ml_score=ml_score,
            final_score=final_score,
            outcome=outcome,
            risk_factors=risk_factors,
            feature_vector_json=json.dumps(feature_vector),
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            is_fraud_ground_truth=is_fraud_ground_truth,
        )

    return log_id


def update_ground_truth(transaction_id: str, is_fraud: bool) -> None:
    """Update the ground-truth label on a prediction log (used for performance tracking)."""
    with neo4j_session() as session:
        session.run("""
            MATCH (pl:PredictionLog {transaction_id: $txn_id})
            SET pl.is_fraud_ground_truth = $is_fraud
        """, txn_id=transaction_id, is_fraud=is_fraud)


def get_recent_predictions(limit: int = 100, hours_back: int = 24) -> list[dict]:
    since = (datetime.utcnow().replace(
        microsecond=0
    ) - __import__("datetime").timedelta(hours=hours_back)).isoformat()

    with neo4j_session() as session:
        records = list(session.run("""
            MATCH (pl:PredictionLog)
            WHERE pl.timestamp >= $since
            RETURN pl ORDER BY pl.timestamp DESC LIMIT $limit
        """, since=since, limit=limit))
    return [dict(r["pl"]) for r in records]
