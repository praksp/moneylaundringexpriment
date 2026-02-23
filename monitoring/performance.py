"""
Model Performance Monitoring
==============================
Tracks model performance over time using labelled prediction logs.
Computes precision, recall, F1, and confusion matrix over rolling windows.

Also provides:
  - Score distribution over time (hourly/daily buckets)
  - Outcome volume trend
  - Processing latency percentiles
"""
import json
from datetime import datetime, timedelta
from db.client import neo4j_session


def get_performance_metrics(days_back: int = 30) -> dict:
    """
    Compute precision/recall/F1 from labelled prediction logs.
    Only includes logs where is_fraud_ground_truth is not null.
    """
    since = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

    with neo4j_session() as session:
        records = list(session.run("""
            MATCH (pl:PredictionLog)
            WHERE pl.timestamp >= $since
              AND pl.is_fraud_ground_truth IS NOT NULL
            RETURN pl.final_score AS score,
                   pl.outcome AS outcome,
                   pl.is_fraud_ground_truth AS is_fraud
        """, since=since))

    if not records:
        return {"error": "No labelled predictions available", "labelled_count": 0}

    tp = fp = tn = fn = 0
    for r in records:
        predicted_fraud = r["outcome"] in ("CHALLENGE", "DECLINE")
        actual_fraud = bool(r["is_fraud"])
        if predicted_fraud and actual_fraud:
            tp += 1
        elif predicted_fraud and not actual_fraud:
            fp += 1
        elif not predicted_fraud and not actual_fraud:
            tn += 1
        else:
            fn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "period_days": days_back,
        "labelled_count": len(records),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fp / max(fp + tn, 1), 4),
        "false_negative_rate": round(fn / max(fn + tp, 1), 4),
    }


def get_score_distribution_over_time(days_back: int = 7, bucket: str = "day") -> list[dict]:
    """
    Score distribution bucketed by time window.
    bucket: 'hour' or 'day'
    """
    since = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

    with neo4j_session() as session:
        records = list(session.run("""
            MATCH (pl:PredictionLog) WHERE pl.timestamp >= $since
            RETURN pl.timestamp AS ts, pl.final_score AS score,
                   pl.outcome AS outcome, pl.processing_time_ms AS latency
            ORDER BY pl.timestamp ASC
        """, since=since))

    if not records:
        return []

    # Group by day
    buckets: dict[str, list] = {}
    for r in records:
        ts_str = str(r["ts"])
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", ""))
            if bucket == "hour":
                key = dt.strftime("%Y-%m-%dT%H:00")
            else:
                key = dt.strftime("%Y-%m-%d")
        except Exception:
            key = ts_str[:10]

        if key not in buckets:
            buckets[key] = []
        buckets[key].append({
            "score": int(r["score"] or 0),
            "outcome": r["outcome"],
            "latency": float(r["latency"] or 0),
        })

    result = []
    for key in sorted(buckets.keys()):
        items = buckets[key]
        scores = [x["score"] for x in items]
        latencies = [x["latency"] for x in items]
        outcomes = [x["outcome"] for x in items]
        result.append({
            "period": key,
            "count": len(items),
            "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "p50_score": round(sorted(scores)[len(scores) // 2], 1) if scores else 0,
            "p95_score": round(sorted(scores)[int(len(scores) * 0.95)], 1) if scores else 0,
            "allow_count": outcomes.count("ALLOW"),
            "challenge_count": outcomes.count("CHALLENGE"),
            "decline_count": outcomes.count("DECLINE"),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if latencies else 0,
        })

    return result


def get_outcome_trend(days_back: int = 30) -> list[dict]:
    """Daily outcome counts for trend charts."""
    since = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

    with neo4j_session() as session:
        records = list(session.run("""
            MATCH (pl:PredictionLog) WHERE pl.timestamp >= $since
            RETURN substring(pl.timestamp, 0, 10) AS day,
                   pl.outcome AS outcome, count(*) AS c
            ORDER BY day ASC
        """, since=since))

    result: dict[str, dict] = {}
    for r in records:
        day = str(r["day"])
        if day not in result:
            result[day] = {"date": day, "ALLOW": 0, "CHALLENGE": 0, "DECLINE": 0, "total": 0}
        outcome = str(r["outcome"] or "ALLOW")
        result[day][outcome] = int(r["c"])
        result[day]["total"] += int(r["c"])

    return list(result.values())


def get_top_risk_factors(days_back: int = 7, top_n: int = 15) -> list[dict]:
    """Most frequently triggered risk factors in recent evaluations."""
    since = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

    with neo4j_session() as session:
        records = list(session.run("""
            MATCH (pl:PredictionLog) WHERE pl.timestamp >= $since
            UNWIND pl.risk_factors AS factor
            RETURN factor, count(*) AS frequency,
                   avg(pl.final_score) AS avg_score_when_triggered
            ORDER BY frequency DESC LIMIT $top_n
        """, since=since, top_n=top_n))

    return [
        {
            "factor": r["factor"],
            "frequency": int(r["frequency"]),
            "avg_score_when_triggered": round(float(r["avg_score_when_triggered"] or 0), 1),
        }
        for r in records
    ]


def get_summary_stats() -> dict:
    """Overall system statistics for the dashboard."""
    with neo4j_session() as session:
        since_24h = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        since_7d = (datetime.utcnow() - timedelta(days=7)).isoformat()

        total = session.run("MATCH (pl:PredictionLog) RETURN count(pl) AS c").single()
        last_24h = session.run(
            "MATCH (pl:PredictionLog) WHERE pl.timestamp >= $since RETURN count(pl) AS c",
            since=since_24h
        ).single()
        outcomes_7d = list(session.run("""
            MATCH (pl:PredictionLog) WHERE pl.timestamp >= $since
            RETURN pl.outcome AS outcome, count(*) AS c
        """, since=since_7d))

        avg_score = session.run("""
            MATCH (pl:PredictionLog) WHERE pl.timestamp >= $since
            RETURN avg(pl.final_score) AS avg_s, avg(pl.processing_time_ms) AS avg_ms
        """, since=since_7d).single()

        latest_drift = session.run("""
            MATCH (dr:DriftReport)
            RETURN dr.overall_alert_level AS alert, dr.computed_at AS ts
            ORDER BY dr.computed_at DESC LIMIT 1
        """).single()

    outcome_counts = {r["outcome"]: int(r["c"]) for r in outcomes_7d if r["outcome"]}
    total_7d = sum(outcome_counts.values())

    return {
        "total_evaluations": int(total["c"]) if total else 0,
        "evaluations_24h": int(last_24h["c"]) if last_24h else 0,
        "outcome_7d": {
            **outcome_counts,
            "total": total_7d,
            "decline_rate_pct": round(
                outcome_counts.get("DECLINE", 0) / max(total_7d, 1) * 100, 2
            ),
        },
        "avg_score_7d": round(float(avg_score["avg_s"] or 0), 1) if avg_score else 0,
        "avg_latency_ms": round(float(avg_score["avg_ms"] or 0), 2) if avg_score else 0,
        "latest_drift_alert": latest_drift["alert"] if latest_drift else "UNKNOWN",
        "latest_drift_at": str(latest_drift["ts"]) if latest_drift else None,
    }
