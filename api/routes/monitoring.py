"""Model monitoring and drift detection API routes."""
import json
from fastapi import APIRouter, HTTPException, Query, Body
from monitoring.performance import (
    get_performance_metrics, get_score_distribution_over_time,
    get_outcome_trend, get_top_risk_factors, get_summary_stats,
)
from monitoring.drift import (
    compute_and_store_drift_report, get_drift_history,
    fetch_recent_prediction_scores, fetch_outcome_distribution,
)
from monitoring.logger import get_recent_predictions, update_ground_truth
from risk.features import FeatureVector

router = APIRouter(prefix="/monitoring", tags=["Model Monitoring"])


@router.get("/summary")
async def monitoring_summary():
    """Dashboard summary: evaluation counts, outcomes, drift status, latency."""
    return get_summary_stats()


@router.get("/performance")
async def model_performance(days_back: int = Query(default=30, ge=1, le=90)):
    """
    Precision / Recall / F1 from labelled prediction logs.
    Labels are set when transactions are confirmed as fraud or legitimate.
    """
    return get_performance_metrics(days_back=days_back)


@router.get("/score-distribution")
async def score_distribution(
    days_back: int = Query(default=7, ge=1, le=90),
    bucket: str = Query(default="day", description="'hour' or 'day'"),
):
    """Score distribution over time bucketed by hour or day."""
    return {"data": get_score_distribution_over_time(days_back=days_back, bucket=bucket)}


@router.get("/outcome-trend")
async def outcome_trend(days_back: int = Query(default=30, ge=1, le=90)):
    """Daily counts of ALLOW / CHALLENGE / DECLINE outcomes."""
    return {"data": get_outcome_trend(days_back=days_back)}


@router.get("/risk-factors")
async def top_risk_factors(
    days_back: int = Query(default=7, ge=1, le=30),
    top_n: int = Query(default=15, ge=5, le=30),
):
    """Most frequently triggered risk factors in recent evaluations."""
    return {"data": get_top_risk_factors(days_back=days_back, top_n=top_n)}


@router.post("/drift/compute")
async def compute_drift():
    """
    Compute a new drift report comparing recent predictions to the training distribution.
    Stores the report as a DriftReport node in Neo4j.
    """
    # Load training reference from saved model metadata
    try:
        import joblib
        from config.settings import settings
        from pathlib import Path
        meta_path = Path(settings.model_path).parent / "training_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            reference_scores = meta.get("training_scores", [])
            reference_vectors = meta.get("training_vectors_sample", [])
        else:
            # Fall back to recent predictions as reference if no metadata
            reference_scores = fetch_recent_prediction_scores(hours_back=720)  # 30 days
            reference_vectors = []
    except Exception:
        reference_scores = fetch_recent_prediction_scores(hours_back=720)
        reference_vectors = []

    if len(reference_scores) < 10:
        raise HTTPException(
            status_code=400,
            detail="Not enough predictions to compute drift (need ≥10). "
                   "Run some evaluations first."
        )

    report = compute_and_store_drift_report(
        reference_scores=reference_scores,
        reference_vectors=reference_vectors,
        feature_names=FeatureVector.feature_names(),
        evaluation_hours=168,
    )
    # Remove large nested feature drift dict from response for readability
    response = {k: v for k, v in report.items() if k != "feature_drift"}
    response["feature_drift_summary"] = {
        k: {"psi": v["psi"], "alert_level": v["alert_level"]}
        for k, v in report.get("feature_drift", {}).items()
        if v["psi"] > 0.05
    }
    return response


@router.get("/drift/history")
async def drift_history(limit: int = Query(default=10, ge=1, le=50)):
    """Recent drift reports for trend display."""
    return {"reports": get_drift_history(limit=limit)}


@router.get("/predictions/recent")
async def recent_predictions(
    limit: int = Query(default=50, ge=1, le=200),
    hours_back: int = Query(default=24, ge=1, le=168),
):
    """Recent prediction logs for inspection."""
    preds = get_recent_predictions(limit=limit, hours_back=hours_back)
    # Strip large feature vectors from list view
    for p in preds:
        p.pop("feature_vector_json", None)
    return {"count": len(preds), "predictions": preds}


@router.post("/predictions/{transaction_id}/label")
async def label_prediction(
    transaction_id: str,
    is_fraud: bool = Body(..., embed=True),
):
    """
    Provide ground-truth label for a transaction.
    Used to compute precision/recall/F1 for performance monitoring.
    """
    update_ground_truth(transaction_id, is_fraud)
    return {"transaction_id": transaction_id, "is_fraud": is_fraud, "status": "labelled"}
