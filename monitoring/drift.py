"""
Model Drift Detection
======================
Uses Population Stability Index (PSI) to detect:
  1. Score distribution drift    : have recent risk scores shifted vs. training?
  2. Feature drift               : have individual feature distributions changed?
  3. Outcome drift               : has the ALLOW/CHALLENGE/DECLINE ratio changed?
  4. Prediction volume changes   : sudden spikes/drops in evaluation volume

PSI interpretation:
  PSI < 0.10  → No significant change (GREEN)
  PSI < 0.20  → Moderate change, monitor (YELLOW / WARNING)
  PSI ≥ 0.20  → Significant change, retrain likely needed (RED / CRITICAL)

The reference distribution is built from the training data (stored at model save time).
"""
import json
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from db.client import neo4j_session

PSI_WARNING_THRESHOLD = 0.10
PSI_CRITICAL_THRESHOLD = 0.20

# Score bins for PSI (0–999 split into 10 buckets)
SCORE_BINS = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
SCORE_BIN_LABELS = ["0-100", "101-200", "201-300", "301-400", "401-500",
                    "501-600", "601-700", "701-800", "801-900", "901-999"]


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """
    Compute Population Stability Index between two distributions.
    Both arrays must sum to 1.0 (proportions).
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-6
    expected = np.clip(expected, eps, None)
    actual = np.clip(actual, eps, None)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _score_distribution(scores: list[int]) -> np.ndarray:
    """Bin scores into SCORE_BINS and return proportion array."""
    if not scores:
        return np.ones(len(SCORE_BINS) - 1) / (len(SCORE_BINS) - 1)
    hist, _ = np.histogram(scores, bins=SCORE_BINS)
    return hist.astype(float) / max(hist.sum(), 1)


def compute_score_psi(reference_scores: list[int], current_scores: list[int]) -> dict:
    """Compare score distributions via PSI."""
    ref_dist = _score_distribution(reference_scores)
    cur_dist = _score_distribution(current_scores)
    psi_val = _psi(ref_dist, cur_dist)

    return {
        "psi": round(psi_val, 4),
        "alert_level": _alert_level(psi_val),
        "reference_distribution": {SCORE_BIN_LABELS[i]: round(float(ref_dist[i]), 4)
                                    for i in range(len(SCORE_BIN_LABELS))},
        "current_distribution": {SCORE_BIN_LABELS[i]: round(float(cur_dist[i]), 4)
                                  for i in range(len(SCORE_BIN_LABELS))},
    }


def compute_feature_psi(reference_vectors: list[list[float]],
                         current_vectors: list[list[float]],
                         feature_names: list[str]) -> dict[str, dict]:
    """
    Compute PSI per feature across reference and current feature vectors.
    For binary features uses observed proportion directly.
    For continuous features bins into 10 equal-width buckets.
    """
    if not reference_vectors or not current_vectors:
        return {}

    ref_arr = np.array(reference_vectors)
    cur_arr = np.array(current_vectors)
    results = {}

    for i, fname in enumerate(feature_names):
        ref_col = ref_arr[:, i]
        cur_col = cur_arr[:, i]

        # Determine if binary (only 0 or 1 values)
        unique_vals = np.unique(np.concatenate([ref_col, cur_col]))
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0.0, 1.0}):
            # Binary feature: just compare proportions
            ref_prop = np.array([np.mean(ref_col == 0), np.mean(ref_col == 1)])
            cur_prop = np.array([np.mean(cur_col == 0), np.mean(cur_col == 1)])
            psi_val = _psi(ref_prop, cur_prop)
        else:
            # Continuous: bin into 10 buckets using reference range
            min_val = ref_col.min()
            max_val = ref_col.max()
            if min_val == max_val:
                psi_val = 0.0
            else:
                bins = np.linspace(min_val, max_val, 11)
                ref_hist, _ = np.histogram(ref_col, bins=bins)
                cur_hist, _ = np.histogram(cur_col, bins=bins)
                psi_val = _psi(ref_hist.astype(float), cur_hist.astype(float))

        results[fname] = {
            "psi": round(psi_val, 4),
            "alert_level": _alert_level(psi_val),
            "ref_mean": round(float(ref_col.mean()), 4),
            "cur_mean": round(float(cur_col.mean()), 4),
            "mean_drift": round(float(cur_col.mean() - ref_col.mean()), 4),
        }

    return results


def _alert_level(psi: float) -> str:
    if psi < PSI_WARNING_THRESHOLD:
        return "OK"
    elif psi < PSI_CRITICAL_THRESHOLD:
        return "WARNING"
    return "CRITICAL"


def fetch_recent_prediction_scores(hours_back: int = 168) -> list[int]:
    """Pull final_score from recent PredictionLog entries."""
    since = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()
    with neo4j_session() as session:
        records = list(session.run("""
            MATCH (pl:PredictionLog) WHERE pl.timestamp >= $since
            RETURN pl.final_score AS score
        """, since=since))
    return [int(r["score"]) for r in records if r["score"] is not None]


def fetch_recent_feature_vectors(hours_back: int = 168) -> list[list[float]]:
    """Pull feature vectors from recent PredictionLog entries."""
    since = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()
    with neo4j_session() as session:
        records = list(session.run("""
            MATCH (pl:PredictionLog) WHERE pl.timestamp >= $since
              AND pl.feature_vector_json IS NOT NULL
            RETURN pl.feature_vector_json AS fv
            LIMIT 500
        """, since=since))
    vectors = []
    for r in records:
        try:
            vectors.append(json.loads(r["fv"]))
        except Exception:
            pass
    return vectors


def fetch_outcome_distribution(hours_back: int = 168) -> dict[str, int]:
    since = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()
    with neo4j_session() as session:
        records = list(session.run("""
            MATCH (pl:PredictionLog) WHERE pl.timestamp >= $since
            RETURN pl.outcome AS outcome, count(*) AS c
        """, since=since))
    return {r["outcome"]: int(r["c"]) for r in records if r["outcome"]}


def compute_and_store_drift_report(
    reference_scores: list[int],
    reference_vectors: list[list[float]],
    feature_names: list[str],
    evaluation_hours: int = 168,
) -> dict:
    """
    Full drift computation and persist as DriftReport node.
    """
    current_scores = fetch_recent_prediction_scores(evaluation_hours)
    current_vectors = fetch_recent_feature_vectors(evaluation_hours)
    current_outcomes = fetch_outcome_distribution(evaluation_hours)

    score_drift = compute_score_psi(reference_scores, current_scores) if current_scores else {
        "psi": 0.0, "alert_level": "OK",
        "reference_distribution": {}, "current_distribution": {},
    }

    feature_drift = {}
    if reference_vectors and current_vectors:
        feature_drift = compute_feature_psi(reference_vectors, current_vectors, feature_names)

    # Top drifted features
    top_drifted = sorted(feature_drift.items(), key=lambda x: x[1]["psi"], reverse=True)[:10]

    # Overall alert
    all_psis = [score_drift["psi"]] + [v["psi"] for v in feature_drift.values()]
    max_psi = max(all_psis) if all_psis else 0.0
    overall_alert = _alert_level(max_psi)

    # ── Build human-readable drift causes ─────────────────────────────────────
    drift_causes = _explain_drift_causes(
        score_drift=score_drift,
        feature_drift=feature_drift,
        current_outcomes=current_outcomes,
        reference_scores=reference_scores,
        current_scores=current_scores,
        top_drifted=top_drifted,
    )

    report = {
        "id": str(uuid.uuid4()),
        "computed_at": datetime.utcnow().isoformat(),
        "evaluation_window_hours": evaluation_hours,
        "reference_sample_size": len(reference_scores),
        "current_sample_size": len(current_scores),
        "score_distribution_psi": score_drift["psi"],
        "score_alert_level": score_drift["alert_level"],
        "max_feature_psi": round(max_psi, 4),
        "overall_alert_level": overall_alert,
        "features_in_warning": sum(1 for v in feature_drift.values()
                                   if v["alert_level"] == "WARNING"),
        "features_in_critical": sum(1 for v in feature_drift.values()
                                    if v["alert_level"] == "CRITICAL"),
        "top_drifted_features": [
            {"feature": k, "psi": v["psi"], "alert": v["alert_level"],
             "ref_mean": v["ref_mean"], "cur_mean": v["cur_mean"],
             "mean_drift": v["mean_drift"],
             "direction": "↑ increased" if v["mean_drift"] > 0 else "↓ decreased"}
            for k, v in top_drifted
        ],
        "drift_causes": drift_causes,
        "outcome_distribution": current_outcomes,
        "score_distribution": {
            "reference": score_drift.get("reference_distribution", {}),
            "current": score_drift.get("current_distribution", {}),
        },
        "feature_drift": feature_drift,
        "drift_detected": overall_alert in ("WARNING", "CRITICAL"),
    }

    # Persist to Neo4j
    with neo4j_session() as session:
        session.run("""
            MERGE (dr:DriftReport {id: $id})
            SET dr += {
                computed_at: $computed_at,
                evaluation_window_hours: $evaluation_window_hours,
                reference_sample_size: $reference_sample_size,
                current_sample_size: $current_sample_size,
                score_distribution_psi: $score_distribution_psi,
                max_feature_psi: $max_feature_psi,
                overall_alert_level: $overall_alert_level,
                features_in_warning: $features_in_warning,
                features_in_critical: $features_in_critical,
                drift_detected: $drift_detected,
                top_drifted_json: $top_drifted_json,
                outcome_distribution_json: $outcome_distribution_json,
                score_distribution_json: $score_distribution_json,
                drift_causes_json: $drift_causes_json
            }
        """,
            id=report["id"],
            computed_at=report["computed_at"],
            evaluation_window_hours=report["evaluation_window_hours"],
            reference_sample_size=report["reference_sample_size"],
            current_sample_size=report["current_sample_size"],
            score_distribution_psi=report["score_distribution_psi"],
            max_feature_psi=report["max_feature_psi"],
            overall_alert_level=report["overall_alert_level"],
            features_in_warning=report["features_in_warning"],
            features_in_critical=report["features_in_critical"],
            drift_detected=report["drift_detected"],
            top_drifted_json=json.dumps(report["top_drifted_features"]),
            outcome_distribution_json=json.dumps(report["outcome_distribution"]),
            score_distribution_json=json.dumps(report["score_distribution"]),
            drift_causes_json=json.dumps(report["drift_causes"]),
        )

    return report


# ── Human-readable drift cause explanation ────────────────────────────────────

_FEATURE_DESCRIPTIONS: dict[str, str] = {
    "is_cross_border":        "cross-border transactions",
    "is_wire":                "wire transfers",
    "is_crypto":              "cryptocurrency transactions",
    "is_cash":                "cash transactions",
    "is_high_amount":         "high-value transactions (>$10k)",
    "is_very_high_amount":    "very high-value transactions (>$50k)",
    "in_structuring_band":    "structuring-band amounts ($9k–$10k)",
    "is_round_amount":        "round-number amounts",
    "is_night_transaction":   "after-hours transactions",
    "is_weekend":             "weekend transactions",
    "is_new_sender_account":  "new sender accounts (<30d)",
    "is_dormant_sender":      "dormant accounts suddenly active",
    "sender_is_pep":          "politically exposed person (PEP) senders",
    "sender_is_sanctioned":   "sanctioned senders",
    "ip_is_tor":              "Tor network originating IPs",
    "ip_is_vpn":              "VPN originating IPs",
    "ip_country_mismatch":    "IP / account country mismatches",
    "is_high_velocity_1h":    "high-velocity activity (1h window)",
    "is_high_velocity_24h":   "high-velocity activity (24h window)",
    "is_structuring":         "structured transaction patterns",
    "is_round_trip":          "round-trip fund movements",
    "device_shared":          "shared device usage",
    "is_deep_layering":       "deep layering (3+ hops)",
    "receiver_country_risk":  "high-risk receiver jurisdictions",
    "beneficiary_country_risk": "high-risk beneficiary jurisdictions",
    "sender_to_high_risk":    "transfers to high-risk corridors",
    "sender_to_tax_haven":    "transfers to tax havens",
}


def _explain_drift_causes(
    score_drift: dict,
    feature_drift: dict,
    current_outcomes: dict,
    reference_scores: list[int],
    current_scores: list[int],
    top_drifted: list[tuple],
) -> list[dict]:
    """
    Produce a list of plain-English drift cause explanations with severity,
    category, and recommended action.
    """
    causes: list[dict] = []

    # ── Score distribution shift ───────────────────────────────────────────────
    if score_drift["psi"] >= PSI_WARNING_THRESHOLD and current_scores and reference_scores:
        ref_avg = float(np.mean(reference_scores))
        cur_avg = float(np.mean(current_scores))
        delta   = cur_avg - ref_avg
        direction = "higher" if delta > 0 else "lower"
        causes.append({
            "category": "Score Distribution",
            "severity": score_drift["alert_level"],
            "psi": score_drift["psi"],
            "title": f"Risk score distribution has shifted {direction}",
            "detail": (
                f"The average risk score has moved from {ref_avg:.0f} "
                f"to {cur_avg:.0f} ({delta:+.0f} pts). "
                f"PSI = {score_drift['psi']:.4f} — "
                + ("significant change, model retraining recommended."
                   if score_drift["psi"] >= PSI_CRITICAL_THRESHOLD
                   else "moderate change, monitor closely.")
            ),
            "recommendation": (
                "Retrain the model with recent data to recalibrate scores."
                if score_drift["psi"] >= PSI_CRITICAL_THRESHOLD
                else "Continue monitoring — run drift checks more frequently."
            ),
        })

    # ── Feature-level drift ────────────────────────────────────────────────────
    for fname, fdata in top_drifted:
        if fdata["psi"] < PSI_WARNING_THRESHOLD:
            break
        desc = _FEATURE_DESCRIPTIONS.get(fname, fname.replace("_", " "))
        direction = "increased" if fdata["mean_drift"] > 0 else "decreased"
        pct_change = (
            abs(fdata["mean_drift"]) / max(abs(fdata["ref_mean"]), 1e-9) * 100
        )
        causes.append({
            "category": "Feature Drift",
            "severity": fdata["alert_level"],
            "psi": fdata["psi"],
            "feature": fname,
            "title": f"Feature '{desc}' has {direction} significantly",
            "detail": (
                f"Proportion of {desc} changed from {fdata['ref_mean']:.3f} "
                f"(training) to {fdata['cur_mean']:.3f} (current) — "
                f"a {pct_change:.0f}% shift. PSI = {fdata['psi']:.4f}."
            ),
            "recommendation": (
                f"Investigate why {desc} have {'increased' if fdata['mean_drift'] > 0 else 'decreased'}. "
                "This may indicate a change in transaction mix or emerging fraud typology."
            ),
        })

    # ── Outcome distribution shift ─────────────────────────────────────────────
    total_current = sum(current_outcomes.values()) or 1
    decline_pct = current_outcomes.get("DECLINE", 0) / total_current * 100
    if decline_pct > 20:
        causes.append({
            "category": "Outcome Shift",
            "severity": "WARNING" if decline_pct < 35 else "CRITICAL",
            "psi": None,
            "title": f"Decline rate elevated at {decline_pct:.0f}%",
            "detail": (
                f"In the current evaluation window, {decline_pct:.0f}% of transactions "
                f"were DECLINED ({current_outcomes.get('DECLINE', 0)} out of {sum(current_outcomes.values())}). "
                "This is above the expected threshold."
            ),
            "recommendation": (
                "Review recent DECLINE decisions for false positives. "
                "Consider adjusting risk thresholds or investigating for fraud wave."
            ),
        })
    elif current_outcomes.get("DECLINE", 0) == 0 and sum(current_outcomes.values()) > 10:
        causes.append({
            "category": "Outcome Shift",
            "severity": "WARNING",
            "psi": None,
            "title": "No transactions declined — model may be under-scoring",
            "detail": "Zero DECLINE outcomes in the current window with an active evaluation set is unusual.",
            "recommendation": "Verify that the risk engine is correctly calibrated and evaluating recent transactions.",
        })

    # ── No drift detected ──────────────────────────────────────────────────────
    if not causes:
        if not current_scores:
            causes.append({
                "category": "Data Coverage",
                "severity": "WARNING",
                "psi": None,
                "title": "No recent predictions to evaluate",
                "detail": "There are no prediction logs in the evaluation window. Drift cannot be computed accurately.",
                "recommendation": "Submit transactions through the risk engine to generate prediction logs.",
            })
        else:
            causes.append({
                "category": "Status",
                "severity": "OK",
                "psi": score_drift["psi"],
                "title": "No significant drift detected",
                "detail": (
                    f"All PSI values are below the warning threshold ({PSI_WARNING_THRESHOLD}). "
                    f"Score PSI = {score_drift['psi']:.4f}. "
                    "The model is performing consistently with its training distribution."
                ),
                "recommendation": "Continue periodic drift checks every 7 days.",
            })

    return causes


def get_drift_history(limit: int = 20) -> list[dict]:
    """Return recent drift reports for trend display."""
    with neo4j_session() as session:
        records = list(session.run("""
            MATCH (dr:DriftReport)
            RETURN dr ORDER BY dr.computed_at DESC LIMIT $limit
        """, limit=limit))
    results = []
    for r in records:
        d = dict(r["dr"])
        for key in (
            "top_drifted_json",
            "outcome_distribution_json",
            "score_distribution_json",
            "drift_causes_json",
        ):
            if key in d:
                try:
                    d[key.replace("_json", "")] = json.loads(d.pop(key))
                except Exception:
                    d.pop(key, None)
        results.append(d)
    return results
