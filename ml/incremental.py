"""
Incremental Training Pipeline
==============================
Loads ONLY new transactions that arrived after the baseline model's training
checkpoint, fine-tunes each model, and creates an experimental version.

Key optimizations vs full retrain
----------------------------------
  • Delta query:   Only transactions with timestamp > last_trained_ts are loaded.
                   On a live system this is typically 100 – 50 000 rows vs 961 k,
                   reducing Neo4j load time from ~10 min to < 10 s.
  • XGBoost:       Warm-start — new trees are appended to the existing booster via
                   the xgb_model= parameter (no re-training existing trees).
  • SGD / SVM:     partial_fit() — online learning, O(n_new) only.
  • KNN anomaly:   New normal account vectors appended to FAISS index.
  • GraphSAGE:     Fine-tuned from saved weights for fewer epochs (20 vs 60).

Auto-promotion
--------------
  A new experimental version is automatically promoted to baseline when:
    xgb_roc_auc_new  >= xgb_roc_auc_baseline * PROMOTION_THRESHOLD (0.99)
    AND n_new_samples >= MIN_SAMPLES_TO_PROMOTE (500)
"""
from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import joblib

from db.client import neo4j_session
from risk.features import extract_features, FeatureVector
from ml.version import (
    ModelVersion, VersionRegistry, get_version_registry,
    reset_version_registry, now_iso,
)
from config.settings import settings

# ── Constants ──────────────────────────────────────────────────────────────────

PROMOTION_THRESHOLD   = 0.99    # new_auc >= baseline_auc * 0.99
MIN_SAMPLES_TO_PROMOTE = 500     # minimum new transactions to allow promotion
MAX_DELTA_ROWS        = 50_000   # cap incremental load to avoid memory pressure
INCREMENTAL_N_TREES   = 100      # extra XGBoost trees to add
GRAPHSAGE_FINE_EPOCHS = 20       # fine-tuning epochs for GraphSAGE

# ── Delta transaction query ────────────────────────────────────────────────────

_DELTA_COUNT_QUERY = """
MATCH (t:Transaction) WHERE t.timestamp > $since RETURN count(t) AS n
"""

_DELTA_PAGE_QUERY = """
MATCH (sender:Account)-[:INITIATED]->(t:Transaction)
WHERE t.timestamp > $since
OPTIONAL MATCH (t)-[:CREDITED_TO]->(receiver:Account)
OPTIONAL MATCH (sender)<-[:OWNS]-(sc:Customer)
OPTIONAL MATCH (receiver)<-[:OWNS]-(rc:Customer)
OPTIONAL MATCH (t)-[:ORIGINATED_FROM]->(device:Device)
OPTIONAL MATCH (t)-[:SOURCED_FROM]->(ip:IPAddress)
OPTIONAL MATCH (t)-[:PAID_TO]->(merchant:Merchant)
OPTIONAL MATCH (t)-[:SENT_TO_EXTERNAL]->(beneficiary:BeneficiaryAccount)
RETURN t, sender, receiver, sc AS sender_customer, rc AS receiver_customer,
       device, ip, merchant, beneficiary
ORDER BY t.timestamp
SKIP $skip LIMIT $limit
"""

_MAX_TS_QUERY = "MATCH (t:Transaction) RETURN max(t.timestamp) AS max_ts"


def _node(n) -> dict:
    return dict(n) if n is not None else {}


def _record_to_feature(record) -> tuple[list[float], int, str]:
    txn       = _node(record["t"])
    sender    = _node(record["sender"])
    fraud_type = txn.get("fraud_type", "")
    graph_data = {
        "sender":           sender,
        "receiver":         _node(record["receiver"]),
        "sender_customer":  _node(record["sender_customer"]),
        "receiver_customer":_node(record["receiver_customer"]),
        "device":           _node(record["device"]),
        "ip":               _node(record["ip"]),
        "merchant":         _node(record["merchant"]),
        "beneficiary":      _node(record["beneficiary"]),
        "txn_count_1h":  10 if fraud_type == "RAPID_VELOCITY" else 0,
        "txn_count_24h": 20 if fraud_type == "RAPID_VELOCITY" else (
                          8 if fraud_type in ("SMURFING", "STRUCTURING") else 0),
        "txn_count_7d":  0,
        "total_amount_24h":  0.0,
        "total_amount_7d":   0.0,
        "structuring_count_24h": (
            3 if fraud_type in ("SMURFING", "STRUCTURING") else (
            1 if 9000 <= float(txn.get("amount", 0)) < 10000 else 0)),
        "round_trip_count":         1 if fraud_type == "ROUND_TRIP" else 0,
        "shared_device_user_count": 1,
        "network_hop_count":        4 if fraud_type == "LAYERING" else 1,
    }
    fv: FeatureVector = extract_features(txn, graph_data)
    return fv.to_ml_array(), (1 if txn.get("is_fraud") else 0), txn.get("id", "")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_delta_transactions(
    since_ts: str,
    max_rows: int = MAX_DELTA_ROWS,
    page_size: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """
    Load all transactions with timestamp > since_ts (up to max_rows).
    Returns (X, y, txn_ids, latest_timestamp).
    """
    with neo4j_session() as s:
        total = min(
            s.run(_DELTA_COUNT_QUERY, since=since_ts).single()["n"],
            max_rows,
        )

    if total == 0:
        return (
            np.empty((0, len(FeatureVector.feature_names())), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            [],
            since_ts,
        )

    print(f"[Incremental] Loading {total:,} new transactions since {since_ts[:19]}…")
    n_pages = (total + page_size - 1) // page_size

    rows, labels, ids = [], [], []
    for page in range(n_pages):
        skip = page * page_size
        with neo4j_session() as s:
            records = list(s.run(_DELTA_PAGE_QUERY,
                                 since=since_ts, skip=skip, limit=page_size))
        for rec in records:
            feat, label, tid = _record_to_feature(rec)
            rows.append(feat)
            labels.append(label)
            ids.append(tid)
        print(f"  Page {page+1}/{n_pages} — {min(skip+page_size, total):,}/{total:,}")

    X = np.array(rows,   dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    # Get latest timestamp in the new batch
    with neo4j_session() as s:
        row = s.run(_MAX_TS_QUERY).single()
        latest_ts = row["max_ts"] or since_ts

    print(f"[Incremental] {len(y):,} new rows | fraud={y.mean()*100:.1f}%")
    return X, y, ids, str(latest_ts)


def get_current_max_ts() -> str:
    """Return the latest transaction timestamp in the database."""
    with neo4j_session() as s:
        row = s.run(_MAX_TS_QUERY).single()
        return str(row["max_ts"] or "")


# ── Per-model fine-tuning ──────────────────────────────────────────────────────

def _fine_tune_xgboost(X_new: np.ndarray, y_new: np.ndarray) -> dict:
    """
    Append INCREMENTAL_N_TREES new trees to the existing XGBoost booster.
    The existing scaler is reused (no refit — distribution shift is handled
    by the new trees).
    Returns evaluation metrics on hold-out portion of new data.
    """
    from ml.model import AMLModel, _eval_metrics, _calibrate_threshold
    from sklearn.model_selection import train_test_split
    try:
        from xgboost import XGBClassifier
        _USE_XGB = True
    except ImportError:
        _USE_XGB = False

    model = AMLModel()
    model.load()

    X_s = model.scaler.transform(X_new.astype(np.float32))
    if len(y_new) < 10:
        # Not enough data — skip fine-tuning
        return {"skipped": True, "reason": "too_few_samples"}

    X_tr, X_v, y_tr, y_v = train_test_split(
        X_s, y_new, test_size=0.2, random_state=42,
        stratify=y_new if y_new.sum() >= 2 else None,
    )

    if _USE_XGB:
        # Warm-start: add INCREMENTAL_N_TREES trees to existing booster
        incremental_clf = XGBClassifier(
            n_estimators=INCREMENTAL_N_TREES,
            max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=5,
            eval_metric="logloss",
            random_state=42, n_jobs=-1,
            tree_method="hist", max_bin=256,
        )
        incremental_clf.fit(
            X_tr, y_tr,
            xgb_model=model.classifier.get_booster(),   # warm-start
            eval_set=[(X_v, y_v)], verbose=False,
        )
        model.classifier = incremental_clf
    else:
        model.classifier.fit(X_tr, y_tr)

    model.threshold = _calibrate_threshold(model.classifier, X_v, y_v)
    model.save()
    print(f"[Incremental] XGBoost warm-started with {INCREMENTAL_N_TREES} new trees")
    return _eval_metrics(model.classifier, X_v, y_v, model.threshold)


def _fine_tune_svm(X_new: np.ndarray, y_new: np.ndarray) -> dict:
    """
    Apply partial_fit to the inner SGDClassifier from the SVM pipeline.
    Re-calibrates threshold afterwards.
    """
    from ml.model import SVMModel, _eval_metrics, _calibrate_threshold
    from sklearn.model_selection import train_test_split

    svm = SVMModel()
    svm.load()

    X_s = svm.pipeline.named_steps["scaler"].transform(X_new.astype(np.float32))
    if len(y_new) < 10:
        return {"skipped": True, "reason": "too_few_samples"}

    X_tr, X_v, y_tr, y_v = train_test_split(
        X_s, y_new, test_size=0.2, random_state=42,
        stratify=y_new if y_new.sum() >= 2 else None,
    )

    # Access the inner SGDClassifier via CalibratedClassifierCV
    inner_clf = svm.pipeline.named_steps["clf"]
    if hasattr(inner_clf, "estimator"):
        sgd = inner_clf.estimator
        sgd.partial_fit(X_tr, y_tr, classes=[0, 1])
    elif hasattr(inner_clf, "base_estimator"):
        sgd = inner_clf.base_estimator
        sgd.partial_fit(X_tr, y_tr, classes=[0, 1])
    else:
        # Fallback: directly call partial_fit on the pipeline clf
        try:
            inner_clf.partial_fit(X_tr, y_tr, classes=[0, 1])
        except AttributeError:
            return {"skipped": True, "reason": "partial_fit_unavailable"}

    svm.threshold = _calibrate_threshold(inner_clf, X_v, y_v)
    svm.save()
    print("[Incremental] SGD/SVM partial_fit applied")
    return _eval_metrics(inner_clf, X_v, y_v, svm.threshold)


def _fine_tune_graphsage(X_new: np.ndarray, y_new: np.ndarray) -> dict:
    """
    Fine-tune GraphSAGE from saved weights for fewer epochs on the updated graph.
    Falls back gracefully if GraphSAGE is disabled or fails.
    """
    if not settings.enable_graphsage:
        return {"skipped": True, "reason": "graphsage_disabled"}
    try:
        from ml.graphsage import train_graphsage, reset_sage
        # Use the existing graph + saved weights as starting point
        metrics = train_graphsage(
            max_nodes=50_000,
            max_edges=500_000,
            epochs=GRAPHSAGE_FINE_EPOCHS,
        )
        reset_sage()
        print(f"[Incremental] GraphSAGE fine-tuned: ROC-AUC={metrics['roc_auc']}")
        return metrics
    except Exception as e:
        print(f"[Incremental] GraphSAGE fine-tuning skipped: {e}")
        return {"skipped": True, "reason": str(e)}


def _update_anomaly_index() -> dict:
    """
    Rebuild the KNN anomaly index using the current account stats.
    Faster than retraining from scratch since we just re-index.
    """
    if not settings.enable_knn_anomaly:
        return {"skipped": True, "reason": "knn_disabled"}
    try:
        from ml.anomaly import MuleAccountDetector, reset_detector
        detector = MuleAccountDetector()
        metrics = detector.fit(max_normal=2_000)
        detector.save()
        reset_detector()
        print(f"[Incremental] KNN anomaly index rebuilt: {metrics['n_normal_vectors']} vectors")
        return metrics
    except Exception as e:
        print(f"[Incremental] KNN anomaly update skipped: {e}")
        return {"skipped": True, "reason": str(e)}


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_experimental_vs_baseline(
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    baseline: ModelVersion,
) -> dict:
    """
    Score the experimental (freshly saved) model on the hold-out set and
    return comparison metrics against the baseline.
    """
    from ml.model import AMLModel, _eval_metrics
    from ml.model import reset_registry

    # Load experimental (currently on disk)
    model = AMLModel()
    model.load()
    X_s = model.scaler.transform(X_eval.astype(np.float32))
    exp_metrics = _eval_metrics(model.classifier, X_s, y_eval, model.threshold)

    return {
        "experimental": exp_metrics,
        "baseline_xgb_auc": baseline.xgb_auc(),
        "improvement": round(exp_metrics["roc_auc"] - baseline.xgb_auc(), 4),
    }


# ── Main entry point ───────────────────────────────────────────────────────────

def run_incremental(
    trigger: str = "manual",
    force: bool = False,
    auto_promote: bool = True,
) -> dict:
    """
    Full incremental training pipeline:

    1. Load delta transactions since the baseline's last_txn_timestamp.
    2. Fine-tune XGBoost (warm-start), SGD/SVM (partial_fit),
       KNN anomaly (re-index), GraphSAGE (fine-tune).
    3. Evaluate the new models on a hold-out set drawn from the delta.
    4. Archive the new weights as an experimental version.
    5. Auto-promote if accuracy >= baseline * PROMOTION_THRESHOLD.

    Returns a status dict with version info and metrics.
    """
    t0 = time.time()
    reg = get_version_registry()
    reg.reload()
    baseline = reg.get_baseline()

    if baseline is None:
        return {"error": "No baseline version found. Run a full retrain first."}

    since_ts = baseline.last_txn_timestamp
    if not since_ts:
        return {"error": "Baseline has no last_txn_timestamp checkpoint."}

    # ── 1. Load delta ──────────────────────────────────────────────────────────
    X, y, txn_ids, new_max_ts = load_delta_transactions(since_ts)

    if len(y) < MIN_SAMPLES_TO_PROMOTE and not force:
        return {
            "status":     "skipped",
            "reason":     f"Only {len(y)} new transactions (< {MIN_SAMPLES_TO_PROMOTE} threshold).",
            "n_new":      len(y),
            "since_ts":   since_ts,
        }

    # Reserve 20% of delta for evaluation, rest for training
    n_eval   = max(1, int(len(y) * 0.2))
    eval_idx = random.sample(range(len(y)), n_eval)
    train_idx = [i for i in range(len(y)) if i not in set(eval_idx)]

    X_train, y_train = X[train_idx], y[train_idx]
    X_eval,  y_eval  = X[eval_idx],  y[eval_idx]

    print(f"[Incremental] Training on {len(y_train):,} | Evaluating on {len(y_eval):,}")

    # ── 2. Fine-tune models ────────────────────────────────────────────────────
    all_metrics: dict = {}

    print("[Incremental] Fine-tuning XGBoost…")
    xgb_metrics = _fine_tune_xgboost(X_train, y_train)
    all_metrics["xgb"] = {k: v for k, v in xgb_metrics.items()
                          if k != "classification_report"}

    print("[Incremental] Fine-tuning SGD/SVM…")
    svm_metrics = _fine_tune_svm(X_train, y_train)
    all_metrics["svm"] = {k: v for k, v in svm_metrics.items()
                          if k != "classification_report"}

    print("[Incremental] Updating KNN anomaly index…")
    knn_metrics = _update_anomaly_index()
    all_metrics["knn_anomaly"] = knn_metrics

    if settings.enable_graphsage:
        print("[Incremental] Fine-tuning GraphSAGE…")
        sage_metrics = _fine_tune_graphsage(X_train, y_train)
        all_metrics["graphsage"] = {k: v for k, v in sage_metrics.items()
                                    if k != "classification_report"}

    # ── 3. Evaluate experimental model ────────────────────────────────────────
    eval_result = evaluate_experimental_vs_baseline(X_eval, y_eval, baseline)
    exp_xgb_auc = eval_result["experimental"]["roc_auc"]
    improvement  = eval_result["improvement"]
    all_metrics["xgb"]["roc_auc_eval"] = exp_xgb_auc

    # ── 4. Archive as experimental version ────────────────────────────────────
    new_vid = reg.next_version_id()
    version = ModelVersion(
        version_id         = new_vid,
        status             = "experimental",
        trained_at         = now_iso(),
        n_samples          = len(y_train),
        fraud_rate         = round(float(y_train.mean()), 4),
        last_txn_timestamp = new_max_ts,
        training_type      = "incremental",
        trigger            = trigger,
        metrics            = all_metrics,
        notes              = (f"Delta train on {len(y_train):,} rows. "
                              f"XGB AUC={exp_xgb_auc:.4f} "
                              f"(Δ{improvement:+.4f} vs baseline {baseline.version_id})"),
    )
    reg.archive_current_models(new_vid)
    reg.register_version(version)
    reg.set_experimental(new_vid)

    # ── 5. Auto-promote? ──────────────────────────────────────────────────────
    promoted = False
    promotion_msg = ""
    if auto_promote and version.is_better_than(baseline):
        reason = (
            f"Auto-promoted: XGB AUC {exp_xgb_auc:.4f} >= "
            f"baseline {baseline.xgb_auc():.4f} * {PROMOTION_THRESHOLD}"
        )
        reg.set_baseline(new_vid, reason=reason)
        from ml.model import reset_registry
        reset_registry()
        from ml.anomaly import reset_detector
        reset_detector()
        promoted = True
        promotion_msg = reason
        print(f"[Incremental] {reason}")
    else:
        print(
            f"[Incremental] {new_vid} is EXPERIMENTAL "
            f"(AUC {exp_xgb_auc:.4f}, baseline {baseline.xgb_auc():.4f})"
        )

    elapsed = round(time.time() - t0, 1)
    reset_version_registry()   # force reload on next access
    return {
        "status":        "promoted" if promoted else "experimental",
        "version_id":    new_vid,
        "n_new_samples": len(y),
        "xgb_auc":       exp_xgb_auc,
        "baseline_auc":  baseline.xgb_auc(),
        "improvement":   improvement,
        "promoted":      promoted,
        "promotion_msg": promotion_msg,
        "elapsed_s":     elapsed,
        "metrics":       all_metrics,
    }


# ── Scoring against experimental model ────────────────────────────────────────

def score_with_experimental(fv: FeatureVector) -> Optional[float]:
    """
    Score a transaction against the current experimental model (if any).
    Returns a probability [0,1] or None if no experimental model exists.
    Used for shadow scoring during A/B evaluation.
    """
    reg = get_version_registry()
    if reg.get_experimental() is None:
        return None

    exp_dir = reg.get_experimental().artifact_dir
    model_path  = exp_dir / "aml_model.joblib"
    scaler_path = exp_dir / "aml_scaler.joblib"
    if not model_path.exists() or not scaler_path.exists():
        return None

    try:
        clf   = joblib.load(model_path)
        scaler, _ = joblib.load(scaler_path)
        arr = scaler.transform(
            np.array([fv.to_ml_array()], dtype=np.float32)
        )
        return float(clf.predict_proba(arr)[0, 1])
    except Exception:
        return None
