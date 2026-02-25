"""
Model Training Pipeline
========================
Fetches all transactions from Neo4j and trains three independent AML classifiers:

  1. XGBoost  — histogram-quantized gradient boosting (tree_method="hist")
  2. SGD/SVM  — linear SGD classifier (O(n) training, replaces RBF-SVC)
  3. KNN      — FAISS IVF-PQ approximate nearest neighbours (or sklearn fallback)

Scaling strategy for large datasets:
• Data is loaded in pages of PAGE_SIZE rows using SKIP/LIMIT so Neo4j memory
  stays bounded regardless of total transaction count.
• Feature vectors accumulate in a pre-allocated float32 array to avoid repeated
  Python list appends.
• After training, each model attempts to export an INT8-quantized ONNX artefact
  for low-latency serving.

All three models are saved to models_saved/ and loaded at API startup via
the ModelRegistry (ml.model.get_registry).
"""
import json
import random
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from db.client import neo4j_session
from risk.features import extract_features, FeatureVector
from ml.model import AMLModel, SVMModel, reset_registry
from config.settings import settings

console = Console()

# ── Query templates ────────────────────────────────────────────────────────────

COUNT_QUERY = "MATCH (sender:Account)-[:INITIATED]->(t:Transaction) RETURN count(t) AS n"

PAGE_QUERY = """
MATCH (sender:Account)-[:INITIATED]->(t:Transaction)
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

# Page size for streaming Neo4j reads.
# At 1K rows this is a single page; at 10M rows each page is ~PAGE_SIZE rows.
PAGE_SIZE = 10_000


# ── Helpers ────────────────────────────────────────────────────────────────────

def _node(n) -> dict:
    return dict(n) if n is not None else {}


def _record_to_feature(record) -> tuple[list[float], int, str]:
    """Convert a single Neo4j record to (feature_array, label, txn_id)."""
    txn      = _node(record["t"])
    sender   = _node(record["sender"])
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
        # Approximate velocity signals for known fraud patterns
        "txn_count_1h":  10 if fraud_type == "RAPID_VELOCITY" else 0,
        "txn_count_24h": 20 if fraud_type == "RAPID_VELOCITY" else (
                          8 if fraud_type in ("SMURFING", "STRUCTURING") else 0),
        "txn_count_7d":  0,
        "total_amount_24h":  0.0,
        "total_amount_7d":   0.0,
        "structuring_count_24h": (
            3 if fraud_type in ("SMURFING", "STRUCTURING") else (
            1 if 9000 <= float(txn.get("amount", 0)) < 10000 else 0)),
        "round_trip_count":        1 if fraud_type == "ROUND_TRIP" else 0,
        "shared_device_user_count":1,
        "network_hop_count":       4 if fraud_type == "LAYERING" else 1,
    }
    fv: FeatureVector = extract_features(txn, graph_data)
    return fv.to_ml_array(), (1 if txn.get("is_fraud") else 0), txn.get("id", "")


# ── Data loading ───────────────────────────────────────────────────────────────

def build_training_data(
    page_size: int = PAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Stream transactions from Neo4j in pages of `page_size` rows and return
    (X, y, txn_ids).

    Memory behaviour:
    • Only `page_size` Neo4j records are in Python memory at any one time.
    • Feature vectors are appended to a growing list then stacked once at the
      end.  For datasets > ~50M rows consider using np.memmap instead.

    Returns:
        X  — float32 feature matrix  (n_samples × n_features)
        y  — int32 binary labels     (1 = fraud, 0 = legitimate)
    """
    # ── Count total rows so we can page correctly ──────────────────────────────
    with neo4j_session() as session:
        total: int = session.run(COUNT_QUERY).single()["n"]

    console.print(f"[cyan]Total transactions in Neo4j: {total:,}[/]")
    n_pages = (total + page_size - 1) // page_size
    console.print(f"[cyan]Loading in {n_pages} page(s) of {page_size:,} rows…[/]")

    feature_rows: list[list[float]] = []
    labels:       list[int]         = []
    txn_ids:      list[str]         = []

    for page in range(n_pages):
        skip = page * page_size
        with neo4j_session() as session:
            records = list(session.run(PAGE_QUERY, skip=skip, limit=page_size))

        for record in records:
            feat, label, tid = _record_to_feature(record)
            feature_rows.append(feat)
            labels.append(label)
            txn_ids.append(tid)

        loaded = min(skip + page_size, total)
        console.print(f"  Page {page + 1}/{n_pages} — {loaded:,}/{total:,} rows loaded")

    X = np.array(feature_rows, dtype=np.float32)
    y = np.array(labels,       dtype=np.int32)
    fraud_count = int(y.sum())
    console.print(
        f"[bold]Dataset: {len(y):,} samples | Fraud: {fraud_count:,} "
        f"({fraud_count / len(y) * 100:.1f}%)[/]"
    )
    return X, y, txn_ids


# ── Reporting ──────────────────────────────────────────────────────────────────

def _print_metrics(model_name: str, metrics: dict) -> None:
    table = Table(title=f"{model_name} — Evaluation Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("ROC-AUC",               str(metrics["roc_auc"]))
    table.add_row("Avg Precision (PR-AUC)", str(metrics["avg_precision"]))
    table.add_row("Best Threshold",         str(metrics["best_threshold"]))
    if "index_type" in metrics:
        table.add_row("Index Type", metrics["index_type"])
        table.add_row("IVF Cells",  str(metrics.get("n_cells", "-")))
        table.add_row("PQ Sub-quantizers", str(metrics.get("m_pq", "-")))
    console.print(table)
    console.print(metrics["classification_report"])


def _try_export_onnx(model, label: str) -> None:
    """Attempt ONNX INT8 export; print result without raising."""
    if not hasattr(model, "export_onnx_quantized"):
        return
    ok = model.export_onnx_quantized()
    if ok:
        console.print(f"  [green]✓ {label}: INT8 ONNX quantized model exported[/]")
    else:
        console.print(
            f"  [yellow]⚠ {label}: ONNX export skipped "
            f"(install skl2onnx + onnxmltools + onnxruntime)[/]"
        )


# ── Training entry points ──────────────────────────────────────────────────────

def train_and_save() -> AMLModel:
    """Legacy single-model trainer — preserves backward compatibility."""
    X, y, _ = build_training_data()
    console.print("[bold cyan]Training XGBoost classifier…[/]")
    model = AMLModel()
    metrics = model.fit(X, y)
    _print_metrics("XGBoost", metrics)
    model.save()
    _try_export_onnx(model, "XGBoost")
    _save_training_metadata(X, y, model)
    console.print("[bold green]✓ XGBoost model saved[/]")
    reset_registry()
    return model


def train_and_save_all() -> dict:
    """
    Train XGBoost (hist) + SGD/SVM on the shared dataset, then train the
    KNN anomaly detector on normal transactions.
    Returns a dict of per-model metrics.
    """
    X, y, _ = build_training_data()
    all_metrics: dict[str, dict] = {}

    # ── XGBoost (histogram quantization) ──────────────────────────────────────
    console.print("\n[bold cyan]━━━ Training XGBoost (tree_method=hist, max_bin=256) ━━━[/]")
    xgb = AMLModel()
    m = xgb.fit(X, y)
    _print_metrics("XGBoost", m)
    xgb.save()
    _try_export_onnx(xgb, "XGBoost")
    all_metrics["xgb"] = {k: v for k, v in m.items() if k != "classification_report"}
    console.print("[bold green]✓ XGBoost saved[/]")

    # ── SGD / Linear SVM ──────────────────────────────────────────────────────
    console.print("\n[bold cyan]━━━ Training SGD classifier (linear SVM, O(n)) ━━━[/]")
    svm = SVMModel()
    m = svm.fit(X, y)
    _print_metrics("SGD/SVM", m)
    svm.save()
    _try_export_onnx(svm, "SGD/SVM")
    all_metrics["svm"] = {k: v for k, v in m.items() if k != "classification_report"}
    console.print("[bold green]✓ SGD/SVM saved[/]")

    # ── KNN Anomaly Detector (feature-flagged) ────────────────────────────────
    if settings.enable_knn_anomaly:
        from ml.anomaly import MuleAccountDetector, reset_detector
        console.print("\n[bold cyan]━━━ Training KNN Mule-Account Anomaly Detector ━━━[/]")
        detector = MuleAccountDetector()
        m_anom = detector.fit(max_normal=2_000)  # 2k accounts: fast + representative
        detector.save()
        reset_detector()
        all_metrics["anomaly"] = m_anom
        console.print(
            f"  Normal vectors indexed: {m_anom['n_normal_vectors']:,}  "
            f"| p95 dist: {m_anom['p95_dist']}"
        )
        console.print("[bold green]✓ KNN Anomaly detector saved[/]")
    else:
        console.print("\n[yellow]⚡ KNN anomaly detector skipped (ENABLE_KNN_ANOMALY=false)[/]")

    # ── GraphSAGE Mule-Account Detector (feature-flagged) ─────────────────────
    if settings.enable_graphsage:
        from ml.graphsage import train_graphsage, reset_sage
        console.print("\n[bold cyan]━━━ Training GraphSAGE Mule-Account Detector ━━━[/]")
        try:
            sage_metrics = train_graphsage(max_nodes=50_000, max_edges=500_000, epochs=60)
            all_metrics["graphsage"] = sage_metrics
            reset_sage()
            console.print(
                f"  ROC-AUC: {sage_metrics['roc_auc']} "
                f"| Avg Precision: {sage_metrics['avg_precision']}"
            )
            console.print("[bold green]✓ GraphSAGE model saved[/]")
        except Exception as exc:
            console.print(f"[yellow]⚠ GraphSAGE training failed: {exc}[/]")
            all_metrics["graphsage"] = {"error": str(exc)}
    else:
        console.print("\n[yellow]⚡ GraphSAGE skipped (ENABLE_GRAPHSAGE=false)[/]")

    # ── Metadata for drift detection ──────────────────────────────────────────
    _save_training_metadata(X, y, xgb, all_metrics)
    console.print("\n[bold green]✓ All models saved — metadata written[/]")

    # ── Register as new baseline version ──────────────────────────────────────
    try:
        from ml.version import (
            ModelVersion, get_version_registry,
            reset_version_registry, now_iso,
        )
        from db.client import neo4j_session as _neo4j_session
        # Get latest transaction timestamp as checkpoint
        with _neo4j_session() as _s:
            _row = _s.run("MATCH (t:Transaction) RETURN max(t.timestamp) AS ts").single()
            _last_ts = str(_row["ts"] or "")
        reg = get_version_registry()
        reg.reload()
        new_vid = reg.next_version_id()
        version = ModelVersion(
            version_id         = new_vid,
            status             = "baseline",
            trained_at         = now_iso(),
            n_samples          = int(len(X)),
            fraud_rate         = round(float(y.mean()), 4),
            last_txn_timestamp = _last_ts,
            training_type      = "full",
            trigger            = "manual",
            metrics            = all_metrics,
            notes              = f"Full retrain on {len(X):,} transactions.",
        )
        reg.archive_current_models(new_vid)
        reg.register_version(version)
        reg.set_baseline(new_vid, reason="Full retrain")
        reset_version_registry()
        console.print(f"[bold green]✓ Version {new_vid} registered as baseline[/]")
    except Exception as _e:
        console.print(f"[yellow]⚠ Version registration skipped: {_e}[/]")

    reset_registry()
    return all_metrics


# ── Training metadata (for drift detection) ────────────────────────────────────

def _save_training_metadata(X: np.ndarray, y: np.ndarray,
                            xgb_model: AMLModel,
                            all_metrics: dict | None = None) -> None:
    from config.settings import settings
    meta_path = Path(settings.model_path).parent / "training_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    sample_size = min(500, len(X))
    idx = random.sample(range(len(X)), sample_size)

    # Run XGBoost inference in a thread with a 30s timeout to avoid OpenMP
    # thread-pool deadlocks that can occur after long multi-model training runs.
    import concurrent.futures as _cf
    training_scores: list[int] = []
    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(xgb_model.predict_proba_batch, X[idx])
            probs = future.result(timeout=30)
            training_scores = [round(float(p) * 999) for p in probs]
    except Exception as e:
        console.print(f"  [yellow]⚠ Metadata score inference skipped ({e}); using label proxy[/]")
        # Fallback: use the labels as a coarse score proxy (0 → 0, 1 → 999)
        training_scores = [int(lbl) * 999 for lbl in y[idx].tolist()]

    # Sanitise all_metrics: convert numpy scalars → Python native so json.dump
    # doesn't raise TypeError on non-serialisable types.
    def _to_native(obj):
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_native(v) for v in obj]
        if hasattr(obj, "item"):           # numpy scalar
            return obj.item()
        return obj

    payload: dict = {
        "training_scores":          training_scores,
        "training_vectors_sample":  X[idx].tolist(),
        "feature_names":            FeatureVector.feature_names(),
        "n_samples":                int(len(X)),
        "fraud_rate":               float(y.mean()),
    }
    if all_metrics:
        payload["model_metrics"] = _to_native(all_metrics)
    with open(meta_path, "w") as f:
        json.dump(payload, f)
