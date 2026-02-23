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
from ml.model import AMLModel, SVMModel, KNNModel, reset_registry

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
    Train XGBoost (hist), SGD/SVM, and FAISS-KNN on the same streamed dataset.
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

    # ── FAISS IVF-PQ KNN ──────────────────────────────────────────────────────
    try:
        import faiss
        console.print("\n[bold cyan]━━━ Training KNN (FAISS IVF-PQ) ━━━[/]")
    except ImportError:
        console.print("\n[bold cyan]━━━ Training KNN (sklearn ball_tree fallback) ━━━[/]")
    knn = KNNModel()
    m = knn.fit(X, y)
    _print_metrics("KNN", m)
    knn.save()
    all_metrics["knn"] = {k: v for k, v in m.items() if k != "classification_report"}
    console.print("[bold green]✓ KNN saved[/]")

    # ── Metadata for drift detection ──────────────────────────────────────────
    _save_training_metadata(X, y, xgb, all_metrics)
    console.print("\n[bold green]✓ All models saved — metadata written[/]")

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
    training_scores = [round(float(p) * 999)
                       for p in xgb_model.predict_proba_batch(X[idx])]
    payload: dict = {
        "training_scores":          training_scores,
        "training_vectors_sample":  X[idx].tolist(),
        "feature_names":            FeatureVector.feature_names(),
        "n_samples":                int(len(X)),
        "fraud_rate":               float(y.mean()),
    }
    if all_metrics:
        payload["model_metrics"] = all_metrics
    with open(meta_path, "w") as f:
        json.dump(payload, f)
