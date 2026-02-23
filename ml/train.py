"""
Model Training Pipeline
========================
Fetches all transactions from Neo4j, extracts features, and trains three
independent AML classifiers:

  1. XGBoost — gradient-boosted trees
  2. SVM     — RBF kernel Support Vector Machine
  3. KNN     — calibrated k-Nearest Neighbours

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

ALL_TRANSACTIONS_QUERY = """
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
"""


def _node(n) -> dict:
    return dict(n) if n is not None else {}


def build_training_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns (X, y, txn_ids).
      X  — feature matrix (n_samples × n_features)
      y  — binary labels (1 = fraud, 0 = legitimate)
    """
    feature_rows: list[list[float]] = []
    labels: list[int] = []
    txn_ids: list[str] = []

    console.print("[cyan]Fetching transactions from Neo4j…[/]")
    with neo4j_session() as session:
        records = list(session.run(ALL_TRANSACTIONS_QUERY))

    console.print(f"[cyan]Building feature vectors for {len(records)} transactions…[/]")

    for i, record in enumerate(records):
        txn = _node(record["t"])
        sender = _node(record["sender"])
        fraud_type = txn.get("fraud_type", "")

        graph_data = {
            "sender": sender,
            "receiver": _node(record["receiver"]),
            "sender_customer": _node(record["sender_customer"]),
            "receiver_customer": _node(record["receiver_customer"]),
            "device": _node(record["device"]),
            "ip": _node(record["ip"]),
            "merchant": _node(record["merchant"]),
            "beneficiary": _node(record["beneficiary"]),
            # Approximate velocity signals for known fraud patterns
            "txn_count_1h":  10 if fraud_type == "RAPID_VELOCITY" else 0,
            "txn_count_24h": 20 if fraud_type == "RAPID_VELOCITY" else (
                             8 if fraud_type in ("SMURFING", "STRUCTURING") else 0),
            "txn_count_7d": 0,
            "total_amount_24h": 0.0,
            "total_amount_7d": 0.0,
            "structuring_count_24h": 3 if fraud_type in ("SMURFING", "STRUCTURING") else (
                                     1 if 9000 <= float(txn.get("amount", 0)) < 10000 else 0),
            "round_trip_count": 1 if fraud_type == "ROUND_TRIP" else 0,
            "shared_device_user_count": 1,
            "network_hop_count": 4 if fraud_type == "LAYERING" else 1,
        }

        fv: FeatureVector = extract_features(txn, graph_data)
        feature_rows.append(fv.to_ml_array())
        labels.append(1 if txn.get("is_fraud") else 0)
        txn_ids.append(txn.get("id", ""))

        if (i + 1) % 200 == 0:
            console.print(f"  {i + 1}/{len(records)} processed")

    X = np.array(feature_rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    fraud_count = int(y.sum())
    console.print(f"[bold]Dataset: {len(y)} samples | Fraud: {fraud_count} "
                  f"({fraud_count / len(y) * 100:.1f}%)[/]")
    return X, y, txn_ids


def _print_metrics(model_name: str, metrics: dict) -> None:
    table = Table(title=f"{model_name} — Evaluation Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("ROC-AUC", str(metrics["roc_auc"]))
    table.add_row("Avg Precision (PR-AUC)", str(metrics["avg_precision"]))
    table.add_row("Best Threshold", str(metrics["best_threshold"]))
    console.print(table)
    console.print(metrics["classification_report"])


def train_and_save() -> AMLModel:
    """Legacy single-model trainer — preserves backward compatibility."""
    X, y, _ = build_training_data()
    console.print("[bold cyan]Training XGBoost classifier…[/]")
    model = AMLModel()
    metrics = model.fit(X, y)
    _print_metrics("XGBoost", metrics)
    model.save()
    _save_training_metadata(X, y, model)
    console.print("[bold green]✓ XGBoost model saved[/]")
    reset_registry()
    return model


def train_and_save_all() -> dict:
    """
    Train XGBoost, SVM, and KNN models on the same dataset.
    Returns a dict of per-model metrics.
    """
    X, y, _ = build_training_data()
    all_metrics: dict[str, dict] = {}

    # ── XGBoost ──────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]━━━ Training XGBoost ━━━[/]")
    xgb = AMLModel()
    m = xgb.fit(X, y)
    _print_metrics("XGBoost", m)
    xgb.save()
    all_metrics["xgb"] = {k: v for k, v in m.items() if k != "classification_report"}
    console.print("[bold green]✓ XGBoost saved[/]")

    # ── SVM ───────────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]━━━ Training SVM (RBF kernel) ━━━[/]")
    svm = SVMModel()
    m = svm.fit(X, y)
    _print_metrics("SVM", m)
    svm.save()
    all_metrics["svm"] = {k: v for k, v in m.items() if k != "classification_report"}
    console.print("[bold green]✓ SVM saved[/]")

    # ── KNN ───────────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]━━━ Training KNN (k=7, isotonic calibration) ━━━[/]")
    knn = KNNModel()
    m = knn.fit(X, y)
    _print_metrics("KNN", m)
    knn.save()
    all_metrics["knn"] = {k: v for k, v in m.items() if k != "classification_report"}
    console.print("[bold green]✓ KNN saved[/]")

    # ── Metadata for drift detection ─────────────────────────────────────────
    _save_training_metadata(X, y, xgb, all_metrics)
    console.print("\n[bold green]✓ All models saved — metadata written[/]")

    reset_registry()
    return all_metrics


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
        "training_scores": training_scores,
        "training_vectors_sample": X[idx].tolist(),
        "feature_names": FeatureVector.feature_names(),
        "n_samples": int(len(X)),
        "fraud_rate": float(y.mean()),
    }
    if all_metrics:
        payload["model_metrics"] = all_metrics
    with open(meta_path, "w") as f:
        json.dump(payload, f)
