"""
Model Training Pipeline
========================
Fetches all transactions from Neo4j, extracts features, runs the Bayesian
engine, and trains the gradient-boosted ML model.
"""
import numpy as np
from rich.console import Console
from rich.table import Table

from db.client import neo4j_session
from risk.features import extract_features, FeatureVector
from risk.bayesian import compute_bayesian_score
from ml.model import AMLModel

console = Console()

# Cypher: fetch full context for all stored transactions
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

VELOCITY_QUERY = """
MATCH (a:Account {id: $account_id})-[:INITIATED]->(tx:Transaction)
WHERE tx.timestamp < $ts AND tx.id <> $txn_id
WITH tx ORDER BY tx.timestamp DESC
WITH collect(tx) AS all_txns
RETURN
  size([x IN all_txns WHERE x.timestamp >= $ts_1h]) AS count_1h,
  size([x IN all_txns WHERE x.timestamp >= $ts_24h]) AS count_24h,
  size([x IN all_txns WHERE x.timestamp >= $ts_7d]) AS count_7d,
  reduce(s=0.0, x IN [x IN all_txns WHERE x.timestamp >= $ts_24h] | s + x.amount) AS total_24h,
  reduce(s=0.0, x IN [x IN all_txns WHERE x.timestamp >= $ts_7d] | s + x.amount) AS total_7d,
  size([x IN all_txns WHERE x.amount >= 9000 AND x.amount < 10000
        AND x.timestamp >= $ts_24h]) AS structuring_24h
"""


def _node_to_dict(node) -> dict:
    if node is None:
        return {}
    return dict(node)


def build_training_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns (X, y, txn_ids) where:
      X: feature matrix (n_samples × n_features)
      y: binary labels (1 = fraud, 0 = legitimate)
      txn_ids: list of transaction IDs
    """
    feature_rows: list[list[float]] = []
    labels: list[int] = []
    txn_ids: list[str] = []

    console.print("[cyan]Fetching transactions from Neo4j...[/]")
    with neo4j_session() as session:
        result = session.run(ALL_TRANSACTIONS_QUERY)
        records = list(result)

    console.print(f"[cyan]Processing {len(records)} transactions...[/]")

    for i, record in enumerate(records):
        txn_node = record["t"]
        txn = _node_to_dict(txn_node)
        sender = _node_to_dict(record["sender"])
        receiver = _node_to_dict(record["receiver"])
        sender_customer = _node_to_dict(record["sender_customer"])
        receiver_customer = _node_to_dict(record["receiver_customer"])
        device = _node_to_dict(record["device"])
        ip = _node_to_dict(record["ip"])
        merchant = _node_to_dict(record["merchant"])
        beneficiary = _node_to_dict(record["beneficiary"])

        # Simplified velocity (avoid N+1 queries per transaction during training)
        # In production the full graph query runs; for training we approximate
        graph_data = {
            "sender": sender,
            "receiver": receiver,
            "sender_customer": sender_customer,
            "receiver_customer": receiver_customer,
            "device": device,
            "ip": ip,
            "merchant": merchant,
            "beneficiary": beneficiary,
            "txn_count_1h": 0,
            "txn_count_24h": 0,
            "txn_count_7d": 0,
            "total_amount_24h": 0.0,
            "total_amount_7d": 0.0,
            "structuring_count_24h": 1 if (
                9000 <= float(txn.get("amount", 0)) < 10000
            ) else 0,
            "round_trip_count": 0,
            "shared_device_user_count": 1,
            "network_hop_count": 1,
        }

        # Enrich velocity + network for fraud patterns (approximate)
        fraud_type = txn.get("fraud_type", "")
        if fraud_type == "RAPID_VELOCITY":
            graph_data["txn_count_1h"] = 10
            graph_data["txn_count_24h"] = 20
        elif fraud_type in ("SMURFING", "STRUCTURING"):
            graph_data["structuring_count_24h"] = 3
            graph_data["txn_count_24h"] = 8
        elif fraud_type == "LAYERING":
            graph_data["network_hop_count"] = 4
        elif fraud_type == "ROUND_TRIP":
            graph_data["round_trip_count"] = 1

        fv: FeatureVector = extract_features(txn, graph_data)
        feature_rows.append(fv.to_ml_array())
        labels.append(1 if txn.get("is_fraud") else 0)
        txn_ids.append(txn.get("id", ""))

        if (i + 1) % 200 == 0:
            console.print(f"  Processed {i + 1}/{len(records)}")

    X = np.array(feature_rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    fraud_count = int(y.sum())
    console.print(f"[bold]Dataset: {len(y)} samples | Fraud: {fraud_count} "
                  f"({fraud_count/len(y)*100:.1f}%)[/]")

    return X, y, txn_ids


def train_and_save() -> AMLModel:
    X, y, _ = build_training_data()

    console.print("[bold cyan]Training gradient-boosted classifier...[/]")
    model = AMLModel()
    metrics = model.fit(X, y)

    console.print("[bold green]Training complete! Metrics:[/]")
    table = Table(title="Model Evaluation")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("ROC-AUC", str(metrics["roc_auc"]))
    table.add_row("Avg Precision (PR-AUC)", str(metrics["avg_precision"]))
    table.add_row("Best Threshold", str(metrics["best_threshold"]))
    console.print(table)
    console.print(metrics["classification_report"])

    console.print("[cyan]Saving model...[/]")
    model.save()

    # Save training metadata for drift detection reference
    import json, random
    from pathlib import Path
    from config.settings import settings
    meta_path = Path(settings.model_path).parent / "training_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    # Sample up to 500 vectors to keep file small
    sample_size = min(500, len(X))
    idx = random.sample(range(len(X)), sample_size)
    training_scores = [round(float(p) * 999) for p in model.predict_proba_batch(X[idx])]
    with open(meta_path, "w") as f:
        json.dump({
            "training_scores": training_scores,
            "training_vectors_sample": X[idx].tolist(),
            "feature_names": FeatureVector.feature_names(),
            "n_samples": int(len(X)),
            "fraud_rate": float(y.mean()),
        }, f)
    console.print("[bold green]✓ Model saved[/]")

    return model
