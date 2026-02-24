"""
ml/anomaly.py — KNN-based Mule Account Anomaly Detector
=========================================================
Repurposes FAISS KNN for unsupervised anomaly detection on mule accounts.

How it works
------------
1.  During training, the detector builds a FAISS IndexFlatL2 index on
    NORMAL (non-fraud) transaction feature vectors ONLY.

2.  At inference, it computes the average L2 distance of an account's
    transactions to their K nearest normal neighbours.
    • Short distance → account behaves like normal customers → low anomaly
    • Large distance → account's behaviour is unusual compared to normal → high anomaly

3.  Mule-account composite score (0–100) combines:
    a. KNN distance anomaly (structural deviation from normal behaviour)
    b. Rule-based mule indicators derived from graph topology:
       - Pass-through ratio (inbound ≈ outbound volume)
       - Rapid disbursement (low hold time)
       - High unique sender count (aggregating from many sources)
       - Structuring patterns (sub-10k transactions)
       - Geographic spread (cross-border)

4.  Results are stored in Neo4j as Account.anomaly_score and
    Account.mule_suspect properties for fast API queries.

Persistence
-----------
  models_saved/anomaly_index.faiss  — FAISS index (normal transaction vectors)
  models_saved/anomaly_meta.pkl     — scaler + threshold + stats
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    import faiss
    _USE_FAISS = True
except ImportError:
    _USE_FAISS = False

from db.client import neo4j_session
from risk.features import extract_features, FeatureVector

# ── Lightweight normal-transaction loader ─────────────────────────────────────

_NORMAL_TXN_QUERY = """
MATCH (sender:Account)-[:INITIATED]->(t:Transaction {is_fraud: false})
OPTIONAL MATCH (t)-[:CREDITED_TO]->(receiver:Account)
OPTIONAL MATCH (sender)<-[:OWNS]-(sc:Customer)
OPTIONAL MATCH (receiver)<-[:OWNS]-(rc:Customer)
OPTIONAL MATCH (t)-[:ORIGINATED_FROM]->(device:Device)
OPTIONAL MATCH (t)-[:SOURCED_FROM]->(ip:IPAddress)
OPTIONAL MATCH (t)-[:SENT_TO_EXTERNAL]->(ben:BeneficiaryAccount)
RETURN t, sender, receiver, sc AS sender_customer, rc AS receiver_customer,
       device, ip, null AS merchant, ben AS beneficiary
ORDER BY rand()
LIMIT $limit
"""


def _load_normal_transactions(limit: int = 100_000) -> np.ndarray:
    """
    Load up to `limit` normal transaction feature vectors directly,
    without building a large Python list-of-lists first.
    Uses random ORDER BY rand() for a representative sample.
    Returns float32 ndarray of shape (n, n_features).
    """
    with neo4j_session() as s:
        records = list(s.run(_NORMAL_TXN_QUERY, limit=limit))

    rows: list[np.ndarray] = []
    for rec in records:
        try:
            txn    = dict(rec["t"])
            sender = dict(rec["sender"] or {})
            graph_data = {
                "sender":           sender,
                "receiver":         dict(rec["receiver"] or {}) if rec["receiver"] else {},
                "sender_customer":  dict(rec["sender_customer"] or {}) if rec["sender_customer"] else {},
                "receiver_customer":dict(rec["receiver_customer"] or {}) if rec["receiver_customer"] else {},
                "device":           dict(rec["device"] or {}) if rec["device"] else {},
                "ip":               dict(rec["ip"] or {}) if rec["ip"] else {},
                "merchant":         {},
                "beneficiary":      dict(rec["beneficiary"] or {}) if rec["beneficiary"] else {},
                "txn_count_1h": 0, "txn_count_24h": 0, "txn_count_7d": 0,
                "total_amount_24h": 0.0, "total_amount_7d": 0.0,
                "structuring_count_24h": 0, "round_trip_count": 0,
                "shared_device_user_count": 1, "network_hop_count": 1,
            }
            fv = extract_features(txn, graph_data)
            rows.append(np.array(fv.to_ml_array(), dtype=np.float32))
        except Exception:
            continue

    if not rows:
        return np.empty((0, len(FeatureVector.feature_names())), dtype=np.float32)
    return np.vstack(rows)

_MODELS_DIR  = Path("models_saved")
_INDEX_FILE  = str(_MODELS_DIR / "anomaly_index.faiss")
_META_FILE   = str(_MODELS_DIR / "anomaly_meta.pkl")

_K           = 11       # neighbours for distance computation
_EVAL_CAP    = 1_000    # subsample for calibration stats (speed)


# ── Account-level feature extraction from Neo4j ────────────────────────────────

ACCOUNT_STATS_QUERY = """
MATCH (a:Account {id: $account_id})
OPTIONAL MATCH (a)-[:INITIATED]->(t_out:Transaction)
OPTIONAL MATCH (t_in:Transaction)-[:CREDITED_TO]->(a)
WITH a,
     count(DISTINCT t_out) AS out_count,
     count(DISTINCT t_in)  AS in_count,
     coalesce(sum(t_out.amount), 0) AS out_volume,
     coalesce(sum(t_in.amount),  0) AS in_volume,
     coalesce(avg(t_out.amount), 0) AS avg_out,
     coalesce(avg(t_in.amount),  0) AS avg_in
OPTIONAL MATCH (a)-[:INITIATED]->(tx_out:Transaction)
WHERE tx_out.timestamp >= $since_30d
OPTIONAL MATCH (tx_in:Transaction)-[:CREDITED_TO]->(a)
WHERE tx_in.timestamp >= $since_30d
WITH a, out_count, in_count, out_volume, in_volume, avg_out, avg_in,
     count(DISTINCT tx_out) AS out_30d,
     count(DISTINCT tx_in)  AS in_30d,
     coalesce(sum(tx_out.amount), 0) AS out_vol_30d,
     coalesce(sum(tx_in.amount),  0) AS in_vol_30d
OPTIONAL MATCH (sender_acct:Account)-[:INITIATED]->(txn_in:Transaction)-[:CREDITED_TO]->(a)
WHERE txn_in.timestamp >= $since_30d
WITH a, out_count, in_count, out_volume, in_volume, avg_out, avg_in,
     out_30d, in_30d, out_vol_30d, in_vol_30d,
     count(DISTINCT sender_acct) AS unique_senders_30d
OPTIONAL MATCH (a)-[:INITIATED]->(struct_tx:Transaction)
WHERE struct_tx.amount >= 9000 AND struct_tx.amount < 10000
     AND struct_tx.timestamp >= $since_30d
WITH a, out_count, in_count, out_volume, in_volume, avg_out, avg_in,
     out_30d, in_30d, out_vol_30d, in_vol_30d,
     unique_senders_30d, count(struct_tx) AS structuring_30d
RETURN
  out_count, in_count, out_volume, in_volume, avg_out, avg_in,
  out_30d, in_30d, out_vol_30d, in_vol_30d,
  unique_senders_30d, structuring_30d,
  a.country AS country, a.account_type AS account_type,
  a.typical_transaction_size AS typical_txn_size,
  a.average_monthly_volume   AS avg_monthly_vol
"""

TXN_FEATURES_QUERY = """
MATCH (a:Account {id: $account_id})-[:INITIATED]->(t:Transaction)
OPTIONAL MATCH (t)-[:CREDITED_TO]->(recv:Account)
OPTIONAL MATCH (a)<-[:OWNS]-(sc:Customer)
OPTIONAL MATCH (recv)<-[:OWNS]-(rc:Customer)
OPTIONAL MATCH (t)-[:ORIGINATED_FROM]->(d:Device)
OPTIONAL MATCH (t)-[:SOURCED_FROM]->(ip:IPAddress)
OPTIONAL MATCH (t)-[:SENT_TO_EXTERNAL]->(ben:BeneficiaryAccount)
RETURN t, a AS sender, recv AS receiver,
       sc AS sender_customer, rc AS receiver_customer,
       d AS device, ip, null AS merchant, ben AS beneficiary
LIMIT 200
"""

BATCH_ACCOUNTS_QUERY = """
MATCH (a:Account)
WHERE NOT EXISTS(a.anomaly_score)   // skip already scored unless forced
RETURN a.id AS id
SKIP $skip LIMIT $limit
"""

BATCH_ALL_ACCOUNTS_QUERY = """
MATCH (a:Account) RETURN a.id AS id SKIP $skip LIMIT $limit
"""


def _account_feature_vector(stats: dict) -> np.ndarray:
    """
    Build a numeric feature vector from account-level statistics.
    Returns shape (8,) float32 array.
    """
    out_vol = float(stats.get("out_volume") or 0)
    in_vol  = float(stats.get("in_volume")  or 1)
    out_30  = float(stats.get("out_vol_30d") or 0)
    in_30   = float(stats.get("in_vol_30d")  or 1)

    pass_through_ratio = min(out_vol / max(in_vol, 1), 5.0)
    unique_senders     = min(float(stats.get("unique_senders_30d") or 0), 100)
    structuring        = min(float(stats.get("structuring_30d") or 0), 50)
    out_count_30       = min(float(stats.get("out_30d") or 0), 500)
    in_count_30        = min(float(stats.get("in_30d")  or 0), 500)
    avg_out_amt        = min(float(stats.get("avg_out") or 0), 100_000)
    avg_in_amt         = min(float(stats.get("avg_in")  or 0), 100_000)
    monthly_vol        = min(float(stats.get("avg_monthly_vol") or 0), 1_000_000)

    return np.array([
        pass_through_ratio,
        unique_senders,
        structuring,
        out_count_30,
        in_count_30,
        avg_out_amt,
        avg_in_amt,
        monthly_vol,
    ], dtype=np.float32)


# ── Main detector class ────────────────────────────────────────────────────────

class MuleAccountDetector:
    """KNN-based unsupervised anomaly detector for mule accounts."""

    def __init__(self):
        self._scaler: Optional[StandardScaler] = None
        self._index  = None          # FAISS IndexFlatL2 or None
        self._normal_labels: Optional[np.ndarray] = None
        self._p95_dist: float  = 1.0   # 95th percentile of normal distances (calibration)
        self.is_trained: bool  = False

    # ── Training ───────────────────────────────────────────────────────────────

    # Number of normal-transaction vectors to load and index.
    # Kept small to avoid memory pressure — 80k is more than sufficient to
    # represent the distribution of legitimate behaviour.
    _MAX_INDEX = 80_000

    def fit(self, X: np.ndarray | None = None, y: np.ndarray | None = None) -> dict:
        """
        Build the anomaly index.

        When called from train_and_save_all() X and y are available (already loaded).
        To avoid duplicating the 1GB list-of-lists allocation, we instead load a
        fresh small sample of normal transactions directly from Neo4j.
        X and y are accepted but ignored — the dedicated loader is always used.
        """
        print(f"  Loading up to {self._MAX_INDEX:,} normal transactions from Neo4j…")
        X_normal = _load_normal_transactions(limit=self._MAX_INDEX)
        n_normal = len(X_normal)
        if n_normal == 0:
            print("  ⚠ No normal transactions found — detector not trained.")
            return {"n_normal_vectors": 0, "p95_dist": 0.0}
        print(f"  Building anomaly index on {n_normal:,} normal transaction vectors…")

        self._scaler = StandardScaler()
        X_s = self._scaler.fit_transform(X_normal).astype(np.float32)

        if _USE_FAISS:
            n_features = X_s.shape[1]
            self._index = faiss.IndexFlatL2(n_features)
            self._index.add(np.ascontiguousarray(X_s))
        else:
            # Fallback: store raw scaled vectors for brute-force numpy search
            self._index = X_s

        # Calibrate p95 distance on a subsample of normal vectors
        cap = min(_EVAL_CAP, n_normal)
        rng = np.random.default_rng(42)
        sample = X_s[rng.choice(n_normal, cap, replace=False)]
        dists  = self._query_distances(sample)         # (cap,) mean distances
        self._p95_dist = float(np.percentile(dists, 95)) or 1.0

        self.is_trained = True
        return {
            "n_normal_vectors": n_normal,
            "p95_dist": round(self._p95_dist, 4),
        }

    # ── Distance computation ───────────────────────────────────────────────────

    def _query_distances(self, X_scaled: np.ndarray) -> np.ndarray:
        """Returns mean-NN distance for each row. Shape: (n_queries,)"""
        X_scaled = np.ascontiguousarray(X_scaled, dtype=np.float32)
        if _USE_FAISS and hasattr(self._index, "search"):
            k = min(_K, self._index.ntotal)
            dists, _ = self._index.search(X_scaled, k)
            return dists.mean(axis=1)
        else:
            # numpy brute-force fallback
            dists_all = []
            for row in X_scaled:
                d = np.sum((self._index - row) ** 2, axis=1)
                k = min(_K, len(d))
                dists_all.append(float(np.sort(d)[:k].mean()))
            return np.array(dists_all, dtype=np.float32)

    def anomaly_score_from_dist(self, mean_dist: float) -> float:
        """Convert mean distance → 0–100 anomaly score (higher = more suspicious)."""
        ratio = mean_dist / max(self._p95_dist, 1e-9)
        return min(100.0, round(ratio * 50, 2))

    # ── Per-account scoring ────────────────────────────────────────────────────

    def score_account(self, account_id: str) -> dict:
        """
        Compute anomaly score for a single account.
        Returns a dict with anomaly_score, indicators, and explanation.
        """
        if not self.is_trained:
            return {"account_id": account_id, "anomaly_score": 0, "error": "Detector not trained"}

        now = datetime.utcnow()
        since_30d = (now - timedelta(days=30)).isoformat()

        # ── Step 1: Fetch account statistics ──────────────────────────────────
        with neo4j_session() as s:
            stats_row = s.run(
                ACCOUNT_STATS_QUERY,
                account_id=account_id,
                since_30d=since_30d,
            ).single()
            if stats_row is None:
                return {"account_id": account_id, "anomaly_score": 0, "error": "Not found"}
            stats = dict(stats_row)

            # Fetch transaction features (up to 200 recent)
            txn_records = list(s.run(TXN_FEATURES_QUERY, account_id=account_id))

        # ── Step 2: KNN distance on transaction features ───────────────────────
        knn_anomaly = 0.0
        if txn_records:
            feature_rows = []
            for rec in txn_records:
                try:
                    txn = dict(rec["t"])
                    graph_data = {
                        "sender":           dict(rec["sender"] or {}),
                        "receiver":         dict(rec["receiver"] or {}) if rec["receiver"] else {},
                        "sender_customer":  dict(rec["sender_customer"] or {}) if rec["sender_customer"] else {},
                        "receiver_customer":dict(rec["receiver_customer"] or {}) if rec["receiver_customer"] else {},
                        "device":           dict(rec["device"] or {}) if rec["device"] else {},
                        "ip":               dict(rec["ip"] or {}) if rec["ip"] else {},
                        "merchant":         {},
                        "beneficiary":      dict(rec["beneficiary"] or {}) if rec["beneficiary"] else {},
                        "txn_count_1h": 0, "txn_count_24h": 0, "txn_count_7d": 0,
                        "total_amount_24h": 0.0, "total_amount_7d": 0.0,
                        "structuring_count_24h": 0, "round_trip_count": 0,
                        "shared_device_user_count": 1, "network_hop_count": 1,
                    }
                    fv = extract_features(txn, graph_data)
                    feature_rows.append(fv.to_ml_array())
                except Exception:
                    continue

            if feature_rows:
                X_txn = np.array(feature_rows, dtype=np.float32)
                X_scaled = self._scaler.transform(X_txn).astype(np.float32)
                mean_dists = self._query_distances(X_scaled)
                knn_anomaly = self.anomaly_score_from_dist(float(mean_dists.mean()))

        # ── Step 3: Rule-based mule indicators ────────────────────────────────
        out_vol  = float(stats.get("out_volume") or 0)
        in_vol   = float(stats.get("in_volume")  or 0)
        ptr      = out_vol / max(in_vol, 1)             # pass-through ratio
        u_send   = int(stats.get("unique_senders_30d") or 0)
        struct   = int(stats.get("structuring_30d")    or 0)
        out_30   = int(stats.get("out_30d")            or 0)

        is_pass_through = 0.8 <= ptr <= 1.2 and in_vol > 5000
        high_sender_count = u_send >= 5
        structuring_risk  = struct >= 2
        rapid_disbursement = out_30 >= 10 and ptr > 0.7

        rule_score = 0
        if is_pass_through:     rule_score += 35
        if high_sender_count:   rule_score += 25
        if structuring_risk:    rule_score += 25
        if rapid_disbursement:  rule_score += 15

        # ── Step 4: Composite score ────────────────────────────────────────────
        composite = round(0.5 * knn_anomaly + 0.5 * rule_score, 1)

        indicators = []
        if is_pass_through:     indicators.append("PASS_THROUGH")
        if high_sender_count:   indicators.append("HIGH_SENDER_COUNT")
        if structuring_risk:    indicators.append("STRUCTURING")
        if rapid_disbursement:  indicators.append("RAPID_DISBURSEMENT")

        return {
            "account_id":         account_id,
            "anomaly_score":      composite,
            "knn_distance_score": round(knn_anomaly, 1),
            "rule_score":         rule_score,
            "is_mule_suspect":    composite >= 40,
            "indicators":         indicators,
            "pass_through_ratio": round(ptr, 3),
            "unique_senders_30d": u_send,
            "structuring_30d":    struct,
            "in_volume":          round(in_vol, 2),
            "out_volume":         round(out_vol, 2),
        }

    # ── Batch scoring ──────────────────────────────────────────────────────────

    def scan_all_accounts(
        self,
        batch_size: int = 200,
        force: bool = False,
        progress_cb=None,
    ) -> list[dict]:
        """
        Score all accounts in Neo4j and persist results as Account properties.
        Returns list of suspect accounts (anomaly_score >= 40).
        """
        query = BATCH_ALL_ACCOUNTS_QUERY if force else BATCH_ACCOUNTS_QUERY

        # Count
        with neo4j_session() as s:
            total = s.run("MATCH (a:Account) RETURN count(a) AS n").single()["n"]

        results = []
        processed = 0
        for skip in range(0, total, batch_size):
            with neo4j_session() as s:
                ids = [r["id"] for r in s.run(query, skip=skip, limit=batch_size)]

            for account_id in ids:
                try:
                    result = self.score_account(account_id)
                    # Persist to Neo4j
                    with neo4j_session() as s:
                        s.run("""
                            MATCH (a:Account {id: $id})
                            SET a.anomaly_score    = $score,
                                a.mule_suspect     = $suspect,
                                a.mule_indicators  = $indicators,
                                a.anomaly_scored_at = $ts
                        """,
                        id=account_id,
                        score=result["anomaly_score"],
                        suspect=result["is_mule_suspect"],
                        indicators=result["indicators"],
                        ts=datetime.utcnow().isoformat())
                    if result["is_mule_suspect"]:
                        results.append(result)
                except Exception:
                    pass

            processed += len(ids)
            if progress_cb:
                progress_cb(processed, total)

        return results

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self):
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if _USE_FAISS and self._index is not None:
            faiss.write_index(self._index, _INDEX_FILE)
        joblib.dump(
            (self._scaler, self._p95_dist, self.is_trained, _USE_FAISS,
             self._index if not _USE_FAISS else None),
            _META_FILE,
        )

    def load(self):
        meta_path = Path(_META_FILE)
        if not meta_path.exists():
            raise FileNotFoundError(f"Anomaly detector not found: {meta_path}")
        self._scaler, self._p95_dist, self.is_trained, used_faiss, fallback_index = \
            joblib.load(_META_FILE)
        if _USE_FAISS and Path(_INDEX_FILE).exists():
            self._index = faiss.read_index(_INDEX_FILE)
        elif fallback_index is not None:
            self._index = fallback_index


# ── Singleton ──────────────────────────────────────────────────────────────────

_detector: Optional[MuleAccountDetector] = None


def get_detector() -> MuleAccountDetector:
    global _detector
    if _detector is None:
        _detector = MuleAccountDetector()
        try:
            _detector.load()
        except FileNotFoundError:
            pass
    return _detector


def reset_detector():
    global _detector
    _detector = None
