"""
ml/anomaly.py — KNN-based Mule Account Anomaly Detector
=========================================================
Repurposes FAISS KNN for unsupervised anomaly detection on mule accounts.

How it works
------------
1.  Training: load account-level feature vectors (8 dims) for a sample of
    accounts via a single batch Cypher query, then build a FAISS index on
    the LOW-RISK subset (pass_through_ratio < 0.6).

2.  Scoring: for a given account, compute account-level feature vector in
    ONE Neo4j query and compare it to the FAISS index.
    • Short L2 distance → account behaves like normal customers → low anomaly
    • Large distance → structurally unusual → high anomaly

3.  Composite score (0–100) combines KNN distance anomaly + rule-based
    mule indicators:
       - Pass-through ratio (inbound ≈ outbound volume)
       - Rapid disbursement (high out_30 with ptr > 0.7)
       - High unique sender count (aggregating from many sources)
       - Structuring patterns (sub-10k transactions)

4.  Results are stored in Neo4j as Account.anomaly_score and
    Account.mule_suspect for fast API queries.

Persistence
-----------
  models_saved/anomaly_index.faiss  — FAISS index (account feature vectors)
  models_saved/anomaly_meta.pkl     — scaler + threshold + stats
"""

from __future__ import annotations

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

# ── Batch account-stats loader ────────────────────────────────────────────────

_BATCH_STATS_QUERY = """
MATCH (a:Account)
OPTIONAL MATCH (a)-[:INITIATED]->(t_out:Transaction)
OPTIONAL MATCH (t_in:Transaction)-[:CREDITED_TO]->(a)
WITH a,
     coalesce(sum(t_out.amount), 0) AS out_volume,
     coalesce(sum(t_in.amount),  0) AS in_volume,
     coalesce(avg(t_out.amount), 0) AS avg_out,
     coalesce(avg(t_in.amount),  0) AS avg_in
OPTIONAL MATCH (sender_acct:Account)-[:INITIATED]->(txn_in2:Transaction)-[:CREDITED_TO]->(a)
WITH a, out_volume, in_volume, avg_out, avg_in,
     count(DISTINCT sender_acct) AS unique_senders
OPTIONAL MATCH (a)-[:INITIATED]->(struct_tx:Transaction)
  WHERE struct_tx.amount >= 9000 AND struct_tx.amount < 10000
WITH a, out_volume, in_volume, avg_out, avg_in,
     unique_senders, count(struct_tx) AS structuring
OPTIONAL MATCH (a)-[:INITIATED]->(t_30:Transaction)
  WHERE t_30.timestamp >= $since_30d
WITH a, out_volume, in_volume, avg_out, avg_in,
     unique_senders, structuring,
     count(DISTINCT t_30) AS out_30d
RETURN
  a.id                       AS id,
  out_volume, in_volume, avg_out, avg_in,
  unique_senders, structuring, out_30d,
  a.average_monthly_volume   AS avg_monthly_vol
ORDER BY rand()
SKIP $skip
LIMIT $limit
"""


def _load_account_feature_matrix(
    limit: int = 5_000,
) -> tuple[np.ndarray, list[str]]:
    """
    Load account-level statistics for up to `limit` accounts in one query.
    Returns (X float32 (n, 8), account_ids).
    """
    since_30d = (datetime.utcnow() - timedelta(days=30)).isoformat()
    with neo4j_session() as s:
        records = list(s.run(_BATCH_STATS_QUERY, since_30d=since_30d, skip=0, limit=limit))

    rows: list[np.ndarray] = []
    ids: list[str] = []
    for rec in records:
        try:
            fv = _stats_to_feature_vector(dict(rec))
            rows.append(fv)
            ids.append(rec["id"])
        except Exception:
            continue

    if not rows:
        return np.empty((0, 8), dtype=np.float32), []
    return np.vstack(rows).astype(np.float32), ids

_MODELS_DIR  = Path("models_saved")
_INDEX_FILE  = str(_MODELS_DIR / "anomaly_index.faiss")
_META_FILE   = str(_MODELS_DIR / "anomaly_meta.pkl")

_K           = 7        # neighbours for distance computation
_EVAL_CAP    = 500      # subsample for calibration stats (speed)


# ── Single-account stats query (for live scoring) ──────────────────────────────

SINGLE_ACCOUNT_QUERY = """
MATCH (a:Account {id: $account_id})
OPTIONAL MATCH (a)-[:INITIATED]->(t_out:Transaction)
OPTIONAL MATCH (t_in:Transaction)-[:CREDITED_TO]->(a)
WITH a,
     coalesce(sum(t_out.amount), 0) AS out_volume,
     coalesce(sum(t_in.amount),  0) AS in_volume,
     coalesce(avg(t_out.amount), 0) AS avg_out,
     coalesce(avg(t_in.amount),  0) AS avg_in
OPTIONAL MATCH (sender_acct:Account)-[:INITIATED]->(txn_in2:Transaction)-[:CREDITED_TO]->(a)
WITH a, out_volume, in_volume, avg_out, avg_in,
     count(DISTINCT sender_acct) AS unique_senders
OPTIONAL MATCH (a)-[:INITIATED]->(struct_tx:Transaction)
  WHERE struct_tx.amount >= 9000 AND struct_tx.amount < 10000
WITH a, out_volume, in_volume, avg_out, avg_in,
     unique_senders, count(struct_tx) AS structuring
OPTIONAL MATCH (a)-[:INITIATED]->(t_30:Transaction)
  WHERE t_30.timestamp >= $since_30d
WITH a, out_volume, in_volume, avg_out, avg_in,
     unique_senders, structuring,
     count(DISTINCT t_30) AS out_30d
RETURN
  a.id                       AS id,
  out_volume, in_volume, avg_out, avg_in,
  unique_senders, structuring, out_30d,
  a.average_monthly_volume   AS avg_monthly_vol
"""

# Batch query for scanning accounts (with/without existing scores)
_BATCH_SCAN_QUERY = """
MATCH (a:Account)
WHERE NOT EXISTS(a.anomaly_score)
WITH a
MATCH (a)-[:INITIATED]->(t_out:Transaction)
OPTIONAL MATCH (t_in:Transaction)-[:CREDITED_TO]->(a)
WITH a,
     coalesce(sum(t_out.amount), 0) AS out_volume,
     coalesce(sum(t_in.amount),  0) AS in_volume,
     coalesce(avg(t_out.amount), 0) AS avg_out,
     coalesce(avg(t_in.amount),  0) AS avg_in
OPTIONAL MATCH (sender_acct:Account)-[:INITIATED]->(txn_in2:Transaction)-[:CREDITED_TO]->(a)
WITH a, out_volume, in_volume, avg_out, avg_in,
     count(DISTINCT sender_acct) AS unique_senders
OPTIONAL MATCH (a)-[:INITIATED]->(struct_tx:Transaction)
  WHERE struct_tx.amount >= 9000 AND struct_tx.amount < 10000
WITH a, out_volume, in_volume, avg_out, avg_in,
     unique_senders, count(struct_tx) AS structuring
OPTIONAL MATCH (a)-[:INITIATED]->(t_30:Transaction)
  WHERE t_30.timestamp >= $since_30d
WITH a, out_volume, in_volume, avg_out, avg_in,
     unique_senders, structuring,
     count(DISTINCT t_30) AS out_30d
RETURN
  a.id AS id,
  out_volume, in_volume, avg_out, avg_in,
  unique_senders, structuring, out_30d,
  a.average_monthly_volume AS avg_monthly_vol
SKIP $skip LIMIT $limit
"""

_BATCH_SCAN_FORCE_QUERY = """
MATCH (a:Account)
OPTIONAL MATCH (a)-[:INITIATED]->(t_out:Transaction)
OPTIONAL MATCH (t_in:Transaction)-[:CREDITED_TO]->(a)
WITH a,
     coalesce(sum(t_out.amount), 0) AS out_volume,
     coalesce(sum(t_in.amount),  0) AS in_volume,
     coalesce(avg(t_out.amount), 0) AS avg_out,
     coalesce(avg(t_in.amount),  0) AS avg_in
OPTIONAL MATCH (sender_acct:Account)-[:INITIATED]->(txn_in2:Transaction)-[:CREDITED_TO]->(a)
WITH a, out_volume, in_volume, avg_out, avg_in,
     count(DISTINCT sender_acct) AS unique_senders
OPTIONAL MATCH (a)-[:INITIATED]->(struct_tx:Transaction)
  WHERE struct_tx.amount >= 9000 AND struct_tx.amount < 10000
WITH a, out_volume, in_volume, avg_out, avg_in,
     unique_senders, count(struct_tx) AS structuring
OPTIONAL MATCH (a)-[:INITIATED]->(t_30:Transaction)
  WHERE t_30.timestamp >= $since_30d
WITH a, out_volume, in_volume, avg_out, avg_in,
     unique_senders, structuring,
     count(DISTINCT t_30) AS out_30d
RETURN
  a.id AS id,
  out_volume, in_volume, avg_out, avg_in,
  unique_senders, structuring, out_30d,
  a.average_monthly_volume AS avg_monthly_vol
SKIP $skip LIMIT $limit
"""


def _stats_to_feature_vector(stats: dict) -> np.ndarray:
    """
    Build an 8-dim feature vector from account-level statistics row.
    All fields come from the batch/single Cypher queries above.
    """
    out_vol = float(stats.get("out_volume") or 0)
    in_vol  = float(stats.get("in_volume")  or 1)

    pass_through_ratio = min(out_vol / max(in_vol, 1), 5.0)
    unique_senders     = min(float(stats.get("unique_senders") or 0), 100)
    structuring        = min(float(stats.get("structuring") or 0), 50)
    out_30d            = min(float(stats.get("out_30d") or 0), 500)
    avg_out_amt        = min(float(stats.get("avg_out") or 0), 100_000)
    avg_in_amt         = min(float(stats.get("avg_in")  or 0), 100_000)
    monthly_vol        = min(float(stats.get("avg_monthly_vol") or 0), 1_000_000)
    vol_ratio          = min(max(out_vol, in_vol) / max(min(out_vol, in_vol), 1), 50)

    return np.array([
        pass_through_ratio,
        unique_senders,
        structuring,
        out_30d,
        avg_out_amt,
        avg_in_amt,
        monthly_vol,
        vol_ratio,
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

    # Default number of account vectors to sample for the index.
    _MAX_INDEX = 5_000

    def fit(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        max_normal: int | None = None,
    ) -> dict:
        """
        Build the anomaly index on a sample of account feature vectors.

        max_normal — how many accounts to sample (default: _MAX_INDEX = 5 000).
        We load account-level stats (one batch Cypher query) which is much
        faster than loading per-transaction feature vectors.
        X and y are accepted for API compatibility but ignored.
        """
        limit = max_normal if max_normal and max_normal > 0 else self._MAX_INDEX
        print(f"  Loading account stats for up to {limit:,} accounts from Neo4j…")
        X_all, ids = _load_account_feature_matrix(limit=limit)
        n_total = len(X_all)
        if n_total == 0:
            print("  ⚠ No account data found — detector not trained.")
            return {"n_normal_vectors": 0, "p95_dist": 0.0}

        # Use accounts with low pass-through ratio as the "normal" reference
        ptr_col = X_all[:, 0]  # pass_through_ratio is feature index 0
        normal_mask = ptr_col < 0.6
        X_normal = X_all[normal_mask]
        if len(X_normal) < 10:
            # fallback: use all accounts
            X_normal = X_all

        n_normal = len(X_normal)
        print(f"  Building anomaly index on {n_normal:,} normal account vectors "
              f"(out of {n_total:,} total)…")

        self._scaler = StandardScaler()
        X_s = self._scaler.fit_transform(X_normal).astype(np.float32)

        if _USE_FAISS:
            n_features = X_s.shape[1]
            self._index = faiss.IndexFlatL2(n_features)
            self._index.add(np.ascontiguousarray(X_s))
        else:
            self._index = X_s

        # Calibrate p95 distance on a subsample of normal vectors
        cap = min(_EVAL_CAP, n_normal)
        rng = np.random.default_rng(42)
        sample = X_s[rng.choice(n_normal, cap, replace=False)]
        dists  = self._query_distances(sample)
        self._p95_dist = float(np.percentile(dists, 95)) or 1.0

        self.is_trained = True
        return {
            "n_accounts_sampled": n_total,
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
        Compute anomaly score for a single account using ONE Neo4j query.
        Returns a dict with anomaly_score, indicators, and explanation.
        """
        if not self.is_trained:
            return {"account_id": account_id, "anomaly_score": 0, "error": "Detector not trained"}

        since_30d = (datetime.utcnow() - timedelta(days=30)).isoformat()

        with neo4j_session() as s:
            row = s.run(
                SINGLE_ACCOUNT_QUERY,
                account_id=account_id,
                since_30d=since_30d,
            ).single()

        if row is None:
            return {"account_id": account_id, "anomaly_score": 0, "error": "Not found"}

        stats = dict(row)
        return self._score_from_stats(stats)

    def _score_from_stats(self, stats: dict) -> dict:
        """Score an account given pre-fetched stats dict."""
        account_id = stats.get("id", "unknown")

        # ── KNN distance score ────────────────────────────────────────────────
        fv = _stats_to_feature_vector(stats).reshape(1, -1)
        X_s = self._scaler.transform(fv).astype(np.float32)
        dists = self._query_distances(X_s)
        knn_anomaly = self.anomaly_score_from_dist(float(dists[0]))

        # ── Rule-based mule indicators ────────────────────────────────────────
        out_vol = float(stats.get("out_volume") or 0)
        in_vol  = float(stats.get("in_volume")  or 0)
        ptr     = out_vol / max(in_vol, 1)
        u_send  = int(stats.get("unique_senders") or 0)
        struct  = int(stats.get("structuring") or 0)
        out_30  = int(stats.get("out_30d") or 0)

        is_pass_through    = 0.8 <= ptr <= 1.2 and in_vol > 5000
        high_sender_count  = u_send >= 5
        structuring_risk   = struct >= 2
        rapid_disbursement = out_30 >= 10 and ptr > 0.7

        rule_score = 0
        if is_pass_through:     rule_score += 35
        if high_sender_count:   rule_score += 25
        if structuring_risk:    rule_score += 25
        if rapid_disbursement:  rule_score += 15

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
        batch_size: int = 500,
        force: bool = False,
        max_accounts: int = 5_000,
        progress_cb=None,
    ) -> list[dict]:
        """
        Score up to max_accounts accounts using bulk Cypher queries.

        Each batch fetches stats for `batch_size` accounts in one Cypher call,
        scores them in Python, then writes results back in one UNWIND statement.
        This avoids the N+1 query problem of per-account scoring.

        max_accounts — hard cap (default 5 000). Set to 0 for unlimited.
        Returns list of suspect accounts (anomaly_score >= 40).
        """
        if not self.is_trained:
            raise RuntimeError("Detector not trained. Call fit() first.")

        scan_query = _BATCH_SCAN_FORCE_QUERY if force else _BATCH_SCAN_QUERY
        since_30d  = (datetime.utcnow() - timedelta(days=30)).isoformat()

        cap = max_accounts if max_accounts > 0 else 10_000_000
        suspects: list[dict] = []
        processed = 0

        for skip in range(0, cap, batch_size):
            fetch = min(batch_size, cap - skip)
            with neo4j_session() as s:
                rows = list(s.run(scan_query, since_30d=since_30d, skip=skip, limit=fetch))

            if not rows:
                break

            # Score the entire batch in Python (no more Neo4j calls)
            batch_results: list[dict] = []
            for rec in rows:
                try:
                    result = self._score_from_stats(dict(rec))
                    batch_results.append(result)
                except Exception:
                    pass

            if not batch_results:
                processed += len(rows)
                if progress_cb:
                    progress_cb(processed, cap)
                continue

            # Persist back to Neo4j in a single UNWIND
            ts = datetime.utcnow().isoformat()
            write_data = [
                {
                    "id":         r["account_id"],
                    "score":      r["anomaly_score"],
                    "suspect":    r["is_mule_suspect"],
                    "indicators": r["indicators"],
                    "ts":         ts,
                }
                for r in batch_results
            ]
            with neo4j_session() as s:
                s.run(
                    """
                    UNWIND $rows AS row
                    MATCH (a:Account {id: row.id})
                    SET a.anomaly_score     = row.score,
                        a.mule_suspect      = row.suspect,
                        a.mule_indicators   = row.indicators,
                        a.anomaly_scored_at = row.ts
                    """,
                    rows=write_data,
                )

            suspects.extend(r for r in batch_results if r["is_mule_suspect"])
            processed += len(rows)
            print(f"  [Anomaly] Scanned {processed:,} accounts, "
                  f"{len(suspects):,} suspects so far…")

            if progress_cb:
                progress_cb(processed, cap)

            if len(rows) < fetch:
                break  # no more accounts to fetch

        return suspects

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
