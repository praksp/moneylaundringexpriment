"""
ml/graphsage.py  — GraphSAGE Mule-Account Detector
====================================================
A full 2-layer GraphSAGE implementation using scipy.sparse for graph operations
and numpy for differentiable layer computations.  No PyTorch / TF dependency.

Architecture
------------
Input node features (account-level):
  - Normalised transaction statistics  (avg amount, count, velocity)
  - Fraud signal (fraud_ratio, fraud_pattern_flags)
  - Graph structure signals (degree, neighbour fraud ratio)
  - Account attributes (type, country risk, PEP/sanctions flag, KYC level)

GraphSAGE (2 layers, mean aggregator):
  h⁰ᵥ = X_v                                  # initial node features
  hˡᵥ = ReLU( Wˡ · [hˡ⁻¹ᵥ ‖ MEAN(hˡ⁻¹ₙ)] + bˡ )
  hˡᵥ = hˡᵥ / ‖hˡᵥ‖                          # L2 normalise

Output classifier:
  ŷ = σ( w_out · h²ᵥ + b_out )

Trained end-to-end with:
  • Binary cross-entropy loss
  • Mini-batch SGD (Adam)
  • Class-weighted loss to handle ~15% mule prevalence

Mule label derivation
---------------------
An account is labelled a mule when ≥1 of:
  1. ≥ 30% of its transactions are labelled is_fraud=True
  2. It initiated ≥ 2 fraud transactions with types:
     SMURFING | LAYERING | STRUCTURING | ROUND_TRIP | DORMANT_BURST | HIGH_RISK_CORRIDOR

Persistence
-----------
  models_saved/graphsage_model.npz   — learned weight matrices
  models_saved/graphsage_meta.pkl    — scaler + threshold + training stats
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score

from db.client import neo4j_session

# ── Paths ─────────────────────────────────────────────────────────────────────
_MODELS_DIR   = Path("models_saved")
_WEIGHTS_FILE = str(_MODELS_DIR / "graphsage_model.npz")
_META_FILE    = str(_MODELS_DIR / "graphsage_meta.pkl")

# ── Graph extraction queries ───────────────────────────────────────────────────

_NODE_QUERY = """
MATCH (a:Account)
OPTIONAL MATCH (c:Customer)-[:OWNS]->(a)
OPTIONAL MATCH (a)-[:INITIATED]->(t:Transaction)
WITH a, c,
     count(t)                                        AS txn_count,
     coalesce(avg(t.amount_usd), 0)                  AS avg_amount,
     coalesce(sum(t.amount_usd), 0)                  AS total_volume,
     coalesce(sum(CASE WHEN t.is_fraud THEN 1.0 ELSE 0.0 END), 0) AS fraud_count,
     coalesce(sum(CASE WHEN t.fraud_type IN
       ['SMURFING','LAYERING','STRUCTURING','ROUND_TRIP',
        'DORMANT_BURST','HIGH_RISK_CORRIDOR','RAPID_VELOCITY']
       THEN 1.0 ELSE 0.0 END), 0)                    AS pattern_count,
     coalesce(max(CASE WHEN t.is_fraud THEN t.amount_usd ELSE 0 END), 0)
                                                      AS max_fraud_amount
OPTIONAL MATCH (t_in:Transaction)-[:CREDITED_TO]->(a)
WITH a, c, txn_count, avg_amount, total_volume, fraud_count, pattern_count, max_fraud_amount,
     coalesce(count(t_in), 0)                         AS in_count,
     coalesce(sum(t_in.amount_usd), 0)                AS in_volume
RETURN
  a.id                             AS id,
  coalesce(txn_count, 0)           AS out_count,
  coalesce(in_count,  0)           AS in_count,
  coalesce(avg_amount, 0)          AS avg_amount,
  coalesce(total_volume, 0)        AS out_volume,
  coalesce(in_volume, 0)           AS in_volume,
  coalesce(fraud_count, 0)         AS fraud_count,
  coalesce(pattern_count, 0)       AS pattern_count,
  coalesce(max_fraud_amount, 0)    AS max_fraud_amount,
  CASE a.account_type
    WHEN 'CURRENT'  THEN 0 WHEN 'SAVINGS'  THEN 1
    WHEN 'BUSINESS' THEN 2 WHEN 'PREPAID'  THEN 3 ELSE 4
  END                              AS acct_type_enc,
  CASE COALESCE(c.kyc_level, 'BASIC')
    WHEN 'BASIC'       THEN 0
    WHEN 'SIMPLIFIED'  THEN 1
    WHEN 'ENHANCED'    THEN 2 ELSE 0
  END                              AS kyc_enc,
  CASE COALESCE(c.risk_tier, 'LOW')
    WHEN 'LOW'      THEN 0 WHEN 'MEDIUM'   THEN 1
    WHEN 'HIGH'     THEN 2 WHEN 'CRITICAL' THEN 3 ELSE 0
  END                              AS risk_tier_enc,
  COALESCE(c.pep_flag, false)      AS pep_flag,
  COALESCE(c.sanctions_flag, false) AS sanctions_flag,
  a.country                        AS country
LIMIT $limit
"""

_EDGE_QUERY = """
MATCH (a:Account)-[:INITIATED]->(t:Transaction)-[:CREDITED_TO]->(b:Account)
WHERE a.id IN $account_ids AND b.id IN $account_ids
RETURN DISTINCT a.id AS src, b.id AS dst
LIMIT $limit
"""

_COUNTRY_RISK_MAP = {
    "US": 0.0, "GB": 0.0, "DE": 0.0, "FR": 0.0, "CA": 0.0, "AU": 0.0,
    "SG": 0.0, "JP": 0.0, "KR": 0.0, "NZ": 0.0, "CH": 0.1, "IE": 0.0,
    "SE": 0.0, "NO": 0.0, "DK": 0.0, "FI": 0.0, "NL": 0.0, "BE": 0.0,
    "AT": 0.0, "PT": 0.0, "ES": 0.1, "IT": 0.1, "PL": 0.1, "HK": 0.2,
    "TW": 0.1, "CN": 0.3, "IN": 0.3, "BR": 0.4, "MX": 0.4, "AE": 0.4,
    "SA": 0.4, "TR": 0.4, "ZA": 0.4, "NG": 0.7, "KE": 0.4, "EG": 0.4,
    "RU": 0.8, "MM": 0.8, "VE": 0.8, "PK": 0.7, "AF": 0.9,
    "IR": 1.0, "KP": 1.0, "SY": 1.0,
    "KY": 0.6, "VG": 0.6, "PA": 0.5, "LU": 0.3, "MT": 0.3,
}


# ── Data extraction ────────────────────────────────────────────────────────────

def build_account_graph(
    max_nodes: int = 50_000,
    max_edges: int = 500_000,
) -> tuple[np.ndarray, np.ndarray, list[str], csr_matrix]:
    """
    Extract account node features, mule labels, and the transaction graph from Neo4j.

    Returns:
        X         — float32 feature matrix     (n_nodes × n_features)
        y         — int32 binary mule labels   (n_nodes,)  1=mule, 0=normal
        node_ids  — list of account ID strings (n_nodes,)
        adj       — row-normalised sparse adjacency (n_nodes × n_nodes)
    """
    print(f"  [GraphSAGE] Fetching up to {max_nodes:,} account nodes from Neo4j…")
    t0 = time.perf_counter()

    with neo4j_session() as s:
        rows = s.run(_NODE_QUERY, limit=max_nodes).data()

    print(f"  [GraphSAGE] {len(rows):,} accounts loaded in {time.perf_counter()-t0:.1f}s")

    node_ids: list[str] = []
    feat_rows: list[list[float]] = []
    labels: list[int] = []

    for r in rows:
        out_c = float(r["out_count"] or 0)
        in_c  = float(r["in_count"]  or 0)
        out_v = float(r["out_volume"] or 0)
        in_v  = float(r["in_volume"]  or 0)
        avg_a = float(r["avg_amount"] or 0)
        fr_c  = float(r["fraud_count"] or 0)
        pat_c = float(r["pattern_count"] or 0)
        max_f = float(r["max_fraud_amount"] or 0)

        fraud_ratio = fr_c / max(out_c, 1)
        pass_thru   = out_v / max(in_v, 1) if in_v > 100 else 0.0
        country_risk = _COUNTRY_RISK_MAP.get(r.get("country") or "", 0.2)

        # Mule label
        is_mule = int(fraud_ratio >= 0.30 or pat_c >= 2)

        node_ids.append(r["id"])
        labels.append(is_mule)
        feat_rows.append([
            min(out_c, 1000) / 1000,          # out_count (normalised)
            min(in_c,  1000) / 1000,          # in_count
            min(avg_a, 100_000) / 100_000,    # avg_amount
            min(out_v, 10_000_000) / 1e7,     # out_volume
            min(in_v,  10_000_000) / 1e7,     # in_volume
            fraud_ratio,                       # fraction of fraud txns
            min(fr_c, 50) / 50,               # fraud_count (normalised)
            min(pat_c, 20) / 20,              # pattern_count
            min(max_f, 500_000) / 500_000,    # max_fraud_amount
            float(r.get("acct_type_enc") or 0) / 4,
            float(r.get("kyc_enc")       or 0) / 2,
            float(r.get("risk_tier_enc") or 0) / 3,
            float(r.get("pep_flag")      or False),
            float(r.get("sanctions_flag")or False),
            country_risk,
            min(pass_thru, 5) / 5,            # pass-through ratio (capped at 5×)
        ])

    X = np.array(feat_rows, dtype=np.float32)
    y = np.array(labels,    dtype=np.int32)
    n = len(node_ids)
    idx_map = {nid: i for i, nid in enumerate(node_ids)}

    print(f"  [GraphSAGE] {n:,} nodes | mule prevalence: "
          f"{int(y.sum()):,} ({y.mean()*100:.1f}%)")
    print(f"  [GraphSAGE] Fetching up to {max_edges:,} edges…")

    t1 = time.perf_counter()
    with neo4j_session() as s:
        edge_rows = s.run(_EDGE_QUERY,
                          account_ids=node_ids, limit=max_edges).data()

    src_list, dst_list = [], []
    for er in edge_rows:
        s_idx = idx_map.get(er["src"])
        d_idx = idx_map.get(er["dst"])
        if s_idx is not None and d_idx is not None and s_idx != d_idx:
            src_list.append(s_idx)
            dst_list.append(d_idx)
            # Undirected: add reverse edge too
            src_list.append(d_idx)
            dst_list.append(s_idx)

    # Build symmetric sparse adjacency, row-normalise for mean aggregation
    data = np.ones(len(src_list), dtype=np.float32)
    adj_raw = csr_matrix(
        (data, (src_list, dst_list)),
        shape=(n, n),
        dtype=np.float32,
    )
    # Row-normalise: A_norm = D^{-1} A  (equivalent to mean aggregation)
    deg = np.array(adj_raw.sum(axis=1)).flatten()
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0).astype(np.float32)
    adj = diags(deg_inv) @ adj_raw   # D^{-1} A

    print(f"  [GraphSAGE] {len(src_list)//2:,} unique edges in "
          f"{time.perf_counter()-t1:.1f}s")
    return X, y, node_ids, adj


# ── Adam optimiser state ───────────────────────────────────────────────────────

class _Adam:
    def __init__(self, lr=3e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr; self.b1 = beta1; self.b2 = beta2; self.eps = eps
        self.m: dict = {}; self.v: dict = {}; self.t = 0

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for k in params:
            g = grads[k]
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g ** 2
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ── GraphSAGE model ────────────────────────────────────────────────────────────

N_INPUT   = 16      # must match len(feat_rows[0]) above
N_HIDDEN  = 64
N_OUT     = 32
L2_LAMBDA = 1e-4


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def _l2_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


class GraphSAGEModel:
    """
    2-layer GraphSAGE with mean aggregator.

    Forward pass per layer l (l=1,2):
      agg_v   = row-normalised-adj @ H^{l-1}      # mean of neighbour embeddings
      cat_v   = [H^{l-1} ‖ agg_v]                 # concat self + aggregated
      H^l     = ReLU( cat_v @ Wl + bl )
      H^l     = L2-normalise(H^l)

    Output:
      logits  = H² @ w_out + b_out
      probs   = sigmoid(logits)
    """

    def __init__(self):
        rng = np.random.default_rng(42)

        def _glorot(fan_in, fan_out):
            lim = math.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-lim, lim, (fan_in, fan_out)).astype(np.float32)

        self.params = {
            "W1":    _glorot(N_INPUT  * 2, N_HIDDEN),
            "b1":    np.zeros(N_HIDDEN, dtype=np.float32),
            "W2":    _glorot(N_HIDDEN * 2, N_OUT),
            "b2":    np.zeros(N_OUT,    dtype=np.float32),
            "w_out": _glorot(N_OUT, 1),
            "b_out": np.zeros(1,        dtype=np.float32),
        }
        self.threshold: float = 0.5
        self.is_trained: bool = False
        self._scaler: Optional[StandardScaler] = None
        self._training_stats: dict = {}

    # ── Forward ───────────────────────────────────────────────────────────────

    def _sage_layer(
        self,
        H: np.ndarray,
        adj: csr_matrix,
        W: np.ndarray,
        b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (H_out, pre_relu, agg, cat) for backprop."""
        agg      = adj @ H                                # mean aggregation
        cat      = np.hstack([H, agg])                   # concat self + neigh
        pre_relu = (cat @ W) + b                         # linear transform
        H_out    = _l2_norm(_relu(pre_relu))             # relu + L2-norm
        return H_out, pre_relu, agg, cat

    def forward(
        self,
        X: np.ndarray,
        adj: csr_matrix,
    ) -> tuple[np.ndarray, dict]:
        """Full forward pass. Returns (probs, cache) where cache holds
        intermediate values needed for backprop."""
        p = self.params
        H1, pre1, agg1, cat1 = self._sage_layer(X,  adj, p["W1"], p["b1"])
        H2, pre2, agg2, cat2 = self._sage_layer(H1, adj, p["W2"], p["b2"])
        logits = (H2 @ p["w_out"]).flatten() + p["b_out"][0]
        probs  = _sigmoid(logits)
        cache  = dict(X=X, H1=H1, H2=H2, pre1=pre1, pre2=pre2,
                      agg1=agg1, agg2=agg2, cat1=cat1, cat2=cat2,
                      logits=logits)
        return probs, cache

    # ── Backward ──────────────────────────────────────────────────────────────

    def backward(
        self,
        probs: np.ndarray,
        y: np.ndarray,
        cache: dict,
        adj: csr_matrix,
        weights: np.ndarray,
    ) -> dict:
        """
        Compute gradients of weighted binary cross-entropy w.r.t. all params.
        Uses chain rule through L2-norm → ReLU → linear → mean-aggregation.
        """
        p = self.params
        n = len(y)

        # ── Output layer ──────────────────────────────────────────────────────
        d_logits = (weights * (probs - y))[:, None] / n   # (n, 1)
        grads = {
            "w_out": cache["H2"].T @ d_logits,            # (N_OUT, 1)
            "b_out": np.array([d_logits.sum()]),
        }

        # ── Back through layer 2 ──────────────────────────────────────────────
        d_H2   = d_logits @ p["w_out"].T                  # (n, N_OUT)
        d_H2   = _back_l2_norm(d_H2, cache["H2"])         # through L2-norm
        d_pre2 = d_H2 * (cache["pre2"] > 0)              # through ReLU
        grads["W2"] = cache["cat2"].T @ d_pre2            # (N_HIDDEN*2, N_OUT)
        grads["b2"] = d_pre2.sum(axis=0)
        d_cat2 = d_pre2 @ p["W2"].T                      # (n, N_HIDDEN*2)
        # Gradient splits between self-part and aggregated-part
        d_H1_self = d_cat2[:, :N_HIDDEN]
        d_agg2    = d_cat2[:, N_HIDDEN:]
        # Mean aggregation gradient: d_H1 += adj^T @ d_agg2
        d_H1 = d_H1_self + adj.T @ d_agg2

        # ── Back through layer 1 ──────────────────────────────────────────────
        d_H1   = _back_l2_norm(d_H1, cache["H1"])
        d_pre1 = d_H1 * (cache["pre1"] > 0)
        grads["W1"] = cache["cat1"].T @ d_pre1
        grads["b1"] = d_pre1.sum(axis=0)

        # ── L2 regularisation ─────────────────────────────────────────────────
        for key in ("W1", "W2", "w_out"):
            grads[key] += L2_LAMBDA * p[key]

        return grads

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_raw: np.ndarray,
        y: np.ndarray,
        adj: csr_matrix,
        epochs: int = 60,
        batch_frac: float = 0.3,
        lr: float = 3e-3,
        patience: int = 8,
    ) -> dict:
        """
        Train GraphSAGE with mini-batch Adam SGD.
        Supports class-weighted loss for imbalanced mule prevalence.
        """
        n = len(y)

        # Scale features
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X_raw).astype(np.float32)

        # Class weights for imbalanced labels
        pos_frac = float(y.mean())
        neg_frac = 1 - pos_frac
        w_pos = 1.0 / (2 * max(pos_frac, 1e-3))
        w_neg = 1.0 / (2 * max(neg_frac, 1e-3))
        sample_weights = np.where(y == 1, w_pos, w_neg).astype(np.float32)

        # Train / validation split (stratified)
        rng = np.random.default_rng(42)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_val_pos = max(1, int(len(pos_idx) * 0.15))
        n_val_neg = max(1, int(len(neg_idx) * 0.15))
        val_pos = rng.choice(pos_idx, n_val_pos, replace=False)
        val_neg = rng.choice(neg_idx, n_val_neg, replace=False)
        val_idx = np.concatenate([val_pos, val_neg])
        train_idx = np.setdiff1d(np.arange(n), val_idx)

        opt = _Adam(lr=lr)
        batch_size = max(64, int(len(train_idx) * batch_frac))

        best_auc   = 0.0
        best_params = {k: v.copy() for k, v in self.params.items()}
        no_improve  = 0
        history: list[dict] = []

        print(f"  [GraphSAGE] Training {n:,} nodes | "
              f"train={len(train_idx):,} val={len(val_idx):,} | "
              f"epochs={epochs} batch={batch_size:,}")

        # ── For full-graph aggregation we need all nodes every forward pass.
        #    Mini-batching is done only for the loss / gradient on selected nodes.
        for epoch in range(1, epochs + 1):
            # Mini-batch over training nodes (shuffle)
            perm = rng.permutation(train_idx)
            epoch_loss = 0.0
            for start in range(0, len(perm), batch_size):
                batch = perm[start:start + batch_size]

                # Full-graph forward (needed for correct neighbourhood agg)
                probs, cache = self.forward(X, adj)

                # Compute loss on batch nodes only
                p_b = probs[batch]
                y_b = y[batch].astype(np.float32)
                w_b = sample_weights[batch]
                eps  = 1e-9
                loss = -np.mean(w_b * (
                    y_b * np.log(p_b + eps) + (1 - y_b) * np.log(1 - p_b + eps)
                ))
                epoch_loss += loss

                # Build per-node gradient mask (only batch nodes contribute)
                proxy_probs = probs.copy()
                proxy_y     = y.astype(np.float32)
                proxy_w     = np.zeros(n, dtype=np.float32)
                proxy_w[batch] = sample_weights[batch]

                grads = self.backward(proxy_probs, proxy_y, cache, adj, proxy_w)
                opt.step(self.params, grads)

            # Validation
            val_probs, _ = self.forward(X, adj)
            v_probs = val_probs[val_idx]
            v_y     = y[val_idx]
            if len(np.unique(v_y)) > 1:
                val_auc = roc_auc_score(v_y, v_probs)
            else:
                val_auc = 0.5

            history.append({
                "epoch": epoch,
                "loss": round(float(epoch_loss), 4),
                "val_auc": round(float(val_auc), 4),
            })

            if epoch % 10 == 0 or epoch == 1:
                print(f"    Epoch {epoch:3d}/{epochs} | "
                      f"loss={epoch_loss:.4f} | val_auc={val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                best_params = {k: v.copy() for k, v in self.params.items()}
                no_improve  = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    Early stop at epoch {epoch} (best val_auc={best_auc:.4f})")
                    break

        # Restore best weights
        self.params = best_params

        # Calibrate threshold on validation set
        final_probs, _ = self.forward(X, adj)
        vp = final_probs[val_idx]
        vy = y[val_idx]
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(vy, vp)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        best_th_idx   = int(np.argmax(f1[:-1]))
        self.threshold = float(thresholds[best_th_idx]) if len(thresholds) else 0.5

        # Final metrics
        preds = (vp >= self.threshold).astype(int)
        metrics = {
            "roc_auc":      round(float(best_auc), 4),
            "avg_precision":round(float(average_precision_score(vy, vp)), 4),
            "threshold":    round(self.threshold, 4),
            "n_train":      len(train_idx),
            "n_val":        len(val_idx),
            "mule_rate":    round(float(y.mean()), 4),
            "history":      history,
            "classification_report": classification_report(
                vy, preds, target_names=["normal", "mule"]
            ),
        }

        self._training_stats = metrics
        self.is_trained = True
        print(f"  [GraphSAGE] Training complete | ROC-AUC={best_auc:.4f} "
              f"| threshold={self.threshold:.3f}")
        return metrics

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(
        self, X_raw: np.ndarray, adj: csr_matrix
    ) -> np.ndarray:
        if not self.is_trained:
            return np.zeros(len(X_raw))
        X = self._scaler.transform(X_raw).astype(np.float32)
        probs, _ = self.forward(X, adj)
        return probs

    def predict_account_score(
        self,
        account_id: str,
        all_node_ids: list[str],
        X_all: np.ndarray,
        adj: csr_matrix,
    ) -> float:
        """Return mule probability [0,1] for a single account."""
        if not self.is_trained:
            return 0.0
        try:
            idx = all_node_ids.index(account_id)
        except ValueError:
            return 0.0
        X = self._scaler.transform(X_all).astype(np.float32)
        probs, _ = self.forward(X, adj)
        return float(probs[idx])

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        # np.savez_compressed appends .npz if not already present, so strip it first.
        weights_stem = _WEIGHTS_FILE[:-4] if _WEIGHTS_FILE.endswith(".npz") else _WEIGHTS_FILE
        np.savez_compressed(weights_stem, **{k: v for k, v in self.params.items()})
        joblib.dump(
            (self._scaler, self.threshold, self.is_trained, self._training_stats),
            _META_FILE,
        )

    def load(self) -> None:
        if not Path(_WEIGHTS_FILE).exists():
            raise FileNotFoundError(f"GraphSAGE weights not found: {_WEIGHTS_FILE}")
        data = np.load(_WEIGHTS_FILE)
        for k in data.files:
            self.params[k] = data[k]
        self._scaler, self.threshold, self.is_trained, self._training_stats = \
            joblib.load(_META_FILE)


# ── L2-norm backward ──────────────────────────────────────────────────────────

def _back_l2_norm(
    d_out: np.ndarray,
    h_normed: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Gradient of d_out through the L2-normalisation h_normed = h / ||h||.
    d_in = (I - h_normed h_normed^T) d_out / ||h||
    We approximate ||h|| ≈ 1 since h_normed already has unit norm.
    """
    dot = (d_out * h_normed).sum(axis=1, keepdims=True)
    return d_out - dot * h_normed


# ── Singleton & persistence helpers ───────────────────────────────────────────

_sage_instance: Optional[GraphSAGEModel] = None
# Cached graph data for inference without re-querying
_graph_cache: Optional[tuple] = None   # (X_raw, y, node_ids, adj)


def get_sage() -> GraphSAGEModel:
    global _sage_instance
    if _sage_instance is None:
        _sage_instance = GraphSAGEModel()
        try:
            _sage_instance.load()
        except (FileNotFoundError, Exception):
            pass
    return _sage_instance


def reset_sage() -> None:
    global _sage_instance, _graph_cache
    _sage_instance  = None
    _graph_cache    = None


def get_graph_cache() -> Optional[tuple]:
    return _graph_cache


def set_graph_cache(X, y, node_ids, adj) -> None:
    global _graph_cache
    _graph_cache = (X, y, node_ids, adj)


# ── High-level training entry point ───────────────────────────────────────────

def train_graphsage(
    max_nodes: int = 50_000,
    max_edges: int = 500_000,
    epochs: int = 60,
) -> dict:
    """
    Build graph from Neo4j, train GraphSAGE, save model and write scores
    back to Neo4j as Account.graphsage_mule_score properties.
    """
    print("[GraphSAGE] Building account graph…")
    X, y, node_ids, adj = build_account_graph(max_nodes=max_nodes, max_edges=max_edges)
    set_graph_cache(X, y, node_ids, adj)

    model = GraphSAGEModel()
    print("[GraphSAGE] Starting training…")
    metrics = model.fit(X, y, adj, epochs=epochs)
    model.save()
    reset_sage()

    # Write scores back to Neo4j (batch)
    print("[GraphSAGE] Writing scores to Neo4j…")
    probs = model.predict_proba(X, adj)
    _write_scores_to_neo4j(node_ids, probs, model.threshold)

    print(f"[GraphSAGE] Done. ROC-AUC={metrics['roc_auc']}")
    return metrics


def _write_scores_to_neo4j(
    node_ids: list[str],
    probs: np.ndarray,
    threshold: float,
    batch: int = 1000,
) -> None:
    for start in range(0, len(node_ids), batch):
        chunk_ids   = node_ids[start:start + batch]
        chunk_probs = probs[start:start + batch].tolist()
        rows = [
            {
                "id":    nid,
                "score": round(float(p) * 100, 1),
                "suspect": float(p) >= threshold,
            }
            for nid, p in zip(chunk_ids, chunk_probs)
        ]
        with neo4j_session() as s:
            s.run("""
                UNWIND $rows AS r
                MATCH (a:Account {id: r.id})
                SET a.graphsage_score   = r.score,
                    a.graphsage_suspect = r.suspect,
                    a.graphsage_scored_at = $ts
            """, rows=rows, ts=datetime.utcnow().isoformat())
