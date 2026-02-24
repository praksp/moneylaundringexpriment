"""
Feature Store
==============
Persists computed feature snapshots for customers/accounts in Neo4j.
Features are versioned so drift can be detected over time.

Feature groups stored:
  - Identity features     : KYC, PEP, sanctions, risk tier
  - Behavioral features   : transaction velocity, amounts, patterns
  - Network features      : counterparty count, centrality, hop depth
  - Mule indicators       : turnover ratio, hold time, structuring frequency
  - Risk history features : avg/max score, fraud incident rate
  - Device/IP features    : VPN/Tor usage, device sharing

Node: FeatureSnapshot
  - Links via [:FEATURE_SNAPSHOT_FOR]->(Account)
  - Links via [:BELONGS_TO]->(Customer)
"""
import json
import uuid
from datetime import datetime, timedelta
from typing import Optional
from db.client import neo4j_session

FEATURE_VERSION = "1.0"

COMPUTE_FEATURES_QUERY = """
MATCH (c:Customer {id: $customer_id})-[:OWNS]->(a:Account {id: $account_id})
OPTIONAL MATCH (a)-[:INITIATED]->(t_out:Transaction)
WHERE t_out.timestamp >= $since_30d
OPTIONAL MATCH (a)<-[:CREDITED_TO]-(t_in:Transaction)
WHERE t_in.timestamp >= $since_30d
OPTIONAL MATCH (a)-[:INITIATED]->(t_str:Transaction)
WHERE t_str.amount >= 9000 AND t_str.amount < 10000
  AND t_str.timestamp >= $since_30d
OPTIONAL MATCH (a)-[:INITIATED]->(t_vpn:Transaction)-[:SOURCED_FROM]->(ip:IPAddress)
WHERE ip.is_vpn = true AND t_vpn.timestamp >= $since_30d
OPTIONAL MATCH (a)-[:INITIATED]->(t_tor:Transaction)-[:SOURCED_FROM]->(ip2:IPAddress)
WHERE ip2.is_tor = true AND t_tor.timestamp >= $since_30d
OPTIONAL MATCH (a)-[:INITIATED]->(t_out2:Transaction)-[:CREDITED_TO]->(recv:Account)
WHERE t_out2.timestamp >= $since_30d
OPTIONAL MATCH (a)<-[:CREDITED_TO]-(t_in2:Transaction)<-[:INITIATED]-(sender:Account)
WHERE t_in2.timestamp >= $since_30d
OPTIONAL MATCH (a)-[:BASED_IN]->(country:Country)
WITH a, c, country,
     count(DISTINCT t_out) AS out_count,
     sum(t_out.amount) AS out_vol,
     count(DISTINCT t_in) AS in_count,
     sum(t_in.amount) AS in_vol,
     count(DISTINCT t_str) AS struct_count,
     count(DISTINCT t_vpn) AS vpn_count,
     count(DISTINCT t_tor) AS tor_count,
     count(DISTINCT recv) AS unique_receivers,
     count(DISTINCT sender) AS unique_senders
RETURN a, c, country,
       out_count, out_vol, in_count, in_vol,
       struct_count, vpn_count, tor_count,
       unique_receivers, unique_senders
"""

PREDICTION_HISTORY_QUERY = """
MATCH (a:Account {id: $account_id})-[:INITIATED]->(t:Transaction)
MATCH (pl:PredictionLog {transaction_id: t.id})
WHERE pl.timestamp >= $since_30d
RETURN avg(pl.final_score) AS avg_score,
       max(pl.final_score) AS max_score,
       count(pl) AS eval_count,
       count(CASE WHEN pl.outcome = 'DECLINE' THEN 1 END) AS decline_count
"""


def compute_and_store_feature_snapshot(customer_id: str, account_id: str) -> dict:
    """
    Compute a full feature snapshot for a customer/account and persist it.
    Returns the snapshot dict.
    """
    now = datetime.utcnow()
    since_30d = (now - timedelta(days=30)).isoformat()
    since_7d = (now - timedelta(days=7)).isoformat()

    with neo4j_session() as session:
        # Core feature computation
        rec = session.run(
            COMPUTE_FEATURES_QUERY,
            customer_id=customer_id,
            account_id=account_id,
            since_30d=since_30d,
        ).single()

        if rec is None:
            return {}

        a = dict(rec["a"])
        c = dict(rec["c"])
        country = dict(rec["country"]) if rec["country"] else {}

        out_count = int(rec["out_count"] or 0)
        out_vol = float(rec["out_vol"] or 0)
        in_count = int(rec["in_count"] or 0)
        in_vol = float(rec["in_vol"] or 0)
        struct_count = int(rec["struct_count"] or 0)
        vpn_count = int(rec["vpn_count"] or 0)
        tor_count = int(rec["tor_count"] or 0)
        unique_receivers = int(rec["unique_receivers"] or 0)
        unique_senders = int(rec["unique_senders"] or 0)

        # Prediction history
        pred_rec = session.run(
            PREDICTION_HISTORY_QUERY,
            account_id=account_id, since_30d=since_30d
        ).single()

        avg_score = float(pred_rec["avg_score"] or 0) if pred_rec else 0
        max_score = int(pred_rec["max_score"] or 0) if pred_rec else 0
        eval_count = int(pred_rec["eval_count"] or 0) if pred_rec else 0
        decline_count = int(pred_rec["decline_count"] or 0) if pred_rec else 0

        # Account age
        try:
            created = datetime.fromisoformat(str(a.get("created_at", "")).replace("Z", ""))
            acct_age_days = (now - created).days
        except Exception:
            acct_age_days = 365

        try:
            last_active = datetime.fromisoformat(str(a.get("last_active", "")).replace("Z", ""))
            days_since_active = (now - last_active).days
        except Exception:
            days_since_active = 0

        # Derived features
        turnover_ratio = out_vol / max(in_vol, 1)
        country_risk = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "BLACKLIST": 3}.get(
            country.get("fatf_risk", "LOW"), 0
        )

        snapshot = {
            "id": str(uuid.uuid4()),
            "customer_id": customer_id,
            "account_id": account_id,
            "computed_at": now.isoformat(),
            "feature_version": FEATURE_VERSION,

            # Identity
            "kyc_level": c.get("kyc_level", "BASIC"),
            "pep_flag": bool(c.get("pep_flag", False)),
            "sanctions_flag": bool(c.get("sanctions_flag", False)),
            "risk_tier": c.get("risk_tier", "LOW"),
            "customer_type": c.get("customer_type", "INDIVIDUAL"),
            "country_risk_score": country_risk,
            "is_sanctioned_country": bool(country.get("is_sanctioned", False)),
            "is_tax_haven": bool(country.get("is_tax_haven", False)),

            # Account health
            "account_age_days": acct_age_days,
            "days_since_active": days_since_active,
            "is_new_account": acct_age_days < 30,
            "is_dormant": days_since_active > 90,
            "account_type": a.get("account_type", "CURRENT"),
            "account_balance": float(a.get("balance", 0)),

            # Behavioral (30d)
            "outbound_txn_count_30d": out_count,
            "inbound_txn_count_30d": in_count,
            "outbound_volume_30d": round(out_vol, 2),
            "inbound_volume_30d": round(in_vol, 2),
            "net_flow_30d": round(in_vol - out_vol, 2),
            "turnover_ratio_30d": round(turnover_ratio, 4),
            "avg_outbound_amount_30d": round(out_vol / max(out_count, 1), 2),

            # Mule indicators
            "is_pass_through": 0.7 <= turnover_ratio <= 1.3 and in_vol > 1000,
            "unique_senders_30d": unique_senders,
            "unique_receivers_30d": unique_receivers,
            "structuring_count_30d": struct_count,
            "has_structuring_pattern": struct_count >= 2,
            "high_sender_diversity": unique_senders > 5,

            # Device/IP risk
            "vpn_transaction_count_30d": vpn_count,
            "tor_transaction_count_30d": tor_count,
            "has_tor_activity": tor_count > 0,
            "has_vpn_activity": vpn_count > 0,

            # Risk history (30d)
            "avg_risk_score_30d": round(avg_score, 1),
            "max_risk_score_30d": max_score,
            "evaluation_count_30d": eval_count,
            "decline_count_30d": decline_count,
            "decline_rate_30d": round(decline_count / max(eval_count, 1), 4),

            # Composite mule score (0–100)
            "mule_score": _compute_mule_score(
                is_pass_through=0.7 <= turnover_ratio <= 1.3 and in_vol > 1000,
                high_sender_count=unique_senders > 5,
                has_structuring=struct_count >= 2,
                is_dormant=days_since_active > 90,
                has_tor=tor_count > 0,
                decline_rate=decline_count / max(eval_count, 1),
            ),
        }

        snapshot["is_likely_mule"] = snapshot["mule_score"] >= 50

        # Persist to Neo4j
        session.run("""
            MERGE (fs:FeatureSnapshot {id: $id})
            SET fs += {
                customer_id: $customer_id,
                account_id: $account_id,
                computed_at: $computed_at,
                feature_version: $feature_version,
                mule_score: $mule_score,
                is_likely_mule: $is_likely_mule,
                turnover_ratio_30d: $turnover_ratio_30d,
                unique_senders_30d: $unique_senders_30d,
                structuring_count_30d: $structuring_count_30d,
                has_tor_activity: $has_tor_activity,
                avg_risk_score_30d: $avg_risk_score_30d,
                max_risk_score_30d: $max_risk_score_30d,
                is_dormant: $is_dormant,
                is_pass_through: $is_pass_through,
                account_age_days: $account_age_days,
                risk_tier: $risk_tier,
                pep_flag: $pep_flag
            }
            WITH fs
            MATCH (a:Account {id: $account_id})
            MERGE (fs)-[:FEATURE_SNAPSHOT_FOR]->(a)
            WITH fs
            MATCH (c:Customer {id: $customer_id})
            MERGE (fs)-[:BELONGS_TO]->(c)
        """, **{k: v for k, v in snapshot.items() if isinstance(v, (str, int, float, bool))},
        id=snapshot["id"], customer_id=customer_id, account_id=account_id)

    return snapshot


def get_latest_snapshot(account_id: str) -> Optional[dict]:
    """Retrieve the most recent feature snapshot for an account."""
    with neo4j_session() as session:
        rec = session.run("""
            MATCH (fs:FeatureSnapshot)-[:FEATURE_SNAPSHOT_FOR]->(a:Account {id: $account_id})
            RETURN fs ORDER BY fs.computed_at DESC LIMIT 1
        """, account_id=account_id).single()
    return dict(rec["fs"]) if rec else None


def get_snapshots_over_time(account_id: str, days: int = 30) -> list[dict]:
    """Get all snapshots for drift analysis."""
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()
    with neo4j_session() as session:
        records = list(session.run("""
            MATCH (fs:FeatureSnapshot)-[:FEATURE_SNAPSHOT_FOR]->(a:Account {id: $account_id})
            WHERE fs.computed_at >= $since
            RETURN fs ORDER BY fs.computed_at ASC
        """, account_id=account_id, since=since))
    return [dict(r["fs"]) for r in records]


def list_high_risk_accounts(limit: int = 50) -> list[dict]:
    """
    Return accounts flagged as high-risk.

    Priority order:
    1. Accounts WITH a FeatureSnapshot (manually computed)
    2. Accounts flagged by GraphSAGE or KNN anomaly detector
    3. Accounts with a high fraud-transaction ratio (computed on the fly)
    """
    results: list[dict] = []
    seen: set[str] = set()

    with neo4j_session() as session:
        # ── Tier 1: FeatureSnapshot nodes ──────────────────────────────────────
        records = list(session.run("""
            MATCH (fs:FeatureSnapshot)-[:FEATURE_SNAPSHOT_FOR]->(a:Account)
            MATCH (fs)-[:BELONGS_TO]->(c:Customer)
            WHERE fs.is_likely_mule = true OR fs.mule_score >= 40
               OR fs.has_tor_activity = true
            RETURN a.id AS account_id, a.account_number AS account_number,
                   c.id AS customer_id, c.name AS customer_name,
                   fs.mule_score             AS mule_score,
                   fs.is_likely_mule         AS is_likely_mule,
                   fs.avg_risk_score_30d     AS avg_risk_score,
                   fs.turnover_ratio_30d     AS turnover_ratio,
                   fs.has_tor_activity       AS tor_activity,
                   fs.unique_senders_30d     AS unique_senders,
                   fs.structuring_count_30d  AS structuring_count,
                   fs.outbound_volume_30d    AS out_volume,
                   fs.inbound_volume_30d     AS in_volume,
                   fs.evaluation_count_30d   AS eval_count,
                   fs.decline_count_30d      AS decline_count,
                   fs.pep_flag               AS pep_flag,
                   fs.risk_tier              AS risk_tier,
                   fs.computed_at            AS computed_at,
                   'feature_snapshot'        AS source
            ORDER BY fs.mule_score DESC
            LIMIT $limit
        """, limit=limit))
        for r in records:
            d = dict(r)
            seen.add(d["account_id"])
            results.append(d)

        still_need = limit - len(results)
        if still_need <= 0:
            return results

        # ── Tier 2: GraphSAGE / KNN flagged accounts ─────────────────────────
        records2 = list(session.run("""
            MATCH (a:Account)
            WHERE (a.graphsage_suspect = true OR a.mule_suspect = true)
              AND NOT a.id IN $seen
            OPTIONAL MATCH (c:Customer)-[:OWNS]->(a)
            OPTIONAL MATCH (a)-[:INITIATED]->(t_out:Transaction)
            OPTIONAL MATCH (t_in:Transaction)-[:CREDITED_TO]->(a)
            WITH a, c,
              coalesce(a.graphsage_score, 0)  AS sage_score,
              coalesce(a.anomaly_score, 0)    AS knn_score,
              count(DISTINCT t_out)           AS out_count,
              coalesce(sum(t_out.amount_usd), 0) AS out_vol,
              coalesce(sum(t_in.amount_usd), 0)  AS in_vol,
              coalesce(sum(CASE WHEN t_out.is_fraud THEN 1.0 ELSE 0.0 END), 0) AS fraud_count
            RETURN
              a.id             AS account_id,
              a.account_number AS account_number,
              c.id             AS customer_id,
              c.name           AS customer_name,
              // Synthesise a mule_score from GraphSAGE + KNN
              toInteger((sage_score * 0.6 + knn_score * 0.4))
                               AS mule_score,
              a.graphsage_suspect AS is_likely_mule,
              0                AS avg_risk_score,
              CASE WHEN in_vol > 0 THEN out_vol / in_vol ELSE 0 END
                               AS turnover_ratio,
              false            AS tor_activity,
              null             AS unique_senders,
              null             AS structuring_count,
              out_vol          AS out_volume,
              in_vol           AS in_volume,
              out_count        AS eval_count,
              fraud_count      AS decline_count,
              false            AS pep_flag,
              'UNKNOWN'        AS risk_tier,
              a.graphsage_scored_at AS computed_at,
              'graphsage_knn'  AS source
            ORDER BY sage_score DESC
            LIMIT $n
        """, seen=list(seen), n=still_need))

        for r in records2:
            d = dict(r)
            seen.add(d["account_id"])
            results.append(d)

        still_need2 = limit - len(results)
        if still_need2 <= 0:
            return results

        # ── Tier 3: Accounts with ≥30% fraud txns (computed on the fly) ──────
        records3 = list(session.run("""
            MATCH (a:Account)-[:INITIATED]->(t:Transaction)
            WHERE NOT a.id IN $seen
            WITH a,
              count(t) AS total,
              sum(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) AS fraud_c
            WHERE total > 0 AND toFloat(fraud_c) / total >= 0.30
            OPTIONAL MATCH (c:Customer)-[:OWNS]->(a)
            OPTIONAL MATCH (a)-[:INITIATED]->(t2:Transaction)
            OPTIONAL MATCH (t_in:Transaction)-[:CREDITED_TO]->(a)
            WITH a, c, total, fraud_c,
              coalesce(sum(t2.amount_usd), 0) AS out_vol,
              coalesce(sum(t_in.amount_usd), 0) AS in_vol
            RETURN
              a.id             AS account_id,
              a.account_number AS account_number,
              c.id             AS customer_id,
              c.name           AS customer_name,
              toInteger(toFloat(fraud_c) / total * 100)
                               AS mule_score,
              true             AS is_likely_mule,
              0                AS avg_risk_score,
              CASE WHEN in_vol > 0 THEN out_vol / in_vol ELSE 0 END
                               AS turnover_ratio,
              false            AS tor_activity,
              null             AS unique_senders,
              null             AS structuring_count,
              out_vol          AS out_volume,
              in_vol           AS in_volume,
              total            AS eval_count,
              fraud_c          AS decline_count,
              false            AS pep_flag,
              'HIGH'           AS risk_tier,
              null             AS computed_at,
              'fraud_ratio'    AS source
            ORDER BY mule_score DESC
            LIMIT $n
        """, seen=list(seen), n=still_need2))
        for r in records3:
            results.append(dict(r))

    results.sort(key=lambda x: float(x.get("mule_score") or 0), reverse=True)
    return results[:limit]


def _compute_mule_score(is_pass_through: bool, high_sender_count: bool,
                         has_structuring: bool, is_dormant: bool,
                         has_tor: bool, decline_rate: float) -> int:
    score = 0
    if is_pass_through:
        score += 30
    if high_sender_count:
        score += 20
    if has_structuring:
        score += 15
    if is_dormant:
        score += 10
    if has_tor:
        score += 20
    if decline_rate > 0.3:
        score += 5
    return min(100, score)
