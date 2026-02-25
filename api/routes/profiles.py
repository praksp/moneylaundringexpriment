"""
Customer profile API routes.

All endpoints in this router require the requesting user to hold the
'admin' role.  Customer PII (name, nationality, account numbers, etc.)
is only visible to admins.  Viewer-role users are directed to the
/transactions/aggregate endpoint instead.
"""
import asyncio
import dataclasses
import json

from fastapi import APIRouter, HTTPException, Query, Depends
from profiles.customer_profile import (
    build_customer_profile, list_customers, get_customer_transactions_page,
)
from store.feature_store import (
    compute_and_store_feature_snapshot, get_latest_snapshot,
    list_high_risk_accounts,
)
from db.client import neo4j_session
from auth.dependencies import require_admin
from auth.models import UserInDB
from api.cache import get_cached, set_cached

router = APIRouter(prefix="/profiles", tags=["Customer Profiles"])


def _serialize(obj):
    """Recursively convert dataclasses to dicts."""
    if dataclasses.is_dataclass(obj):
        return {k: _serialize(v) for k, v in dataclasses.asdict(obj).items()}
    elif isinstance(obj, list):
        return [_serialize(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


@router.get("/")
async def list_all_customers(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=25, ge=1, le=100),
    risk_tier: str = Query(default=None, description="Filter: LOW / MEDIUM / HIGH / CRITICAL"),
    search: str = Query(default=None, description="Case-insensitive name search"),
    _admin: UserInDB = Depends(require_admin),
):
    """List customers with summary info. Supports name search and risk-tier filter. Requires admin role."""
    # Cache paginated lists for 60 s (keyed by all params)
    cache_key = f"customer_list:{skip}:{limit}:{risk_tier}:{search}"
    cached = get_cached(cache_key)
    if cached is not None:
        return cached
    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: list_customers(skip=skip, limit=limit, risk_tier=risk_tier, search=search)
    )
    set_cached(cache_key, result, ttl=60)
    return result


@router.get("/high-risk-accounts")
async def get_high_risk_accounts(
    limit: int = Query(default=50, ge=1, le=200),
    _admin: UserInDB = Depends(require_admin),
):
    """Accounts flagged as likely mules or high-risk by the feature store. Requires admin role."""
    cache_key = f"high_risk_accounts:{limit}"
    cached = get_cached(cache_key)
    if cached is not None:
        return {"accounts": cached}
    accounts = await asyncio.get_event_loop().run_in_executor(
        None, lambda: list_high_risk_accounts(limit=limit)
    )
    set_cached(cache_key, accounts, ttl=180)
    return {"accounts": accounts}


@router.get("/{customer_id}")
async def get_customer_profile(
    customer_id: str,
    _admin: UserInDB = Depends(require_admin),
):
    """
    Full customer profile including accounts, transaction history,
    risk profile, network connections, and mule indicators.
    Requires admin role — contains PII.
    """
    cache_key = f"profile:{customer_id}"
    cached = get_cached(cache_key)
    if cached is not None:
        return cached
    profile = await asyncio.get_event_loop().run_in_executor(
        None, lambda: build_customer_profile(customer_id)
    )
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    result = _serialize(profile)
    set_cached(cache_key, result, ttl=120)
    return result


@router.post("/{customer_id}/accounts/{account_id}/feature-snapshot")
async def compute_feature_snapshot(
    customer_id: str,
    account_id: str,
    _admin: UserInDB = Depends(require_admin),
):
    """Compute and persist a feature snapshot. Requires admin role."""
    snapshot = compute_and_store_feature_snapshot(customer_id, account_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Customer or account not found")
    return snapshot


@router.get("/{customer_id}/accounts/{account_id}/feature-snapshot")
async def get_feature_snapshot(
    customer_id: str,
    account_id: str,
    _admin: UserInDB = Depends(require_admin),
):
    """Retrieve the latest feature snapshot for an account. Requires admin role."""
    snapshot = get_latest_snapshot(account_id)
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail=f"No feature snapshot for account {account_id}. POST to compute one."
        )
    return snapshot


# ── Human-readable explanations for each risk factor ──────────────────────────
FACTOR_EXPLANATIONS: dict[str, dict] = {
    "in_structuring_band": {
        "title": "Structuring (CTR Avoidance)",
        "detail": "Amount is between $9,000–$9,999 — just below the $10,000 Currency Transaction Report threshold. This is a classic money-laundering technique called structuring or 'smurfing'.",
        "severity": "critical",
    },
    "is_very_high_amount": {
        "title": "Very High Amount (>$50k)",
        "detail": "Single transaction exceeds $50,000. Large single movements are a strong indicator of layering in money-laundering typologies.",
        "severity": "high",
    },
    "is_high_amount": {
        "title": "High Amount (>$10k)",
        "detail": "Transaction exceeds the $10,000 CTR reporting threshold. Transactions above this level receive heightened AML scrutiny.",
        "severity": "medium",
    },
    "is_round_amount": {
        "title": "Round Number Amount",
        "detail": "Amount is a round number (e.g. $10,000, $50,000). Legitimate transactions are rarely perfectly round — this pattern is associated with structuring or testing transactions.",
        "severity": "low",
    },
    "high_amount_deviation": {
        "title": "Unusual Amount for This Account",
        "detail": "This transaction is significantly larger than the account's typical transaction size, indicating a behavioural anomaly or account takeover.",
        "severity": "high",
    },
    "is_wire": {
        "title": "Wire Transfer",
        "detail": "Wire transfers are difficult to reverse and commonly used in the layering stage of money laundering, particularly for cross-border fund movement.",
        "severity": "low",
    },
    "is_crypto": {
        "title": "Cryptocurrency Transfer",
        "detail": "Cryptocurrency transactions are pseudonymous and irreversible, making them a high-risk channel for placement and layering.",
        "severity": "high",
    },
    "is_cash": {
        "title": "Cash Transaction",
        "detail": "Cash transactions leave no digital trail and are the primary method used in the placement phase of money laundering.",
        "severity": "medium",
    },
    "is_cross_border": {
        "title": "Cross-Border Transfer",
        "detail": "Transfer crosses national jurisdictions, complicating regulatory oversight and is a common layering technique.",
        "severity": "medium",
    },
    "is_night_transaction": {
        "title": "After-Hours Transaction",
        "detail": "Transaction occurred between midnight and 6am — outside normal business hours. Fraudsters often operate when monitoring is reduced.",
        "severity": "low",
    },
    "is_weekend": {
        "title": "Weekend Transaction",
        "detail": "Transaction occurred on a weekend when compliance monitoring may be reduced.",
        "severity": "low",
    },
    "is_new_sender_account": {
        "title": "New Sender Account (<30 days)",
        "detail": "The sending account is less than 30 days old. Mule accounts are frequently newly opened to avoid account history checks.",
        "severity": "high",
    },
    "is_dormant_sender": {
        "title": "Dormant Account Suddenly Active",
        "detail": "This account was inactive for more than 90 days and is now sending significant funds. Dormant accounts are commonly used in burst-fraud patterns.",
        "severity": "high",
    },
    "sender_is_pep": {
        "title": "Sender is a Politically Exposed Person",
        "detail": "The sender is classified as a PEP — a person in a prominent public position with elevated risk of involvement in bribery or corruption.",
        "severity": "high",
    },
    "sender_is_sanctioned": {
        "title": "Sender on Sanctions List",
        "detail": "The sending customer matches a sanctioned entity. Processing this transaction may violate OFAC, UN, or EU sanctions regulations.",
        "severity": "critical",
    },
    "receiver_is_pep": {
        "title": "Receiver is a Politically Exposed Person",
        "detail": "The receiving party is a PEP. Transfers to PEPs require enhanced due diligence under FATF Recommendation 12.",
        "severity": "medium",
    },
    "receiver_is_sanctioned": {
        "title": "Receiver on Sanctions List",
        "detail": "The receiving party is sanctioned. This transaction must be blocked and reported to the relevant regulatory authority.",
        "severity": "critical",
    },
    "sender_risk_tier": {
        "title": "High-Risk Customer Tier",
        "detail": "The customer's risk tier is elevated (HIGH or CRITICAL) based on KYC assessment, prior transaction patterns, and watchlist checks.",
        "severity": "medium",
    },
    "receiver_country_risk": {
        "title": "Receiver in High-Risk Jurisdiction",
        "detail": "The receiving account is based in a country classified as high-risk or blacklisted by FATF, increasing the likelihood of funds being used for illicit purposes.",
        "severity": "high",
    },
    "beneficiary_country_risk": {
        "title": "External Beneficiary in High-Risk Country",
        "detail": "The wire transfer destination is in a FATF grey- or black-listed country, where AML controls are considered weak or non-existent.",
        "severity": "critical",
    },
    "sender_to_high_risk": {
        "title": "Transfer to High-Risk Country",
        "detail": "Funds are being sent to a jurisdiction on the FATF blacklist or grey list (e.g. Iran, North Korea, Myanmar). Such transfers are subject to enhanced scrutiny or prohibition.",
        "severity": "critical",
    },
    "sender_to_tax_haven": {
        "title": "Transfer to Tax Haven",
        "detail": "Destination is a known tax haven (e.g. Cayman Islands, BVI, Panama). These jurisdictions are frequently used to conceal beneficial ownership in layering schemes.",
        "severity": "medium",
    },
    "ip_is_tor": {
        "title": "Transaction from Tor Network",
        "detail": "The transaction originated from a Tor exit node, indicating deliberate anonymisation of the user's identity and location. Strongly associated with fraud.",
        "severity": "critical",
    },
    "ip_is_vpn": {
        "title": "Transaction from VPN",
        "detail": "A VPN was used to mask the originating IP address and location, which may indicate intent to evade geographic controls.",
        "severity": "high",
    },
    "ip_country_mismatch": {
        "title": "IP Country Mismatch",
        "detail": "The IP address's geolocated country does not match the account's registered country, which may indicate account compromise or unauthorised use.",
        "severity": "medium",
    },
    "is_high_velocity_1h": {
        "title": "High Transaction Velocity (1 Hour)",
        "detail": "Five or more transactions were initiated from this account within the last hour, which is a strong indicator of automated fraud or rapid fund disbursement (mule activity).",
        "severity": "critical",
    },
    "is_high_velocity_24h": {
        "title": "High Transaction Velocity (24 Hours)",
        "detail": "Fifteen or more transactions were initiated in the past 24 hours, significantly above normal behavioural patterns for this account type.",
        "severity": "high",
    },
    "is_structuring": {
        "title": "Structuring Pattern Detected",
        "detail": "Multiple transactions in the $9,000–$9,999 range detected within 24 hours. This is the classic structuring pattern used to avoid Currency Transaction Reports.",
        "severity": "critical",
    },
    "is_round_trip": {
        "title": "Round-Trip Funds Movement",
        "detail": "Funds left this account and returned (directly or via intermediaries) within 48 hours. Round-tripping is used to create the appearance of legitimate business activity.",
        "severity": "critical",
    },
    "device_shared": {
        "title": "Device Used by Multiple Customers",
        "detail": "The device initiating this transaction has been used by more than one registered customer. This may indicate account sharing, identity fraud, or a fraud ring.",
        "severity": "high",
    },
    "is_deep_layering": {
        "title": "Deep Layering (3+ Network Hops)",
        "detail": "Funds have passed through 3 or more accounts before reaching this transaction. Layering through multiple accounts is used to obscure the original source of funds.",
        "severity": "critical",
    },
    "merchant_is_gambling": {
        "title": "Gambling Merchant",
        "detail": "Payment is to a gambling establishment (MCC 7995). Gambling platforms are frequently used in placement and integration stages of money laundering.",
        "severity": "medium",
    },
    "merchant_is_fx": {
        "title": "Foreign Exchange Merchant",
        "detail": "Payment is to a currency exchange (MCC 6051). FX services are a common vector for currency conversion and placement.",
        "severity": "medium",
    },
    "merchant_is_high_risk": {
        "title": "High-Risk Merchant Category",
        "detail": "This merchant operates in a category known for elevated money-laundering risk (e.g. cash advance, gambling, FX).",
        "severity": "medium",
    },
}


@router.get("/{customer_id}/transaction-map")
async def get_transaction_map(
    customer_id: str,
    _admin: UserInDB = Depends(require_admin),
):
    """
    Aggregate a customer's transactions by country for the world-map heat-map.
    Returns one entry per country code (sender OR receiver/beneficiary) with
    risk score aggregates, fraud counts, and flow direction.
    """
    with neo4j_session() as session:
        rows = session.run("""
            MATCH (c:Customer {id: $customer_id})-[:OWNS]->(a:Account)-[:INITIATED]->(t:Transaction)
            OPTIONAL MATCH (a)-[:BASED_IN]->(sc:Country)
            OPTIONAL MATCH (t)-[:CREDITED_TO]->(ra:Account)-[:BASED_IN]->(rc:Country)
            OPTIONAL MATCH (t)-[:SENT_TO_EXTERNAL]->(b:BeneficiaryAccount)
            OPTIONAL MATCH (t)-[:HAS_PREDICTION]->(pl:PredictionLog)
            RETURN
                sc.code AS sender_cc, sc.name AS sender_name,
                sc.fatf_risk AS sender_fatf,
                rc.code AS recv_cc, rc.name AS recv_name,
                rc.fatf_risk AS recv_fatf,
                b.country AS bene_cc,
                t.is_fraud AS is_fraud, t.fraud_type AS fraud_type,
                t.amount AS amount, t.currency AS currency,
                COALESCE(pl.final_score, CASE t.is_fraud
                    WHEN true THEN 720 ELSE null END) AS score,
                COALESCE(pl.outcome, CASE t.is_fraud
                    WHEN true THEN 'DECLINE' ELSE null END) AS outcome
        """, customer_id=customer_id).data()

    # Aggregate by country code
    country_stats: dict[str, dict] = {}

    def _add(code: str, name: str, fatf: str, amount: float, score,
             is_fraud: bool, fraud_type: str | None, direction: str):
        if not code:
            return
        if code not in country_stats:
            country_stats[code] = {
                "code": code,
                "name": name or code,
                "fatf_risk": fatf or "LOW",
                "txn_count": 0,
                "fraud_count": 0,
                "total_amount": 0.0,
                "scores": [],
                "directions": set(),
                "fraud_types": set(),
            }
        s = country_stats[code]
        s["txn_count"] += 1
        s["total_amount"] += float(amount or 0)
        if is_fraud:
            s["fraud_count"] += 1
            if fraud_type:
                s["fraud_types"].add(fraud_type)
        if score is not None:
            s["scores"].append(int(score))
        s["directions"].add(direction)

    for r in rows:
        amt = r.get("amount", 0) or 0
        fraud = bool(r.get("is_fraud", False))
        ftype = r.get("fraud_type")
        score = r.get("score")

        _add(r.get("sender_cc"), r.get("sender_name"), r.get("sender_fatf"),
             amt, score, fraud, ftype, "sender")
        _add(r.get("recv_cc"), r.get("recv_name"), r.get("recv_fatf"),
             amt, score, fraud, ftype, "receiver")
        if r.get("bene_cc"):
            _add(r["bene_cc"], r["bene_cc"], "HIGH",
                 amt, score, fraud, ftype, "beneficiary")

    # Compute risk level for each country bucket
    result = []
    for s in country_stats.values():
        scores = s["scores"]
        avg_score = round(sum(scores) / len(scores)) if scores else None
        max_score = max(scores) if scores else None
        fraud_pct = s["fraud_count"] / s["txn_count"] if s["txn_count"] else 0

        if max_score is not None and max_score >= 700:
            risk_level = "CRITICAL"
        elif max_score is not None and max_score >= 400:
            risk_level = "HIGH"
        elif fraud_pct > 0:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        result.append({
            "code": s["code"],
            "name": s["name"],
            "fatf_risk": s["fatf_risk"],
            "txn_count": s["txn_count"],
            "fraud_count": s["fraud_count"],
            "total_amount": round(s["total_amount"], 2),
            "avg_score": avg_score,
            "max_score": max_score,
            "risk_level": risk_level,
            "directions": list(s["directions"]),
            "fraud_types": list(s["fraud_types"]),
        })

    result.sort(key=lambda x: x["txn_count"], reverse=True)
    return {"countries": result, "total_countries": len(result)}


@router.get("/{customer_id}/transactions")
async def list_customer_transactions(
    customer_id: str,
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=500, ge=1, le=500, description="Transactions per page (max 500)"),
    _admin: UserInDB = Depends(require_admin),
):
    """
    Paginated transaction history for a customer.
    Returns 500 transactions per page by default, ordered newest-first.
    """
    result = get_customer_transactions_page(customer_id, page=page, page_size=page_size)
    if result["total"] == 0:
        # Verify the customer actually exists before returning an empty result
        from profiles.customer_profile import CUSTOMER_BASE_QUERY
        from db.client import neo4j_session
        with neo4j_session() as s:
            if s.run(CUSTOMER_BASE_QUERY, customer_id=customer_id).single() is None:
                raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    return result


@router.get("/{customer_id}/transactions/{txn_id}")
async def get_transaction_detail(
    customer_id: str,
    txn_id: str,
    _admin: UserInDB = Depends(require_admin),
):
    """
    Full transaction detail with risk score breakdown and factor explanations.
    Designed to show the analyst exactly why the risk score is the way it is.
    """
    with neo4j_session() as session:
        # Full transaction + graph context
        rec = session.run("""
            MATCH (sender:Account)-[:INITIATED]->(t:Transaction {id: $txn_id})
            OPTIONAL MATCH (t)-[:CREDITED_TO]->(receiver:Account)
            OPTIONAL MATCH (sender)<-[:OWNS]-(sc:Customer)
            OPTIONAL MATCH (receiver)<-[:OWNS]-(rc:Customer)
            OPTIONAL MATCH (t)-[:ORIGINATED_FROM]->(device:Device)
            OPTIONAL MATCH (t)-[:SOURCED_FROM]->(ip:IPAddress)
            OPTIONAL MATCH (t)-[:PAID_TO]->(merchant:Merchant)
            OPTIONAL MATCH (t)-[:SENT_TO_EXTERNAL]->(beneficiary:BeneficiaryAccount)
            OPTIONAL MATCH (sender)-[:BASED_IN]->(sender_country:Country)
            OPTIONAL MATCH (receiver)-[:BASED_IN]->(recv_country:Country)
            RETURN t, sender, receiver, sc AS sender_customer, rc AS receiver_customer,
                   device, ip, merchant, beneficiary, sender_country, recv_country
        """, txn_id=txn_id).single()

        if rec is None:
            raise HTTPException(status_code=404, detail=f"Transaction {txn_id} not found")

        def _n(node):
            return dict(node) if node else None

        txn = _n(rec["t"])
        sender = _n(rec["sender"])
        receiver = _n(rec["receiver"])
        sender_customer = _n(rec["sender_customer"])
        receiver_customer = _n(rec["receiver_customer"])
        device = _n(rec["device"])
        ip_node = _n(rec["ip"])
        merchant = _n(rec["merchant"])
        beneficiary = _n(rec["beneficiary"])
        sender_country = _n(rec["sender_country"])
        recv_country = _n(rec["recv_country"])

        # Verify this transaction belongs to the customer
        if sender_customer and sender_customer.get("id") != customer_id:
            raise HTTPException(status_code=403, detail="Transaction does not belong to this customer")

        # Prediction log for this transaction
        pred_rec = session.run("""
            MATCH (pl:PredictionLog {transaction_id: $txn_id})
            RETURN pl ORDER BY pl.timestamp DESC LIMIT 1
        """, txn_id=txn_id).single()

        prediction = None
        risk_factor_details = []
        if pred_rec:
            pl = dict(pred_rec["pl"])
            # Strip large vector from response
            pl.pop("feature_vector_json", None)
            prediction = pl

            # Enrich each triggered factor with human-readable explanation
            for factor in pl.get("risk_factors", []):
                exp = FACTOR_EXPLANATIONS.get(factor, {
                    "title": factor.replace("_", " ").title(),
                    "detail": f"Risk indicator '{factor}' was detected for this transaction.",
                    "severity": "medium",
                })
                risk_factor_details.append({
                    "factor": factor,
                    "title": exp["title"],
                    "detail": exp["detail"],
                    "severity": exp["severity"],
                })

            # Sort by severity
            sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            risk_factor_details.sort(key=lambda x: sev_order.get(x["severity"], 99))

        # Velocity context for "why" the score is what it is
        velocity_context = session.run("""
            MATCH (a:Account {id: $account_id})-[:INITIATED]->(tx:Transaction)
            WHERE tx.timestamp <= $ts AND tx.id <> $txn_id
            WITH tx ORDER BY tx.timestamp DESC
            WITH collect(tx)[0..50] AS recent
            RETURN
              size([x IN recent WHERE x.timestamp >= $ts_1h]) AS count_1h,
              size([x IN recent WHERE x.timestamp >= $ts_24h]) AS count_24h,
              reduce(s=0.0, x IN [x IN recent WHERE x.amount >= 9000 AND x.amount < 10000] | s + 1) AS structuring_count
        """,
            account_id=sender.get("id", "") if sender else "",
            txn_id=txn_id,
            ts=txn.get("timestamp", ""),
            ts_1h="1970-01-01",
            ts_24h="1970-01-01",
        ).single() if sender else None

    return {
        "transaction": txn,
        "sender_account": sender,
        "receiver_account": receiver,
        "sender_customer": sender_customer,
        "receiver_customer": receiver_customer,
        "device": device,
        "ip_address": ip_node,
        "merchant": merchant,
        "beneficiary": beneficiary,
        "sender_country": sender_country,
        "receiver_country": recv_country,
        "prediction": prediction,
        "risk_factor_details": risk_factor_details,
        "score_explanation": _build_score_explanation(prediction, txn),
    }


def _build_score_explanation(prediction: dict | None, txn: dict | None) -> dict:
    """Build a narrative explanation of the risk score."""
    if not prediction:
        return {"summary": "No risk evaluation has been run for this transaction yet."}

    score = prediction.get("final_score", 0)
    bayesian = prediction.get("bayesian_score", 0)
    ml = prediction.get("ml_score", 0)
    outcome = prediction.get("outcome", "ALLOW")
    factors = prediction.get("risk_factors", [])
    confidence = prediction.get("confidence", 0)

    # Score range narrative
    if score <= 399:
        range_text = "within the safe range (0–399)"
        risk_level = "LOW"
    elif score <= 699:
        range_text = "in the elevated risk range (400–699)"
        risk_level = "MEDIUM"
    else:
        range_text = "in the high-risk range (700–999)"
        risk_level = "HIGH"

    factor_count = len(factors)
    top_factors = factors[:3] if factors else []
    top_factors_readable = [f.replace("_", " ") for f in top_factors]

    # Bayesian vs ML agreement
    diff = abs(bayesian - ml)
    if diff > 200:
        model_agreement = "Note: the Bayesian rule engine and ML model disagreed significantly on this transaction, which may warrant manual review."
    elif diff > 100:
        model_agreement = "The Bayesian engine and ML model had moderate disagreement on this transaction."
    else:
        model_agreement = "Both the Bayesian engine and ML model were in close agreement on this transaction."

    summary_parts = [
        f"This transaction scored {score}/999, placing it {range_text}.",
    ]
    if factor_count > 0:
        summary_parts.append(
            f"{factor_count} risk factor{'s were' if factor_count > 1 else ' was'} triggered, "
            f"led by: {', '.join(top_factors_readable)}."
        )
    else:
        summary_parts.append("No significant risk factors were detected.")

    summary_parts.append(model_agreement)

    if outcome == "DECLINE":
        summary_parts.append(
            "The transaction was DECLINED due to the combination of high-risk signals. "
            "A SAR (Suspicious Activity Report) investigation is recommended."
        )
    elif outcome == "CHALLENGE":
        summary_parts.append(
            "The transaction was CHALLENGED — the customer was asked to provide additional verification "
            "before proceeding."
        )
    else:
        summary_parts.append("The transaction was ALLOWED as risk levels are within acceptable thresholds.")

    return {
        "summary": " ".join(summary_parts),
        "score": score,
        "risk_level": risk_level,
        "bayesian_contribution": f"Bayesian engine weighted score: {round(bayesian * 0.55)} (55% weight)",
        "ml_contribution": f"ML model weighted score: {round(ml * 0.45)} (45% weight)",
        "factor_count": factor_count,
        "confidence_pct": round(confidence * 100, 1),
        "model_agreement": model_agreement,
    }
