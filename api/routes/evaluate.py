"""Transaction risk evaluation routes."""
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional

from db.models import (
    TransactionEvaluationRequest,
    TransactionEvaluationResponse,
    TransactionType,
    TransactionChannel,
)
from risk.engine import evaluate_transaction_by_id, evaluate_transaction_inline

router = APIRouter(prefix="/evaluate", tags=["Risk Evaluation"])


@router.post("/{txn_id}", response_model=TransactionEvaluationResponse)
async def evaluate_by_id(txn_id: str):
    """
    Evaluate an existing transaction stored in the Neo4j graph.
    Returns risk score (0–999), outcome (ALLOW/CHALLENGE/DECLINE),
    risk factors, and challenge question if applicable.
    """
    try:
        result = evaluate_transaction_by_id(txn_id)
        return TransactionEvaluationResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk evaluation error: {str(e)}")


class InlineEvaluationRequest(BaseModel):
    """Evaluate a transaction before it is stored — for real-time pre-submission checks."""
    sender_account_id: str
    receiver_account_id: Optional[str] = None
    amount: float = Field(gt=0)
    currency: str = "USD"
    transaction_type: TransactionType = TransactionType.ACH
    channel: TransactionChannel = TransactionChannel.ONLINE
    description: Optional[str] = None
    # Optional graph context hints (if available at evaluation time)
    sender_country: Optional[str] = "US"
    receiver_country: Optional[str] = "US"
    beneficiary_country: Optional[str] = None
    sender_pep: bool = False
    sender_kyc_level: str = "BASIC"
    sender_account_age_days: int = 365
    is_dormant_sender: bool = False
    ip_is_vpn: bool = False
    ip_is_tor: bool = False
    ip_country_mismatch: bool = False
    txn_count_last_1h: int = 0
    txn_count_last_24h: int = 0


@router.post("/inline/evaluate", response_model=TransactionEvaluationResponse)
async def evaluate_inline(request: InlineEvaluationRequest):
    """
    Evaluate a transaction inline (real-time, before storage).
    Provide transaction details and optional context signals.
    """
    import uuid
    from datetime import datetime

    txn_data = {
        "id": str(uuid.uuid4()),
        "sender_account_id": request.sender_account_id,
        "receiver_account_id": request.receiver_account_id,
        "amount": request.amount,
        "amount_usd": request.amount,
        "currency": request.currency,
        "transaction_type": request.transaction_type.value,
        "channel": request.channel.value,
        "description": request.description,
        "timestamp": datetime.utcnow().isoformat(),
    }

    graph_data = {
        "sender": {
            "id": request.sender_account_id,
            "country": request.sender_country or "US",
            "created_at": (
                datetime.utcnow().replace(
                    year=datetime.utcnow().year - (request.sender_account_age_days // 365 or 1)
                )
            ).isoformat(),
            "last_active": datetime.utcnow().isoformat() if not request.is_dormant_sender else
                           (datetime.utcnow().replace(year=datetime.utcnow().year - 1)).isoformat(),
            "typical_transaction_size": 500,
            "status": "ACTIVE",
        },
        "receiver": {
            "id": request.receiver_account_id or "",
            "country": request.receiver_country or "US",
            "created_at": datetime.utcnow().isoformat(),
        } if request.receiver_account_id else {},
        "sender_customer": {
            "pep_flag": request.sender_pep,
            "sanctions_flag": False,
            "kyc_level": request.sender_kyc_level,
            "risk_tier": "MEDIUM" if request.sender_pep else "LOW",
        },
        "receiver_customer": {},
        "device": {},
        "ip": {
            "is_vpn": request.ip_is_vpn,
            "is_tor": request.ip_is_tor,
            "country": request.receiver_country or "US",
        },
        "merchant": {},
        "beneficiary": {
            "country": request.beneficiary_country,
        } if request.beneficiary_country else {},
        "txn_count_1h": request.txn_count_last_1h,
        "txn_count_24h": request.txn_count_last_24h,
        "txn_count_7d": 0,
        "total_amount_24h": 0.0,
        "total_amount_7d": 0.0,
        "structuring_count_24h": 1 if 9000 <= request.amount < 10000 else 0,
        "round_trip_count": 0,
        "shared_device_user_count": 1,
        "network_hop_count": 1,
        "ip_country_mismatch": request.ip_country_mismatch,
    }

    try:
        result = evaluate_transaction_inline(txn_data, graph_data)
        return TransactionEvaluationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk evaluation error: {str(e)}")


@router.post("/batch/evaluate")
async def evaluate_batch(
    txn_ids: list[str] = Body(..., description="List of transaction IDs to evaluate"),
):
    """Evaluate multiple transactions in sequence. Returns list of risk evaluations."""
    results = []
    errors = []
    for txn_id in txn_ids[:50]:  # Cap at 50 per batch
        try:
            result = evaluate_transaction_by_id(txn_id)
            results.append({
                "transaction_id": txn_id,
                "score": result["risk_score"].score,
                "outcome": result["risk_score"].outcome.value,
                "risk_factors": result["risk_score"].risk_factors[:5],
            })
        except Exception as e:
            errors.append({"transaction_id": txn_id, "error": str(e)})

    return {"evaluated": len(results), "errors": len(errors), "results": results, "failed": errors}
