"""Customer profile API routes."""
from fastapi import APIRouter, HTTPException, Query
from profiles.customer_profile import build_customer_profile, list_customers
from store.feature_store import (
    compute_and_store_feature_snapshot, get_latest_snapshot,
    list_high_risk_accounts,
)
import dataclasses

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
    limit: int = Query(default=20, ge=1, le=100),
    risk_tier: str = Query(default=None, description="Filter: LOW / MEDIUM / HIGH / CRITICAL"),
):
    """List all customers with summary info."""
    return list_customers(skip=skip, limit=limit, risk_tier=risk_tier)


@router.get("/high-risk-accounts")
async def get_high_risk_accounts(limit: int = Query(default=50, ge=1, le=200)):
    """Accounts flagged as likely mules or high-risk by the feature store."""
    return {"accounts": list_high_risk_accounts(limit=limit)}


@router.get("/{customer_id}")
async def get_customer_profile(customer_id: str):
    """
    Full customer profile including accounts, transaction history,
    risk profile, network connections, and mule indicators.
    """
    profile = build_customer_profile(customer_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    return _serialize(profile)


@router.post("/{customer_id}/accounts/{account_id}/feature-snapshot")
async def compute_feature_snapshot(customer_id: str, account_id: str):
    """
    Compute and persist a feature snapshot for a specific account.
    Used to populate the feature store and detect mule patterns.
    """
    snapshot = compute_and_store_feature_snapshot(customer_id, account_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Customer or account not found")
    return snapshot


@router.get("/{customer_id}/accounts/{account_id}/feature-snapshot")
async def get_feature_snapshot(customer_id: str, account_id: str):
    """Retrieve the latest feature snapshot for an account."""
    snapshot = get_latest_snapshot(account_id)
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail=f"No feature snapshot for account {account_id}. POST to compute one."
        )
    return snapshot
