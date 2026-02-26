"""
Transaction submission route.
Accepts a new transaction, stores it in Neo4j, evaluates it, and returns the risk score.
"""
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from db.client import neo4j_session
from db.models import TransactionEvaluationResponse, TransactionType, TransactionChannel
from risk.engine import evaluate_transaction_by_id

router = APIRouter(prefix="/submit", tags=["Transaction Submission"])

# Magic answer that always passes a challenge — used for testing
CHALLENGE_MAGIC_ANSWER = "TEST"


class SubmitTransactionRequest(BaseModel):
    sender_account_id: str = Field(..., description="ID of the sending account")
    receiver_account_id: Optional[str] = Field(None, description="ID of receiving account (internal)")
    beneficiary_country: Optional[str] = Field(None, description="ISO country code for external wire")
    beneficiary_name: Optional[str] = None
    beneficiary_bank_swift: Optional[str] = None
    amount: float = Field(..., gt=0, description="Transaction amount")
    currency: str = Field(default="USD")
    transaction_type: TransactionType = TransactionType.ACH
    channel: TransactionChannel = TransactionChannel.ONLINE
    description: Optional[str] = None
    device_id: Optional[str] = None
    ip_address: Optional[str] = None


@router.post("/transaction", response_model=TransactionEvaluationResponse)
async def submit_transaction(req: SubmitTransactionRequest):
    """
    Submit a new transaction. It will be:
      1. Persisted to the Neo4j graph
      2. Evaluated by the risk engine (Bayesian + ML)
      3. Returned with risk score and outcome decision

    If outcome is CHALLENGE, a challenge question is included in the response.
    """
    txn_id = str(uuid.uuid4())
    reference = f"TXN{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.utcnow().isoformat()

    with neo4j_session() as session:
        # Verify sender account exists and check balance
        sender_check = session.run(
            "MATCH (a:Account {id: $id}) RETURN a.id AS id, a.balance AS balance, a.currency AS currency",
            id=req.sender_account_id
        ).single()
        if sender_check is None:
            raise HTTPException(
                status_code=404,
                detail=f"Sender account {req.sender_account_id} not found. "
                       "Use GET /transactions/ to browse existing accounts."
            )
        
        balance = sender_check["balance"] or 0.0
        currency = sender_check["currency"] or "USD"
        
        # Business Rule: Ensure sufficient balance
        if req.amount > balance:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient funds. Transfer amount ({req.amount:,.2f} {req.currency}) exceeds available balance ({balance:,.2f} {currency}) for account {req.sender_account_id}."
            )

        # Store transaction node
        session.run("""
            MERGE (t:Transaction {id: $id})
            SET t += {
                reference: $reference,
                amount: $amount,
                currency: $currency,
                exchange_rate: 1.0,
                amount_usd: $amount,
                transaction_type: $txn_type,
                channel: $channel,
                status: 'PENDING',
                timestamp: $timestamp,
                description: $description,
                is_fraud: false,
                fraud_type: null
            }
            WITH t
            MATCH (sender:Account {id: $sender_id})
            MERGE (sender)-[:INITIATED]->(t)
        """,
            id=txn_id, reference=reference, amount=req.amount,
            currency=req.currency, txn_type=req.transaction_type.value,
            channel=req.channel.value, timestamp=timestamp,
            description=req.description or "", sender_id=req.sender_account_id,
        )

        # Link receiver if internal
        if req.receiver_account_id:
            session.run("""
                MATCH (t:Transaction {id: $txn_id})
                MATCH (recv:Account {id: $recv_id})
                MERGE (t)-[:CREDITED_TO]->(recv)
            """, txn_id=txn_id, recv_id=req.receiver_account_id)

        # Create beneficiary for external wire
        if req.beneficiary_country and not req.receiver_account_id:
            ben_id = str(uuid.uuid4())
            session.run("""
                MERGE (b:BeneficiaryAccount {id: $id})
                SET b += {
                    account_number: $acct_num,
                    account_name: $name,
                    bank_swift: $swift,
                    country: $country,
                    currency: $currency
                }
                WITH b
                MATCH (t:Transaction {id: $txn_id})
                MERGE (t)-[:SENT_TO_EXTERNAL]->(b)
            """,
                id=ben_id,
                acct_num=f"EXT{uuid.uuid4().hex[:8].upper()}",
                name=req.beneficiary_name or "External Beneficiary",
                swift=req.beneficiary_bank_swift or "UNKNOWN",
                country=req.beneficiary_country,
                currency=req.currency,
                txn_id=txn_id,
            )

        # Link device/IP if provided
        if req.device_id:
            session.run("""
                MATCH (t:Transaction {id: $txn_id})
                MATCH (d:Device {id: $device_id})
                MERGE (t)-[:ORIGINATED_FROM]->(d)
            """, txn_id=txn_id, device_id=req.device_id)

        if req.ip_address:
            session.run("""
                MATCH (t:Transaction {id: $txn_id})
                MATCH (i:IPAddress {ip: $ip})
                MERGE (t)-[:SOURCED_FROM]->(i)
            """, txn_id=txn_id, ip=req.ip_address)

    # Evaluate and return
    try:
        result = evaluate_transaction_by_id(txn_id)
        # Update transaction status
        with neo4j_session() as session:
            status = "COMPLETED" if result["risk_score"].outcome.value != "DECLINE" else "FLAGGED"
            session.run(
                "MATCH (t:Transaction {id: $id}) SET t.status = $status",
                id=txn_id, status=status
            )
        return TransactionEvaluationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


class VerifyChallengeRequest(BaseModel):
    transaction_id: str
    question_id: str
    answer: str


@router.post("/verify-challenge")
async def verify_challenge(req: VerifyChallengeRequest):
    """
    Verify a challenge answer.
    If the answer matches the magic word 'TEST' (case-insensitive),
    the transaction is committed and its status set to COMPLETED.
    Any other answer is rejected.
    """
    if req.answer.strip().upper() != CHALLENGE_MAGIC_ANSWER:
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "message": "Incorrect answer. The transaction remains on hold.",
            },
        )

    with neo4j_session() as session:
        record = session.run(
            "MATCH (t:Transaction {id: $id}) RETURN t.status AS status, t.reference AS ref",
            id=req.transaction_id,
        ).single()

        if record is None:
            raise HTTPException(status_code=404, detail=f"Transaction {req.transaction_id} not found")

        # Commit the transaction
        session.run(
            "MATCH (t:Transaction {id: $id}) SET t.status = 'COMPLETED'",
            id=req.transaction_id,
        )

    return {
        "success": True,
        "transaction_id": req.transaction_id,
        "new_status": "COMPLETED",
        "message": "Challenge passed. Transaction has been committed successfully.",
    }


@router.get("/accounts")
async def list_accounts_for_form(limit: int = 100):
    """Get account list to populate the transaction form dropdowns."""
    with neo4j_session() as session:
        records = list(session.run("""
            MATCH (c:Customer)-[:OWNS]->(a:Account)
            WHERE a.status = 'ACTIVE'
            RETURN a.id AS id, a.account_number AS account_number,
                   a.account_type AS type, a.currency AS currency,
                   a.balance AS balance, a.country AS country,
                   c.name AS customer_name, c.id AS customer_id
            ORDER BY c.name LIMIT $limit
        """, limit=limit))
    return {"accounts": [dict(r) for r in records]}
