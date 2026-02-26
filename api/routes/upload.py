"""
CSV Upload and Bulk Transaction Processing Route.
Handles async ingestion of bulk transactions via CSV, chunking, graph insertion,
and subsequent incremental model training.
"""
import csv
import io
import uuid
import asyncio
import threading
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Depends

from db.client import neo4j_session
from ml.incremental import run_incremental
from auth.dependencies import require_admin
from auth.models import UserInDB

router = APIRouter(prefix="/upload", tags=["Bulk Upload"])

# Tracking state for long-running uploads
class UploadJobState:
    def __init__(self):
        self.running = False
        self.total_records = 0
        self.processed_records = 0
        self.error = None
        self.status = "idle"
        self.training_result = None

_upload_state = UploadJobState()
_upload_lock = threading.Lock()

def _process_csv_background(csv_content: str):
    global _upload_state
    
    try:
        reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(reader)
        total = len(rows)
        
        with _upload_lock:
            _upload_state.total_records = total
            _upload_state.processed_records = 0
            _upload_state.status = "processing_graph"
            
        chunk_size = 5000
        
        for i in range(0, total, chunk_size):
            chunk = rows[i:i + chunk_size]
            
            # Prepare parameters for UNWIND
            tx_params = []
            for row in chunk:
                # Basic validation / defaults
                try:
                    amt = float(row.get("amount", 0))
                except:
                    amt = 0.0
                
                is_fraud = str(row.get("is_fraud", "")).strip().lower() in ("true", "1", "yes", "y")
                
                # Make sure timestamp exists, or default to now
                ts = row.get("timestamp", "").strip()
                if not ts:
                    ts = datetime.utcnow().isoformat()
                    
                tx_params.append({
                    "id": str(uuid.uuid4()),
                    "reference": row.get("reference", f"BLK{uuid.uuid4().hex[:8].upper()}"),
                    "sender_id": row.get("sender_account_id", "").strip(),
                    "receiver_id": row.get("receiver_account_id", "").strip(),
                    "amount": amt,
                    "currency": row.get("currency", "USD").strip() or "USD",
                    "txn_type": row.get("transaction_type", "WIRE").strip() or "WIRE",
                    "channel": row.get("channel", "ONLINE").strip() or "ONLINE",
                    "description": row.get("description", "").strip(),
                    "timestamp": ts,
                    "is_fraud": is_fraud,
                    "fraud_type": row.get("fraud_type", "").strip() or None,
                })
            
            # Ingest chunk via UNWIND
            with neo4j_session() as session:
                session.run("""
                    UNWIND $batch AS row
                    
                    // Match or create sender account if missing (fallback only)
                    MERGE (sender:Account {id: row.sender_id})
                    
                    // Create transaction
                    CREATE (t:Transaction {
                        id: row.id,
                        reference: row.reference,
                        amount: row.amount,
                        currency: row.currency,
                        amount_usd: row.amount,
                        exchange_rate: 1.0,
                        transaction_type: row.txn_type,
                        channel: row.channel,
                        status: 'COMPLETED',
                        timestamp: row.timestamp,
                        description: row.description,
                        is_fraud: row.is_fraud,
                        fraud_type: row.fraud_type
                    })
                    
                    // Link sender
                    CREATE (sender)-[:INITIATED]->(t)
                    
                    // Link receiver if provided
                    WITH row, t
                    WHERE row.receiver_id <> ""
                    MERGE (receiver:Account {id: row.receiver_id})
                    CREATE (t)-[:CREDITED_TO]->(receiver)
                """, batch=tx_params)
                
            with _upload_lock:
                _upload_state.processed_records += len(chunk)

        # Triger incremental model training
        with _upload_lock:
            _upload_state.status = "training_models"
            
        print("[Upload] Graph ingestion complete. Triggering incremental training...")
        train_res = run_incremental(trigger="bulk_upload", force=True, auto_promote=True)
        
        with _upload_lock:
            _upload_state.training_result = train_res
            _upload_state.status = "completed"
            _upload_state.running = False
            
        print(f"[Upload] Finished. Added {total} transactions. Training result: {train_res.get('status')}")

    except Exception as e:
        with _upload_lock:
            _upload_state.error = str(e)
            _upload_state.status = "failed"
            _upload_state.running = False
        print(f"[Upload] Error processing CSV: {e}")


@router.post("/transactions")
async def upload_transactions_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    _admin: UserInDB = Depends(require_admin)
):
    """
    Upload a CSV of transactions. 
    Required columns: sender_account_id, amount
    Optional columns: receiver_account_id, currency, transaction_type, channel, description, timestamp, is_fraud, fraud_type
    Processing happens asynchronously.
    """
    global _upload_state
    
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
    with _upload_lock:
        if _upload_state.running:
            raise HTTPException(status_code=429, detail="An upload job is already running")
            
        _upload_state = UploadJobState()
        _upload_state.running = True
        _upload_state.status = "reading_file"

    content = await file.read()
    csv_text = content.decode("utf-8")
    
    background_tasks.add_task(_process_csv_background, csv_text)
    
    return {
        "status": "accepted",
        "message": "CSV uploaded successfully. Processing in background.",
        "filename": file.filename
    }


@router.get("/status")
def get_upload_status(
    _admin: UserInDB = Depends(require_admin)
):
    """Check the status of the background CSV upload and training job."""
    with _upload_lock:
        return {
            "running": _upload_state.running,
            "status": _upload_state.status,
            "total_records": _upload_state.total_records,
            "processed_records": _upload_state.processed_records,
            "error": _upload_state.error,
            "training_result": _upload_state.training_result
        }
