#!/usr/bin/env python
"""
Generate a sample CSV file for bulk transaction upload.
Pulls existing accounts from Neo4j to ensure valid references and 
creates a mix of legitimate and fraudulent transaction patterns.
"""
import sys
import os
import csv
import uuid
import random
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from db.client import neo4j_session

def generate_csv(filename: str = "sample_transactions.csv", num_records: int = 100):
    print("Fetching existing accounts from Neo4j...")
    with neo4j_session() as session:
        records = session.run("MATCH (a:Account) RETURN a.id AS id LIMIT 500").data()
        account_ids = [r["id"] for r in records]

    if len(account_ids) < 10:
        print("Not enough accounts in the database to generate a sample. Run setup first.")
        sys.exit(1)

    print(f"Generating {num_records} sample transactions into {filename}...")
    
    headers = [
        "reference", "sender_account_id", "receiver_account_id", "amount", 
        "currency", "transaction_type", "channel", "timestamp", "description", 
        "is_fraud", "fraud_type"
    ]
    
    now = datetime.utcnow()
    
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictReader(f, fieldnames=headers)
        # Using csv.writer to write headers
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i in range(num_records):
            sender = random.choice(account_ids)
            receiver = random.choice(account_ids)
            while receiver == sender:
                receiver = random.choice(account_ids)
            
            # Mix in some fraud
            is_fraud = random.random() < 0.15
            fraud_type = ""
            
            if is_fraud:
                fraud_type = random.choice([
                    "STRUCTURING", "SMURFING", "LAYERING", 
                    "ROUND_TRIP", "DORMANT_BURST", "HIGH_RISK_CORRIDOR", "RAPID_VELOCITY"
                ])
                if fraud_type in ("STRUCTURING", "SMURFING"):
                    amount = round(random.uniform(9000, 9999), 2)
                elif fraud_type in ("LAYERING", "HIGH_RISK_CORRIDOR"):
                    amount = round(random.uniform(15000, 50000), 2)
                else:
                    amount = round(random.uniform(500, 5000), 2)
            else:
                amount = round(random.uniform(10, 2000), 2)
            
            # Create timestamp shifted slightly to simulate recent activity
            ts = now - timedelta(hours=random.uniform(0, 48))
            
            writer.writerow([
                f"SAMP{uuid.uuid4().hex[:8].upper()}",
                sender,
                receiver,
                amount,
                "USD",
                random.choice(["WIRE", "ACH", "INTERNAL", "CARD"]),
                random.choice(["ONLINE", "MOBILE", "BRANCH"]),
                ts.isoformat(),
                f"Sample transaction {i+1}",
                str(is_fraud).lower(),
                fraud_type
            ])
            
    print(f"Done. File '{filename}' created.")
    print("This file contains the raw features needed for ingestion. The backend FeatureStore ")
    print("will automatically enrich these with 44 graph features during processing.")

if __name__ == "__main__":
    generate_csv()
