"""
One-shot setup script
======================
Run this once after starting Neo4j:

  python scripts/setup.py

Steps:
  1. Apply Neo4j schema (constraints + indexes)
  2. Generate and persist 1000 sample transactions
  3. Train the ML model on the generated data
  4. Save the model to disk
"""
import sys
import os

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.rule import Rule
import time

console = Console()


def main():
    console.print(Rule("[bold cyan]AML System Setup[/]"))

    # ── Step 1: Verify Neo4j connection ──────────────────────────
    console.print("\n[bold]Step 1: Connecting to Neo4j...[/]")
    try:
        from db.client import get_driver
        driver = get_driver()
        driver.verify_connectivity()
        console.print("[green]✓ Neo4j connected[/]")
    except Exception as e:
        console.print(f"[red]✗ Cannot connect to Neo4j: {e}[/]")
        console.print("[yellow]Make sure Neo4j is running: docker-compose up -d[/]")
        sys.exit(1)

    # ── Step 2: Generate and persist data ────────────────────────
    console.print("\n[bold]Step 2: Generating 1000 sample transactions...[/]")
    from data.generator import AMLDataGenerator
    gen = AMLDataGenerator()
    gen.generate(target_total=1000)
    gen.persist()

    # ── Step 3: Train ML model ────────────────────────────────────
    console.print("\n[bold]Step 3: Training ML model...[/]")
    from ml.train import train_and_save
    model = train_and_save()

    # ── Step 4: Quick sanity check ────────────────────────────────
    console.print("\n[bold]Step 4: Running sanity check on a sample transaction...[/]")
    from db.client import neo4j_session
    with neo4j_session() as session:
        result = session.run(
            "MATCH ()-[:INITIATED]->(t:Transaction) WHERE t.is_fraud = true "
            "RETURN t.id AS id LIMIT 1"
        )
        rec = result.single()

    if rec:
        txn_id = rec["id"]
        console.print(f"[cyan]Evaluating fraud transaction: {txn_id}[/]")
        from risk.engine import evaluate_transaction_by_id
        eval_result = evaluate_transaction_by_id(txn_id)
        rs = eval_result["risk_score"]
        console.print(f"  Score: [bold red]{rs.score}[/] | "
                      f"Outcome: [bold]{rs.outcome.value}[/] | "
                      f"Factors: {', '.join(rs.risk_factors[:3])}")

    console.print(Rule("[bold green]✓ Setup Complete[/]"))
    console.print(
        "\nStart the API server:\n"
        "  [cyan]uvicorn api.main:app --reload --port 8000[/]\n"
        "\nOpen docs:\n"
        "  http://localhost:8000/docs\n"
        "\nOpen Neo4j browser:\n"
        "  http://localhost:7474  (user: neo4j / pass: amlpassword123)\n"
    )


if __name__ == "__main__":
    main()
