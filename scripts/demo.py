"""
Interactive demo – evaluate random transactions and show risk scores.

Usage:
  python scripts/demo.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def score_colour(score: int) -> str:
    if score <= 399:
        return "green"
    elif score <= 699:
        return "yellow"
    return "red"


def outcome_icon(outcome: str) -> str:
    return {"ALLOW": "✅", "CHALLENGE": "⚠️ ", "DECLINE": "🚫"}.get(outcome, "?")


def main():
    console.print(Panel("[bold cyan]AML Risk Engine – Live Demo[/]",
                        subtitle="Evaluating sample transactions"))

    from db.client import neo4j_session
    from risk.engine import evaluate_transaction_by_id

    # Get a mix of fraud and legit transactions
    with neo4j_session() as session:
        fraud_txns = [
            r["id"] for r in session.run(
                "MATCH ()-[:INITIATED]->(t:Transaction {is_fraud: true}) "
                "RETURN t.id AS id LIMIT 5"
            )
        ]
        legit_txns = [
            r["id"] for r in session.run(
                "MATCH ()-[:INITIATED]->(t:Transaction {is_fraud: false}) "
                "RETURN t.id AS id LIMIT 5"
            )
        ]

    sample = fraud_txns + legit_txns

    table = Table(title="Transaction Risk Evaluation Results",
                  box=box.ROUNDED, show_lines=True)
    table.add_column("Transaction ID", style="cyan", max_width=36)
    table.add_column("Score", justify="center")
    table.add_column("Bayesian", justify="center")
    table.add_column("ML Score", justify="center")
    table.add_column("Outcome", justify="center")
    table.add_column("Confidence", justify="center")
    table.add_column("Top Risk Factors", max_width=50)

    for txn_id in sample:
        try:
            result = evaluate_transaction_by_id(txn_id)
            rs = result["risk_score"]
            sc = score_colour(rs.score)
            outcome_str = f"{outcome_icon(rs.outcome.value)} {rs.outcome.value}"
            factors_str = "\n".join(f"• {f}" for f in rs.risk_factors[:3]) or "—"
            table.add_row(
                txn_id[:8] + "...",
                f"[{sc}]{rs.score}[/]",
                str(rs.bayesian_score),
                str(rs.ml_score),
                outcome_str,
                f"{rs.confidence:.2f}",
                factors_str,
            )
        except Exception as e:
            table.add_row(txn_id[:8] + "...", "ERR", "—", "—", "—", "—", str(e))

    console.print(table)

    # Show a challenge question example
    for txn_id in fraud_txns:
        result = evaluate_transaction_by_id(txn_id)
        if result.get("challenge_question"):
            cq = result["challenge_question"]
            console.print(Panel(
                f"[bold yellow]Challenge Question for transaction {txn_id[:8]}...[/]\n\n"
                f"[white]{cq.question}[/]",
                title="CHALLENGE",
                border_style="yellow",
            ))
            break


if __name__ == "__main__":
    main()
