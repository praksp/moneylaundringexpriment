"""
expand_data.py
==============
Generates 10,000 additional transactions and persists them to Neo4j.

Split:
  24% (2,400 txns) → assigned to EXISTING customers using their current accounts
  76% (7,600 txns) → assigned to NEW customers + new accounts created here

Fraud rate:  ~15% across both groups, using the same 7 fraud typologies:
  Structuring, Smurfing, Layering, Round-tripping,
  Dormant Burst, High-Risk Corridor, Rapid Velocity

After writing to Neo4j, all three ML models are retrained on the full dataset.
"""

import random
import uuid
from datetime import datetime, timedelta

from faker import Faker
from rich.console import Console
from rich.table import Table

from db.client import neo4j_session
from data.generator import (
    AMLDataGenerator, _uid, _txn_ref, _account_num,
    _pick_country, _pick_bank, ALL_COUNTRIES, HIGH_RISK_COUNTRIES,
    LOW_RISK_COUNTRIES,
)

console = Console()
fake    = Faker()

# ── Config ─────────────────────────────────────────────────────────────────────

TARGET_NEW_TXNS     = 10_000
EXISTING_RATIO      = 0.24        # 24 % of new txns go to existing customers
NEW_CUSTOMER_RATIO  = 1 - EXISTING_RATIO  # 76 %
FRAUD_RATE          = 0.15        # ~15 % fraud in new batch (same as original)

EXISTING_TXN_COUNT  = round(TARGET_NEW_TXNS * EXISTING_RATIO)   # 2 400
NEW_TXN_COUNT       = TARGET_NEW_TXNS - EXISTING_TXN_COUNT       # 7 600

FRAUD_COUNT_EXISTING = round(EXISTING_TXN_COUNT * FRAUD_RATE)    # ~360
FRAUD_COUNT_NEW      = round(NEW_TXN_COUNT      * FRAUD_RATE)    # ~1 140

# Seed for reproducibility (different from the original 42 to avoid collisions)
random.seed(2026)
Faker.seed(2026)


# ── Fetch existing graph entities ──────────────────────────────────────────────

def fetch_existing_entities() -> dict:
    """Pull all accounts, devices, IPs, and merchants already in Neo4j."""
    console.print("[cyan]Fetching existing entities from Neo4j…[/]")
    with neo4j_session() as session:
        accounts = [
            dict(r["a"]) for r in session.run(
                "MATCH (a:Account) RETURN a"
            )
        ]
        devices = [
            dict(r["d"]) for r in session.run(
                "MATCH (d:Device) RETURN d"
            )
        ]
        ips = [
            dict(r["i"]) for r in session.run(
                "MATCH (i:IPAddress) RETURN i"
            )
        ]
        merchants = [
            dict(r["m"]) for r in session.run(
                "MATCH (m:Merchant) RETURN m"
            )
        ]
        customers = [
            dict(r["c"]) for r in session.run(
                "MATCH (c:Customer) RETURN c"
            )
        ]

    console.print(
        f"  Loaded: {len(customers)} customers, {len(accounts)} accounts, "
        f"{len(devices)} devices, {len(ips)} IPs, {len(merchants)} merchants"
    )
    return {
        "accounts":  accounts,
        "devices":   devices,
        "ips":       ips,
        "merchants": merchants,
        "customers": customers,
    }


# ── New-entity generators (mirrors AMLDataGenerator helpers) ──────────────────

def _make_new_devices(n: int = 200) -> list[dict]:
    types = ["MOBILE", "DESKTOP", "ATM", "POS"]
    return [
        {
            "id":          _uid(),
            "fingerprint": fake.md5(),
            "device_type": random.choices(types, weights=[50, 30, 10, 10])[0],
            "user_agent":  fake.user_agent(),
        }
        for _ in range(n)
    ]


def _make_new_ips(n: int = 150) -> list[dict]:
    country_choices = [c["code"] for c in ALL_COUNTRIES]
    result = []
    for _ in range(n):
        country = random.choice(country_choices)
        is_vpn  = random.random() < 0.08
        is_tor  = random.random() < 0.03
        result.append({
            "ip":       fake.ipv4_public(),
            "country":  country,
            "city":     fake.city(),
            "is_vpn":   is_vpn,
            "is_tor":   is_tor,
            "is_proxy": is_vpn or random.random() < 0.04,
            "asn":      f"AS{random.randint(1000, 65000)}",
            "isp":      fake.company(),
        })
    return result


def _make_new_merchants(n: int = 80) -> list[dict]:
    from data.generator import MCC_CODES
    result = []
    for _ in range(n):
        mcc, cat = random.choice(MCC_CODES)
        country  = _pick_country("low")
        result.append({
            "id":         _uid(),
            "name":       fake.company(),
            "mcc_code":   mcc,
            "category":   cat,
            "country":    country["code"],
            "risk_level": "HIGH" if mcc in ("6011", "6051", "7995") else "LOW",
        })
    return result


def _make_new_customers(n: int) -> list[dict]:
    kyc_levels = ["BASIC", "ENHANCED", "SIMPLIFIED"]
    risk_tiers = ["LOW", "LOW", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    result = []
    for _ in range(n):
        country    = _pick_country("low")
        pep        = random.random() < 0.04
        sanctions  = random.random() < 0.01
        result.append({
            "id":                  _uid(),
            "name":                fake.name(),
            "date_of_birth":       fake.date_of_birth(minimum_age=18, maximum_age=85).isoformat(),
            "nationality":         country["code"],
            "country_of_residence":country["code"],
            "kyc_level":           random.choice(kyc_levels),
            "pep_flag":            pep,
            "sanctions_flag":      sanctions,
            "risk_tier":           random.choices(risk_tiers)[0],
            "created_at":          fake.date_time_between(start_date="-5y", end_date="-1m").isoformat(),
            "customer_type":       "CORPORATE" if random.random() < 0.2 else "INDIVIDUAL",
        })
    return result


def _make_new_accounts(customers: list[dict]) -> tuple[list[dict], dict]:
    """Returns (accounts_list, customer_id → [account_id])."""
    now    = datetime.utcnow()
    types  = ["CURRENT", "SAVINGS", "BUSINESS", "PREPAID"]
    currencies = ["USD", "USD", "USD", "EUR", "GBP", "EUR", "JPY"]
    accounts:  list[dict]       = []
    cust_accts: dict[str, list] = {}

    for customer in customers:
        n_accts = random.randint(1, 3)
        cust_accts[customer["id"]] = []
        for _ in range(n_accts):
            bank_name, bank_swift = _pick_bank()
            country      = _pick_country("low")
            created      = fake.date_time_between(start_date="-4y", end_date="-7d")
            days_inactive = random.randint(0, 365)
            typical      = round(random.uniform(100, 5000), 2)
            acct = {
                "id":                        _uid(),
                "account_number":            _account_num(),
                "customer_id":               customer["id"],
                "account_type":              random.choice(types),
                "currency":                  random.choice(currencies),
                "balance":                   round(random.uniform(500, 100_000), 2),
                "country":                   country["code"],
                "bank_name":                 bank_name,
                "bank_swift":                bank_swift,
                "status":                    "ACTIVE",
                "created_at":                created.isoformat(),
                "last_active":               (now - timedelta(days=days_inactive)).isoformat(),
                "average_monthly_volume":    round(typical * random.uniform(10, 40), 2),
                "typical_transaction_size":  typical,
            }
            accounts.append(acct)
            cust_accts[customer["id"]].append(acct["id"])

    return accounts, cust_accts


# ── Transaction factory (reuses AMLDataGenerator fraud patterns) ──────────────

class BatchGenerator(AMLDataGenerator):
    """
    Extends AMLDataGenerator to operate on a pre-seeded pool of accounts,
    devices, IPs, and merchants instead of always creating new ones.
    """

    def load_pool(
        self,
        accounts: list[dict],
        devices:  list[dict],
        ips:      list[dict],
        merchants:list[dict],
    ) -> None:
        """Replace internal lists with the provided pool."""
        self.accounts  = accounts
        self.devices   = devices
        self.ips       = ips
        self.merchants = merchants
        # Rebuild index
        self._account_index = {a["id"]: a for a in accounts}

    def generate_transactions(
        self,
        target_total: int,
        fraud_target: int,
        base_time: datetime,
    ) -> tuple[list[dict], list[dict]]:
        """
        Generate `target_total` transactions with ~`fraud_target` fraud cases.
        Returns (all_transactions, new_beneficiaries).
        """
        fraud_txns: list[dict] = []
        self.beneficiaries = []   # reset so we collect only new ones

        # Scale fraud pattern counts to hit the fraud_target
        # Original 1 000 txns had ~117 fraud from the recipe below.
        # Scale factor relative to original.
        scale = max(1, round(fraud_target / 117))

        patterns = [
            (self._make_structuring_pattern,    3 * scale),
            (self._make_smurfing_pattern,       2 * scale),
            (self._make_layering_pattern,       4 * scale),
            (self._make_round_trip_pattern,     5 * scale),
            (self._make_dormant_burst_pattern,  5 * scale),
            (self._make_high_risk_corridor,    15 * scale),
            (self._make_rapid_velocity_pattern, 3 * scale),
        ]

        for fn, count in patterns:
            for _ in range(count):
                fraud_txns.extend(fn(base_time))
            if len(fraud_txns) >= fraud_target:
                break

        # Trim to exact fraud target
        random.shuffle(fraud_txns)
        fraud_txns = fraud_txns[:fraud_target]

        # Fill remainder with normal transactions
        normal_needed = target_total - len(fraud_txns)
        normal_txns   = [self._make_normal_txn(base_time) for _ in range(normal_needed)]

        all_txns = fraud_txns + normal_txns
        random.shuffle(all_txns)
        return all_txns, list(self.beneficiaries)


# ── Neo4j writers (thin wrappers around AMLDataGenerator._write_* ) ──────────

def _write_new_devices(devices: list[dict]) -> None:
    with neo4j_session() as s:
        for d in devices:
            s.run(
                "MERGE (d:Device {id: $id}) SET d += {fingerprint:$fingerprint, "
                "device_type:$device_type, user_agent:$user_agent}",
                **d,
            )


def _write_new_ips(ips: list[dict]) -> None:
    with neo4j_session() as s:
        for ip in ips:
            s.run(
                """MERGE (i:IPAddress {ip: $ip})
                   SET i += {country:$country, city:$city, is_vpn:$is_vpn,
                             is_tor:$is_tor, is_proxy:$is_proxy, asn:$asn, isp:$isp}
                   WITH i
                   MERGE (c:Country {code: $country})
                   MERGE (i)-[:GEOLOCATED_IN]->(c)""",
                **ip,
            )


def _write_new_merchants(merchants: list[dict]) -> None:
    with neo4j_session() as s:
        for m in merchants:
            s.run(
                "MERGE (m:Merchant {id: $id}) SET m += {name:$name, mcc_code:$mcc_code,"
                " category:$category, country:$country, risk_level:$risk_level}",
                **m,
            )


def _write_new_customers(customers: list[dict]) -> None:
    with neo4j_session() as s:
        for c in customers:
            s.run(
                """MERGE (cu:Customer {id: $id})
                   SET cu += {name:$name, date_of_birth:$date_of_birth,
                              nationality:$nationality,
                              country_of_residence:$country_of_residence,
                              kyc_level:$kyc_level, pep_flag:$pep_flag,
                              sanctions_flag:$sanctions_flag,
                              risk_tier:$risk_tier, created_at:$created_at,
                              customer_type:$customer_type}
                   WITH cu
                   MERGE (cty:Country {code: $country_of_residence})
                   MERGE (cu)-[:RESIDENT_OF]->(cty)""",
                **c,
            )


def _write_new_accounts(accounts: list[dict]) -> None:
    with neo4j_session() as s:
        for a in accounts:
            s.run(
                """MERGE (ac:Account {id: $id})
                   SET ac += {account_number:$account_number,
                              customer_id:$customer_id,
                              account_type:$account_type, currency:$currency,
                              balance:$balance, country:$country,
                              bank_name:$bank_name, bank_swift:$bank_swift,
                              status:$status, created_at:$created_at,
                              last_active:$last_active,
                              average_monthly_volume:$average_monthly_volume,
                              typical_transaction_size:$typical_transaction_size}
                   WITH ac
                   MATCH (cu:Customer {id: $customer_id})
                   MERGE (cu)-[:OWNS]->(ac)
                   WITH ac
                   MERGE (cty:Country {code: $country})
                   MERGE (ac)-[:BASED_IN]->(cty)""",
                **a,
            )


def _write_new_beneficiaries(bens: list[dict]) -> None:
    with neo4j_session() as s:
        for b in bens:
            s.run(
                """MERGE (b:BeneficiaryAccount {id: $id})
                   SET b += {account_number:$account_number,
                             account_name:$account_name,
                             bank_name:$bank_name, bank_swift:$bank_swift,
                             country:$country, currency:$currency}""",
                **b,
            )


def _write_transactions_to_neo4j(txns: list[dict], label: str) -> None:
    CHUNK = 100
    total = len(txns)
    for start in range(0, total, CHUNK):
        chunk = txns[start:start + CHUNK]
        with neo4j_session() as s:
            for t in chunk:
                s.run(
                    """MERGE (tx:Transaction {id: $id})
                       SET tx += {reference:$reference, amount:$amount,
                                  currency:$currency, exchange_rate:$exchange_rate,
                                  amount_usd:$amount_usd,
                                  transaction_type:$transaction_type,
                                  channel:$channel, status:$status,
                                  timestamp:$timestamp, description:$description,
                                  is_fraud:$is_fraud, fraud_type:$fraud_type}
                       WITH tx
                       MATCH (sender:Account {id: $sender_account_id})
                       MERGE (sender)-[:INITIATED]->(tx)""",
                    **{k: v for k, v in t.items()},
                )
                if t.get("receiver_account_id"):
                    s.run(
                        "MATCH (tx:Transaction {id:$tid}) "
                        "MATCH (r:Account {id:$rid}) MERGE (tx)-[:CREDITED_TO]->(r)",
                        tid=t["id"], rid=t["receiver_account_id"],
                    )
                if t.get("merchant_id"):
                    s.run(
                        "MATCH (tx:Transaction {id:$tid}) "
                        "MATCH (m:Merchant {id:$mid}) MERGE (tx)-[:PAID_TO]->(m)",
                        tid=t["id"], mid=t["merchant_id"],
                    )
                if t.get("device_id"):
                    s.run(
                        "MATCH (tx:Transaction {id:$tid}) "
                        "MATCH (d:Device {id:$did}) MERGE (tx)-[:ORIGINATED_FROM]->(d)",
                        tid=t["id"], did=t["device_id"],
                    )
                if t.get("ip_id"):
                    s.run(
                        "MATCH (tx:Transaction {id:$tid}) "
                        "MATCH (i:IPAddress {ip:$ip}) MERGE (tx)-[:SOURCED_FROM]->(i)",
                        tid=t["id"], ip=t["ip_id"],
                    )
                if t.get("beneficiary_id"):
                    s.run(
                        "MATCH (tx:Transaction {id:$tid}) "
                        "MATCH (b:BeneficiaryAccount {id:$bid}) MERGE (tx)-[:SENT_TO_EXTERNAL]->(b)",
                        tid=t["id"], bid=t["beneficiary_id"],
                    )

        written = min(start + CHUNK, total)
        if written % 1000 == 0 or written == total:
            console.print(f"  [{label}] {written:,}/{total:,} transactions written")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold cyan]AML Data Expansion — 10,000 New Transactions[/]")
    console.print(
        f"  Target: [bold]{TARGET_NEW_TXNS:,}[/] new transactions\n"
        f"  Existing customers (24%): [bold]{EXISTING_TXN_COUNT:,}[/] txns "
        f"(~{FRAUD_COUNT_EXISTING} fraud)\n"
        f"  New customers (76%):      [bold]{NEW_TXN_COUNT:,}[/] txns "
        f"(~{FRAUD_COUNT_NEW} fraud)\n"
    )

    # ── 1. Load existing entities ─────────────────────────────────────────────
    existing = fetch_existing_entities()

    # ── 2. Create new auxiliary entities ─────────────────────────────────────
    console.print("\n[cyan]Creating new auxiliary entities…[/]")
    new_devices   = _make_new_devices(200)
    new_ips       = _make_new_ips(150)
    new_merchants = _make_new_merchants(80)

    console.print(f"  New devices: {len(new_devices)}  |  IPs: {len(new_ips)}  |  Merchants: {len(new_merchants)}")

    _write_new_devices(new_devices)
    _write_new_ips(new_ips)
    _write_new_merchants(new_merchants)
    console.print("  [green]✓ Auxiliary entities written[/]")

    # ── 3. Transactions for EXISTING customers ────────────────────────────────
    console.print("\n[cyan]Generating transactions for EXISTING customers…[/]")
    all_devices   = existing["devices"]   + new_devices
    all_ips       = existing["ips"]       + new_ips
    all_merchants = existing["merchants"] + new_merchants

    gen_existing = BatchGenerator()
    gen_existing.load_pool(
        accounts  = existing["accounts"],
        devices   = all_devices,
        ips       = all_ips,
        merchants = all_merchants,
    )

    base_time = datetime.utcnow()
    txns_existing, bens_existing = gen_existing.generate_transactions(
        target_total = EXISTING_TXN_COUNT,
        fraud_target = FRAUD_COUNT_EXISTING,
        base_time    = base_time,
    )
    fraud_e = sum(1 for t in txns_existing if t["is_fraud"])
    console.print(
        f"  Generated [bold]{len(txns_existing):,}[/] txns "
        f"([red]{fraud_e} fraud[/] / [green]{len(txns_existing)-fraud_e} normal[/])"
    )

    if bens_existing:
        _write_new_beneficiaries(bens_existing)

    console.print(f"  Writing {len(txns_existing):,} transactions for existing customers…")
    _write_transactions_to_neo4j(txns_existing, "existing-customers")
    console.print("  [green]✓ Existing-customer transactions written[/]")

    # ── 4. New customers + their transactions ─────────────────────────────────
    console.print("\n[cyan]Creating new customers and accounts…[/]")

    # Estimate how many new customers we need for ~7 600 txns
    # Original: 120 customers → ~256 accounts → 1 000 txns  (≈ 4 txns/account)
    # Aim for ≈ 4 txns/account, 2 accounts/customer  → ~950 new customers
    new_customers_needed = max(100, round(NEW_TXN_COUNT / (4 * 2)))
    new_customers  = _make_new_customers(new_customers_needed)
    new_accounts, _cust_accts = _make_new_accounts(new_customers)

    console.print(
        f"  New customers: [bold]{len(new_customers):,}[/]  |  "
        f"New accounts: [bold]{len(new_accounts):,}[/]"
    )

    _write_new_customers(new_customers)
    _write_new_accounts(new_accounts)
    console.print("  [green]✓ New customers and accounts written[/]")

    gen_new = BatchGenerator()
    gen_new.load_pool(
        accounts  = new_accounts,
        devices   = all_devices,
        ips       = all_ips,
        merchants = all_merchants,
    )

    txns_new, bens_new = gen_new.generate_transactions(
        target_total = NEW_TXN_COUNT,
        fraud_target = FRAUD_COUNT_NEW,
        base_time    = base_time,
    )
    fraud_n = sum(1 for t in txns_new if t["is_fraud"])
    console.print(
        f"  Generated [bold]{len(txns_new):,}[/] txns "
        f"([red]{fraud_n} fraud[/] / [green]{len(txns_new)-fraud_n} normal[/])"
    )

    if bens_new:
        _write_new_beneficiaries(bens_new)

    console.print(f"  Writing {len(txns_new):,} transactions for new customers…")
    _write_transactions_to_neo4j(txns_new, "new-customers")
    console.print("  [green]✓ New-customer transactions written[/]")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    total_new  = len(txns_existing) + len(txns_new)
    total_fraud = fraud_e + fraud_n

    console.rule("[bold green]Expansion Complete[/]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric",  style="cyan")
    table.add_column("Value",   style="green")
    table.add_row("New transactions",     f"{total_new:,}")
    table.add_row("  ↳ existing customers (24%)", f"{len(txns_existing):,}")
    table.add_row("  ↳ new customers (76%)",       f"{len(txns_new):,}")
    table.add_row("New fraud transactions", f"{total_fraud:,}  ({total_fraud/total_new*100:.1f}%)")
    table.add_row("New customers added",   f"{len(new_customers):,}")
    table.add_row("New accounts added",    f"{len(new_accounts):,}")
    console.print(table)

    # ── 6. Re-verify database totals ─────────────────────────────────────────
    with neo4j_session() as s:
        totals = {
            "Customers":    s.run("MATCH (c:Customer) RETURN count(c) AS n").single()["n"],
            "Accounts":     s.run("MATCH (a:Account)  RETURN count(a) AS n").single()["n"],
            "Transactions": s.run("MATCH (t:Transaction) RETURN count(t) AS n").single()["n"],
            "Fraud txns":   s.run("MATCH (t:Transaction {is_fraud:true}) RETURN count(t) AS n").single()["n"],
        }

    console.print("\n[bold]Database totals after expansion:[/]")
    for k, v in totals.items():
        console.print(f"  {k}: [bold white]{v:,}[/]")

    # ── 7. Retrain all models ─────────────────────────────────────────────────
    console.rule("[bold cyan]Retraining ML Models on Expanded Dataset[/]")
    from ml.train import train_and_save_all
    metrics = train_and_save_all()

    console.rule("[bold green]All Done[/]")
    console.print(
        f"XGBoost ROC-AUC: [bold]{metrics['xgb']['roc_auc']}[/]  |  "
        f"SGD/SVM ROC-AUC: [bold]{metrics['svm']['roc_auc']}[/]  |  "
        f"KNN ROC-AUC: [bold]{metrics['knn']['roc_auc']}[/]"
    )


if __name__ == "__main__":
    main()
