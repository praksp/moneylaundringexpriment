"""
AML Transaction Data Generator
================================
Generates 1000 realistic financial transactions seeded with known
money-laundering patterns for training and testing the risk engine.

Fraud pattern distribution (~15% fraud):
  1. Structuring        : Multiple txns just below $10k reporting threshold
  2. Smurfing           : Many sources → one aggregator account
  3. Layering           : Chain A→B→C→D→E rapid fund movement
  4. Round-tripping     : Money leaves and returns to origin within 48h
  5. Dormant-then-burst : Long-dormant account suddenly sends large amount
  6. High-risk corridor : Transfers to FATF grey/black-listed countries
  7. Rapid velocity     : 10+ transactions from same account within 1 hour
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import Any
from faker import Faker
from db.client import neo4j_session
from db.schema import apply_schema
from rich.console import Console

console = Console()
fake = Faker()
random.seed(42)

# ──────────────────────────────────────────────────────────────────
# Reference data
# ──────────────────────────────────────────────────────────────────

HIGH_RISK_COUNTRIES = [
    {"code": "IR", "name": "Iran", "fatf_risk": "BLACKLIST", "is_sanctioned": True, "is_tax_haven": False},
    {"code": "KP", "name": "North Korea", "fatf_risk": "BLACKLIST", "is_sanctioned": True, "is_tax_haven": False},
    {"code": "SY", "name": "Syria", "fatf_risk": "BLACKLIST", "is_sanctioned": True, "is_tax_haven": False},
    {"code": "MM", "name": "Myanmar", "fatf_risk": "HIGH", "is_sanctioned": False, "is_tax_haven": False},
    {"code": "RU", "name": "Russia", "fatf_risk": "HIGH", "is_sanctioned": False, "is_tax_haven": False},
    {"code": "VE", "name": "Venezuela", "fatf_risk": "HIGH", "is_sanctioned": False, "is_tax_haven": False},
    {"code": "AF", "name": "Afghanistan", "fatf_risk": "HIGH", "is_sanctioned": False, "is_tax_haven": False},
    {"code": "PK", "name": "Pakistan", "fatf_risk": "HIGH", "is_sanctioned": False, "is_tax_haven": False},
]

LOW_RISK_COUNTRIES = [
    {"code": "US", "name": "United States", "fatf_risk": "LOW", "is_sanctioned": False, "is_tax_haven": False},
    {"code": "GB", "name": "United Kingdom", "fatf_risk": "LOW", "is_sanctioned": False, "is_tax_haven": False},
    {"code": "DE", "name": "Germany", "fatf_risk": "LOW", "is_sanctioned": False, "is_tax_haven": False},
    {"code": "FR", "name": "France", "fatf_risk": "LOW", "is_sanctioned": False, "is_tax_haven": False},
    {"code": "CA", "name": "Canada", "fatf_risk": "LOW", "is_sanctioned": False, "is_tax_haven": False},
    {"code": "AU", "name": "Australia", "fatf_risk": "LOW", "is_sanctioned": False, "is_tax_haven": False},
    {"code": "SG", "name": "Singapore", "fatf_risk": "LOW", "is_sanctioned": False, "is_tax_haven": False},
    {"code": "JP", "name": "Japan", "fatf_risk": "LOW", "is_sanctioned": False, "is_tax_haven": False},
]

TAX_HAVEN_COUNTRIES = [
    {"code": "KY", "name": "Cayman Islands", "fatf_risk": "MEDIUM", "is_sanctioned": False, "is_tax_haven": True},
    {"code": "VG", "name": "British Virgin Islands", "fatf_risk": "MEDIUM", "is_sanctioned": False, "is_tax_haven": True},
    {"code": "PA", "name": "Panama", "fatf_risk": "MEDIUM", "is_sanctioned": False, "is_tax_haven": True},
    {"code": "CH", "name": "Switzerland", "fatf_risk": "LOW", "is_sanctioned": False, "is_tax_haven": True},
]

ALL_COUNTRIES = LOW_RISK_COUNTRIES + TAX_HAVEN_COUNTRIES + HIGH_RISK_COUNTRIES

BANKS = [
    ("JPMorgan Chase", "CHASUS33"), ("Bank of America", "BOFAUS3N"),
    ("HSBC", "MRMDGB2L"), ("Deutsche Bank", "DEUTDEDB"),
    ("Barclays", "BARCGB22"), ("Standard Chartered", "SCBLSGSG"),
    ("DBS Bank", "DBSSSGSG"), ("Commonwealth Bank", "CTBAAU2S"),
    ("RBC Royal Bank", "ROYCCAT2"), ("BNP Paribas", "BNPAFRPP"),
]

MCC_CODES = [
    ("5411", "Grocery Stores"), ("5812", "Eating Places"), ("4111", "Transportation"),
    ("7011", "Hotels"), ("5999", "Miscellaneous Retail"), ("6011", "Cash Advance ATM"),
    ("7372", "Computer Programming"), ("5122", "Drugs / Pharmacies"),
    ("6051", "Foreign Exchange"), ("7995", "Gambling"),
]

CHALLENGE_QUESTIONS = [
    "What is the last 4 digits of your registered mobile number?",
    "Please confirm this transaction by entering your one-time passcode sent to your registered email.",
    "What was the destination country of your last international transaction?",
    "Please confirm your identity: what is your mother's maiden name?",
    "Enter the OTP sent to your registered device to authorise this transaction.",
    "What is the primary purpose of this transaction? (e.g. Business payment, Personal transfer, Invoice settlement)",
    "Please confirm: do you personally authorise this transfer of ${amount} to {beneficiary}?",
]


# ──────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────

def _uid() -> str:
    return str(uuid.uuid4())


def _txn_ref() -> str:
    return f"TXN{uuid.uuid4().hex[:8].upper()}"


def _account_num() -> str:
    return f"{random.randint(10000000, 99999999)}"


def _pick_country(risk: str = "any") -> dict:
    if risk == "high":
        return random.choice(HIGH_RISK_COUNTRIES)
    elif risk == "low":
        return random.choice(LOW_RISK_COUNTRIES)
    elif risk == "tax_haven":
        return random.choice(TAX_HAVEN_COUNTRIES)
    else:
        weights = [len(LOW_RISK_COUNTRIES)] * len(LOW_RISK_COUNTRIES) + \
                  [2] * len(TAX_HAVEN_COUNTRIES) + \
                  [1] * len(HIGH_RISK_COUNTRIES)
        return random.choices(ALL_COUNTRIES, weights=weights[:len(ALL_COUNTRIES)])[0]


def _pick_bank() -> tuple:
    return random.choice(BANKS)


# ──────────────────────────────────────────────────────────────────
# Main generator class
# ──────────────────────────────────────────────────────────────────

class AMLDataGenerator:
    def __init__(self):
        self.customers: list[dict] = []
        self.accounts: list[dict] = []
        self.transactions: list[dict] = []
        self.devices: list[dict] = []
        self.ips: list[dict] = []
        self.merchants: list[dict] = []
        self.beneficiaries: list[dict] = []
        self.countries: list[dict] = []
        self._account_index: dict[str, dict] = {}   # account_id → account dict
        self._customer_accounts: dict[str, list] = {}  # customer_id → [account_ids]

    # ── Seeding entities ─────────────────────────────────────────

    def _seed_countries(self) -> None:
        self.countries = ALL_COUNTRIES

    def _seed_devices(self, n: int = 80) -> None:
        types = ["MOBILE", "DESKTOP", "ATM", "POS"]
        for _ in range(n):
            self.devices.append({
                "id": _uid(),
                "fingerprint": fake.md5(),
                "device_type": random.choices(types, weights=[50, 30, 10, 10])[0],
                "user_agent": fake.user_agent(),
            })

    def _seed_ips(self, n: int = 60) -> None:
        country_choices = [c["code"] for c in ALL_COUNTRIES]
        for _ in range(n):
            country = random.choice(country_choices)
            is_vpn = random.random() < 0.08
            is_tor = random.random() < 0.03
            self.ips.append({
                "ip": fake.ipv4_public(),
                "country": country,
                "city": fake.city(),
                "is_vpn": is_vpn,
                "is_tor": is_tor,
                "is_proxy": is_vpn or random.random() < 0.04,
                "asn": f"AS{random.randint(1000, 65000)}",
                "isp": fake.company(),
            })

    def _seed_merchants(self, n: int = 50) -> None:
        for _ in range(n):
            mcc, cat = random.choice(MCC_CODES)
            country = _pick_country("low")
            risk = "HIGH" if mcc in ("6011", "6051", "7995") else "LOW"
            self.merchants.append({
                "id": _uid(),
                "name": fake.company(),
                "mcc_code": mcc,
                "category": cat,
                "country": country["code"],
                "risk_level": risk,
            })

    def _seed_customers(self, n: int = 120) -> None:
        kyc_levels = ["BASIC", "ENHANCED", "SIMPLIFIED"]
        risk_tiers = ["LOW", "LOW", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        for i in range(n):
            country = _pick_country("low")
            pep = random.random() < 0.04
            sanctions = random.random() < 0.01
            self.customers.append({
                "id": _uid(),
                "name": fake.name(),
                "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=85).isoformat(),
                "nationality": country["code"],
                "country_of_residence": country["code"],
                "kyc_level": random.choice(kyc_levels),
                "pep_flag": pep,
                "sanctions_flag": sanctions,
                "risk_tier": random.choices(risk_tiers)[0],
                "created_at": fake.date_time_between(start_date="-5y", end_date="-1m").isoformat(),
                "customer_type": "CORPORATE" if random.random() < 0.2 else "INDIVIDUAL",
            })

    def _seed_accounts(self, accounts_per_customer: tuple = (1, 3)) -> None:
        now = datetime.utcnow()
        types = ["CURRENT", "SAVINGS", "BUSINESS", "PREPAID"]
        currencies = ["USD", "USD", "USD", "EUR", "GBP", "EUR", "JPY"]
        for customer in self.customers:
            n_accounts = random.randint(*accounts_per_customer)
            self._customer_accounts[customer["id"]] = []
            for _ in range(n_accounts):
                bank_name, bank_swift = _pick_bank()
                country = _pick_country("low")
                created = fake.date_time_between(start_date="-4y", end_date="-7d")
                days_since_active = random.randint(0, 365)
                last_active = now - timedelta(days=days_since_active)
                typical = round(random.uniform(100, 5000), 2)
                acct = {
                    "id": _uid(),
                    "account_number": _account_num(),
                    "customer_id": customer["id"],
                    "account_type": random.choice(types),
                    "currency": random.choice(currencies),
                    "balance": round(random.uniform(500, 100_000), 2),
                    "country": country["code"],
                    "bank_name": bank_name,
                    "bank_swift": bank_swift,
                    "status": "ACTIVE",
                    "created_at": created.isoformat(),
                    "last_active": last_active.isoformat(),
                    "average_monthly_volume": round(typical * random.uniform(10, 40), 2),
                    "typical_transaction_size": typical,
                }
                self.accounts.append(acct)
                self._account_index[acct["id"]] = acct
                self._customer_accounts[customer["id"]].append(acct["id"])

    # ── Normal transaction generator ─────────────────────────────

    def _make_normal_txn(self, base_time: datetime) -> dict:
        sender = random.choice(self.accounts)
        receiver = random.choice([a for a in self.accounts if a["id"] != sender["id"]])
        typical = sender.get("typical_transaction_size", 500)
        amount = round(random.uniform(typical * 0.5, typical * 2.0), 2)
        txn_type = random.choices(
            ["WIRE", "ACH", "CARD", "INTERNAL"],
            weights=[10, 40, 35, 15]
        )[0]
        channel = random.choices(
            ["ONLINE", "MOBILE", "BRANCH", "ATM"],
            weights=[30, 40, 20, 10]
        )[0]
        device = random.choice(self.devices) if channel in ("ONLINE", "MOBILE") else None
        ip = random.choice(self.ips) if channel in ("ONLINE", "MOBILE") else None
        merchant = random.choice(self.merchants) if txn_type == "CARD" else None
        offset_hours = random.uniform(0, 24 * 30)
        timestamp = base_time - timedelta(hours=offset_hours)
        return {
            "id": _uid(),
            "reference": _txn_ref(),
            "sender_account_id": sender["id"],
            "receiver_account_id": receiver["id"],
            "beneficiary_id": None,
            "merchant_id": merchant["id"] if merchant else None,
            "amount": amount,
            "currency": sender.get("currency", "USD"),
            "exchange_rate": 1.0,
            "amount_usd": amount,
            "transaction_type": txn_type,
            "channel": channel,
            "status": "COMPLETED",
            "timestamp": timestamp.isoformat(),
            "description": fake.sentence(nb_words=4),
            "device_id": device["id"] if device else None,
            "ip_id": ip["ip"] if ip else None,
            "is_fraud": False,
            "fraud_type": None,
        }

    # ── Fraud pattern generators ─────────────────────────────────

    def _make_structuring_pattern(self, base_time: datetime) -> list[dict]:
        """Multiple transactions just below $10,000 CTR threshold."""
        sender = random.choice(self.accounts)
        receivers = random.sample([a for a in self.accounts if a["id"] != sender["id"]], min(5, len(self.accounts) - 1))
        txns = []
        for i, receiver in enumerate(receivers):
            amount = round(random.uniform(9000, 9999), 2)
            offset = timedelta(hours=random.uniform(0, 6) + i * 0.5)
            device = random.choice(self.devices)
            ip = random.choice(self.ips)
            txns.append({
                "id": _uid(), "reference": _txn_ref(),
                "sender_account_id": sender["id"],
                "receiver_account_id": receiver["id"],
                "beneficiary_id": None, "merchant_id": None,
                "amount": amount, "currency": "USD",
                "exchange_rate": 1.0, "amount_usd": amount,
                "transaction_type": "ACH", "channel": "ONLINE",
                "status": "COMPLETED",
                "timestamp": (base_time - offset).isoformat(),
                "description": "Business payment",
                "device_id": device["id"], "ip_id": ip["ip"],
                "is_fraud": True, "fraud_type": "STRUCTURING",
            })
        return txns

    def _make_smurfing_pattern(self, base_time: datetime) -> list[dict]:
        """Multiple senders aggregating to one account (aggregator)."""
        aggregator = random.choice(self.accounts)
        senders = random.sample([a for a in self.accounts if a["id"] != aggregator["id"]], min(8, len(self.accounts) - 1))
        txns = []
        for i, sender in enumerate(senders):
            amount = round(random.uniform(1000, 4999), 2)
            offset = timedelta(hours=random.uniform(0, 48) + i * 0.25)
            device = random.choice(self.devices)
            ip = random.choice(self.ips)
            txns.append({
                "id": _uid(), "reference": _txn_ref(),
                "sender_account_id": sender["id"],
                "receiver_account_id": aggregator["id"],
                "beneficiary_id": None, "merchant_id": None,
                "amount": amount, "currency": "USD",
                "exchange_rate": 1.0, "amount_usd": amount,
                "transaction_type": "ACH", "channel": "MOBILE",
                "status": "COMPLETED",
                "timestamp": (base_time - offset).isoformat(),
                "description": "Transfer",
                "device_id": device["id"], "ip_id": ip["ip"],
                "is_fraud": True, "fraud_type": "SMURFING",
            })
        return txns

    def _make_layering_pattern(self, base_time: datetime) -> list[dict]:
        """Chain of rapid transfers A→B→C→D→E to obscure fund origin."""
        chain = random.sample(self.accounts, min(5, len(self.accounts)))
        txns = []
        amount = round(random.uniform(20_000, 200_000), 2)
        for i in range(len(chain) - 1):
            amount = round(amount * random.uniform(0.90, 0.99), 2)  # small fee each hop
            offset = timedelta(hours=i * random.uniform(0.5, 3))
            device = random.choice(self.devices)
            ip = random.choice(self.ips)
            txns.append({
                "id": _uid(), "reference": _txn_ref(),
                "sender_account_id": chain[i]["id"],
                "receiver_account_id": chain[i + 1]["id"],
                "beneficiary_id": None, "merchant_id": None,
                "amount": amount, "currency": "USD",
                "exchange_rate": 1.0, "amount_usd": amount,
                "transaction_type": "WIRE", "channel": "ONLINE",
                "status": "COMPLETED",
                "timestamp": (base_time - offset).isoformat(),
                "description": "Invoice settlement",
                "device_id": device["id"], "ip_id": ip["ip"],
                "is_fraud": True, "fraud_type": "LAYERING",
            })
        return txns

    def _make_round_trip_pattern(self, base_time: datetime) -> list[dict]:
        """Money leaves origin, passes through intermediary, returns to origin."""
        origin = random.choice(self.accounts)
        intermediary = random.choice([a for a in self.accounts if a["id"] != origin["id"]])
        amount = round(random.uniform(10_000, 80_000), 2)
        device = random.choice(self.devices)
        ip = random.choice(self.ips)
        txns = [
            {
                "id": _uid(), "reference": _txn_ref(),
                "sender_account_id": origin["id"],
                "receiver_account_id": intermediary["id"],
                "beneficiary_id": None, "merchant_id": None,
                "amount": amount, "currency": "USD",
                "exchange_rate": 1.0, "amount_usd": amount,
                "transaction_type": "WIRE", "channel": "ONLINE",
                "status": "COMPLETED",
                "timestamp": (base_time - timedelta(hours=30)).isoformat(),
                "description": "Consulting fee",
                "device_id": device["id"], "ip_id": ip["ip"],
                "is_fraud": True, "fraud_type": "ROUND_TRIP",
            },
            {
                "id": _uid(), "reference": _txn_ref(),
                "sender_account_id": intermediary["id"],
                "receiver_account_id": origin["id"],
                "beneficiary_id": None, "merchant_id": None,
                "amount": round(amount * 0.95, 2),
                "currency": "USD", "exchange_rate": 1.0,
                "amount_usd": round(amount * 0.95, 2),
                "transaction_type": "WIRE", "channel": "ONLINE",
                "status": "COMPLETED",
                "timestamp": (base_time - timedelta(hours=6)).isoformat(),
                "description": "Refund - overpayment",
                "device_id": device["id"], "ip_id": ip["ip"],
                "is_fraud": True, "fraud_type": "ROUND_TRIP",
            },
        ]
        return txns

    def _make_dormant_burst_pattern(self, base_time: datetime) -> list[dict]:
        """Dormant account suddenly active with large outbound."""
        account = random.choice(self.accounts)
        # Mark account as dormant (last active > 90 days ago)
        account["last_active"] = (base_time - timedelta(days=random.randint(120, 365))).isoformat()
        receivers = random.sample([a for a in self.accounts if a["id"] != account["id"]], min(3, len(self.accounts) - 1))
        amount_each = round(random.uniform(15_000, 50_000), 2)
        txns = []
        device = random.choice(self.devices)
        ip = random.choice(self.ips)
        for i, receiver in enumerate(receivers):
            txns.append({
                "id": _uid(), "reference": _txn_ref(),
                "sender_account_id": account["id"],
                "receiver_account_id": receiver["id"],
                "beneficiary_id": None, "merchant_id": None,
                "amount": amount_each, "currency": "USD",
                "exchange_rate": 1.0, "amount_usd": amount_each,
                "transaction_type": "WIRE", "channel": "ONLINE",
                "status": "COMPLETED",
                "timestamp": (base_time - timedelta(hours=i * 2)).isoformat(),
                "description": "Urgent payment",
                "device_id": device["id"], "ip_id": ip["ip"],
                "is_fraud": True, "fraud_type": "DORMANT_BURST",
            })
        return txns

    def _make_high_risk_corridor(self, base_time: datetime) -> list[dict]:
        """Transfers to/from high-risk or sanctioned jurisdictions."""
        sender = random.choice(self.accounts)
        high_risk_country = random.choice(HIGH_RISK_COUNTRIES)
        bank_name, bank_swift = _pick_bank()
        ben_id = _uid()
        ben = {
            "id": ben_id,
            "account_number": _account_num(),
            "account_name": fake.name(),
            "bank_name": bank_name,
            "bank_swift": bank_swift,
            "country": high_risk_country["code"],
            "currency": "USD",
        }
        self.beneficiaries.append(ben)
        amount = round(random.uniform(5_000, 100_000), 2)
        device = random.choice(self.devices)
        ip = random.choice([i for i in self.ips if i["country"] == high_risk_country["code"]] or self.ips)
        return [{
            "id": _uid(), "reference": _txn_ref(),
            "sender_account_id": sender["id"],
            "receiver_account_id": None,
            "beneficiary_id": ben_id,
            "merchant_id": None,
            "amount": amount, "currency": "USD",
            "exchange_rate": 1.0, "amount_usd": amount,
            "transaction_type": "WIRE", "channel": "ONLINE",
            "status": "COMPLETED",
            "timestamp": (base_time - timedelta(hours=random.uniform(0, 72))).isoformat(),
            "description": "International wire",
            "device_id": device["id"], "ip_id": ip["ip"],
            "is_fraud": True, "fraud_type": "HIGH_RISK_CORRIDOR",
        }]

    def _make_rapid_velocity_pattern(self, base_time: datetime) -> list[dict]:
        """10+ transactions from one account within 1 hour."""
        sender = random.choice(self.accounts)
        receivers = random.sample([a for a in self.accounts if a["id"] != sender["id"]], min(10, len(self.accounts) - 1))
        txns = []
        device = random.choice(self.devices)
        ip = random.choice(self.ips)
        for i, receiver in enumerate(receivers):
            amount = round(random.uniform(500, 2000), 2)
            offset = timedelta(minutes=i * 4)
            txns.append({
                "id": _uid(), "reference": _txn_ref(),
                "sender_account_id": sender["id"],
                "receiver_account_id": receiver["id"],
                "beneficiary_id": None, "merchant_id": None,
                "amount": amount, "currency": "USD",
                "exchange_rate": 1.0, "amount_usd": amount,
                "transaction_type": "ACH", "channel": "API",
                "status": "COMPLETED",
                "timestamp": (base_time - offset).isoformat(),
                "description": "Automated payment",
                "device_id": device["id"], "ip_id": ip["ip"],
                "is_fraud": True, "fraud_type": "RAPID_VELOCITY",
            })
        return txns

    # ── Orchestrate data generation ───────────────────────────────

    def generate(self, target_total: int = 1000) -> None:
        console.print("[bold cyan]Seeding reference data...[/]")
        self._seed_countries()
        self._seed_devices()
        self._seed_ips()
        self._seed_merchants()
        self._seed_customers(120)
        self._seed_accounts((1, 3))

        base_time = datetime.utcnow()

        # --- Fraud patterns (≈ 15% of target) ---
        console.print("[bold yellow]Generating fraud patterns...[/]")
        fraud_txns: list[dict] = []

        # Structuring: 3 instances × ~5 txns = 15 txns
        for _ in range(3):
            fraud_txns.extend(self._make_structuring_pattern(base_time))

        # Smurfing: 2 instances × ~8 txns = 16 txns
        for _ in range(2):
            fraud_txns.extend(self._make_smurfing_pattern(base_time))

        # Layering: 4 instances × ~4 txns = 16 txns
        for _ in range(4):
            fraud_txns.extend(self._make_layering_pattern(base_time))

        # Round-tripping: 5 instances × 2 txns = 10 txns
        for _ in range(5):
            fraud_txns.extend(self._make_round_trip_pattern(base_time))

        # Dormant burst: 5 instances × ~3 txns = 15 txns
        for _ in range(5):
            fraud_txns.extend(self._make_dormant_burst_pattern(base_time))

        # High-risk corridor: 15 instances × 1 txn = 15 txns
        for _ in range(15):
            fraud_txns.extend(self._make_high_risk_corridor(base_time))

        # Rapid velocity: 3 instances × 10 txns = 30 txns
        for _ in range(3):
            fraud_txns.extend(self._make_rapid_velocity_pattern(base_time))

        console.print(f"[yellow]Generated {len(fraud_txns)} fraud transactions[/]")

        # --- Normal transactions to fill remaining quota ---
        normal_needed = max(0, target_total - len(fraud_txns))
        console.print(f"[bold green]Generating {normal_needed} normal transactions...[/]")
        normal_txns = [self._make_normal_txn(base_time) for _ in range(normal_needed)]

        self.transactions = fraud_txns + normal_txns
        random.shuffle(self.transactions)

        fraud_count = sum(1 for t in self.transactions if t["is_fraud"])
        console.print(f"[bold]Total: {len(self.transactions)} transactions | "
                      f"Fraud: {fraud_count} ({fraud_count/len(self.transactions)*100:.1f}%)[/]")

    # ── Write to Neo4j ────────────────────────────────────────────

    def _write_countries(self, session) -> None:
        for c in self.countries:
            session.run(
                """MERGE (c:Country {code: $code})
                   SET c += {name: $name, fatf_risk: $fatf_risk,
                             is_sanctioned: $is_sanctioned, is_tax_haven: $is_tax_haven}""",
                **c
            )

    def _write_devices(self, session) -> None:
        for d in self.devices:
            session.run(
                """MERGE (d:Device {id: $id})
                   SET d += {fingerprint: $fingerprint, device_type: $device_type,
                             user_agent: $user_agent}""",
                **d
            )

    def _write_ips(self, session) -> None:
        for ip in self.ips:
            session.run(
                """MERGE (i:IPAddress {ip: $ip})
                   SET i += {country: $country, city: $city, is_vpn: $is_vpn,
                             is_tor: $is_tor, is_proxy: $is_proxy, asn: $asn, isp: $isp}
                   WITH i
                   MERGE (c:Country {code: $country})
                   MERGE (i)-[:GEOLOCATED_IN]->(c)""",
                **ip
            )

    def _write_merchants(self, session) -> None:
        for m in self.merchants:
            session.run(
                """MERGE (m:Merchant {id: $id})
                   SET m += {name: $name, mcc_code: $mcc_code, category: $category,
                             country: $country, risk_level: $risk_level}""",
                **m
            )

    def _write_customers(self, session) -> None:
        for c in self.customers:
            session.run(
                """MERGE (cu:Customer {id: $id})
                   SET cu += {name: $name, date_of_birth: $date_of_birth,
                              nationality: $nationality,
                              country_of_residence: $country_of_residence,
                              kyc_level: $kyc_level, pep_flag: $pep_flag,
                              sanctions_flag: $sanctions_flag,
                              risk_tier: $risk_tier, created_at: $created_at,
                              customer_type: $customer_type}
                   WITH cu
                   MERGE (c:Country {code: $country_of_residence})
                   MERGE (cu)-[:RESIDENT_OF]->(c)""",
                **c
            )

    def _write_accounts(self, session) -> None:
        for a in self.accounts:
            session.run(
                """MERGE (ac:Account {id: $id})
                   SET ac += {account_number: $account_number,
                              customer_id: $customer_id,
                              account_type: $account_type, currency: $currency,
                              balance: $balance, country: $country,
                              bank_name: $bank_name, bank_swift: $bank_swift,
                              status: $status, created_at: $created_at,
                              last_active: $last_active,
                              average_monthly_volume: $average_monthly_volume,
                              typical_transaction_size: $typical_transaction_size}
                   WITH ac
                   MATCH (cu:Customer {id: $customer_id})
                   MERGE (cu)-[:OWNS]->(ac)
                   WITH ac
                   MERGE (c:Country {code: $country})
                   MERGE (ac)-[:BASED_IN]->(c)""",
                **a
            )

    def _write_beneficiaries(self, session) -> None:
        for b in self.beneficiaries:
            session.run(
                """MERGE (b:BeneficiaryAccount {id: $id})
                   SET b += {account_number: $account_number,
                             account_name: $account_name,
                             bank_name: $bank_name, bank_swift: $bank_swift,
                             country: $country, currency: $currency}""",
                **b
            )

    def _write_transactions(self, session) -> None:
        for t in self.transactions:
            params = {k: v for k, v in t.items()}
            session.run(
                """MERGE (tx:Transaction {id: $id})
                   SET tx += {reference: $reference, amount: $amount,
                              currency: $currency, exchange_rate: $exchange_rate,
                              amount_usd: $amount_usd,
                              transaction_type: $transaction_type,
                              channel: $channel, status: $status,
                              timestamp: $timestamp, description: $description,
                              is_fraud: $is_fraud, fraud_type: $fraud_type}
                   WITH tx
                   MATCH (sender:Account {id: $sender_account_id})
                   MERGE (sender)-[:INITIATED]->(tx)""",
                **params
            )
            if t.get("receiver_account_id"):
                session.run(
                    """MATCH (tx:Transaction {id: $txn_id})
                       MATCH (receiver:Account {id: $receiver_id})
                       MERGE (tx)-[:CREDITED_TO]->(receiver)""",
                    txn_id=t["id"], receiver_id=t["receiver_account_id"]
                )
            if t.get("merchant_id"):
                session.run(
                    """MATCH (tx:Transaction {id: $txn_id})
                       MATCH (m:Merchant {id: $merchant_id})
                       MERGE (tx)-[:PAID_TO]->(m)""",
                    txn_id=t["id"], merchant_id=t["merchant_id"]
                )
            if t.get("device_id"):
                session.run(
                    """MATCH (tx:Transaction {id: $txn_id})
                       MATCH (d:Device {id: $device_id})
                       MERGE (tx)-[:ORIGINATED_FROM]->(d)""",
                    txn_id=t["id"], device_id=t["device_id"]
                )
            if t.get("ip_id"):
                session.run(
                    """MATCH (tx:Transaction {id: $txn_id})
                       MATCH (i:IPAddress {ip: $ip})
                       MERGE (tx)-[:SOURCED_FROM]->(i)""",
                    txn_id=t["id"], ip=t["ip_id"]
                )
            if t.get("beneficiary_id"):
                session.run(
                    """MATCH (tx:Transaction {id: $txn_id})
                       MATCH (b:BeneficiaryAccount {id: $ben_id})
                       MERGE (tx)-[:SENT_TO_EXTERNAL]->(b)""",
                    txn_id=t["id"], ben_id=t["beneficiary_id"]
                )

    def persist(self) -> None:
        console.print("[bold cyan]Applying schema constraints & indexes...[/]")
        with neo4j_session() as session:
            apply_schema(session)

        console.print("[bold cyan]Writing countries...[/]")
        with neo4j_session() as session:
            self._write_countries(session)

        console.print("[bold cyan]Writing devices...[/]")
        with neo4j_session() as session:
            self._write_devices(session)

        console.print("[bold cyan]Writing IP addresses...[/]")
        with neo4j_session() as session:
            self._write_ips(session)

        console.print("[bold cyan]Writing merchants...[/]")
        with neo4j_session() as session:
            self._write_merchants(session)

        console.print("[bold cyan]Writing customers...[/]")
        with neo4j_session() as session:
            self._write_customers(session)

        console.print("[bold cyan]Writing accounts...[/]")
        with neo4j_session() as session:
            self._write_accounts(session)

        console.print("[bold cyan]Writing beneficiaries...[/]")
        with neo4j_session() as session:
            self._write_beneficiaries(session)

        console.print(f"[bold cyan]Writing {len(self.transactions)} transactions...[/]")
        # Batch in chunks for performance
        chunk_size = 50
        for i in range(0, len(self.transactions), chunk_size):
            chunk = self.transactions[i:i + chunk_size]
            with neo4j_session() as session:
                self._write_transactions(session)
                # just write this chunk
            if (i + chunk_size) % 200 == 0 or i + chunk_size >= len(self.transactions):
                console.print(f"  Written {min(i + chunk_size, len(self.transactions))}/{len(self.transactions)}")

        console.print("[bold green]✓ All data persisted to Neo4j[/]")

    def run(self, target_total: int = 1000) -> None:
        self.generate(target_total)
        self.persist()
