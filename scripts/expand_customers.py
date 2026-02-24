"""
expand_customers.py
===================
Adds 50,000 new customers with a realistic GLOBAL country distribution
spanning 60+ countries across all continents, then generates ~3 transactions
per customer to maintain a healthy world-map heatmap.

Key improvements over expand_data.py:
  • 60+ countries weighted by real-world financial-hub importance
  • Optimised Neo4j writes using UNWIND batches (≈10× faster)
  • Cross-border transactions between all new regions
  • ~15 % fraud rate preserved
  • Models retrained on the enlarged dataset

Estimated new records:
  Customers:    50 000
  Accounts:    ~100 000  (1-3 per customer)
  Transactions: ~150 000  (≈3 per customer, +fraud patterns)
"""

import random
import uuid
from datetime import datetime, timedelta

from faker import Faker
from rich.console import Console
from rich.table import Table

from db.client import neo4j_session
from data.generator import (
    AMLDataGenerator, _uid, _txn_ref, _account_num, _pick_bank,
    MCC_CODES,
)

console = Console()
fake    = Faker()
random.seed(2028)
Faker.seed(2028)

# ── Config ─────────────────────────────────────────────────────────────────────

NEW_CUSTOMERS      = 50_000
TXNS_PER_CUSTOMER  = 3          # average; fraud patterns add more
FRAUD_RATE         = 0.15
BATCH_SIZE         = 500        # UNWIND batch size for Neo4j writes

# ── Extended global country list ───────────────────────────────────────────────
# Each entry: (code, name, fatf_risk, is_sanctioned, is_tax_haven, weight)
# weight = relative sampling probability

WORLD_COUNTRIES: list[dict] = [
    # ── North America ──────────────────────────────────────────────────────────
    {"code":"US","name":"United States",    "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":14},
    {"code":"CA","name":"Canada",           "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":7},
    {"code":"MX","name":"Mexico",           "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":5},
    # ── Western Europe ─────────────────────────────────────────────────────────
    {"code":"GB","name":"United Kingdom",   "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":10},
    {"code":"DE","name":"Germany",          "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":8},
    {"code":"FR","name":"France",           "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":7},
    {"code":"ES","name":"Spain",            "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":5},
    {"code":"IT","name":"Italy",            "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":5},
    {"code":"NL","name":"Netherlands",      "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":5},
    {"code":"CH","name":"Switzerland",      "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":True, "weight":4},
    {"code":"SE","name":"Sweden",           "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":3},
    {"code":"NO","name":"Norway",           "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":3},
    {"code":"DK","name":"Denmark",          "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":3},
    {"code":"FI","name":"Finland",          "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":2},
    {"code":"BE","name":"Belgium",          "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":3},
    {"code":"AT","name":"Austria",          "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":3},
    {"code":"PT","name":"Portugal",         "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":2},
    {"code":"IE","name":"Ireland",          "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":2},
    # ── Eastern Europe ─────────────────────────────────────────────────────────
    {"code":"PL","name":"Poland",           "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":3},
    {"code":"CZ","name":"Czech Republic",   "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":2},
    {"code":"HU","name":"Hungary",          "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":2},
    {"code":"RO","name":"Romania",          "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":2},
    {"code":"UA","name":"Ukraine",          "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":2},
    # ── Asia-Pacific ───────────────────────────────────────────────────────────
    {"code":"SG","name":"Singapore",        "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":8},
    {"code":"JP","name":"Japan",            "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":7},
    {"code":"AU","name":"Australia",        "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":6},
    {"code":"NZ","name":"New Zealand",      "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":2},
    {"code":"KR","name":"South Korea",      "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":6},
    {"code":"HK","name":"Hong Kong",        "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":5},
    {"code":"TW","name":"Taiwan",           "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":4},
    {"code":"CN","name":"China",            "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":10},
    {"code":"IN","name":"India",            "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":9},
    {"code":"TH","name":"Thailand",         "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":4},
    {"code":"MY","name":"Malaysia",         "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":4},
    {"code":"ID","name":"Indonesia",        "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":4},
    {"code":"PH","name":"Philippines",      "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":3},
    {"code":"VN","name":"Vietnam",          "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":3},
    {"code":"BD","name":"Bangladesh",       "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":2},
    # ── Middle East ────────────────────────────────────────────────────────────
    {"code":"AE","name":"United Arab Emirates","fatf_risk":"MEDIUM","is_sanctioned":False,"is_tax_haven":False,"weight":6},
    {"code":"SA","name":"Saudi Arabia",     "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":5},
    {"code":"QA","name":"Qatar",            "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":3},
    {"code":"KW","name":"Kuwait",           "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":2},
    {"code":"IL","name":"Israel",           "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":3},
    {"code":"TR","name":"Turkey",           "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":4},
    {"code":"JO","name":"Jordan",           "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":2},
    # ── Africa ─────────────────────────────────────────────────────────────────
    {"code":"ZA","name":"South Africa",     "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":4},
    {"code":"NG","name":"Nigeria",          "fatf_risk":"HIGH",   "is_sanctioned":False,"is_tax_haven":False,"weight":3},
    {"code":"KE","name":"Kenya",            "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":3},
    {"code":"GH","name":"Ghana",            "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":2},
    {"code":"EG","name":"Egypt",            "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":3},
    {"code":"MA","name":"Morocco",          "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":2},
    {"code":"ET","name":"Ethiopia",         "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":2},
    {"code":"TZ","name":"Tanzania",         "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":1},
    # ── Latin America ──────────────────────────────────────────────────────────
    {"code":"BR","name":"Brazil",           "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":7},
    {"code":"AR","name":"Argentina",        "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":4},
    {"code":"CL","name":"Chile",            "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":False,"weight":3},
    {"code":"CO","name":"Colombia",         "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":3},
    {"code":"PE","name":"Peru",             "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":False,"weight":2},
    # ── Tax Havens ─────────────────────────────────────────────────────────────
    {"code":"KY","name":"Cayman Islands",   "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":True, "weight":2},
    {"code":"VG","name":"British Virgin Islands","fatf_risk":"MEDIUM","is_sanctioned":False,"is_tax_haven":True,"weight":2},
    {"code":"PA","name":"Panama",           "fatf_risk":"MEDIUM", "is_sanctioned":False,"is_tax_haven":True, "weight":2},
    {"code":"LU","name":"Luxembourg",       "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":True, "weight":2},
    {"code":"MT","name":"Malta",            "fatf_risk":"LOW",    "is_sanctioned":False,"is_tax_haven":True, "weight":1},
    # ── High-risk / Sanctioned (small weight but present for fraud patterns) ───
    {"code":"RU","name":"Russia",           "fatf_risk":"HIGH",   "is_sanctioned":False,"is_tax_haven":False,"weight":2},
    {"code":"MM","name":"Myanmar",          "fatf_risk":"HIGH",   "is_sanctioned":False,"is_tax_haven":False,"weight":1},
    {"code":"VE","name":"Venezuela",        "fatf_risk":"HIGH",   "is_sanctioned":False,"is_tax_haven":False,"weight":1},
    {"code":"PK","name":"Pakistan",         "fatf_risk":"HIGH",   "is_sanctioned":False,"is_tax_haven":False,"weight":2},
    {"code":"AF","name":"Afghanistan",      "fatf_risk":"HIGH",   "is_sanctioned":False,"is_tax_haven":False,"weight":1},
    {"code":"IR","name":"Iran",             "fatf_risk":"BLACKLIST","is_sanctioned":True,"is_tax_haven":False,"weight":1},
    {"code":"KP","name":"North Korea",      "fatf_risk":"BLACKLIST","is_sanctioned":True,"is_tax_haven":False,"weight":1},
    {"code":"SY","name":"Syria",            "fatf_risk":"BLACKLIST","is_sanctioned":True,"is_tax_haven":False,"weight":1},
]

HIGH_RISK_WORLD  = [c for c in WORLD_COUNTRIES if c["fatf_risk"] in ("HIGH","BLACKLIST")]
LOW_RISK_WORLD   = [c for c in WORLD_COUNTRIES if c["fatf_risk"] == "LOW"]

_WEIGHTS = [c["weight"] for c in WORLD_COUNTRIES]


def _pick_world_country(risk: str = "any") -> dict:
    if risk == "high":
        return random.choice(HIGH_RISK_WORLD)
    elif risk == "low":
        return random.choice(LOW_RISK_WORLD)
    return random.choices(WORLD_COUNTRIES, weights=_WEIGHTS)[0]


# ── Entity factories ───────────────────────────────────────────────────────────

def make_customers(n: int) -> list[dict]:
    kyc_levels = ["BASIC", "ENHANCED", "SIMPLIFIED"]
    risk_tiers = ["LOW", "LOW", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    result = []
    for _ in range(n):
        country   = _pick_world_country()
        pep       = random.random() < 0.04
        sanctions = random.random() < 0.01
        result.append({
            "id":                   _uid(),
            "name":                 fake.name(),
            "date_of_birth":        fake.date_of_birth(minimum_age=18, maximum_age=85).isoformat(),
            "nationality":          country["code"],
            "country_of_residence": country["code"],
            "kyc_level":            random.choice(kyc_levels),
            "pep_flag":             pep,
            "sanctions_flag":       sanctions,
            "risk_tier":            random.choices(risk_tiers)[0],
            "created_at":           fake.date_time_between(start_date="-5y", end_date="-1m").isoformat(),
            "customer_type":        "CORPORATE" if random.random() < 0.2 else "INDIVIDUAL",
        })
    return result


def make_accounts(customers: list[dict]) -> tuple[list[dict], dict]:
    now    = datetime.utcnow()
    types  = ["CURRENT", "SAVINGS", "BUSINESS", "PREPAID"]
    currs  = ["USD", "USD", "USD", "EUR", "GBP", "EUR", "JPY", "AED", "SGD", "INR", "BRL", "AUD"]
    accounts: list[dict]       = []
    cust_map: dict[str, list]  = {}

    for customer in customers:
        n_accts = random.randint(1, 3)
        cust_map[customer["id"]] = []
        for _ in range(n_accts):
            bank_name, bank_swift = _pick_bank()
            country   = _pick_world_country()
            typical   = round(random.uniform(100, 8000), 2)
            created   = fake.date_time_between(start_date="-4y", end_date="-7d")
            last_act  = now - timedelta(days=random.randint(0, 365))
            acct = {
                "id":                       _uid(),
                "account_number":           _account_num(),
                "customer_id":              customer["id"],
                "account_type":             random.choice(types),
                "currency":                 random.choice(currs),
                "balance":                  round(random.uniform(500, 100_000), 2),
                "country":                  country["code"],
                "bank_name":                bank_name,
                "bank_swift":               bank_swift,
                "status":                   "ACTIVE",
                "created_at":               created.isoformat(),
                "last_active":              last_act.isoformat(),
                "average_monthly_volume":   round(typical * random.uniform(10, 40), 2),
                "typical_transaction_size": typical,
            }
            accounts.append(acct)
            cust_map[customer["id"]].append(acct["id"])

    return accounts, cust_map


# ── Transaction generator ──────────────────────────────────────────────────────

class WorldBatchGenerator(AMLDataGenerator):
    """Uses the global world country pool for more diverse transactions."""

    def load_pool(self, accounts, devices, ips, merchants):
        self.accounts  = accounts
        self.devices   = devices
        self.ips       = ips
        self.merchants = merchants
        self._account_index = {a["id"]: a for a in accounts}

    # ── Fast overrides: avoid O(n) list comprehensions ────────────────────────
    # For pools of 1k+ accounts the collision probability is negligible, so
    # we just sample from the full list and retry once if there's a collision.

    def _two_accounts(self):
        """Return two distinct accounts without filtering the full list."""
        a = random.choice(self.accounts)
        b = random.choice(self.accounts)
        if b["id"] == a["id"] and len(self.accounts) > 1:
            b = random.choice(self.accounts)
        return a, b

    def _n_accounts(self, n: int):
        """Return n distinct accounts quickly."""
        k = min(n, len(self.accounts))
        return random.sample(self.accounts, k)

    def _make_structuring_pattern(self, base_time):
        sender    = random.choice(self.accounts)
        receivers = self._n_accounts(5)
        txns = []
        for i, receiver in enumerate(receivers):
            amount = round(random.uniform(9000, 9999), 2)
            device = random.choice(self.devices)
            ip     = random.choice(self.ips)
            txns.append({
                "id": _uid(), "reference": _txn_ref(),
                "sender_account_id": sender["id"],
                "receiver_account_id": receiver["id"],
                "beneficiary_id": None, "merchant_id": None,
                "amount": amount, "currency": "USD", "exchange_rate": 1.0, "amount_usd": amount,
                "transaction_type": "ACH", "channel": "ONLINE", "status": "COMPLETED",
                "timestamp": (base_time - timedelta(hours=random.uniform(0, 6) + i * 0.5)).isoformat(),
                "description": "Business payment",
                "device_id": device["id"], "ip_id": ip["ip"],
                "is_fraud": True, "fraud_type": "STRUCTURING",
            })
        return txns

    def _make_smurfing_pattern(self, base_time):
        accounts  = self._n_accounts(9)
        aggregator, senders = accounts[0], accounts[1:]
        txns = []
        for i, sender in enumerate(senders):
            amount = round(random.uniform(1000, 4999), 2)
            device = random.choice(self.devices)
            ip     = random.choice(self.ips)
            txns.append({
                "id": _uid(), "reference": _txn_ref(),
                "sender_account_id": sender["id"],
                "receiver_account_id": aggregator["id"],
                "beneficiary_id": None, "merchant_id": None,
                "amount": amount, "currency": "USD", "exchange_rate": 1.0, "amount_usd": amount,
                "transaction_type": "ACH", "channel": "MOBILE", "status": "COMPLETED",
                "timestamp": (base_time - timedelta(hours=random.uniform(0, 48) + i * 0.25)).isoformat(),
                "description": "Transfer",
                "device_id": device["id"], "ip_id": ip["ip"],
                "is_fraud": True, "fraud_type": "SMURFING",
            })
        return txns

    def _make_layering_pattern(self, base_time):
        chain  = self._n_accounts(5)
        amount = round(random.uniform(20_000, 200_000), 2)
        txns   = []
        for i in range(len(chain) - 1):
            amount = round(amount * random.uniform(0.90, 0.99), 2)
            device = random.choice(self.devices)
            ip     = random.choice(self.ips)
            txns.append({
                "id": _uid(), "reference": _txn_ref(),
                "sender_account_id":   chain[i]["id"],
                "receiver_account_id": chain[i + 1]["id"],
                "beneficiary_id": None, "merchant_id": None,
                "amount": amount, "currency": "USD", "exchange_rate": 1.0, "amount_usd": amount,
                "transaction_type": "WIRE", "channel": "ONLINE", "status": "COMPLETED",
                "timestamp": (base_time - timedelta(hours=i * random.uniform(0.5, 3))).isoformat(),
                "description": "Invoice settlement",
                "device_id": device["id"], "ip_id": ip["ip"],
                "is_fraud": True, "fraud_type": "LAYERING",
            })
        return txns

    def _make_round_trip_pattern(self, base_time):
        origin, intermediary = self._n_accounts(2)
        amount = round(random.uniform(10_000, 80_000), 2)
        device = random.choice(self.devices)
        ip     = random.choice(self.ips)
        return [
            {"id": _uid(), "reference": _txn_ref(),
             "sender_account_id": origin["id"], "receiver_account_id": intermediary["id"],
             "beneficiary_id": None, "merchant_id": None,
             "amount": amount, "currency": "USD", "exchange_rate": 1.0, "amount_usd": amount,
             "transaction_type": "WIRE", "channel": "ONLINE", "status": "COMPLETED",
             "timestamp": (base_time - timedelta(hours=30)).isoformat(),
             "description": "Consulting fee",
             "device_id": device["id"], "ip_id": ip["ip"],
             "is_fraud": True, "fraud_type": "ROUND_TRIP"},
            {"id": _uid(), "reference": _txn_ref(),
             "sender_account_id": intermediary["id"], "receiver_account_id": origin["id"],
             "beneficiary_id": None, "merchant_id": None,
             "amount": round(amount * 0.95, 2), "currency": "USD", "exchange_rate": 1.0,
             "amount_usd": round(amount * 0.95, 2),
             "transaction_type": "WIRE", "channel": "ONLINE", "status": "COMPLETED",
             "timestamp": (base_time - timedelta(hours=6)).isoformat(),
             "description": "Refund - overpayment",
             "device_id": device["id"], "ip_id": ip["ip"],
             "is_fraud": True, "fraud_type": "ROUND_TRIP"},
        ]

    def _make_dormant_burst_pattern(self, base_time):
        account   = random.choice(self.accounts)
        account["last_active"] = (base_time - timedelta(days=random.randint(120, 365))).isoformat()
        receivers = self._n_accounts(3)
        amount_each = round(random.uniform(15_000, 50_000), 2)
        device = random.choice(self.devices)
        ip     = random.choice(self.ips)
        txns   = []
        for i, receiver in enumerate(receivers):
            txns.append({
                "id": _uid(), "reference": _txn_ref(),
                "sender_account_id": account["id"], "receiver_account_id": receiver["id"],
                "beneficiary_id": None, "merchant_id": None,
                "amount": amount_each, "currency": "USD", "exchange_rate": 1.0, "amount_usd": amount_each,
                "transaction_type": "WIRE", "channel": "ONLINE", "status": "COMPLETED",
                "timestamp": (base_time - timedelta(hours=i * 2)).isoformat(),
                "description": "Urgent payment",
                "device_id": device["id"], "ip_id": ip["ip"],
                "is_fraud": True, "fraud_type": "DORMANT_BURST",
            })
        return txns

    def _make_rapid_velocity_pattern(self, base_time):
        sender    = random.choice(self.accounts)
        receivers = self._n_accounts(10)
        device    = random.choice(self.devices)
        ip        = random.choice(self.ips)
        txns      = []
        for i, receiver in enumerate(receivers):
            txns.append({
                "id": _uid(), "reference": _txn_ref(),
                "sender_account_id": sender["id"], "receiver_account_id": receiver["id"],
                "beneficiary_id": None, "merchant_id": None,
                "amount": round(random.uniform(500, 2000), 2),
                "currency": "USD", "exchange_rate": 1.0,
                "amount_usd": round(random.uniform(500, 2000), 2),
                "transaction_type": "ACH", "channel": "API", "status": "COMPLETED",
                "timestamp": (base_time - timedelta(minutes=i * 4)).isoformat(),
                "description": "Automated payment",
                "device_id": device["id"], "ip_id": ip["ip"],
                "is_fraud": True, "fraud_type": "RAPID_VELOCITY",
            })
        return txns

    # Override country picker to use global pool
    def _make_normal_txn(self, base_time: datetime) -> dict:
        sender   = random.choice(self.accounts)
        # Avoid O(n) filter: just pick another random account; collision probability
        # is negligible with pools of 10k+ accounts.
        receiver = random.choice(self.accounts)
        typical  = sender.get("typical_transaction_size", 500)
        amount   = round(random.uniform(typical * 0.5, typical * 2.0), 2)
        txn_type = random.choices(["WIRE","ACH","CARD","INTERNAL"], weights=[10,40,35,15])[0]
        channel  = random.choices(["ONLINE","MOBILE","BRANCH","ATM"], weights=[30,40,20,10])[0]
        device   = random.choice(self.devices) if channel in ("ONLINE","MOBILE") else None
        ip       = random.choice(self.ips)     if channel in ("ONLINE","MOBILE") else None
        merchant = random.choice(self.merchants) if txn_type == "CARD" else None
        timestamp= base_time - timedelta(hours=random.uniform(0, 24 * 60))
        return {
            "id": _uid(), "reference": _txn_ref(),
            "sender_account_id": sender["id"],
            "receiver_account_id": receiver["id"],
            "beneficiary_id": None,
            "merchant_id": merchant["id"] if merchant else None,
            "amount": amount, "currency": sender.get("currency","USD"),
            "exchange_rate": 1.0, "amount_usd": amount,
            "transaction_type": txn_type, "channel": channel,
            "status": "COMPLETED", "timestamp": timestamp.isoformat(),
            "description": fake.sentence(nb_words=4),
            "device_id": device["id"] if device else None,
            "ip_id": ip["ip"] if ip else None,
            "is_fraud": False, "fraud_type": None,
        }

    def _make_high_risk_corridor(self, base_time: datetime) -> list[dict]:
        """Override to use the new world country list for high-risk transfers."""
        sender         = random.choice(self.accounts)
        high_risk_cty  = random.choice(HIGH_RISK_WORLD)
        bank_name, bank_swift = _pick_bank()
        ben_id = _uid()
        ben = {
            "id": ben_id,
            "account_number": _account_num(),
            "account_name":   fake.name(),
            "bank_name":      bank_name,
            "bank_swift":     bank_swift,
            "country":        high_risk_cty["code"],
            "currency":       "USD",
        }
        self.beneficiaries.append(ben)
        amount = round(random.uniform(5_000, 100_000), 2)
        device = random.choice(self.devices)
        ip     = random.choice(self.ips)
        return [{
            "id": _uid(), "reference": _txn_ref(),
            "sender_account_id": sender["id"],
            "receiver_account_id": None, "beneficiary_id": ben_id,
            "merchant_id": None,
            "amount": amount, "currency": "USD", "exchange_rate": 1.0, "amount_usd": amount,
            "transaction_type": "WIRE", "channel": "ONLINE", "status": "COMPLETED",
            "timestamp": (base_time - timedelta(hours=random.uniform(0, 72))).isoformat(),
            "description": "International wire",
            "device_id": device["id"], "ip_id": ip["ip"],
            "is_fraud": True, "fraud_type": "HIGH_RISK_CORRIDOR",
        }]

    def generate_transactions(
        self,
        target_total: int,
        fraud_target: int,
        base_time: datetime,
        fraud_pool_accounts: list | None = None,
    ):
        """
        fraud_pool_accounts: small account list used for fraud patterns.
            Keeping this small (e.g. new_accounts only) avoids O(n) list
            comprehensions over the full 100k+ account pool.
        """
        # Temporarily swap self.accounts to the smaller fraud pool
        full_accounts = self.accounts
        if fraud_pool_accounts is not None:
            self.accounts = fraud_pool_accounts

        fraud_txns: list[dict] = []
        self.beneficiaries = []
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

        # Restore full pool for normal transactions
        self.accounts = full_accounts

        random.shuffle(fraud_txns)
        fraud_txns = fraud_txns[:fraud_target]
        normal_needed = target_total - len(fraud_txns)
        normal_txns   = [self._make_normal_txn(base_time) for _ in range(normal_needed)]
        all_txns = fraud_txns + normal_txns
        random.shuffle(all_txns)
        return all_txns, list(self.beneficiaries)


# ── UNWIND batch writers (≈10× faster than row-by-row) ───────────────────────

def _run_batch(session, query: str, rows: list[dict]) -> None:
    if not rows:
        return
    for start in range(0, len(rows), BATCH_SIZE):
        session.run(query, rows=rows[start:start + BATCH_SIZE])


def write_countries(countries: list[dict]) -> None:
    with neo4j_session() as s:
        _run_batch(s, """
            UNWIND $rows AS r
            MERGE (c:Country {code: r.code})
            SET c += {name: r.name, fatf_risk: r.fatf_risk,
                      is_sanctioned: r.is_sanctioned, is_tax_haven: r.is_tax_haven}
        """, countries)


def write_customers(customers: list[dict]) -> None:
    with neo4j_session() as s:
        _run_batch(s, """
            UNWIND $rows AS r
            MERGE (cu:Customer {id: r.id})
            SET cu += {name: r.name, date_of_birth: r.date_of_birth,
                       nationality: r.nationality,
                       country_of_residence: r.country_of_residence,
                       kyc_level: r.kyc_level, pep_flag: r.pep_flag,
                       sanctions_flag: r.sanctions_flag,
                       risk_tier: r.risk_tier, created_at: r.created_at,
                       customer_type: r.customer_type}
            WITH cu, r
            MERGE (cty:Country {code: r.country_of_residence})
            MERGE (cu)-[:RESIDENT_OF]->(cty)
        """, customers)


def write_accounts(accounts: list[dict]) -> None:
    with neo4j_session() as s:
        _run_batch(s, """
            UNWIND $rows AS r
            MERGE (ac:Account {id: r.id})
            SET ac += {account_number: r.account_number,
                       customer_id: r.customer_id,
                       account_type: r.account_type, currency: r.currency,
                       balance: r.balance, country: r.country,
                       bank_name: r.bank_name, bank_swift: r.bank_swift,
                       status: r.status, created_at: r.created_at,
                       last_active: r.last_active,
                       average_monthly_volume: r.average_monthly_volume,
                       typical_transaction_size: r.typical_transaction_size}
            WITH ac, r
            MATCH (cu:Customer {id: r.customer_id})
            MERGE (cu)-[:OWNS]->(ac)
            WITH ac, r
            MERGE (cty:Country {code: r.country})
            MERGE (ac)-[:BASED_IN]->(cty)
        """, accounts)


def write_beneficiaries(bens: list[dict]) -> None:
    if not bens:
        return
    with neo4j_session() as s:
        _run_batch(s, """
            UNWIND $rows AS r
            MERGE (b:BeneficiaryAccount {id: r.id})
            SET b += {account_number: r.account_number,
                      account_name: r.account_name,
                      bank_name: r.bank_name, bank_swift: r.bank_swift,
                      country: r.country, currency: r.currency}
        """, bens)


def write_transactions_fast(txns: list[dict], label: str) -> None:
    """
    Writes transactions using 5 UNWIND passes (one per relationship type).
    ≈10× faster than individual row writes.
    """
    total = len(txns)

    # ── Pass 1: Transaction nodes ──────────────────────────────────────────────
    console.print(f"  [{label}] Creating {total:,} transaction nodes…")
    with neo4j_session() as s:
        _run_batch(s, """
            UNWIND $rows AS r
            MERGE (tx:Transaction {id: r.id})
            SET tx += {reference: r.reference, amount: r.amount,
                       currency: r.currency, exchange_rate: r.exchange_rate,
                       amount_usd: r.amount_usd,
                       transaction_type: r.transaction_type,
                       channel: r.channel, status: r.status,
                       timestamp: r.timestamp, description: r.description,
                       is_fraud: r.is_fraud, fraud_type: r.fraud_type}
        """, txns)

    # ── Pass 2: INITIATED (sender → tx) ───────────────────────────────────────
    console.print(f"  [{label}] Linking INITIATED…")
    with neo4j_session() as s:
        _run_batch(s, """
            UNWIND $rows AS r
            MATCH (tx:Transaction {id: r.id})
            MATCH (a:Account {id: r.sender_account_id})
            MERGE (a)-[:INITIATED]->(tx)
        """, txns)

    # ── Pass 3: CREDITED_TO (tx → receiver) ───────────────────────────────────
    recv = [t for t in txns if t.get("receiver_account_id")]
    if recv:
        console.print(f"  [{label}] Linking CREDITED_TO ({len(recv):,})…")
        with neo4j_session() as s:
            _run_batch(s, """
                UNWIND $rows AS r
                MATCH (tx:Transaction {id: r.id})
                MATCH (a:Account {id: r.receiver_account_id})
                MERGE (tx)-[:CREDITED_TO]->(a)
            """, recv)

    # ── Pass 4: SENT_TO_EXTERNAL (tx → beneficiary) ───────────────────────────
    bene = [t for t in txns if t.get("beneficiary_id")]
    if bene:
        console.print(f"  [{label}] Linking SENT_TO_EXTERNAL ({len(bene):,})…")
        with neo4j_session() as s:
            _run_batch(s, """
                UNWIND $rows AS r
                MATCH (tx:Transaction {id: r.id})
                MATCH (b:BeneficiaryAccount {id: r.beneficiary_id})
                MERGE (tx)-[:SENT_TO_EXTERNAL]->(b)
            """, bene)

    # ── Pass 5: ORIGINATED_FROM + SOURCED_FROM (device / IP) ─────────────────
    devs = [t for t in txns if t.get("device_id")]
    ips  = [t for t in txns if t.get("ip_id")]
    if devs:
        with neo4j_session() as s:
            _run_batch(s, """
                UNWIND $rows AS r
                MATCH (tx:Transaction {id: r.id})
                MATCH (d:Device {id: r.device_id})
                MERGE (tx)-[:ORIGINATED_FROM]->(d)
            """, devs)
    if ips:
        with neo4j_session() as s:
            _run_batch(s, """
                UNWIND $rows AS r
                MATCH (tx:Transaction {id: r.id})
                MATCH (i:IPAddress {ip: r.ip_id})
                MERGE (tx)-[:SOURCED_FROM]->(i)
            """, ips)

    console.print(f"  [{label}] [green]✓ {total:,} transactions written[/]")


# ── Aux entities ───────────────────────────────────────────────────────────────

def make_devices(n: int = 300) -> list[dict]:
    types = ["MOBILE","DESKTOP","ATM","POS"]
    return [{
        "id": _uid(), "fingerprint": fake.md5(),
        "device_type": random.choices(types, weights=[50,30,10,10])[0],
        "user_agent": fake.user_agent(),
    } for _ in range(n)]


def make_ips(n: int = 250) -> list[dict]:
    result = []
    for _ in range(n):
        cty    = _pick_world_country()
        is_vpn = random.random() < 0.08
        is_tor = random.random() < 0.03
        result.append({
            "ip": fake.ipv4_public(), "country": cty["code"],
            "city": fake.city(), "is_vpn": is_vpn, "is_tor": is_tor,
            "is_proxy": is_vpn or random.random() < 0.04,
            "asn": f"AS{random.randint(1000,65000)}", "isp": fake.company(),
        })
    return result


def make_merchants(n: int = 120) -> list[dict]:
    result = []
    for _ in range(n):
        mcc, cat = random.choice(MCC_CODES)
        cty      = _pick_world_country("low")
        result.append({
            "id": _uid(), "name": fake.company(),
            "mcc_code": mcc, "category": cat,
            "country": cty["code"],
            "risk_level": "HIGH" if mcc in ("6011","6051","7995") else "LOW",
        })
    return result


def write_devices(devices: list[dict]) -> None:
    with neo4j_session() as s:
        _run_batch(s, """
            UNWIND $rows AS r
            MERGE (d:Device {id: r.id})
            SET d += {fingerprint: r.fingerprint, device_type: r.device_type,
                      user_agent: r.user_agent}
        """, devices)


def write_ips(ips: list[dict]) -> None:
    with neo4j_session() as s:
        _run_batch(s, """
            UNWIND $rows AS r
            MERGE (i:IPAddress {ip: r.ip})
            SET i += {country: r.country, city: r.city, is_vpn: r.is_vpn,
                      is_tor: r.is_tor, is_proxy: r.is_proxy, asn: r.asn, isp: r.isp}
            WITH i, r
            MERGE (c:Country {code: r.country})
            MERGE (i)-[:GEOLOCATED_IN]->(c)
        """, ips)


def write_merchants(merchants: list[dict]) -> None:
    with neo4j_session() as s:
        _run_batch(s, """
            UNWIND $rows AS r
            MERGE (m:Merchant {id: r.id})
            SET m += {name: r.name, mcc_code: r.mcc_code, category: r.category,
                      country: r.country, risk_level: r.risk_level}
        """, merchants)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(skip_entities: bool = False):
    """
    skip_entities=True → skip customer/account creation and only generate
    transactions for the accounts already in Neo4j (useful after aborted runs).
    """
    console.rule("[bold cyan]Global Customer Expansion — 50,000 New Customers[/]")

    # ── Step 1: Persist new world countries ───────────────────────────────────
    console.print("[cyan]Merging world country nodes…[/]")
    write_countries(WORLD_COUNTRIES)
    console.print(f"  [green]✓ {len(WORLD_COUNTRIES)} country nodes upserted[/]")

    # ── Step 2: Create / load auxiliary entities ───────────────────────────────
    console.print("\n[cyan]Loading all entities from Neo4j…[/]")
    with neo4j_session() as s:
        existing_accounts = [dict(r["a"]) for r in s.run("MATCH (a:Account) RETURN a")]
        existing_devices  = [dict(r["d"]) for r in s.run("MATCH (d:Device)  RETURN d")]
        existing_ips      = [dict(r["i"]) for r in s.run("MATCH (i:IPAddress) RETURN i")]
        existing_merchants= [dict(r["m"]) for r in s.run("MATCH (m:Merchant) RETURN m")]
        existing_customers= s.run("MATCH (c:Customer) RETURN count(c) AS n").single()["n"]
    console.print(
        f"  Existing: {existing_customers:,} customers | {len(existing_accounts):,} accounts | "
        f"{len(existing_devices)} devices | {len(existing_ips)} IPs"
    )

    new_accounts: list[dict] = []

    if not skip_entities:
        console.print("\n[cyan]Creating new auxiliary entities…[/]")
        devices   = make_devices(300)
        ips       = make_ips(250)
        merchants = make_merchants(120)
        write_devices(devices)
        write_ips(ips)
        write_merchants(merchants)
        console.print(f"  Devices: {len(devices)}  |  IPs: {len(ips)}  |  Merchants: {len(merchants)}")

        all_devices   = existing_devices   + devices
        all_ips       = existing_ips       + ips
        all_merchants = existing_merchants + merchants

        # ── Step 3: Create new customers + accounts ───────────────────────────
        console.print(f"\n[cyan]Creating {NEW_CUSTOMERS:,} new customers…[/]")
        new_customers = make_customers(NEW_CUSTOMERS)
        new_accounts, _ = make_accounts(new_customers)

        console.print(f"  Writing customers ({len(new_customers):,})…")
        write_customers(new_customers)
        console.print(f"  Writing accounts  ({len(new_accounts):,})…")
        write_accounts(new_accounts)
        console.print(f"  [green]✓ {len(new_customers):,} customers / {len(new_accounts):,} accounts written[/]")
    else:
        console.print("\n[yellow]⚡ Skipping customer/account creation (skip_entities=True)[/]")
        all_devices   = existing_devices
        all_ips       = existing_ips
        all_merchants = existing_merchants

    # All accounts for transaction pool (new + existing)
    all_accounts  = new_accounts + existing_accounts if new_accounts else existing_accounts

    # Fraud patterns run on a capped sub-pool to avoid O(n) list comprehensions
    FRAUD_POOL_CAP = 10_000
    fraud_pool = random.sample(all_accounts, min(FRAUD_POOL_CAP, len(all_accounts)))

    # ── Step 4: Generate transactions ─────────────────────────────────────────
    # When skipping entity creation, generate 3 txns per account that has none
    # (accounts created in the aborted runs).  Otherwise use the standard ratio.
    if skip_entities:
        # Find how many new accounts have 0 transactions
        with neo4j_session() as s:
            no_txn_accts = s.run("""
                MATCH (a:Account)
                WHERE NOT (a)-[:INITIATED]->(:Transaction)
                RETURN count(a) AS n
            """).single()["n"]
        target_txns  = no_txn_accts * TXNS_PER_CUSTOMER
        console.print(
            f"\n[cyan]{no_txn_accts:,} accounts have no transactions → "
            f"generating {target_txns:,} transactions[/]"
        )
    else:
        target_txns = (len(new_accounts) if new_accounts else len(all_accounts)) * TXNS_PER_CUSTOMER

    fraud_target = round(target_txns * FRAUD_RATE)
    console.print(f"  Fraud target: {fraud_target:,}  (~{FRAUD_RATE*100:.0f}%)")

    gen = WorldBatchGenerator()
    gen.load_pool(
        accounts  = all_accounts,
        devices   = all_devices,
        ips       = all_ips,
        merchants = all_merchants,
    )

    base_time = datetime.utcnow()

    # Generate in chunks to show progress and avoid huge in-memory lists
    CHUNK = 50_000
    total_written = 0
    chunks_needed = max(1, (target_txns + CHUNK - 1) // CHUNK)

    for chunk_i in range(chunks_needed):
        chunk_target = min(CHUNK, target_txns - total_written)
        chunk_fraud  = round(chunk_target * FRAUD_RATE)
        console.print(f"\n  Chunk {chunk_i+1}/{chunks_needed}: {chunk_target:,} transactions…")

        txns, bens = gen.generate_transactions(
            target_total        = chunk_target,
            fraud_target        = chunk_fraud,
            base_time           = base_time - timedelta(hours=chunk_i * 24),
            fraud_pool_accounts = fraud_pool,
        )

        fraud_count = sum(1 for t in txns if t["is_fraud"])
        console.print(
            f"    Generated {len(txns):,} "
            f"([red]{fraud_count} fraud[/] / [green]{len(txns)-fraud_count} normal[/])"
        )

        if bens:
            write_beneficiaries(bens)

        write_transactions_fast(txns, f"chunk-{chunk_i+1}")
        total_written += len(txns)

    # ── Step 5: Final DB totals ────────────────────────────────────────────────
    with neo4j_session() as s:
        db = {
            "Customers":    s.run("MATCH (c:Customer) RETURN count(c) AS n").single()["n"],
            "Accounts":     s.run("MATCH (a:Account)  RETURN count(a) AS n").single()["n"],
            "Transactions": s.run("MATCH (t:Transaction) RETURN count(t) AS n").single()["n"],
            "Fraud txns":   s.run("MATCH (t:Transaction {is_fraud:true}) RETURN count(t) AS n").single()["n"],
            "Countries":    s.run("MATCH (c:Country) RETURN count(c) AS n").single()["n"],
        }
        cty_dist = s.run("""
            MATCH (c:Customer)-[:RESIDENT_OF]->(cty:Country)
            RETURN cty.code AS cc, count(c) AS n
            ORDER BY n DESC LIMIT 10
        """).data()

    console.rule("[bold green]Expansion Complete[/]")
    tbl = Table(show_header=True, header_style="bold cyan")
    tbl.add_column("Metric"); tbl.add_column("Value", style="green")
    for k, v in db.items():
        tbl.add_row(k, f"{v:,}")
    console.print(tbl)

    console.print("\n[bold]Top 10 customer countries:[/]")
    for r in cty_dist:
        console.print(f"  {r['cc']}: {r['n']:,}")

    # ── Step 6: Retrain all models ─────────────────────────────────────────────
    console.rule("[bold cyan]Retraining ML Models on Expanded Dataset[/]")
    from ml.train import train_and_save_all
    metrics = train_and_save_all()

    console.rule("[bold green]All Done[/]")
    console.print(
        f"XGBoost: {metrics['xgb']['roc_auc']}  |  "
        f"SGD/SVM: {metrics['svm']['roc_auc']}  |  "
        f"KNN: {metrics['knn']['roc_auc']}"
    )


if __name__ == "__main__":
    import sys
    # Pass --skip-entities to only generate transactions (after aborted runs)
    skip = "--skip-entities" in sys.argv
    main(skip_entities=skip)
