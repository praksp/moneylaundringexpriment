"""
Feature Extractor
==================
Pulls contextual signals from the Neo4j graph for a given transaction.
Returns a FeatureVector used by both the Bayesian engine and the ML model.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional
import math

HIGH_RISK_COUNTRY_CODES = {
    "IR", "KP", "SY", "MM", "RU", "VE", "AF", "PK",
}
TAX_HAVEN_CODES = {"KY", "VG", "PA", "CH"}
STRUCTURING_THRESHOLD = 10_000.0
STRUCTURING_WINDOW = (9_000.0, 9_999.99)
HIGH_AMOUNT_THRESHOLD = 10_000.0
VERY_HIGH_AMOUNT_THRESHOLD = 50_000.0
VELOCITY_WINDOW_1H = 1
VELOCITY_WINDOW_24H = 24
VELOCITY_WINDOW_7D = 168
DORMANT_DAYS = 90


@dataclass
class FeatureVector:
    # ── Transaction-level ───────────────────────────────────────
    txn_id: str = ""
    amount_usd: float = 0.0
    is_wire: bool = False
    is_crypto: bool = False
    is_cash: bool = False
    is_card: bool = False
    is_cross_border: bool = False

    # Amount risk
    is_high_amount: bool = False            # > $10k
    is_very_high_amount: bool = False       # > $50k
    is_round_amount: bool = False           # Amount ends in 000
    in_structuring_band: bool = False       # $9000 - $9999.99
    log_amount: float = 0.0

    # Time features
    hour_of_day: int = 0
    is_night_transaction: bool = False      # 00:00 - 05:59
    is_weekend: bool = False
    day_of_week: int = 0

    # ── Sender account features ──────────────────────────────────
    sender_account_age_days: int = 0
    is_new_sender_account: bool = False     # < 30 days
    is_dormant_sender: bool = False         # last_active > 90 days ago
    sender_amount_deviation: float = 0.0   # amount / typical_size ratio

    # ── Receiver account features ────────────────────────────────
    receiver_account_age_days: int = 0
    is_new_receiver_account: bool = False

    # ── Customer / KYC features ─────────────────────────────────
    sender_is_pep: bool = False
    sender_is_sanctioned: bool = False
    sender_risk_tier_score: int = 0         # LOW=0 MED=1 HIGH=2 CRITICAL=3
    receiver_is_pep: bool = False
    receiver_is_sanctioned: bool = False

    # ── Geographic risk ──────────────────────────────────────────
    sender_country_risk: int = 0            # 0=low 1=medium 2=high 3=blacklist
    receiver_country_risk: int = 0
    beneficiary_country_risk: int = 0
    sender_to_high_risk: bool = False
    sender_to_tax_haven: bool = False
    ip_is_vpn: bool = False
    ip_is_tor: bool = False
    ip_country_mismatch: bool = False       # IP country ≠ sender account country

    # ── Velocity features (from graph queries) ───────────────────
    txn_count_1h: int = 0
    txn_count_24h: int = 0
    txn_count_7d: int = 0
    total_amount_24h: float = 0.0
    total_amount_7d: float = 0.0
    is_high_velocity_1h: bool = False       # ≥ 5 txns in 1h
    is_high_velocity_24h: bool = False      # ≥ 15 txns in 24h

    # ── Network / graph features ─────────────────────────────────
    structuring_count_24h: int = 0
    is_structuring: bool = False            # ≥ 2 structuring-band txns in 24h
    round_trip_count: int = 0
    is_round_trip: bool = False
    shared_device_user_count: int = 0
    device_shared: bool = False             # device used by > 1 customer
    network_hop_count: int = 0
    is_deep_layering: bool = False          # ≥ 3 hops in fund chain

    # ── Merchant risk ─────────────────────────────────────────────
    merchant_is_high_risk: bool = False
    merchant_is_gambling: bool = False
    merchant_is_fx: bool = False

    def to_ml_array(self) -> list[float]:
        """Return ordered numeric feature array for ML model."""
        return [
            self.log_amount,
            float(self.is_high_amount),
            float(self.is_very_high_amount),
            float(self.is_round_amount),
            float(self.in_structuring_band),
            float(self.is_wire),
            float(self.is_crypto),
            float(self.is_cash),
            float(self.is_cross_border),
            float(self.is_night_transaction),
            float(self.is_weekend),
            float(self.hour_of_day) / 23.0,
            min(self.sender_account_age_days, 3650) / 3650.0,
            float(self.is_new_sender_account),
            float(self.is_dormant_sender),
            min(self.sender_amount_deviation, 50.0) / 50.0,
            float(self.sender_is_pep),
            float(self.sender_is_sanctioned),
            float(self.sender_risk_tier_score) / 3.0,
            float(self.receiver_is_pep),
            float(self.receiver_is_sanctioned),
            float(self.sender_country_risk) / 3.0,
            float(self.receiver_country_risk) / 3.0,
            float(self.beneficiary_country_risk) / 3.0,
            float(self.sender_to_high_risk),
            float(self.sender_to_tax_haven),
            float(self.ip_is_vpn),
            float(self.ip_is_tor),
            float(self.ip_country_mismatch),
            min(self.txn_count_1h, 20) / 20.0,
            min(self.txn_count_24h, 50) / 50.0,
            min(self.txn_count_7d, 200) / 200.0,
            float(self.is_high_velocity_1h),
            float(self.is_high_velocity_24h),
            float(self.is_structuring),
            float(self.structuring_count_24h) / 10.0,
            float(self.is_round_trip),
            float(self.device_shared),
            min(self.shared_device_user_count, 10) / 10.0,
            min(self.network_hop_count, 5) / 5.0,
            float(self.is_deep_layering),
            float(self.merchant_is_high_risk),
            float(self.merchant_is_gambling),
            float(self.merchant_is_fx),
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "log_amount", "is_high_amount", "is_very_high_amount",
            "is_round_amount", "in_structuring_band",
            "is_wire", "is_crypto", "is_cash", "is_cross_border",
            "is_night_transaction", "is_weekend", "hour_norm",
            "sender_age_norm", "is_new_sender", "is_dormant_sender",
            "amount_deviation_norm",
            "sender_pep", "sender_sanctions", "sender_risk_tier_norm",
            "receiver_pep", "receiver_sanctions",
            "sender_country_risk_norm", "receiver_country_risk_norm",
            "beneficiary_country_risk_norm",
            "to_high_risk", "to_tax_haven",
            "ip_vpn", "ip_tor", "ip_country_mismatch",
            "velocity_1h_norm", "velocity_24h_norm", "velocity_7d_norm",
            "is_high_velocity_1h", "is_high_velocity_24h",
            "is_structuring", "structuring_count_norm",
            "is_round_trip", "device_shared", "device_users_norm",
            "hop_count_norm", "is_deep_layering",
            "merchant_high_risk", "merchant_gambling", "merchant_fx",
        ]


def _country_risk_score(country_code: str) -> int:
    FATF_BLACKLIST = {"IR", "KP", "SY"}
    FATF_HIGH = {"MM", "RU", "VE", "AF", "PK", "LA", "TZ", "YE"}
    FATF_MEDIUM = {"KY", "VG", "PA", "CH", "MX", "TR", "UA", "TH"}
    if country_code in FATF_BLACKLIST:
        return 3
    if country_code in FATF_HIGH:
        return 2
    if country_code in FATF_MEDIUM:
        return 1
    return 0


def _risk_tier_score(tier: str) -> int:
    return {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}.get(tier, 0)


def extract_features(txn_data: dict, graph_data: dict) -> FeatureVector:
    """
    Build a FeatureVector from transaction dict + pre-fetched graph context.

    Args:
        txn_data: Raw transaction dict from DB or API request
        graph_data: Dict containing sender, receiver, customer, device, ip, velocity counts etc.
    """
    fv = FeatureVector()
    fv.txn_id = txn_data.get("id", "")

    now = datetime.utcnow()
    ts_raw = txn_data.get("timestamp", now.isoformat())
    if isinstance(ts_raw, str):
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00").replace("+00:00", ""))
        except Exception:
            ts = now
    else:
        ts = ts_raw

    # ── Amount features ──
    amount = float(txn_data.get("amount_usd") or txn_data.get("amount", 0))
    fv.amount_usd = amount
    fv.log_amount = math.log1p(amount)
    fv.is_high_amount = amount >= HIGH_AMOUNT_THRESHOLD
    fv.is_very_high_amount = amount >= VERY_HIGH_AMOUNT_THRESHOLD
    fv.is_round_amount = (amount % 1000 == 0) and amount >= 1000
    fv.in_structuring_band = STRUCTURING_WINDOW[0] <= amount <= STRUCTURING_WINDOW[1]

    # ── Transaction type ──
    txn_type = txn_data.get("transaction_type", "")
    fv.is_wire = txn_type == "WIRE"
    fv.is_crypto = txn_type == "CRYPTO"
    fv.is_cash = txn_type == "CASH"
    fv.is_card = txn_type == "CARD"

    # ── Time features ──
    fv.hour_of_day = ts.hour
    fv.day_of_week = ts.weekday()
    fv.is_night_transaction = ts.hour < 6
    fv.is_weekend = ts.weekday() >= 5

    # ── Sender account ──
    sender = graph_data.get("sender", {})
    if sender:
        try:
            created = datetime.fromisoformat(str(sender.get("created_at", now.isoformat())).replace("Z", ""))
            fv.sender_account_age_days = (now - created).days
        except Exception:
            fv.sender_account_age_days = 365
        fv.is_new_sender_account = fv.sender_account_age_days < 30

        try:
            last_active_raw = sender.get("last_active")
            if last_active_raw:
                last_active = datetime.fromisoformat(str(last_active_raw).replace("Z", ""))
                fv.is_dormant_sender = (now - last_active).days > DORMANT_DAYS
        except Exception:
            pass

        typical = float(sender.get("typical_transaction_size", 500) or 500)
        fv.sender_amount_deviation = amount / typical if typical > 0 else 1.0

        sender_country = sender.get("country", "US")
        fv.sender_country_risk = _country_risk_score(sender_country)

    # ── Receiver account ──
    receiver = graph_data.get("receiver", {})
    if receiver:
        try:
            created = datetime.fromisoformat(str(receiver.get("created_at", now.isoformat())).replace("Z", ""))
            fv.receiver_account_age_days = (now - created).days
        except Exception:
            fv.receiver_account_age_days = 365
        fv.is_new_receiver_account = fv.receiver_account_age_days < 30

        receiver_country = receiver.get("country", "US")
        fv.receiver_country_risk = _country_risk_score(receiver_country)
        fv.is_cross_border = sender.get("country", "US") != receiver_country if sender else False
        fv.sender_to_high_risk = receiver_country in HIGH_RISK_COUNTRY_CODES
        fv.sender_to_tax_haven = receiver_country in TAX_HAVEN_CODES

    # ── Beneficiary (external wire) ──
    beneficiary = graph_data.get("beneficiary", {})
    if beneficiary:
        ben_country = beneficiary.get("country", "US")
        fv.beneficiary_country_risk = _country_risk_score(ben_country)
        fv.sender_to_high_risk = ben_country in HIGH_RISK_COUNTRY_CODES
        fv.is_cross_border = True

    # ── Sender customer ──
    sender_customer = graph_data.get("sender_customer", {})
    if sender_customer:
        fv.sender_is_pep = bool(sender_customer.get("pep_flag", False))
        fv.sender_is_sanctioned = bool(sender_customer.get("sanctions_flag", False))
        fv.sender_risk_tier_score = _risk_tier_score(sender_customer.get("risk_tier", "LOW"))

    # ── Receiver customer ──
    receiver_customer = graph_data.get("receiver_customer", {})
    if receiver_customer:
        fv.receiver_is_pep = bool(receiver_customer.get("pep_flag", False))
        fv.receiver_is_sanctioned = bool(receiver_customer.get("sanctions_flag", False))

    # ── Device / IP ──
    device = graph_data.get("device", {})
    ip = graph_data.get("ip", {})
    if ip:
        fv.ip_is_vpn = bool(ip.get("is_vpn", False))
        fv.ip_is_tor = bool(ip.get("is_tor", False))
        ip_country = ip.get("country", "US")
        sender_country = sender.get("country", "US") if sender else "US"
        fv.ip_country_mismatch = ip_country != sender_country

    # ── Velocity (from pre-computed graph queries) ──
    fv.txn_count_1h = int(graph_data.get("txn_count_1h", 0))
    fv.txn_count_24h = int(graph_data.get("txn_count_24h", 0))
    fv.txn_count_7d = int(graph_data.get("txn_count_7d", 0))
    fv.total_amount_24h = float(graph_data.get("total_amount_24h", 0.0))
    fv.total_amount_7d = float(graph_data.get("total_amount_7d", 0.0))
    fv.is_high_velocity_1h = fv.txn_count_1h >= 5
    fv.is_high_velocity_24h = fv.txn_count_24h >= 15

    # ── Structuring / network ──
    fv.structuring_count_24h = int(graph_data.get("structuring_count_24h", 0))
    fv.is_structuring = fv.structuring_count_24h >= 2 or fv.in_structuring_band
    fv.round_trip_count = int(graph_data.get("round_trip_count", 0))
    fv.is_round_trip = fv.round_trip_count > 0
    fv.shared_device_user_count = int(graph_data.get("shared_device_user_count", 1))
    fv.device_shared = fv.shared_device_user_count > 1
    fv.network_hop_count = int(graph_data.get("network_hop_count", 1))
    fv.is_deep_layering = fv.network_hop_count >= 3

    # ── Merchant ──
    merchant = graph_data.get("merchant", {})
    if merchant:
        fv.merchant_is_high_risk = merchant.get("risk_level", "LOW") == "HIGH"
        mcc = merchant.get("mcc_code", "")
        fv.merchant_is_gambling = mcc == "7995"
        fv.merchant_is_fx = mcc == "6051"

    return fv
