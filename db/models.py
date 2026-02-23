"""
Pydantic models representing graph nodes and relationships.
Follows the ISO 20022 financial messaging standard for field naming.
"""
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


def new_id() -> str:
    return str(uuid.uuid4())


# ──────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────

class AccountType(str, Enum):
    CURRENT = "CURRENT"
    SAVINGS = "SAVINGS"
    BUSINESS = "BUSINESS"
    CRYPTO = "CRYPTO"
    PREPAID = "PREPAID"


class TransactionType(str, Enum):
    WIRE = "WIRE"           # International wire transfer
    ACH = "ACH"             # Domestic ACH
    CARD = "CARD"           # Card payment
    CRYPTO = "CRYPTO"       # Cryptocurrency transfer
    CASH = "CASH"           # Cash deposit/withdrawal
    INTERNAL = "INTERNAL"   # Internal transfer between own accounts


class TransactionChannel(str, Enum):
    ONLINE = "ONLINE"
    MOBILE = "MOBILE"
    BRANCH = "BRANCH"
    ATM = "ATM"
    API = "API"


class TransactionStatus(str, Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REVERSED = "REVERSED"
    FLAGGED = "FLAGGED"


class KYCLevel(str, Enum):
    BASIC = "BASIC"         # Name + ID verified
    ENHANCED = "ENHANCED"   # Full KYC with docs
    SIMPLIFIED = "SIMPLIFIED"  # Low-risk simplified


class RiskTier(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TransactionOutcome(str, Enum):
    ALLOW = "ALLOW"
    CHALLENGE = "CHALLENGE"
    DECLINE = "DECLINE"


# ──────────────────────────────────────────────
# Graph Node Models
# ──────────────────────────────────────────────

class Country(BaseModel):
    code: str                       # ISO 3166-1 alpha-2
    name: str
    fatf_risk: str = "LOW"          # LOW / MEDIUM / HIGH / BLACKLIST
    is_sanctioned: bool = False
    is_tax_haven: bool = False


class Customer(BaseModel):
    id: str = Field(default_factory=new_id)
    name: str
    date_of_birth: Optional[str] = None
    nationality: str                # ISO country code
    country_of_residence: str
    kyc_level: KYCLevel = KYCLevel.BASIC
    pep_flag: bool = False          # Politically Exposed Person
    sanctions_flag: bool = False
    risk_tier: RiskTier = RiskTier.LOW
    created_at: datetime = Field(default_factory=datetime.utcnow)
    customer_type: str = "INDIVIDUAL"  # INDIVIDUAL / CORPORATE


class Account(BaseModel):
    id: str = Field(default_factory=new_id)
    account_number: str
    customer_id: str
    account_type: AccountType = AccountType.CURRENT
    currency: str = "USD"           # ISO 4217
    balance: float = 0.0
    country: str                    # ISO country code (bank jurisdiction)
    bank_name: str
    bank_swift: str
    status: str = "ACTIVE"          # ACTIVE / FROZEN / CLOSED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: Optional[datetime] = None
    average_monthly_volume: float = 0.0
    typical_transaction_size: float = 0.0


class Transaction(BaseModel):
    id: str = Field(default_factory=new_id)
    reference: str = Field(default_factory=lambda: f"TXN{uuid.uuid4().hex[:8].upper()}")
    sender_account_id: str
    receiver_account_id: Optional[str] = None   # None for external wire
    beneficiary_id: Optional[str] = None
    merchant_id: Optional[str] = None
    amount: float
    currency: str = "USD"
    exchange_rate: float = 1.0
    amount_usd: float = 0.0         # Normalised to USD
    transaction_type: TransactionType
    channel: TransactionChannel
    status: TransactionStatus = TransactionStatus.PENDING
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    description: Optional[str] = None
    device_id: Optional[str] = None
    ip_id: Optional[str] = None
    # Fraud labels (set during data generation / investigation)
    is_fraud: bool = False
    fraud_type: Optional[str] = None  # STRUCTURING / LAYERING / SMURFING / ROUND_TRIP


class Device(BaseModel):
    id: str = Field(default_factory=new_id)
    fingerprint: str
    device_type: str = "MOBILE"     # MOBILE / DESKTOP / ATM / POS
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None


class IPAddress(BaseModel):
    ip: str
    country: str
    city: Optional[str] = None
    is_vpn: bool = False
    is_tor: bool = False
    is_proxy: bool = False
    asn: Optional[str] = None
    isp: Optional[str] = None


class Merchant(BaseModel):
    id: str = Field(default_factory=new_id)
    name: str
    mcc_code: str                   # Merchant Category Code (ISO 18245)
    category: str
    country: str
    risk_level: str = "LOW"


class BeneficiaryAccount(BaseModel):
    id: str = Field(default_factory=new_id)
    account_number: str
    account_name: str
    bank_name: str
    bank_swift: str
    country: str
    currency: str = "USD"


# ──────────────────────────────────────────────
# API Request / Response Models
# ──────────────────────────────────────────────

class TransactionEvaluationRequest(BaseModel):
    transaction_id: Optional[str] = None   # Evaluate existing transaction in DB
    # -- OR -- provide full transaction details inline:
    sender_account_id: Optional[str] = None
    receiver_account_id: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = "USD"
    transaction_type: Optional[TransactionType] = None
    channel: Optional[TransactionChannel] = None
    device_id: Optional[str] = None
    ip_id: Optional[str] = None
    beneficiary_country: Optional[str] = None


class RiskScore(BaseModel):
    score: int                      # 0-999
    bayesian_score: int
    ml_score: int
    outcome: TransactionOutcome
    risk_factors: list[str]
    confidence: float               # 0.0 - 1.0
    explanation: str


class ChallengeQuestion(BaseModel):
    question: str
    question_id: str
    transaction_id: str


class TransactionEvaluationResponse(BaseModel):
    transaction_id: str
    risk_score: RiskScore
    challenge_question: Optional[ChallengeQuestion] = None
    processing_time_ms: float
