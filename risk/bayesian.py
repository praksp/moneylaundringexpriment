"""
Bayesian Risk Engine
=====================
Computes a fraud risk probability using Bayesian inference (log-odds / Naive Bayes).

Method:
  1. Start with prior log-odds derived from the base fraud rate P(fraud) = 0.02
  2. For each risk factor present in the FeatureVector, add log(LR)
     where LR = P(factor | fraud) / P(factor | legitimate)
  3. Convert posterior log-odds → probability via sigmoid
  4. Scale probability to 0–999 integer score

Risk factor likelihood ratios (LR) are calibrated against standard
AML typology research and FATF risk indicators.
"""

import math
from dataclasses import dataclass, field
from risk.features import FeatureVector
from config.settings import settings


# ──────────────────────────────────────────────────────────────────
# Likelihood Ratios: P(indicator | fraud) / P(indicator | legitimate)
# LR > 1  →  raises risk
# LR < 1  →  lowers risk (protective factor)
# ──────────────────────────────────────────────────────────────────

LIKELIHOOD_RATIOS: dict[str, float] = {
    # Amount indicators
    "in_structuring_band": 18.0,     # Classic CTR avoidance
    "is_very_high_amount": 12.0,     # > $50k single transfer
    "is_high_amount": 5.0,           # > $10k
    "is_round_amount": 3.5,          # Round figures = less natural

    # Transaction type
    "is_wire": 3.0,                  # Wires harder to reverse
    "is_crypto": 4.5,                # Pseudonymous, irreversible
    "is_cash": 3.0,                  # Cash = no paper trail
    "is_cross_border": 4.0,          # International = more risk

    # Time
    "is_night_transaction": 2.0,     # After-hours activity
    "is_weekend": 1.3,

    # Account age / activity
    "is_new_sender_account": 4.5,    # Mule accounts are often new
    "is_dormant_sender": 6.0,        # Sudden activity after long dormancy

    # Customer risk
    "sender_is_pep": 5.5,            # Politically Exposed Person
    "sender_is_sanctioned": 50.0,    # Sanctions hit = near-certain decline
    "receiver_is_pep": 4.0,
    "receiver_is_sanctioned": 40.0,

    # Geographic risk (evaluated as per country_risk_score levels)
    "sender_to_high_risk": 9.0,
    "sender_to_tax_haven": 3.0,

    # IP / device
    "ip_is_tor": 20.0,               # Tor = strong anonymity indicator
    "ip_is_vpn": 8.0,                # VPN = moderate anonymity indicator
    "ip_country_mismatch": 4.0,

    # Velocity
    "is_high_velocity_1h": 12.0,     # Burst pattern within 1 hour
    "is_high_velocity_24h": 8.0,

    # Network patterns
    "is_structuring": 20.0,
    "is_round_trip": 15.0,
    "device_shared": 7.0,            # Device used by multiple customers
    "is_deep_layering": 14.0,

    # Merchant
    "merchant_is_gambling": 4.0,
    "merchant_is_fx": 3.5,
    "merchant_is_high_risk": 3.0,

    # Amount deviation from typical
    "high_amount_deviation": 6.0,    # Amount > 5× typical for that account
}

# Country risk score level multipliers (applied to base amount-independent score)
COUNTRY_RISK_LR = {0: 1.0, 1: 2.5, 2: 5.0, 3: 25.0}

# Protective / legitimacy factors that DECREASE risk
PROTECTIVE_LR: dict[str, float] = {
    "enhanced_kyc_sender": 0.5,      # Enhanced KYC lowers prior
    "old_account": 0.7,              # Account age > 3 years
    "low_risk_customer": 0.8,
}


@dataclass
class BayesianRiskResult:
    score: int                       # 0–999
    probability: float               # Posterior P(fraud)
    log_odds: float
    triggered_factors: list[str] = field(default_factory=list)
    factor_contributions: dict[str, float] = field(default_factory=dict)


def _sigmoid(x: float) -> float:
    """Stable sigmoid / logistic function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _prob_to_score(prob: float) -> int:
    """Map fraud probability [0, 1] → integer score [0, 999]."""
    return max(0, min(999, round(prob * 999)))


def compute_bayesian_score(fv: FeatureVector) -> BayesianRiskResult:
    """
    Run the full Bayesian risk computation on a FeatureVector.
    Returns a BayesianRiskResult with the 0–999 risk score.
    """
    base_fraud_rate = settings.base_fraud_rate
    prior_odds = base_fraud_rate / (1.0 - base_fraud_rate)
    log_prior = math.log(prior_odds)

    log_posterior = log_prior
    triggered: list[str] = []
    contributions: dict[str, float] = {}

    def _apply_lr(factor_name: str, lr: float) -> None:
        nonlocal log_posterior
        delta = math.log(lr)
        log_posterior += delta
        contributions[factor_name] = round(delta, 4)
        if lr > 1.0:
            triggered.append(factor_name)

    # ── Amount factors ──────────────────────────────────────────
    if fv.in_structuring_band:
        _apply_lr("in_structuring_band", LIKELIHOOD_RATIOS["in_structuring_band"])
    elif fv.is_very_high_amount:
        _apply_lr("is_very_high_amount", LIKELIHOOD_RATIOS["is_very_high_amount"])
    elif fv.is_high_amount:
        _apply_lr("is_high_amount", LIKELIHOOD_RATIOS["is_high_amount"])

    if fv.is_round_amount:
        _apply_lr("is_round_amount", LIKELIHOOD_RATIOS["is_round_amount"])

    if fv.sender_amount_deviation > 5.0:
        _apply_lr("high_amount_deviation", LIKELIHOOD_RATIOS["high_amount_deviation"])
    elif fv.sender_amount_deviation > 2.0:
        _apply_lr("high_amount_deviation", 2.5)

    # ── Transaction type ────────────────────────────────────────
    if fv.is_wire:
        _apply_lr("is_wire", LIKELIHOOD_RATIOS["is_wire"])
    if fv.is_crypto:
        _apply_lr("is_crypto", LIKELIHOOD_RATIOS["is_crypto"])
    if fv.is_cash:
        _apply_lr("is_cash", LIKELIHOOD_RATIOS["is_cash"])
    if fv.is_cross_border:
        _apply_lr("is_cross_border", LIKELIHOOD_RATIOS["is_cross_border"])

    # ── Temporal signals ────────────────────────────────────────
    if fv.is_night_transaction:
        _apply_lr("is_night_transaction", LIKELIHOOD_RATIOS["is_night_transaction"])
    if fv.is_weekend:
        _apply_lr("is_weekend", LIKELIHOOD_RATIOS["is_weekend"])

    # ── Account health ──────────────────────────────────────────
    if fv.is_new_sender_account:
        _apply_lr("is_new_sender_account", LIKELIHOOD_RATIOS["is_new_sender_account"])
    elif fv.sender_account_age_days > 1095:   # > 3 years = protective
        _apply_lr("old_account", PROTECTIVE_LR["old_account"])

    if fv.is_dormant_sender:
        _apply_lr("is_dormant_sender", LIKELIHOOD_RATIOS["is_dormant_sender"])

    # ── Customer risk ───────────────────────────────────────────
    if fv.sender_is_sanctioned:
        _apply_lr("sender_is_sanctioned", LIKELIHOOD_RATIOS["sender_is_sanctioned"])
    elif fv.sender_is_pep:
        _apply_lr("sender_is_pep", LIKELIHOOD_RATIOS["sender_is_pep"])

    if fv.receiver_is_sanctioned:
        _apply_lr("receiver_is_sanctioned", LIKELIHOOD_RATIOS["receiver_is_sanctioned"])
    elif fv.receiver_is_pep:
        _apply_lr("receiver_is_pep", LIKELIHOOD_RATIOS["receiver_is_pep"])

    # Risk tier multiplier
    tier_lr = {0: 1.0, 1: 1.5, 2: 3.0, 3: 6.0}.get(fv.sender_risk_tier_score, 1.0)
    if tier_lr > 1.0:
        _apply_lr("sender_risk_tier", tier_lr)

    # ── Geographic risk ─────────────────────────────────────────
    receiver_country_lr = COUNTRY_RISK_LR.get(fv.receiver_country_risk, 1.0)
    if receiver_country_lr > 1.0:
        _apply_lr("receiver_country_risk", receiver_country_lr)

    beneficiary_country_lr = COUNTRY_RISK_LR.get(fv.beneficiary_country_risk, 1.0)
    if beneficiary_country_lr > 1.0:
        _apply_lr("beneficiary_country_risk", beneficiary_country_lr)

    if fv.sender_to_high_risk:
        _apply_lr("sender_to_high_risk", LIKELIHOOD_RATIOS["sender_to_high_risk"])
    if fv.sender_to_tax_haven:
        _apply_lr("sender_to_tax_haven", LIKELIHOOD_RATIOS["sender_to_tax_haven"])

    # ── IP / device ─────────────────────────────────────────────
    if fv.ip_is_tor:
        _apply_lr("ip_is_tor", LIKELIHOOD_RATIOS["ip_is_tor"])
    elif fv.ip_is_vpn:
        _apply_lr("ip_is_vpn", LIKELIHOOD_RATIOS["ip_is_vpn"])
    if fv.ip_country_mismatch:
        _apply_lr("ip_country_mismatch", LIKELIHOOD_RATIOS["ip_country_mismatch"])

    # ── Velocity ────────────────────────────────────────────────
    if fv.is_high_velocity_1h:
        _apply_lr("is_high_velocity_1h", LIKELIHOOD_RATIOS["is_high_velocity_1h"])
    elif fv.is_high_velocity_24h:
        _apply_lr("is_high_velocity_24h", LIKELIHOOD_RATIOS["is_high_velocity_24h"])

    # ── Network / typology patterns ─────────────────────────────
    if fv.is_structuring:
        _apply_lr("is_structuring", LIKELIHOOD_RATIOS["is_structuring"])
    if fv.is_round_trip:
        _apply_lr("is_round_trip", LIKELIHOOD_RATIOS["is_round_trip"])
    if fv.device_shared:
        _apply_lr("device_shared", LIKELIHOOD_RATIOS["device_shared"])
    if fv.is_deep_layering:
        _apply_lr("is_deep_layering", LIKELIHOOD_RATIOS["is_deep_layering"])

    # ── Merchant ────────────────────────────────────────────────
    if fv.merchant_is_gambling:
        _apply_lr("merchant_is_gambling", LIKELIHOOD_RATIOS["merchant_is_gambling"])
    if fv.merchant_is_fx:
        _apply_lr("merchant_is_fx", LIKELIHOOD_RATIOS["merchant_is_fx"])
    if fv.merchant_is_high_risk:
        _apply_lr("merchant_is_high_risk", LIKELIHOOD_RATIOS["merchant_is_high_risk"])

    posterior_prob = _sigmoid(log_posterior)
    score = _prob_to_score(posterior_prob)

    return BayesianRiskResult(
        score=score,
        probability=round(posterior_prob, 6),
        log_odds=round(log_posterior, 4),
        triggered_factors=triggered,
        factor_contributions=contributions,
    )
