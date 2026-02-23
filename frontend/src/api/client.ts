import axios from 'axios'

export const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
})

// ── Types ──────────────────────────────────────────────────────────────

export interface RiskScore {
  score: number
  bayesian_score: number
  ml_score: number
  outcome: 'ALLOW' | 'CHALLENGE' | 'DECLINE'
  risk_factors: string[]
  confidence: number
  explanation: string
}

export interface ChallengeQuestion {
  question: string
  question_id: string
  transaction_id: string
}

export interface EvaluationResponse {
  transaction_id: string
  risk_score: RiskScore
  challenge_question: ChallengeQuestion | null
  processing_time_ms: number
}

export interface Account {
  id: string
  account_number: string
  type: string
  currency: string
  balance: number
  country: string
  customer_name: string
  customer_id: string
}

export interface Transaction {
  id: string
  reference: string
  amount: number
  currency: string
  transaction_type: string
  channel: string
  timestamp: string
  is_fraud: boolean
  fraud_type: string | null
  sender_id: string
  receiver_id: string | null
  risk_score?: number
  outcome?: string
}

export interface CustomerProfile {
  customer_id: string
  name: string
  customer_type: string
  nationality: string
  country_of_residence: string
  kyc_level: string
  pep_flag: boolean
  sanctions_flag: boolean
  risk_tier: string
  created_at: string
  accounts: AccountSummary[]
  recent_transactions: TransactionSummary[]
  risk_profile: RiskProfile
  network_connections: NetworkConnection[]
  velocity: Record<string, number>
  mule_indicators: MuleIndicators
  generated_at: string
}

export interface AccountSummary {
  id: string
  account_number: string
  account_type: string
  currency: string
  balance: number
  country: string
  bank_name: string
  status: string
  created_at: string
  last_active: string
  is_dormant: boolean
}

export interface TransactionSummary {
  id: string
  reference: string
  amount: number
  currency: string
  transaction_type: string
  channel: string
  timestamp: string
  is_fraud: boolean
  fraud_type: string | null
  sender_account_id: string
  receiver_account_id: string | null
  receiver_name: string | null
  risk_score: number | null
  outcome: string | null
}

export interface RiskProfile {
  total_evaluations: number
  avg_score_30d: number
  max_score_30d: number
  allow_count: number
  challenge_count: number
  decline_count: number
  fraud_incident_count: number
  current_risk_tier: string
  risk_trend: string
}

export interface NetworkConnection {
  customer_id: string
  customer_name: string
  risk_tier: string
  is_pep: boolean
  connection_type: string
  shared_devices: string[]
}

export interface MuleIndicators {
  mule_score: number
  is_likely_mule: boolean
  is_pass_through: boolean
  turnover_ratio: number
  unique_senders_30d: number
  high_sender_count: boolean
  avg_hold_time_hours: number
  rapid_disbursement: boolean
  structuring_incidents_30d: number
  structuring_risk: boolean
  inbound_volume_30d: number
  outbound_volume_30d: number
}

export interface MonitoringSummary {
  total_evaluations: number
  evaluations_24h: number
  outcome_7d: { ALLOW: number; CHALLENGE: number; DECLINE: number; total: number; decline_rate_pct: number }
  avg_score_7d: number
  avg_latency_ms: number
  latest_drift_alert: 'OK' | 'WARNING' | 'CRITICAL' | 'UNKNOWN'
  latest_drift_at: string | null
}

export interface OutcomeTrendPoint {
  date: string
  ALLOW: number
  CHALLENGE: number
  DECLINE: number
  total: number
}

export interface ScoreDistPoint {
  period: string
  count: number
  avg_score: number
  p50_score: number
  p95_score: number
  allow_count: number
  challenge_count: number
  decline_count: number
  avg_latency_ms: number
}

export interface RiskFactor {
  factor: string
  frequency: number
  avg_score_when_triggered: number
}

// ── API calls ───────────────────────────────────────────────────────────

export const submitTransaction = (data: object) =>
  api.post<EvaluationResponse>('/submit/transaction', data).then(r => r.data)

export const getAccounts = () =>
  api.get<{ accounts: Account[] }>('/submit/accounts').then(r => r.data.accounts)

export const listTransactions = (skip = 0, limit = 20, fraudOnly = false) =>
  api.get<{ total: number; transactions: Transaction[] }>('/transactions/', {
    params: { skip, limit, fraud_only: fraudOnly },
  }).then(r => r.data)

export const getTransactionStats = () =>
  api.get('/transactions/stats/summary').then(r => r.data)

export const evaluateTransaction = (txnId: string) =>
  api.post<EvaluationResponse>(`/evaluate/${txnId}`).then(r => r.data)

export const getCustomerProfile = (customerId: string) =>
  api.get<CustomerProfile>(`/profiles/${customerId}`).then(r => r.data)

export const listCustomers = (skip = 0, limit = 20, riskTier?: string) =>
  api.get('/profiles/', { params: { skip, limit, risk_tier: riskTier } }).then(r => r.data)

export const getHighRiskAccounts = () =>
  api.get('/profiles/high-risk-accounts').then(r => r.data)

export const computeFeatureSnapshot = (customerId: string, accountId: string) =>
  api.post(`/profiles/${customerId}/accounts/${accountId}/feature-snapshot`).then(r => r.data)

export const getMonitoringSummary = () =>
  api.get<MonitoringSummary>('/monitoring/summary').then(r => r.data)

export const getOutcomeTrend = (daysBack = 30) =>
  api.get<{ data: OutcomeTrendPoint[] }>('/monitoring/outcome-trend', {
    params: { days_back: daysBack },
  }).then(r => r.data.data)

export const getScoreDistribution = (daysBack = 7) =>
  api.get<{ data: ScoreDistPoint[] }>('/monitoring/score-distribution', {
    params: { days_back: daysBack },
  }).then(r => r.data.data)

export const getTopRiskFactors = (daysBack = 7) =>
  api.get<{ data: RiskFactor[] }>('/monitoring/risk-factors', {
    params: { days_back: daysBack },
  }).then(r => r.data.data)

export const computeDrift = () =>
  api.post('/monitoring/drift/compute').then(r => r.data)

export const getDriftHistory = () =>
  api.get('/monitoring/drift/history').then(r => r.data)

export const labelPrediction = (transactionId: string, isFraud: boolean) =>
  api.post(`/monitoring/predictions/${transactionId}/label`, { is_fraud: isFraud }).then(r => r.data)
