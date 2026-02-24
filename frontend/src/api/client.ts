import axios from 'axios'

export const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
})

// ── Types ──────────────────────────────────────────────────────────────

export interface ModelScore {
  score: number           // 0-999
  probability: number     // 0.0-1.0
  label: string           // "Bayesian Engine", "XGBoost", "SVM (RBF)", "KNN (k=7)"
  short: string           // "bayesian", "xgb", "svm", "knn"
  is_trained: boolean
  weight_pct: number      // % weight in the ensemble
}

export interface RiskScore {
  score: number
  bayesian_score: number
  ml_score: number        // XGBoost score
  svm_score: number
  knn_score: number
  model_scores: ModelScore[]
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
  risk_factors: string[]
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

// ── Auth types ──────────────────────────────────────────────────────────

export interface AuthUser {
  id: string
  username: string
  role: 'admin' | 'viewer'
  full_name?: string
  is_active: boolean
}

export interface AggregateWorldMapEntry {
  code: string
  name: string
  fatf_risk: string
  txn_count: number
  fraud_count: number
  total_amount: number
  avg_score: number | null
  max_score: number | null
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  directions: string[]
  fraud_types: string[]
  txn_types: string[]
}

// ── API calls ───────────────────────────────────────────────────────────

export const getAggregateWorldMap = () =>
  api.get<{ countries: AggregateWorldMapEntry[]; total_countries: number; summary: Record<string, number> }>(
    '/transactions/aggregate/world-map'
  ).then(r => r.data)

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

export const listCustomers = (skip = 0, limit = 25, riskTier?: string, search?: string) =>
  api.get('/profiles/', { params: { skip, limit, risk_tier: riskTier || undefined, search: search || undefined } }).then(r => r.data)

export const getHighRiskAccounts = () =>
  api.get('/profiles/high-risk-accounts').then(r => r.data)

export interface PagedTransactions {
  total: number
  page: number
  page_size: number
  total_pages: number
  transactions: TransactionSummary[]
}

export const getCustomerTransactions = (
  customerId: string,
  page = 1,
  pageSize = 500,
) =>
  api
    .get<PagedTransactions>(`/profiles/${customerId}/transactions`, {
      params: { page, page_size: pageSize },
    })
    .then(r => r.data)

export const computeFeatureSnapshot = (customerId: string, accountId: string) =>
  api.post(`/profiles/${customerId}/accounts/${accountId}/feature-snapshot`).then(r => r.data)

export interface RiskFactorDetail {
  factor: string
  title: string
  detail: string
  severity: 'critical' | 'high' | 'medium' | 'low'
}

export interface ScoreExplanation {
  summary: string
  score: number
  risk_level: string
  bayesian_contribution: string
  ml_contribution: string
  factor_count: number
  confidence_pct: number
  model_agreement: string
}

export interface TransactionDetail {
  transaction: Record<string, unknown>
  sender_account: Record<string, unknown> | null
  receiver_account: Record<string, unknown> | null
  sender_customer: Record<string, unknown> | null
  receiver_customer: Record<string, unknown> | null
  device: Record<string, unknown> | null
  ip_address: Record<string, unknown> | null
  merchant: Record<string, unknown> | null
  beneficiary: Record<string, unknown> | null
  sender_country: Record<string, unknown> | null
  receiver_country: Record<string, unknown> | null
  prediction: Record<string, unknown> | null
  risk_factor_details: RiskFactorDetail[]
  score_explanation: ScoreExplanation
}

export const getTransactionDetail = (customerId: string, txnId: string) =>
  api.get<TransactionDetail>(`/profiles/${customerId}/transactions/${txnId}`).then(r => r.data)

export interface CountryMapEntry {
  code: string
  name: string
  fatf_risk: string
  txn_count: number
  fraud_count: number
  total_amount: number
  avg_score: number | null
  max_score: number | null
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  directions: string[]
  fraud_types: string[]
}

export const getTransactionMap = (customerId: string) =>
  api.get<{ countries: CountryMapEntry[]; total_countries: number }>(
    `/profiles/${customerId}/transaction-map`
  ).then(r => r.data)

export const verifyChallenge = (payload: {
  transaction_id: string
  question_id: string
  answer: string
}) =>
  api.post('/submit/verify-challenge', payload).then(r => r.data)

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

// ── Anomaly Detection ─────────────────────────────────────────────────────────

export interface AnomalyAccountResult {
  account_id: string
  anomaly_score: number
  knn_distance_score: number
  rule_score: number
  is_mule_suspect: boolean
  indicators: string[]
  pass_through_ratio: number
  unique_senders_30d: number
  structuring_30d: number
  in_volume: number
  out_volume: number
  customer_id: string | null
  customer_name: string | null
  customer_country: string | null
  account_number: string | null
  bank_name: string | null
  account_type: string | null
  scored_at: string | null
}

export interface SuspectListResponse {
  total: number
  page: number
  page_size: number
  total_pages: number
  suspects: AnomalyAccountResult[]
  detector_trained: boolean
}

export interface AnomalySummary {
  total_accounts: number
  scored_accounts: number
  mule_suspects: number
  high_risk_accounts: number
  suspect_rate_pct: number
  indicator_distribution: { indicator: string; freq: number }[]
  detector_trained: boolean
  coverage_pct: number
}

export const getAnomalySummary = () =>
  api.get<AnomalySummary>('/anomaly/summary').then(r => r.data)

export const listMuleSuspects = (page = 1, pageSize = 50) =>
  api.get<SuspectListResponse>('/anomaly/suspects', {
    params: { page, page_size: pageSize },
  }).then(r => r.data)

export const getAccountAnomaly = (accountId: string) =>
  api.get<AnomalyAccountResult>(`/anomaly/accounts/${accountId}`).then(r => r.data)

export const getCustomerAnomaly = (customerId: string) =>
  api.get(`/anomaly/customers/${customerId}`).then(r => r.data)

export const trainAnomalyDetector = () =>
  api.post('/anomaly/train').then(r => r.data)

export const scanAccounts = (force = false) =>
  api.post('/anomaly/scan', null, { params: { force } }).then(r => r.data)
