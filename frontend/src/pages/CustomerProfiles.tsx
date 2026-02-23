import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import {
  Search, User, AlertTriangle, TrendingUp, TrendingDown, Minus,
  Shield, ChevronRight, X, ArrowRight, Monitor, Wifi, MapPin,
  Clock, Hash, Building2, CreditCard, ChevronDown, ChevronUp,
  ExternalLink, Info,
} from 'lucide-react'
import {
  listCustomers, getCustomerProfile, computeFeatureSnapshot,
  getTransactionDetail,
  type CustomerProfile, type TransactionSummary, type RiskFactorDetail,
} from '../api/client'
import TransactionWorldMap from '../components/TransactionWorldMap'
import clsx from 'clsx'

// ── Constants ─────────────────────────────────────────────────────────────────

const TIER_COLORS: Record<string, string> = {
  LOW: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  MEDIUM: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
  HIGH: 'text-orange-400 bg-orange-500/10 border-orange-500/20',
  CRITICAL: 'text-red-400 bg-red-500/10 border-red-500/20',
}

const OUTCOME_COLORS = {
  ALLOW: { bg: 'bg-emerald-500/10 border-emerald-500/30', text: 'text-emerald-400', dot: 'bg-emerald-400' },
  CHALLENGE: { bg: 'bg-amber-500/10 border-amber-500/30', text: 'text-amber-400', dot: 'bg-amber-400' },
  DECLINE: { bg: 'bg-red-500/10 border-red-500/30', text: 'text-red-400', dot: 'bg-red-400' },
}

const SEVERITY_STYLES: Record<string, string> = {
  critical: 'bg-red-500/10 border-red-500/30 text-red-300',
  high: 'bg-orange-500/10 border-orange-500/30 text-orange-300',
  medium: 'bg-amber-500/10 border-amber-500/30 text-amber-300',
  low: 'bg-blue-500/10 border-blue-500/30 text-blue-300',
}

const SEVERITY_BADGE: Record<string, string> = {
  critical: 'bg-red-500/20 text-red-300',
  high: 'bg-orange-500/20 text-orange-300',
  medium: 'bg-amber-500/20 text-amber-300',
  low: 'bg-blue-500/20 text-blue-300',
}

// ── Sub-components ────────────────────────────────────────────────────────────

const TrendIcon = ({ trend }: { trend: string }) =>
  trend === 'RISING' ? <TrendingUp size={13} className="text-red-400" />
  : trend === 'FALLING' ? <TrendingDown size={13} className="text-emerald-400" />
  : <Minus size={13} className="text-slate-400" />

function ScoreGauge({ score, size = 'sm' }: { score: number; size?: 'sm' | 'lg' }) {
  const r = size === 'lg' ? 54 : 36
  const sw = size === 'lg' ? 10 : 7
  const viewBox = size === 'lg' ? 120 : 80
  const cx = viewBox / 2
  const circumference = 2 * Math.PI * r
  const pct = Math.min(score / 999, 1)
  const dash = pct * circumference
  const color = score <= 399 ? '#10b981' : score <= 699 ? '#f59e0b' : '#ef4444'
  const dim = size === 'lg' ? 'w-28 h-28' : 'w-20 h-20'

  return (
    <div className={clsx('relative mx-auto', dim)}>
      <svg viewBox={`0 0 ${viewBox} ${viewBox}`} className="w-full h-full -rotate-90">
        <circle cx={cx} cy={cx} r={r} fill="none" stroke="#1e293b" strokeWidth={sw} />
        <circle cx={cx} cy={cx} r={r} fill="none" stroke={color} strokeWidth={sw}
          strokeDasharray={`${dash} ${circumference}`} strokeLinecap="round"
          style={{ transition: 'stroke-dasharray 0.6s ease' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={clsx('font-bold text-white', size === 'lg' ? 'text-2xl' : 'text-base')}>{score}</span>
        <span className="text-xs text-slate-500">/999</span>
      </div>
    </div>
  )
}

// ── Transaction detail drawer ─────────────────────────────────────────────────

function RiskFactorCard({ factor, expanded, onToggle }: {
  factor: RiskFactorDetail; expanded: boolean; onToggle: () => void
}) {
  return (
    <button
      onClick={onToggle}
      className={clsx(
        'w-full text-left rounded-lg border p-3 transition-colors',
        SEVERITY_STYLES[factor.severity],
        'hover:opacity-90',
      )}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={clsx('text-xs font-bold px-1.5 py-0.5 rounded uppercase tracking-wide', SEVERITY_BADGE[factor.severity])}>
            {factor.severity}
          </span>
          <span className="text-sm font-medium">{factor.title}</span>
        </div>
        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </div>
      {expanded && (
        <p className="mt-2 text-xs leading-relaxed opacity-80 text-left">{factor.detail}</p>
      )}
    </button>
  )
}

function TransactionDrawer({ customerId, txnId, txnSummary, onClose }: {
  customerId: string
  txnId: string
  txnSummary: TransactionSummary
  onClose: () => void
}) {
  const [expandedFactor, setExpandedFactor] = useState<string | null>(null)

  const { data, isLoading, isError } = useQuery({
    queryKey: ['txn-detail', txnId],
    queryFn: () => getTransactionDetail(customerId, txnId),
  })

  const pred = data?.prediction as Record<string, unknown> | null
  const score = pred ? Number(pred.final_score) : null
  const outcome = pred ? String(pred.outcome) : txnSummary.outcome
  const outcomeStyle = outcome ? OUTCOME_COLORS[outcome as keyof typeof OUTCOME_COLORS] : OUTCOME_COLORS.ALLOW
  const factors = data?.risk_factor_details || []
  const explanation = data?.score_explanation

  return (
    <div className="fixed inset-0 z-50 flex">
      {/* Backdrop */}
      <div className="flex-1 bg-black/60" onClick={onClose} />

      {/* Drawer */}
      <div className="w-full max-w-2xl bg-slate-950 border-l border-slate-800 overflow-y-auto flex flex-col shadow-2xl">
        {/* Header */}
        <div className="sticky top-0 z-10 bg-slate-950 border-b border-slate-800 px-6 py-4 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <Hash size={14} className="text-slate-500" />
              <span className="font-mono text-sm font-semibold text-white">{txnSummary.reference}</span>
              {txnSummary.is_fraud && (
                <span className="text-xs px-2 py-0.5 rounded bg-red-500/20 text-red-300 border border-red-500/30">
                  FRAUD PATTERN
                </span>
              )}
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              {txnSummary.transaction_type} · {txnSummary.channel} ·{' '}
              {new Date(txnSummary.timestamp).toLocaleString()}
            </p>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-slate-800 rounded-lg transition-colors">
            <X size={16} className="text-slate-400" />
          </button>
        </div>

        <div className="flex-1 p-6 space-y-6">
          {isLoading && (
            <div className="text-center py-16 text-slate-500">Loading transaction details…</div>
          )}

          {isError && (
            <div className="text-center py-16 text-red-400">
              Could not load transaction details. This transaction may not have been evaluated yet.
            </div>
          )}

          {data && (
            <>
              {/* Amount + parties */}
              <div className="bg-slate-900 rounded-xl border border-slate-800 p-5">
                <div className="text-center mb-5">
                  <p className="text-3xl font-bold text-white">
                    {txnSummary.currency}{' '}
                    {txnSummary.amount.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                  </p>
                  {txnSummary.is_fraud && txnSummary.fraud_type && (
                    <p className="text-xs text-red-400 mt-1 font-medium">
                      Fraud type: {txnSummary.fraud_type.replace(/_/g, ' ')}
                    </p>
                  )}
                </div>

                {/* Flow diagram */}
                <div className="flex items-center gap-3 text-sm">
                  <div className="flex-1 bg-slate-800 rounded-lg p-3">
                    <p className="text-xs text-slate-500 mb-1 flex items-center gap-1">
                      <Building2 size={11} /> Sender
                    </p>
                    <p className="font-medium text-white text-xs truncate">
                      {data.sender_customer?.name as string || 'Unknown'}
                    </p>
                    <p className="text-slate-500 text-xs font-mono">
                      {data.sender_account?.account_number as string || '—'}
                    </p>
                    {data.sender_country && (
                      <p className="text-xs text-slate-500 flex items-center gap-1 mt-1">
                        <MapPin size={10} />
                        {data.sender_country.name as string} ({data.sender_country.code as string})
                        {data.sender_country.fatf_risk !== 'LOW' && (
                          <span className="text-amber-400 font-medium ml-1">
                            {data.sender_country.fatf_risk as string}
                          </span>
                        )}
                      </p>
                    )}
                  </div>

                  <ArrowRight size={20} className="text-slate-500 flex-shrink-0" />

                  <div className="flex-1 bg-slate-800 rounded-lg p-3">
                    <p className="text-xs text-slate-500 mb-1 flex items-center gap-1">
                      <Building2 size={11} /> Receiver
                    </p>
                    {data.receiver_account ? (
                      <>
                        <p className="font-medium text-white text-xs truncate">
                          {data.receiver_customer?.name as string || 'Internal Account'}
                        </p>
                        <p className="text-slate-500 text-xs font-mono">
                          {data.receiver_account.account_number as string}
                        </p>
                        {data.receiver_country && (
                          <p className="text-xs text-slate-500 flex items-center gap-1 mt-1">
                            <MapPin size={10} />
                            {data.receiver_country.name as string}
                            {(data.receiver_country.fatf_risk as string) !== 'LOW' && (
                              <span className="text-red-400 font-medium ml-1">
                                {data.receiver_country.fatf_risk as string}
                              </span>
                            )}
                          </p>
                        )}
                      </>
                    ) : data.beneficiary ? (
                      <>
                        <p className="font-medium text-white text-xs truncate">
                          {data.beneficiary.account_name as string}
                        </p>
                        <p className="text-slate-500 text-xs">{data.beneficiary.bank_name as string}</p>
                        <p className="text-xs text-red-400 flex items-center gap-1 mt-1">
                          <MapPin size={10} /> External: {data.beneficiary.country as string}
                        </p>
                      </>
                    ) : (
                      <p className="text-slate-500 text-xs">No receiver</p>
                    )}
                  </div>
                </div>

                {/* Device / IP */}
                {(data.device || data.ip_address) && (
                  <div className="mt-3 flex gap-2 flex-wrap">
                    {data.device && (
                      <div className="flex items-center gap-1.5 text-xs bg-slate-700/50 px-2.5 py-1.5 rounded-lg">
                        <Monitor size={11} className="text-slate-400" />
                        <span className="text-slate-300">{data.device.device_type as string}</span>
                      </div>
                    )}
                    {data.ip_address && (
                      <div className={clsx(
                        'flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg',
                        data.ip_address.is_tor ? 'bg-red-500/15 text-red-300' :
                        data.ip_address.is_vpn ? 'bg-amber-500/15 text-amber-300' :
                        'bg-slate-700/50 text-slate-300'
                      )}>
                        <Wifi size={11} />
                        <span>{data.ip_address.ip as string} ({data.ip_address.country as string})</span>
                        {!!data.ip_address.is_tor && <span className="font-bold">TOR</span>}
                        {!!data.ip_address.is_vpn && !data.ip_address.is_tor && <span className="font-bold">VPN</span>}
                      </div>
                    )}
                    {data.merchant && (
                      <div className="flex items-center gap-1.5 text-xs bg-slate-700/50 px-2.5 py-1.5 rounded-lg">
                        <CreditCard size={11} className="text-slate-400" />
                        <span className="text-slate-300">{data.merchant.name as string} ({data.merchant.mcc_code as string})</span>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Test-data fraud explanation (no PredictionLog yet) */}
              {!pred && txnSummary.is_fraud && txnSummary.fraud_type && (
                <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <AlertTriangle size={16} className="text-red-400" />
                    <h3 className="text-sm font-semibold text-red-300">Fraud Pattern Detected (Test Data)</h3>
                  </div>
                  <p className="text-sm text-red-200/80 leading-relaxed">
                    This is a seeded test transaction flagged as{' '}
                    <span className="font-bold text-red-300">
                      {txnSummary.fraud_type.replace(/_/g, ' ')}
                    </span>
                    . It has not yet been passed through the live risk engine.
                    Risk scores shown below are derived from the known fraud pattern.
                  </p>
                  {txnSummary.risk_score !== null && (
                    <div className="mt-3 flex items-center gap-3">
                      <span className="text-2xl font-bold text-red-300">{txnSummary.risk_score}</span>
                      <span className="text-xs text-red-400">/999 · inferred score</span>
                      <span className="ml-auto text-xs font-bold bg-red-500/20 text-red-300 px-2.5 py-1 rounded">
                        {txnSummary.outcome}
                      </span>
                    </div>
                  )}
                  {txnSummary.risk_factors.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {txnSummary.risk_factors.map(f => (
                        <span key={f} className="text-xs px-2 py-0.5 rounded bg-red-500/15 text-red-300 border border-red-500/20">
                          {f.replace(/_/g, ' ')}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Risk Score */}
              {pred && score !== null ? (
                <div className={clsx('rounded-xl border p-5', outcomeStyle.bg)}>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-slate-200">Risk Score Breakdown</h3>
                    <span className={clsx(
                      'text-sm font-bold px-3 py-1 rounded-full border',
                      outcomeStyle.bg, outcomeStyle.text
                    )}>
                      {outcome}
                    </span>
                  </div>

                  <div className="flex items-center gap-6">
                    <ScoreGauge score={score} size="lg" />
                    <div className="flex-1 space-y-2.5">
                      {/* All model bars */}
                      {[
                        { label: 'Bayesian', key: 'bayesian_score', weight: '40%', colour: '#a78bfa' },
                        { label: 'XGBoost',  key: 'ml_score',       weight: '30%', colour: '#60a5fa' },
                        { label: 'SVM',      key: 'svm_score',      weight: '20%', colour: '#34d399' },
                        { label: 'KNN',      key: 'knn_score',      weight: '10%', colour: '#fbbf24' },
                      ].map(({ label, key, weight, colour }) => {
                        const s = Number((pred as Record<string, unknown>)[key] ?? 0)
                        return (
                          <div key={label}>
                            <div className="flex justify-between text-xs mb-1">
                              <span className="text-slate-400">{label}
                                <span className="text-slate-600 ml-1">({weight})</span>
                              </span>
                              <span className={clsx('font-bold',
                                s <= 399 ? 'text-emerald-400' : s <= 699 ? 'text-amber-400' : 'text-red-400'
                              )}>{s}</span>
                            </div>
                            <div className="h-2 bg-slate-900 rounded-full overflow-hidden">
                              <div className="h-full rounded-full transition-all"
                                style={{ width: `${(s / 999) * 100}%`, backgroundColor: colour }} />
                            </div>
                          </div>
                        )
                      })}
                      <div className="flex gap-4 text-xs text-slate-400 pt-1">
                        <span>Confidence: <span className="text-white font-medium">
                          {((pred.confidence as number) * 100).toFixed(0)}%
                        </span></span>
                        <span>Eval time: <span className="text-white font-medium">
                          {(pred.processing_time_ms as number)?.toFixed(1)}ms
                        </span></span>
                      </div>
                    </div>
                  </div>

                  {/* Score explanation */}
                  {explanation && (
                    <div className="mt-4 bg-slate-900/60 rounded-lg p-4">
                      <p className="text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wide flex items-center gap-1">
                        <Info size={11} /> Why this score?
                      </p>
                      <p className="text-sm text-slate-300 leading-relaxed">{explanation.summary}</p>
                      <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-slate-500">
                        <span>{explanation.bayesian_contribution}</span>
                        <span>{explanation.ml_contribution}</span>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="bg-slate-900 border border-dashed border-slate-700 rounded-xl p-8 text-center">
                  <Clock size={24} className="text-slate-600 mx-auto mb-2" />
                  <p className="text-slate-500 text-sm">No risk evaluation recorded for this transaction.</p>
                  <p className="text-slate-600 text-xs mt-1">
                    Transactions submitted via the Submit page are evaluated automatically.
                  </p>
                </div>
              )}

              {/* Risk factors */}
              {factors.length > 0 && (
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold text-slate-300">
                      Risk Factors Triggered ({factors.length})
                    </h3>
                    <p className="text-xs text-slate-500">Click to expand explanation</p>
                  </div>
                  <div className="space-y-2">
                    {factors.map(f => (
                      <RiskFactorCard
                        key={f.factor}
                        factor={f}
                        expanded={expandedFactor === f.factor}
                        onToggle={() => setExpandedFactor(prev => prev === f.factor ? null : f.factor)}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Transaction metadata */}
              <div className="bg-slate-900 rounded-xl border border-slate-800 p-5">
                <h3 className="text-sm font-semibold text-slate-300 mb-3">Transaction Details</h3>
                <div className="grid grid-cols-2 gap-x-6 gap-y-2.5 text-xs">
                  {[
                    ['ID', String(data.transaction.id).slice(0, 18) + '…'],
                    ['Reference', data.transaction.reference as string],
                    ['Type', data.transaction.transaction_type as string],
                    ['Channel', data.transaction.channel as string],
                    ['Currency', data.transaction.currency as string],
                    ['Exchange Rate', data.transaction.exchange_rate as string],
                    ['Status', data.transaction.status as string],
                    ['Timestamp', new Date(data.transaction.timestamp as string).toLocaleString()],
                    ['Description', (data.transaction.description as string) || '—'],
                    ['Fraud Label', data.transaction.is_fraud ? `YES — ${data.transaction.fraud_type}` : 'No'],
                  ].map(([label, value]) => (
                    <div key={label} className="py-1 border-b border-slate-800/50">
                      <p className="text-slate-500">{label}</p>
                      <p className={clsx('text-white font-medium mt-0.5',
                        label === 'Fraud Label' && data.transaction.is_fraud ? 'text-red-400' : ''
                      )}>{value}</p>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Transaction row ───────────────────────────────────────────────────────────

function TransactionRow({ txn, onClick }: { txn: TransactionSummary; onClick: () => void }) {
  const outcome = txn.outcome
  const outcomeColor = outcome
    ? { ALLOW: 'text-emerald-400', CHALLENGE: 'text-amber-400', DECLINE: 'text-red-400' }[outcome] || 'text-slate-400'
    : 'text-slate-500'
  const dotColor = txn.is_fraud
    ? 'bg-red-400'
    : outcome
      ? { ALLOW: 'bg-emerald-400', CHALLENGE: 'bg-amber-400', DECLINE: 'bg-red-400' }[outcome] || 'bg-slate-500'
      : 'bg-slate-600'

  return (
    <button
      onClick={onClick}
      className="w-full flex items-center justify-between text-xs bg-slate-800/40 hover:bg-slate-800 rounded-lg px-3 py-2.5 transition-colors group border border-transparent hover:border-slate-700"
    >
      <div className="flex items-center gap-2.5 min-w-0">
        <div className={clsx('w-2 h-2 rounded-full flex-shrink-0', dotColor)} />
        <div className="min-w-0 text-left">
          <p className="text-white font-mono font-medium">{txn.reference}</p>
          <p className="text-slate-500">
            {txn.transaction_type} · {txn.channel} · {txn.timestamp?.slice(0, 10)}
            {txn.is_fraud && txn.fraud_type && (
              <span className="ml-1.5 text-red-400 font-medium">
                · {txn.fraud_type.replace(/_/g, ' ')}
              </span>
            )}
          </p>
        </div>
      </div>
      <div className="flex items-center gap-3 flex-shrink-0 ml-2">
        <div className="text-right">
          <p className="text-white font-medium">
            {txn.currency} {txn.amount.toLocaleString(undefined, { minimumFractionDigits: 2 })}
          </p>
          <div className="flex items-center gap-1.5 justify-end mt-0.5">
            {outcome && <span className={clsx('font-medium', outcomeColor)}>{outcome}</span>}
            {txn.risk_score !== null && txn.risk_score !== undefined && (
              <span className={clsx(
                'font-bold px-1.5 py-0.5 rounded',
                txn.risk_score <= 399 ? 'bg-emerald-500/20 text-emerald-300'
                : txn.risk_score <= 699 ? 'bg-amber-500/20 text-amber-300'
                : 'bg-red-500/20 text-red-300'
              )}>{txn.risk_score}</span>
            )}
            {txn.is_fraud && (
              <span className="bg-red-500/20 text-red-300 px-1.5 py-0.5 rounded font-medium">FRAUD</span>
            )}
          </div>
        </div>
        <ExternalLink size={12} className="text-slate-600 group-hover:text-slate-400 transition-colors" />
      </div>
    </button>
  )
}

// ── Profile panel ─────────────────────────────────────────────────────────────

function ProfilePanel({ profile, onTransactionClick }: {
  profile: CustomerProfile
  onTransactionClick: (txn: TransactionSummary) => void
}) {
  const snapshotMutation = useMutation({
    mutationFn: ({ cid, aid }: { cid: string; aid: string }) =>
      computeFeatureSnapshot(cid, aid),
  })
  const tier = profile.risk_tier
  const mi = profile.mule_indicators

  return (
    <div className="space-y-5">
      {/* Identity */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-5">
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-11 h-11 rounded-full bg-slate-700 flex items-center justify-center text-lg font-bold text-white">
              {profile.name[0]?.toUpperCase()}
            </div>
            <div>
              <h2 className="font-semibold text-white">{profile.name}</h2>
              <p className="text-xs text-slate-500">{profile.customer_type} · {profile.country_of_residence}</p>
            </div>
          </div>
          <span className={clsx('text-xs px-2.5 py-1 rounded border font-medium', TIER_COLORS[tier])}>
            {tier}
          </span>
        </div>
        <div className="grid grid-cols-3 gap-3 text-xs">
          {[
            ['KYC Level', profile.kyc_level],
            ['Nationality', profile.nationality],
            ['Customer Since', profile.created_at?.slice(0, 10)],
          ].map(([label, value]) => (
            <div key={label} className="bg-slate-800/50 rounded-lg p-2.5">
              <p className="text-slate-500 mb-1">{label}</p>
              <p className="text-white font-medium">{value}</p>
            </div>
          ))}
        </div>
        {(profile.pep_flag || profile.sanctions_flag) && (
          <div className="mt-3 flex gap-2">
            {profile.pep_flag && (
              <span className="flex items-center gap-1 text-xs px-2 py-1 rounded bg-amber-500/15 text-amber-300 border border-amber-500/20">
                <Shield size={11} /> Politically Exposed Person
              </span>
            )}
            {profile.sanctions_flag && (
              <span className="flex items-center gap-1 text-xs px-2 py-1 rounded bg-red-500/15 text-red-300 border border-red-500/20">
                <AlertTriangle size={11} /> Sanctions Flag
              </span>
            )}
          </div>
        )}
      </div>

      {/* Risk Profile */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-slate-300">Risk Profile (30d)</h3>
          <div className="flex items-center gap-1.5 text-xs text-slate-400">
            <TrendIcon trend={profile.risk_profile.risk_trend} />
            <span className={clsx(
              profile.risk_profile.risk_trend === 'RISING' ? 'text-red-400'
              : profile.risk_profile.risk_trend === 'FALLING' ? 'text-emerald-400'
              : 'text-slate-400'
            )}>{profile.risk_profile.risk_trend}</span>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-3 text-center text-xs mb-4">
          <div className="bg-emerald-500/10 rounded-lg p-2.5">
            <p className="text-emerald-400 text-lg font-bold">{profile.risk_profile.allow_count}</p>
            <p className="text-slate-400">ALLOW</p>
          </div>
          <div className="bg-amber-500/10 rounded-lg p-2.5">
            <p className="text-amber-400 text-lg font-bold">{profile.risk_profile.challenge_count}</p>
            <p className="text-slate-400">CHALLENGE</p>
          </div>
          <div className="bg-red-500/10 rounded-lg p-2.5">
            <p className="text-red-400 text-lg font-bold">{profile.risk_profile.decline_count}</p>
            <p className="text-slate-400">DECLINE</p>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2 text-xs text-slate-400">
          <span>Avg Score (30d): <span className="text-white font-medium">{profile.risk_profile.avg_score_30d}</span></span>
          <span>Max Score (30d): <span className={clsx('font-medium',
            profile.risk_profile.max_score_30d > 700 ? 'text-red-400'
            : profile.risk_profile.max_score_30d > 400 ? 'text-amber-400'
            : 'text-white'
          )}>{profile.risk_profile.max_score_30d}</span></span>
          <span>Total Evals: <span className="text-white font-medium">{profile.risk_profile.total_evaluations}</span></span>
          <span>Fraud Incidents: <span className={clsx('font-medium',
            profile.risk_profile.fraud_incident_count > 0 ? 'text-red-400' : 'text-white'
          )}>{profile.risk_profile.fraud_incident_count}</span></span>
        </div>
      </div>

      {/* World Map */}
      <TransactionWorldMap customerId={profile.customer_id} />

      {/* Mule Indicators */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-slate-300">Mule Account Indicators</h3>
          <div className={clsx(
            'text-xs font-bold px-2.5 py-1 rounded',
            mi.is_likely_mule
              ? 'bg-red-500/20 text-red-300 border border-red-500/30'
              : 'bg-slate-700 text-slate-400'
          )}>
            Score: {mi.mule_score}/100{mi.is_likely_mule ? ' ⚠ LIKELY MULE' : ''}
          </div>
        </div>
        <div className="space-y-2.5">
          {[
            { label: 'Pass-through Account', value: mi.is_pass_through, info: `Turnover ratio: ${mi.turnover_ratio?.toFixed(2)}` },
            { label: 'High Sender Diversity', value: mi.high_sender_count, info: `${mi.unique_senders_30d} unique senders (30d)` },
            { label: 'Rapid Disbursement', value: mi.rapid_disbursement, info: `Avg hold time: ${mi.avg_hold_time_hours}h` },
            { label: 'Structuring Pattern', value: mi.structuring_risk, info: `${mi.structuring_incidents_30d} incidents` },
          ].map(({ label, value, info }) => (
            <div key={label} className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className={clsx('w-2 h-2 rounded-full flex-shrink-0',
                  value ? 'bg-red-400' : 'bg-slate-600'
                )} />
                <span className={clsx('text-sm', value ? 'text-red-300' : 'text-slate-400')}>{label}</span>
              </div>
              <span className="text-xs text-slate-500">{info}</span>
            </div>
          ))}
        </div>
        <div className="mt-4 grid grid-cols-2 gap-2 text-xs text-slate-500">
          <span>Inbound 30d: <span className="text-white">${mi.inbound_volume_30d?.toLocaleString()}</span></span>
          <span>Outbound 30d: <span className="text-white">${mi.outbound_volume_30d?.toLocaleString()}</span></span>
        </div>
      </div>

      {/* Accounts */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-5">
        <h3 className="text-sm font-semibold text-slate-300 mb-3">Accounts ({profile.accounts.length})</h3>
        <div className="space-y-2">
          {profile.accounts.map(acct => (
            <div key={acct.id} className="flex items-center justify-between text-xs bg-slate-800/50 rounded-lg px-3 py-2.5">
              <div>
                <p className="text-white font-medium font-mono">{acct.account_number}</p>
                <p className="text-slate-500">{acct.account_type} · {acct.bank_name} · {acct.country}</p>
              </div>
              <div className="text-right space-y-0.5">
                <p className="text-white font-medium">{acct.currency} {acct.balance.toLocaleString()}</p>
                {acct.is_dormant && (
                  <p className="text-amber-400 flex items-center gap-1 justify-end">
                    <Clock size={10} /> Dormant
                  </p>
                )}
                <button
                  onClick={() => snapshotMutation.mutate({ cid: profile.customer_id, aid: acct.id })}
                  className="text-blue-400 hover:text-blue-300"
                >
                  {snapshotMutation.isPending ? 'Computing…' : 'Compute snapshot'}
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Transaction history — clickable */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-5">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-slate-300">
            Transaction History ({profile.recent_transactions.length})
          </h3>
          <span className="text-xs text-slate-500">Click any row for risk details</span>
        </div>
        <div className="space-y-1.5 max-h-96 overflow-y-auto">
          {profile.recent_transactions.length === 0 ? (
            <p className="text-xs text-slate-500 text-center py-6">No transactions in last 30 days</p>
          ) : (
            profile.recent_transactions.map(t => (
              <TransactionRow
                key={t.id}
                txn={t}
                onClick={() => onTransactionClick(t)}
              />
            ))
          )}
        </div>
      </div>

      {/* Network connections */}
      {profile.network_connections.length > 0 && (
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-5">
          <h3 className="text-sm font-semibold text-slate-300 mb-3">
            Network Connections ({profile.network_connections.length})
          </h3>
          <div className="space-y-2">
            {profile.network_connections.map(conn => (
              <div key={conn.customer_id} className="flex items-center justify-between text-xs bg-slate-800/40 rounded-lg px-3 py-2">
                <div>
                  <p className="text-white font-medium">{conn.customer_name}</p>
                  <p className="text-slate-500">{conn.connection_type.replace(/_/g, ' ')}</p>
                </div>
                <span className={clsx('px-2 py-0.5 rounded border text-xs font-medium',
                  TIER_COLORS[conn.risk_tier] || TIER_COLORS.LOW
                )}>{conn.risk_tier}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Customer list card ─────────────────────────────────────────────────────────

interface CustomerRow {
  id: string; name: string; type: string; country: string; risk_tier: string
  account_count: number; pep_flag: boolean; sanctions_flag: boolean; kyc_level: string
}

function CustomerCard({ customer, isSelected, onClick }: {
  customer: CustomerRow; isSelected: boolean; onClick: () => void
}) {
  const tier = customer.risk_tier || 'LOW'
  return (
    <button
      onClick={onClick}
      className={clsx(
        'w-full bg-slate-900 border rounded-xl p-4 text-left transition-all group',
        isSelected
          ? 'border-blue-500/50 bg-slate-800/60'
          : 'border-slate-800 hover:border-slate-600',
      )}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className={clsx(
            'w-9 h-9 rounded-full flex items-center justify-center font-medium text-sm transition-colors',
            isSelected ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-300'
          )}>
            {customer.name?.[0]?.toUpperCase()}
          </div>
          <div>
            <p className="text-sm font-medium text-white">{customer.name}</p>
            <p className="text-xs text-slate-500">{customer.type} · {customer.country}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className={clsx('text-xs px-2 py-0.5 rounded border', TIER_COLORS[tier])}>
            {tier}
          </span>
          <ChevronRight size={14} className={clsx('transition-colors',
            isSelected ? 'text-blue-400 rotate-90' : 'text-slate-600 group-hover:text-slate-400'
          )} />
        </div>
      </div>
      <div className="mt-3 flex items-center gap-4 text-xs text-slate-500">
        <span>{customer.account_count} acct{customer.account_count !== 1 ? 's' : ''}</span>
        {customer.pep_flag && <span className="text-amber-400 font-medium">PEP</span>}
        {customer.sanctions_flag && <span className="text-red-400 font-medium">SANCTIONS</span>}
        <span className="ml-auto">{customer.kyc_level}</span>
      </div>
    </button>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function CustomerProfiles() {
  const [search, setSearch] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [riskFilter, setRiskFilter] = useState('')
  const [selectedTxn, setSelectedTxn] = useState<TransactionSummary | null>(null)

  const { data, isLoading } = useQuery({
    queryKey: ['customers', riskFilter],
    queryFn: () => listCustomers(0, 50, riskFilter || undefined),
  })

  const { data: profile, isLoading: profileLoading } = useQuery({
    queryKey: ['profile', selectedId],
    queryFn: () => getCustomerProfile(selectedId!),
    enabled: !!selectedId,
  })

  const customers = (data?.customers || []) as CustomerRow[]
  const filtered = customers.filter(c =>
    !search || c.name?.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white">Customer Profiles</h1>
        <p className="text-slate-400 text-sm mt-1">
          {data?.total || 0} customers · Click a customer, then click any transaction to see the full risk score breakdown
        </p>
      </div>

      {/* Filters */}
      <div className="flex gap-3 mb-5">
        <div className="relative flex-1 max-w-xs">
          <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            placeholder="Search by name…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="w-full pl-9 pr-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500"
          />
        </div>
        <select
          value={riskFilter}
          onChange={e => setRiskFilter(e.target.value)}
          className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
        >
          <option value="">All Risk Tiers</option>
          {['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].map(t => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
        {/* Customer list */}
        <div className="xl:col-span-2 space-y-2 max-h-[calc(100vh-220px)] overflow-y-auto pr-1">
          {isLoading ? (
            <div className="text-center py-12 text-slate-500">Loading customers…</div>
          ) : filtered.length === 0 ? (
            <div className="text-center py-12 text-slate-500">No customers found</div>
          ) : (
            filtered.map(c => (
              <CustomerCard
                key={c.id}
                customer={c}
                isSelected={selectedId === c.id}
                onClick={() => setSelectedId(c.id)}
              />
            ))
          )}
        </div>

        {/* Profile panel */}
        <div className="xl:col-span-3 max-h-[calc(100vh-220px)] overflow-y-auto pr-1">
          {selectedId === null ? (
            <div className="bg-slate-900 border border-slate-800 border-dashed rounded-xl p-16 text-center">
              <User size={40} className="text-slate-600 mx-auto mb-3" />
              <p className="text-slate-400 text-sm font-medium">Select a customer to view their profile</p>
              <p className="text-slate-600 text-xs mt-1">
                Then click any transaction row to see the full risk score breakdown
              </p>
            </div>
          ) : profileLoading ? (
            <div className="flex items-center justify-center py-20">
              <div className="w-6 h-6 border-2 border-blue-500/30 border-t-blue-400 rounded-full animate-spin" />
            </div>
          ) : profile ? (
            <ProfilePanel
              profile={profile}
              onTransactionClick={setSelectedTxn}
            />
          ) : (
            <div className="text-center py-20 text-red-400">Profile not found</div>
          )}
        </div>
      </div>

      {/* Transaction detail drawer */}
      {selectedTxn && selectedId && (
        <TransactionDrawer
          customerId={selectedId}
          txnId={selectedTxn.id}
          txnSummary={selectedTxn}
          onClose={() => setSelectedTxn(null)}
        />
      )}
    </div>
  )
}
