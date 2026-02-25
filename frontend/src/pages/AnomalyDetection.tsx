import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  ShieldAlert, Search, AlertTriangle, TrendingUp, Activity,
  RefreshCw, Zap, ChevronLeft, ChevronRight, Info,
} from 'lucide-react'
import {
  getAnomalySummary, listMuleSuspects, getAccountAnomaly,
  trainAnomalyDetector, scanAccounts,
  type AnomalyAccountResult,
} from '../api/client'

// ── Helpers ───────────────────────────────────────────────────────────────────

const INDICATOR_META: Record<string, { label: string; color: string; desc: string }> = {
  PASS_THROUGH:     { label: 'Pass-through', color: 'bg-red-100 text-red-700',    desc: 'Inbound ≈ outbound volume — funds not held' },
  HIGH_SENDER_COUNT:{ label: 'Many senders', color: 'bg-orange-100 text-orange-700', desc: 'Receives from 5+ distinct sources in 30 days' },
  STRUCTURING:      { label: 'Structuring',  color: 'bg-purple-100 text-purple-700', desc: 'Repeated sub-$10k transactions (CTR avoidance)' },
  RAPID_DISBURSEMENT:{ label: 'Rapid disbursement', color: 'bg-yellow-100 text-yellow-700', desc: 'Funds moved out quickly after receipt' },
}

function ScoreGauge({ score }: { score: number }) {
  const pct   = Math.min(100, Math.max(0, score))
  const color  = pct >= 70 ? '#ef4444' : pct >= 40 ? '#f97316' : '#22c55e'
  const angle  = -135 + (pct / 100) * 270
  const risk   = pct >= 70 ? 'HIGH' : pct >= 40 ? 'MEDIUM' : 'LOW'
  const rColor = pct >= 70 ? 'text-red-600' : pct >= 40 ? 'text-orange-500' : 'text-green-600'

  return (
    <div className="flex flex-col items-center gap-1">
      <svg viewBox="0 0 120 80" className="w-28 h-20">
        {/* Background arc */}
        <path d="M 15 75 A 50 50 0 1 1 105 75" fill="none" stroke="#e5e7eb" strokeWidth="10" strokeLinecap="round"/>
        {/* Score arc */}
        <path d="M 15 75 A 50 50 0 1 1 105 75" fill="none" stroke={color}
          strokeWidth="10" strokeLinecap="round"
          strokeDasharray={`${(pct / 100) * 157} 157`}/>
        {/* Needle */}
        <g transform={`rotate(${angle}, 60, 75)`}>
          <line x1="60" y1="75" x2="60" y2="32" stroke="#374151" strokeWidth="2.5" strokeLinecap="round"/>
          <circle cx="60" cy="75" r="4" fill="#374151"/>
        </g>
        <text x="60" y="72" textAnchor="middle" fontSize="18" fontWeight="bold" fill={color}>{score.toFixed(0)}</text>
      </svg>
      <span className={`text-xs font-bold ${rColor}`}>{risk} ANOMALY</span>
    </div>
  )
}

function IndicatorBadge({ ind }: { ind: string }) {
  const meta = INDICATOR_META[ind] ?? { label: ind, color: 'bg-gray-100 text-gray-700', desc: '' }
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${meta.color}`}
      title={meta.desc}>
      {meta.label}
    </span>
  )
}

function SummaryCard({ label, value, sub, icon: Icon, color }:
  { label: string; value: string | number; sub?: string; icon: React.ElementType; color: string }) {
  return (
    <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4 flex items-start gap-3">
      <div className={`p-2 rounded-lg ${color}`}>
        <Icon className="w-5 h-5 text-white" />
      </div>
      <div>
        <p className="text-2xl font-bold text-gray-900">{typeof value === 'number' ? value.toLocaleString() : value}</p>
        <p className="text-sm text-gray-500">{label}</p>
        {sub && <p className="text-xs text-gray-400 mt-0.5">{sub}</p>}
      </div>
    </div>
  )
}

// ── Account Detail Drawer ─────────────────────────────────────────────────────

function AccountDetailDrawer({
  accountId,
  prefill,
  onClose,
}: {
  accountId: string
  prefill?: AnomalyAccountResult
  onClose: () => void
}) {
  const { data: liveData, isLoading, isError, error } = useQuery({
    queryKey: ['account-anomaly', accountId],
    queryFn: () => getAccountAnomaly(accountId),
    retry: 1,
  })

  // Use live data if available, fall back to pre-filled data from the suspects table
  const data = liveData ?? prefill

  return (
    <div className="fixed inset-0 z-50 flex justify-end" onClick={onClose}>
      <div className="w-full max-w-md bg-white shadow-2xl border-l border-gray-200 overflow-y-auto"
        onClick={e => e.stopPropagation()}>
        <div className="sticky top-0 bg-white border-b border-gray-100 px-5 py-4 flex items-center justify-between">
          <div>
            <h2 className="font-semibold text-gray-900">Account Anomaly Detail</h2>
            {data && <p className="text-xs text-gray-400 font-mono mt-0.5">{accountId.slice(0,8)}…</p>}
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 p-1 rounded text-lg leading-none">✕</button>
        </div>

        {isLoading && !prefill && (
          <div className="p-8 text-center text-gray-400">
            <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2" />
            <p className="text-sm">Loading account details…</p>
          </div>
        )}

        {isError && !prefill && (
          <div className="p-6 m-4 bg-red-50 border border-red-200 rounded-xl text-sm">
            <p className="font-semibold text-red-700 mb-1">Could not load live score</p>
            <p className="text-red-600 text-xs">
              {(error as Error)?.message?.includes('503')
                ? 'The anomaly detector is not trained yet. Click "Train Detector" first.'
                : 'An error occurred fetching account details.'}
            </p>
          </div>
        )}

        {isLoading && prefill && (
          <div className="px-5 pt-3 pb-0">
            <div className="flex items-center gap-2 text-xs text-indigo-500 bg-indigo-50 rounded-lg px-3 py-2">
              <RefreshCw className="w-3 h-3 animate-spin" />
              Refreshing with live score…
            </div>
          </div>
        )}

        {data && (
          <div className="p-5 space-y-5">
            {/* Score */}
            <div className="flex flex-col items-center bg-gray-50 rounded-xl p-4">
              <ScoreGauge score={data.anomaly_score} />
              <p className="text-sm text-gray-500 mt-1">Composite Mule Score</p>
              <div className="mt-2 flex gap-2 text-xs">
                <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded-full">
                  KNN: {(data.knn_distance_score ?? 0).toFixed(1)}
                </span>
                <span className="px-2 py-0.5 bg-purple-100 text-purple-700 rounded-full">
                  Rules: {data.rule_score ?? 0}
                </span>
              </div>
            </div>

            {/* Account info */}
            <div className="grid grid-cols-2 gap-2 text-sm">
              {([
                ['Account #', data.account_number ?? '—'],
                ['Bank',      data.bank_name ?? '—'],
                ['Type',      data.account_type ?? '—'],
                ['Country',   data.customer_country ?? '—'],
                ['Customer',  data.customer_name ?? '—'],
                ['Suspect',   data.is_mule_suspect ? '⚠ Yes' : '✓ No'],
              ] as [string, string][]).map(([k, v]) => (
                <div key={k} className={`rounded-lg p-2 ${
                  k === 'Suspect' && data.is_mule_suspect ? 'bg-red-50' : 'bg-gray-50'
                }`}>
                  <p className="text-gray-500 text-xs">{k}</p>
                  <p className={`font-medium truncate ${
                    k === 'Suspect' && data.is_mule_suspect ? 'text-red-700' : 'text-gray-800'
                  }`}>{v}</p>
                </div>
              ))}
            </div>

            {/* Score breakdown bars */}
            <div>
              <h3 className="text-sm font-semibold text-gray-700 mb-2">Score Breakdown</h3>
              <div className="space-y-2">
                {([
                  { label: 'KNN Distance Score', val: data.knn_distance_score ?? 0, color: 'bg-blue-500' },
                  { label: 'Rule-based Score',   val: data.rule_score ?? 0,         color: 'bg-purple-500' },
                  { label: 'Composite Score',    val: data.anomaly_score,           color: 'bg-orange-500' },
                ] as { label: string; val: number; color: string }[]).map(({ label, val, color }) => (
                  <div key={label}>
                    <div className="flex justify-between text-xs text-gray-500 mb-0.5">
                      <span>{label}</span>
                      <span className="font-medium tabular-nums">{val.toFixed(1)} / 100</span>
                    </div>
                    <div className="h-2 rounded-full bg-gray-100">
                      <div className={`h-2 rounded-full ${color} transition-all`}
                        style={{ width: `${Math.min(val, 100)}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Risk indicators */}
            {(data.indicators?.length ?? 0) > 0 ? (
              <div>
                <h3 className="text-sm font-semibold text-gray-700 mb-2">
                  Risk Indicators ({data.indicators.length})
                </h3>
                <div className="flex flex-wrap gap-2 mb-3">
                  {data.indicators.map(ind => <IndicatorBadge key={ind} ind={ind} />)}
                </div>
                <div className="space-y-2">
                  {data.indicators.map(ind => {
                    const meta = INDICATOR_META[ind]
                    return meta ? (
                      <div key={ind} className="flex gap-2 text-xs text-gray-600 bg-yellow-50 border border-yellow-100 rounded-lg p-2.5">
                        <Info className="w-3.5 h-3.5 text-yellow-500 flex-shrink-0 mt-0.5" />
                        <div>
                          <p className="font-medium text-gray-700">{meta.label}</p>
                          <p className="text-gray-500 mt-0.5">{meta.desc}</p>
                        </div>
                      </div>
                    ) : null
                  })}
                </div>
              </div>
            ) : (
              <div className="bg-green-50 border border-green-100 rounded-lg p-3 text-xs text-green-700">
                No specific rule-based indicators triggered. Anomaly score is driven by KNN distance.
              </div>
            )}

            {/* Financial signals */}
            <div>
              <h3 className="text-sm font-semibold text-gray-700 mb-2">Financial Signals</h3>
              <div className="grid grid-cols-2 gap-2 text-sm">
                {([
                  ['Pass-through ratio',  `${(data.pass_through_ratio ?? 0).toFixed(3)}×`],
                  ['Unique senders 30d',  String(data.unique_senders_30d ?? 0)],
                  ['Structuring txns 30d',String(data.structuring_30d ?? 0)],
                  ['Inbound volume',      `$${(data.in_volume ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`],
                  ['Outbound volume',     `$${(data.out_volume ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`],
                  ['Net flow',            `$${((data.in_volume ?? 0) - (data.out_volume ?? 0)).toLocaleString(undefined, { maximumFractionDigits: 0 })}`],
                ] as [string, string][]).map(([k, v]) => (
                  <div key={k} className="bg-gray-50 rounded-lg p-2">
                    <p className="text-gray-500 text-xs">{k}</p>
                    <p className="font-semibold text-gray-800 tabular-nums">{v}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {!data && !isLoading && (
          <div className="p-8 text-center text-gray-400">
            <AlertTriangle className="w-8 h-8 mx-auto mb-2 opacity-40" />
            <p className="text-sm">No data available for this account.</p>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Main Page ─────────────────────────────────────────────────────────────────

const SAMPLE_OPTIONS = [
  { label: '1 000',  value: 1_000 },
  { label: '5 000',  value: 5_000 },
  { label: '10 000', value: 10_000 },
  { label: '25 000', value: 25_000 },
]

export default function AnomalyDetection() {
  const qc = useQueryClient()
  const [page, setPage]         = useState(1)
  const [search, setSearch]     = useState('')
  const [selectedAcct, setSelectedAcct] = useState<string | null>(null)
  const [selectedPrefill, setSelectedPrefill] = useState<AnomalyAccountResult | undefined>(undefined)
  const [maxNormal, setMaxNormal]   = useState(5_000)
  const [maxAccounts, setMaxAccounts] = useState(5_000)
  const PAGE_SIZE = 50

  const { data: summary } = useQuery({
    queryKey: ['anomaly-summary'],
    queryFn: getAnomalySummary,
    refetchInterval: 30_000,
  })

  const { data: suspects, isLoading } = useQuery({
    queryKey: ['anomaly-suspects', page],
    queryFn: () => listMuleSuspects(page, PAGE_SIZE),
    placeholderData: (prev) => prev,
  })

  const trainMutation = useMutation({
    mutationFn: () => trainAnomalyDetector(maxNormal, maxAccounts),
    onSuccess: () => {
      setTimeout(() => {
        qc.invalidateQueries({ queryKey: ['anomaly-summary'] })
        qc.invalidateQueries({ queryKey: ['anomaly-suspects'] })
      }, 4000)
    },
  })

  const scanMutation = useMutation({
    mutationFn: () => scanAccounts(true, maxAccounts),
    onSuccess: () => {
      setTimeout(() => {
        qc.invalidateQueries({ queryKey: ['anomaly-summary'] })
        qc.invalidateQueries({ queryKey: ['anomaly-suspects'] })
      }, 5000)
    },
  })

  const filtered = suspects?.suspects.filter(s =>
    !search ||
    s.customer_name?.toLowerCase().includes(search.toLowerCase()) ||
    s.account_number?.toLowerCase().includes(search.toLowerCase()) ||
    s.customer_country?.toLowerCase().includes(search.toLowerCase()),
  ) ?? []

  const totalPages = suspects?.total_pages ?? 1

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
            <ShieldAlert className="w-7 h-7 text-red-500" />
            Mule Account Anomaly Detection
          </h1>
          <p className="text-sm text-gray-500 mt-0.5">
            KNN-powered detection of money-mule accounts — trained on normal transaction patterns
          </p>
        </div>

        {/* Controls panel */}
        <div className="flex flex-col gap-2 min-w-[340px]">
          {/* Sample size selectors */}
          <div className="flex items-center gap-2 bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 text-xs">
            <Info className="w-3.5 h-3.5 text-gray-400 shrink-0" />
            <label className="text-gray-500 whitespace-nowrap">Train on</label>
            <select
              value={maxNormal}
              onChange={e => setMaxNormal(Number(e.target.value))}
              className="border border-gray-300 rounded px-1.5 py-0.5 bg-white text-gray-700"
            >
              {SAMPLE_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label} txns</option>
              ))}
            </select>
            <label className="text-gray-500 whitespace-nowrap">· Scan</label>
            <select
              value={maxAccounts}
              onChange={e => setMaxAccounts(Number(e.target.value))}
              className="border border-gray-300 rounded px-1.5 py-0.5 bg-white text-gray-700"
            >
              {SAMPLE_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label} accts</option>
              ))}
            </select>
          </div>

          {/* Action buttons */}
          <div className="flex gap-2">
            <button
              onClick={() => trainMutation.mutate()}
              disabled={trainMutation.isPending}
              className="flex-1 flex items-center justify-center gap-1.5 px-4 py-2 bg-indigo-600 text-white text-sm font-medium
                rounded-lg hover:bg-indigo-700 disabled:opacity-60 transition-colors"
            >
              {trainMutation.isPending
                ? <RefreshCw className="w-4 h-4 animate-spin" />
                : <Zap className="w-4 h-4" />}
              {trainMutation.isPending ? 'Training…' : 'Train Detector'}
            </button>
            <button
              onClick={() => scanMutation.mutate()}
              disabled={scanMutation.isPending || !summary?.detector_trained}
              className="flex-1 flex items-center justify-center gap-1.5 px-4 py-2 bg-orange-600 text-white text-sm font-medium
                rounded-lg hover:bg-orange-700 disabled:opacity-60 transition-colors"
            >
              {scanMutation.isPending
                ? <RefreshCw className="w-4 h-4 animate-spin" />
                : <Activity className="w-4 h-4" />}
              {scanMutation.isPending ? 'Scanning…' : 'Scan Accounts'}
            </button>
          </div>

          {/* In-progress banner */}
          {(trainMutation.isPending || scanMutation.isPending) && (
            <p className="text-xs text-indigo-600 text-center animate-pulse">
              {trainMutation.isPending
                ? `Training on ${maxNormal.toLocaleString()} transactions + scanning ${maxAccounts.toLocaleString()} accounts…`
                : `Scanning ${maxAccounts.toLocaleString()} accounts for anomalies…`}
            </p>
          )}
          {(trainMutation.isSuccess || scanMutation.isSuccess) && (
            <p className="text-xs text-green-600 text-center">
              ✓ Complete — results refreshing…
            </p>
          )}
        </div>
      </div>

      {/* Status banner if detector not trained */}
      {summary && !summary.detector_trained && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 flex gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-amber-800">Detector not trained</p>
            <p className="text-sm text-amber-700">
              Click <strong>Train Detector</strong> to build the KNN anomaly index on normal transactions.
              Training runs in the background (~1–2 minutes).
            </p>
          </div>
        </div>
      )}

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <SummaryCard
          label="Accounts Scanned"
          value={summary?.scored_accounts ?? 0}
          sub={`${summary?.coverage_pct ?? 0}% coverage`}
          icon={Activity}
          color="bg-blue-500"
        />
        <SummaryCard
          label="Mule Suspects"
          value={summary?.mule_suspects ?? 0}
          sub={`${summary?.suspect_rate_pct ?? 0}% of scanned`}
          icon={AlertTriangle}
          color="bg-orange-500"
        />
        <SummaryCard
          label="High-Risk Accounts"
          value={summary?.high_risk_accounts ?? 0}
          sub="Score ≥ 70"
          icon={ShieldAlert}
          color="bg-red-500"
        />
        <SummaryCard
          label="Total Accounts"
          value={summary?.total_accounts ?? 0}
          icon={TrendingUp}
          color="bg-indigo-500"
        />
      </div>

      {/* Indicator distribution */}
      {summary && summary.indicator_distribution.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Indicator Distribution</h3>
          <div className="flex flex-wrap gap-3">
            {summary.indicator_distribution.map(({ indicator, freq }) => {
              const meta = INDICATOR_META[indicator]
              return (
                <div key={indicator} className="flex items-center gap-2">
                  <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${meta?.color ?? 'bg-gray-100 text-gray-700'}`}>
                    {meta?.label ?? indicator}
                  </span>
                  <span className="text-xs text-gray-500">{freq.toLocaleString()} accounts</span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
        <input
          type="text"
          placeholder="Filter by customer name, account number, or country…"
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="w-full pl-10 pr-4 py-2.5 border border-gray-200 rounded-xl text-sm
            focus:outline-none focus:ring-2 focus:ring-indigo-300 bg-white"
        />
      </div>

      {/* Suspects table */}
      <div className="bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
        <div className="px-5 py-3 border-b border-gray-100 flex items-center justify-between">
          <h2 className="font-semibold text-gray-900">
            Mule Suspects
            {suspects && (
              <span className="ml-2 text-sm font-normal text-gray-500">
                ({suspects.total.toLocaleString()} total)
              </span>
            )}
          </h2>
          {isLoading && <RefreshCw className="w-4 h-4 animate-spin text-gray-400" />}
        </div>

        {suspects?.total === 0 ? (
          <div className="py-16 text-center text-gray-400">
            <ShieldAlert className="w-12 h-12 mx-auto mb-3 opacity-30" />
            <p className="font-medium">No mule suspects detected</p>
            <p className="text-sm">
              {summary?.detector_trained
                ? 'Run a scan to analyse accounts'
                : 'Train the detector first, then run a scan'}
            </p>
          </div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 border-b border-gray-100">
                  <tr>
                    {['Anomaly Score', 'Customer', 'Account', 'Country', 'Indicators',
                      'Pass-through', 'Senders', 'Structuring', 'Action'].map(h => (
                      <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-50">
                  {filtered.map((s: AnomalyAccountResult) => (
                    <tr key={s.account_id}
                      className="hover:bg-gray-50 transition-colors cursor-pointer"
                      onClick={() => { setSelectedPrefill(s); setSelectedAcct(s.account_id) }}
                    >
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div className={`w-2 h-2 rounded-full flex-shrink-0 ${
                            s.anomaly_score >= 70 ? 'bg-red-500' :
                            s.anomaly_score >= 40 ? 'bg-orange-400' : 'bg-green-400'
                          }`} />
                          <span className={`font-bold tabular-nums ${
                            s.anomaly_score >= 70 ? 'text-red-600' :
                            s.anomaly_score >= 40 ? 'text-orange-600' : 'text-green-600'
                          }`}>
                            {s.anomaly_score.toFixed(1)}
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <p className="font-medium text-gray-900 truncate max-w-[140px]">
                          {s.customer_name ?? '—'}
                        </p>
                        <p className="text-xs text-gray-400 truncate">{s.customer_id?.slice(0, 8)}…</p>
                      </td>
                      <td className="px-4 py-3 font-mono text-gray-700 text-xs">
                        {s.account_number ?? '—'}
                      </td>
                      <td className="px-4 py-3 text-gray-600">{s.customer_country ?? '—'}</td>
                      <td className="px-4 py-3">
                        <div className="flex flex-wrap gap-1">
                          {s.indicators.map(ind => <IndicatorBadge key={ind} ind={ind} />)}
                        </div>
                      </td>
                      <td className="px-4 py-3 tabular-nums text-gray-700">
                        {s.pass_through_ratio.toFixed(2)}×
                      </td>
                      <td className="px-4 py-3 text-gray-700">{s.unique_senders_30d}</td>
                      <td className="px-4 py-3 text-gray-700">{s.structuring_30d}</td>
                      <td className="px-4 py-3">
                        <button
                          onClick={e => { e.stopPropagation(); setSelectedPrefill(s); setSelectedAcct(s.account_id) }}
                          className="text-xs text-indigo-600 hover:text-indigo-800 font-medium"
                        >
                          Details →
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="px-5 py-3 border-t border-gray-100 flex items-center justify-between text-sm text-gray-500">
                <span>
                  Page {page} of {totalPages} — {suspects?.total.toLocaleString()} suspects
                </span>
                <div className="flex gap-1">
                  <button
                    onClick={() => setPage(p => Math.max(1, p - 1))}
                    disabled={page === 1}
                    className="p-1.5 rounded-lg hover:bg-gray-100 disabled:opacity-40 transition-colors"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  {Array.from({ length: Math.min(7, totalPages) }, (_, i) => {
                    const n = page <= 4 ? i + 1
                      : page >= totalPages - 3 ? totalPages - 6 + i
                      : page - 3 + i
                    if (n < 1 || n > totalPages) return null
                    return (
                      <button
                        key={n}
                        onClick={() => setPage(n)}
                        className={`w-8 h-8 rounded-lg text-sm font-medium transition-colors
                          ${n === page ? 'bg-indigo-600 text-white' : 'hover:bg-gray-100 text-gray-600'}`}
                      >
                        {n}
                      </button>
                    )
                  })}
                  <button
                    onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                    disabled={page === totalPages}
                    className="p-1.5 rounded-lg hover:bg-gray-100 disabled:opacity-40 transition-colors"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* How it works card */}
      <div className="bg-gradient-to-br from-indigo-50 to-blue-50 rounded-xl border border-indigo-100 p-5">
        <h3 className="text-sm font-semibold text-indigo-800 mb-3">How anomaly detection works</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-indigo-700">
          <div className="flex gap-2">
            <div className="w-6 h-6 rounded-full bg-indigo-600 text-white flex items-center justify-center flex-shrink-0 font-bold text-xs">1</div>
            <div>
              <p className="font-semibold mb-0.5">Build KNN index</p>
              <p>FAISS IndexFlatL2 is built from <em>normal</em> transaction feature vectors only, establishing a baseline of legitimate behaviour.</p>
            </div>
          </div>
          <div className="flex gap-2">
            <div className="w-6 h-6 rounded-full bg-indigo-600 text-white flex items-center justify-center flex-shrink-0 font-bold text-xs">2</div>
            <div>
              <p className="font-semibold mb-0.5">Compute distances</p>
              <p>For each account, its transactions are projected into feature space. The mean distance to K nearest normal neighbours measures how far the account deviates from legitimate behaviour.</p>
            </div>
          </div>
          <div className="flex gap-2">
            <div className="w-6 h-6 rounded-full bg-indigo-600 text-white flex items-center justify-center flex-shrink-0 font-bold text-xs">3</div>
            <div>
              <p className="font-semibold mb-0.5">Composite mule score</p>
              <p>KNN distance score (50%) + rule-based mule indicators (50%) produce a 0–100 composite score. Accounts ≥ 40 are flagged as mule suspects.</p>
            </div>
          </div>
        </div>
      </div>

      {/* Account detail drawer */}
      {selectedAcct && (
        <AccountDetailDrawer
          accountId={selectedAcct}
          prefill={selectedPrefill}
          onClose={() => { setSelectedAcct(null); setSelectedPrefill(undefined) }}
        />
      )}
    </div>
  )
}
