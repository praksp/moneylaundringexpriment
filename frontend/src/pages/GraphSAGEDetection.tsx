import { useState } from 'react'
import { useQuery, useMutation, useQueryClient, keepPreviousData } from '@tanstack/react-query'
import {
  Brain, Network, RefreshCw, AlertTriangle, ShieldCheck,
  ChevronLeft, ChevronRight, TrendingUp, BarChart3, GitCompare,
  Zap, Info, CheckCircle2, XCircle, ArrowUpRight, ArrowDownLeft,
  ShieldAlert, Activity,
} from 'lucide-react'
import {
  getGraphSAGESummary, listGraphSAGESuspects, getGraphSAGEComparison, trainGraphSAGE,
  getGraphSAGEAccount,
} from '../api/client'
import type { GraphSAGESuspect, GraphSAGESummary, GraphSAGEAccountDetail } from '../api/client'

// ── Helpers ────────────────────────────────────────────────────────────────────

function scoreColor(score: number) {
  if (score >= 75) return 'text-red-400'
  if (score >= 50) return 'text-orange-400'
  if (score >= 30) return 'text-yellow-400'
  return 'text-green-400'
}

function scoreBg(score: number) {
  if (score >= 75) return 'bg-red-900/40 border-red-700/40'
  if (score >= 50) return 'bg-orange-900/40 border-orange-700/40'
  if (score >= 30) return 'bg-yellow-900/40 border-yellow-700/40'
  return 'bg-green-900/40 border-green-700/40'
}

function StatCard({ label, value, sub, color = 'text-white' }: {
  label: string; value: string | number; sub?: string; color?: string
}) {
  return (
    <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-5">
      <p className="text-sm text-slate-400">{label}</p>
      <p className={`text-3xl font-bold mt-1 ${color}`}>{value}</p>
      {sub && <p className="text-xs text-slate-500 mt-1">{sub}</p>}
    </div>
  )
}

// ── Feature bar ───────────────────────────────────────────────────────────────

function FeatureRow({ feat }: { feat: GraphSAGEAccountDetail['features'][0] }) {
  const pct = typeof feat.value === 'number'
    ? Math.min(100, (feat.value / (Number(feat.threshold) || 100)) * 100)
    : feat.triggered ? 100 : 0

  return (
    <div className={`rounded-lg border p-3 space-y-1.5 ${feat.triggered
      ? 'bg-red-900/20 border-red-700/40'
      : 'bg-slate-800/40 border-slate-700/30'}`}>
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          {feat.triggered
            ? <AlertTriangle className="w-3.5 h-3.5 text-red-400 shrink-0" />
            : <CheckCircle2  className="w-3.5 h-3.5 text-slate-600 shrink-0" />}
          <span className={`text-xs font-medium truncate ${feat.triggered ? 'text-red-200' : 'text-slate-400'}`}>
            {feat.name}
          </span>
        </div>
        <div className="text-right shrink-0">
          <span className={`text-sm font-bold ${feat.triggered ? 'text-red-300' : 'text-slate-400'}`}>
            {String(feat.value)}{feat.unit}
          </span>
          <span className="text-slate-600 text-xs ml-1">/ {String(feat.threshold)}{feat.unit}</span>
        </div>
      </div>
      <div className="h-1.5 bg-slate-700/60 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${feat.triggered ? 'bg-red-500' : 'bg-slate-600'}`}
          style={{ width: `${Math.min(100, pct)}%` }}
        />
      </div>
      <p className="text-[10px] text-slate-500">{feat.description}</p>
    </div>
  )
}

// ── Account drawer ─────────────────────────────────────────────────────────────

function AccountDrawer({
  account,
  onClose,
}: {
  account: GraphSAGESuspect
  onClose: () => void
}) {
  const [activeTab, setActiveTab] = useState<'features' | 'transactions'>('features')

  const { data: detail, isLoading } = useQuery<GraphSAGEAccountDetail>({
    queryKey: ['graphsage-account-detail', account.account_id],
    queryFn: () => getGraphSAGEAccount(account.account_id),
    staleTime: 60_000,
  })

  const sage = account.graphsage_score
  const knn  = account.knn_anomaly_score ?? 0

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-end bg-black/60" onClick={onClose}>
      <div
        className="h-full w-full max-w-2xl bg-slate-900 border-l border-slate-700 overflow-y-auto shadow-2xl flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-5 border-b border-slate-700 flex items-start justify-between shrink-0">
          <div>
            <h3 className="font-semibold text-white flex items-center gap-2">
              <Network className="w-4 h-4 text-purple-400" />
              Mule Account Analysis
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              {account.account_number ?? account.account_id}
              {account.bank_name && <span className="ml-2 text-slate-500">· {account.bank_name}</span>}
            </p>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-white text-xl shrink-0">✕</button>
        </div>

        {/* Score banner */}
        <div className={`p-4 border-b border-slate-700/50 ${scoreBg(sage)}`}>
          <div className="flex items-center gap-4">
            <div className="text-center">
              <p className={`text-3xl font-bold ${scoreColor(sage)}`}>{sage.toFixed(0)}</p>
              <p className="text-xs text-slate-400">SAGE score</p>
            </div>
            {account.knn_anomaly_score != null && (
              <>
                <div className="text-slate-600 text-xl">·</div>
                <div className="text-center">
                  <p className={`text-3xl font-bold ${scoreColor(knn)}`}>{knn.toFixed(0)}</p>
                  <p className="text-xs text-slate-400">KNN score</p>
                </div>
              </>
            )}
            <div className="ml-auto text-right space-y-1">
              {detail?.mule_label_reason && (
                <p className="text-xs text-red-300">
                  <AlertTriangle className="w-3 h-3 inline mr-1" />
                  {detail.mule_label_reason}
                </p>
              )}
              {detail?.triggered_count != null && (
                <p className="text-xs text-slate-400">
                  {detail.triggered_count} of {detail.features?.length ?? 0} indicators triggered
                </p>
              )}
              <div className="flex items-center gap-2 text-xs justify-end">
                {sage >= 50 && knn >= 50
                  ? <span className="text-red-300">Flagged by BOTH models</span>
                  : sage >= 50
                  ? <span className="text-orange-300">GraphSAGE flagged</span>
                  : <span className="text-yellow-300">KNN flagged</span>}
              </div>
            </div>
          </div>
        </div>

        {/* Customer + account meta */}
        <div className="px-5 py-3 border-b border-slate-700/30 grid grid-cols-2 gap-x-6 gap-y-1 text-xs shrink-0">
          {[
            ['Customer',   detail?.customer_name   ?? account.customer_name   ?? '—'],
            ['Country',    detail?.customer_country ?? account.customer_country ?? '—'],
            ['Account',    detail?.account_type    ?? account.account_type    ?? '—'],
            ['Risk tier',  detail?.risk_tier       ?? '—'],
            ['KYC',        detail?.kyc_level       ?? '—'],
            ['PEP',        detail?.pep_flag ? '🔴 YES' : '✅ No'],
          ].map(([k, v]) => (
            <div key={k} className="flex justify-between py-0.5">
              <span className="text-slate-500">{k}</span>
              <span className="text-slate-300 font-medium">{v}</span>
            </div>
          ))}
        </div>

        {/* Tab bar */}
        <div className="flex border-b border-slate-700/50 shrink-0">
          {[
            { id: 'features',      label: `Features (${detail?.features?.length ?? '…'})`,       icon: Activity },
            { id: 'transactions',  label: `Fraud Txns (${detail?.fraud_txn_count ?? '…'})`,      icon: ShieldAlert },
          ].map(t => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id as 'features' | 'transactions')}
              className={`flex items-center gap-1.5 px-4 py-3 text-sm border-b-2 transition-colors ${
                activeTab === t.id
                  ? 'border-purple-500 text-purple-300'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              <t.icon className="w-3.5 h-3.5" />
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="flex-1 overflow-y-auto p-5">
          {isLoading ? (
            <div className="flex items-center justify-center h-32 text-slate-400 text-sm">
              <RefreshCw className="w-4 h-4 animate-spin mr-2" /> Loading account analysis…
            </div>
          ) : activeTab === 'features' ? (
            <div className="space-y-3">
              {/* Feature summary */}
              {detail?.feature_summary && (
                <div className="grid grid-cols-3 gap-2 mb-4">
                  {[
                    { label: 'Fraud ratio', value: `${detail.feature_summary.fraud_ratio_pct}%`,   alert: detail.feature_summary.fraud_ratio_pct >= 30 },
                    { label: 'Fraud txns',  value: `${detail.feature_summary.fraud_count}`,        alert: detail.feature_summary.fraud_count > 0 },
                    { label: 'Pass-thru',   value: `${detail.feature_summary.pass_through}×`,      alert: detail.feature_summary.pass_through >= 0.7 },
                    { label: 'Senders',     value: `${detail.feature_summary.unique_senders}`,     alert: detail.feature_summary.unique_senders > 10 },
                    { label: 'Outbound',    value: `$${Number(detail.feature_summary.out_volume_usd).toLocaleString(undefined, {maximumFractionDigits: 0})}`, alert: false },
                    { label: 'Patterns',    value: `${detail.feature_summary.pattern_count}`,      alert: detail.feature_summary.pattern_count >= 2 },
                  ].map(item => (
                    <div key={item.label} className={`rounded-lg p-2 text-center border ${item.alert ? 'bg-red-900/20 border-red-700/30' : 'bg-slate-800/40 border-slate-700/30'}`}>
                      <p className="text-[10px] text-slate-500">{item.label}</p>
                      <p className={`text-sm font-bold mt-0.5 ${item.alert ? 'text-red-300' : 'text-slate-300'}`}>{item.value}</p>
                    </div>
                  ))}
                </div>
              )}

              {/* Per-feature rows */}
              {detail?.features?.map(f => (
                <FeatureRow key={f.name} feat={f} />
              ))}
            </div>
          ) : (
            /* Transaction list */
            <div className="space-y-2">
              {!detail?.fraud_transactions?.length ? (
                <div className="text-center text-slate-500 py-10">
                  <CheckCircle2 className="w-8 h-8 mx-auto mb-2 opacity-30" />
                  No fraud transactions found for this account.
                </div>
              ) : (
                detail.fraud_transactions.map((t, i) => (
                  <div key={t.txn_id ?? i} className="bg-slate-800/50 border border-slate-700/40 rounded-xl p-3 space-y-2">
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-2">
                        {t.direction === 'OUTBOUND'
                          ? <ArrowUpRight   className="w-4 h-4 text-red-400   shrink-0" />
                          : <ArrowDownLeft  className="w-4 h-4 text-blue-400  shrink-0" />}
                        <div>
                          <p className="text-xs text-slate-300 font-medium">{t.counterparty ?? 'Unknown'}</p>
                          <p className="text-[10px] text-slate-500">
                            {t.country ?? '—'}
                            {t.timestamp && ` · ${String(t.timestamp).slice(0, 10)}`}
                          </p>
                        </div>
                      </div>
                      <div className="text-right shrink-0">
                        <p className="text-sm font-semibold text-white">
                          ${Number(t.amount).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </p>
                        <p className="text-[10px] text-slate-500">{t.currency}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 flex-wrap">
                      {t.fraud_type && (
                        <span className="text-[10px] px-2 py-0.5 rounded border bg-red-900/30 border-red-700/40 text-red-300 font-medium">
                          {t.fraud_type}
                        </span>
                      )}
                      {t.is_fraud && (
                        <span className="text-[10px] px-2 py-0.5 rounded border bg-orange-900/30 border-orange-700/40 text-orange-300">
                          FRAUD
                        </span>
                      )}
                      {t.outcome && (
                        <span className={`text-[10px] px-2 py-0.5 rounded border font-medium ${
                          t.outcome === 'DECLINE' ? 'bg-red-900/30 border-red-700/40 text-red-300'
                          : t.outcome === 'CHALLENGE' ? 'bg-yellow-900/30 border-yellow-700/40 text-yellow-300'
                          : 'bg-green-900/30 border-green-700/40 text-green-300'
                        }`}>{t.outcome}</span>
                      )}
                      <span className={`text-[10px] px-2 py-0.5 rounded border ${
                        t.direction === 'OUTBOUND'
                          ? 'bg-red-900/20 border-red-700/30 text-red-400'
                          : 'bg-blue-900/20 border-blue-700/30 text-blue-400'
                      }`}>
                        {t.direction}
                      </span>
                    </div>
                    <p className="text-[10px] font-mono text-slate-600">
                      {t.txn_id?.slice(0, 20)}…
                    </p>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Model info panel ───────────────────────────────────────────────────────────

interface EpochEntry { epoch: number; loss: number; val_auc: number }

function TrainingCurve({ entries }: { entries: EpochEntry[] }) {
  if (entries.length <= 1) return null
  return (
    <div>
      <p className="text-xs text-slate-500 mb-2">Val AUC over epochs</p>
      <div className="h-16 flex items-end gap-0.5">
        {entries.map(h => {
          const pct = Math.max(0, (h.val_auc - 0.5) / 0.5) * 100
          return (
            <div key={h.epoch} className="flex-1 flex flex-col items-center justify-end">
              <div
                className="w-full bg-purple-500/60 rounded-sm min-h-[2px]"
                style={{ height: `${Math.max(2, pct)}%` }}
                title={`Epoch ${h.epoch}: ${h.val_auc}`}
              />
            </div>
          )
        })}
      </div>
    </div>
  )
}

function ModelInfoPanel({ stats }: { stats: Record<string, unknown> }) {
  const roc = stats.roc_auc as number | undefined
  const ap  = stats.avg_precision as number | undefined
  const thr = stats.threshold as number | undefined
  const n   = stats.n_train as number | undefined
  const mr  = stats.mule_rate as number | undefined
  const trainHistory: EpochEntry[] = Array.isArray(stats.history)
    ? (stats.history as EpochEntry[])
    : []
  const lastAuc: number | undefined = trainHistory.length
    ? trainHistory[trainHistory.length - 1].val_auc
    : roc

  return (
    <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-5 space-y-4">
      <div className="flex items-center gap-2">
        <BarChart3 className="w-4 h-4 text-purple-400" />
        <h3 className="font-semibold text-white text-sm">GraphSAGE Training Stats</h3>
      </div>
      <div className="grid grid-cols-2 gap-3">
        {([
          ['ROC-AUC',       roc != null ? roc.toFixed(3) : '—'],
          ['Avg Precision', ap  != null ? ap.toFixed(3)  : '—'],
          ['Threshold',     thr != null ? thr.toFixed(3) : '—'],
          ['Train Nodes',   n   != null ? n.toLocaleString() : '—'],
          ['Mule Rate',     mr  != null ? `${(mr * 100).toFixed(1)}%` : '—'],
          ['Val AUC',       lastAuc != null ? lastAuc.toString() : '—'],
        ] as [string, string][]).map(([k, v]) => (
          <div key={k} className="bg-slate-900/50 rounded-lg p-2.5 text-center">
            <p className="text-xs text-slate-500">{k}</p>
            <p className="text-sm font-semibold text-purple-300 mt-0.5">{v}</p>
          </div>
        ))}
      </div>

      <TrainingCurve entries={trainHistory} />

      {stats.classification_report != null ? (
        <details className="text-xs">
          <summary className="text-slate-400 cursor-pointer hover:text-slate-200">
            Classification report
          </summary>
          <pre className="mt-2 p-2 bg-slate-900 rounded text-slate-300 overflow-x-auto text-[10px] leading-tight">
            {String(stats.classification_report)}
          </pre>
        </details>
      ) : null}
    </div>
  )
}

// ── Comparison panel ───────────────────────────────────────────────────────────

function ComparisonPanel() {
  const { data, isLoading } = useQuery({
    queryKey: ['graphsage-comparison'],
    queryFn:  () => getGraphSAGEComparison(200),
    staleTime: 60_000,
  })

  if (isLoading) return (
    <div className="flex items-center justify-center h-40 text-slate-400 text-sm">
      <RefreshCw className="w-4 h-4 animate-spin mr-2" /> Loading comparison…
    </div>
  )
  if (!data || !data.accounts.length) return (
    <div className="text-center text-slate-500 text-sm py-10">
      No comparison data. Train GraphSAGE and run the KNN scan first.
    </div>
  )

  const items = data.accounts.slice(0, 100)
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-sm text-slate-400">
          Model agreement: <span className="text-purple-300 font-semibold">{data.agreement_pct}%</span>
          &nbsp;({data.total} accounts)
        </p>
      </div>
      <div className="overflow-x-auto rounded-xl border border-slate-700/50">
        <table className="w-full text-sm">
          <thead className="bg-slate-800/80">
            <tr className="text-left text-xs text-slate-400">
              <th className="px-3 py-2">Account</th>
              <th className="px-3 py-2">Customer</th>
              <th className="px-3 py-2 text-center">GraphSAGE</th>
              <th className="px-3 py-2 text-center">KNN Anomaly</th>
              <th className="px-3 py-2 text-center">Agreement</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700/30">
            {items.map((a, i) => {
              const agree = !!a.sage_suspect === !!a.knn_suspect
              return (
                <tr key={i} className="hover:bg-slate-700/20 transition-colors">
                  <td className="px-3 py-2 font-mono text-xs text-slate-300">
                    {a.account_number ?? a.account_id.slice(0, 12)}
                  </td>
                  <td className="px-3 py-2 text-slate-300">{a.customer_name ?? '—'}</td>
                  <td className="px-3 py-2 text-center">
                    <span className={`font-semibold ${scoreColor(a.sage_score)}`}>
                      {a.sage_score.toFixed(1)}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-center">
                    <span className={`font-semibold ${scoreColor(a.knn_score)}`}>
                      {a.knn_score.toFixed(1)}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-center">
                    {agree
                      ? <CheckCircle2 className="w-4 h-4 text-green-400 mx-auto" />
                      : <XCircle     className="w-4 h-4 text-red-400   mx-auto" />}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────────

export default function GraphSAGEDetection() {
  const [page, setPage]           = useState(1)
  const [pageSize]                = useState(50)
  const [selected, setSelected]   = useState<GraphSAGESuspect | null>(null)
  const [activeTab, setActiveTab] = useState<'suspects' | 'comparison' | 'howto'>('suspects')
  const [trainParams, setTrainParams] = useState({ max_nodes: 50000, epochs: 60 })

  const qc = useQueryClient()

  const { data: summary, isLoading: summaryLoading } = useQuery<GraphSAGESummary>({
    queryKey: ['graphsage-summary'],
    queryFn:  getGraphSAGESummary,
    staleTime: 30_000,
  })

  const { data: suspects, isLoading: suspectsLoading } = useQuery({
    queryKey: ['graphsage-suspects', page, pageSize],
    queryFn:  () => listGraphSAGESuspects(page, pageSize),
    placeholderData: keepPreviousData,
    staleTime: 30_000,
  })

  const trainMutation = useMutation({
    mutationFn: () => trainGraphSAGE({
      max_nodes: trainParams.max_nodes,
      epochs:    trainParams.epochs,
    }),
    onSuccess: () => {
      setTimeout(() => {
        qc.invalidateQueries({ queryKey: ['graphsage-summary'] })
        qc.invalidateQueries({ queryKey: ['graphsage-suspects'] })
        qc.invalidateQueries({ queryKey: ['graphsage-comparison'] })
      }, 2000)
    },
  })

  const tabs = [
    { id: 'suspects',   label: 'Suspect Accounts', icon: AlertTriangle },
    { id: 'comparison', label: 'Model Comparison',  icon: GitCompare   },
    { id: 'howto',      label: 'How It Works',      icon: Info         },
  ] as const

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/20 rounded-lg">
            <Network className="w-6 h-6 text-purple-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">GraphSAGE Mule Detection</h1>
            <p className="text-slate-400 text-sm mt-0.5">
              Graph neural network trained on account transaction neighbourhoods
            </p>
          </div>
        </div>

        {/* Train button + params */}
        <div className="flex items-center gap-3">
          <div className="flex gap-2 text-sm">
            <div className="flex items-center gap-1.5">
              <label className="text-slate-400 text-xs">Nodes</label>
              <select
                value={trainParams.max_nodes}
                onChange={e => setTrainParams(p => ({ ...p, max_nodes: +e.target.value }))}
                className="bg-slate-700 border border-slate-600 rounded px-2 py-1 text-slate-200 text-xs"
              >
                {[10000, 25000, 50000, 100000].map(v => (
                  <option key={v} value={v}>{v.toLocaleString()}</option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-1.5">
              <label className="text-slate-400 text-xs">Epochs</label>
              <select
                value={trainParams.epochs}
                onChange={e => setTrainParams(p => ({ ...p, epochs: +e.target.value }))}
                className="bg-slate-700 border border-slate-600 rounded px-2 py-1 text-slate-200 text-xs"
              >
                {[20, 40, 60, 100].map(v => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </div>
          </div>
          <button
            onClick={() => trainMutation.mutate()}
            disabled={trainMutation.isPending}
            className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-sm font-medium disabled:opacity-50 transition-colors"
          >
            <Brain className={`w-4 h-4 ${trainMutation.isPending ? 'animate-pulse' : ''}`} />
            {trainMutation.isPending ? 'Training…' : 'Train GraphSAGE'}
          </button>
        </div>
      </div>

      {trainMutation.isPending && (
        <div className="flex items-center gap-3 p-4 bg-purple-900/30 border border-purple-700/40 rounded-xl text-sm text-purple-200">
          <RefreshCw className="w-4 h-4 animate-spin" />
          GraphSAGE training running in background. This may take 2–5 minutes. The page will auto-refresh when done.
        </div>
      )}

      {trainMutation.isSuccess && (
        <div className="flex items-center gap-3 p-4 bg-green-900/30 border border-green-700/40 rounded-xl text-sm text-green-200">
          <CheckCircle2 className="w-4 h-4" />
          Training started! Scores will appear once the background task completes.
        </div>
      )}

      {/* Summary stats */}
      {summaryLoading ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-5 animate-pulse h-24" />
          ))}
        </div>
      ) : summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard
            label="Total Accounts"
            value={(summary.total_accounts ?? 0).toLocaleString()}
            sub={`${summary.coverage_pct ?? 0}% scored`}
          />
          <StatCard
            label="GraphSAGE Suspects"
            value={(summary.graphsage_suspects ?? 0).toLocaleString()}
            color="text-red-400"
            sub="score ≥ threshold"
          />
          <StatCard
            label="High Confidence ≥80"
            value={(summary.high_confidence ?? 0).toLocaleString()}
            color="text-orange-400"
          />
          <StatCard
            label="Flagged by Both"
            value={(summary.flagged_by_both ?? 0).toLocaleString()}
            color="text-purple-400"
            sub="GraphSAGE + KNN"
          />
        </div>
      )}

      {/* Model status bar */}
      <div className="flex items-center gap-4 p-3 bg-slate-800/40 rounded-xl border border-slate-700/30 text-sm">
        <div className="flex items-center gap-2">
          {summary?.model_trained
            ? <Zap className="w-4 h-4 text-green-400" />
            : <XCircle className="w-4 h-4 text-slate-500" />}
          <span className={summary?.model_trained ? 'text-green-400' : 'text-slate-500'}>
            GraphSAGE {summary?.model_trained ? 'trained' : 'not trained'}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-blue-400" />
          <span className="text-slate-400">
            Feature flag: <span className={summary?.feature_flag_on ? 'text-green-400' : 'text-red-400'}>
              {summary?.feature_flag_on ? 'ON' : 'OFF'}
            </span>
            <span className="text-slate-500 ml-2 text-xs">(env: ENABLE_GRAPHSAGE)</span>
          </span>
        </div>
        {summary?.model_trained && summary.training_stats?.roc_auc != null && (
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-purple-400" />
            <span className="text-slate-400">
              ROC-AUC: <span className="text-purple-300 font-semibold">
                {Number(summary.training_stats.roc_auc).toFixed(4)}
              </span>
            </span>
          </div>
        )}
      </div>

      {/* Two-column layout: suspects/tabs + model info */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2 space-y-4">
          {/* Tabs */}
          <div className="flex gap-1 bg-slate-800/60 p-1 rounded-xl border border-slate-700/50">
            {tabs.map(t => (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                className={`flex items-center gap-2 flex-1 justify-center py-2 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === t.id
                    ? 'bg-purple-600 text-white'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <t.icon className="w-4 h-4" />
                {t.label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          {activeTab === 'suspects' && (
            <div className="space-y-3">
              {suspectsLoading ? (
                <div className="flex items-center justify-center h-32 text-slate-400 text-sm">
                  <RefreshCw className="w-4 h-4 animate-spin mr-2" /> Loading suspects…
                </div>
              ) : !suspects?.suspects.length ? (
                <div className="text-center text-slate-500 py-12">
                  <Network className="w-12 h-12 mx-auto mb-3 opacity-30" />
                  <p>No suspects scored yet.</p>
                  <p className="text-xs mt-1">Click <strong>Train GraphSAGE</strong> to run the model.</p>
                </div>
              ) : (
                <>
                  <div className="overflow-x-auto rounded-xl border border-slate-700/50">
                    <table className="w-full text-sm">
                      <thead className="bg-slate-800/80">
                        <tr className="text-left text-xs text-slate-400">
                          <th className="px-3 py-2.5">Account</th>
                          <th className="px-3 py-2.5">Customer</th>
                          <th className="px-3 py-2.5">Country</th>
                          <th className="px-3 py-2.5 text-center">SAGE Score</th>
                          <th className="px-3 py-2.5 text-center">KNN Score</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-700/30">
                        {suspects.suspects.map(s => (
                          <tr
                            key={s.account_id}
                            className="hover:bg-slate-700/20 transition-colors cursor-pointer"
                            onClick={() => setSelected(s)}
                          >
                            <td className="px-3 py-2.5 font-mono text-xs text-slate-300">
                              {s.account_number ?? s.account_id.slice(0, 14)}
                            </td>
                            <td className="px-3 py-2.5 text-slate-300">
                              {s.customer_name ?? <span className="text-slate-600">—</span>}
                            </td>
                            <td className="px-3 py-2.5 text-slate-400 text-xs">
                              {s.customer_country ?? '—'}
                            </td>
                            <td className="px-3 py-2.5 text-center">
                              <span className={`font-semibold ${scoreColor(s.graphsage_score)}`}>
                                {s.graphsage_score.toFixed(1)}
                              </span>
                            </td>
                            <td className="px-3 py-2.5 text-center">
                              {s.knn_anomaly_score != null
                                ? <span className={`font-semibold ${scoreColor(s.knn_anomaly_score)}`}>
                                    {s.knn_anomaly_score.toFixed(1)}
                                  </span>
                                : <span className="text-slate-600 text-xs">n/a</span>}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Pagination */}
                  {suspects.total_pages > 1 && (
                    <div className="flex items-center justify-between text-sm text-slate-400">
                      <span>
                        Page {suspects.page} / {suspects.total_pages}
                        &nbsp;·&nbsp; {suspects.total.toLocaleString()} suspects
                      </span>
                      <div className="flex gap-2">
                        <button
                          onClick={() => setPage(p => Math.max(1, p - 1))}
                          disabled={page <= 1}
                          className="p-1.5 rounded-lg bg-slate-700 hover:bg-slate-600 disabled:opacity-40 disabled:cursor-not-allowed"
                        >
                          <ChevronLeft className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => setPage(p => Math.min(suspects.total_pages, p + 1))}
                          disabled={page >= suspects.total_pages}
                          className="p-1.5 rounded-lg bg-slate-700 hover:bg-slate-600 disabled:opacity-40 disabled:cursor-not-allowed"
                        >
                          <ChevronRight className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {activeTab === 'comparison' && <ComparisonPanel />}

          {activeTab === 'howto' && (
            <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-5 space-y-4 text-sm text-slate-300">
              <h3 className="font-semibold text-white flex items-center gap-2">
                <Network className="w-4 h-4 text-purple-400" /> GraphSAGE Architecture
              </h3>
              <p>
                GraphSAGE (<em>Graph Sample and Aggregate</em>) is a graph neural network that learns
                embeddings by sampling and aggregating each node's local neighbourhood. Unlike
                transductive methods, it is <strong>inductive</strong> — it generalises to unseen accounts.
              </p>
              <div className="space-y-2">
                {[
                  ['Graph construction', 'Account nodes, edges from initiated → credited_to transaction relationships'],
                  ['Node features (16)',  'Txn count, avg amount, fraud ratio, pattern count, pass-through ratio, PEP/sanctions flags, country risk, KYC level, account type'],
                  ['Layer 1',            'Mean-aggregate 1-hop neighbours → ReLU → L2-normalise → 64-dim embedding'],
                  ['Layer 2',            'Mean-aggregate 2-hop neighbours → ReLU → L2-normalise → 32-dim embedding'],
                  ['Output',             'Sigmoid classifier: mule probability 0–1 (×100 = mule score)'],
                  ['Training',           'Mini-batch Adam SGD, binary cross-entropy, class-weighted for ~15% mule prevalence'],
                  ['Mule label',         'Account has ≥30% fraud transactions OR ≥2 fraud pattern transactions (SMURFING/LAYERING/STRUCTURING)'],
                ].map(([k, v]) => (
                  <div key={k} className="flex gap-3">
                    <span className="text-purple-400 font-medium min-w-[140px]">{k}</span>
                    <span className="text-slate-400">{v}</span>
                  </div>
                ))}
              </div>
              <div className="mt-4 p-3 bg-purple-900/20 border border-purple-700/30 rounded-lg text-xs text-purple-200">
                <strong>KNN vs GraphSAGE:</strong> KNN anomaly detection scores individual transaction
                feature vectors in isolation. GraphSAGE captures <em>structural patterns</em> in the
                transaction graph — mule accounts often sit at the junction of multiple unrelated senders,
                which is invisible to KNN but visible to SAGE.
              </div>
            </div>
          )}
        </div>

        {/* Right column: model info */}
        <div>
          {summary?.model_trained && Object.keys(summary.training_stats ?? {}).length > 0
            ? <ModelInfoPanel stats={summary.training_stats} />
            : (
              <div className="bg-slate-800/60 border border-slate-700/50 border-dashed rounded-xl p-6 text-center text-slate-500 space-y-3">
                <Brain className="w-10 h-10 mx-auto opacity-30" />
                <p className="text-sm">No model trained yet.</p>
                <p className="text-xs">
                  Set node count & epochs above, then click{' '}
                  <strong className="text-purple-400">Train GraphSAGE</strong>.
                </p>
              </div>
            )}
        </div>
      </div>

      {/* Account detail drawer */}
      {selected && <AccountDrawer account={selected} onClose={() => setSelected(null)} />}
    </div>
  )
}
