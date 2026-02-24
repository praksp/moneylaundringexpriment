import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Database, AlertTriangle, RefreshCw, ChevronRight,
  ChevronDown, ShieldAlert, Activity, TrendingUp, Users,
  ArrowUpRight, ArrowDownLeft, Info,
} from 'lucide-react'
import { getHighRiskAccounts, computeFeatureSnapshot } from '../api/client'
import clsx from 'clsx'

// ── Types ─────────────────────────────────────────────────────────────────────

interface HighRiskAccount {
  account_id: string
  account_number: string | null
  customer_id: string | null
  customer_name: string | null
  mule_score: number
  is_likely_mule: boolean
  avg_risk_score: number
  turnover_ratio: number
  tor_activity: boolean
  unique_senders: number | null
  structuring_count: number | null
  out_volume: number | null
  in_volume: number | null
  eval_count: number | null
  decline_count: number | null
  pep_flag: boolean
  risk_tier: string | null
  computed_at: string | null
  source: 'feature_snapshot' | 'graphsage_knn' | 'fraud_ratio' | string
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function scoreColor(s: number) {
  return s >= 70 ? 'text-red-400' : s >= 40 ? 'text-amber-400' : 'text-emerald-400'
}

function SourceBadge({ source }: { source: string }) {
  const map: Record<string, { label: string; cls: string }> = {
    feature_snapshot: { label: 'Feature Snapshot', cls: 'bg-purple-900/40 text-purple-300 border-purple-700/40' },
    graphsage_knn:    { label: 'GraphSAGE + KNN',  cls: 'bg-blue-900/40   text-blue-300   border-blue-700/40'   },
    fraud_ratio:      { label: 'Fraud Ratio',       cls: 'bg-red-900/40    text-red-300    border-red-700/40'    },
  }
  const s = map[source] ?? { label: source, cls: 'bg-slate-700 text-slate-300 border-slate-600' }
  return (
    <span className={clsx('text-[10px] px-2 py-0.5 rounded border font-medium', s.cls)}>
      {s.label}
    </span>
  )
}

function MiniBar({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = Math.min(100, Math.max(0, (value / Math.max(max, 1)) * 100))
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div className={clsx('h-full rounded-full', color)} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

// ── Feature indicator pill ─────────────────────────────────────────────────────

function FeaturePill({
  label, triggered, value,
}: { label: string; triggered: boolean; value?: string }) {
  return (
    <div className={clsx(
      'flex items-center gap-1.5 text-xs px-2 py-1 rounded-lg border',
      triggered
        ? 'bg-red-900/30 border-red-700/40 text-red-300'
        : 'bg-slate-800/50 border-slate-700/30 text-slate-500',
    )}>
      {triggered
        ? <AlertTriangle className="w-3 h-3 shrink-0" />
        : <span className="w-3 h-3 rounded-full border border-slate-600 inline-block shrink-0" />}
      <span>{label}</span>
      {value && <span className="font-semibold ml-0.5">{value}</span>}
    </div>
  )
}

// ── Expanded row detail ────────────────────────────────────────────────────────

function AccountDetail({
  row,
  onRecompute,
  recomputing,
}: {
  row: HighRiskAccount
  onRecompute: () => void
  recomputing: boolean
}) {
  const tr = Number(row.turnover_ratio)
  const outV = Number(row.out_volume ?? 0)
  const inV  = Number(row.in_volume  ?? 0)
  const fraudCount = Number(row.decline_count ?? 0)
  const totalTxns  = Number(row.eval_count    ?? 0)
  const fraudRatio = totalTxns > 0 ? (fraudCount / totalTxns) * 100 : 0

  const indicators = [
    {
      label: 'Fraud ratio ≥30%',
      triggered: fraudRatio >= 30,
      value: `${fraudRatio.toFixed(1)}%`,
    },
    {
      label: 'Pass-through account',
      triggered: tr >= 0.7 && tr <= 5 && inV > 1000,
      value: `${tr.toFixed(2)}×`,
    },
    {
      label: 'High sender diversity',
      triggered: (row.unique_senders ?? 0) > 10,
      value: `${row.unique_senders ?? '—'} senders`,
    },
    {
      label: 'Structuring pattern',
      triggered: (row.structuring_count ?? 0) >= 2,
      value: `${row.structuring_count ?? '—'} txns`,
    },
    {
      label: 'Tor / VPN activity',
      triggered: row.tor_activity,
      value: row.tor_activity ? 'YES' : 'no',
    },
    {
      label: 'PEP / Sanctions',
      triggered: row.pep_flag,
      value: row.pep_flag ? 'YES' : 'no',
    },
    {
      label: 'High risk tier',
      triggered: ['HIGH', 'CRITICAL'].includes(row.risk_tier ?? ''),
      value: row.risk_tier ?? '—',
    },
  ]

  return (
    <tr>
      <td colSpan={8} className="px-5 py-4 bg-slate-900/70 border-b border-slate-700/40">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

          {/* Volume stats */}
          <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/40 space-y-3">
            <p className="text-xs font-medium text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5" /> Transaction Volumes
            </p>
            {[
              { label: 'Outbound volume',   val: `$${outV.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, icon: ArrowUpRight,   cls: 'text-red-400' },
              { label: 'Inbound volume',    val: `$${inV.toLocaleString(undefined,  { maximumFractionDigits: 0 })}`, icon: ArrowDownLeft,  cls: 'text-green-400' },
              { label: 'Turnover ratio',    val: `${tr.toFixed(2)}×`,       icon: TrendingUp,    cls: 'text-amber-400' },
              { label: 'Fraud txns / total',val: `${fraudCount} / ${totalTxns}`, icon: ShieldAlert, cls: 'text-red-400' },
              { label: 'Unique senders',    val: String(row.unique_senders ?? '—'), icon: Users, cls: 'text-blue-400' },
            ].map(({ label, val, icon: Icon, cls }) => (
              <div key={label} className="flex items-center justify-between text-sm">
                <span className="text-slate-500 flex items-center gap-1">
                  <Icon className={clsx('w-3.5 h-3.5', cls)} />
                  {label}
                </span>
                <span className="text-slate-200 font-medium">{val}</span>
              </div>
            ))}
          </div>

          {/* Triggered indicators */}
          <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/40">
            <p className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-1.5">
              <AlertTriangle className="w-3.5 h-3.5" /> Risk Indicators
            </p>
            <div className="flex flex-wrap gap-1.5">
              {indicators.map(ind => (
                <FeaturePill
                  key={ind.label}
                  label={ind.label}
                  triggered={ind.triggered}
                  value={ind.value}
                />
              ))}
            </div>
            <p className="text-xs text-slate-600 mt-3">
              {indicators.filter(i => i.triggered).length} of {indicators.length} indicators triggered
            </p>
          </div>

          {/* Mule score breakdown */}
          <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/40 space-y-3">
            <p className="text-xs font-medium text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5" /> Detection Score
            </p>
            <div className="text-center py-2">
              <p className={clsx('text-4xl font-bold', scoreColor(row.mule_score))}>
                {row.mule_score}
              </p>
              <p className="text-xs text-slate-500 mt-1">Composite mule score (0–100)</p>
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-slate-500">Fraud ratio</span>
                <span className={fraudRatio >= 30 ? 'text-red-400 font-semibold' : 'text-slate-400'}>
                  {fraudRatio.toFixed(1)}%
                  {fraudRatio >= 30 && ' ⚡'}
                </span>
              </div>
              <MiniBar value={fraudRatio} max={100} color={fraudRatio >= 30 ? 'bg-red-500' : 'bg-slate-500'} />
              <div className="flex justify-between mt-1">
                <span className="text-slate-500">Turnover</span>
                <span className={tr >= 0.7 && tr <= 5 ? 'text-amber-400 font-semibold' : 'text-slate-400'}>
                  {tr.toFixed(2)}×
                  {(tr >= 0.7 && tr <= 5) && ' ⚡'}
                </span>
              </div>
              <MiniBar value={Math.min(tr, 5)} max={5} color={(tr >= 0.7 && tr <= 5) ? 'bg-amber-500' : 'bg-slate-500'} />
            </div>

            {row.customer_id && (
              <button
                onClick={onRecompute}
                disabled={recomputing}
                className="w-full mt-2 text-xs text-blue-400 hover:text-blue-300 border border-blue-700/40 rounded-lg py-1.5 hover:bg-blue-900/20 transition-colors disabled:opacity-50"
              >
                {recomputing ? 'Computing…' : '↺ Compute full feature snapshot'}
              </button>
            )}
            <p className="text-[10px] text-slate-600 text-center">
              Source: <span className="text-slate-500">{row.source?.replace('_', ' ')}</span>
              {row.computed_at && ` · ${String(row.computed_at).slice(0, 10)}`}
            </p>
          </div>
        </div>
      </td>
    </tr>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────────

export default function FeatureStore() {
  const qc = useQueryClient()
  const [expanded, setExpanded] = useState<string | null>(null)

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['high-risk-accounts'],
    queryFn: getHighRiskAccounts,
  })

  const recomputeMutation = useMutation({
    mutationFn: ({ cid, aid }: { cid: string; aid: string }) =>
      computeFeatureSnapshot(cid, aid),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['high-risk-accounts'] }),
  })

  const accounts = ((data?.accounts || []) as HighRiskAccount[])
    .sort((a, b) => Number(b.mule_score) - Number(a.mule_score))

  const criticalCount = accounts.filter(a => Number(a.mule_score) >= 70).length
  const warningCount  = accounts.filter(a => Number(a.mule_score) >= 40 && Number(a.mule_score) < 70).length

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Feature Store</h1>
          <p className="text-slate-400 text-sm mt-1">
            Mule indicators and behavioral features — auto-populated from GraphSAGE, KNN anomaly, and fraud transaction data.
          </p>
        </div>
        <button
          onClick={() => refetch()}
          className="flex items-center gap-2 px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm rounded-lg border border-slate-700 transition-colors"
        >
          <RefreshCw size={14} />
          Refresh
        </button>
      </div>

      {/* Summary cards */}
      {!isLoading && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: 'Total flagged',   value: accounts.length,  color: 'text-white',       sub: 'high-risk accounts' },
            { label: 'Critical ≥70',   value: criticalCount,    color: 'text-red-400',     sub: 'immediate review' },
            { label: 'Warning 40–69',  value: warningCount,     color: 'text-amber-400',   sub: 'monitor closely' },
            { label: 'With snapshots', value: accounts.filter(a => a.source === 'feature_snapshot').length,
              color: 'text-purple-400', sub: 'fully computed' },
          ].map(c => (
            <div key={c.label} className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4">
              <p className="text-xs text-slate-400">{c.label}</p>
              <p className={clsx('text-3xl font-bold mt-1', c.color)}>{c.value}</p>
              <p className="text-xs text-slate-600 mt-0.5">{c.sub}</p>
            </div>
          ))}
        </div>
      )}

      {/* Info banner about data sources */}
      <div className="flex items-start gap-3 p-3 bg-slate-800/40 border border-slate-700/30 rounded-xl text-xs text-slate-400">
        <Info className="w-4 h-4 shrink-0 mt-0.5 text-blue-400" />
        <span>
          Accounts are auto-populated from three sources:
          <span className="text-purple-300 mx-1">FeatureSnapshot</span> (manually computed),
          <span className="text-blue-300 mx-1">GraphSAGE + KNN</span> (model-flagged suspects),
          and <span className="text-red-300 mx-1">Fraud Ratio</span> (accounts with ≥30% fraud transactions).
          Click any row to expand full indicator details. Use "Compute full feature snapshot" to generate a persistent snapshot.
        </span>
      </div>

      {/* Feature groups reference */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[
          { label: 'Mule Indicators',  desc: 'Turnover ratio, pass-through, senders',  color: 'border-red-500/30' },
          { label: 'Behavioral',       desc: 'Velocity, volume, structuring patterns',  color: 'border-amber-500/30' },
          { label: 'Network',          desc: 'Counterparties, fraud type distribution', color: 'border-blue-500/30' },
          { label: 'Risk History',     desc: 'GraphSAGE score, KNN anomaly, tier',      color: 'border-purple-500/30' },
        ].map(g => (
          <div key={g.label} className={clsx('bg-slate-900 rounded-lg border p-3', g.color)}>
            <p className="text-sm font-medium text-white">{g.label}</p>
            <p className="text-xs text-slate-500 mt-0.5">{g.desc}</p>
          </div>
        ))}
      </div>

      {/* Main table */}
      {isLoading ? (
        <div className="text-center py-16 text-slate-500">
          <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2" />
          Loading feature store…
        </div>
      ) : accounts.length === 0 ? (
        <div className="text-center py-16 bg-slate-900 border border-dashed border-slate-700 rounded-xl">
          <Database size={36} className="text-slate-600 mx-auto mb-3" />
          <p className="text-slate-400 text-sm font-medium">No high-risk accounts found</p>
          <p className="text-slate-500 text-xs mt-1">
            Train the GraphSAGE model or run the KNN anomaly scan to auto-populate this store.
          </p>
        </div>
      ) : (
        <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
          <div className="px-5 py-3 border-b border-slate-800 flex items-center justify-between">
            <p className="text-sm font-semibold text-slate-300">
              {accounts.length} High-Risk Account{accounts.length !== 1 ? 's' : ''}
            </p>
            <span className="text-xs text-slate-500">
              Click any row to expand indicator details
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-slate-400 border-b border-slate-800">
                  <th className="text-left py-3 px-4 w-6"></th>
                  <th className="text-left py-3 px-4">Customer</th>
                  <th className="text-left py-3 px-4">Account</th>
                  <th className="text-left py-3 px-4">Mule Score</th>
                  <th className="text-left py-3 px-4">Turnover</th>
                  <th className="text-left py-3 px-4">Fraud txns</th>
                  <th className="text-left py-3 px-4">Source</th>
                  <th className="text-left py-3 px-4">Date</th>
                </tr>
              </thead>
              <tbody>
                {accounts.map((row) => {
                  const isOpen = expanded === row.account_id
                  const score = Number(row.mule_score)
                  const fraudCount = Number(row.decline_count ?? 0)
                  const totalTxns  = Number(row.eval_count    ?? 0)
                  const fraudRatio = totalTxns > 0 ? (fraudCount / totalTxns) * 100 : 0

                  return (
                    <>
                      <tr
                        key={row.account_id}
                        className={clsx(
                          'border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors cursor-pointer',
                          isOpen && 'bg-slate-800/20',
                        )}
                        onClick={() => setExpanded(isOpen ? null : row.account_id)}
                      >
                        <td className="py-3 px-4">
                          {isOpen
                            ? <ChevronDown className="w-4 h-4 text-slate-400" />
                            : <ChevronRight className="w-4 h-4 text-slate-500" />}
                        </td>
                        <td className="py-3 px-4">
                          <p className="text-white text-xs font-medium">{row.customer_name ?? '—'}</p>
                          <p className="text-slate-500 text-xs font-mono">
                            {row.customer_id ? row.customer_id.slice(0, 8) + '…' : '—'}
                          </p>
                        </td>
                        <td className="py-3 px-4 font-mono text-xs text-slate-300">
                          {row.account_number ?? row.account_id?.slice(0, 12) ?? '—'}
                        </td>
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-2">
                            <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                              <div
                                className={clsx(
                                  'h-full rounded-full',
                                  score >= 70 ? 'bg-red-500' : score >= 40 ? 'bg-amber-500' : 'bg-emerald-500',
                                )}
                                style={{ width: `${Math.min(score, 100)}%` }}
                              />
                            </div>
                            <span className={clsx('text-xs font-bold', scoreColor(score))}>{score}</span>
                          </div>
                          {row.is_likely_mule && (
                            <span className="text-xs text-red-400 flex items-center gap-1 mt-0.5">
                              <AlertTriangle size={10} /> Likely Mule
                            </span>
                          )}
                        </td>
                        <td className="py-3 px-4 text-xs text-slate-300">
                          {Number(row.turnover_ratio).toFixed(2)}×
                        </td>
                        <td className="py-3 px-4 text-xs">
                          {totalTxns > 0 ? (
                            <span className={fraudRatio >= 30 ? 'text-red-400 font-semibold' : 'text-slate-400'}>
                              {fraudCount}/{totalTxns}
                              <span className="text-slate-500 ml-1">({fraudRatio.toFixed(0)}%)</span>
                            </span>
                          ) : (
                            <span className="text-slate-600">—</span>
                          )}
                        </td>
                        <td className="py-3 px-4">
                          <SourceBadge source={row.source} />
                        </td>
                        <td className="py-3 px-4 text-xs text-slate-500">
                          {row.computed_at ? String(row.computed_at).slice(0, 10) : '—'}
                        </td>
                      </tr>
                      {isOpen && (
                        <AccountDetail
                          key={`${row.account_id}-detail`}
                          row={row}
                          onRecompute={() =>
                            recomputeMutation.mutate({
                              cid: row.customer_id ?? '',
                              aid: row.account_id,
                            })
                          }
                          recomputing={recomputeMutation.isPending}
                        />
                      )}
                    </>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Feature schema reference */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-5">
        <h3 className="text-sm font-semibold text-slate-300 mb-3">Feature Schema Reference</h3>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-x-8 gap-y-2 text-xs">
          {[
            ['mule_score',             '0–100 composite mule likelihood'],
            ['turnover_ratio_30d',     'outbound ÷ inbound volume'],
            ['unique_senders_30d',     'distinct accounts sending funds in'],
            ['structuring_count_30d',  'txns in $9k–$9.99k band (structuring)'],
            ['has_tor_activity',       'any transaction via Tor exit node'],
            ['is_dormant',             'last_active > 90 days ago'],
            ['account_age_days',       'days since account creation'],
            ['avg_risk_score_30d',     'mean Bayesian+ML score'],
            ['decline_rate_30d',       'fraction of evaluations DECLINED'],
            ['is_pass_through',        'outbound ≈ inbound volume (relay account)'],
            ['outbound_volume_30d',    'total USD sent out in 30 days'],
            ['fraud_txn_ratio',        'fraction of transactions marked is_fraud'],
          ].map(([feat, desc]) => (
            <div key={feat} className="py-1 border-b border-slate-800/50">
              <code className="text-blue-400">{feat}</code>
              <p className="text-slate-500 mt-0.5">{desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
