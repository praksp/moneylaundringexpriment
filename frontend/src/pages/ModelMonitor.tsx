import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip,
  CartesianGrid, ResponsiveContainer, Legend, PieChart, Pie, Cell,
} from 'recharts'
import {
  Activity, AlertTriangle, CheckCircle, XCircle,
  RefreshCw, TrendingUp, Clock, GitBranch, Zap, ArrowUpCircle, Archive,
} from 'lucide-react'
import {
  getMonitoringSummary, getOutcomeTrend, getScoreDistribution,
  getTopRiskFactors, computeDrift, getDriftHistory,
  listModelVersions, getCurrentVersions, getTrainingStatus,
  triggerIncrementalTrain, promoteVersion, retireVersion,
  type ModelVersionMeta,
  type VersionsResponse, type CurrentVersionsResponse, type TrainingStatusResponse,
} from '../api/client'
import clsx from 'clsx'

const OUTCOME_COLORS = { ALLOW: '#10b981', CHALLENGE: '#f59e0b', DECLINE: '#ef4444' }
const ALERT_COLORS = { OK: 'text-emerald-400', WARNING: 'text-amber-400', CRITICAL: 'text-red-400', UNKNOWN: 'text-slate-400' }
const ALERT_BG = { OK: 'bg-emerald-500/10 border-emerald-500/20', WARNING: 'bg-amber-500/10 border-amber-500/20', CRITICAL: 'bg-red-500/10 border-red-500/20', UNKNOWN: 'bg-slate-800 border-slate-700' }

function StatCard({ title, value, sub, icon: Icon, color = 'text-white' }: {
  title: string; value: string | number; sub?: string
  icon: React.ElementType; color?: string
}) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">{title}</p>
        <Icon size={16} className="text-slate-500" />
      </div>
      <p className={clsx('text-2xl font-bold', color)}>{value}</p>
      {sub && <p className="text-xs text-slate-500 mt-1">{sub}</p>}
    </div>
  )
}

export default function ModelMonitor() {
  const qc = useQueryClient()
  const summary = useQuery({ queryKey: ['monitoring-summary'], queryFn: getMonitoringSummary, refetchInterval: 30_000 })
  const trend = useQuery({ queryKey: ['outcome-trend'], queryFn: () => getOutcomeTrend(14) })
  const scoreDist = useQuery({ queryKey: ['score-dist'], queryFn: () => getScoreDistribution(7) })
  const riskFactors = useQuery({ queryKey: ['risk-factors'], queryFn: () => getTopRiskFactors(7) })
  const driftHistory = useQuery({ queryKey: ['drift-history'], queryFn: getDriftHistory })

  // Model versioning
  const versionsQ  = useQuery({ queryKey: ['model-versions'], queryFn: listModelVersions })
  const currentQ   = useQuery({ queryKey: ['model-versions-current'], queryFn: getCurrentVersions })
  const trainJobQ  = useQuery({
    queryKey: ['train-status'],
    queryFn: getTrainingStatus,
    refetchInterval: (data: unknown) => {
      const d = data as { running?: boolean } | undefined
      return d?.running ? 3000 : false
    },
  })

  const driftMutation = useMutation({
    mutationFn: computeDrift,
    onSuccess: () => {
      driftHistory.refetch()
      summary.refetch()
    },
  })

  const trainMutation = useMutation({
    mutationFn: ({ force, autoPromote }: { force: boolean; autoPromote: boolean }) =>
      triggerIncrementalTrain(force, autoPromote),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['train-status'] })
    },
  })

  const promoteMutation = useMutation({
    mutationFn: (versionId: string) => promoteVersion(versionId),
    onMutate: async (versionId) => {
      await qc.cancelQueries({ queryKey: ['model-versions'] })
      await qc.cancelQueries({ queryKey: ['model-versions-current'] })
      const prevVersions = qc.getQueryData(['model-versions'])
      const prevCurrent = qc.getQueryData(['model-versions-current'])

      qc.setQueryData(['model-versions'], (old: any) => {
        if (!old || !old.versions) return old
        return {
          ...old,
          versions: old.versions.map((v: any) => {
            if (v.version_id === versionId) return { ...v, status: 'baseline' }
            if (v.status === 'baseline') return { ...v, status: 'retired' }
            return v
          })
        }
      })

      return { prevVersions, prevCurrent }
    },
    onError: (_err, _variables, context) => {
      if (context?.prevVersions) qc.setQueryData(['model-versions'], context.prevVersions)
      if (context?.prevCurrent) qc.setQueryData(['model-versions-current'], context.prevCurrent)
    },
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ['model-versions'] })
      qc.invalidateQueries({ queryKey: ['model-versions-current'] })
    },
  })

  const retireMutation = useMutation({
    mutationFn: (versionId: string) => retireVersion(versionId),
    onMutate: async (versionId) => {
      await qc.cancelQueries({ queryKey: ['model-versions'] })
      const prevVersions = qc.getQueryData(['model-versions'])

      qc.setQueryData(['model-versions'], (old: any) => {
        if (!old || !old.versions) return old
        return {
          ...old,
          versions: old.versions.map((v: any) =>
            v.version_id === versionId ? { ...v, status: 'retired' } : v
          )
        }
      })

      return { prevVersions }
    },
    onError: (_err, _variables, context) => {
      if (context?.prevVersions) qc.setQueryData(['model-versions'], context.prevVersions)
    },
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ['model-versions'] })
    },
  })

  const s = summary.data
  const alert = (s?.latest_drift_alert || 'UNKNOWN') as keyof typeof ALERT_COLORS

  const outcomePieData = s ? [
    { name: 'Allow', value: s.outcome_7d.ALLOW, color: OUTCOME_COLORS.ALLOW },
    { name: 'Challenge', value: s.outcome_7d.CHALLENGE, color: OUTCOME_COLORS.CHALLENGE },
    { name: 'Decline', value: s.outcome_7d.DECLINE, color: OUTCOME_COLORS.DECLINE },
  ] : []

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Model Monitor</h1>
          <p className="text-slate-400 text-sm mt-1">Drift detection · Performance metrics · Prediction analytics</p>
        </div>
        <button
          onClick={() => driftMutation.mutate()}
          disabled={driftMutation.isPending}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 text-white text-sm font-medium rounded-lg transition-colors"
        >
          <RefreshCw size={14} className={driftMutation.isPending ? 'animate-spin' : ''} />
          {driftMutation.isPending ? 'Computing…' : 'Run Drift Check'}
        </button>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Evaluations" value={s?.total_evaluations?.toLocaleString() ?? '—'}
          sub={`${s?.evaluations_24h ?? 0} in last 24h`} icon={Activity}
        />
        <StatCard
          title="Avg Risk Score (7d)" value={s?.avg_score_7d ?? '—'}
          sub="0–999 scale" icon={TrendingUp}
          color={s?.avg_score_7d
            ? s.avg_score_7d <= 399 ? 'text-emerald-400' : s.avg_score_7d <= 699 ? 'text-amber-400' : 'text-red-400'
            : 'text-white'}
        />
        <StatCard
          title="Decline Rate (7d)" value={`${s?.outcome_7d.decline_rate_pct ?? 0}%`}
          sub={`${s?.outcome_7d.DECLINE ?? 0} declined transactions`} icon={XCircle}
          color={(s?.outcome_7d.decline_rate_pct ?? 0) > 15 ? 'text-red-400' : 'text-white'}
        />
        <StatCard
          title="Avg Latency" value={`${s?.avg_latency_ms ?? 0}ms`}
          sub="Evaluation time" icon={Clock}
        />
      </div>

      {/* Drift alert */}
      {s && (
        <div className={clsx('rounded-xl border p-4 flex items-center justify-between', ALERT_BG[alert])}>
          <div className="flex items-center gap-3">
            {alert === 'OK' ? <CheckCircle size={18} className="text-emerald-400" />
              : alert === 'CRITICAL' ? <XCircle size={18} className="text-red-400" />
              : <AlertTriangle size={18} className="text-amber-400" />}
            <div>
              <p className={clsx('font-semibold text-sm', ALERT_COLORS[alert])}>
                Drift Status: {alert}
              </p>
              <p className="text-xs text-slate-400">
                {s.latest_drift_at
                  ? `Last computed: ${new Date(s.latest_drift_at).toLocaleString()}`
                  : 'No drift report computed yet. Click "Run Drift Check".'}
              </p>
            </div>
          </div>
          {driftHistory.data?.reports?.[0] && (
            <div className="text-xs text-right text-slate-400">
              <p>Score PSI: <span className="text-white">{driftHistory.data.reports[0].score_distribution_psi?.toFixed(3)}</span></p>
              <p>Max Feature PSI: <span className="text-white">{driftHistory.data.reports[0].max_feature_psi?.toFixed(3)}</span></p>
            </div>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Outcome trend chart */}
        <div className="lg:col-span-2 bg-slate-900 rounded-xl border border-slate-800 p-5">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">Outcome Trend (14d)</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={trend.data || []} barSize={10}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }}
                tickFormatter={v => v.slice(5)} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8 }}
                labelStyle={{ color: '#94a3b8' }} itemStyle={{ color: '#e2e8f0', fontSize: 12 }}
              />
              <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
              <Bar dataKey="ALLOW" fill={OUTCOME_COLORS.ALLOW} stackId="a" />
              <Bar dataKey="CHALLENGE" fill={OUTCOME_COLORS.CHALLENGE} stackId="a" />
              <Bar dataKey="DECLINE" fill={OUTCOME_COLORS.DECLINE} stackId="a" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Outcome pie */}
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-5">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">Outcome Distribution (7d)</h3>
          {outcomePieData.some(d => d.value > 0) ? (
            <>
              <ResponsiveContainer width="100%" height={160}>
                <PieChart>
                  <Pie data={outcomePieData} cx="50%" cy="50%" innerRadius={45}
                    outerRadius={70} dataKey="value" paddingAngle={3}>
                    {outcomePieData.map(entry => (
                      <Cell key={entry.name} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8 }}
                    itemStyle={{ fontSize: 12 }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="space-y-1.5 mt-2">
                {outcomePieData.map(d => (
                  <div key={d.name} className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                      <div className="w-2.5 h-2.5 rounded-full" style={{ background: d.color }} />
                      <span className="text-slate-400">{d.name}</span>
                    </div>
                    <span className="text-white font-medium">{d.value}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="text-center py-12 text-slate-500 text-sm">No data yet</div>
          )}
        </div>
      </div>

      {/* Score distribution over time */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-5">
        <h3 className="text-sm font-semibold text-slate-300 mb-4">Risk Score Over Time (7d daily)</h3>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={scoreDist.data || []}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="period" tick={{ fill: '#64748b', fontSize: 11 }}
              tickFormatter={v => v.slice(5)} />
            <YAxis domain={[0, 999]} tick={{ fill: '#64748b', fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8 }}
              itemStyle={{ color: '#e2e8f0', fontSize: 12 }}
            />
            <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
            <Line type="monotone" dataKey="avg_score" name="Avg Score" stroke="#60a5fa" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="p50_score" name="P50" stroke="#818cf8" strokeWidth={1.5} dot={false} strokeDasharray="4 4" />
            <Line type="monotone" dataKey="p95_score" name="P95" stroke="#f87171" strokeWidth={1.5} dot={false} strokeDasharray="4 4" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Top risk factors */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-slate-300">Top Risk Factors (7d)</h3>
          <span className="text-xs text-slate-500">Frequency of triggering in evaluations</span>
        </div>
        <div className="space-y-2">
          {(riskFactors.data || []).slice(0, 12).map(f => {
            const max = riskFactors.data?.[0]?.frequency || 1
            const pct = (f.frequency / max) * 100
            return (
              <div key={f.factor} className="flex items-center gap-3">
                <div className="w-40 text-xs text-slate-400 truncate">{f.factor.replace(/_/g, ' ')}</div>
                <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 rounded-full transition-all"
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <div className="w-10 text-right text-xs text-slate-400">{f.frequency}</div>
                <div className="w-16 text-right text-xs text-slate-500">
                  avg {f.avg_score_when_triggered}
                </div>
              </div>
            )
          })}
          {(riskFactors.data || []).length === 0 && (
            <p className="text-center py-8 text-slate-500 text-sm">
              No risk factors data. Submit some transactions to see results.
            </p>
          )}
        </div>
      </div>

      {/* ── Model Versioning ────────────────────────────────────────────── */}
      <ModelVersioningPanel
        versionsData={versionsQ.data}
        currentData={currentQ.data}
        trainJob={trainJobQ.data}
        onTrain={(force, autoPromote) => trainMutation.mutate({ force, autoPromote })}
        onPromote={(vid) => promoteMutation.mutate(vid)}
        onRetire={(vid) => retireMutation.mutate(vid)}
        trainPending={trainMutation.isPending}
        promotePending={promoteMutation.isPending}
      />

      {/* Drift history */}
      {driftHistory.data?.reports?.length > 0 && (
        <div className="space-y-4">
          {driftHistory.data.reports.map((r: Record<string, unknown>, i: number) => {
            const lvl = (r.overall_alert_level as string) as keyof typeof ALERT_COLORS
            const causes = (r.drift_causes as DriftCause[] | undefined) ?? []
            const topFeatures = (r.top_drifted_features as TopDriftedFeature[] | undefined) ?? []
            return (
              <DriftReportCard key={i} report={r} lvl={lvl} causes={causes} topFeatures={topFeatures} />
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── Model Versioning Panel ────────────────────────────────────────────────────

const STATUS_COLORS: Record<string, string> = {
  baseline:     'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
  experimental: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
  retired:      'bg-slate-700/40 text-slate-500 border-slate-700',
}

const STATUS_ICONS: Record<string, React.ElementType> = {
  baseline:     CheckCircle,
  experimental: Zap,
  retired:      Archive,
}

function VersionBadge({ status }: { status: string }) {
  const Icon = STATUS_ICONS[status] ?? Activity
  return (
    <span className={clsx('inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border', STATUS_COLORS[status] ?? STATUS_COLORS.retired)}>
      <Icon size={10} />
      {status}
    </span>
  )
}

function AucBar({ label, value, baseline }: { label: string; value: number; baseline?: number }) {
  const pct = Math.min(value * 100, 100)
  const delta = baseline !== undefined ? value - baseline : null
  return (
    <div>
      <div className="flex justify-between text-xs mb-0.5">
        <span className="text-slate-400">{label}</span>
        <span className="text-white font-mono">
          {value.toFixed(4)}
          {delta !== null && (
            <span className={clsx('ml-1', delta >= 0 ? 'text-emerald-400' : 'text-red-400')}>
              ({delta >= 0 ? '+' : ''}{delta.toFixed(4)})
            </span>
          )}
        </span>
      </div>
      <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
        <div
          className={clsx('h-full rounded-full', delta !== null && delta < 0 ? 'bg-amber-500' : 'bg-blue-500')}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

function VersionCard({
  version, isBaseline, baselineAuc, onPromote, onRetire, promotePending,
}: {
  version: ModelVersionMeta
  isBaseline: boolean
  baselineAuc?: number
  onPromote: () => void
  onRetire: () => void
  promotePending: boolean
}) {
  const [expanded, setExpanded] = useState(false)
  const ts = new Date(version.trained_at).toLocaleString()
  const checkpoint = version.last_txn_timestamp
    ? new Date(version.last_txn_timestamp).toLocaleDateString()
    : '—'

  return (
    <div className={clsx('rounded-xl border p-4', isBaseline ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-slate-800 bg-slate-900')}>
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-sm font-bold text-white font-mono">{version.version_id}</span>
          <VersionBadge status={version.status} />
          <span className={clsx('text-xs px-1.5 py-0.5 rounded border',
            version.training_type === 'incremental'
              ? 'bg-purple-500/10 text-purple-300 border-purple-500/30'
              : 'bg-slate-700/40 text-slate-400 border-slate-700')}>
            {version.training_type}
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {version.status === 'experimental' && !promotePending && (
            <button
              onClick={onPromote}
              className="flex items-center gap-1 px-2.5 py-1 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-medium rounded-lg transition-colors"
            >
              <ArrowUpCircle size={12} />
              Promote
            </button>
          )}
          {version.status !== 'baseline' && version.status !== 'retired' && (
            <button
              onClick={onRetire}
              className="px-2.5 py-1 bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs font-medium rounded-lg transition-colors"
            >
              Retire
            </button>
          )}
          <button
            onClick={() => setExpanded(e => !e)}
            className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
          >
            {expanded ? '▲ Less' : '▼ More'}
          </button>
        </div>
      </div>

      {/* Metrics row */}
      <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
        <div>
          <p className="text-slate-500 mb-0.5">XGB ROC-AUC</p>
          <p className={clsx('font-bold tabular-nums',
            !isBaseline && baselineAuc !== undefined
              ? version.xgb_auc >= baselineAuc ? 'text-emerald-400' : 'text-amber-400'
              : 'text-white')}>
            {version.xgb_auc ? version.xgb_auc.toFixed(4) : '—'}
          </p>
        </div>
        <div>
          <p className="text-slate-500 mb-0.5">Samples</p>
          <p className="text-white font-medium">{version.n_samples.toLocaleString()}</p>
        </div>
        <div>
          <p className="text-slate-500 mb-0.5">Trained</p>
          <p className="text-slate-300">{ts}</p>
        </div>
        <div>
          <p className="text-slate-500 mb-0.5">Checkpoint</p>
          <p className="text-slate-300">{checkpoint}</p>
        </div>
      </div>

      {/* Expanded: AUC bars + notes */}
      {expanded && (
        <div className="mt-4 space-y-3 pt-3 border-t border-slate-800">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <AucBar label="XGBoost ROC-AUC" value={version.xgb_auc} baseline={isBaseline ? undefined : baselineAuc} />
            {version.svm_auc > 0 && (
              <AucBar label="SVM ROC-AUC" value={version.svm_auc} />
            )}
          </div>
          {version.metrics?.graphsage?.roc_auc && (
            <AucBar label="GraphSAGE ROC-AUC" value={Number(version.metrics.graphsage.roc_auc)} />
          )}
          {version.notes && (
            <p className="text-xs text-slate-400 italic">{version.notes}</p>
          )}
          {version.promotion_reason && (
            <p className="text-xs text-emerald-400">↑ {version.promotion_reason}</p>
          )}
          {version.trigger && (
            <p className="text-xs text-slate-500">Trigger: {version.trigger} · Fraud rate: {(version.fraud_rate * 100).toFixed(1)}%</p>
          )}
        </div>
      )}
    </div>
  )
}

function ModelVersioningPanel({
  versionsData, currentData, trainJob,
  onTrain, onPromote, onRetire,
  trainPending, promotePending,
}: {
  versionsData: VersionsResponse | undefined
  currentData: CurrentVersionsResponse | undefined
  trainJob: TrainingStatusResponse | undefined
  onTrain: (force: boolean, autoPromote: boolean) => void
  onPromote: (vid: string) => void
  onRetire: (vid: string) => void
  trainPending: boolean
  promotePending: boolean
}) {
  const [showAll, setShowAll] = useState(false)
  const versions = versionsData?.versions ?? []
  const baseline = currentData?.baseline
  const experimental = currentData?.experimental
  const comparison = currentData?.comparison
  const baselineAuc = baseline?.xgb_auc ?? 0
  const visibleVersions = showAll ? versions : versions.slice(0, 5)

  const jobRunning = trainJob?.running
  const jobResult = trainJob?.result as Record<string, unknown> | null

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 space-y-5">
      {/* Section header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <GitBranch size={16} className="text-blue-400" />
          <h3 className="text-sm font-semibold text-white">Model Versions</h3>
          {versions.length > 0 && (
            <span className="text-xs text-slate-500">{versions.length} version{versions.length !== 1 ? 's' : ''}</span>
          )}
        </div>
        <button
          onClick={() => onTrain(false, true)}
          disabled={trainPending || jobRunning}
          className="flex items-center gap-2 px-3 py-1.5 bg-purple-600 hover:bg-purple-500 disabled:bg-slate-700 text-white text-xs font-medium rounded-lg transition-colors"
        >
          <Zap size={12} className={jobRunning ? 'animate-pulse' : ''} />
          {jobRunning ? 'Training…' : 'Incremental Train'}
        </button>
      </div>

      {/* Training job status banner */}
      {trainJob && (trainJob.running || trainJob.result || trainJob.error) && (
        <div className={clsx('rounded-lg border p-3 text-xs',
          trainJob.running
            ? 'bg-purple-500/10 border-purple-500/30 text-purple-300'
            : trainJob.error
              ? 'bg-red-500/10 border-red-500/30 text-red-300'
              : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300')}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {trainJob.running
                ? <RefreshCw size={12} className="animate-spin" />
                : trainJob.error
                  ? <XCircle size={12} />
                  : <CheckCircle size={12} />}
              <span className="font-medium">
                {trainJob.running
                  ? `Training in progress… (started ${trainJob.started_at ? new Date(trainJob.started_at).toLocaleTimeString() : ''})`
                  : trainJob.error
                    ? `Training failed: ${trainJob.error}`
                    : `Training complete — ${(jobResult?.version_id as string) ?? ''} ${(jobResult?.promoted as boolean) ? 'auto-promoted ✓' : 'is experimental'}`}
              </span>
            </div>
            {jobResult && (
              <span className="font-mono text-white">
                AUC {typeof jobResult.xgb_auc === 'number' ? (jobResult.xgb_auc as number).toFixed(4) : '—'}
                {typeof jobResult.improvement === 'number' && (
                  <span className={clsx('ml-1', (jobResult.improvement as number) >= 0 ? 'text-emerald-400' : 'text-amber-400')}>
                    ({(jobResult.improvement as number) >= 0 ? '+' : ''}{(jobResult.improvement as number).toFixed(4)})
                  </span>
                )}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Baseline vs Experimental comparison */}
      {baseline && experimental && comparison && (
        <div className="grid grid-cols-2 gap-4 p-3 rounded-lg bg-slate-800/40 border border-slate-700">
          <div>
            <p className="text-xs font-semibold text-slate-400 mb-1">Baseline ({baseline.version_id})</p>
            <p className="text-lg font-bold text-white tabular-nums">{baseline.xgb_auc.toFixed(4)}</p>
            <p className="text-xs text-slate-500">{baseline.n_samples.toLocaleString()} samples · {baseline.training_type}</p>
          </div>
          <div>
            <p className="text-xs font-semibold text-slate-400 mb-1">Experimental ({experimental.version_id})</p>
            <p className={clsx('text-lg font-bold tabular-nums',
              comparison.xgb_auc_delta >= 0 ? 'text-emerald-400' : 'text-amber-400')}>
              {experimental.xgb_auc.toFixed(4)}
              <span className="text-sm ml-1">
                ({comparison.xgb_auc_delta >= 0 ? '+' : ''}{comparison.xgb_auc_delta.toFixed(4)})
              </span>
            </p>
            <p className="text-xs text-slate-500">{experimental.n_samples.toLocaleString()} samples · {experimental.training_type}</p>
          </div>
          <div className="col-span-2 pt-2 border-t border-slate-700">
            <div className="flex items-center justify-between">
              <span className={clsx('text-xs font-medium',
                comparison.would_auto_promote ? 'text-emerald-400' : 'text-amber-400')}>
                {comparison.would_auto_promote
                  ? '✓ Meets auto-promotion criteria'
                  : `⚠ Below promotion threshold (need ≥ baseline × ${comparison.promotion_threshold})`}
              </span>
              <button
                onClick={() => onPromote(experimental.version_id)}
                disabled={promotePending}
                className="flex items-center gap-1 px-3 py-1 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 text-white text-xs font-medium rounded-lg transition-colors"
              >
                <ArrowUpCircle size={12} />
                Promote {experimental.version_id}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* No versions yet */}
      {versions.length === 0 && (
        <div className="text-center py-8 text-slate-500 text-sm">
          <GitBranch size={24} className="mx-auto mb-2 text-slate-700" />
          <p>No model versions yet.</p>
          <p className="text-xs mt-1">Run a full retrain to create the baseline (v1), then use "Incremental Train" for subsequent updates.</p>
        </div>
      )}

      {/* Version list */}
      <div className="space-y-3">
        {visibleVersions.map(v => (
          <VersionCard
            key={v.version_id}
            version={v}
            isBaseline={v.version_id === versionsData?.baseline_id}
            baselineAuc={baselineAuc}
            onPromote={() => onPromote(v.version_id)}
            onRetire={() => onRetire(v.version_id)}
            promotePending={promotePending}
          />
        ))}
      </div>

      {versions.length > 5 && (
        <button
          onClick={() => setShowAll(s => !s)}
          className="w-full text-xs text-slate-500 hover:text-slate-300 py-1 transition-colors"
        >
          {showAll ? '▲ Show less' : `▼ Show all ${versions.length} versions`}
        </button>
      )}
    </div>
  )
}

// ── Types ─────────────────────────────────────────────────────────────────────

interface DriftCause {
  category: string
  severity: string
  psi: number | null
  title: string
  detail: string
  recommendation: string
  feature?: string
}

interface TopDriftedFeature {
  feature: string
  psi: number
  alert: string
  ref_mean: number
  cur_mean: number
  mean_drift: number
  direction: string
}

// ── Drift Report Card ─────────────────────────────────────────────────────────

function DriftReportCard({
  report, lvl, causes, topFeatures,
}: {
  report: Record<string, unknown>
  lvl: keyof typeof ALERT_COLORS
  causes: DriftCause[]
  topFeatures: TopDriftedFeature[]
}) {
  const [expanded, setExpanded] = React.useState(false)

  const CAUSE_BG: Record<string, string> = {
    OK:       'bg-emerald-500/10 border-emerald-500/30',
    WARNING:  'bg-amber-500/10 border-amber-500/30',
    CRITICAL: 'bg-red-500/10 border-red-500/30',
  }
  const CAUSE_ICON: Record<string, string> = {
    OK: '✓', WARNING: '⚠', CRITICAL: '✕',
  }

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-5">
      {/* Header row */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <span className={clsx('font-bold text-sm', ALERT_COLORS[lvl])}>
            {CAUSE_ICON[lvl] ?? '?'} {lvl}
          </span>
          <span className="text-slate-400 text-xs">
            {String(report.computed_at).slice(0, 16).replace('T', ' ')}
          </span>
          {Boolean(report.drift_detected) && (
            <span className="px-2 py-0.5 text-xs bg-red-500/20 text-red-400 rounded-full border border-red-500/30">
              Drift Detected
            </span>
          )}
        </div>
        <div className="flex items-center gap-4 text-xs text-slate-400">
          <span>Score PSI: <span className="text-white">{Number(report.score_distribution_psi).toFixed(3)}</span></span>
          <span>Max Feature PSI: <span className="text-white">{Number(report.max_feature_psi).toFixed(3)}</span></span>
          <span className="text-amber-400">{Number(report.features_in_warning)} warn</span>
          <span className="text-red-400">{Number(report.features_in_critical)} crit</span>
          <button
            onClick={() => setExpanded(x => !x)}
            className="px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded text-slate-300 transition-colors"
          >
            {expanded ? 'Collapse' : 'Details →'}
          </button>
        </div>
      </div>

      {/* Drift causes — always shown (summary) */}
      {causes.length > 0 && (
        <div className="space-y-2">
          {causes.slice(0, expanded ? causes.length : 2).map((c, j) => (
            <div key={j} className={clsx('rounded-lg border p-3', CAUSE_BG[c.severity] ?? 'bg-slate-800 border-slate-700')}>
              <div className="flex items-start gap-2">
                <span className={clsx('text-xs font-bold mt-0.5', ALERT_COLORS[c.severity as keyof typeof ALERT_COLORS] ?? 'text-slate-400')}>
                  [{c.category}]
                </span>
                <div className="flex-1">
                  <p className="text-sm font-semibold text-slate-200">{c.title}</p>
                  {expanded && (
                    <>
                      <p className="text-xs text-slate-400 mt-1">{c.detail}</p>
                      <p className="text-xs text-blue-400 mt-1.5">
                        → {c.recommendation}
                      </p>
                    </>
                  )}
                </div>
              </div>
            </div>
          ))}
          {!expanded && causes.length > 2 && (
            <p className="text-xs text-slate-500 text-center">
              +{causes.length - 2} more causes — click Details to expand
            </p>
          )}
        </div>
      )}

      {/* Top drifted features table (expanded only) */}
      {expanded && topFeatures.length > 0 && (
        <div className="mt-4">
          <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
            Top Drifted Features
          </h4>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-500 border-b border-slate-800">
                  {['Feature', 'PSI', 'Alert', 'Training Mean', 'Current Mean', 'Change'].map(h => (
                    <th key={h} className="text-left py-1.5 pr-3 font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {topFeatures.map((f, fi) => (
                  <tr key={fi} className="border-b border-slate-800/40">
                    <td className="py-1.5 pr-3 text-slate-300 font-mono">{f.feature.replace(/_/g, ' ')}</td>
                    <td className="py-1.5 pr-3 tabular-nums">{f.psi.toFixed(4)}</td>
                    <td className={clsx('py-1.5 pr-3 font-medium', ALERT_COLORS[f.alert as keyof typeof ALERT_COLORS] ?? 'text-slate-400')}>
                      {f.alert}
                    </td>
                    <td className="py-1.5 pr-3 tabular-nums text-slate-400">{f.ref_mean.toFixed(3)}</td>
                    <td className="py-1.5 pr-3 tabular-nums text-slate-300">{f.cur_mean.toFixed(3)}</td>
                    <td className={clsx('py-1.5 tabular-nums font-medium',
                      f.mean_drift > 0 ? 'text-red-400' : 'text-emerald-400')}>
                      {f.direction} ({f.mean_drift > 0 ? '+' : ''}{f.mean_drift.toFixed(3)})
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
