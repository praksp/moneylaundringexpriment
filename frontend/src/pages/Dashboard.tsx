import { useQuery } from '@tanstack/react-query'
import { useMemo } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Cell } from 'recharts'
import { ShieldAlert, TrendingUp, Users, Activity } from 'lucide-react'
import { getTransactionStats, getMonitoringSummary } from '../api/client'
import clsx from 'clsx'
import { Link } from 'react-router-dom'

export default function Dashboard() {
  const stats = useQuery({ queryKey: ['txn-stats'], queryFn: getTransactionStats, refetchInterval: 30_000 })
  const monitor = useQuery({ queryKey: ['monitoring-summary'], queryFn: getMonitoringSummary, refetchInterval: 30_000 })

  const s = stats.data
  const m = monitor.data

  const fraudByType = useMemo(() => {
    return Object.entries(s?.fraud_by_type || {}).map(([type, count]) => ({
      type: type?.replace('_', ' ') || 'UNKNOWN',
      count,
    }))
  }, [s?.fraud_by_type])

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">AML Risk Engine</h1>
        <p className="text-slate-400 text-sm mt-1">
          Neo4j graph database · Bayesian + XGBoost risk scoring · Real-time evaluation
        </p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: 'Total Transactions', value: s?.total_transactions?.toLocaleString() ?? '—', sub: `$${((s?.total_volume_usd || 0) / 1e6).toFixed(2)}M volume`, icon: Activity, color: 'text-white' },
          { label: 'Fraud Rate', value: `${s?.fraud_rate_pct ?? 0}%`, sub: `${s?.fraud_count ?? 0} flagged transactions`, icon: ShieldAlert, color: (s?.fraud_rate_pct || 0) > 10 ? 'text-red-400' : 'text-amber-400' },
          { label: 'Customers', value: s?.total_customers?.toLocaleString() ?? '—', sub: `${s?.total_accounts ?? 0} linked accounts`, icon: Users, color: 'text-blue-400' },
          { label: 'Avg Risk Score', value: m?.avg_score_7d ?? '—', sub: 'Last 7 days', icon: TrendingUp, color: (m?.avg_score_7d || 0) > 500 ? 'text-red-400' : 'text-emerald-400' },
        ].map(({ label, value, sub, icon: Icon, color }) => (
          <div key={label} className="bg-slate-900 border border-slate-800 rounded-xl p-5">
            <div className="flex items-center justify-between mb-3">
              <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">{label}</p>
              <Icon size={16} className="text-slate-500" />
            </div>
            <p className={clsx('text-2xl font-bold', color)}>{value}</p>
            <p className="text-xs text-slate-500 mt-1">{sub}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Fraud by type */}
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-5">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">Fraud Patterns Detected</h3>
          {fraudByType.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={fraudByType} layout="vertical" barSize={14}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                <XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} />
                <YAxis type="category" dataKey="type" tick={{ fill: '#94a3b8', fontSize: 11 }} width={130} />
                <Tooltip
                  contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8 }}
                  itemStyle={{ color: '#e2e8f0', fontSize: 12 }}
                />
                <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                  {fraudByType.map((_, i) => (
                    <Cell key={i} fill={['#ef4444', '#f97316', '#f59e0b', '#eab308', '#84cc16', '#22c55e', '#14b8a6'][i % 7]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-center py-12 text-slate-500 text-sm">Loading…</p>
          )}
        </div>

        {/* Risk engine status */}
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 space-y-4">
          <h3 className="text-sm font-semibold text-slate-300">Risk Engine Status</h3>

          <div className="space-y-3">
            {[
              { label: 'Neo4j Graph DB', status: 'ONLINE', detail: `${s?.total_transactions || 0} transactions` },
              { label: 'Bayesian Engine', status: 'ONLINE', detail: '20+ likelihood ratios' },
              { label: 'XGBoost Model', status: m ? 'ONLINE' : 'LOADING', detail: 'ROC-AUC 1.0 on training' },
              { label: 'Feature Store', status: 'ONLINE', detail: '44 graph features' },
              { label: 'Drift Monitor', status: m?.latest_drift_alert === 'CRITICAL' ? 'ALERT' : 'ONLINE', detail: m?.latest_drift_alert || '—' },
            ].map(({ label, status, detail }) => (
              <div key={label} className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <div className={clsx('w-2 h-2 rounded-full', {
                    'bg-emerald-400': status === 'ONLINE',
                    'bg-amber-400': status === 'LOADING',
                    'bg-red-400 animate-pulse': status === 'ALERT',
                  })} />
                  <span className="text-sm text-slate-300">{label}</span>
                </div>
                <span className="text-xs text-slate-500">{detail}</span>
              </div>
            ))}
          </div>

          {/* Thresholds */}
          <div className="border-t border-slate-800 pt-4">
            <p className="text-xs text-slate-400 font-semibold mb-3 uppercase tracking-wide">Score Thresholds</p>
            <div className="flex gap-0 rounded-lg overflow-hidden text-xs text-center font-medium">
              <div className="flex-1 bg-emerald-500/20 text-emerald-300 py-2">
                <div>0–399</div><div className="text-emerald-400 font-bold">ALLOW</div>
              </div>
              <div className="flex-1 bg-amber-500/20 text-amber-300 py-2">
                <div>400–699</div><div className="text-amber-400 font-bold">CHALLENGE</div>
              </div>
              <div className="flex-1 bg-red-500/20 text-red-300 py-2">
                <div>700–999</div><div className="text-red-400 font-bold">DECLINE</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Quick actions */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[
          { label: 'Submit Transaction', desc: 'Evaluate in real-time', to: '/submit', color: 'border-blue-500/30 hover:border-blue-500/60' },
          { label: 'Customer Profiles', desc: 'View risk history', to: '/customers', color: 'border-slate-700 hover:border-slate-500' },
          { label: 'Feature Store', desc: 'Mule indicators', to: '/features', color: 'border-slate-700 hover:border-slate-500' },
          { label: 'Model Monitor', desc: 'Drift & performance', to: '/monitor', color: 'border-slate-700 hover:border-slate-500' },
        ].map(({ label, desc, to, color }) => (
          <Link
            key={to}
            to={to}
            className={clsx('bg-slate-900 rounded-xl border p-4 transition-colors hover:bg-slate-800', color)}
          >
            <p className="text-sm font-semibold text-white">{label}</p>
            <p className="text-xs text-slate-500 mt-0.5">{desc}</p>
          </Link>
        ))}
      </div>
    </div>
  )
}
