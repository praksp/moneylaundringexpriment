import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Database, AlertTriangle, RefreshCw } from 'lucide-react'
import { getHighRiskAccounts, computeFeatureSnapshot } from '../api/client'
import clsx from 'clsx'

export default function FeatureStore() {
  const qc = useQueryClient()
  const { data, isLoading, refetch } = useQuery({
    queryKey: ['high-risk-accounts'],
    queryFn: getHighRiskAccounts,
  })

  const recomputeMutation = useMutation({
    mutationFn: ({ cid, aid }: { cid: string; aid: string }) =>
      computeFeatureSnapshot(cid, aid),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['high-risk-accounts'] }),
  })

  interface HighRiskAccount {
    customer_id: string; customer_name: string; account_id: string
    account_number: string; mule_score: number; is_likely_mule: boolean
    turnover_ratio: number; tor_activity: boolean; avg_risk_score: number
    computed_at: string
  }
  const accounts = (data?.accounts || []) as HighRiskAccount[]

  const mulelScore = (score: number) =>
    score >= 70 ? 'text-red-400' : score >= 40 ? 'text-amber-400' : 'text-emerald-400'

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white">Feature Store</h1>
          <p className="text-slate-400 text-sm mt-1">
            Pre-computed mule account indicators and behavioral features per account.
            Stored in Neo4j as <code className="text-blue-400 text-xs">FeatureSnapshot</code> nodes.
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

      {/* Feature groups legend */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-8">
        {[
          { label: 'Mule Indicators', desc: 'Turnover ratio, hold time, senders', color: 'border-red-500/30' },
          { label: 'Behavioral', desc: 'Velocity, volume, structuring', color: 'border-amber-500/30' },
          { label: 'Network', desc: 'Counterparties, shared devices', color: 'border-blue-500/30' },
          { label: 'Risk History', desc: 'Avg/max score, decline rate', color: 'border-purple-500/30' },
        ].map(g => (
          <div key={g.label} className={clsx('bg-slate-900 rounded-lg border p-3', g.color)}>
            <p className="text-sm font-medium text-white">{g.label}</p>
            <p className="text-xs text-slate-500 mt-0.5">{g.desc}</p>
          </div>
        ))}
      </div>

      {isLoading ? (
        <div className="text-center py-16 text-slate-500">Loading feature store…</div>
      ) : accounts.length === 0 ? (
        <div className="text-center py-16 bg-slate-900 border border-dashed border-slate-700 rounded-xl">
          <Database size={36} className="text-slate-600 mx-auto mb-3" />
          <p className="text-slate-400 text-sm font-medium">No feature snapshots yet</p>
          <p className="text-slate-500 text-xs mt-1">
            Go to a customer profile and click "Compute snapshot" on an account to populate the store.
          </p>
        </div>
      ) : (
        <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
          <div className="px-5 py-3 border-b border-slate-800 flex items-center justify-between">
            <p className="text-sm font-semibold text-slate-300">
              {accounts.length} High-Risk Account{accounts.length !== 1 ? 's' : ''}
            </p>
            <span className="text-xs text-slate-500">Sorted by mule score descending</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-slate-400 border-b border-slate-800">
                  {[
                    'Customer', 'Account', 'Mule Score', 'Turnover', 'Senders 30d',
                    'Tor Activity', 'Avg Risk Score', 'Computed At', '',
                  ].map(h => (
                    <th key={h} className="text-left py-3 px-4 font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
          {accounts.map((row: HighRiskAccount, i: number) => {
            const score = Number(row.mule_score)
                  return (
                    <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                      <td className="py-3 px-4">
                        <p className="text-white text-xs font-medium">{row.customer_name as string}</p>
                        <p className="text-slate-500 text-xs font-mono">{String(row.customer_id).slice(0, 8)}…</p>
                      </td>
                      <td className="py-3 px-4 font-mono text-xs text-slate-300">{row.account_number}</td>
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-2 bg-slate-800 rounded-full overflow-hidden">
                            <div
                              className={clsx('h-full rounded-full', score >= 70 ? 'bg-red-500' : score >= 40 ? 'bg-amber-500' : 'bg-emerald-500')}
                              style={{ width: `${score}%` }}
                            />
                          </div>
                          <span className={clsx('text-xs font-bold', mulelScore(score))}>{score}</span>
                        </div>
                        {row.is_likely_mule && (
                          <span className="text-xs text-red-400 flex items-center gap-1 mt-1">
                            <AlertTriangle size={10} /> Likely Mule
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-xs text-slate-300">
                        {Number(row.turnover_ratio).toFixed(2)}
                      </td>
                      <td className="py-3 px-4 text-xs text-slate-300">—</td>
                      <td className="py-3 px-4">
                        {row.tor_activity ? (
                          <span className="text-xs text-red-400 font-medium">YES</span>
                        ) : (
                          <span className="text-xs text-slate-500">No</span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-xs text-slate-300">
                        {Number(row.avg_risk_score).toFixed(0)}
                      </td>
                      <td className="py-3 px-4 text-xs text-slate-500">
                        {String(row.computed_at).slice(0, 10)}
                      </td>
                      <td className="py-3 px-4">
                        <button
                          onClick={() => recomputeMutation.mutate({
                            cid: row.customer_id as string,
                            aid: row.account_id as string,
                          })}
                          className="text-xs text-blue-400 hover:text-blue-300"
                        >
                          Recompute
                        </button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Feature schema reference */}
      <div className="mt-8 bg-slate-900 rounded-xl border border-slate-800 p-5">
        <h3 className="text-sm font-semibold text-slate-300 mb-3">Feature Schema Reference</h3>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-x-8 gap-y-2 text-xs">
          {[
            ['mule_score', '0–100 composite mule likelihood'],
            ['turnover_ratio_30d', 'outbound / inbound volume'],
            ['unique_senders_30d', 'distinct accounts sending funds in'],
            ['structuring_count_30d', 'txns in $9k–$9.99k band'],
            ['has_tor_activity', 'any transaction via Tor exit node'],
            ['is_dormant', 'last_active > 90 days ago'],
            ['account_age_days', 'days since account creation'],
            ['avg_risk_score_30d', 'mean Bayesian+ML score'],
            ['decline_rate_30d', 'fraction of evaluations DECLINED'],
            ['is_pass_through', 'turnover 0.7–1.3 with >$1k inbound'],
            ['outbound_volume_30d', 'total USD sent out in 30 days'],
            ['vpn_transaction_count_30d', 'transactions originating from VPN'],
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
