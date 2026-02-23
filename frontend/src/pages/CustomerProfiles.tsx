import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Search, User, AlertTriangle, TrendingUp, TrendingDown, Minus, Shield, ChevronRight } from 'lucide-react'
import { listCustomers, getCustomerProfile, computeFeatureSnapshot, type CustomerProfile } from '../api/client'
import clsx from 'clsx'
import { useMutation } from '@tanstack/react-query'

const TIER_COLORS: Record<string, string> = {
  LOW: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  MEDIUM: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
  HIGH: 'text-orange-400 bg-orange-500/10 border-orange-500/20',
  CRITICAL: 'text-red-400 bg-red-500/10 border-red-500/20',
}

const TrendIcon = ({ trend }: { trend: string }) =>
  trend === 'RISING' ? <TrendingUp size={13} className="text-red-400" />
  : trend === 'FALLING' ? <TrendingDown size={13} className="text-emerald-400" />
  : <Minus size={13} className="text-slate-400" />

interface CustomerRow {
  id: string; name: string; type: string; country: string; risk_tier: string
  account_count: number; pep_flag: boolean; sanctions_flag: boolean; kyc_level: string
}

function CustomerCard({ customer, onClick }: { customer: CustomerRow; onClick: () => void }) {
  const tier = customer.risk_tier || 'LOW'
  return (
    <button
      onClick={onClick}
      className="w-full bg-slate-900 border border-slate-800 hover:border-slate-600 rounded-xl p-4 text-left transition-colors group"
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-full bg-slate-700 flex items-center justify-center text-slate-300 font-medium text-sm">
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
          <ChevronRight size={14} className="text-slate-600 group-hover:text-slate-400 transition-colors" />
        </div>
      </div>
      <div className="mt-3 flex items-center gap-4 text-xs text-slate-500">
        <span>{customer.account_count} account{customer.account_count !== 1 ? 's' : ''}</span>
        {customer.pep_flag && <span className="text-amber-400">PEP</span>}
        {customer.sanctions_flag && <span className="text-red-400">SANCTIONS</span>}
        <span className="ml-auto">{customer.kyc_level}</span>
      </div>
    </button>
  )
}

function ProfilePanel({ profile }: { profile: CustomerProfile }) {
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
            {profile.risk_profile.risk_trend}
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
          <span>Avg Score: <span className="text-white">{profile.risk_profile.avg_score_30d}</span></span>
          <span>Max Score: <span className="text-white">{profile.risk_profile.max_score_30d}</span></span>
          <span>Evaluations: <span className="text-white">{profile.risk_profile.total_evaluations}</span></span>
          <span>Fraud Incidents: <span className="text-red-400 font-medium">{profile.risk_profile.fraud_incident_count}</span></span>
        </div>
      </div>

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
            Score: {mi.mule_score}/100
          </div>
        </div>
        <div className="space-y-2.5">
          {[
            { label: 'Pass-through Account', value: mi.is_pass_through, info: `Turnover: ${mi.turnover_ratio?.toFixed(2)}` },
            { label: 'High Sender Diversity', value: mi.high_sender_count, info: `${mi.unique_senders_30d} unique senders` },
            { label: 'Rapid Disbursement', value: mi.rapid_disbursement, info: `Avg hold: ${mi.avg_hold_time_hours}h` },
            { label: 'Structuring Pattern', value: mi.structuring_risk, info: `${mi.structuring_incidents_30d} incidents` },
          ].map(({ label, value, info }) => (
            <div key={label} className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-2">
                <div className={clsx('w-2 h-2 rounded-full', value ? 'bg-red-400' : 'bg-slate-600')} />
                <span className="text-slate-300">{label}</span>
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
        <h3 className="text-sm font-semibold text-slate-300 mb-3">Accounts</h3>
        <div className="space-y-2">
          {profile.accounts.map(acct => (
            <div key={acct.id} className="flex items-center justify-between text-xs bg-slate-800/50 rounded-lg px-3 py-2.5">
              <div>
                <p className="text-white font-medium">{acct.account_number}</p>
                <p className="text-slate-500">{acct.account_type} · {acct.bank_name}</p>
              </div>
              <div className="text-right">
                <p className="text-white">{acct.currency} {acct.balance.toLocaleString()}</p>
                {acct.is_dormant && <p className="text-amber-400">Dormant</p>}
                <button
                  onClick={() => snapshotMutation.mutate({ cid: profile.customer_id, aid: acct.id })}
                  className="text-blue-400 hover:text-blue-300 mt-0.5"
                >
                  {snapshotMutation.isPending ? 'Computing…' : 'Compute snapshot'}
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recent transactions */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-5">
        <h3 className="text-sm font-semibold text-slate-300 mb-3">Recent Transactions</h3>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {profile.recent_transactions.length === 0 && (
            <p className="text-xs text-slate-500 text-center py-4">No transactions in last 30 days</p>
          )}
          {profile.recent_transactions.map(t => (
            <div key={t.id} className="flex items-center justify-between text-xs bg-slate-800/40 rounded-lg px-3 py-2">
              <div>
                <p className="text-white font-mono">{t.reference}</p>
                <p className="text-slate-500">{t.transaction_type} · {t.timestamp?.slice(0, 10)}</p>
              </div>
              <div className="text-right">
                <p className="text-white font-medium">{t.currency} {t.amount.toLocaleString()}</p>
                {t.outcome && (
                  <span className={clsx('text-xs',
                    t.outcome === 'ALLOW' ? 'text-emerald-400'
                    : t.outcome === 'CHALLENGE' ? 'text-amber-400'
                    : 'text-red-400'
                  )}>{t.outcome}</span>
                )}
                {t.is_fraud && <span className="text-red-400 ml-1">FRAUD</span>}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Network */}
      {profile.network_connections.length > 0 && (
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-5">
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Network Connections</h3>
          <div className="space-y-2">
            {profile.network_connections.map(conn => (
              <div key={conn.customer_id} className="flex items-center justify-between text-xs bg-slate-800/40 rounded-lg px-3 py-2">
                <div>
                  <p className="text-white">{conn.customer_name}</p>
                  <p className="text-slate-500">{conn.connection_type.replace('_', ' ')}</p>
                </div>
                <span className={clsx('px-2 py-0.5 rounded border text-xs',
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

export default function CustomerProfiles() {
  const [search, setSearch] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [riskFilter, setRiskFilter] = useState('')

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
          {data?.total || 0} customers · Graph-linked transaction history, mule indicators and risk scores
        </p>
      </div>

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
        <div className="xl:col-span-2 space-y-2 max-h-[calc(100vh-200px)] overflow-y-auto">
          {isLoading ? (
            <div className="text-center py-12 text-slate-500">Loading…</div>
          ) : filtered.length === 0 ? (
            <div className="text-center py-12 text-slate-500">No customers found</div>
          ) : (
            filtered.map(c => (
              <CustomerCard
                key={c.id}
                customer={c}
                onClick={() => setSelectedId(c.id)}
              />
            ))
          )}
        </div>

        {/* Profile panel */}
        <div className="xl:col-span-3 max-h-[calc(100vh-200px)] overflow-y-auto">
          {selectedId === null ? (
            <div className="bg-slate-900 border border-slate-800 border-dashed rounded-xl p-16 text-center">
              <User size={40} className="text-slate-600 mx-auto mb-3" />
              <p className="text-slate-500 text-sm">Select a customer to view their full profile</p>
            </div>
          ) : profileLoading ? (
            <div className="text-center py-20 text-slate-500">Loading profile…</div>
          ) : profile ? (
            <ProfilePanel profile={profile} />
          ) : (
            <div className="text-center py-20 text-red-400">Profile not found</div>
          )}
        </div>
      </div>
    </div>
  )
}
