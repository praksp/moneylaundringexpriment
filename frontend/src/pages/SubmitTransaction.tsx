import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Send, Info } from 'lucide-react'
import { getAccounts, submitTransaction, type EvaluationResponse } from '../api/client'
import RiskScoreCard from '../components/RiskScoreCard'
import clsx from 'clsx'

const TXN_TYPES = ['ACH', 'WIRE', 'CARD', 'CRYPTO', 'CASH', 'INTERNAL']
const CHANNELS = ['ONLINE', 'MOBILE', 'BRANCH', 'ATM', 'API']
const CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD']
const HIGH_RISK_COUNTRIES = ['IR', 'KP', 'SY', 'MM', 'RU', 'VE', 'AF', 'PK']

const Field = ({ label, required, children, hint }: {
  label: string; required?: boolean; children: React.ReactNode; hint?: string
}) => (
  <div className="space-y-1.5">
    <label className="text-sm font-medium text-slate-300">
      {label}{required && <span className="text-red-400 ml-1">*</span>}
    </label>
    {children}
    {hint && <p className="text-xs text-slate-500">{hint}</p>}
  </div>
)

const Select = ({ value, onChange, options, placeholder }: {
  value: string; onChange: (v: string) => void
  options: { value: string; label: string }[]; placeholder?: string
}) => (
  <select
    value={value}
    onChange={e => onChange(e.target.value)}
    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/30"
  >
    {placeholder && <option value="">{placeholder}</option>}
    {options.map(o => (
      <option key={o.value} value={o.value}>{o.label}</option>
    ))}
  </select>
)

export default function SubmitTransaction() {
  const { data: accounts = [] } = useQuery({ queryKey: ['accounts'], queryFn: getAccounts })
  const [result, setResult] = useState<EvaluationResponse | null>(null)

  const [form, setForm] = useState({
    sender_account_id: '',
    receiver_account_id: '',
    amount: '',
    currency: 'USD',
    transaction_type: 'ACH',
    channel: 'ONLINE',
    description: '',
    beneficiary_country: '',
    beneficiary_name: '',
  })

  const mutation = useMutation({
    mutationFn: submitTransaction,
    onSuccess: (data) => setResult(data),
  })

  const set = (k: string, v: string) => setForm(f => ({ ...f, [k]: v }))

  const senderAccount = accounts.find(a => a.id === form.sender_account_id)
  const isHighRiskCountry = HIGH_RISK_COUNTRIES.includes(form.beneficiary_country)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!form.sender_account_id || !form.amount) return
    const payload: Record<string, unknown> = {
      sender_account_id: form.sender_account_id,
      amount: parseFloat(form.amount),
      currency: form.currency,
      transaction_type: form.transaction_type,
      channel: form.channel,
      description: form.description || undefined,
    }
    if (form.receiver_account_id) payload.receiver_account_id = form.receiver_account_id
    if (form.beneficiary_country) {
      payload.beneficiary_country = form.beneficiary_country
      payload.beneficiary_name = form.beneficiary_name || 'External Beneficiary'
    }
    mutation.mutate(payload)
  }

  const accountOptions = accounts.map(a => ({
    value: a.id,
    label: `${a.customer_name} — ${a.account_number} (${a.currency} ${a.type})`,
  }))

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">New Transaction</h1>
        <p className="text-slate-400 mt-1 text-sm">
          Submit a financial transaction for real-time AML risk evaluation.
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Sender */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-6 space-y-4">
            <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wide">Sender</h2>

            <Field label="Sender Account" required>
              <Select
                value={form.sender_account_id}
                onChange={v => set('sender_account_id', v)}
                options={accountOptions}
                placeholder="Select sender account…"
              />
            </Field>

            {senderAccount && (
              <div className="flex items-start gap-2 bg-slate-800/50 rounded-lg p-3">
                <Info size={14} className="text-blue-400 mt-0.5 flex-shrink-0" />
                <div className="text-xs text-slate-400 space-y-0.5">
                  <p>Balance: <span className="text-white font-medium">
                    {senderAccount.currency} {senderAccount.balance.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                  </span></p>
                  <p>Country: <span className="text-white">{senderAccount.country}</span></p>
                  <p>Customer: <span className="text-white">{senderAccount.customer_name}</span></p>
                </div>
              </div>
            )}
          </div>

          {/* Transaction details */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-6 space-y-4">
            <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wide">Transaction Details</h2>

            <div className="grid grid-cols-2 gap-4">
              <Field label="Amount" required>
                <div className="relative">
                  <input
                    type="number" step="0.01" min="0.01"
                    value={form.amount}
                    onChange={e => set('amount', e.target.value)}
                    placeholder="0.00"
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500"
                  />
                  {form.amount && parseFloat(form.amount) >= 9000 && parseFloat(form.amount) < 10000 && (
                    <div className="absolute -bottom-5 left-0 text-xs text-amber-400 flex items-center gap-1">
                      <AlertBadge /> Structuring range ($9,000–$9,999)
                    </div>
                  )}
                </div>
              </Field>
              <Field label="Currency">
                <Select
                  value={form.currency}
                  onChange={v => set('currency', v)}
                  options={CURRENCIES.map(c => ({ value: c, label: c }))}
                />
              </Field>
            </div>

            <div className="grid grid-cols-2 gap-4 mt-2">
              <Field label="Type">
                <Select
                  value={form.transaction_type}
                  onChange={v => set('transaction_type', v)}
                  options={TXN_TYPES.map(t => ({ value: t, label: t }))}
                />
              </Field>
              <Field label="Channel">
                <Select
                  value={form.channel}
                  onChange={v => set('channel', v)}
                  options={CHANNELS.map(c => ({ value: c, label: c }))}
                />
              </Field>
            </div>

            <Field label="Description">
              <input
                type="text"
                value={form.description}
                onChange={e => set('description', e.target.value)}
                placeholder="Payment description…"
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500"
              />
            </Field>
          </div>

          {/* Receiver */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-6 space-y-4">
            <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wide">
              Receiver <span className="text-slate-600 font-normal normal-case">(internal or external)</span>
            </h2>

            <Field label="Internal Receiver Account" hint="Leave blank for external wire transfer">
              <Select
                value={form.receiver_account_id}
                onChange={v => set('receiver_account_id', v)}
                options={accountOptions.filter(a => a.value !== form.sender_account_id)}
                placeholder="Select internal receiver…"
              />
            </Field>

            {(!form.receiver_account_id || form.transaction_type === 'WIRE') && (
              <>
                <Field label="External Beneficiary Country" hint="ISO code, e.g. US, GB, IR">
                  <div className="relative">
                    <input
                      type="text"
                      value={form.beneficiary_country}
                      onChange={e => set('beneficiary_country', e.target.value.toUpperCase())}
                      placeholder="Country code…"
                      maxLength={2}
                      className={clsx(
                        'w-full bg-slate-800 border rounded-lg px-3 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none',
                        isHighRiskCountry
                          ? 'border-red-500 focus:border-red-400'
                          : 'border-slate-700 focus:border-blue-500'
                      )}
                    />
                    {isHighRiskCountry && (
                      <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-red-400 font-medium">
                        HIGH RISK
                      </span>
                    )}
                  </div>
                </Field>
                <Field label="Beneficiary Name">
                  <input
                    type="text"
                    value={form.beneficiary_name}
                    onChange={e => set('beneficiary_name', e.target.value)}
                    placeholder="Beneficiary name…"
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500"
                  />
                </Field>
              </>
            )}
          </div>

          <button
            type="submit"
            disabled={mutation.isPending || !form.sender_account_id || !form.amount}
            className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed text-white font-semibold rounded-xl transition-colors"
          >
            {mutation.isPending ? (
              <>
                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Evaluating…
              </>
            ) : (
              <>
                <Send size={17} />
                Submit & Evaluate
              </>
            )}
          </button>

          {mutation.isError && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-sm text-red-300">
              {(mutation.error as Error).message}
            </div>
          )}
        </form>

        {/* Result */}
        <div>
          {result ? (
            <RiskScoreCard
              result={result}
              onChallengeAnswer={(ans) => {
                // In production this would validate the answer
                console.log('Challenge answer:', ans)
              }}
            />
          ) : (
            <div className="bg-slate-900 rounded-xl border border-slate-800 border-dashed p-12 text-center">
              <Send size={36} className="text-slate-600 mx-auto mb-3" />
              <p className="text-slate-500 text-sm">Risk evaluation result will appear here</p>
              <p className="text-slate-600 text-xs mt-1">
                Score 0–399: ALLOW · 400–699: CHALLENGE · 700–999: DECLINE
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function AlertBadge() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
      <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>
  )
}
