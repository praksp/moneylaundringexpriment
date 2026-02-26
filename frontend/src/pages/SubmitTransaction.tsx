import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Send, Info, Upload, CheckCircle, RefreshCw } from 'lucide-react'
import { getAccounts, submitTransaction, uploadTransactionsCSV, getUploadStatus, type EvaluationResponse } from '../api/client'
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

function SingleTransactionTab({ accounts }: { accounts: any[] }) {
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
            {(mutation.error as any).response?.data?.detail || (mutation.error as Error).message}
          </div>
        )}
      </form>

      {/* Result */}
      <div>
        {result ? (
          <RiskScoreCard result={result} />
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
  )
}

function BulkUploadTab() {
  const [file, setFile] = useState<File | null>(null)
  const [errorMsg, setErrorMsg] = useState('')

  const statusQ = useQuery({
    queryKey: ['upload-status'],
    queryFn: getUploadStatus,
    refetchInterval: (data: any) => data?.running ? 2000 : false,
  })

  const uploadMutation = useMutation({
    mutationFn: uploadTransactionsCSV,
    onSuccess: () => statusQ.refetch(),
    onError: (err: any) => setErrorMsg(err.response?.data?.detail || err.message),
  })

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0])
      setErrorMsg('')
    }
  }

  const handleUpload = () => {
    if (file) uploadMutation.mutate(file)
  }

  const s = statusQ.data
  const isRunning = s?.running

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-8 max-w-2xl">
      <h2 className="text-lg font-bold text-white mb-4">Bulk Transaction Upload</h2>
      <p className="text-slate-400 text-sm mb-6">
        Upload a CSV file to ingest historical transactions in bulk. The system will
        automatically extract graph features and trigger incremental model training.
      </p>

      {!isRunning ? (
        <div className="space-y-6">
          <div className="border-2 border-dashed border-slate-700 rounded-xl p-8 text-center hover:bg-slate-800/30 transition-colors">
            <Upload size={32} className="text-blue-500 mx-auto mb-3" />
            <p className="text-slate-300 font-medium mb-1">Select CSV File</p>
            <p className="text-slate-500 text-xs mb-4">Must contain sender_account_id and amount</p>
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="block w-full text-sm text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-500 mx-auto cursor-pointer max-w-xs"
            />
          </div>

          {errorMsg && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-sm text-red-300">
              {errorMsg}
            </div>
          )}

          {s?.status === 'completed' && !s.error && (
            <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg px-4 py-4">
              <div className="flex items-center gap-2 text-emerald-400 font-medium mb-2">
                <CheckCircle size={16} /> Previous Upload Complete
              </div>
              <ul className="text-xs text-emerald-300/80 space-y-1 ml-6 list-disc">
                <li>Processed: {s.processed_records} records</li>
                <li>Training Result: {s.training_result?.status as string}</li>
                {s.training_result && Boolean(s.training_result.improvement) && (
                  <li>Model Improvement: {(s.training_result.improvement as number) > 0 ? '+' : ''}{(s.training_result.improvement as number).toFixed(4)} AUC</li>
                )}
              </ul>
            </div>
          )}

          {s?.status === 'failed' && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-sm text-red-300">
              Previous upload failed: {s.error}
            </div>
          )}

          <div className="flex items-center gap-4 pt-4 border-t border-slate-800">
            <button
              onClick={handleUpload}
              disabled={!file || uploadMutation.isPending}
              className="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-semibold rounded-lg transition-colors flex items-center gap-2"
            >
              {uploadMutation.isPending ? <RefreshCw size={16} className="animate-spin" /> : <Upload size={16} />}
              Upload & Train
            </button>
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault()
                alert("Run 'python scripts/generate_sample_csv.py' on the backend to generate a sample file.")
              }}
              className="text-sm text-blue-400 hover:text-blue-300 transition-colors"
            >
              How to get a sample CSV?
            </a>
          </div>
        </div>
      ) : (
        <div className="border border-slate-700 rounded-xl p-8 text-center bg-slate-800/20">
          <RefreshCw size={32} className="text-blue-500 mx-auto mb-4 animate-spin" />
          <h3 className="text-white font-medium mb-2">Processing Upload...</h3>
          <p className="text-slate-400 text-sm mb-4">Status: <span className="text-blue-300">{s.status.replace(/_/g, ' ')}</span></p>
          
          <div className="w-full max-w-sm mx-auto bg-slate-800 rounded-full h-2.5 mb-2 overflow-hidden">
            <div 
              className="bg-blue-500 h-2.5 rounded-full transition-all duration-500" 
              style={{ width: `${s.total_records ? (s.processed_records / s.total_records) * 100 : 0}%` }}
            />
          </div>
          <p className="text-xs text-slate-500">{s.processed_records} / {s.total_records} records inserted</p>
        </div>
      )}
    </div>
  )
}

export default function SubmitTransactionWrapper() {
  const { data: accounts = [] } = useQuery({ queryKey: ['accounts'], queryFn: getAccounts })
  const [tab, setTab] = useState<'single' | 'bulk'>('single')

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">New Transaction</h1>
          <p className="text-slate-400 mt-1 text-sm">
            Submit transactions for real-time risk evaluation or bulk ingest via CSV.
          </p>
        </div>
      </div>

      <div className="flex space-x-1 bg-slate-900/50 p-1 rounded-xl w-fit mb-8 border border-slate-800">
        <button
          onClick={() => setTab('single')}
          className={clsx(
            'px-4 py-2 text-sm font-medium rounded-lg transition-colors',
            tab === 'single' ? 'bg-slate-800 text-white shadow-sm border border-slate-700' : 'text-slate-400 hover:text-slate-200'
          )}
        >
          Single Evaluation
        </button>
        <button
          onClick={() => setTab('bulk')}
          className={clsx(
            'px-4 py-2 text-sm font-medium rounded-lg transition-colors flex items-center gap-2',
            tab === 'bulk' ? 'bg-slate-800 text-white shadow-sm border border-slate-700' : 'text-slate-400 hover:text-slate-200'
          )}
        >
          Bulk Upload <span className="bg-purple-500/20 text-purple-300 px-1.5 py-0.5 rounded text-[10px] uppercase">New</span>
        </button>
      </div>

      {tab === 'single' ? (
        <SingleTransactionTab accounts={accounts} />
      ) : (
        <BulkUploadTab />
      )}
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
