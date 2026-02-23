import { useState, type FormEvent } from 'react'
import { ShieldAlert, Lock, User, Eye, EyeOff, AlertCircle } from 'lucide-react'
import { useAuth } from '../context/AuthContext'

export default function Login() {
  const { login, isLoading } = useAuth()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [showPw, setShowPw]     = useState(false)
  const [error, setError]       = useState<string | null>(null)

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)
    try {
      await login(username, password)
    } catch {
      setError('Invalid username or password. Please try again.')
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo / branding */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-blue-600/20 border border-blue-500/30 mb-4">
            <ShieldAlert className="text-blue-400" size={32} />
          </div>
          <h1 className="text-2xl font-bold text-white">AML Risk Engine</h1>
          <p className="text-slate-400 text-sm mt-1">
            Anti-Money Laundering Detection System
          </p>
        </div>

        {/* Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 shadow-2xl">
          <h2 className="text-lg font-semibold text-white mb-1">Sign in</h2>
          <p className="text-slate-400 text-sm mb-6">
            Enter your credentials to access the platform
          </p>

          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Username */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1.5">
                Username
              </label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={16} />
                <input
                  type="text"
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  required
                  autoFocus
                  placeholder="Enter username"
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg pl-9 pr-4 py-2.5
                             text-white placeholder-slate-500 text-sm
                             focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                             transition-all"
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1.5">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={16} />
                <input
                  type={showPw ? 'text' : 'password'}
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  required
                  placeholder="Enter password"
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg pl-9 pr-10 py-2.5
                             text-white placeholder-slate-500 text-sm
                             focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                             transition-all"
                />
                <button
                  type="button"
                  onClick={() => setShowPw(v => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors"
                >
                  {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            {/* Error */}
            {error && (
              <div className="flex items-start gap-2 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2.5">
                <AlertCircle className="text-red-400 flex-shrink-0 mt-0.5" size={15} />
                <p className="text-red-400 text-sm">{error}</p>
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              disabled={isLoading || !username || !password}
              className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500
                         text-white font-semibold py-2.5 rounded-lg text-sm transition-all
                         focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
                         focus:ring-offset-slate-900"
            >
              {isLoading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Signing in…
                </span>
              ) : (
                'Sign in'
              )}
            </button>
          </form>

          {/* Hint cards */}
          <div className="mt-6 space-y-2">
            <p className="text-xs text-slate-500 font-medium uppercase tracking-wider mb-2">
              Available accounts
            </p>
            {[
              { role: 'Admin', username: 'admin', password: 'password', color: 'blue', desc: 'Full access · Customer profiles · PII visible' },
              { role: 'Viewer', username: 'viewer', password: 'viewer123', color: 'emerald', desc: 'Aggregated view · No customer PII' },
            ].map(acc => (
              <button
                key={acc.username}
                type="button"
                onClick={() => { setUsername(acc.username); setPassword(acc.password) }}
                className="w-full text-left bg-slate-800/60 hover:bg-slate-800 border border-slate-700/50
                           rounded-lg px-3 py-2.5 transition-colors group"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${
                      acc.color === 'blue'
                        ? 'bg-blue-600/20 text-blue-400'
                        : 'bg-emerald-600/20 text-emerald-400'
                    }`}>
                      {acc.role}
                    </span>
                    <span className="text-sm text-white font-mono">{acc.username}</span>
                    <span className="text-xs text-slate-500 font-mono">/ {acc.password}</span>
                  </div>
                  <span className="text-xs text-slate-500 group-hover:text-slate-400">click to fill →</span>
                </div>
                <p className="text-xs text-slate-500 mt-0.5 pl-0.5">{acc.desc}</p>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
