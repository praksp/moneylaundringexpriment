import clsx from 'clsx'
import { CheckCircle, AlertTriangle, XCircle, Clock } from 'lucide-react'
import type { EvaluationResponse } from '../api/client'

interface Props {
  result: EvaluationResponse
  onChallengeAnswer?: (answer: string) => void
}

const OUTCOME_CONFIG = {
  ALLOW: {
    icon: CheckCircle,
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/30',
    text: 'text-emerald-400',
    badge: 'bg-emerald-500/20 text-emerald-300',
    label: 'Transaction Approved',
  },
  CHALLENGE: {
    icon: AlertTriangle,
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/30',
    text: 'text-amber-400',
    badge: 'bg-amber-500/20 text-amber-300',
    label: 'Verification Required',
  },
  DECLINE: {
    icon: XCircle,
    bg: 'bg-red-500/10',
    border: 'border-red-500/30',
    text: 'text-red-400',
    badge: 'bg-red-500/20 text-red-300',
    label: 'Transaction Declined',
  },
}

function ScoreGauge({ score }: { score: number }) {
  const pct = (score / 999) * 100
  const color = score <= 399 ? '#10b981' : score <= 699 ? '#f59e0b' : '#ef4444'
  const circumference = 2 * Math.PI * 54
  const dash = (pct / 100) * circumference

  return (
    <div className="relative w-36 h-36 mx-auto">
      <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
        <circle cx="60" cy="60" r="54" fill="none" stroke="#1e293b" strokeWidth="12" />
        <circle
          cx="60" cy="60" r="54" fill="none"
          stroke={color} strokeWidth="12"
          strokeDasharray={`${dash} ${circumference}`}
          strokeLinecap="round"
          style={{ transition: 'stroke-dasharray 0.8s ease' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-bold text-white">{score}</span>
        <span className="text-xs text-slate-400">/ 999</span>
      </div>
    </div>
  )
}

export default function RiskScoreCard({ result, onChallengeAnswer }: Props) {
  const { risk_score, challenge_question, processing_time_ms } = result
  const cfg = OUTCOME_CONFIG[risk_score.outcome]
  const Icon = cfg.icon

  return (
    <div className={clsx('rounded-xl border p-6 space-y-5', cfg.bg, cfg.border)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Icon size={22} className={cfg.text} />
          <h2 className={clsx('text-lg font-semibold', cfg.text)}>{cfg.label}</h2>
        </div>
        <span className={clsx('text-xs font-medium px-2.5 py-1 rounded-full', cfg.badge)}>
          {risk_score.outcome}
        </span>
      </div>

      {/* Score gauge */}
      <ScoreGauge score={risk_score.score} />

      {/* Score breakdown */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-slate-800/60 rounded-lg p-3 text-center">
          <p className="text-xs text-slate-400 mb-1">Bayesian Score</p>
          <p className="text-xl font-bold text-white">{risk_score.bayesian_score}</p>
          <p className="text-xs text-slate-500">55% weight</p>
        </div>
        <div className="bg-slate-800/60 rounded-lg p-3 text-center">
          <p className="text-xs text-slate-400 mb-1">ML Score</p>
          <p className="text-xl font-bold text-white">{risk_score.ml_score}</p>
          <p className="text-xs text-slate-500">45% weight</p>
        </div>
      </div>

      {/* Explanation */}
      <p className="text-sm text-slate-300 leading-relaxed">{risk_score.explanation}</p>

      {/* Risk factors */}
      {risk_score.risk_factors.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wide">
            Risk Factors Triggered
          </p>
          <div className="flex flex-wrap gap-2">
            {risk_score.risk_factors.map(f => (
              <span key={f} className="text-xs px-2 py-1 rounded bg-red-500/15 text-red-300 border border-red-500/20">
                {f.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Challenge question */}
      {challenge_question && (
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-4 space-y-3">
          <p className="text-sm font-semibold text-amber-300">Challenge Question</p>
          <p className="text-sm text-amber-200">{challenge_question.question}</p>
          {onChallengeAnswer && (
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="Your answer…"
                className="flex-1 bg-slate-800 border border-slate-600 rounded px-3 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-amber-400"
                onKeyDown={e => {
                  if (e.key === 'Enter') {
                    onChallengeAnswer((e.target as HTMLInputElement).value)
                  }
                }}
              />
              <button
                onClick={() => {
                  const inp = document.querySelector('input[placeholder="Your answer…"]') as HTMLInputElement
                  onChallengeAnswer(inp?.value || '')
                }}
                className="px-4 py-2 bg-amber-500 hover:bg-amber-400 text-black text-sm font-medium rounded transition-colors"
              >
                Submit
              </button>
            </div>
          )}
        </div>
      )}

      {/* Meta */}
      <div className="flex items-center gap-4 pt-1 border-t border-slate-700/50 text-xs text-slate-500">
        <div className="flex items-center gap-1">
          <Clock size={12} />
          {processing_time_ms.toFixed(1)}ms
        </div>
        <div>Confidence: {(risk_score.confidence * 100).toFixed(0)}%</div>
        <div className="truncate font-mono">{result.transaction_id.slice(0, 8)}…</div>
      </div>
    </div>
  )
}
