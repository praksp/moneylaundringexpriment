/**
 * RiskScoreCard
 * =============
 * Displays the full risk evaluation result with:
 *  - Large ensemble score gauge
 *  - Side-by-side model comparison (Bayesian / XGBoost / SVM / KNN)
 *  - Ensemble weight visualisation
 *  - Risk factors list
 *  - Challenge answer flow
 */
import { useState } from 'react'
import clsx from 'clsx'
import {
  CheckCircle, AlertTriangle, XCircle, Clock, Send, KeyRound, ShieldCheck,
  Brain, Zap, Network, Cpu,
} from 'lucide-react'
import type { EvaluationResponse, ModelScore } from '../api/client'
import { verifyChallenge } from '../api/client'

// ── Config ────────────────────────────────────────────────────────────────────

const OUTCOME_CONFIG = {
  ALLOW: {
    icon: CheckCircle, bg: 'bg-emerald-500/10', border: 'border-emerald-500/30',
    text: 'text-emerald-400', badge: 'bg-emerald-500/20 text-emerald-300',
    label: 'Transaction Approved',
  },
  CHALLENGE: {
    icon: AlertTriangle, bg: 'bg-amber-500/10', border: 'border-amber-500/30',
    text: 'text-amber-400', badge: 'bg-amber-500/20 text-amber-300',
    label: 'Verification Required',
  },
  DECLINE: {
    icon: XCircle, bg: 'bg-red-500/10', border: 'border-red-500/30',
    text: 'text-red-400', badge: 'bg-red-500/20 text-red-300',
    label: 'Transaction Declined',
  },
}

const MODEL_META: Record<string, { icon: React.FC<{ size: number; className?: string; style?: React.CSSProperties }>; colour: string; desc: string }> = {
  bayesian: {
    icon: Brain,
    colour: '#a78bfa',
    desc: 'Rule-based likelihood ratios over 30+ AML risk factors',
  },
  xgb: {
    icon: Zap,
    colour: '#60a5fa',
    desc: 'Gradient-boosted trees trained on 1003 labelled transactions',
  },
  svm: {
    icon: Cpu,
    colour: '#34d399',
    desc: 'RBF-kernel SVM with Platt probability calibration',
  },
  knn: {
    icon: Network,
    colour: '#fbbf24',
    desc: 'Distance-weighted k=7 nearest neighbours with isotonic calibration',
  },
}

type ChallengeState = 'idle' | 'submitting' | 'passed' | 'failed'

// ── Score gauge ───────────────────────────────────────────────────────────────

function ScoreGauge({ score }: { score: number }) {
  const pct = score / 999
  const color = score <= 399 ? '#10b981' : score <= 699 ? '#f59e0b' : '#ef4444'
  const r = 54
  const c = 2 * Math.PI * r
  const dash = pct * c
  return (
    <div className="relative w-36 h-36 mx-auto">
      <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
        <circle cx="60" cy="60" r={r} fill="none" stroke="#1e293b" strokeWidth="12" />
        <circle cx="60" cy="60" r={r} fill="none" stroke={color} strokeWidth="12"
          strokeDasharray={`${dash} ${c}`} strokeLinecap="round"
          style={{ transition: 'stroke-dasharray 0.8s ease' }} />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-bold text-white">{score}</span>
        <span className="text-xs text-slate-400">/ 999</span>
      </div>
    </div>
  )
}

// ── Mini score bar for each model ─────────────────────────────────────────────

function ModelScoreBar({ ms, isEnsembleBest }: { ms: ModelScore; isEnsembleBest: boolean }) {
  const meta = MODEL_META[ms.short] || MODEL_META.bayesian
  const Icon = meta.icon
  const barColor = ms.score <= 399 ? '#10b981' : ms.score <= 699 ? '#f59e0b' : '#ef4444'
  const scoreColor = ms.score <= 399 ? 'text-emerald-400' : ms.score <= 699 ? 'text-amber-400' : 'text-red-400'

  return (
    <div className={clsx(
      'rounded-xl border p-4 space-y-3 transition-all',
      isEnsembleBest
        ? 'border-blue-500/40 bg-blue-500/5'
        : 'border-slate-700 bg-slate-800/40',
    )}>
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg flex items-center justify-center"
            style={{ backgroundColor: meta.colour + '20', border: `1px solid ${meta.colour}30` }}>
            <Icon size={14} className="opacity-90" style={{ color: meta.colour } as React.CSSProperties} />
          </div>
          <div>
            <p className="text-xs font-semibold text-slate-200">{ms.label}</p>
            <p className="text-xs text-slate-500">{ms.weight_pct}% weight</p>
          </div>
        </div>
        <div className="text-right">
          <p className={clsx('text-xl font-bold', scoreColor)}>{ms.score}</p>
          {!ms.is_trained && (
            <p className="text-xs text-amber-400">not trained</p>
          )}
        </div>
      </div>

      {/* Score bar */}
      <div className="space-y-1">
        <div className="h-2.5 bg-slate-900 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{
              width: `${(ms.score / 999) * 100}%`,
              backgroundColor: barColor,
              boxShadow: `0 0 6px ${barColor}60`,
            }}
          />
        </div>
        {/* Weight contribution bar */}
        <div className="h-1 bg-slate-900 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full opacity-50"
            style={{
              width: `${(ms.score / 999) * (ms.weight_pct / 100) * 100}%`,
              backgroundColor: meta.colour,
            }}
          />
        </div>
        <p className="text-xs text-slate-600">{meta.desc}</p>
      </div>

      {/* Contribution label */}
      <div className="text-xs text-slate-500 border-t border-slate-800 pt-2">
        Weighted contribution:{' '}
        <span className="font-mono font-medium text-slate-300">
          {Math.round(ms.score * ms.weight_pct / 100)}
        </span>
        <span className="text-slate-600"> pts to ensemble</span>
      </div>
    </div>
  )
}

// ── Model agreement indicator ─────────────────────────────────────────────────

function ModelAgreement({ scores }: { scores: ModelScore[] }) {
  if (scores.length < 2) return null
  const vals = scores.map(s => s.score)
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length
  const std = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length)
  const pct = Math.round(100 - (std / 999) * 100)

  const label = pct >= 85 ? 'Strong agreement'
    : pct >= 65 ? 'Moderate agreement'
    : 'Low agreement'
  const colour = pct >= 85 ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30'
    : pct >= 65 ? 'text-amber-400 bg-amber-500/10 border-amber-500/30'
    : 'text-red-400 bg-red-500/10 border-red-500/30'

  return (
    <div className={clsx('flex items-center justify-between text-xs px-3 py-2 rounded-lg border', colour)}>
      <span className="font-medium">{label} across models</span>
      <span className="font-bold font-mono">{pct}%</span>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

interface Props { result: EvaluationResponse }

export default function RiskScoreCard({ result }: Props) {
  const { risk_score, challenge_question, processing_time_ms, transaction_id } = result
  const cfg = OUTCOME_CONFIG[risk_score.outcome]
  const Icon = cfg.icon
  const [answer, setAnswer] = useState('')
  const [challengeState, setChallengeState] = useState<ChallengeState>('idle')
  const [errorMsg, setErrorMsg] = useState('')

  const modelScores: ModelScore[] = risk_score.model_scores?.length
    ? risk_score.model_scores
    : [
        { score: risk_score.bayesian_score, probability: risk_score.bayesian_score/999, label: 'Bayesian Engine', short: 'bayesian', is_trained: true, weight_pct: 40 },
        { score: risk_score.ml_score,       probability: risk_score.ml_score/999,       label: 'XGBoost',        short: 'xgb',      is_trained: risk_score.ml_score > 0,  weight_pct: 30 },
        { score: risk_score.svm_score ?? 0, probability: (risk_score.svm_score??0)/999, label: 'SVM (RBF)',      short: 'svm',      is_trained: (risk_score.svm_score??0) > 0, weight_pct: 20 },
        { score: risk_score.knn_score ?? 0, probability: (risk_score.knn_score??0)/999, label: 'KNN (k=7)',      short: 'knn',      is_trained: (risk_score.knn_score??0) > 0, weight_pct: 10 },
      ]

  const highestContributor = modelScores.reduce((best, m) =>
    (m.score * m.weight_pct) > (best.score * best.weight_pct) ? m : best, modelScores[0])

  const handleChallengeSubmit = async () => {
    if (!answer.trim() || !challenge_question) return
    setChallengeState('submitting')
    setErrorMsg('')
    try {
      await verifyChallenge({ transaction_id, question_id: challenge_question.question_id, answer: answer.trim() })
      setChallengeState('passed')
    } catch (err: unknown) {
      setChallengeState('failed')
      const apiErr = err as { response?: { data?: { detail?: { message?: string } | string } } }
      const detail = apiErr?.response?.data?.detail
      const msg = typeof detail === 'object' && detail !== null && 'message' in detail
        ? (detail as { message: string }).message
        : typeof detail === 'string' ? detail
        : 'Incorrect answer. Please try again.'
      setErrorMsg(msg)
    }
  }

  return (
    <div className={clsx('rounded-xl border p-6 space-y-6', cfg.bg, cfg.border)}>

      {/* ── Header ── */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Icon size={22} className={cfg.text} />
          <h2 className={clsx('text-lg font-semibold', cfg.text)}>{cfg.label}</h2>
        </div>
        <span className={clsx('text-xs font-medium px-2.5 py-1 rounded-full', cfg.badge)}>
          {risk_score.outcome}
        </span>
      </div>

      {/* ── Ensemble score gauge + explanation ── */}
      <div className="flex items-center gap-5">
        <div className="flex-shrink-0">
          <ScoreGauge score={risk_score.score} />
          <p className="text-center text-xs text-slate-400 mt-1">Ensemble Score</p>
        </div>
        <div className="flex-1 space-y-3">
          <p className="text-sm text-slate-300 leading-relaxed">{risk_score.explanation}</p>
          <ModelAgreement scores={modelScores} />
          <div className="flex items-center gap-4 text-xs text-slate-500">
            <span>Confidence: <span className="text-white font-medium">{(risk_score.confidence * 100).toFixed(0)}%</span></span>
            <span>Highest driver: <span style={{ color: MODEL_META[highestContributor.short]?.colour }}
              className="font-medium">{highestContributor.label}</span></span>
          </div>
        </div>
      </div>

      {/* ── Model comparison grid ── */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
            Model Comparison — Side by Side
          </p>
          <p className="text-xs text-slate-500">Ensemble = Bayesian×40% + XGB×30% + SVM×20% + KNN×10%</p>
        </div>
        <div className="grid grid-cols-2 gap-3">
          {modelScores.map(ms => (
            <ModelScoreBar
              key={ms.short}
              ms={ms}
              isEnsembleBest={ms.short === highestContributor.short}
            />
          ))}
        </div>
      </div>

      {/* ── Risk factors ── */}
      {risk_score.risk_factors.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wide">
            Risk Factors Triggered ({risk_score.risk_factors.length})
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

      {/* ── Challenge question ── */}
      {challenge_question && (
        <div className="border border-amber-500/30 rounded-xl overflow-hidden">
          {challengeState === 'passed' && (
            <div className="bg-emerald-500/10 border-b border-emerald-500/30 px-5 py-4 flex items-center gap-3">
              <ShieldCheck size={20} className="text-emerald-400 flex-shrink-0" />
              <div>
                <p className="text-sm font-semibold text-emerald-300">Challenge Passed — Transaction Committed</p>
                <p className="text-xs text-emerald-400/80 mt-0.5">Transaction status set to COMPLETED.</p>
              </div>
            </div>
          )}
          <div className="bg-amber-500/5 px-5 py-4 space-y-4">
            <div className="flex items-start gap-2">
              <KeyRound size={16} className="text-amber-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-xs font-semibold text-amber-400 uppercase tracking-wide mb-1">Challenge Question</p>
                <p className="text-sm text-amber-200 leading-relaxed">{challenge_question.question}</p>
              </div>
            </div>
            {challengeState !== 'passed' && (
              <div className="space-y-2">
                <div className="flex gap-2">
                  <input type="text" value={answer}
                    onChange={e => { setAnswer(e.target.value); setChallengeState('idle'); setErrorMsg('') }}
                    onKeyDown={e => e.key === 'Enter' && handleChallengeSubmit()}
                    placeholder="Enter your answer…"
                    disabled={challengeState === 'submitting'}
                    className={clsx(
                      'flex-1 bg-slate-900 border rounded-lg px-3 py-2.5 text-sm text-white',
                      'placeholder-slate-500 focus:outline-none transition-colors',
                      challengeState === 'failed' ? 'border-red-500' : 'border-slate-600 focus:border-amber-400',
                    )}
                  />
                  <button onClick={handleChallengeSubmit}
                    disabled={!answer.trim() || challengeState === 'submitting'}
                    className={clsx(
                      'flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all',
                      challengeState === 'submitting'
                        ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                        : 'bg-amber-500 hover:bg-amber-400 active:scale-95 text-black',
                    )}>
                    {challengeState === 'submitting'
                      ? <span className="w-4 h-4 border-2 border-slate-500/30 border-t-slate-400 rounded-full animate-spin" />
                      : <Send size={14} />}
                    {challengeState === 'submitting' ? 'Verifying…' : 'Verify'}
                  </button>
                </div>
                {challengeState === 'failed' && errorMsg && (
                  <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2">
                    <XCircle size={14} className="text-red-400 flex-shrink-0" />
                    <p className="text-xs text-red-300">{errorMsg}</p>
                  </div>
                )}
                <p className="text-xs text-slate-500">
                  Tip: enter <span className="font-mono text-amber-400/70">TEST</span> to approve this transaction
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Footer ── */}
      <div className="flex items-center gap-4 pt-1 border-t border-slate-700/50 text-xs text-slate-500">
        <div className="flex items-center gap-1">
          <Clock size={12} />{processing_time_ms.toFixed(1)}ms
        </div>
        <div>Confidence: {(risk_score.confidence * 100).toFixed(0)}%</div>
        <div className="ml-auto font-mono truncate">{transaction_id.slice(0, 8)}…</div>
      </div>
    </div>
  )
}
