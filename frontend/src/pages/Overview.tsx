/**
 * Global Overview — visible to all authenticated users (admin + viewer).
 * Shows aggregated transaction data and a world heatmap.
 * NO customer PII is displayed here.
 */
import { useQuery } from '@tanstack/react-query'
import {
  Globe, TrendingUp, AlertTriangle, BarChart2,
  ShieldOff, Layers, CheckCircle,
} from 'lucide-react'
import { getAggregateWorldMap, getTransactionStats, type AggregateWorldMapEntry } from '../api/client'
import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import * as topojson from 'topojson-client'
import clsx from 'clsx'

// ── Risk colour helpers ────────────────────────────────────────────────────────

const RISK_COLOR: Record<string, string> = {
  LOW:      '#22c55e',
  MEDIUM:   '#f59e0b',
  HIGH:     '#ef4444',
  CRITICAL: '#a855f7',
}

const RISK_BG: Record<string, string> = {
  LOW:      'bg-green-500/20 text-green-400 border-green-500/30',
  MEDIUM:   'bg-amber-500/20 text-amber-400 border-amber-500/30',
  HIGH:     'bg-red-500/20 text-red-400 border-red-500/30',
  CRITICAL: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
}

// ── World Map component ────────────────────────────────────────────────────────

function WorldHeatMap({ countries }: { countries: AggregateWorldMapEntry[] }) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [tooltip, setTooltip] = useState<{ x: number; y: number; entry: AggregateWorldMapEntry } | null>(null)

  useEffect(() => {
    if (!svgRef.current || countries.length === 0) return
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const W = svgRef.current.clientWidth || 900
    const H = 420

    const projection = d3.geoNaturalEarth1()
      .scale(W / 6.5)
      .translate([W / 2, H / 2])

    const path = d3.geoPath().projection(projection)

    // Ocean gradient
    const defs = svg.append('defs')
    const grad = defs.append('linearGradient')
      .attr('id', 'ocean-grad').attr('x1', '0%').attr('y1', '0%').attr('x2', '0%').attr('y2', '100%')
    grad.append('stop').attr('offset', '0%').attr('stop-color', '#0f172a')
    grad.append('stop').attr('offset', '100%').attr('stop-color', '#1e293b')

    svg.append('rect').attr('width', W).attr('height', H).attr('fill', 'url(#ocean-grad)')

    // Graticule
    svg.append('path')
      .datum(d3.geoGraticule()())
      .attr('d', path as never)
      .attr('fill', 'none').attr('stroke', '#334155').attr('stroke-width', 0.3)

    const byCode = new Map(countries.map(c => [c.code, c]))

    fetch('https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json')
      .then(r => r.json())
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      .then((world: any) => {
        const features = topojson.feature(world, world.objects.countries) as unknown as {
          features: { id: string; properties: { name: string } }[]
        }

        // Country fills
        svg.append('g').selectAll('path')
          .data(features.features)
          .join('path')
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          .attr('d', path as any)
          .attr('fill', d => {
            const entry = byCode.get(d.id) || byCode.get(d.properties.name)
            return entry ? d3.color(RISK_COLOR[entry.risk_level])!.copy({ opacity: 0.45 }).formatHsl() : '#1e293b'
          })
          .attr('stroke', '#334155').attr('stroke-width', 0.4)

        // Country borders
        svg.append('path')
          .datum(topojson.mesh(world, world.objects.countries))
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          .attr('d', path as any)
          .attr('fill', 'none').attr('stroke', '#475569').attr('stroke-width', 0.2)

        // Bubbles at centroids
        countries.forEach(entry => {
          const feature = features.features
            .find(f => f.properties.name === entry.name || f.properties.name === entry.code)
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const centroid = feature ? projection(path.centroid(feature as any)) : null
          if (!centroid) return

          const r = Math.max(5, Math.min(30, Math.sqrt(entry.txn_count) * 4))
          const color = RISK_COLOR[entry.risk_level]

          if (entry.fraud_count > 0) {
            svg.append('circle')
              .attr('cx', centroid[0]).attr('cy', centroid[1]).attr('r', r + 4)
              .attr('fill', 'none').attr('stroke', color).attr('stroke-width', 1.5)
              .attr('opacity', 0.4)
              .append('animate')
              .attr('attributeName', 'r').attr('from', r).attr('to', r + 10)
              .attr('dur', '2s').attr('repeatCount', 'indefinite')
          }

          svg.append('circle')
            .attr('cx', centroid[0]).attr('cy', centroid[1]).attr('r', r)
            .attr('fill', color).attr('fill-opacity', 0.75)
            .attr('stroke', color).attr('stroke-width', 1)
            .attr('cursor', 'pointer')
            .on('mousemove', (event: MouseEvent) => {
              const rect = svgRef.current!.getBoundingClientRect()
              setTooltip({ x: event.clientX - rect.left, y: event.clientY - rect.top, entry })
            })
            .on('mouseleave', () => setTooltip(null))
        })
      })
  }, [countries])

  return (
    <div className="relative w-full">
      <svg ref={svgRef} className="w-full" style={{ height: 420 }} />
      {tooltip && (
        <div
          className="absolute z-10 bg-slate-800 border border-slate-600 rounded-lg p-3 shadow-xl text-xs pointer-events-none min-w-[200px]"
          style={{ left: tooltip.x + 12, top: tooltip.y - 10 }}
        >
          <div className="flex items-center justify-between mb-1.5">
            <span className="font-bold text-white">{tooltip.entry.name}</span>
            <span className={clsx('px-1.5 py-0.5 rounded text-[10px] font-bold border', RISK_BG[tooltip.entry.risk_level])}>
              {tooltip.entry.risk_level}
            </span>
          </div>
          <div className="space-y-1 text-slate-300">
            <div className="flex justify-between gap-4">
              <span className="text-slate-400">Transactions</span>
              <span className="font-semibold">{tooltip.entry.txn_count.toLocaleString()}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-slate-400">Fraud cases</span>
              <span className={tooltip.entry.fraud_count > 0 ? 'text-red-400 font-semibold' : 'font-semibold'}>
                {tooltip.entry.fraud_count}
              </span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-slate-400">Volume</span>
              <span className="font-semibold">${(tooltip.entry.total_amount / 1000).toFixed(0)}K</span>
            </div>
            {tooltip.entry.avg_score !== null && (
              <div className="flex justify-between gap-4">
                <span className="text-slate-400">Avg score</span>
                <span className="font-semibold">{tooltip.entry.avg_score}/999</span>
              </div>
            )}
            <div className="flex justify-between gap-4">
              <span className="text-slate-400">FATF</span>
              <span className="font-semibold">{tooltip.entry.fatf_risk}</span>
            </div>
            {tooltip.entry.fraud_types.length > 0 && (
              <div className="pt-1 border-t border-slate-700">
                <span className="text-slate-400">Fraud patterns: </span>
                <span className="text-red-400">{tooltip.entry.fraud_types.join(', ')}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Stat card ─────────────────────────────────────────────────────────────────

function StatCard({ label, value, icon: Icon, sub, accent = 'blue' }: {
  label: string; value: string | number; icon: React.ElementType
  sub?: string; accent?: string
}) {
  const colors: Record<string, string> = {
    blue:    'text-blue-400 bg-blue-500/10',
    red:     'text-red-400 bg-red-500/10',
    amber:   'text-amber-400 bg-amber-500/10',
    emerald: 'text-emerald-400 bg-emerald-500/10',
    purple:  'text-purple-400 bg-purple-500/10',
  }
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
      <div className="flex items-center justify-between mb-3">
        <p className="text-sm text-slate-400">{label}</p>
        <div className={clsx('w-8 h-8 rounded-lg flex items-center justify-center', colors[accent])}>
          <Icon size={16} className={colors[accent].split(' ')[0]} />
        </div>
      </div>
      <p className="text-2xl font-bold text-white">{value}</p>
      {sub && <p className="text-xs text-slate-500 mt-1">{sub}</p>}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Overview() {
  const { data: mapData, isLoading: mapLoading } = useQuery({
    queryKey: ['aggregate-world-map'],
    queryFn: getAggregateWorldMap,
  })
  const { data: stats } = useQuery({
    queryKey: ['txn-stats'],
    queryFn: getTransactionStats,
  })

  const countries = mapData?.countries ?? []
  const summary   = mapData?.summary ?? {}

  const topCountries = [...countries]
    .sort((a, b) => b.txn_count - a.txn_count)
    .slice(0, 8)

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Globe size={22} className="text-blue-400" />
            Global Transaction Overview
          </h1>
          <p className="text-slate-400 text-sm mt-0.5">
            Aggregated view — no customer information displayed
          </p>
        </div>
        <div className="flex items-center gap-1.5 bg-emerald-500/10 border border-emerald-500/30 rounded-lg px-3 py-1.5">
          <CheckCircle size={14} className="text-emerald-400" />
          <span className="text-xs text-emerald-400 font-medium">PII Protected</span>
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Total Transactions"
          value={(summary.total_transactions ?? stats?.total_transactions ?? 0).toLocaleString()}
          icon={Layers}
          accent="blue"
        />
        <StatCard
          label="Fraud Detected"
          value={(summary.total_fraud ?? stats?.fraud_count ?? 0).toLocaleString()}
          icon={AlertTriangle}
          sub={`${summary.fraud_rate_pct ?? stats?.fraud_rate_pct ?? 0}% fraud rate`}
          accent="red"
        />
        <StatCard
          label="Countries Involved"
          value={summary.total_countries ?? countries.length}
          icon={Globe}
          accent="amber"
        />
        <StatCard
          label="High-Risk Countries"
          value={summary.high_risk_countries ?? 0}
          icon={ShieldOff}
          sub="FATF grey/blacklist"
          accent="purple"
        />
      </div>

      {/* World Map */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-white">Transaction Heatmap</h2>
            <p className="text-xs text-slate-400 mt-0.5">Circle size = transaction count · Colour = risk level</p>
          </div>
          <div className="flex items-center gap-3">
            {Object.entries(RISK_COLOR).map(([level, color]) => (
              <span key={level} className="flex items-center gap-1 text-xs text-slate-400">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                {level}
              </span>
            ))}
          </div>
        </div>
        <div className="p-4">
          {mapLoading ? (
            <div className="h-[420px] flex items-center justify-center">
              <div className="w-8 h-8 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
            </div>
          ) : (
            <WorldHeatMap countries={countries} />
          )}
        </div>
      </div>

      {/* Top countries table */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-800">
          <h2 className="text-sm font-semibold text-white flex items-center gap-2">
            <BarChart2 size={16} className="text-blue-400" />
            Top Countries by Activity
          </h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800">
                {['Country', 'Risk', 'Transactions', 'Fraud Cases', 'Volume (USD)', 'Avg Score', 'FATF'].map(h => (
                  <th key={h} className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {topCountries.map(c => (
                <tr key={c.code} className="hover:bg-slate-800/50 transition-colors">
                  <td className="px-4 py-3 font-medium text-white">
                    <span>{c.name}</span>
                    <span className="ml-1.5 text-slate-500 text-xs">{c.code}</span>
                  </td>
                  <td className="px-4 py-3">
                    <span className={clsx('px-2 py-0.5 rounded text-xs font-semibold border', RISK_BG[c.risk_level])}>
                      {c.risk_level}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-slate-300">{c.txn_count.toLocaleString()}</td>
                  <td className="px-4 py-3">
                    {c.fraud_count > 0 ? (
                      <span className="text-red-400 font-semibold">{c.fraud_count}</span>
                    ) : (
                      <span className="text-slate-500">0</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-slate-300">
                    ${c.total_amount >= 1_000_000
                      ? `${(c.total_amount / 1_000_000).toFixed(1)}M`
                      : `${(c.total_amount / 1000).toFixed(0)}K`}
                  </td>
                  <td className="px-4 py-3 text-slate-300">
                    {c.avg_score !== null ? (
                      <span className={c.avg_score >= 700 ? 'text-red-400' : c.avg_score >= 400 ? 'text-amber-400' : 'text-green-400'}>
                        {c.avg_score}
                      </span>
                    ) : '—'}
                  </td>
                  <td className="px-4 py-3">
                    <span className={clsx('text-xs', {
                      'text-red-400': c.fatf_risk === 'BLACKLIST' || c.fatf_risk === 'HIGH',
                      'text-amber-400': c.fatf_risk === 'MEDIUM',
                      'text-slate-400': c.fatf_risk === 'LOW',
                    })}>
                      {c.fatf_risk}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Fraud type breakdown */}
      {countries.some(c => c.fraud_types.length > 0) && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
          <h2 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
            <TrendingUp size={16} className="text-red-400" />
            Fraud Pattern Distribution
          </h2>
          <div className="flex flex-wrap gap-2">
            {Array.from(
              countries.reduce((acc, c) => {
                c.fraud_types.forEach(ft => acc.set(ft, (acc.get(ft) ?? 0) + c.fraud_count))
                return acc
              }, new Map<string, number>())
            )
              .sort((a, b) => b[1] - a[1])
              .map(([ft, count]) => (
                <div
                  key={ft}
                  className="bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-1.5"
                >
                  <span className="text-xs font-medium text-red-300">{ft}</span>
                  <span className="ml-1.5 text-xs text-red-500">{count} cases</span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  )
}
