/**
 * TransactionWorldMap
 * -------------------
 * D3-powered Natural Earth SVG world map that plots customer transaction activity
 * as coloured circles at each country centroid.
 *
 * Circle size  → transaction count
 * Circle colour → max risk level (green / amber / orange / red)
 * Pulsing ring  → country has fraud-labelled transactions
 */
import React, { useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import * as d3 from 'd3'
import * as topojson from 'topojson-client'
import type { Topology, GeometryCollection } from 'topojson-specification'
import { getTransactionMap, type CountryMapEntry } from '../api/client'
import clsx from 'clsx'

// ── Constants ─────────────────────────────────────────────────────────────────

const RISK_COLORS = {
  LOW:      { fill: '#10b981', stroke: '#059669', label: 'Low (0–399)',      badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30' },
  MEDIUM:   { fill: '#f59e0b', stroke: '#d97706', label: 'Medium (400–699)', badge: 'bg-amber-500/20 text-amber-300 border-amber-500/30' },
  HIGH:     { fill: '#f97316', stroke: '#ea580c', label: 'High (400–699)',   badge: 'bg-orange-500/20 text-orange-300 border-orange-500/30' },
  CRITICAL: { fill: '#ef4444', stroke: '#dc2626', label: 'Critical (700+)',  badge: 'bg-red-500/20 text-red-300 border-red-500/30' },
}

const WORLD_ATLAS_URL =
  'https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json'

// ── Types ─────────────────────────────────────────────────────────────────────

interface TooltipState {
  x: number
  y: number
  entry: CountryMapEntry
}

// ── Component ─────────────────────────────────────────────────────────────────

export default React.memo(function TransactionWorldMap({ customerId }: { customerId: string }) {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [tooltip, setTooltip] = useState<TooltipState | null>(null)
  const [worldData, setWorldData] = useState<Topology | null>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 420 })

  const { data, isLoading } = useQuery({
    queryKey: ['txn-map', customerId],
    queryFn: () => getTransactionMap(customerId),
    staleTime: 30_000,
  })

  // Fetch world topology once
  useEffect(() => {
    fetch(WORLD_ATLAS_URL)
      .then(r => r.json())
      .then(setWorldData)
      .catch(console.error)
  }, [])

  // Observe container width for responsive sizing
  useEffect(() => {
    if (!containerRef.current) return
    const ro = new ResizeObserver(entries => {
      const { width } = entries[0].contentRect
      setDimensions({ width, height: Math.round(width * 0.52) })
    })
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [])

  // Draw map whenever data or dimensions change
  useEffect(() => {
    if (!svgRef.current || !worldData || !data) return

    const { width, height } = dimensions
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const projection = d3.geoNaturalEarth1()
      .scale(width / 6.5)
      .translate([width / 2, height / 2 + 20])

    const path = d3.geoPath().projection(projection)

    // Background gradient
    const defs = svg.append('defs')
    const grad = defs.append('linearGradient')
      .attr('id', 'ocean-grad')
      .attr('x1', '0%').attr('y1', '0%').attr('x2', '0%').attr('y2', '100%')
    grad.append('stop').attr('offset', '0%').attr('stop-color', '#0f172a')
    grad.append('stop').attr('offset', '100%').attr('stop-color', '#0c1324')

    svg.append('rect')
      .attr('width', width).attr('height', height)
      .attr('fill', 'url(#ocean-grad)')
      .attr('rx', 12)

    // Graticule (grid lines)
    const graticule = d3.geoGraticule()()
    svg.append('path')
      .datum(graticule)
      .attr('d', path)
      .attr('fill', 'none')
      .attr('stroke', '#1e3a5f')
      .attr('stroke-width', 0.3)
      .attr('opacity', 0.5)

    // Country shapes
    const countries = topojson.feature(
      worldData as Topology<{ countries: GeometryCollection }>,
      (worldData as Topology<{ countries: GeometryCollection }>).objects.countries,
    )

    // Build country → entry lookup
    const entryByCode = new Map<string, CountryMapEntry>()
    data.countries.forEach(e => entryByCode.set(e.code, e))

    // Country ISO numeric → alpha-2 mapping (subset of world)
    const NUM_TO_ALPHA2: Record<string, string> = {
      '840': 'US', '826': 'GB', '036': 'AU', '124': 'CA', '276': 'DE',
      '250': 'FR', '392': 'JP', '702': 'SG', '356': 'IN', '156': 'CN',
      '076': 'BR', '484': 'MX', '410': 'KR', '528': 'NL', '380': 'IT',
      '724': 'ES', '756': 'CH', '752': 'SE', '578': 'NO', '208': 'DK',
      '246': 'FI', '040': 'AT', '056': 'BE', '620': 'PT', '372': 'IE',
      '616': 'PL', '203': 'CZ', '348': 'HU', '300': 'GR', '191': 'HR',
      '804': 'UA', '643': 'RU', '792': 'TR', '682': 'SA', '784': 'AE',
      '376': 'IL', '818': 'EG', '710': 'ZA', '566': 'NG', '404': 'KE',
      '360': 'ID', '458': 'MY', '764': 'TH', '704': 'VN', '608': 'PH',
      '554': 'NZ', '032': 'AR', '152': 'CL', '170': 'CO', '604': 'PE',
      '858': 'UY', '364': 'IR', '408': 'KP', '760': 'SY', '104': 'MM',
      '862': 'VE', '050': 'BD', '586': 'PK', '004': 'AF', '144': 'LK',
      '688': 'RS', '100': 'BG', '498': 'MD', '703': 'SK', '440': 'LT',
      '428': 'LV', '233': 'EE', '417': 'KG', '398': 'KZ', '860': 'UZ',
    }

    // Draw country fills
    svg.append('g')
      .selectAll('path')
      .data((countries as d3.GeoPermissibleObjects & { features: unknown[] }).features)
      .join('path')
      .attr('d', d => path(d as d3.GeoPermissibleObjects) ?? '')
      .attr('fill', (d: unknown) => {
        const feat = d as { id?: string }
        const alpha = NUM_TO_ALPHA2[String(feat.id ?? '').padStart(3, '0')]
        const entry = alpha ? entryByCode.get(alpha) : undefined
        if (!entry) return '#1e293b'
        const base = RISK_COLORS[entry.risk_level]
        return base.fill + '22'
      })
      .attr('stroke', '#334155')
      .attr('stroke-width', 0.4)

    // Country borders
    svg.append('path')
      .datum(topojson.mesh(
        worldData as Topology<{ countries: GeometryCollection }>,
        (worldData as Topology<{ countries: GeometryCollection }>).objects.countries,
        (a, b) => a !== b,
      ))
      .attr('d', path)
      .attr('fill', 'none')
      .attr('stroke', '#475569')
      .attr('stroke-width', 0.3)

    // ── Place circles ─────────────────────────────────────────────────────────

    // Centroid lookup from features
    const centroids = new Map<string, [number, number]>()
    ;(countries as d3.GeoPermissibleObjects & { features: unknown[] }).features.forEach((f: unknown) => {
      const feat = f as { id?: string }
      const alpha = NUM_TO_ALPHA2[String(feat.id ?? '').padStart(3, '0')]
      if (!alpha) return
      const centroid = projection(d3.geoCentroid(f as d3.GeoPermissibleObjects))
      if (centroid && isFinite(centroid[0]) && isFinite(centroid[1])) {
        centroids.set(alpha, centroid as [number, number])
      }
    })

    const maxCount = Math.max(...data.countries.map(e => e.txn_count))
    const rScale = d3.scaleSqrt().domain([1, maxCount]).range([6, 28])

    // Pulse rings for fraud countries
    const fraudCountries = data.countries.filter(e => e.fraud_count > 0)
    fraudCountries.forEach(entry => {
      const pos = centroids.get(entry.code)
      if (!pos) return
      const col = RISK_COLORS[entry.risk_level]
      const r = rScale(entry.txn_count)

      const pulseG = svg.append('g')
        .attr('transform', `translate(${pos[0]},${pos[1]})`)

      // Outer pulsing ring
      pulseG.append('circle')
        .attr('r', r + 4)
        .attr('fill', 'none')
        .attr('stroke', col.stroke)
        .attr('stroke-width', 1.5)
        .attr('opacity', 0.6)
        .append('animate')
        .attr('attributeName', 'r')
        .attr('values', `${r + 2};${r + 10};${r + 2}`)
        .attr('dur', '2.5s')
        .attr('repeatCount', 'indefinite')

      pulseG.select('circle')
        .append('animate')
        .attr('attributeName', 'opacity')
        .attr('values', '0.6;0;0.6')
        .attr('dur', '2.5s')
        .attr('repeatCount', 'indefinite')
    })

    // Main circles
    const circleG = svg.append('g')
    data.countries.forEach(entry => {
      const pos = centroids.get(entry.code)
      if (!pos) return

      const col = RISK_COLORS[entry.risk_level]
      const r = rScale(entry.txn_count)

      const g = circleG.append('g')
        .attr('transform', `translate(${pos[0]},${pos[1]})`)
        .style('cursor', 'pointer')

      // Glow
      g.append('circle')
        .attr('r', r + 5)
        .attr('fill', col.fill)
        .attr('opacity', 0.08)

      // Main circle
      g.append('circle')
        .attr('r', r)
        .attr('fill', col.fill)
        .attr('fill-opacity', 0.85)
        .attr('stroke', col.stroke)
        .attr('stroke-width', 1.5)

      // Count label
      if (r > 10) {
        g.append('text')
          .attr('text-anchor', 'middle')
          .attr('dy', '0.35em')
          .attr('fill', 'white')
          .attr('font-size', Math.min(r * 0.7, 13))
          .attr('font-weight', '700')
          .attr('pointer-events', 'none')
          .text(entry.txn_count)
      }

      // Country code label below circle
      g.append('text')
        .attr('text-anchor', 'middle')
        .attr('dy', r + 12)
        .attr('fill', '#94a3b8')
        .attr('font-size', 9)
        .attr('pointer-events', 'none')
        .text(entry.code)

      // Hover interaction
      g.on('mouseenter', function (event: MouseEvent) {
        d3.select(this).select('circle:nth-child(2)')
          .attr('fill-opacity', 1)
          .attr('stroke-width', 2.5)
        const rect = svgRef.current!.getBoundingClientRect()
        setTooltip({
          x: event.clientX - rect.left,
          y: event.clientY - rect.top,
          entry,
        })
      })
      .on('mousemove', function (event: MouseEvent) {
        const rect = svgRef.current!.getBoundingClientRect()
        setTooltip(prev => prev ? { ...prev, x: event.clientX - rect.left, y: event.clientY - rect.top } : null)
      })
      .on('mouseleave', function () {
        d3.select(this).select('circle:nth-child(2)')
          .attr('fill-opacity', 0.85)
          .attr('stroke-width', 1.5)
        setTooltip(null)
      })
    })

  }, [worldData, data, dimensions])

  const countries = data?.countries ?? []

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-5">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-200">Transaction Geography</h3>
          <p className="text-xs text-slate-500 mt-0.5">
            {isLoading ? 'Loading…' : `${countries.length} countr${countries.length !== 1 ? 'ies' : 'y'} involved — circle size = volume, colour = risk level`}
          </p>
        </div>
        {/* Legend */}
        <div className="flex items-center gap-2 flex-wrap justify-end">
          {(Object.entries(RISK_COLORS) as [string, typeof RISK_COLORS.LOW][]).map(([key, val]) => (
            <span key={key} className={clsx('text-xs px-2 py-0.5 rounded border font-medium', val.badge)}>
              {key}
            </span>
          ))}
          <span className="text-xs text-slate-500 border border-slate-600 rounded px-2 py-0.5">
            ~ pulsing = fraud
          </span>
        </div>
      </div>

      {/* Map */}
      <div ref={containerRef} className="relative w-full rounded-lg overflow-hidden">
        {(isLoading || !worldData) && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-950/80 z-10 rounded-lg">
            <div className="flex flex-col items-center gap-2">
              <div className="w-6 h-6 border-2 border-blue-500/30 border-t-blue-400 rounded-full animate-spin" />
              <p className="text-xs text-slate-400">Loading world map…</p>
            </div>
          </div>
        )}

        {!isLoading && worldData && countries.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <p className="text-slate-500 text-sm">No transaction geography data available</p>
          </div>
        )}

        <svg
          ref={svgRef}
          width={dimensions.width}
          height={dimensions.height}
          className="w-full h-auto block"
        />

        {/* Tooltip */}
        {tooltip && (
          <div
            className="absolute z-20 pointer-events-none"
            style={{
              left: tooltip.x + 14,
              top: tooltip.y - 10,
              transform: tooltip.x > dimensions.width * 0.65 ? 'translateX(-110%)' : undefined,
            }}
          >
            <div className="bg-slate-800 border border-slate-600 rounded-xl shadow-2xl p-4 min-w-52 text-xs space-y-2.5">
              <div className="flex items-center justify-between gap-3">
                <span className="font-semibold text-sm text-white">{tooltip.entry.name}</span>
                <span className={clsx(
                  'text-xs font-bold px-2 py-0.5 rounded border',
                  RISK_COLORS[tooltip.entry.risk_level].badge
                )}>{tooltip.entry.risk_level}</span>
              </div>

              <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-slate-300">
                <span className="text-slate-500">Transactions</span>
                <span className="font-bold text-white text-right">{tooltip.entry.txn_count}</span>

                <span className="text-slate-500">Fraud events</span>
                <span className={clsx('font-bold text-right', tooltip.entry.fraud_count > 0 ? 'text-red-400' : 'text-emerald-400')}>
                  {tooltip.entry.fraud_count}
                </span>

                <span className="text-slate-500">Total amount</span>
                <span className="font-medium text-white text-right">
                  ${tooltip.entry.total_amount.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </span>

                {tooltip.entry.avg_score !== null && (
                  <>
                    <span className="text-slate-500">Avg risk score</span>
                    <span className={clsx('font-bold text-right',
                      tooltip.entry.avg_score <= 399 ? 'text-emerald-400'
                      : tooltip.entry.avg_score <= 699 ? 'text-amber-400'
                      : 'text-red-400'
                    )}>{tooltip.entry.avg_score}</span>
                  </>
                )}

                {tooltip.entry.max_score !== null && (
                  <>
                    <span className="text-slate-500">Max risk score</span>
                    <span className={clsx('font-bold text-right',
                      tooltip.entry.max_score <= 399 ? 'text-emerald-400'
                      : tooltip.entry.max_score <= 699 ? 'text-amber-400'
                      : 'text-red-400'
                    )}>{tooltip.entry.max_score}</span>
                  </>
                )}

                <span className="text-slate-500">FATF risk</span>
                <span className={clsx('font-medium text-right',
                  tooltip.entry.fatf_risk === 'HIGH' || tooltip.entry.fatf_risk === 'BLACKLIST'
                    ? 'text-red-400'
                    : tooltip.entry.fatf_risk === 'MEDIUM' ? 'text-amber-400'
                    : 'text-emerald-400'
                )}>{tooltip.entry.fatf_risk}</span>

                <span className="text-slate-500">Role</span>
                <span className="text-white text-right capitalize">{tooltip.entry.directions.join(' & ')}</span>
              </div>

              {tooltip.entry.fraud_types.length > 0 && (
                <div className="pt-1 border-t border-slate-700">
                  <p className="text-slate-500 mb-1">Fraud patterns:</p>
                  <div className="flex flex-wrap gap-1">
                    {tooltip.entry.fraud_types.map(ft => (
                      <span key={ft} className="bg-red-500/15 text-red-300 px-1.5 py-0.5 rounded text-xs">
                        {ft.replace(/_/g, ' ')}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Country summary bar */}
      {countries.length > 0 && (
        <div className="mt-4 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
          {countries.slice(0, 8).map(entry => {
            const col = RISK_COLORS[entry.risk_level]
            return (
              <div
                key={entry.code}
                className="flex items-center gap-2 bg-slate-800/50 rounded-lg px-3 py-2 text-xs"
              >
                <div className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{ backgroundColor: col.fill }} />
                <div className="min-w-0">
                  <p className="text-white font-medium truncate">{entry.name}</p>
                  <p className="text-slate-500">
                    {entry.txn_count} txn{entry.txn_count !== 1 ? 's' : ''}
                    {entry.fraud_count > 0 && (
                      <span className="text-red-400 ml-1">· {entry.fraud_count} fraud</span>
                    )}
                  </p>
                </div>
                {entry.max_score !== null && (
                  <span className={clsx('ml-auto font-bold flex-shrink-0',
                    entry.max_score <= 399 ? 'text-emerald-400'
                    : entry.max_score <= 699 ? 'text-amber-400'
                    : 'text-red-400'
                  )}>{entry.max_score}</span>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
})
