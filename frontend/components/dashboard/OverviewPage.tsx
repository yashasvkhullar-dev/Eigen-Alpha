"use client"

import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer
} from "recharts"
import { cumulativeReturns, monthlyReturnsHeatmap, sparklinePortfolio, sparklineSharpe, sparklineDrawdown, sparklineStocks } from "@/lib/mock-data"

const TEAL = "#00d2a8"
const RED = "#ff4757"

function Sparkline({ data, color }: { data: number[]; color: string }) {
  const min = Math.min(...data)
  const max = Math.max(...data)
  const h = 32
  const w = 80
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w
    const y = h - ((v - min) / (max - min || 1)) * h
    return `${x},${y}`
  }).join(" ")
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" />
    </svg>
  )
}

const metricCards = [
  { label: "Portfolio Return", value: "+18.4%", sub: "YTD 2025", color: TEAL, sparkData: sparklinePortfolio, positive: true },
  { label: "Sharpe Ratio", value: "1.24", sub: "Annualised", color: TEAL, sparkData: sparklineSharpe, positive: true },
  { label: "Max Drawdown", value: "-11.2%", sub: "Peak to trough", color: RED, sparkData: sparklineDrawdown, positive: false },
  { label: "Active Stocks", value: "47", sub: "Current holdings", color: TEAL, sparkData: sparklineStocks, positive: true },
]

const years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

function cellColor(v: number) {
  if (v > 6) return "#006644"
  if (v > 3) return "#009958"
  if (v > 0) return "#00cc77"
  if (v > -3) return "#cc3344"
  if (v > -6) return "#991133"
  return "#660022"
}

function cellText(v: number) {
  return v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1)
}

export function OverviewPage() {
  const heatByYear: Record<number, Record<string, number>> = {}
  for (const d of monthlyReturnsHeatmap) {
    if (!heatByYear[d.year]) heatByYear[d.year] = {}
    heatByYear[d.year][d.month] = d.value
  }

  return (
    <div className="space-y-5">
      {/* Metric Cards */}
      <div className="grid grid-cols-4 gap-4">
        {metricCards.map((c) => (
          <div key={c.label} className="rounded-xl border border-[#2a2d3a] bg-[#1a1d27] p-4 flex flex-col gap-2">
            <p className="text-xs text-[#8892a4] uppercase tracking-wider">{c.label}</p>
            <div className="flex items-end justify-between">
              <span className="text-2xl font-semibold" style={{ color: c.color }}>{c.value}</span>
              <Sparkline data={c.sparkData} color={c.color} />
            </div>
            <p className="text-xs text-[#8892a4]">{c.sub}</p>
          </div>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-2 gap-4">
        {/* Cumulative Returns */}
        <div className="rounded-xl border border-[#2a2d3a] bg-[#1a1d27] p-4">
          <p className="text-sm font-medium text-[#e2e8f0] mb-4">Cumulative Returns</p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={cumulativeReturns} margin={{ top: 4, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
              <XAxis dataKey="month" tick={{ fill: "#8892a4", fontSize: 10 }} tickLine={false} axisLine={false} interval={2} />
              <YAxis tick={{ fill: "#8892a4", fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={(v) => `${v}%`} />
              <Tooltip
                contentStyle={{ background: "#1a1d27", border: "1px solid #2a2d3a", borderRadius: 8 }}
                labelStyle={{ color: "#e2e8f0", fontSize: 11 }}
                itemStyle={{ fontSize: 11 }}
                formatter={(v: number) => [`${v.toFixed(2)}%`]}
              />
              <Line type="monotone" dataKey="eigen" stroke={TEAL} strokeWidth={2} dot={false} name="EigenAlpha" />
              <Line type="monotone" dataKey="nifty" stroke="#6b7280" strokeWidth={2} dot={false} name="Nifty 500" />
            </LineChart>
          </ResponsiveContainer>
          <div className="flex gap-4 mt-2">
            <div className="flex items-center gap-1.5"><span className="w-3 h-0.5 rounded-full inline-block" style={{ background: TEAL }} /><span className="text-xs text-[#8892a4]">EigenAlpha</span></div>
            <div className="flex items-center gap-1.5"><span className="w-3 h-0.5 rounded-full inline-block" style={{ background: "#6b7280" }} /><span className="text-xs text-[#8892a4]">Nifty 500</span></div>
          </div>
        </div>

        {/* Monthly Returns Heatmap */}
        <div className="rounded-xl border border-[#2a2d3a] bg-[#1a1d27] p-4">
          <p className="text-sm font-medium text-[#e2e8f0] mb-3">Monthly Returns Heatmap</p>
          <div className="overflow-x-auto">
            <table className="w-full text-[9px]" style={{ borderCollapse: "separate", borderSpacing: "2px" }}>
              <thead>
                <tr>
                  <th className="text-[#8892a4] text-left pr-2 font-normal w-8">Year</th>
                  {months.map((m) => (
                    <th key={m} className="text-[#8892a4] font-normal text-center">{m}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {years.map((yr) => (
                  <tr key={yr}>
                    <td className="text-[#8892a4] pr-2 font-medium">{yr}</td>
                    {months.map((m) => {
                      const v = heatByYear[yr]?.[m] ?? 0
                      return (
                        <td
                          key={m}
                          className="text-center rounded"
                          style={{ background: cellColor(v), color: "#e2e8f0", padding: "3px 0" }}
                          title={`${yr} ${m}: ${cellText(v)}%`}
                        >
                          {cellText(v)}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex items-center gap-2 mt-3 justify-end">
            <span className="text-[9px] text-[#8892a4]">Negative</span>
            <div className="flex gap-0.5">
              {["#660022","#991133","#cc3344","#00cc77","#009958","#006644"].map((c) => (
                <span key={c} className="w-4 h-2.5 rounded-sm inline-block" style={{ background: c }} />
              ))}
            </div>
            <span className="text-[9px] text-[#8892a4]">Positive</span>
          </div>
        </div>
      </div>
    </div>
  )
}
