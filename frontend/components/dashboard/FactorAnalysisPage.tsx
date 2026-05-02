"use client"

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Line, ComposedChart, ReferenceLine,
  Cell
} from "recharts"
import { factorStats, factorICData, correlationMatrix, icDecayData } from "@/lib/mock-data"

const TEAL = "#00d2a8"
const RED = "#ff4757"

function factorRollingMean(data: typeof factorICData, key: "momentum"|"realVol"|"volTrend", window=6) {
  return data.map((d, i) => {
    const slice = data.slice(Math.max(0, i - window + 1), i + 1)
    const avg = slice.reduce((s, x) => s + x[key], 0) / slice.length
    return { ...d, rolling: parseFloat(avg.toFixed(4)) }
  })
}

function corrColor(v: number) {
  if (v >= 0.8) return "#006644"
  if (v >= 0.4) return "#009958"
  if (v >= 0.1) return "#1d4d3a"
  if (v >= -0.1) return "#2a2d3a"
  if (v >= -0.4) return "#4d1d1d"
  return "#660022"
}

const corrFactors = ["Momentum", "Realised Vol", "Volume Trend"]

export function FactorAnalysisPage() {
  return (
    <div className="space-y-5">
      {/* Factor Cards */}
      <div className="grid grid-cols-3 gap-4">
        {factorStats.map((f) => {
          const key = f.key as "momentum" | "realVol" | "volTrend"
          const chartData = factorRollingMean(factorICData, key, 6)
          return (
            <div key={f.name} className="rounded-xl border border-[#2a2d3a] bg-[#1a1d27] p-4">
              <p className="text-sm font-semibold text-[#e2e8f0] mb-3">{f.name}</p>
              <div className="grid grid-cols-3 gap-2 mb-4">
                <div>
                  <p className="text-[10px] text-[#8892a4] uppercase tracking-wide">Mean IC</p>
                  <p className="text-base font-semibold mt-0.5" style={{ color: f.meanIC >= 0 ? TEAL : RED }}>
                    {f.meanIC >= 0 ? "+" : ""}{f.meanIC.toFixed(3)}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-[#8892a4] uppercase tracking-wide">IR</p>
                  <p className="text-base font-semibold mt-0.5" style={{ color: f.IR >= 0 ? TEAL : RED }}>
                    {f.IR >= 0 ? "+" : ""}{f.IR.toFixed(2)}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-[#8892a4] uppercase tracking-wide">+ve Months</p>
                  <p className="text-base font-semibold mt-0.5 text-[#e2e8f0]">{f.posMonths}%</p>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={100}>
                <ComposedChart data={chartData} margin={{ top: 0, right: 4, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" vertical={false} />
                  <XAxis dataKey="month" tick={false} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#8892a4", fontSize: 9 }} tickLine={false} axisLine={false} tickFormatter={(v) => v.toFixed(2)} width={32} />
                  <ReferenceLine y={0} stroke="#2a2d3a" />
                  <Tooltip
                    contentStyle={{ background: "#1a1d27", border: "1px solid #2a2d3a", borderRadius: 8, fontSize: 10 }}
                    formatter={(v: number) => [v.toFixed(4)]}
                  />
                  <Bar dataKey={key} maxBarSize={6} radius={[1,1,0,0]}>
                    {chartData.map((d, i) => (
                      <Cell key={i} fill={d[key] >= 0 ? TEAL : RED} />
                    ))}
                  </Bar>
                  <Line type="monotone" dataKey="rolling" stroke="#f7931a" strokeWidth={1.5} dot={false} name="6M Rolling" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          )
        })}
      </div>

      {/* Correlation Matrix */}
      <div className="rounded-xl border border-[#2a2d3a] bg-[#1a1d27] p-4">
        <p className="text-sm font-medium text-[#e2e8f0] mb-4">Factor Correlation Matrix</p>
        <div className="flex justify-center">
          <table className="text-xs" style={{ borderSpacing: "4px", borderCollapse: "separate" }}>
            <thead>
              <tr>
                <th className="w-28" />
                {corrFactors.map((f) => (
                  <th key={f} className="text-[#8892a4] font-medium text-center px-2 py-1 w-28">{f}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {correlationMatrix.map((row) => (
                <tr key={row.factor}>
                  <td className="text-[#8892a4] pr-4 font-medium text-right">{row.factor}</td>
                  {corrFactors.map((f) => {
                    const v = row[f as keyof typeof row] as number
                    return (
                      <td
                        key={f}
                        className="text-center rounded-lg text-[#e2e8f0] font-semibold"
                        style={{ background: corrColor(v), width: 112, height: 48, verticalAlign: "middle" }}
                      >
                        {v.toFixed(2)}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* IC Decay */}
      <div className="rounded-xl border border-[#2a2d3a] bg-[#1a1d27] p-4">
        <p className="text-sm font-medium text-[#e2e8f0] mb-4">IC Decay</p>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={icDecayData} barGap={4} margin={{ top: 0, right: 12, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" vertical={false} />
            <XAxis dataKey="lag" tick={{ fill: "#8892a4", fontSize: 11 }} tickLine={false} axisLine={false} />
            <YAxis tick={{ fill: "#8892a4", fontSize: 11 }} tickLine={false} axisLine={false} tickFormatter={(v) => v.toFixed(3)} width={44} />
            <ReferenceLine y={0} stroke="#2a2d3a" />
            <Tooltip
              contentStyle={{ background: "#1a1d27", border: "1px solid #2a2d3a", borderRadius: 8, fontSize: 11 }}
              formatter={(v: number) => [v.toFixed(4)]}
            />
            <Bar dataKey="momentum" name="Momentum" fill={TEAL} radius={[2,2,0,0]} maxBarSize={20} />
            <Bar dataKey="realVol" name="Realised Vol" fill={RED} radius={[2,2,0,0]} maxBarSize={20} />
            <Bar dataKey="volTrend" name="Volume Trend" fill="#f7931a" radius={[2,2,0,0]} maxBarSize={20} />
          </BarChart>
        </ResponsiveContainer>
        <div className="flex gap-5 mt-2 justify-center">
          {[["Momentum", TEAL], ["Realised Vol", RED], ["Volume Trend", "#f7931a"]].map(([label, color]) => (
            <div key={label} className="flex items-center gap-1.5">
              <span className="w-3 h-2 rounded-sm inline-block" style={{ background: color }} />
              <span className="text-xs text-[#8892a4]">{label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
