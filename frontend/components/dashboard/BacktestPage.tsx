"use client"

import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, AreaChart, Area, ReferenceLine
} from "recharts"
import { quintileData, drawdownData, performanceTable } from "@/lib/mock-data"

const quintileColors = ["#ff4757", "#f7931a", "#f7b731", "#00b4d8", "#00d2a8"]

const perf = performanceTable

function fmt(v: number, pct = false) {
  const sign = v >= 0 ? "+" : ""
  return pct ? `${sign}${v.toFixed(1)}%` : `${sign}${v.toFixed(2)}`
}

function PerfCell({ v, bold = false, colorize = false }: { v: string; bold?: boolean; colorize?: boolean }) {
  const num = parseFloat(v.replace(/[+%]/g, ""))
  const color = colorize ? (num >= 0 ? "#00d2a8" : "#ff4757") : "#e2e8f0"
  return (
    <td
      className="px-3 py-2.5 text-right text-xs"
      style={{ color, fontWeight: bold ? 600 : 400 }}
    >
      {v}
    </td>
  )
}

export function BacktestPage() {
  return (
    <div className="space-y-5">
      {/* Quintile Performance */}
      <div className="rounded-xl border border-[#2a2d3a] bg-[#1a1d27] p-4">
        <p className="text-sm font-medium text-[#e2e8f0] mb-4">Quintile Performance (2020–2025)</p>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={quintileData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
            <XAxis
              dataKey="month" tick={{ fill: "#8892a4", fontSize: 10 }}
              tickLine={false} axisLine={false} interval={3}
            />
            <YAxis
              tick={{ fill: "#8892a4", fontSize: 10 }}
              tickLine={false} axisLine={false}
              tickFormatter={(v) => `${v}%`} width={36}
            />
            <ReferenceLine y={0} stroke="#2a2d3a" />
            <Tooltip
              contentStyle={{ background: "#1a1d27", border: "1px solid #2a2d3a", borderRadius: 8, fontSize: 11 }}
              formatter={(v: number) => [`${v.toFixed(2)}%`]}
            />
            {["Q1","Q2","Q3","Q4","Q5"].map((q, i) => (
              <Line
                key={q} type="monotone" dataKey={q}
                stroke={quintileColors[i]} strokeWidth={1.5}
                dot={false} name={q}
              />
            ))}
            <Line
              type="monotone" dataKey="LS"
              stroke="#ffffff" strokeWidth={2} strokeDasharray="5 3"
              dot={false} name="L/S Spread"
            />
          </LineChart>
        </ResponsiveContainer>
        <div className="flex gap-4 mt-2 flex-wrap">
          {["Q1","Q2","Q3","Q4","Q5"].map((q, i) => (
            <div key={q} className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 rounded inline-block" style={{ background: quintileColors[i] }} />
              <span className="text-xs text-[#8892a4]">{q}</span>
            </div>
          ))}
          <div className="flex items-center gap-1.5">
            <span className="w-5 h-0.5 rounded inline-block" style={{ background: "#ffffff", borderTop: "1px dashed #ffffff" }} />
            <span className="text-xs text-[#8892a4]">L/S Spread</span>
          </div>
        </div>
      </div>

      {/* Drawdown Chart */}
      <div className="rounded-xl border border-[#2a2d3a] bg-[#1a1d27] p-4">
        <p className="text-sm font-medium text-[#e2e8f0] mb-4">Drawdown</p>
        <ResponsiveContainer width="100%" height={150}>
          <AreaChart data={drawdownData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
            <XAxis
              dataKey="month" tick={{ fill: "#8892a4", fontSize: 10 }}
              tickLine={false} axisLine={false} interval={3}
            />
            <YAxis
              tick={{ fill: "#8892a4", fontSize: 10 }}
              tickLine={false} axisLine={false}
              tickFormatter={(v) => `${v}%`} width={36}
            />
            <ReferenceLine y={0} stroke="#2a2d3a" />
            <Tooltip
              contentStyle={{ background: "#1a1d27", border: "1px solid #2a2d3a", borderRadius: 8, fontSize: 11 }}
              formatter={(v: number) => [`${v.toFixed(2)}%`, "Drawdown"]}
            />
            <defs>
              <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ff4757" stopOpacity={0.5} />
                <stop offset="95%" stopColor="#ff4757" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <Area
              type="monotone" dataKey="drawdown"
              stroke="#ff4757" strokeWidth={1.5}
              fill="url(#ddGrad)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Performance Table */}
      <div className="rounded-xl border border-[#2a2d3a] bg-[#1a1d27] p-4">
        <p className="text-sm font-medium text-[#e2e8f0] mb-4">Strategy Performance Summary</p>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-[#2a2d3a]">
                {["Strategy","Ann. Return","Volatility","Sharpe","Calmar","Max Drawdown","Win Rate"].map((h) => (
                  <th key={h} className="px-3 py-2 text-left text-[10px] font-medium uppercase tracking-wider text-[#8892a4]">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {perf.map((row, i) => (
                <tr
                  key={row.strategy}
                  className="border-b border-[#2a2d3a] last:border-0 hover:bg-[#1f2235] transition-colors"
                >
                  <td className="px-3 py-2.5 text-xs font-semibold text-[#e2e8f0]">{row.strategy}</td>
                  <PerfCell v={fmt(row.annReturn, true)} colorize bold />
                  <PerfCell v={`${row.vol.toFixed(1)}%`} />
                  <PerfCell v={row.sharpe.toFixed(2)} colorize />
                  <PerfCell v={row.calmar.toFixed(2)} colorize />
                  <PerfCell v={`${row.maxDD.toFixed(1)}%`} colorize />
                  <PerfCell v={`${row.winRate.toFixed(1)}%`} colorize />
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
