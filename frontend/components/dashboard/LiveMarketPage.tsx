"use client"

import { useState, useMemo } from "react"
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer
} from "recharts"
import { liveMarketData, generateIntradayData, clusterColors } from "@/lib/mock-data"

const TEAL = "#00d2a8"
const RED = "#ff4757"

function SignalBadge({ signal }: { signal: string }) {
  const config = {
    BUY: { bg: "rgba(0,210,168,0.15)", color: TEAL, border: "rgba(0,210,168,0.3)" },
    HOLD: { bg: "rgba(136,146,164,0.12)", color: "#8892a4", border: "rgba(136,146,164,0.25)" },
    SELL: { bg: "rgba(255,71,87,0.15)", color: RED, border: "rgba(255,71,87,0.3)" },
  }[signal] ?? { bg: "rgba(136,146,164,0.12)", color: "#8892a4", border: "rgba(136,146,164,0.25)" }

  return (
    <span
      className="px-2 py-0.5 rounded text-[10px] font-bold tracking-wide"
      style={{ background: config.bg, color: config.color, border: `1px solid ${config.border}` }}
    >
      {signal}
    </span>
  )
}

function ScoreBar({ value }: { value: number }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-[#2a2d3a]">
        <div
          className="h-full rounded-full"
          style={{ width: `${value * 100}%`, background: value > 0.6 ? TEAL : value > 0.3 ? "#f7931a" : RED }}
        />
      </div>
      <span className="text-xs tabular-nums" style={{ color: value > 0.6 ? TEAL : value > 0.3 ? "#f7931a" : RED }}>
        {value.toFixed(2)}
      </span>
    </div>
  )
}

export function LiveMarketPage() {
  const [search, setSearch] = useState("")
  const [selectedTicker, setSelectedTicker] = useState("RELIANCE")
  const [lastUpdated] = useState(() => {
    const now = new Date()
    return now.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit" })
  })
  const [refreshKey, setRefreshKey] = useState(0)

  const sorted = useMemo(() => {
    return [...liveMarketData]
      .filter((r) => r.ticker.toLowerCase().includes(search.toLowerCase()))
      .sort((a, b) => b.composite - a.composite)
  }, [search])

  const topQ = sorted[0]?.composite ?? 0
  const botQ = sorted[sorted.length - 1]?.composite ?? 0
  const topThreshold = topQ - (topQ - botQ) * 0.2
  const botThreshold = botQ + (topQ - botQ) * 0.2

  const intradayData = useMemo(
    () => generateIntradayData(selectedTicker + refreshKey),
    [selectedTicker, refreshKey]
  )

  const priceNow = intradayData[intradayData.length - 1]?.price ?? 0
  const priceStart = intradayData[0]?.price ?? 0
  const priceDelta = priceNow - priceStart
  const pricePct = (priceDelta / priceStart) * 100

  return (
    <div className="space-y-5">
      {/* Controls */}
      <div className="flex items-center gap-3">
        <div className="relative flex-1 max-w-xs">
          <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-[#8892a4]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            type="text"
            placeholder="Search ticker…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-3 py-2 rounded-lg text-xs text-[#e2e8f0] placeholder-[#8892a4] outline-none focus:ring-1 focus:ring-[#00d2a8]"
            style={{ background: "#1a1d27", border: "1px solid #2a2d3a" }}
          />
        </div>
        <button
          onClick={() => setRefreshKey((k) => k + 1)}
          className="flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium text-[#0f1117] transition-opacity hover:opacity-90 active:scale-95"
          style={{ background: TEAL }}
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Refresh
        </button>
        <div className="flex items-center gap-2 text-xs text-[#8892a4] ml-auto">
          <span className="w-2 h-2 rounded-full bg-[#00d2a8] animate-pulse inline-block" />
          Last updated: {lastUpdated}
        </div>
      </div>

      {/* Live Factor Scores Table */}
      <div className="rounded-xl border border-[#2a2d3a] bg-[#1a1d27] overflow-hidden">
        <div className="px-4 py-3 border-b border-[#2a2d3a]">
          <p className="text-sm font-medium text-[#e2e8f0]">Live Factor Scores</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-[#2a2d3a]">
                {["Ticker","Momentum","Realised Vol","Volume Trend","Composite Score","Cluster","Signal"].map((h) => (
                  <th key={h} className="px-4 py-2.5 text-left text-[10px] font-medium uppercase tracking-wider text-[#8892a4] whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((row) => {
                const isTop = row.composite >= topThreshold
                const isBot = row.composite <= botThreshold
                const rowBg = isTop
                  ? "rgba(0,210,168,0.06)"
                  : isBot
                  ? "rgba(255,71,87,0.06)"
                  : "transparent"
                const isSelected = row.ticker === selectedTicker

                return (
                  <tr
                    key={row.ticker}
                    onClick={() => setSelectedTicker(row.ticker)}
                    className="border-b border-[#2a2d3a] last:border-0 cursor-pointer hover:bg-[#1f2235] transition-colors"
                    style={{ background: isSelected ? "rgba(0,210,168,0.1)" : rowBg }}
                  >
                    <td className="px-4 py-2.5 text-xs font-bold text-[#e2e8f0] tracking-wide">{row.ticker}</td>
                    <td className="px-4 py-2.5 text-xs tabular-nums" style={{ color: row.momentum > 0.5 ? TEAL : "#8892a4" }}>
                      {row.momentum.toFixed(2)}
                    </td>
                    <td className="px-4 py-2.5 text-xs tabular-nums" style={{ color: row.realVol > 0.5 ? RED : "#8892a4" }}>
                      {row.realVol.toFixed(2)}
                    </td>
                    <td className="px-4 py-2.5 text-xs tabular-nums" style={{ color: row.volTrend > 0.5 ? TEAL : "#8892a4" }}>
                      {row.volTrend.toFixed(2)}
                    </td>
                    <td className="px-4 py-2.5 w-40">
                      <ScoreBar value={row.composite} />
                    </td>
                    <td className="px-4 py-2.5">
                      <span className="flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full inline-block" style={{ background: clusterColors[row.cluster] }} />
                        <span className="text-xs text-[#8892a4]">C{row.cluster}</span>
                      </span>
                    </td>
                    <td className="px-4 py-2.5">
                      <SignalBadge signal={row.signal} />
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Intraday Chart */}
      <div className="rounded-xl border border-[#2a2d3a] bg-[#1a1d27] p-4">
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="text-sm font-medium text-[#e2e8f0]">{selectedTicker} — Intraday (NSE)</p>
            <p className="text-xs text-[#8892a4] mt-0.5">5-min candles · Today</p>
          </div>
          <div className="text-right">
            <p className="text-lg font-bold text-[#e2e8f0]">₹{priceNow.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
            <p className="text-xs font-semibold" style={{ color: priceDelta >= 0 ? TEAL : RED }}>
              {priceDelta >= 0 ? "+" : ""}{priceDelta.toFixed(2)} ({pricePct >= 0 ? "+" : ""}{pricePct.toFixed(2)}%)
            </p>
          </div>
        </div>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={intradayData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
            <XAxis
              dataKey="time" tick={{ fill: "#8892a4", fontSize: 10 }}
              tickLine={false} axisLine={false} interval={12}
            />
            <YAxis
              tick={{ fill: "#8892a4", fontSize: 10 }} tickLine={false} axisLine={false}
              domain={["auto", "auto"]} tickFormatter={(v) => `₹${v.toFixed(0)}`} width={56}
            />
            <Tooltip
              contentStyle={{ background: "#1a1d27", border: "1px solid #2a2d3a", borderRadius: 8, fontSize: 11 }}
              formatter={(v: number) => [`₹${v.toFixed(2)}`, "Price"]}
            />
            <Line
              type="monotone" dataKey="price"
              stroke={priceDelta >= 0 ? TEAL : RED}
              strokeWidth={1.5} dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
