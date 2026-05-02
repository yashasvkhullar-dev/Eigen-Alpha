"use client"

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ScatterChart, Scatter, Cell,
  PieChart, Pie, Legend
} from "recharts"
import { portfolioWeights, clusterAllocation, clusterColors, efficientFrontierDots, efficientFrontierLine, optimalPoint } from "@/lib/mock-data"

const TEAL = "#00d2a8"
const GOLD = "#f7b731"

interface CustomLegendProps {
  payload?: Array<{ color: string; value: string }>
}

function CustomLegend({ payload }: CustomLegendProps) {
  if (!payload) return null
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-1">
      {payload.map((entry) => (
        <div key={entry.value} className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full inline-block flex-shrink-0" style={{ background: entry.color }} />
          <span className="text-[10px] text-[#8892a4] truncate">{entry.value}</span>
        </div>
      ))}
    </div>
  )
}

// Efficient frontier scatter: combine random dots + frontier line + optimal point
function EfficientFrontierChart() {
  const frontierDots = efficientFrontierLine.map((p) => ({ vol: p.vol, ret: p.ret, type: "frontier" }))
  const optDot = [{ vol: optimalPoint.vol, ret: optimalPoint.ret, type: "optimal" }]
  const feasible = efficientFrontierDots.map((d) => ({ ...d, type: "feasible" }))

  return (
    <ResponsiveContainer width="100%" height={260}>
      <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
        <XAxis
          type="number" dataKey="vol" name="Volatility"
          domain={[7, 24]} tickFormatter={(v) => `${v}%`}
          tick={{ fill: "#8892a4", fontSize: 10 }} tickLine={false} axisLine={false}
          label={{ value: "Volatility (%)", position: "insideBottom", offset: -4, fill: "#8892a4", fontSize: 10 }}
        />
        <YAxis
          type="number" dataKey="ret" name="Return"
          domain={[4, 22]} tickFormatter={(v) => `${v}%`}
          tick={{ fill: "#8892a4", fontSize: 10 }} tickLine={false} axisLine={false}
          label={{ value: "Return (%)", angle: -90, position: "insideLeft", fill: "#8892a4", fontSize: 10 }}
        />
        <Tooltip
          cursor={false}
          contentStyle={{ background: "#1a1d27", border: "1px solid #2a2d3a", borderRadius: 8, fontSize: 11 }}
          formatter={(v: number, name: string) => [`${v.toFixed(2)}%`, name]}
        />
        {/* Feasible portfolios */}
        <Scatter data={feasible} fill="#3a3d4a" opacity={0.6} name="Feasible Portfolios" r={2} />
        {/* Efficient frontier line */}
        <Scatter
          data={frontierDots}
          fill="none"
          line={{ stroke: TEAL, strokeWidth: 2 }}
          lineType="joint"
          name="Efficient Frontier"
          r={0}
        />
        {/* Optimal point */}
        <Scatter data={optDot} name="Optimal Sharpe" r={8} fill={GOLD} shape="star" />
      </ScatterChart>
    </ResponsiveContainer>
  )
}

export function PortfolioPage() {
  return (
    <div className="space-y-5">
      {/* Top row: weights + donut */}
      <div className="grid grid-cols-[1fr_340px] gap-4">
        {/* Portfolio Weights */}
        <div className="rounded-xl border border-[#2a2d3a] bg-[#1a1d27] p-4">
          <p className="text-sm font-medium text-[#e2e8f0] mb-4">Portfolio Weights — Top 20 Holdings</p>
          <ResponsiveContainer width="100%" height={340}>
            <BarChart
              data={portfolioWeights}
              layout="vertical"
              margin={{ top: 0, right: 40, left: 60, bottom: 0 }}
              barSize={12}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" horizontal={false} />
              <XAxis
                type="number" tickFormatter={(v) => `${v}%`}
                tick={{ fill: "#8892a4", fontSize: 10 }} tickLine={false} axisLine={false}
              />
              <YAxis
                type="category" dataKey="ticker"
                tick={{ fill: "#e2e8f0", fontSize: 10 }} tickLine={false} axisLine={false} width={60}
              />
              <Tooltip
                contentStyle={{ background: "#1a1d27", border: "1px solid #2a2d3a", borderRadius: 8, fontSize: 11 }}
                formatter={(v: number, _: string, props) => [`${v.toFixed(2)}%`, `Cluster ${props.payload.cluster}`]}
              />
              <Bar dataKey="weight" radius={[0, 3, 3, 0]} name="Weight">
                {portfolioWeights.map((entry) => (
                  <Cell key={entry.ticker} fill={clusterColors[entry.cluster]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Cluster Donut */}
        <div className="rounded-xl border border-[#2a2d3a] bg-[#1a1d27] p-4 flex flex-col">
          <p className="text-sm font-medium text-[#e2e8f0] mb-4">Cluster Allocation</p>
          <div className="flex-1 flex flex-col items-center justify-center">
            <PieChart width={280} height={200}>
              <Pie
                data={clusterAllocation}
                cx={140} cy={95}
                innerRadius={55} outerRadius={85}
                paddingAngle={2}
                dataKey="value"
                nameKey="name"
              >
                {clusterAllocation.map((entry) => (
                  <Cell key={entry.name} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ background: "#1a1d27", border: "1px solid #2a2d3a", borderRadius: 8, fontSize: 11 }}
                formatter={(v: number) => [`${v.toFixed(1)}%`]}
              />
              <Legend content={<CustomLegend />} />
            </PieChart>
          </div>
        </div>
      </div>

      {/* Efficient Frontier */}
      <div className="rounded-xl border border-[#2a2d3a] bg-[#1a1d27] p-4">
        <div className="flex items-center justify-between mb-2">
          <p className="text-sm font-medium text-[#e2e8f0]">Efficient Frontier</p>
          <div className="flex gap-4 items-center">
            <div className="flex items-center gap-1.5"><span className="w-3 h-0.5 rounded inline-block" style={{ background: TEAL }} /><span className="text-xs text-[#8892a4]">Efficient Frontier</span></div>
            <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded inline-block" style={{ background: "#3a3d4a" }} /><span className="text-xs text-[#8892a4]">Feasible Portfolios</span></div>
            <div className="flex items-center gap-1.5"><span className="text-sm">★</span><span className="text-xs text-[#8892a4]" style={{ color: GOLD }}> Optimal Sharpe</span></div>
          </div>
        </div>
        <EfficientFrontierChart />
      </div>
    </div>
  )
}
