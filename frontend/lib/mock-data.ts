// ─── Overview ────────────────────────────────────────────────────────────────

export const sparklinePortfolio = [8, 11, 9, 14, 13, 16, 15, 18, 17, 18.4]
export const sparklineSharpe = [0.8, 0.9, 1.0, 1.1, 1.05, 1.15, 1.2, 1.18, 1.22, 1.24]
export const sparklineDrawdown = [-5, -7, -9, -11, -8, -10, -11.2, -9, -10, -11.2]
export const sparklineStocks = [38, 41, 43, 40, 44, 46, 45, 47, 46, 47]

export const cumulativeReturns = [
  { month: "Jan'24", eigen: 0, nifty: 0 },
  { month: "Feb'24", eigen: 2.1, nifty: 1.4 },
  { month: "Mar'24", eigen: 4.8, nifty: 2.9 },
  { month: "Apr'24", eigen: 3.2, nifty: 1.5 },
  { month: "May'24", eigen: 6.7, nifty: 3.8 },
  { month: "Jun'24", eigen: 5.9, nifty: 2.6 },
  { month: "Jul'24", eigen: 9.4, nifty: 5.2 },
  { month: "Aug'24", eigen: 8.1, nifty: 4.1 },
  { month: "Sep'24", eigen: 11.3, nifty: 6.4 },
  { month: "Oct'24", eigen: 13.2, nifty: 7.8 },
  { month: "Nov'24", eigen: 15.8, nifty: 9.1 },
  { month: "Dec'24", eigen: 14.7, nifty: 8.3 },
  { month: "Jan'25", eigen: 16.9, nifty: 9.7 },
  { month: "Feb'25", eigen: 18.4, nifty: 10.9 },
]

export const monthlyReturnsHeatmap: { year: number; month: string; value: number }[] = []
const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
const yearReturns: Record<number, number[]> = {
  2015: [1.2, -2.1, 3.4, 0.8, -1.5, 2.2, 1.8, -0.7, 2.9, 1.4, -0.9, 1.6],
  2016: [-3.2, 2.8, 4.1, 1.7, 2.4, -1.1, 3.8, 2.2, 1.5, -2.8, -4.2, 1.9],
  2017: [4.3, 2.9, 3.2, 1.8, 3.7, 2.4, 4.8, 1.2, 3.6, 4.1, 2.7, 3.9],
  2018: [1.4, -3.8, 2.1, -1.2, -2.7, -4.1, 2.9, -3.2, -5.1, -4.7, 1.8, -2.3],
  2019: [3.2, 1.8, 4.6, 2.3, -2.9, 1.4, 3.7, -1.8, 2.4, 3.8, 2.1, 4.2],
  2020: [-5.2, -3.4, -12.8, 8.7, 5.3, 4.2, 6.8, 4.1, 5.7, 3.9, 7.2, 5.4],
  2021: [3.1, 4.8, 2.7, 3.9, 5.2, 4.1, 3.8, 5.6, 4.3, 2.8, -1.9, 4.7],
  2022: [-2.8, -3.6, 2.4, -4.2, -5.1, -3.8, 4.9, 2.3, -3.7, 3.8, 1.2, -2.4],
  2023: [4.2, 1.9, 3.7, 2.8, 4.1, 3.6, 5.2, 2.7, 3.1, 4.8, 3.3, 4.6],
  2024: [3.8, 2.4, 4.9, 1.2, 3.6, 2.1, 4.7, 2.9, 3.4, 4.2, 2.8, 3.7],
}
for (const [year, vals] of Object.entries(yearReturns)) {
  months.forEach((m, i) => {
    monthlyReturnsHeatmap.push({ year: Number(year), month: m, value: vals[i] })
  })
}

// ─── Factor Analysis ──────────────────────────────────────────────────────────

export const factorICData = Array.from({ length: 24 }, (_, i) => ({
  month: `M${i + 1}`,
  momentum: parseFloat((Math.random() * 0.16 - 0.04).toFixed(3)),
  realVol: parseFloat((Math.random() * 0.14 - 0.05).toFixed(3)),
  volTrend: parseFloat((Math.random() * 0.12 - 0.04).toFixed(3)),
}))

export const factorStats = [
  {
    name: "Momentum (12-1M)",
    key: "momentum",
    meanIC: 0.062,
    IR: 0.81,
    posMonths: 61.4,
    color: "#00d2a8",
  },
  {
    name: "Realised Volatility",
    key: "realVol",
    meanIC: -0.047,
    IR: -0.63,
    posMonths: 38.2,
    color: "#ff4757",
  },
  {
    name: "Volume Trend",
    key: "volTrend",
    meanIC: 0.038,
    IR: 0.52,
    posMonths: 54.8,
    color: "#f7931a",
  },
]

export const correlationMatrix = [
  { factor: "Momentum", Momentum: 1.0, "Realised Vol": -0.18, "Volume Trend": 0.24 },
  { factor: "Realised Vol", Momentum: -0.18, "Realised Vol": 1.0, "Volume Trend": -0.31 },
  { factor: "Volume Trend", Momentum: 0.24, "Realised Vol": -0.31, "Volume Trend": 1.0 },
]

export const icDecayData = [
  { lag: "1M", momentum: 0.062, realVol: -0.047, volTrend: 0.038 },
  { lag: "2M", momentum: 0.051, realVol: -0.039, volTrend: 0.031 },
  { lag: "3M", momentum: 0.044, realVol: -0.028, volTrend: 0.024 },
  { lag: "4M", momentum: 0.033, realVol: -0.019, volTrend: 0.017 },
  { lag: "5M", momentum: 0.021, realVol: -0.011, volTrend: 0.009 },
  { lag: "6M", momentum: 0.012, realVol: -0.006, volTrend: 0.004 },
]

// ─── Portfolio ────────────────────────────────────────────────────────────────

export const portfolioWeights = [
  { ticker: "RELIANCE", weight: 5.8, cluster: 1 },
  { ticker: "TCS", weight: 5.2, cluster: 2 },
  { ticker: "HDFCBANK", weight: 4.9, cluster: 3 },
  { ticker: "INFY", weight: 4.4, cluster: 2 },
  { ticker: "ICICIBANK", weight: 4.1, cluster: 3 },
  { ticker: "HINDUNILVR", weight: 3.8, cluster: 4 },
  { ticker: "BAJFINANCE", weight: 3.5, cluster: 3 },
  { ticker: "SBIN", weight: 3.2, cluster: 3 },
  { ticker: "BHARTIARTL", weight: 3.0, cluster: 5 },
  { ticker: "KOTAKBANK", weight: 2.9, cluster: 3 },
  { ticker: "LT", weight: 2.7, cluster: 6 },
  { ticker: "ASIANPAINT", weight: 2.5, cluster: 4 },
  { ticker: "AXISBANK", weight: 2.4, cluster: 3 },
  { ticker: "MARUTI", weight: 2.2, cluster: 7 },
  { ticker: "SUNPHARMA", weight: 2.1, cluster: 8 },
  { ticker: "TITAN", weight: 2.0, cluster: 4 },
  { ticker: "WIPRO", weight: 1.9, cluster: 2 },
  { ticker: "ULTRACEMCO", weight: 1.8, cluster: 6 },
  { ticker: "NESTLEIND", weight: 1.7, cluster: 4 },
  { ticker: "POWERGRID", weight: 1.6, cluster: 5 },
]

export const clusterAllocation = [
  { name: "Cluster 1 · Energy", value: 12.4, color: "#00d2a8" },
  { name: "Cluster 2 · IT", value: 18.2, color: "#3b82f6" },
  { name: "Cluster 3 · BFSI", value: 24.6, color: "#f7931a" },
  { name: "Cluster 4 · FMCG", value: 13.8, color: "#a855f7" },
  { name: "Cluster 5 · Telecom", value: 9.2, color: "#ec4899" },
  { name: "Cluster 6 · Infra", value: 8.7, color: "#eab308" },
  { name: "Cluster 7 · Auto", value: 7.4, color: "#06b6d4" },
  { name: "Cluster 8 · Pharma", value: 5.7, color: "#84cc16" },
]

export const clusterColors: Record<number, string> = {
  1: "#00d2a8",
  2: "#3b82f6",
  3: "#f7931a",
  4: "#a855f7",
  5: "#ec4899",
  6: "#eab308",
  7: "#06b6d4",
  8: "#84cc16",
}

// Efficient frontier
export const efficientFrontierDots = Array.from({ length: 200 }, () => {
  const vol = 8 + Math.random() * 14
  const ret = vol * 0.7 + (Math.random() - 0.5) * 6
  return { vol: parseFloat(vol.toFixed(2)), ret: parseFloat(ret.toFixed(2)) }
})

export const efficientFrontierLine = [
  { vol: 9.2, ret: 8.1 },
  { vol: 10.0, ret: 10.4 },
  { vol: 11.2, ret: 12.7 },
  { vol: 12.5, ret: 14.8 },
  { vol: 13.8, ret: 16.3 },
  { vol: 15.1, ret: 17.2 },
  { vol: 16.4, ret: 17.8 },
  { vol: 18.0, ret: 18.0 },
]

export const optimalPoint = { vol: 13.8, ret: 16.3, sharpe: 1.24 }

// ─── Backtest ────────────────────────────────────────────────────────────────

const months2 = [
  "Jan'20","Apr'20","Jul'20","Oct'20",
  "Jan'21","Apr'21","Jul'21","Oct'21",
  "Jan'22","Apr'22","Jul'22","Oct'22",
  "Jan'23","Apr'23","Jul'23","Oct'23",
  "Jan'24","Apr'24","Jul'24","Oct'24",
  "Jan'25",
]

function buildQuintile(base: number, vol: number) {
  let v = 0
  return months2.map((m) => {
    v += (Math.random() - 0.5) * vol + base / months2.length
    return { month: m, value: parseFloat(v.toFixed(2)) }
  })
}

export const quintileData = months2.map((m, i) => {
  const q1 = -12 + i * (-0.3) + (Math.random() - 0.5) * 2
  const q2 = -4 + i * 0.1 + (Math.random() - 0.5) * 1.5
  const q3 = 2 + i * 0.5 + (Math.random() - 0.5) * 1.5
  const q4 = 7 + i * 0.7 + (Math.random() - 0.5) * 1.5
  const q5 = 14 + i * 0.9 + (Math.random() - 0.5) * 2
  return {
    month: m,
    Q1: parseFloat(q1.toFixed(2)),
    Q2: parseFloat(q2.toFixed(2)),
    Q3: parseFloat(q3.toFixed(2)),
    Q4: parseFloat(q4.toFixed(2)),
    Q5: parseFloat(q5.toFixed(2)),
    LS: parseFloat((q5 - q1).toFixed(2)),
  }
})

export const drawdownData = months2.map((m, i) => {
  const base = -Math.abs(Math.sin(i * 0.5) * 11 + Math.random() * 3)
  return { month: m, drawdown: parseFloat(base.toFixed(2)) }
})

export const performanceTable = [
  { strategy: "Q5 Long", annReturn: 24.8, vol: 14.2, sharpe: 1.48, calmar: 2.21, maxDD: -11.2, winRate: 63.4 },
  { strategy: "Q1 Short", annReturn: -8.3, vol: 12.7, sharpe: -0.65, calmar: -0.74, maxDD: -21.4, winRate: 34.8 },
  { strategy: "L/S Spread", annReturn: 18.4, vol: 9.8, sharpe: 1.88, calmar: 1.64, maxDD: -11.2, winRate: 61.2 },
  { strategy: "Markowitz", annReturn: 16.9, vol: 11.4, sharpe: 1.24, calmar: 1.51, maxDD: -11.2, winRate: 58.7 },
  { strategy: "Nifty 500", annReturn: 12.1, vol: 13.8, sharpe: 0.88, calmar: 0.92, maxDD: -28.6, winRate: 52.3 },
]

// ─── Live Market ──────────────────────────────────────────────────────────────

export const liveMarketData = [
  { ticker: "RELIANCE", momentum: 0.82, realVol: 0.31, volTrend: 0.74, composite: 0.78, cluster: 1, signal: "BUY" },
  { ticker: "TCS", momentum: 0.71, realVol: 0.24, volTrend: 0.68, composite: 0.71, cluster: 2, signal: "BUY" },
  { ticker: "INFY", momentum: 0.65, realVol: 0.29, volTrend: 0.61, composite: 0.64, cluster: 2, signal: "BUY" },
  { ticker: "HDFCBANK", momentum: 0.58, realVol: 0.41, volTrend: 0.52, composite: 0.57, cluster: 3, signal: "HOLD" },
  { ticker: "ICICIBANK", momentum: 0.54, realVol: 0.44, volTrend: 0.49, composite: 0.53, cluster: 3, signal: "HOLD" },
  { ticker: "BHARTIARTL", momentum: 0.61, realVol: 0.33, volTrend: 0.58, composite: 0.60, cluster: 5, signal: "BUY" },
  { ticker: "BAJFINANCE", momentum: 0.49, realVol: 0.52, volTrend: 0.44, composite: 0.47, cluster: 3, signal: "HOLD" },
  { ticker: "HINDUNILVR", momentum: 0.38, realVol: 0.22, volTrend: 0.41, composite: 0.39, cluster: 4, signal: "HOLD" },
  { ticker: "MARUTI", momentum: 0.44, realVol: 0.48, volTrend: 0.39, composite: 0.42, cluster: 7, signal: "HOLD" },
  { ticker: "SBIN", momentum: 0.31, realVol: 0.61, volTrend: 0.28, composite: 0.31, cluster: 3, signal: "HOLD" },
  { ticker: "TITAN", momentum: 0.27, realVol: 0.37, volTrend: 0.24, composite: 0.26, cluster: 4, signal: "HOLD" },
  { ticker: "LT", momentum: 0.22, realVol: 0.58, volTrend: 0.19, composite: 0.22, cluster: 6, signal: "HOLD" },
  { ticker: "WIPRO", momentum: 0.18, realVol: 0.63, volTrend: 0.15, composite: 0.17, cluster: 2, signal: "SELL" },
  { ticker: "AXISBANK", momentum: 0.14, realVol: 0.71, volTrend: 0.12, composite: 0.13, cluster: 3, signal: "SELL" },
  { ticker: "SUNPHARMA", momentum: 0.09, realVol: 0.74, volTrend: 0.08, composite: 0.08, cluster: 8, signal: "SELL" },
]

export function generateIntradayData(ticker: string) {
  const seed = ticker.charCodeAt(0) + ticker.charCodeAt(1)
  const base = 500 + (seed % 3000)
  const points = 78 // 9:15 to 15:30 in 5-min intervals
  let price = base
  return Array.from({ length: points }, (_, i) => {
    price = price + (Math.random() - 0.48) * (price * 0.003)
    const hours = 9 + Math.floor((i * 5 + 15) / 60)
    const mins = (i * 5 + 15) % 60
    return {
      time: `${hours}:${String(mins).padStart(2, "0")}`,
      price: parseFloat(price.toFixed(2)),
    }
  })
}
