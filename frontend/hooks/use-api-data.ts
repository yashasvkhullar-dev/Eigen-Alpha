"use client"

import { useState, useEffect, useCallback } from "react"
import { fetchAPI } from "@/lib/api-client"
import * as mock from "@/lib/mock-data"

// ─── Types ──────────────────────────────────────────────────────────

interface OverviewData {
  available: boolean
  metrics: {
    portfolio_return: number
    sharpe_ratio: number
    max_drawdown: number
    active_stocks: number
  }
  sparklines: {
    portfolio: number[]
    sharpe: number[]
    drawdown: number[]
    stocks: number[]
  }
  cumulative_returns: Array<{ month: string; eigen: number; nifty: number }>
  monthly_returns_heatmap: Array<{ year: number; month: string; value: number }>
}

interface FactorData {
  available: boolean
  factorICData: Array<Record<string, any>>
  factorStats: Array<{
    name: string
    key: string
    meanIC: number
    IR: number
    posMonths: number
    color: string
  }>
  correlationMatrix: Array<Record<string, any>>
  icDecayData: Array<Record<string, any>>
}

interface PortfolioData {
  available: boolean
  portfolioWeights: Array<{ ticker: string; weight: number; cluster: number }>
  clusterAllocation: Array<{ name: string; value: number; color: string }>
  clusterColors: Record<number, string>
  efficientFrontierDots: Array<{ vol: number; ret: number }>
  efficientFrontierLine: Array<{ vol: number; ret: number }>
  optimalPoint: { vol: number; ret: number; sharpe: number }
}

interface BacktestData {
  available: boolean
  quintileData: Array<Record<string, any>>
  drawdownData: Array<{ month: string; drawdown: number }>
  performanceTable: Array<{
    strategy: string
    annReturn: number
    vol: number
    sharpe: number
    calmar: number
    maxDD: number
    winRate: number
  }>
}

interface LiveMarketData {
  available: boolean
  liveMarketData: Array<{
    ticker: string
    momentum: number
    realVol: number
    volTrend: number
    composite: number
    cluster: number
    signal: string
  }>
}

// ─── Generic hook ───────────────────────────────────────────────────

function useAPIData<T>(endpoint: string, fallback: T) {
  const [data, setData] = useState<T>(fallback)
  const [loading, setLoading] = useState(true)
  const [isLive, setIsLive] = useState(false)

  const refresh = useCallback(() => {
    setLoading(true)
    fetchAPI<T>(endpoint, fallback).then((result) => {
      setData(result)
      setIsLive(result !== fallback)
      setLoading(false)
    })
  }, [endpoint])

  useEffect(() => {
    refresh()
  }, [refresh])

  return { data, loading, isLive, refresh }
}

// ─── Page-specific hooks ────────────────────────────────────────────

export function useOverviewData() {
  const fallback: OverviewData = {
    available: false,
    metrics: {
      portfolio_return: 18.4,
      sharpe_ratio: 1.24,
      max_drawdown: -11.2,
      active_stocks: 47,
    },
    sparklines: {
      portfolio: mock.sparklinePortfolio,
      sharpe: mock.sparklineSharpe,
      drawdown: mock.sparklineDrawdown,
      stocks: mock.sparklineStocks,
    },
    cumulative_returns: mock.cumulativeReturns,
    monthly_returns_heatmap: mock.monthlyReturnsHeatmap,
  }

  return useAPIData("/overview", fallback)
}

export function useFactorData() {
  const fallback: FactorData = {
    available: false,
    factorICData: mock.factorICData,
    factorStats: mock.factorStats,
    correlationMatrix: mock.correlationMatrix,
    icDecayData: mock.icDecayData,
  }

  return useAPIData("/factors", fallback)
}

export function usePortfolioData() {
  const fallback: PortfolioData = {
    available: false,
    portfolioWeights: mock.portfolioWeights,
    clusterAllocation: mock.clusterAllocation,
    clusterColors: mock.clusterColors,
    efficientFrontierDots: mock.efficientFrontierDots,
    efficientFrontierLine: mock.efficientFrontierLine,
    optimalPoint: mock.optimalPoint,
  }

  return useAPIData("/portfolio", fallback)
}

export function useBacktestData() {
  const fallback: BacktestData = {
    available: false,
    quintileData: mock.quintileData,
    drawdownData: mock.drawdownData,
    performanceTable: mock.performanceTable,
  }

  return useAPIData("/backtest", fallback)
}

export function useLiveMarketData() {
  const fallback: LiveMarketData = {
    available: false,
    liveMarketData: mock.liveMarketData,
  }

  return useAPIData("/live", fallback)
}
