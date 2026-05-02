"use client"

import { useState, useMemo } from "react"
import { Sidebar } from "@/components/dashboard/Sidebar"
import { OverviewPage } from "@/components/dashboard/OverviewPage"
import { FactorAnalysisPage } from "@/components/dashboard/FactorAnalysisPage"
import { PortfolioPage } from "@/components/dashboard/PortfolioPage"
import { BacktestPage } from "@/components/dashboard/BacktestPage"
import { LiveMarketPage } from "@/components/dashboard/LiveMarketPage"

type Page = "overview" | "factor" | "portfolio" | "backtest" | "live"

const PAGE_TITLES: Record<Page, { title: string; sub: string }> = {
  overview: { title: "Overview", sub: "Portfolio summary · Feb 2025" },
  factor: { title: "Factor Analysis", sub: "IC, IR, and cross-sectional signal diagnostics" },
  portfolio: { title: "Portfolio Construction", sub: "Weights, clusters & efficient frontier" },
  backtest: { title: "Backtest Results", sub: "Jan 2020 – Feb 2025 · Quintile analysis" },
  live: { title: "Live Market", sub: "Real-time factor scores · NSE" },
}

function isMarketOpen() {
  const now = new Date()
  // Convert to IST (UTC+5:30)
  const ist = new Date(now.toLocaleString("en-US", { timeZone: "Asia/Kolkata" }))
  const day = ist.getDay()
  const hour = ist.getHours()
  const min = ist.getMinutes()
  const minutes = hour * 60 + min
  // Mon-Fri, 9:15 to 15:30
  return day >= 1 && day <= 5 && minutes >= 555 && minutes < 930
}

export default function EigenAlpha() {
  const [activePage, setActivePage] = useState<Page>("overview")
  const marketOpen = useMemo(() => isMarketOpen(), [])

  const { title, sub } = PAGE_TITLES[activePage]

  return (
    <div
      className="flex h-screen w-screen overflow-hidden"
      style={{ background: "#0f1117", fontFamily: "var(--font-inter, Inter, system-ui, sans-serif)" }}
    >
      <Sidebar activePage={activePage} onNavigate={setActivePage} marketOpen={marketOpen} />

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Topbar */}
        <header
          className="flex-shrink-0 flex items-center justify-between px-6 py-4"
          style={{ borderBottom: "1px solid #2a2d3a", background: "#0f1117" }}
        >
          <div>
            <h1 className="text-base font-semibold text-[#e2e8f0] tracking-tight">{title}</h1>
            <p className="text-xs text-[#8892a4] mt-0.5">{sub}</p>
          </div>
          <div className="flex items-center gap-3">
            {/* Universe badge */}
            <div
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs text-[#8892a4]"
              style={{ background: "#1a1d27", border: "1px solid #2a2d3a" }}
            >
              <span className="text-[#e2e8f0] font-medium">Universe:</span> Nifty 500
            </div>
            {/* Rebalance badge */}
            <div
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs text-[#8892a4]"
              style={{ background: "#1a1d27", border: "1px solid #2a2d3a" }}
            >
              <span className="text-[#e2e8f0] font-medium">Rebalance:</span> Monthly
            </div>
            {/* Phase badge */}
            <div
              className="px-3 py-1.5 rounded-lg text-xs font-semibold"
              style={{ background: "rgba(0,210,168,0.12)", color: "#00d2a8", border: "1px solid rgba(0,210,168,0.25)" }}
            >
              Phase 0 · Research
            </div>
          </div>
        </header>

        {/* Scrollable page content */}
        <main className="flex-1 overflow-y-auto px-6 py-5" style={{ background: "#0f1117" }}>
          {activePage === "overview" && <OverviewPage />}
          {activePage === "factor" && <FactorAnalysisPage />}
          {activePage === "portfolio" && <PortfolioPage />}
          {activePage === "backtest" && <BacktestPage />}
          {activePage === "live" && <LiveMarketPage />}
        </main>
      </div>
    </div>
  )
}
