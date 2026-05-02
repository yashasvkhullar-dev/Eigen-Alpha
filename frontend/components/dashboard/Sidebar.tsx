"use client"

type Page = "overview" | "factor" | "portfolio" | "backtest" | "live"

interface NavItem {
  id: Page
  label: string
  icon: React.ReactNode
}

const navItems: NavItem[] = [
  {
    id: "overview",
    label: "Overview",
    icon: (
      <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M4 5a1 1 0 011-1h4a1 1 0 011 1v5a1 1 0 01-1 1H5a1 1 0 01-1-1V5zm10 0a1 1 0 011-1h4a1 1 0 011 1v2a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 15a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1v-4zm10-4a1 1 0 011-1h4a1 1 0 011 1v8a1 1 0 01-1 1h-4a1 1 0 01-1-1v-8z" />
      </svg>
    ),
  },
  {
    id: "factor",
    label: "Factor Analysis",
    icon: (
      <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
  },
  {
    id: "portfolio",
    label: "Portfolio",
    icon: (
      <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
      </svg>
    ),
  },
  {
    id: "backtest",
    label: "Backtest",
    icon: (
      <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    ),
  },
  {
    id: "live",
    label: "Live Market",
    icon: (
      <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
  },
]

interface SidebarProps {
  activePage: Page
  onNavigate: (page: Page) => void
  marketOpen: boolean
}

export function Sidebar({ activePage, onNavigate, marketOpen }: SidebarProps) {
  return (
    <aside
      className="flex flex-col h-full"
      style={{
        width: 220,
        minWidth: 220,
        background: "#131620",
        borderRight: "1px solid #2a2d3a",
      }}
    >
      {/* Logo */}
      <div className="px-5 py-5 border-b border-[#2a2d3a]">
        <div className="flex items-center gap-2.5">
          <div
            className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
            style={{ background: "rgba(0,210,168,0.15)", border: "1px solid rgba(0,210,168,0.3)" }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="#00d2a8" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <div>
            <p className="text-sm font-bold text-white tracking-tight">EigenAlpha</p>
            <p className="text-[10px] text-[#8892a4]">v0.1 · Phase 0</p>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {navItems.map((item) => {
          const active = activePage === item.id
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left text-sm font-medium transition-all"
              style={{
                background: active ? "rgba(0,210,168,0.12)" : "transparent",
                color: active ? "#00d2a8" : "#8892a4",
                border: active ? "1px solid rgba(0,210,168,0.2)" : "1px solid transparent",
              }}
              onMouseEnter={(e) => {
                if (!active) {
                  e.currentTarget.style.color = "#e2e8f0"
                  e.currentTarget.style.background = "rgba(255,255,255,0.04)"
                }
              }}
              onMouseLeave={(e) => {
                if (!active) {
                  e.currentTarget.style.color = "#8892a4"
                  e.currentTarget.style.background = "transparent"
                }
              }}
            >
              <span style={{ color: active ? "#00d2a8" : "inherit" }}>{item.icon}</span>
              {item.label}
            </button>
          )
        })}
      </nav>

      {/* Market Status */}
      <div className="px-5 py-4 border-t border-[#2a2d3a]">
        <div
          className="flex items-center gap-2.5 px-3 py-2.5 rounded-lg"
          style={{ background: marketOpen ? "rgba(0,210,168,0.08)" : "rgba(255,71,87,0.08)", border: `1px solid ${marketOpen ? "rgba(0,210,168,0.2)" : "rgba(255,71,87,0.2)"}` }}
        >
          <span
            className="w-2 h-2 rounded-full flex-shrink-0"
            style={{
              background: marketOpen ? "#00d2a8" : "#ff4757",
              boxShadow: marketOpen ? "0 0 6px #00d2a8" : "0 0 6px #ff4757",
              animation: marketOpen ? "pulse 2s infinite" : "none",
            }}
          />
          <div>
            <p className="text-[11px] font-semibold" style={{ color: marketOpen ? "#00d2a8" : "#ff4757" }}>
              Market: {marketOpen ? "LIVE" : "CLOSED"}
            </p>
            <p className="text-[9px] text-[#8892a4]">NSE · {marketOpen ? "09:15 – 15:30 IST" : "Opens 09:15 IST"}</p>
          </div>
        </div>
      </div>
    </aside>
  )
}
