"""
EigenAlpha Phase 0 — Global Configuration
==========================================

Central configuration module for the EigenAlpha factor research engine.
All hyperparameters, date ranges, universe definitions, and pipeline
constants are defined here to ensure consistency across all modules.

This file serves as the single source of truth for every tunable
parameter in the pipeline.  Changing a value here propagates to every
module that imports it — no magic numbers are scattered through the
codebase.

Design philosophy:
    - All constants are module-level variables with type annotations.
    - Each constant has an inline comment explaining its purpose.
    - Academic or regulatory references are cited where applicable.
    - The ticker list is intentionally kept as a static snapshot;
      see data/universe.py for the survivorship bias disclaimer.

Usage:
    from config import START_DATE, END_DATE, RISK_FREE_RATE
    from config import NIFTY500_TICKERS
"""

from typing import List

# ══════════════════════════════════════════════════════════════════════
# DATE RANGE
# ══════════════════════════════════════════════════════════════════════

START_DATE: str = "2014-01-01"
END_DATE: str = "2024-12-31"

# ══════════════════════════════════════════════════════════════════════
# RISK PARAMETERS
# ══════════════════════════════════════════════════════════════════════

RISK_FREE_RATE: float = 0.065      # RBI repo rate proxy, annualised
MIN_HISTORY_MONTHS: int = 24       # minimum months of history required

# ══════════════════════════════════════════════════════════════════════
# FACTOR PARAMETERS
# ══════════════════════════════════════════════════════════════════════

FACTOR_WINSOR_LOWER: float = 2.5   # lower winsorise percentile
FACTOR_WINSOR_UPPER: float = 97.5  # upper winsorise percentile
FACTOR_WINSOR_PERCENTILE: tuple = (FACTOR_WINSOR_LOWER, FACTOR_WINSOR_UPPER)

MOMENTUM_LONG_WINDOW: int = 252    # trading days in 12 months
MOMENTUM_SHORT_WINDOW: int = 21    # trading days in 1 month (skip)
VOL_WINDOW: int = 20               # realised vol lookback (days)
VOL_ANNUALISATION: float = 252.0   # annualisation factor for daily vol
VOLUME_TREND_WINDOW: int = 20      # volume regression window (days)

# ══════════════════════════════════════════════════════════════════════
# PCA PARAMETERS
# ══════════════════════════════════════════════════════════════════════

N_PCA_COMPONENTS: int = 50         # max components to compute
VARIANCE_THRESHOLD: float = 0.80   # minimum explained variance for k selection
PCA_VARIANCE_THRESHOLD: float = VARIANCE_THRESHOLD  # alias for backward compat

# ══════════════════════════════════════════════════════════════════════
# CLUSTERING PARAMETERS
# ══════════════════════════════════════════════════════════════════════

N_CLUSTERS: int = 8
KMEANS_RANDOM_STATE: int = 42
KMEANS_N_INIT: int = 20

# ══════════════════════════════════════════════════════════════════════
# BACKTEST PARAMETERS
# ══════════════════════════════════════════════════════════════════════

REBALANCE_FREQ: str = "M"                      # monthly
TOP_QUINTILE: float = 0.20                     # top 20 %
BOTTOM_QUINTILE: float = 0.20                  # bottom 20 %
TOP_QUINTILE_THRESHOLD: float = 0.80           # = 1 - TOP_QUINTILE
BOTTOM_QUINTILE_THRESHOLD: float = 0.20        # = BOTTOM_QUINTILE

# ══════════════════════════════════════════════════════════════════════
# OPTIMISATION PARAMETERS
# ══════════════════════════════════════════════════════════════════════

R_TARGET_ANNUALISED: float = 0.12              # target annual return for Markowitz
MAX_SINGLE_STOCK_WEIGHT: float = 0.05          # 5 % cap per stock

# ══════════════════════════════════════════════════════════════════════
# OUTPUT PATHS
# ══════════════════════════════════════════════════════════════════════

OUTPUT_DIR: str = "outputs"
CACHE_DIR: str = "data/cache"
DATA_DIR: str = "data_cache"                   # legacy alias
TEARSHEET_PATH: str = "outputs/tearsheet.png"
METRICS_PATH: str = "outputs/metrics.json"

# ══════════════════════════════════════════════════════════════════════
# BENCHMARK
# ══════════════════════════════════════════════════════════════════════

BENCHMARK_TICKER: str = "^NSEI"  # Nifty 50 Total Return Index

# ══════════════════════════════════════════════════════════════════════
# NIFTY 500 TICKER UNIVERSE
# ══════════════════════════════════════════════════════════════════════
#
# Static list of ~100 large-cap and upper-mid-cap Nifty 500 constituents
# in Yahoo Finance .NS (NSE) format.
#
# ⚠ SURVIVORSHIP BIAS WARNING:
# This list represents current (late-2024) index constituents.
# Historical backtests using this list will overstate performance
# because delisted/bankrupt stocks are excluded.  See data/universe.py
# for the full academic disclaimer.
#
# Sector coverage:
#   Financials            ~22 names
#   Information Technology ~12
#   Energy / Oil & Gas     ~8
#   Consumer Staples       ~12
#   Consumer Discretionary ~10
#   Industrials            ~10
#   Healthcare / Pharma    ~12
#   Materials / Metals     ~12
#   Utilities / Power       ~6
#   Telecom / Media         ~3
#   Real Estate             ~3
#   Others                  ~5

NIFTY500_TICKERS: List[str] = [
    # ── Financials ──────────────────────────────────────────────────────
    "HDFCBANK.NS",      # HDFC Bank
    "ICICIBANK.NS",     # ICICI Bank
    "KOTAKBANK.NS",     # Kotak Mahindra Bank
    "SBIN.NS",          # State Bank of India
    "AXISBANK.NS",      # Axis Bank
    "INDUSINDBK.NS",    # IndusInd Bank
    "BAJFINANCE.NS",    # Bajaj Finance
    "BAJAJFINSV.NS",    # Bajaj Finserv
    "HDFCLIFE.NS",      # HDFC Life Insurance
    "SBILIFE.NS",       # SBI Life Insurance
    "ICICIPRULI.NS",    # ICICI Prudential Life
    "ICICIGI.NS",       # ICICI Lombard General Insurance
    "MUTHOOTFIN.NS",    # Muthoot Finance
    "CHOLAFIN.NS",      # Cholamandalam Investment & Finance
    "MANAPPURAM.NS",    # Manappuram Finance
    "PFC.NS",           # Power Finance Corporation
    "RECLTD.NS",        # REC Limited
    "IRFC.NS",          # Indian Railway Finance Corporation
    "BANKBARODA.NS",    # Bank of Baroda
    "PNB.NS",           # Punjab National Bank
    "CANBK.NS",         # Canara Bank
    "UNIONBANK.NS",     # Union Bank of India

    # ── Information Technology ──────────────────────────────────────────
    "TCS.NS",           # Tata Consultancy Services
    "INFY.NS",          # Infosys
    "HCLTECH.NS",       # HCL Technologies
    "WIPRO.NS",         # Wipro
    "TECHM.NS",         # Tech Mahindra
    "LTIM.NS",          # LTIMindtree
    "PERSISTENT.NS",    # Persistent Systems
    "COFORGE.NS",       # Coforge
    "MPHASIS.NS",       # Mphasis
    "LTTS.NS",          # L&T Technology Services
    "TATAELXSI.NS",     # Tata Elxsi
    "OFSS.NS",          # Oracle Financial Services

    # ── Energy / Oil & Gas ──────────────────────────────────────────────
    "RELIANCE.NS",      # Reliance Industries
    "ONGC.NS",          # Oil & Natural Gas Corporation
    "IOC.NS",           # Indian Oil Corporation
    "BPCL.NS",          # Bharat Petroleum
    "HINDPETRO.NS",     # Hindustan Petroleum
    "GAIL.NS",          # GAIL India
    "PETRONET.NS",      # Petronet LNG
    "ADANIENT.NS",      # Adani Enterprises

    # ── Consumer Staples ────────────────────────────────────────────────
    "HINDUNILVR.NS",    # Hindustan Unilever
    "ITC.NS",           # ITC Limited
    "NESTLEIND.NS",     # Nestle India
    "BRITANNIA.NS",     # Britannia Industries
    "DABUR.NS",         # Dabur India
    "MARICO.NS",        # Marico
    "GODREJCP.NS",      # Godrej Consumer Products
    "COLPAL.NS",        # Colgate-Palmolive India
    "TATACONSUM.NS",    # Tata Consumer Products
    "MCDOWELL-N.NS",    # United Spirits (McDowell's)
    "UBL.NS",           # United Breweries
    "PIDILITIND.NS",    # Pidilite Industries

    # ── Consumer Discretionary / Automobiles ────────────────────────────
    "TITAN.NS",         # Titan Company
    "MARUTI.NS",        # Maruti Suzuki
    "TATAMOTORS.NS",    # Tata Motors
    "BAJAJ-AUTO.NS",    # Bajaj Auto
    "EICHERMOT.NS",     # Eicher Motors (Royal Enfield)
    "HEROMOTOCO.NS",    # Hero MotoCorp
    "MOTHERSON.NS",     # Samvardhana Motherson
    "BOSCHLTD.NS",      # Bosch India
    "TVSMOTOR.NS",      # TVS Motor
    "ASHOKLEY.NS",      # Ashok Leyland

    # ── Industrials / Capital Goods ─────────────────────────────────────
    "LT.NS",            # Larsen & Toubro
    "SIEMENS.NS",       # Siemens India
    "ABB.NS",           # ABB India
    "HAVELLS.NS",       # Havells India
    "VOLTAS.NS",        # Voltas
    "CUMMINSIND.NS",    # Cummins India
    "BEL.NS",           # Bharat Electronics
    "HAL.NS",           # Hindustan Aeronautics
    "BHEL.NS",          # Bharat Heavy Electricals
    "IRCTC.NS",         # Indian Railway Catering & Tourism

    # ── Healthcare / Pharma ─────────────────────────────────────────────
    "SUNPHARMA.NS",     # Sun Pharmaceutical
    "DRREDDY.NS",       # Dr. Reddy's Laboratories
    "CIPLA.NS",         # Cipla
    "DIVISLAB.NS",      # Divi's Laboratories
    "APOLLOHOSP.NS",    # Apollo Hospitals
    "FORTIS.NS",        # Fortis Healthcare
    "LUPIN.NS",         # Lupin
    "AUROPHARMA.NS",    # Aurobindo Pharma
    "TORNTPHARM.NS",    # Torrent Pharmaceuticals
    "GLENMARK.NS",      # Glenmark Pharmaceuticals
    "IPCALAB.NS",       # Ipca Laboratories
    "ALKEM.NS",         # Alkem Laboratories

    # ── Materials / Metals / Cement ─────────────────────────────────────
    "TATASTEEL.NS",     # Tata Steel
    "JSWSTEEL.NS",      # JSW Steel
    "HINDALCO.NS",      # Hindalco Industries
    "VEDL.NS",          # Vedanta Limited
    "COALINDIA.NS",     # Coal India
    "NMDC.NS",          # NMDC
    "SAIL.NS",          # Steel Authority of India
    "ULTRACEMCO.NS",    # UltraTech Cement
    "AMBUJACEM.NS",     # Ambuja Cements
    "SHREECEM.NS",      # Shree Cement
    "ASIANPAINT.NS",    # Asian Paints
    "BERGERPAINTS.NS",  # Berger Paints
    "GRASIM.NS",        # Grasim Industries (Aditya Birla)

    # ── Utilities / Power ───────────────────────────────────────────────
    "NTPC.NS",          # NTPC Limited
    "POWERGRID.NS",     # Power Grid Corporation
    "TATAPOWER.NS",     # Tata Power
    "ADANIGREEN.NS",    # Adani Green Energy
    "ADANIPORTS.NS",    # Adani Ports & SEZ
    "NHPC.NS",          # NHPC Limited

    # ── Telecom / Media ─────────────────────────────────────────────────
    "BHARTIARTL.NS",    # Bharti Airtel
    "NAUKRI.NS",        # Info Edge (Naukri.com)
    "INDIAMART.NS",     # IndiaMART InterMESH

    # ── Real Estate ─────────────────────────────────────────────────────
    "DLF.NS",           # DLF Limited
    "GODREJPROP.NS",    # Godrej Properties
    "OBEROIRLTY.NS",    # Oberoi Realty

    # ── Mid-cap additions ───────────────────────────────────────────────
    "POLYCAB.NS",       # Polycab India (cables)
    "DEEPAKNTR.NS",     # Deepak Nitrite (chemicals)
    "ATUL.NS",          # Atul Limited (chemicals)
    "ASTRAL.NS",        # Astral Ltd (pipes)
    "SYNGENE.NS",       # Syngene International (pharma CRO)
    "LALPATHLAB.NS",    # Dr Lal PathLabs (diagnostics)
    "TRENT.NS",         # Trent (Westside retail)
    "DMART.NS",         # Avenue Supermarts (D-Mart)
    "PAGEIND.NS",       # Page Industries (Jockey)
    "JUBLFOOD.NS",      # Jubilant FoodWorks (Domino's)
    "ZOMATO.NS",        # Zomato
    "PIIND.NS",         # PI Industries (agrochemicals)
    "SRF.NS",           # SRF Limited (chemicals / packaging)
    "BALKRISIND.NS",    # Balkrishna Industries (tyres)
    "SHRIRAMFIN.NS",    # Shriram Finance
    "BANDHANBNK.NS",    # Bandhan Bank
    "FEDERALBNK.NS",    # Federal Bank
    "IDFCFIRSTB.NS",    # IDFC First Bank
    "AUBANK.NS",        # AU Small Finance Bank
    "SBICARD.NS",       # SBI Cards
]

# Re-export as UNIVERSE_TICKERS for pipeline.py compatibility
UNIVERSE_TICKERS: List[str] = NIFTY500_TICKERS
