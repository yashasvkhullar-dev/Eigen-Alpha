"""
EigenAlpha — Nifty 500 Ticker Universe
=======================================

This module defines the static list of Nifty 500 constituent tickers used
throughout the EigenAlpha pipeline.  Tickers are in Yahoo Finance `.NS`
(National Stock Exchange of India) format.

**Survivorship Bias Disclaimer**
---------------------------------
The ticker list below represents the *current* (as of late-2024) constituents
of the Nifty 500 index.  Using a point-in-time constituent list for historical
backtests introduces **survivorship bias**:

    1. Stocks that went bankrupt, were delisted, or dropped out of the
       index during the backtest window are *excluded* from this list.
    2. Stocks that were promoted into the index after the backtest start
       date are *included*, even though they would not have been investible
       at the start.
    3. Both effects inflate backtest performance relative to a truly
       implementable strategy.

This is a known and accepted limitation of Phase 0.  Future phases will
address this by integrating historical index composition data from NSE
or a commercial data vendor (e.g., CMIE Prowess, Bloomberg).

Academic reference:
    Elton, E. J., Gruber, M. J., & Blake, C. R. (1996).
    Survivorship Bias and Mutual Fund Performance.
    *Review of Financial Studies*, 9(4), 1097–1120.

Usage:
    from data.universe import NIFTY500_TICKERS
    print(f"Universe size: {len(NIFTY500_TICKERS)} tickers")
"""

from typing import List

# ──────────────────────────────────────────────────────────────────────────────
# Nifty 500 constituents — large-cap & selected mid-cap tickers (.NS format)
# ──────────────────────────────────────────────────────────────────────────────
# This list covers approximately 120 of the most liquid large-cap and
# upper-mid-cap names across all major GICS sectors represented in the
# Indian market.  The list is intentionally kept manageable for Phase 0
# while being broad enough to permit meaningful cross-sectional analysis
# and PCA decomposition.
#
# Sector coverage (approximate):
#   Financials      ~25 names
#   Information Technology ~12
#   Energy / Oil & Gas   ~8
#   Consumer Staples     ~10
#   Consumer Discretionary ~10
#   Industrials         ~10
#   Healthcare / Pharma  ~12
#   Materials / Metals   ~10
#   Utilities / Power    ~6
#   Telecom / Media      ~5
#   Real Estate          ~5
#   Others               ~7
# ──────────────────────────────────────────────────────────────────────────────

NIFTY500_TICKERS: List[str] = [
    # ── Financials ──────────────────────────────────────────────────────────
    "HDFCBANK.NS",     # HDFC Bank
    "ICICIBANK.NS",    # ICICI Bank
    "KOTAKBANK.NS",    # Kotak Mahindra Bank
    "SBIN.NS",         # State Bank of India
    "AXISBANK.NS",     # Axis Bank
    "INDUSINDBK.NS",   # IndusInd Bank
    "BANDHANBNK.NS",   # Bandhan Bank
    "FEDERALBNK.NS",   # Federal Bank
    "IDFCFIRSTB.NS",   # IDFC First Bank
    "PNB.NS",          # Punjab National Bank
    "BANKBARODA.NS",   # Bank of Baroda
    "CANBK.NS",        # Canara Bank
    "AUBANK.NS",       # AU Small Finance Bank
    "BAJFINANCE.NS",   # Bajaj Finance
    "BAJAJFINSV.NS",   # Bajaj Finserv
    "HDFCLIFE.NS",     # HDFC Life Insurance
    "SBILIFE.NS",      # SBI Life Insurance
    "ICICIPRULI.NS",   # ICICI Prudential Life
    "ICICIGI.NS",      # ICICI Lombard General Insurance
    "MUTHOOTFIN.NS",   # Muthoot Finance
    "CHOLAFIN.NS",     # Cholamandalam Investment
    "SHRIRAMFIN.NS",   # Shriram Finance
    "M&MFIN.NS",       # Mahindra & Mahindra Financial
    "PFC.NS",          # Power Finance Corporation
    "RECLTD.NS",       # REC Limited

    # ── Information Technology ──────────────────────────────────────────────
    "TCS.NS",          # Tata Consultancy Services
    "INFY.NS",         # Infosys
    "HCLTECH.NS",      # HCL Technologies
    "WIPRO.NS",        # Wipro
    "TECHM.NS",        # Tech Mahindra
    "LTIM.NS",         # LTIMindtree
    "PERSISTENT.NS",   # Persistent Systems
    "COFORGE.NS",      # Coforge
    "MPHASIS.NS",      # Mphasis
    "LTTS.NS",         # L&T Technology Services
    "TATAELXSI.NS",    # Tata Elxsi
    "OFSS.NS",         # Oracle Financial Services

    # ── Energy / Oil & Gas ──────────────────────────────────────────────────
    "RELIANCE.NS",     # Reliance Industries
    "ONGC.NS",         # Oil & Natural Gas Corporation
    "IOC.NS",          # Indian Oil Corporation
    "BPCL.NS",         # Bharat Petroleum
    "HINDPETRO.NS",    # Hindustan Petroleum
    "GAIL.NS",         # GAIL India
    "PETRONET.NS",     # Petronet LNG
    "ADANIENT.NS",     # Adani Enterprises

    # ── Consumer Staples ────────────────────────────────────────────────────
    "HINDUNILVR.NS",   # Hindustan Unilever
    "ITC.NS",          # ITC Limited
    "NESTLEIND.NS",    # Nestle India
    "BRITANNIA.NS",    # Britannia Industries
    "DABUR.NS",        # Dabur India
    "MARICO.NS",       # Marico
    "GODREJCP.NS",     # Godrej Consumer Products
    "COLPAL.NS",       # Colgate-Palmolive India
    "TATACONSUM.NS",   # Tata Consumer Products
    "UBL.NS",          # United Breweries

    # ── Consumer Discretionary ──────────────────────────────────────────────
    "TITAN.NS",        # Titan Company
    "TRENT.NS",        # Trent (Westside)
    "PAGEIND.NS",      # Page Industries
    "BATAINDIA.NS",    # Bata India
    "MANYAVAR.NS",     # Vedant Fashions (Manyavar)
    "JUBLFOOD.NS",     # Jubilant FoodWorks
    "ZOMATO.NS",       # Zomato
    "NYKAA.NS",        # FSN E-Commerce (Nykaa)
    "DMART.NS",        # Avenue Supermarts (D-Mart)
    "TVSMOTOR.NS",     # TVS Motor

    # ── Automobiles ─────────────────────────────────────────────────────────
    "MARUTI.NS",       # Maruti Suzuki
    "M&M.NS",          # Mahindra & Mahindra
    "TATAMOTORS.NS",   # Tata Motors
    "BAJAJ-AUTO.NS",   # Bajaj Auto
    "EICHERMOT.NS",    # Eicher Motors (Royal Enfield)
    "HEROMOTOCO.NS",   # Hero MotoCorp
    "ASHOKLEY.NS",     # Ashok Leyland

    # ── Healthcare / Pharma ─────────────────────────────────────────────────
    "SUNPHARMA.NS",    # Sun Pharmaceutical
    "DRREDDY.NS",      # Dr. Reddy's Laboratories
    "CIPLA.NS",        # Cipla
    "DIVISLAB.NS",     # Divi's Laboratories
    "APOLLOHOSP.NS",   # Apollo Hospitals
    "FORTIS.NS",       # Fortis Healthcare
    "MAXHEALTH.NS",    # Max Healthcare
    "AUROPHARMA.NS",   # Aurobindo Pharma
    "BIOCON.NS",       # Biocon
    "LUPIN.NS",        # Lupin
    "TORNTPHARM.NS",   # Torrent Pharmaceuticals
    "ALKEM.NS",        # Alkem Laboratories

    # ── Industrials / Capital Goods ─────────────────────────────────────────
    "LT.NS",           # Larsen & Toubro
    "SIEMENS.NS",      # Siemens India
    "ABB.NS",          # ABB India
    "HAVELLS.NS",      # Havells India
    "VOLTAS.NS",       # Voltas
    "CUMMINSIND.NS",   # Cummins India
    "BEL.NS",          # Bharat Electronics
    "HAL.NS",          # Hindustan Aeronautics
    "BHEL.NS",         # Bharat Heavy Electricals
    "IRCTC.NS",        # Indian Railway Catering & Tourism

    # ── Materials / Metals ───────────────────────────────────────────────────
    "TATASTEEL.NS",    # Tata Steel
    "JSWSTEEL.NS",     # JSW Steel
    "HINDALCO.NS",     # Hindalco Industries
    "VEDL.NS",         # Vedanta Limited
    "COALINDIA.NS",    # Coal India
    "NMDC.NS",         # NMDC
    "UPL.NS",          # UPL Limited
    "PIDILITIND.NS",   # Pidilite Industries
    "AMBUJACEM.NS",    # Ambuja Cements
    "ULTRACEMCO.NS",   # UltraTech Cement

    # ── Utilities / Power ────────────────────────────────────────────────────
    "NTPC.NS",         # NTPC Limited
    "POWERGRID.NS",    # Power Grid Corporation
    "TATAPOWER.NS",    # Tata Power
    "ADANIGREEN.NS",   # Adani Green Energy
    "ADANIPORTS.NS",   # Adani Ports & SEZ
    "NHPC.NS",         # NHPC Limited

    # ── Telecom / Media ──────────────────────────────────────────────────────
    "BHARTIARTL.NS",   # Bharti Airtel
    "IDEA.NS",         # Vodafone Idea
    "INDIAMART.NS",    # IndiaMART InterMESH
    "NAUKRI.NS",       # Info Edge (Naukri.com)
    "POLICYBZR.NS",    # PB Fintech (PolicyBazaar)

    # ── Real Estate ──────────────────────────────────────────────────────────
    "DLF.NS",          # DLF Limited
    "GODREJPROP.NS",   # Godrej Properties
    "OBEROIRLTY.NS",   # Oberoi Realty
    "PRESTIGE.NS",     # Prestige Estates
    "LODHA.NS",        # Macrotech Developers (Lodha)

    # ── Others / Conglomerates ───────────────────────────────────────────────
    "ASIANPAINT.NS",   # Asian Paints
    "BERGEPAINT.NS",   # Berger Paints
    "SBICARD.NS",      # SBI Cards
    "PIIND.NS",        # PI Industries
    "SRF.NS",          # SRF Limited
    "BALKRISIND.NS",   # Balkrishna Industries
    "MOTHERSON.NS",    # Samvardhana Motherson
]

# Sector mapping for cluster-vs-sector analysis
# Maps ticker (without .NS suffix) to approximate GICS sector
SECTOR_MAP: dict = {
    "HDFCBANK": "Financials", "ICICIBANK": "Financials", "KOTAKBANK": "Financials",
    "SBIN": "Financials", "AXISBANK": "Financials", "INDUSINDBK": "Financials",
    "BANDHANBNK": "Financials", "FEDERALBNK": "Financials", "IDFCFIRSTB": "Financials",
    "PNB": "Financials", "BANKBARODA": "Financials", "CANBK": "Financials",
    "AUBANK": "Financials", "BAJFINANCE": "Financials", "BAJAJFINSV": "Financials",
    "HDFCLIFE": "Financials", "SBILIFE": "Financials", "ICICIPRULI": "Financials",
    "ICICIGI": "Financials", "MUTHOOTFIN": "Financials", "CHOLAFIN": "Financials",
    "SHRIRAMFIN": "Financials", "M&MFIN": "Financials", "PFC": "Financials",
    "RECLTD": "Financials",
    "TCS": "Information Technology", "INFY": "Information Technology",
    "HCLTECH": "Information Technology", "WIPRO": "Information Technology",
    "TECHM": "Information Technology", "LTIM": "Information Technology",
    "PERSISTENT": "Information Technology", "COFORGE": "Information Technology",
    "MPHASIS": "Information Technology", "LTTS": "Information Technology",
    "TATAELXSI": "Information Technology", "OFSS": "Information Technology",
    "RELIANCE": "Energy", "ONGC": "Energy", "IOC": "Energy", "BPCL": "Energy",
    "HINDPETRO": "Energy", "GAIL": "Energy", "PETRONET": "Energy",
    "ADANIENT": "Energy",
    "HINDUNILVR": "Consumer Staples", "ITC": "Consumer Staples",
    "NESTLEIND": "Consumer Staples", "BRITANNIA": "Consumer Staples",
    "DABUR": "Consumer Staples", "MARICO": "Consumer Staples",
    "GODREJCP": "Consumer Staples", "COLPAL": "Consumer Staples",
    "TATACONSUM": "Consumer Staples", "UBL": "Consumer Staples",
    "TITAN": "Consumer Discretionary", "TRENT": "Consumer Discretionary",
    "PAGEIND": "Consumer Discretionary", "BATAINDIA": "Consumer Discretionary",
    "MANYAVAR": "Consumer Discretionary", "JUBLFOOD": "Consumer Discretionary",
    "ZOMATO": "Consumer Discretionary", "NYKAA": "Consumer Discretionary",
    "DMART": "Consumer Discretionary", "TVSMOTOR": "Consumer Discretionary",
    "MARUTI": "Consumer Discretionary", "M&M": "Consumer Discretionary",
    "TATAMOTORS": "Consumer Discretionary", "BAJAJ-AUTO": "Consumer Discretionary",
    "EICHERMOT": "Consumer Discretionary", "HEROMOTOCO": "Consumer Discretionary",
    "ASHOKLEY": "Consumer Discretionary",
    "SUNPHARMA": "Healthcare", "DRREDDY": "Healthcare", "CIPLA": "Healthcare",
    "DIVISLAB": "Healthcare", "APOLLOHOSP": "Healthcare", "FORTIS": "Healthcare",
    "MAXHEALTH": "Healthcare", "AUROPHARMA": "Healthcare", "BIOCON": "Healthcare",
    "LUPIN": "Healthcare", "TORNTPHARM": "Healthcare", "ALKEM": "Healthcare",
    "LT": "Industrials", "SIEMENS": "Industrials", "ABB": "Industrials",
    "HAVELLS": "Industrials", "VOLTAS": "Industrials", "CUMMINSIND": "Industrials",
    "BEL": "Industrials", "HAL": "Industrials", "BHEL": "Industrials",
    "IRCTC": "Industrials",
    "TATASTEEL": "Materials", "JSWSTEEL": "Materials", "HINDALCO": "Materials",
    "VEDL": "Materials", "COALINDIA": "Materials", "NMDC": "Materials",
    "UPL": "Materials", "PIDILITIND": "Materials", "AMBUJACEM": "Materials",
    "ULTRACEMCO": "Materials",
    "NTPC": "Utilities", "POWERGRID": "Utilities", "TATAPOWER": "Utilities",
    "ADANIGREEN": "Utilities", "ADANIPORTS": "Utilities", "NHPC": "Utilities",
    "BHARTIARTL": "Communication Services", "IDEA": "Communication Services",
    "INDIAMART": "Communication Services", "NAUKRI": "Communication Services",
    "POLICYBZR": "Communication Services",
    "DLF": "Real Estate", "GODREJPROP": "Real Estate", "OBEROIRLTY": "Real Estate",
    "PRESTIGE": "Real Estate", "LODHA": "Real Estate",
    "ASIANPAINT": "Materials", "BERGEPAINT": "Materials",
    "SBICARD": "Financials", "PIIND": "Materials", "SRF": "Materials",
    "BALKRISIND": "Consumer Discretionary", "MOTHERSON": "Consumer Discretionary",
}


def get_tickers() -> List[str]:
    """Return the full Nifty 500 ticker list.

    Returns:
        List[str]: Ticker symbols in Yahoo Finance `.NS` format.
    """
    return NIFTY500_TICKERS.copy()


def get_sector_map() -> dict:
    """Return the ticker-to-sector mapping dictionary.

    Returns:
        dict: Mapping of ticker (without `.NS` suffix) to GICS sector name.
    """
    return SECTOR_MAP.copy()
