"""
EigenAlpha — Data Ingestion & Preprocessing Package
====================================================

Provides:
    - universe   : Static Nifty 500 ticker list
    - loader     : yfinance data download pipeline
    - preprocessor : Returns computation, winsorisation, z-scoring
"""

from data.universe import NIFTY500_TICKERS
from data.loader import DataLoader
from data.preprocessor import Preprocessor

__all__ = ["NIFTY500_TICKERS", "DataLoader", "Preprocessor"]
