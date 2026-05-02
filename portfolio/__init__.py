"""
EigenAlpha — Portfolio Construction Package
===========================================

Provides:
    - optimizer  : MarkowitzOptimizer — mean-variance optimisation
    - backtest   : Backtester — monthly rebalance backtesting engine
    - tearsheet  : TearsheetGenerator — professional tearsheet output
"""

from portfolio.optimizer import MarkowitzOptimizer
from portfolio.backtest import Backtester
from portfolio.tearsheet import TearsheetGenerator

__all__ = ["MarkowitzOptimizer", "Backtester", "TearsheetGenerator"]
