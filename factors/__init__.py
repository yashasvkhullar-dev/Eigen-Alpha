"""
EigenAlpha — Factors Package
============================

Provides:
    - engine       : FactorEngine — computes momentum, volatility, volume trend
    - ic_analysis  : InformationCoefficient — IC/IR statistics and plots
"""

from factors.engine import FactorEngine
from factors.ic_analysis import InformationCoefficient

__all__ = ["FactorEngine", "InformationCoefficient"]
