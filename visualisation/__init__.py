"""
Visualisation Package — EigenAlpha
====================================

Provides plotting utilities for exploratory data analysis (EDA) and
PCA/clustering visualisation. All plots follow a consistent visual style
using matplotlib and seaborn.

Modules:
    eda: EDADashboard class for factor distribution and correlation analysis.
    pca_plots: PCA scree charts, cluster scatter, and sector heatmaps.
"""

from visualisation.eda import EDADashboard
from visualisation.pca_plots import PCAPlotter

__all__ = ["EDADashboard", "PCAPlotter"]
