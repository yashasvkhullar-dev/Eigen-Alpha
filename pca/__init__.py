"""
EigenAlpha — PCA & Clustering Package
=====================================

Provides:
    - decompose : CovarianceDecomposer — covariance matrix, eigendecomposition, PCA
    - cluster   : MarketClusterer — K-Means clustering, silhouette analysis
"""

from pca.decompose import CovarianceDecomposer
from pca.cluster import MarketClusterer

__all__ = ["CovarianceDecomposer", "MarketClusterer"]
