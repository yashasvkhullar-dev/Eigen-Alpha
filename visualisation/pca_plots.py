"""
PCA & Clustering Visualisation Module
=======================================

This module provides specialised visualisation tools for the PCA decomposition
and K-Means clustering stages of the EigenAlpha pipeline. The plots help the
researcher understand:

1. **Scree chart**: How many principal components are needed to explain the
   majority of variance in the Nifty 500 return covariance matrix. In typical
   equity markets, 10–30 PCs explain ~80% of variance, with the first PC
   capturing the "market factor" (β to the index).

2. **PCA scatter**: 2D or 3D projection of stocks onto principal components,
   revealing the geometric structure of the return covariance.

3. **Cluster visualisation**: K-Means clusters overlaid on the PC projection,
   showing how mathematical clustering differs from traditional sector
   classification. Often, stocks from different GICS sectors cluster together
   because they share common risk exposures.

Usage:
    >>> from visualisation.pca_plots import PCAPlotter
    >>> plotter = PCAPlotter(decomposer, clusterer)
    >>> plotter.scree_chart()
    >>> plotter.cluster_scatter_2d()

Author: EigenAlpha Research
"""

import logging
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class PCAPlotter:
    """Visualisation tools for PCA decomposition and K-Means clustering.

    Attributes:
        decomposer: CovarianceDecomposer instance with fitted PCA.
        clusterer: Optional MarketClusterer instance with fitted K-Means.

    Args:
        decomposer: A fitted CovarianceDecomposer object.
            Must have completed eigendecomposition and PCA fitting.
        clusterer: Optional fitted MarketClusterer object.
            Must have computed PC scores and cluster labels.

    Raises:
        ValueError: If decomposer has not been fitted.
    """

    def __init__(
        self,
        decomposer: "CovarianceDecomposer",
        clusterer: Optional["MarketClusterer"] = None,
    ) -> None:
        if decomposer is None:
            raise ValueError("decomposer must not be None.")
        self.decomposer = decomposer
        self.clusterer = clusterer
        logger.info("PCAPlotter initialised.")

    def scree_chart(
        self,
        n_components: int = 30,
        variance_threshold: float = 0.80,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """Plot PCA scree chart with cumulative variance line.

        The scree chart is the standard diagnostic for determining the
        dimensionality of the return covariance matrix. It plots the explained
        variance ratio of each principal component as a bar, overlaid with a
        cumulative line. A horizontal threshold line marks the target variance
        level (default 80%).

        Args:
            n_components: Maximum number of components to display.
            variance_threshold: Cumulative variance threshold to mark on chart.
            save_path: If provided, saves the figure to this path.
            figsize: Figure dimensions.

        Returns:
            matplotlib.figure.Figure: The generated figure.

        Raises:
            ValueError: If PCA has not been fitted on the decomposer.
        """
        if not hasattr(self.decomposer, "pca") or self.decomposer.pca is None:
            raise ValueError(
                "PCA has not been fitted. Call decomposer.fit_pca() first."
            )

        pca = self.decomposer.pca
        n_show = min(n_components, len(pca.explained_variance_ratio_))
        var_ratio = pca.explained_variance_ratio_[:n_show]
        cum_var = np.cumsum(var_ratio)

        fig, ax1 = plt.subplots(figsize=figsize)

        # Bar chart: individual variance
        bars = ax1.bar(
            range(1, n_show + 1), var_ratio * 100,
            color="#1a73e8", alpha=0.6, edgecolor="white", linewidth=0.5,
            label="Individual Variance",
        )

        # Highlight the first PC (market factor)
        if n_show > 0:
            bars[0].set_color("#0d47a1")
            bars[0].set_alpha(0.9)

        ax1.set_xlabel("Principal Component", fontsize=11)
        ax1.set_ylabel("Variance Explained (%)", fontsize=11, color="#1a73e8")
        ax1.tick_params(axis="y", labelcolor="#1a73e8")

        # Line chart: cumulative variance (secondary axis)
        ax2 = ax1.twinx()
        ax2.plot(
            range(1, n_show + 1), cum_var * 100,
            color="#c62828", marker="o", markersize=4, linewidth=2.0,
            label="Cumulative Variance",
        )

        # Threshold line
        ax2.axhline(
            y=variance_threshold * 100,
            color="#888888", linestyle="--", linewidth=1.5, alpha=0.7,
        )
        ax2.text(
            n_show * 0.95, variance_threshold * 100 + 1,
            f"{variance_threshold:.0%} threshold",
            ha="right", fontsize=9, color="#888888",
        )

        # Mark the component where threshold is crossed
        if hasattr(self.decomposer, "select_components"):
            k = self.decomposer.select_components(variance_threshold)
            ax2.axvline(
                x=k, color="#43a047", linestyle="-.", linewidth=1.5, alpha=0.7,
            )
            ax2.text(
                k + 0.3, cum_var[min(k - 1, n_show - 1)] * 100 - 3,
                f"k = {k}", fontsize=10, fontweight="bold", color="#43a047",
            )

        ax2.set_ylabel("Cumulative Variance (%)", fontsize=11, color="#c62828")
        ax2.tick_params(axis="y", labelcolor="#c62828")
        ax2.set_ylim(0, 105)

        ax1.set_title(
            "PCA Scree Chart — Explained Variance by Component\n"
            f"(PC1 = Market Factor: {var_ratio[0] * 100:.1f}% of total variance)",
            fontsize=13, fontweight="bold",
        )

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2, labels1 + labels2,
            loc="center right", fontsize=9,
        )

        ax1.set_xticks(range(1, n_show + 1, max(1, n_show // 15)))
        ax1.grid(True, alpha=0.2, axis="y")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info("Scree chart saved to %s", save_path)

        return fig

    def eigenvalue_spectrum(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """Plot the eigenvalue spectrum with Marchenko-Pastur theoretical bound.

        Compares the empirical eigenvalue distribution against the
        Marchenko-Pastur distribution to identify eigenvalues that represent
        genuine signal vs. noise. Eigenvalues above the MP upper bound are
        considered signal components.

        Academic reference:
            Marchenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues
            for some sets of random matrices. *Math. USSR Sbornik*, 1(4), 457.

        Args:
            save_path: If provided, saves the figure to this path.
            figsize: Figure dimensions.

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        if self.decomposer.eigenvalues is None:
            raise ValueError("Eigendecomposition not performed.")

        eigenvalues = self.decomposer.eigenvalues

        fig, ax = plt.subplots(figsize=figsize)

        # Empirical eigenvalue spectrum
        ax.bar(
            range(1, len(eigenvalues) + 1), eigenvalues,
            color="#1a73e8", alpha=0.5, edgecolor="white", linewidth=0.3,
        )

        # Marchenko-Pastur upper bound for random matrices
        # λ+ = σ² (1 + √(N/T))²
        if hasattr(self.decomposer, "return_matrix"):
            T, N = self.decomposer.return_matrix.shape
            q = N / T
            sigma_sq = np.mean(eigenvalues)
            lambda_plus = sigma_sq * (1 + np.sqrt(q)) ** 2
            lambda_minus = sigma_sq * (1 - np.sqrt(q)) ** 2

            ax.axhline(
                y=lambda_plus, color="#c62828", linestyle="--",
                linewidth=1.5, label=f"MP Upper Bound (λ+ = {lambda_plus:.4f})",
            )
            ax.axhline(
                y=lambda_minus, color="#ff7043", linestyle="--",
                linewidth=1.0, alpha=0.7,
                label=f"MP Lower Bound (λ- = {lambda_minus:.4f})",
            )

            # Count signal eigenvalues
            n_signal = np.sum(eigenvalues > lambda_plus)
            ax.text(
                0.98, 0.95,
                f"{n_signal} eigenvalues above MP bound\n(signal components)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.5",
                                       facecolor="white", alpha=0.8),
            )

        ax.set_xlabel("Eigenvalue Index (descending)", fontsize=11)
        ax.set_ylabel("Eigenvalue", fontsize=11)
        ax.set_title(
            "Eigenvalue Spectrum vs Marchenko-Pastur Bound",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.2, axis="y")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info("Eigenvalue spectrum saved to %s", save_path)

        return fig

    def cluster_scatter_2d(
        self,
        annotate_top_n: int = 5,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
    ) -> plt.Figure:
        """Scatter plot of stocks on PC1 vs PC2, coloured by K-Means cluster.

        Each point represents a stock, positioned by its projection onto the
        first two principal components. Colours indicate cluster membership.
        Notable stocks within each cluster are annotated with ticker labels.

        Args:
            annotate_top_n: Number of stocks to annotate per cluster.
                Selects the stocks closest to the cluster centroid.
            save_path: If provided, saves the figure to this path.
            figsize: Figure dimensions.

        Returns:
            matplotlib.figure.Figure: The generated figure.

        Raises:
            ValueError: If clusterer is not provided or not fitted.
        """
        if self.clusterer is None:
            raise ValueError("MarketClusterer not provided.")
        if self.clusterer.pc_scores is None:
            raise ValueError("PC scores not computed. Call get_stock_pc_scores() first.")
        if self.clusterer.cluster_labels is None:
            raise ValueError("Clusters not fitted. Call fit_kmeans() first.")

        scores = self.clusterer.pc_scores
        labels = self.clusterer.cluster_labels

        fig, ax = plt.subplots(figsize=figsize)

        # Get unique clusters
        unique_clusters = sorted(labels.unique())
        cmap = plt.cm.get_cmap("tab10", len(unique_clusters))

        for cluster_id in unique_clusters:
            mask = labels == cluster_id
            tickers_in_cluster = mask[mask].index
            cluster_scores = scores.loc[
                scores.index.intersection(tickers_in_cluster)
            ]

            if cluster_scores.empty:
                continue

            ax.scatter(
                cluster_scores["PC1"], cluster_scores["PC2"],
                c=[cmap(cluster_id)], s=25, alpha=0.6,
                edgecolors="white", linewidths=0.3,
                label=f"Cluster {cluster_id} (n={len(cluster_scores)})",
            )

            # Annotate top_n stocks (closest to centroid)
            centroid = cluster_scores[["PC1", "PC2"]].mean()
            dists = np.sqrt(
                (cluster_scores["PC1"] - centroid["PC1"]) ** 2 +
                (cluster_scores["PC2"] - centroid["PC2"]) ** 2
            )
            closest = dists.nsmallest(annotate_top_n).index
            for ticker in closest:
                ax.annotate(
                    ticker.replace(".NS", ""),
                    xy=(scores.loc[ticker, "PC1"], scores.loc[ticker, "PC2"]),
                    fontsize=6, alpha=0.8,
                    textcoords="offset points", xytext=(5, 3),
                )

        ax.set_xlabel("PC1 (Market Factor)", fontsize=11)
        ax.set_ylabel("PC2", fontsize=11)
        ax.set_title(
            f"Market Clusters — K-Means on PCA Projections (k={len(unique_clusters)})",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.2)
        ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color="black", linewidth=0.5, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info("Cluster scatter saved to %s", save_path)

        return fig

    def cluster_scatter_3d(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
    ) -> plt.Figure:
        """3D scatter plot of stocks on PC1, PC2, PC3, coloured by cluster.

        Args:
            save_path: If provided, saves the figure to this path.
            figsize: Figure dimensions.

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        if self.clusterer is None or self.clusterer.pc_scores is None:
            raise ValueError("Clusterer not fitted or PC scores not computed.")

        scores = self.clusterer.pc_scores
        labels = self.clusterer.cluster_labels

        if "PC3" not in scores.columns:
            logger.warning("PC3 not available in scores. Falling back to 2D.")
            return self.cluster_scatter_2d(save_path=save_path, figsize=figsize)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        unique_clusters = sorted(labels.unique())
        cmap = plt.cm.get_cmap("tab10", len(unique_clusters))

        for cluster_id in unique_clusters:
            mask = labels == cluster_id
            tickers = mask[mask].index
            cs = scores.loc[scores.index.intersection(tickers)]
            if cs.empty:
                continue

            ax.scatter(
                cs["PC1"], cs["PC2"], cs["PC3"],
                c=[cmap(cluster_id)], s=20, alpha=0.6,
                label=f"Cluster {cluster_id}",
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(
            "3D Market Clusters (PC1 × PC2 × PC3)",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=7, loc="upper left")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info("3D cluster scatter saved to %s", save_path)

        return fig

    def silhouette_plot(
        self,
        k_range: range = range(3, 15),
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """Plot silhouette scores for different values of k.

        The silhouette score measures how well-defined the clusters are:
        - Values near +1: well-separated clusters.
        - Values near 0: overlapping clusters.
        - Values near -1: misassigned samples.

        The optimal k is the one that maximises the silhouette score.

        Args:
            k_range: Range of k values to evaluate.
            save_path: If provided, saves the figure to this path.
            figsize: Figure dimensions.

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        if self.clusterer is None:
            raise ValueError("MarketClusterer not provided.")

        sil_scores = self.clusterer.silhouette_analysis(k_range)

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            sil_scores.index, sil_scores.values,
            marker="o", markersize=8, linewidth=2.0, color="#1a73e8",
        )

        # Highlight optimal k
        optimal_k = sil_scores.idxmax()
        optimal_score = sil_scores.max()
        ax.scatter(
            [optimal_k], [optimal_score],
            s=150, color="#c62828", zorder=5, edgecolors="white", linewidths=2,
        )
        ax.annotate(
            f"Optimal k = {optimal_k}\n(score = {optimal_score:.3f})",
            xy=(optimal_k, optimal_score),
            xytext=(optimal_k + 1, optimal_score - 0.02),
            fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#c62828"),
        )

        ax.set_xlabel("Number of Clusters (k)", fontsize=11)
        ax.set_ylabel("Silhouette Score", fontsize=11)
        ax.set_title(
            "Silhouette Analysis for Optimal k Selection",
            fontsize=13, fontweight="bold",
        )
        ax.set_xticks(list(k_range))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info("Silhouette plot saved to %s", save_path)

        return fig

    def cluster_vs_sector_heatmap(
        self,
        sector_map: Dict[str, str],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8),
    ) -> plt.Figure:
        """Cross-tabulate K-Means clusters vs GICS sectors as a heatmap.

        This visualisation demonstrates that mathematical clusters derived from
        return covariance structure cut across traditional sector boundaries.
        Stocks from different sectors often cluster together because they share
        common risk factor exposures beyond sector membership.

        Args:
            sector_map: Dictionary mapping ticker → sector name.
            save_path: If provided, saves the figure to this path.
            figsize: Figure dimensions.

        Returns:
            matplotlib.figure.Figure: The generated figure.

        Raises:
            ValueError: If clusterer is not fitted.
        """
        if self.clusterer is None or self.clusterer.cluster_labels is None:
            raise ValueError("Clusters not fitted.")

        labels = self.clusterer.cluster_labels

        # Map tickers to sectors
        sector_series = pd.Series(sector_map).reindex(labels.index)
        sector_series = sector_series.fillna("Unknown")

        # Cross-tabulation
        cross_tab = pd.crosstab(
            labels.rename("Cluster"),
            sector_series.rename("Sector"),
        )

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            cross_tab, ax=ax, annot=True, fmt="d",
            cmap="YlOrRd", linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Number of Stocks"},
        )

        ax.set_title(
            "Cluster vs Sector Heatmap\n"
            "(Mathematical clusters vs traditional sector classification)",
            fontsize=13, fontweight="bold",
        )
        ax.set_ylabel("K-Means Cluster", fontsize=11)
        ax.set_xlabel("Sector", fontsize=11)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info("Cluster vs sector heatmap saved to %s", save_path)

        return fig
