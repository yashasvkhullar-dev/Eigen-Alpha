"""
EigenAlpha — Market Clustering via PCA
=======================================

The ``MarketClusterer`` class projects stocks into principal component
space and identifies natural groupings using K-Means clustering.  This
reveals the latent factor structure of the equity market — clusters that
often cut across traditional sector boundaries.

The key insight is that mathematical (statistical) clusters capture
co-movement structures that GICS sector classifications miss.  Two
stocks in different sectors may behave similarly because they share
exposure to common macro factors (e.g., domestic consumption,
export-driven growth, credit cycle sensitivity).

Academic references:
    Ahn, S., Horenstein, A. R. (2013). Eigenvalue Ratio Test for the
    Number of Factors. *Econometrica*, 81(3), 1203–1227.

    Rousseeuw, P. J. (1987). Silhouettes: a Graphical Aid to the
    Interpretation and Validation of Cluster Analysis.
    *Journal of Computational and Applied Mathematics*, 20, 53–65.

Usage:
    from pca.cluster import MarketClusterer
    clusterer = MarketClusterer(decomposer, n_clusters=8)
    labels = clusterer.fit_kmeans()
    clusterer.plot_clusters_2d()
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from pca.decompose import CovarianceDecomposer

logger = logging.getLogger(__name__)


class MarketClusterer:
    """Cluster stocks based on their PCA factor loadings.

    Projects each stock (column of the return matrix) onto the top
    principal components and applies K-Means clustering to identify
    groups of stocks with similar statistical behaviour.

    Attributes:
        decomposer (CovarianceDecomposer): Fitted PCA decomposer.
        n_clusters (int): Number of clusters for K-Means.
        pc_scores (pd.DataFrame): Stock-level PC scores.
        cluster_labels (pd.Series): Cluster assignment per ticker.
        kmeans (KMeans): Fitted KMeans object.
    """

    def __init__(
        self,
        decomposer: CovarianceDecomposer,
        n_clusters: int = 8,
    ) -> None:
        """Initialise the MarketClusterer.

        Args:
            decomposer: A ``CovarianceDecomposer`` instance with PCA
                already fitted (i.e., ``fit_pca()`` has been called).
            n_clusters: Number of clusters for K-Means.  Defaults to 8.

        Raises:
            RuntimeError: If PCA has not been fitted on the decomposer.
            ValueError: If ``n_clusters`` < 2.
        """
        if decomposer.pca is None:
            raise RuntimeError(
                "PCA must be fitted on the CovarianceDecomposer before "
                "clustering. Call decomposer.fit_pca() first."
            )
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2. Got {n_clusters}.")

        self.decomposer = decomposer
        self.n_clusters = n_clusters
        self.pc_scores: Optional[pd.DataFrame] = None
        self.cluster_labels: Optional[pd.Series] = None
        self.kmeans: Optional[KMeans] = None

        logger.info("MarketClusterer initialised with n_clusters=%d", n_clusters)

    # ──────────────────────────────────────────────────────────────────────
    # PC Score Projection
    # ──────────────────────────────────────────────────────────────────────

    def get_stock_pc_scores(self, n_dims: int = 3) -> pd.DataFrame:
        """Project each stock onto the top principal components.

        Each stock is a column in the return matrix (T observations).
        We transpose so that stocks become rows, then transform using
        the fitted PCA to get each stock's loading on the top PCs.

        Args:
            n_dims: Number of PC dimensions to retain.  Defaults to 3.

        Returns:
            pd.DataFrame: DataFrame with index=ticker, columns=
                ``['PC1', 'PC2', ..., 'PC{n_dims}']``.
        """
        n_dims = min(n_dims, self.decomposer.pca.n_components_)

        # PCA was fit on (T × N) data.  components_ has shape
        # (n_components, N), where each row is a PC direction in
        # stock-space.  components_[j, i] = loading of stock i on PC j.
        # Transposing gives us an (N × n_components) matrix where each
        # row is a stock's coordinate vector in PC space — exactly the
        # "factor loadings" we need for clustering.
        loadings = self.decomposer.pca.components_[:n_dims, :].T  # (N, n_dims)

        col_names = [f"PC{i+1}" for i in range(n_dims)]
        self.pc_scores = pd.DataFrame(
            loadings,
            index=self.decomposer.tickers,
            columns=col_names,
        )

        logger.info(
            "Stock PC scores: %d stocks × %d dimensions",
            len(self.pc_scores),
            n_dims,
        )

        return self.pc_scores

    # ──────────────────────────────────────────────────────────────────────
    # K-Means Clustering
    # ──────────────────────────────────────────────────────────────────────

    def fit_kmeans(self) -> pd.Series:
        """Fit K-Means clustering on stock PC scores.

        Uses ``n_init=20`` for robustness (multiple random initialisations)
        and ``random_state=42`` for reproducibility.

        Returns:
            pd.Series: Cluster labels indexed by ticker
                (values 0 to n_clusters-1).
        """
        if self.pc_scores is None:
            self.get_stock_pc_scores(n_dims=3)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=20,
            max_iter=300,
        )
        labels = self.kmeans.fit_predict(self.pc_scores.values)

        self.cluster_labels = pd.Series(
            labels,
            index=self.pc_scores.index,
            name="cluster_id",
        )

        # Log cluster sizes
        sizes = self.cluster_labels.value_counts().sort_index()
        logger.info(
            "K-Means clustering: %d clusters. Sizes: %s",
            self.n_clusters,
            sizes.to_dict(),
        )

        return self.cluster_labels

    # ──────────────────────────────────────────────────────────────────────
    # Silhouette Analysis
    # ──────────────────────────────────────────────────────────────────────

    def silhouette_analysis(
        self, k_range: Optional[range] = None
    ) -> pd.Series:
        """Evaluate clustering quality across a range of k values.

        For each k, fits K-Means and computes the silhouette score.
        Higher silhouette scores indicate better-defined clusters.

        Academic reference:
            Rousseeuw, P. J. (1987). Silhouettes: a Graphical Aid to the
            Interpretation and Validation of Cluster Analysis.

        Args:
            k_range: Range of k values to test.  Defaults to ``range(3, 15)``.

        Returns:
            pd.Series: Silhouette scores indexed by k.
        """
        if k_range is None:
            k_range = range(3, 15)

        if self.pc_scores is None:
            self.get_stock_pc_scores(n_dims=3)

        scores = {}
        data = self.pc_scores.values

        for k in k_range:
            if k >= len(data):
                logger.warning("k=%d >= n_samples=%d, skipping.", k, len(data))
                continue

            km = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = km.fit_predict(data)

            # Academic reference: Rousseeuw (1987) — silhouette coefficient
            score = silhouette_score(data, labels)
            scores[k] = score

        result = pd.Series(scores, name="silhouette_score")
        result.index.name = "n_clusters"

        best_k = result.idxmax()
        logger.info(
            "Silhouette analysis: best k=%d (score=%.4f). All: %s",
            best_k,
            result[best_k],
            {k: f"{v:.4f}" for k, v in scores.items()},
        )

        return result

    # ──────────────────────────────────────────────────────────────────────
    # Visualisation: 2D Cluster Scatter
    # ──────────────────────────────────────────────────────────────────────

    def plot_clusters_2d(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Scatter plot of stocks in PC1–PC2 space, coloured by cluster.

        Annotates 5 notable stocks per cluster with ticker labels for
        interpretability.

        Args:
            ax: Optional matplotlib Axes.  If ``None``, creates a new figure.

        Returns:
            matplotlib.axes.Axes: The plot axes.

        Raises:
            RuntimeError: If clustering has not been performed yet.
        """
        if self.cluster_labels is None:
            raise RuntimeError("Call fit_kmeans() before plotting clusters.")
        if self.pc_scores is None or "PC1" not in self.pc_scores.columns:
            raise RuntimeError("PC scores must include at least PC1 and PC2.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        cmap = plt.cm.get_cmap("tab10", self.n_clusters)

        for cid in range(self.n_clusters):
            mask = self.cluster_labels == cid
            tickers_in_cluster = self.pc_scores.index[mask]
            x = self.pc_scores.loc[tickers_in_cluster, "PC1"]
            y = self.pc_scores.loc[tickers_in_cluster, "PC2"]

            ax.scatter(
                x, y,
                c=[cmap(cid)],
                label=f"Cluster {cid} (n={mask.sum()})",
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
                s=60,
            )

            # Annotate up to 5 stocks per cluster (largest absolute PC1)
            notable = x.abs().nlargest(5).index
            for ticker in notable:
                short_name = ticker.replace(".NS", "")
                ax.annotate(
                    short_name,
                    (x[ticker], y[ticker]),
                    fontsize=7,
                    alpha=0.8,
                    ha="center",
                    va="bottom",
                )

        ax.set_xlabel("PC1", fontsize=11)
        ax.set_ylabel("PC2", fontsize=11)
        ax.set_title(
            f"Stock Clusters in PCA Space (k={self.n_clusters})",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=8,
        )
        ax.grid(alpha=0.3)
        plt.tight_layout()

        return ax

    # ──────────────────────────────────────────────────────────────────────
    # Cluster vs Sector Heatmap
    # ──────────────────────────────────────────────────────────────────────

    def cluster_vs_sector_heatmap(
        self,
        sector_map: dict,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Cross-tabulate cluster assignments against GICS sectors.

        This visualisation demonstrates that mathematical clusters
        (based on return co-movement) often cut across traditional
        sector boundaries — a key insight for factor-based portfolio
        construction.

        Args:
            sector_map: Dictionary mapping ticker (without ``.NS`` suffix)
                to sector name.
            ax: Optional matplotlib Axes.

        Returns:
            matplotlib.axes.Axes: The heatmap axes.

        Raises:
            RuntimeError: If clustering has not been performed yet.
        """
        if self.cluster_labels is None:
            raise RuntimeError("Call fit_kmeans() before plotting heatmap.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        # Map tickers to sectors
        sectors = pd.Series(
            {
                t: sector_map.get(t.replace(".NS", ""), "Unknown")
                for t in self.cluster_labels.index
            },
            name="sector",
        )

        # Cross-tabulation
        cross_tab = pd.crosstab(
            self.cluster_labels,
            sectors,
            margins=False,
        )
        cross_tab.index.name = "Cluster"

        # Plot heatmap
        sns.heatmap(
            cross_tab,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            linewidths=0.5,
            ax=ax,
        )

        ax.set_title(
            "Cluster vs Sector Distribution\n"
            "(Mathematical clusters cut across sector boundaries)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_ylabel("Cluster ID")
        ax.set_xlabel("GICS Sector")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        return ax
