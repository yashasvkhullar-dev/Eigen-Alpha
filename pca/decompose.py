"""
EigenAlpha — Covariance Decomposition & PCA
============================================

The ``CovarianceDecomposer`` class performs:

1. **Sample covariance estimation** — both the pandas pairwise-complete
   method and the explicit matrix product (1/T) R^T R.
2. **Eigendecomposition** — using ``numpy.linalg.eigh`` for symmetric
   matrices (numerically stable, O(N^2) storage).
3. **PCA via scikit-learn** — full sklearn ``PCA`` fit with explained
   variance accounting.
4. **Component selection** — automatic determination of the number of
   principal components needed to explain a target fraction of total
   variance.

The return matrix R (T × N) is the fundamental input:
    - T = number of time periods (trading days)
    - N = number of assets

Academic references:
    Laloux, L., Cizeau, P., Bouchaud, J.-P., & Potters, M. (1999).
    Noise Dressing of Financial Correlation Matrices. *Physical Review
    Letters*, 83(7), 1467.

    Marchenko, V. A., & Pastur, L. A. (1967). Distribution of Eigenvalues
    for Some Sets of Random Matrices. *Mathematics of the USSR-Sbornik*,
    1(4), 457–483.

Usage:
    from pca.decompose import CovarianceDecomposer
    decomposer = CovarianceDecomposer(return_matrix)
    eigenvalues, eigenvectors = decomposer.eigendecompose()
    decomposer.fit_pca(n_components=50)
    k = decomposer.select_components(variance_threshold=0.80)
"""

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class CovarianceDecomposer:
    """Decompose the return covariance matrix using eigenanalysis and PCA.

    This class is central to understanding the factor structure of the
    equity cross-section.  The eigenvalues reveal how variance is
    distributed across independent risk dimensions, while the
    eigenvectors identify the asset loadings on each principal component.

    Attributes:
        return_matrix (pd.DataFrame): Wide-format return matrix (T × N).
        cov_matrix (np.ndarray): Sample covariance matrix (N × N).
        sigma_matrix (np.ndarray): Matrix product covariance (1/T) R^T R.
        eigenvalues (np.ndarray): Eigenvalues sorted descending.
        eigenvectors (np.ndarray): Eigenvectors as columns, matching eigenvalues.
        pca (PCA): Fitted sklearn PCA object.
        explained_variance_ratio (np.ndarray): Fraction of variance per PC.
        cumulative_variance (np.ndarray): Cumulative explained variance.
    """

    def __init__(self, return_matrix: pd.DataFrame) -> None:
        """Initialise with a return matrix.

        Args:
            return_matrix: Wide-format DataFrame with index=Date (T rows),
                columns=Tickers (N columns), values=log returns.

        Raises:
            ValueError: If return_matrix is empty or has fewer than 2 columns.
        """
        if return_matrix.empty:
            raise ValueError("return_matrix must not be empty.")
        if return_matrix.shape[1] < 2:
            raise ValueError(
                f"return_matrix must have at least 2 columns (assets). "
                f"Got {return_matrix.shape[1]}."
            )

        self.return_matrix = return_matrix.copy()
        self.tickers = list(return_matrix.columns)
        self.T, self.N = return_matrix.shape

        # Initialised by compute methods
        self.cov_matrix: Optional[np.ndarray] = None
        self.sigma_matrix: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.pca: Optional[PCA] = None
        self.explained_variance_ratio: Optional[np.ndarray] = None
        self.cumulative_variance: Optional[np.ndarray] = None

        logger.info(
            "CovarianceDecomposer initialised: T=%d dates, N=%d assets",
            self.T,
            self.N,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Covariance Estimation
    # ──────────────────────────────────────────────────────────────────────

    def compute_covariance(self) -> np.ndarray:
        """Compute the sample covariance matrix using two methods.

        Method 1 (pandas): Uses ``DataFrame.cov()`` which handles NaN
        values via pairwise complete observations.  This is the preferred
        method when data has missing values.

        Method 2 (matrix product): Computes ``Σ = (1/T) R^T R`` after
        filling NaN with 0.  This is the pure linear algebra formulation
        used in random matrix theory.

        The two methods produce slightly different results when there are
        missing values.  Both are stored for comparison.

        Returns:
            np.ndarray: The pandas-based covariance matrix (N × N).
        """
        # Method 1: Pandas pairwise covariance (handles NaN)
        self.cov_matrix = self.return_matrix.cov().values

        # Method 2: Matrix product (1/T) R^T @ R
        # Fill NaN with 0 for matrix multiplication
        R = self.return_matrix.fillna(0).values
        T = R.shape[0]
        # Academic reference: Laloux et al. (1999) — sample covariance
        self.sigma_matrix = (1.0 / T) * (R.T @ R)

        logger.info(
            "Covariance matrices computed: shape (%d, %d). "
            "Frobenius norm difference between methods: %.6f",
            self.cov_matrix.shape[0],
            self.cov_matrix.shape[1],
            np.linalg.norm(self.cov_matrix - self.sigma_matrix),
        )

        return self.cov_matrix

    # ──────────────────────────────────────────────────────────────────────
    # Eigendecomposition
    # ──────────────────────────────────────────────────────────────────────

    def eigendecompose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Perform eigendecomposition of the covariance matrix.

        Uses ``numpy.linalg.eigh`` (the symmetric matrix variant) which
        is faster and more numerically stable than ``numpy.linalg.eig``
        for real symmetric matrices.

        Eigenvalues and eigenvectors are sorted in descending order of
        eigenvalue magnitude.

        Returns:
            Tuple[np.ndarray, np.ndarray]: ``(eigenvalues, eigenvectors)``
                where eigenvalues is a 1D array of length N sorted
                descending, and eigenvectors is an N×N matrix with
                eigenvectors as columns.

        Raises:
            RuntimeError: If covariance matrix has not been computed yet.
        """
        if self.cov_matrix is None:
            self.compute_covariance()

        # Academic reference: Marchenko & Pastur (1967) — random matrix theory
        # Use eigh for symmetric matrices (covariance is always symmetric)
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix)

        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]

        # Log key statistics
        total_var = self.eigenvalues.sum()
        top1_pct = self.eigenvalues[0] / total_var * 100
        top5_pct = self.eigenvalues[:5].sum() / total_var * 100

        logger.info(
            "Eigendecomposition: %d eigenvalues. "
            "Top-1 explains %.1f%%, Top-5 explain %.1f%% of total variance",
            len(self.eigenvalues),
            top1_pct,
            top5_pct,
        )

        return self.eigenvalues, self.eigenvectors

    # ──────────────────────────────────────────────────────────────────────
    # scikit-learn PCA
    # ──────────────────────────────────────────────────────────────────────

    def fit_pca(self, n_components: int = 50) -> PCA:
        """Fit PCA using scikit-learn.

        Args:
            n_components: Number of principal components to retain.
                Capped at ``min(T, N)``.

        Returns:
            sklearn.decomposition.PCA: The fitted PCA object.
        """
        max_components = min(self.T, self.N)
        n_components = min(n_components, max_components)

        # Fill NaN with 0 for PCA (sklearn does not handle NaN)
        data = self.return_matrix.fillna(0).values

        self.pca = PCA(n_components=n_components)
        self.pca.fit(data)

        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.cumulative_variance = np.cumsum(self.explained_variance_ratio)

        logger.info(
            "PCA fitted: %d components. "
            "Cumulative variance explained: %.1f%% (top-5), %.1f%% (all %d)",
            n_components,
            self.cumulative_variance[min(4, n_components - 1)] * 100,
            self.cumulative_variance[-1] * 100,
            n_components,
        )

        return self.pca

    # ──────────────────────────────────────────────────────────────────────
    # Component Selection
    # ──────────────────────────────────────────────────────────────────────

    def select_components(self, variance_threshold: float = 0.80) -> int:
        """Select the minimum number of PCs to explain a target variance.

        Args:
            variance_threshold: Fraction of total variance to explain
                (e.g. 0.80 for 80%).  Must be in (0, 1].

        Returns:
            int: Minimum k such that the first k PCs explain at least
                ``variance_threshold`` of total variance.

        Raises:
            ValueError: If ``variance_threshold`` is not in (0, 1].
            RuntimeError: If PCA has not been fitted yet.
        """
        if not (0 < variance_threshold <= 1.0):
            raise ValueError(
                f"variance_threshold must be in (0, 1]. Got {variance_threshold}."
            )
        if self.cumulative_variance is None:
            raise RuntimeError("PCA must be fitted first. Call fit_pca().")

        # Find minimum k where cumulative variance >= threshold
        k = int(np.searchsorted(self.cumulative_variance, variance_threshold) + 1)
        k = min(k, len(self.cumulative_variance))

        logger.info(
            "Component selection: k=%d PCs explain %.1f%% of variance "
            "(threshold=%.0f%%)",
            k,
            self.cumulative_variance[k - 1] * 100,
            variance_threshold * 100,
        )

        return k

    # ──────────────────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────────────────

    def plot_scree(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot a scree chart with explained and cumulative variance.

        The scree plot is the standard diagnostic for determining how many
        principal components to retain.  It shows:
            - Bar chart of explained variance per component.
            - Line chart of cumulative explained variance.
            - Horizontal dashed line at the 80% threshold.
            - Annotation of the selected k.

        Args:
            ax: Optional matplotlib Axes.  If ``None``, creates a new figure.

        Returns:
            matplotlib.axes.Axes: The plot axes.

        Raises:
            RuntimeError: If PCA has not been fitted yet.
        """
        if self.explained_variance_ratio is None:
            raise RuntimeError("PCA must be fitted first. Call fit_pca().")

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        n = len(self.explained_variance_ratio)
        x = np.arange(1, n + 1)

        # Bar chart: individual explained variance
        ax.bar(
            x,
            self.explained_variance_ratio * 100,
            color="#3498db",
            alpha=0.7,
            label="Individual",
        )

        # Line chart: cumulative explained variance
        ax2 = ax.twinx()
        ax2.plot(
            x,
            self.cumulative_variance * 100,
            color="#e74c3c",
            linewidth=2,
            marker="o",
            markersize=3,
            label="Cumulative",
        )

        # 80% threshold line
        ax2.axhline(
            y=80,
            color="#2c3e50",
            linestyle="--",
            alpha=0.6,
            label="80% Threshold",
        )

        # Mark selected k
        k = self.select_components(0.80)
        ax2.annotate(
            f"k={k}",
            xy=(k, self.cumulative_variance[k - 1] * 100),
            xytext=(k + 2, self.cumulative_variance[k - 1] * 100 - 5),
            arrowprops=dict(arrowstyle="->", color="#2c3e50"),
            fontsize=11,
            fontweight="bold",
            color="#2c3e50",
        )

        ax.set_xlabel("Principal Component", fontsize=11)
        ax.set_ylabel("Explained Variance (%)", fontsize=11, color="#3498db")
        ax2.set_ylabel("Cumulative Variance (%)", fontsize=11, color="#e74c3c")
        ax.set_title(
            "PCA Scree Chart — Explained Variance by Component",
            fontsize=13,
            fontweight="bold",
        )

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

        ax.set_xlim(0.5, min(n + 0.5, 50.5))
        plt.tight_layout()

        return ax
