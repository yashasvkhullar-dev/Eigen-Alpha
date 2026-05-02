"""
Exploratory Data Analysis Dashboard — Factor Diagnostics
==========================================================

This module provides a comprehensive suite of EDA visualisations designed for
factor research. The EDADashboard class produces publication-quality plots that
answer the fundamental questions a quant researcher asks before committing to a
factor-based strategy:

1. **Factor Distributions**: Are the factors approximately normal after
   winsorisation and z-scoring? Heavy tails or bimodality suggest the need for
   non-linear models or further preprocessing.

2. **Factor Correlations**: Are the factors sufficiently orthogonal? Low
   inter-factor correlation (|ρ| < 0.3) is desirable for building diversified
   multi-factor portfolios. Highly correlated factors provide redundant information.

3. **Signal Persistence (Autocorrelation)**: How quickly does the factor signal
   decay? Factors with high 1-month autocorrelation and slow decay are more
   tradeable because they require lower turnover.

4. **Portfolio Turnover**: What fraction of the top quintile changes each month?
   High turnover implies high transaction costs, which erode real-world alpha.
   For Indian equities, round-trip costs (brokerage + STT + impact) can be
   50–100 bps for institutional investors.

Usage:
    >>> from visualisation.eda import EDADashboard
    >>> dashboard = EDADashboard(factor_data, prices)
    >>> dashboard.factor_distributions()
    >>> dashboard.factor_correlation_heatmap()

Author: EigenAlpha Research
"""

import logging
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Style configuration
sns.set_style("whitegrid")
FACTOR_COLS = ["momentum_12_1", "realized_vol", "volume_trend"]
FACTOR_LABELS = {
    "momentum_12_1": "Momentum (12-1)",
    "realized_vol": "Realized Volatility (20d)",
    "volume_trend": "Volume Trend (20d)",
}
FACTOR_COLORS = {
    "momentum_12_1": "#1a73e8",
    "realized_vol": "#e53935",
    "volume_trend": "#43a047",
}


class EDADashboard:
    """Exploratory data analysis dashboard for factor research.

    Produces a suite of diagnostic plots that help the researcher understand
    the statistical properties of computed factors before proceeding to
    portfolio construction.

    Attributes:
        factor_data (pd.DataFrame): Long-format factor data with columns
            [date, ticker, momentum_12_1, realized_vol, volume_trend].
        prices (pd.DataFrame): Wide-format price matrix (dates × tickers).
        available_factors (list): Factor columns present in the data.

    Args:
        factor_data: Long-format DataFrame with computed factor scores.
            Must contain columns ['date', 'ticker'] plus at least one factor.
        prices: Wide-format adjusted close prices.
            Index = DatetimeIndex, columns = ticker symbols.

    Raises:
        ValueError: If factor_data is empty or missing required columns.
    """

    def __init__(
        self, factor_data: pd.DataFrame, prices: pd.DataFrame
    ) -> None:
        if factor_data.empty:
            raise ValueError("factor_data must not be empty.")
        required = {"date", "ticker"}
        missing = required - set(factor_data.columns)
        if missing:
            raise ValueError(f"factor_data missing columns: {missing}")

        self.factor_data = factor_data.copy()
        self.prices = prices.copy()

        # Identify which factor columns are present
        self.available_factors = [
            c for c in FACTOR_COLS if c in self.factor_data.columns
        ]
        if not self.available_factors:
            logger.warning("No standard factor columns found in factor_data.")

        logger.info(
            "EDADashboard initialised with %d factors, %d observations.",
            len(self.available_factors),
            len(self.factor_data),
        )

    def factor_distributions(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5),
    ) -> plt.Figure:
        """Plot KDE + histogram for each factor.

        Creates one subplot per factor showing the empirical distribution after
        winsorisation and z-scoring. Overlays a standard normal density for
        reference. Reports skewness and kurtosis in the title.

        Args:
            save_path: If provided, saves the figure to this path.
            figsize: Figure dimensions (width, height) in inches.

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        n_factors = len(self.available_factors)
        if n_factors == 0:
            logger.warning("No factors available for distribution plot.")
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.text(0.5, 0.5, "No factor data available",
                    ha="center", va="center", transform=ax.transAxes)
            return fig

        fig, axes = plt.subplots(1, n_factors, figsize=figsize)
        if n_factors == 1:
            axes = [axes]

        for i, factor in enumerate(self.available_factors):
            ax = axes[i]
            values = self.factor_data[factor].dropna()
            color = FACTOR_COLORS.get(factor, "#666666")

            # Histogram + KDE
            ax.hist(
                values, bins=80, density=True, alpha=0.4,
                color=color, edgecolor="white", linewidth=0.5,
            )
            sns.kdeplot(values, ax=ax, color=color, linewidth=1.5)

            # Reference: standard normal
            x_range = np.linspace(values.min(), values.max(), 200)
            from scipy.stats import norm
            ax.plot(
                x_range, norm.pdf(x_range, 0, 1),
                color="#999999", linestyle="--", linewidth=1.0,
                label="N(0,1)",
            )

            skew = values.skew()
            kurt = values.kurtosis()
            label = FACTOR_LABELS.get(factor, factor)
            ax.set_title(
                f"{label}\nSkew={skew:.2f}, Kurt={kurt:.2f}",
                fontsize=10, fontweight="bold",
            )
            ax.set_xlabel("Z-Score", fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.legend(fontsize=7)

        fig.suptitle(
            "Factor Distributions (after winsorisation + z-scoring)",
            fontsize=12, fontweight="bold", y=1.02,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info("Factor distributions saved to %s", save_path)

        return fig

    def factor_correlation_heatmap(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 7),
    ) -> plt.Figure:
        """Plot cross-sectional average correlation heatmap between factors.

        For each date, computes the Pearson correlation matrix across stocks.
        The heatmap shows the time-averaged correlation. Low inter-factor
        correlation (|ρ| < 0.3) is desirable for portfolio diversification.

        Args:
            save_path: If provided, saves the figure to this path.
            figsize: Figure dimensions.

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        if len(self.available_factors) < 2:
            logger.warning("Need at least 2 factors for correlation heatmap.")
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.text(0.5, 0.5, "Insufficient factors",
                    ha="center", va="center", transform=ax.transAxes)
            return fig

        # Compute per-date cross-sectional correlations
        dates = self.factor_data["date"].unique()
        corr_matrices = []

        for date in dates:
            snapshot = self.factor_data[self.factor_data["date"] == date]
            if len(snapshot) < 20:
                continue
            corr = snapshot[self.available_factors].corr()
            corr_matrices.append(corr)

        if not corr_matrices:
            logger.warning("No valid cross-sections for correlation computation.")
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        # Average across dates
        avg_corr = pd.concat(corr_matrices).groupby(level=0).mean()
        avg_corr = avg_corr.reindex(
            index=self.available_factors, columns=self.available_factors
        )

        # Rename for display
        display_labels = [FACTOR_LABELS.get(f, f) for f in self.available_factors]
        avg_corr.index = display_labels
        avg_corr.columns = display_labels

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        mask = np.triu(np.ones_like(avg_corr, dtype=bool), k=1)

        sns.heatmap(
            avg_corr, ax=ax, mask=mask,
            annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, linewidths=1,
            annot_kws={"size": 12, "fontweight": "bold"},
            cbar_kws={"shrink": 0.8, "label": "Correlation"},
        )

        ax.set_title(
            "Cross-Sectional Average Factor Correlations\n"
            "(Target: |ρ| < 0.3 for factor orthogonality)",
            fontsize=11, fontweight="bold",
        )

        # Annotate whether factors are sufficiently orthogonal
        max_abs_corr = avg_corr.values[np.tril_indices_from(avg_corr.values, k=-1)]
        max_abs = np.max(np.abs(max_abs_corr)) if len(max_abs_corr) > 0 else 0
        status = "✓ Factors are sufficiently orthogonal" if max_abs < 0.3 else \
                 "⚠ High inter-factor correlation detected"
        ax.text(
            0.5, -0.05, status,
            ha="center", va="top", transform=ax.transAxes,
            fontsize=10, style="italic",
            color="#2e7d32" if max_abs < 0.3 else "#c62828",
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info("Correlation heatmap saved to %s", save_path)

        return fig

    def autocorrelation_decay(
        self,
        max_lag: int = 12,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """Plot cross-sectional rank autocorrelation at lags 1M through 12M.

        Signal persistence measures how stable factor rankings are over time.
        A factor with high rank autocorrelation at lag 1 (e.g., > 0.8) and
        slow decay is more tradeable because it requires lower turnover.

        Args:
            max_lag: Maximum lag in months to compute autocorrelation.
            save_path: If provided, saves the figure to this path.
            figsize: Figure dimensions.

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        for factor in self.available_factors:
            # Pivot to wide: date × ticker
            pivot = self.factor_data.pivot_table(
                index="date", columns="ticker", values=factor
            )
            dates = pivot.index.sort_values()

            autocorrs = []
            for lag in range(1, max_lag + 1):
                corrs = []
                for i in range(lag, len(dates)):
                    current = pivot.loc[dates[i]].dropna()
                    lagged = pivot.loc[dates[i - lag]].dropna()
                    common = current.index.intersection(lagged.index)
                    if len(common) >= 20:
                        # Spearman rank correlation
                        rank_corr = current[common].rank().corr(lagged[common].rank())
                        corrs.append(rank_corr)
                autocorrs.append(np.mean(corrs) if corrs else np.nan)

            color = FACTOR_COLORS.get(factor, "#666666")
            label = FACTOR_LABELS.get(factor, factor)
            ax.plot(
                range(1, max_lag + 1), autocorrs,
                marker="o", markersize=5, linewidth=1.5,
                color=color, label=label,
            )

        ax.set_xlabel("Lag (months)", fontsize=10)
        ax.set_ylabel("Rank Autocorrelation", fontsize=10)
        ax.set_title(
            "Factor Signal Persistence — Cross-Sectional Rank Autocorrelation",
            fontsize=12, fontweight="bold",
        )
        ax.set_xticks(range(1, max_lag + 1))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info("Autocorrelation decay plot saved to %s", save_path)

        return fig

    def turnover_analysis(
        self,
        n_quantiles: int = 5,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 5),
    ) -> plt.Figure:
        """Analyse and plot monthly portfolio turnover for the top quintile.

        Turnover = fraction of the top quintile that changes each month.
        High turnover means high transaction costs. For Indian equities,
        typical round-trip costs are:
        - Retail: ~50 bps (brokerage + STT + exchange charges)
        - Institutional: ~20–30 bps (negotiated brokerage)

        Args:
            n_quantiles: Number of quantile buckets.
            save_path: If provided, saves the figure to this path.
            figsize: Figure dimensions.

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        n_factors = len(self.available_factors)
        if n_factors == 0:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.text(0.5, 0.5, "No factors available",
                    ha="center", va="center", transform=ax.transAxes)
            return fig

        fig, axes = plt.subplots(1, n_factors, figsize=figsize)
        if n_factors == 1:
            axes = [axes]

        for i, factor in enumerate(self.available_factors):
            ax = axes[i]
            pivot = self.factor_data.pivot_table(
                index="date", columns="ticker", values=factor
            )
            dates = sorted(pivot.index)
            prev_top = set()
            turnover_vals = []
            turnover_dates = []

            for date in dates:
                scores = pivot.loc[date].dropna()
                if len(scores) < n_quantiles * 5:
                    continue
                try:
                    labels = pd.qcut(
                        scores, q=n_quantiles, labels=False, duplicates="drop"
                    )
                except ValueError:
                    continue

                top = set(labels[labels == n_quantiles - 1].index)
                if prev_top and len(top) > 0:
                    new_entries = top - prev_top
                    turnover = len(new_entries) / len(top)
                    turnover_vals.append(turnover)
                    turnover_dates.append(date)
                prev_top = top

            if turnover_vals:
                color = FACTOR_COLORS.get(factor, "#666666")
                ax.bar(
                    range(len(turnover_vals)), turnover_vals,
                    color=color, alpha=0.6,
                )
                mean_turnover = np.mean(turnover_vals)
                ax.axhline(
                    y=mean_turnover, color="black", linestyle="--",
                    linewidth=1.0, label=f"Mean: {mean_turnover:.1%}",
                )

            label = FACTOR_LABELS.get(factor, factor)
            ax.set_title(f"Turnover: {label}", fontsize=10, fontweight="bold")
            ax.set_ylabel("Top Quintile Turnover", fontsize=9)
            ax.set_xlabel("Month", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            "Monthly Top-Quintile Turnover Analysis",
            fontsize=12, fontweight="bold", y=1.02,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info("Turnover analysis saved to %s", save_path)

        return fig
