"""
Tearsheet Generator — Professional Factor Research Report
===========================================================

This module produces a comprehensive 3×3 subplot tearsheet that summarises the
entire EigenAlpha Phase 0 pipeline in a single high-resolution image. The layout
is inspired by the tearsheet conventions used by major quantitative asset managers
(AQR, Two Sigma, Man AHL) and academic factor research papers.

The tearsheet includes:
    Row 1: Cumulative returns, drawdown chart, monthly returns heatmap.
    Row 2: Quintile cumulative returns, factor IC timeseries (2 panels).
    Row 3: PCA scree chart, cluster scatter, portfolio weights.

A performance summary table is overlaid as a text annotation, reporting the
standard metrics: Sharpe, Calmar, MaxDD, Ann. Return, Ann. Vol, and IR.

Usage:
    >>> from portfolio.tearsheet import TearsheetGenerator
    >>> ts = TearsheetGenerator(port_ret, bench_ret, factor_data, ic_results)
    >>> ts.generate("outputs/tearsheet.png")

Author: EigenAlpha Research
"""

import logging
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Consistent colour palette across the entire tearsheet
COLORS = {
    "portfolio": "#1a73e8",
    "benchmark": "#aaaaaa",
    "positive": "#2e7d32",
    "negative": "#c62828",
    "q1": "#ef5350",
    "q2": "#ff7043",
    "q3": "#ffa726",
    "q4": "#66bb6a",
    "q5": "#1a73e8",
    "grid": "#e0e0e0",
    "bg": "#fafafa",
    "text": "#212121",
}


class TearsheetGenerator:
    """Generates a publication-quality 3×3 tearsheet summarising the full pipeline.

    The tearsheet is the primary deliverable of the EigenAlpha Phase 0 research
    project. It condenses all quantitative results—factor efficacy, PCA structure,
    clustering quality, and portfolio performance—into a single visual artefact
    suitable for presentation to a portfolio manager or investment committee.

    Attributes:
        portfolio_returns (pd.Series): Monthly portfolio return series.
        benchmark_returns (pd.Series): Monthly benchmark return series.
        factor_data (pd.DataFrame): Long-format factor data.
        ic_results (dict): IC analysis results keyed by factor name.
        quintile_returns (pd.DataFrame): Optional quintile backtest results.
        pca_decomposer: Optional CovarianceDecomposer instance.
        clusterer: Optional MarketClusterer instance.
        portfolio_weights (pd.Series): Optional portfolio weight allocation.
        metrics (dict): Computed performance metrics.

    Args:
        portfolio_returns: Monthly return series for the strategy.
        benchmark_returns: Monthly return series for the benchmark (Nifty 50).
        factor_data: Long-format DataFrame with factor scores.
        ic_results: Dictionary of IC analysis results, keyed by factor name.
            Each value should have keys: 'ic_series', 'mean_ic', 'ir'.
        quintile_returns: Optional quintile backtest DataFrame (Q1–Q5 + spread).
        pca_decomposer: Optional CovarianceDecomposer for scree chart.
        clusterer: Optional MarketClusterer for cluster scatter.
        portfolio_weights: Optional Series of portfolio weights per ticker.
        metrics: Optional pre-computed performance metrics dict.

    Raises:
        ValueError: If portfolio_returns is empty.
    """

    def __init__(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        factor_data: pd.DataFrame,
        ic_results: Dict[str, Any],
        quintile_returns: Optional[pd.DataFrame] = None,
        pca_decomposer: Optional[Any] = None,
        clusterer: Optional[Any] = None,
        portfolio_weights: Optional[pd.Series] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        if portfolio_returns.empty:
            raise ValueError("portfolio_returns must not be empty.")

        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        self.factor_data = factor_data
        self.ic_results = ic_results
        self.quintile_returns = quintile_returns
        self.pca_decomposer = pca_decomposer
        self.clusterer = clusterer
        self.portfolio_weights = portfolio_weights
        self.metrics = metrics or {}

        logger.info("TearsheetGenerator initialised.")

    def generate(self, output_path: str = "tearsheet.png") -> None:
        """Generate the full 3×3 tearsheet and save to disk.

        Produces a 18×12 inch figure at 150 DPI with 9 subplots arranged in
        a 3×3 grid, plus a text-based performance summary table.

        Args:
            output_path: File path to save the tearsheet image.
                Supported formats: PNG, PDF, SVG.

        Returns:
            None. Saves the figure to output_path.
        """
        logger.info("Generating tearsheet → %s", output_path)

        fig, axes = plt.subplots(
            3, 3, figsize=(18, 12), dpi=150,
            facecolor=COLORS["bg"],
        )
        fig.suptitle(
            "EigenAlpha Phase 0 — Factor Research Tearsheet",
            fontsize=16, fontweight="bold", color=COLORS["text"],
            y=0.98,
        )

        # Row 1
        self._plot_cumulative_returns(axes[0, 0])
        self._plot_drawdown(axes[0, 1])
        self._plot_monthly_heatmap(axes[0, 2])

        # Row 2
        self._plot_quintile_returns(axes[1, 0])
        self._plot_ic_timeseries(axes[1, 1], factor_name="momentum_12_1")
        self._plot_ic_timeseries_multi(axes[1, 2])

        # Row 3
        self._plot_scree(axes[2, 0])
        self._plot_cluster_scatter(axes[2, 1])
        self._plot_portfolio_weights(axes[2, 2])

        # Performance table annotation
        self._add_performance_table(fig)

        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        fig.savefig(output_path, bbox_inches="tight", facecolor=COLORS["bg"])
        plt.close(fig)

        logger.info("Tearsheet saved to %s", output_path)

    # ─── Row 1: Performance ─────────────────────────────────────────

    def _plot_cumulative_returns(self, ax: plt.Axes) -> None:
        """Plot cumulative returns: portfolio vs benchmark."""
        cum_port = (1 + self.portfolio_returns).cumprod()
        ax.plot(
            cum_port.index, cum_port.values,
            color=COLORS["portfolio"], linewidth=1.5, label="Portfolio",
        )

        if not self.benchmark_returns.empty:
            # Align benchmark to portfolio date range
            common = cum_port.index.intersection(self.benchmark_returns.index)
            if len(common) > 0:
                bench_aligned = self.benchmark_returns.loc[common]
                cum_bench = (1 + bench_aligned).cumprod()
                ax.plot(
                    cum_bench.index, cum_bench.values,
                    color=COLORS["benchmark"], linewidth=1.2,
                    linestyle="--", label="Nifty 50",
                )

        ax.set_title("Cumulative Returns", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3, color=COLORS["grid"])
        ax.set_ylabel("Growth of ₹1", fontsize=8)

    def _plot_drawdown(self, ax: plt.Axes) -> None:
        """Plot underwater (drawdown) chart."""
        cum = (1 + self.portfolio_returns).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max

        ax.fill_between(
            drawdown.index, drawdown.values, 0,
            color=COLORS["negative"], alpha=0.4,
        )
        ax.plot(
            drawdown.index, drawdown.values,
            color=COLORS["negative"], linewidth=0.8,
        )

        ax.set_title("Drawdown", fontsize=10, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(True, alpha=0.3, color=COLORS["grid"])
        ax.set_ylabel("Drawdown", fontsize=8)

    def _plot_monthly_heatmap(self, ax: plt.Axes) -> None:
        """Plot monthly returns heatmap (year × month)."""
        returns = self.portfolio_returns.copy()
        returns.index = pd.to_datetime(returns.index)

        monthly_table = pd.DataFrame({
            "year": returns.index.year,
            "month": returns.index.month,
            "return": returns.values,
        })
        pivot = monthly_table.pivot_table(
            index="year", columns="month", values="return", aggfunc="sum"
        )
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ][:len(pivot.columns)]

        sns.heatmap(
            pivot, ax=ax, cmap="RdYlGn", center=0,
            annot=True, fmt=".1%", annot_kws={"size": 6},
            linewidths=0.5, cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Monthly Returns Heatmap", fontsize=10, fontweight="bold")
        ax.set_ylabel("")
        ax.set_xlabel("")

    # ─── Row 2: Factor Analysis ─────────────────────────────────────

    def _plot_quintile_returns(self, ax: plt.Axes) -> None:
        """Plot cumulative returns for each quintile."""
        if self.quintile_returns is None or self.quintile_returns.empty:
            ax.text(0.5, 0.5, "No quintile data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title("Quintile Returns", fontsize=10, fontweight="bold")
            return

        q_colors = [COLORS["q1"], COLORS["q2"], COLORS["q3"],
                     COLORS["q4"], COLORS["q5"]]

        for i, col in enumerate(self.quintile_returns.columns):
            if col == "Long_Short":
                continue
            cum = (1 + self.quintile_returns[col].fillna(0)).cumprod()
            color = q_colors[i] if i < len(q_colors) else "#888888"
            ax.plot(cum.index, cum.values, label=col, color=color, linewidth=1.2)

        ax.set_title("Q1–Q5 Quintile Cumulative Returns", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        ax.grid(True, alpha=0.3, color=COLORS["grid"])
        ax.set_ylabel("Growth of ₹1", fontsize=8)

    def _plot_ic_timeseries(
        self, ax: plt.Axes, factor_name: str = "momentum_12_1"
    ) -> None:
        """Plot IC bar chart for a single factor with rolling mean overlay."""
        if factor_name not in self.ic_results:
            ax.text(0.5, 0.5, f"No IC data for {factor_name}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_title(f"IC: {factor_name}", fontsize=10, fontweight="bold")
            return

        ic_data = self.ic_results[factor_name]
        if "ic_series" not in ic_data:
            ax.text(0.5, 0.5, "IC series not available",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_title(f"IC: {factor_name}", fontsize=10, fontweight="bold")
            return

        ic_series = ic_data["ic_series"]
        colors = [COLORS["positive"] if v >= 0 else COLORS["negative"]
                  for v in ic_series.values]
        ax.bar(ic_series.index, ic_series.values, color=colors, alpha=0.6, width=25)

        # 6-month rolling mean
        rolling_ic = ic_series.rolling(6, min_periods=3).mean()
        ax.plot(rolling_ic.index, rolling_ic.values,
                color=COLORS["portfolio"], linewidth=1.5, label="6M Rolling IC")

        # Threshold lines
        ax.axhline(y=0.05, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(y=-0.05, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(y=0, color="black", linewidth=0.5)

        ax.set_title(f"IC: Momentum (12-1)", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, color=COLORS["grid"])

    def _plot_ic_timeseries_multi(self, ax: plt.Axes) -> None:
        """Plot rolling IC for volatility and volume trend factors."""
        plotted = False
        for factor_name, label, color in [
            ("realized_vol", "Realized Vol", "#ff7043"),
            ("volume_trend", "Volume Trend", "#ab47bc"),
        ]:
            if factor_name in self.ic_results and "ic_series" in self.ic_results[factor_name]:
                ic_series = self.ic_results[factor_name]["ic_series"]
                rolling_ic = ic_series.rolling(6, min_periods=3).mean()
                ax.plot(rolling_ic.index, rolling_ic.values,
                        label=f"{label} (6M Rolling IC)", linewidth=1.2, color=color)
                plotted = True

        if not plotted:
            ax.text(0.5, 0.5, "No IC data for Vol/Volume",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)

        ax.axhline(y=0.05, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(y=-0.05, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_title("IC: Volatility & Volume Trend", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, color=COLORS["grid"])

    # ─── Row 3: PCA & Portfolio ──────────────────────────────────────

    def _plot_scree(self, ax: plt.Axes) -> None:
        """Plot PCA scree chart with cumulative variance line."""
        if self.pca_decomposer is None or not hasattr(self.pca_decomposer, "pca"):
            ax.text(0.5, 0.5, "No PCA data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title("PCA Scree Chart", fontsize=10, fontweight="bold")
            return

        pca = self.pca_decomposer.pca
        if pca is None:
            ax.text(0.5, 0.5, "PCA not fitted", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title("PCA Scree Chart", fontsize=10, fontweight="bold")
            return

        n_show = min(20, len(pca.explained_variance_ratio_))
        var_ratio = pca.explained_variance_ratio_[:n_show]
        cum_var = np.cumsum(var_ratio)

        ax.bar(range(1, n_show + 1), var_ratio,
               color=COLORS["portfolio"], alpha=0.6, label="Individual")
        ax2 = ax.twinx()
        ax2.plot(range(1, n_show + 1), cum_var,
                 color=COLORS["negative"], marker="o", markersize=3,
                 linewidth=1.2, label="Cumulative")
        ax2.axhline(y=0.80, color="#888888", linestyle="--", linewidth=0.8)
        ax2.set_ylabel("Cumulative Variance", fontsize=8)
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

        ax.set_title("PCA Scree Chart", fontsize=10, fontweight="bold")
        ax.set_xlabel("Principal Component", fontsize=8)
        ax.set_ylabel("Variance Explained", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")

    def _plot_cluster_scatter(self, ax: plt.Axes) -> None:
        """Plot K-Means cluster scatter on PC1 vs PC2."""
        if self.clusterer is None:
            ax.text(0.5, 0.5, "No cluster data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title("Market Clusters (PC1 vs PC2)", fontsize=10, fontweight="bold")
            return

        if not hasattr(self.clusterer, "pc_scores") or self.clusterer.pc_scores is None:
            ax.text(0.5, 0.5, "Cluster scores not computed", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title("Market Clusters (PC1 vs PC2)", fontsize=10, fontweight="bold")
            return

        scores = self.clusterer.pc_scores
        labels = self.clusterer.cluster_labels

        if labels is None:
            ax.text(0.5, 0.5, "Clusters not fitted", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title("Market Clusters (PC1 vs PC2)", fontsize=10, fontweight="bold")
            return

        scatter = ax.scatter(
            scores["PC1"], scores["PC2"],
            c=labels.reindex(scores.index).values,
            cmap="tab10", alpha=0.6, s=15, edgecolors="none",
        )
        ax.set_title("Market Clusters (PC1 vs PC2)", fontsize=10, fontweight="bold")
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.grid(True, alpha=0.3, color=COLORS["grid"])

    def _plot_portfolio_weights(self, ax: plt.Axes) -> None:
        """Plot top 20 portfolio weights as a horizontal bar chart."""
        if self.portfolio_weights is None or self.portfolio_weights.empty:
            ax.text(0.5, 0.5, "No weight data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title("Top 20 Portfolio Weights", fontsize=10, fontweight="bold")
            return

        top_20 = self.portfolio_weights.nlargest(20).sort_values()
        ax.barh(
            range(len(top_20)), top_20.values,
            color=COLORS["portfolio"], alpha=0.7,
        )
        ax.set_yticks(range(len(top_20)))
        ax.set_yticklabels(
            [t.replace(".NS", "") for t in top_20.index], fontsize=6
        )
        ax.set_title("Top 20 Portfolio Weights", fontsize=10, fontweight="bold")
        ax.set_xlabel("Weight", fontsize=8)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(True, alpha=0.3, color=COLORS["grid"], axis="x")

    # ─── Performance Table ───────────────────────────────────────────

    def _add_performance_table(self, fig: plt.Figure) -> None:
        """Add a text-based performance summary table to the figure."""
        if not self.metrics:
            return

        lines = [
            "─── Performance Summary ───",
            f"Ann. Return:  {self.metrics.get('annualised_return', 0) * 100:>8.2f}%",
            f"Ann. Vol:     {self.metrics.get('annualised_volatility', 0) * 100:>8.2f}%",
            f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):>8.2f}",
            f"Max Drawdown: {self.metrics.get('max_drawdown', 0) * 100:>8.2f}%",
            f"Calmar Ratio: {self.metrics.get('calmar_ratio', 0):>8.2f}",
            f"Win Rate:     {self.metrics.get('win_rate', 0) * 100:>8.1f}%",
        ]

        # Add benchmark-relative if available
        if "information_ratio" in self.metrics:
            lines.extend([
                f"Info Ratio:   {self.metrics.get('information_ratio', 0):>8.2f}",
                f"Alpha (ann):  {self.metrics.get('alpha', 0) * 100:>8.2f}%",
                f"Beta:         {self.metrics.get('beta', 0):>8.2f}",
            ])

        text = "\n".join(lines)

        fig.text(
            0.01, 0.01, text,
            fontsize=7, fontfamily="monospace",
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9),
        )
