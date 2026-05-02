"""
EigenAlpha — Information Coefficient Analysis
==============================================

The ``InformationCoefficient`` class implements the standard suite of
factor evaluation metrics used in professional quantitative research:

- **IC (Information Coefficient)**: Spearman rank correlation between
  factor values at time *t* and forward returns over the subsequent
  period.  A useful factor typically has |IC| > 0.05.
- **IR (Information Ratio)**: IC_mean / IC_std — measures the consistency
  of the factor's predictive signal.  IR > 0.5 is considered strong.
- **IC Decay**: IC computed at increasing forward lags to measure how
  long the signal persists before decaying to noise.

Academic references:
    Grinold, R. C., & Kahn, R. N. (2000). *Active Portfolio Management*,
    2nd ed. McGraw-Hill. — The foundational text on IC/IR framework.

    Qian, E., Hua, R., & Sorensen, E. (2007). *Quantitative Equity
    Portfolio Management*. Chapman & Hall/CRC. — Chapter 4 on factor
    evaluation.

Usage:
    from factors.ic_analysis import InformationCoefficient
    ic = InformationCoefficient(factor_data, forward_returns)
    summary = ic.ic_summary("momentum_12_1")
    print(f"Mean IC: {summary['mean_ic']:.4f}, IR: {summary['ir']:.4f}")
"""

import logging
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class InformationCoefficient:
    """Evaluate factor quality using Information Coefficient analysis.

    The IC framework is the industry standard for assessing whether a
    factor has predictive power for future returns.

    Attributes:
        factor_data (pd.DataFrame): Long-format factor data with columns
            ``[date, ticker, <factor_columns>]``.
        forward_returns (pd.DataFrame): Long-format forward returns with
            columns ``[date, ticker, forward_1m]``.
    """

    def __init__(
        self,
        factor_data: pd.DataFrame,
        forward_returns: pd.DataFrame,
    ) -> None:
        """Initialise the InformationCoefficient analyser.

        Args:
            factor_data: Long-format DataFrame with columns
                ``[date, ticker, <factor_cols>]``.
            forward_returns: Long-format DataFrame with columns
                ``[date, ticker, forward_1m]`` where ``forward_1m`` is the
                simple return over the next month.

        Raises:
            ValueError: If required columns are missing or DataFrames are empty.
        """
        if factor_data.empty:
            raise ValueError("factor_data must not be empty.")
        if forward_returns.empty:
            raise ValueError("forward_returns must not be empty.")

        required_cols = {"date", "ticker"}
        if not required_cols.issubset(factor_data.columns):
            raise ValueError(
                f"factor_data must contain columns {required_cols}. "
                f"Got: {set(factor_data.columns)}"
            )
        if not required_cols.issubset(forward_returns.columns):
            raise ValueError(
                f"forward_returns must contain columns {required_cols}. "
                f"Got: {set(forward_returns.columns)}"
            )

        # Merge factor data with forward returns
        self.merged = factor_data.merge(
            forward_returns, on=["date", "ticker"], how="inner"
        )

        self.factor_cols = [
            c for c in factor_data.columns if c not in ("date", "ticker")
        ]

        # Auto-detect the forward return column name
        fwd_candidates = [
            c for c in forward_returns.columns
            if c not in ("date", "ticker")
        ]
        if not fwd_candidates:
            raise ValueError(
                "forward_returns must contain at least one return column "
                "beyond 'date' and 'ticker'."
            )
        self.return_col = fwd_candidates[0]

        logger.info(
            "IC analyser initialised: %d observations, %d dates, factors: %s",
            len(self.merged),
            self.merged["date"].nunique(),
            self.factor_cols,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Core IC Computation
    # ──────────────────────────────────────────────────────────────────────

    def compute_ic(self, factor_col: str) -> pd.Series:
        """Compute time series of Spearman rank IC values.

        For each date *t*, computes the Spearman rank correlation between
        the cross-section of factor values and forward 1-month returns.

        Academic reference:
            Grinold, R. C., & Kahn, R. N. (2000). Active Portfolio
            Management, 2nd ed. — Definition of IC as rank correlation.

        Args:
            factor_col: Name of the factor column to evaluate.

        Returns:
            pd.Series: IC values indexed by date (one per month).

        Raises:
            ValueError: If ``factor_col`` is not in the merged data.
        """
        if factor_col not in self.merged.columns:
            raise ValueError(
                f"Factor column '{factor_col}' not found. "
                f"Available: {self.factor_cols}"
            )

        def _spearman_ic(group: pd.DataFrame) -> float:
            """Compute Spearman rank correlation for a single cross-section."""
            x = group[factor_col].values
            y = group[self.return_col].values

            # Need at least 5 observations for meaningful correlation
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 5:
                return np.nan

            # Academic reference: Spearman rank correlation
            corr, _ = stats.spearmanr(x[mask], y[mask])
            return corr

        ic_series = self.merged.groupby("date").apply(_spearman_ic)
        ic_series.name = f"IC_{factor_col}"
        ic_series = ic_series.dropna()

        logger.info(
            "IC(%s): %d periods, mean=%.4f, std=%.4f",
            factor_col,
            len(ic_series),
            ic_series.mean(),
            ic_series.std(),
        )

        return ic_series

    # ──────────────────────────────────────────────────────────────────────
    # Summary Statistics
    # ──────────────────────────────────────────────────────────────────────

    def ic_summary(self, factor_col: str) -> Dict[str, float]:
        """Compute summary statistics for the IC time series.

        Args:
            factor_col: Name of the factor column to evaluate.

        Returns:
            dict: Dictionary with keys:
                - ``mean_ic`` (float): Mean IC across all periods.
                - ``ic_std`` (float): Standard deviation of IC.
                - ``ir`` (float): Information Ratio = mean_ic / ic_std.
                - ``ic_t_stat`` (float): t-statistic = mean / (std / √n).
                - ``pct_positive`` (float): Percentage of months with IC > 0.

        Raises:
            ValueError: If ``factor_col`` is not a valid factor.
        """
        ic = self.compute_ic(factor_col)
        n = len(ic)

        mean_ic = ic.mean()
        ic_std = ic.std()

        # Academic reference: Grinold & Kahn (2000) — IR = IC / σ(IC)
        ir = mean_ic / ic_std if ic_std > 0 else 0.0

        # t-statistic: test H0: mean_ic = 0
        ic_t_stat = mean_ic / (ic_std / np.sqrt(n)) if ic_std > 0 else 0.0

        pct_positive = (ic > 0).mean()

        summary = {
            "mean_ic": mean_ic,
            "ic_std": ic_std,
            "ir": ir,
            "ic_t_stat": ic_t_stat,
            "pct_positive": pct_positive,
        }

        logger.info(
            "IC Summary (%s): mean=%.4f, std=%.4f, IR=%.3f, t=%.2f, %%pos=%.1f%%",
            factor_col,
            mean_ic,
            ic_std,
            ir,
            ic_t_stat,
            pct_positive,
        )

        return summary

    # ──────────────────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────────────────

    def plot_ic_timeseries(
        self,
        factor_col: str,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Plot IC values over time as a bar chart with rolling mean.

        Args:
            factor_col: Name of the factor column to plot.
            ax: Optional matplotlib Axes.  If ``None``, creates a new figure.

        Returns:
            matplotlib.axes.Axes: The plot axes.
        """
        ic = self.compute_ic(factor_col)
        summary = self.ic_summary(factor_col)

        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 5))

        # Bar chart coloured green (positive) / red (negative)
        colours = ["#2ecc71" if v > 0 else "#e74c3c" for v in ic.values]
        ax.bar(ic.index, ic.values, color=colours, alpha=0.7, width=25)

        # 6-month rolling mean IC
        rolling_ic = ic.rolling(window=6, min_periods=3).mean()
        ax.plot(
            rolling_ic.index,
            rolling_ic.values,
            color="#2c3e50",
            linewidth=2,
            label="6M Rolling Mean IC",
        )

        # Rule-of-thumb threshold lines at IC = ±0.05
        ax.axhline(y=0.05, color="#3498db", linestyle="--", alpha=0.6, label="IC = ±0.05")
        ax.axhline(y=-0.05, color="#3498db", linestyle="--", alpha=0.6)
        ax.axhline(y=0, color="black", linewidth=0.5)

        # Annotations
        ax.set_title(
            f"IC Time Series — {factor_col}\n"
            f"Mean IC={summary['mean_ic']:.4f}  IR={summary['ir']:.3f}  "
            f"t-stat={summary['ic_t_stat']:.2f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Information Coefficient (Spearman)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        return ax

    # ──────────────────────────────────────────────────────────────────────
    # IC Decay
    # ──────────────────────────────────────────────────────────────────────

    def ic_decay(self, factor_col: str, max_lag: int = 6) -> pd.Series:
        """Compute IC at increasing forward lags to measure signal persistence.

        For each lag k (1 to ``max_lag``), compute the Spearman rank
        correlation between the factor at time *t* and the return from
        *t+k-1* to *t+k* months.

        A factor with slow IC decay retains predictive power over longer
        horizons, making it more useful for lower-frequency strategies.

        Args:
            factor_col: Name of the factor column.
            max_lag: Maximum forward lag in months.  Defaults to 6.

        Returns:
            pd.Series: IC values indexed by lag (1 to max_lag).
        """
        if factor_col not in self.merged.columns:
            raise ValueError(f"Factor column '{factor_col}' not found.")

        decay = {}

        # Pivot returns to wide format for easy lag computation
        dates = sorted(self.merged["date"].unique())

        for lag in range(1, max_lag + 1):
            ic_values = []

            for i, date in enumerate(dates):
                if i + lag >= len(dates):
                    break

                future_date = dates[i + lag]

                # Current factor values
                current = self.merged[self.merged["date"] == date][
                    ["ticker", factor_col]
                ].set_index("ticker")

                # Future return
                future = self.merged[self.merged["date"] == future_date][
                    ["ticker", self.return_col]
                ].set_index("ticker")

                # Align
                aligned = current.join(future, how="inner")
                if len(aligned) < 5:
                    continue

                x = aligned[factor_col].values
                y = aligned[self.return_col].values
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() < 5:
                    continue

                corr, _ = stats.spearmanr(x[mask], y[mask])
                ic_values.append(corr)

            decay[lag] = np.mean(ic_values) if ic_values else np.nan

        decay_series = pd.Series(decay, name=f"IC_decay_{factor_col}")
        decay_series.index.name = "lag_months"

        logger.info(
            "IC Decay (%s): %s",
            factor_col,
            {k: f"{v:.4f}" for k, v in decay.items()},
        )

        return decay_series
