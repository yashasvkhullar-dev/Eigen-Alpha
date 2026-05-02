"""
Backtesting Engine for Factor-Based Portfolio Strategies
=========================================================

This module implements a professional-grade backtesting framework for evaluating
factor-based portfolio strategies on Indian equities. It supports both quintile
long-short backtests (the standard academic approach) and walk-forward Markowitz
optimised backtests.

Key Design Principles:
    - **No look-ahead bias**: All portfolio construction decisions use only data
      available at the time of rebalancing. The walk-forward methodology ensures
      that optimisation parameters are estimated from historical data only.
    - **Transaction cost awareness**: While this Phase 0 implementation does not
      deduct explicit transaction costs, turnover metrics are tracked to enable
      future integration of realistic cost models (brokerage + STT + SEBI charges
      for Indian equities).
    - **Monthly rebalancing**: Aligns with the monthly frequency of factor signals,
      which is the standard cadence in academic factor research.

Academic References:
    - Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on
      stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56.
    - DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive
      Diversification. *Review of Financial Studies*, 22(5), 1915–1953.

Usage:
    >>> from portfolio.backtest import Backtester
    >>> bt = Backtester(prices, factor_data, benchmark)
    >>> quintile_returns = bt.run_quintile_backtest('momentum_12_1')
    >>> metrics = bt.compute_metrics(quintile_returns['Q5'])

Author: EigenAlpha Research
"""

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config import RISK_FREE_RATE, REBALANCE_FREQ, TOP_QUINTILE, BOTTOM_QUINTILE

logger = logging.getLogger(__name__)


class Backtester:
    """Monthly-rebalance backtesting engine for factor and optimised portfolios.

    This class implements two complementary backtesting methodologies:

    1. **Quintile Backtest**: The standard academic approach. Each month, stocks
       are sorted into quintiles based on factor scores. Returns are computed for
       equal-weighted portfolios within each quintile, plus a long-short spread
       (Q5 minus Q1). This tests whether a factor has monotonic return predictability.

    2. **Markowitz Backtest**: A walk-forward optimised portfolio that uses the
       MarkowitzOptimizer to construct minimum-variance portfolios tilted toward
       high factor-score clusters. This tests whether the PCA + clustering +
       optimisation pipeline adds value over naive quintile sorting.

    Attributes:
        prices (pd.DataFrame): Wide-format price matrix (dates × tickers).
        factor_data (pd.DataFrame): Long-format factor data with columns
            [date, ticker, momentum_12_1, realized_vol, volume_trend].
        benchmark (pd.Series): Benchmark return series (e.g., Nifty 50).
        monthly_prices (pd.DataFrame): Month-end resampled price matrix.
        monthly_returns (pd.DataFrame): Monthly simple return matrix.

    Args:
        prices: Wide-format DataFrame of adjusted close prices.
            Index = DatetimeIndex, columns = ticker symbols.
        factor_data: Long-format DataFrame containing computed factor scores.
            Must contain columns ['date', 'ticker'] plus factor columns.
        benchmark: Series of benchmark prices or returns.
            Index = DatetimeIndex.

    Raises:
        ValueError: If prices DataFrame is empty or has fewer than 2 dates.
        ValueError: If factor_data does not contain required columns.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        factor_data: pd.DataFrame,
        benchmark: pd.Series,
    ) -> None:
        # --- Input validation ---
        if prices.empty:
            raise ValueError("prices DataFrame must not be empty.")
        if len(prices) < 2:
            raise ValueError(
                "prices DataFrame must contain at least 2 dates to compute returns."
            )
        required_cols = {"date", "ticker"}
        missing = required_cols - set(factor_data.columns)
        if missing:
            raise ValueError(
                f"factor_data is missing required columns: {missing}"
            )

        self.prices = prices.copy()
        self.factor_data = factor_data.copy()
        self.benchmark = benchmark.copy()

        # Precompute month-end prices and returns
        self.monthly_prices = self.prices.resample("ME").last()
        self.monthly_returns = self.monthly_prices.pct_change().dropna(how="all")

        # Compute benchmark monthly returns
        if isinstance(self.benchmark, pd.Series):
            bench_monthly = self.benchmark.resample("ME").last()
            self.benchmark_monthly_returns = bench_monthly.pct_change().dropna()
        else:
            self.benchmark_monthly_returns = pd.Series(dtype=float)

        logger.info(
            "Backtester initialised: %d months, %d stocks",
            len(self.monthly_returns),
            len(self.monthly_returns.columns),
        )

    def run_quintile_backtest(
        self,
        factor_col: str,
        n_quantiles: int = 5,
        long_only: bool = False,
    ) -> pd.DataFrame:
        """Run a quintile-sorted long-short backtest for a given factor.

        Each month-end date:
        1. Rank all stocks by the factor score.
        2. Assign stocks to quintiles Q1 (bottom 20%) through Q5 (top 20%).
        3. Compute the equal-weighted return of each quintile in the following month.
        4. Compute the long-short spread: Q5 return minus Q1 return.

        This is the canonical test for whether a factor has monotonic return
        predictability across the cross-section, following the methodology of
        Fama & French (1993).

        Args:
            factor_col: Name of the factor column in self.factor_data to sort on.
                Must be one of the computed factor columns.
            n_quantiles: Number of quantile buckets. Default is 5 (quintiles).
            long_only: If True, only return Q5 (top quintile) portfolio returns.
                If False, also compute and return the long-short spread.

        Returns:
            pd.DataFrame: Monthly returns for each quintile and the spread.
                Columns are ['Q1', 'Q2', ..., 'Q5', 'Long_Short'] (or just
                ['Q5'] if long_only=True).
                Index = DatetimeIndex at month-end frequency.

        Raises:
            ValueError: If factor_col is not found in factor_data columns.
        """
        if factor_col not in self.factor_data.columns:
            raise ValueError(
                f"Factor column '{factor_col}' not found in factor_data. "
                f"Available columns: {list(self.factor_data.columns)}"
            )

        logger.info("Running quintile backtest for factor: %s", factor_col)

        # Pivot factor data to wide format: date × ticker
        factor_pivot = self.factor_data.pivot_table(
            index="date", columns="ticker", values=factor_col
        )

        # Align dates between factor scores and forward returns
        common_dates = factor_pivot.index.intersection(self.monthly_returns.index)
        # We need factor scores at date t to predict returns at date t+1
        # So the last factor date we can use is the second-to-last return date
        factor_dates = sorted(common_dates)

        quintile_returns = {f"Q{q}": [] for q in range(1, n_quantiles + 1)}
        return_dates = []

        for i in range(len(factor_dates) - 1):
            formation_date = factor_dates[i]
            holding_date = factor_dates[i + 1]

            # Get factor scores for this month (cross-section)
            scores = factor_pivot.loc[formation_date].dropna()
            if len(scores) < n_quantiles * 5:
                # Need at least 5 stocks per quintile for meaningful results
                continue

            # Get forward returns for the holding period
            if holding_date not in self.monthly_returns.index:
                continue
            fwd_returns = self.monthly_returns.loc[holding_date]

            # Only keep stocks that have both scores and returns
            common_tickers = scores.index.intersection(fwd_returns.dropna().index)
            if len(common_tickers) < n_quantiles * 5:
                continue

            scores = scores[common_tickers]
            fwd_returns = fwd_returns[common_tickers]

            # Assign quintiles using pd.qcut
            try:
                quintile_labels = pd.qcut(
                    scores, q=n_quantiles, labels=False, duplicates="drop"
                )
            except ValueError:
                # Too few unique values for requested quantiles
                continue

            # Compute equal-weighted return for each quintile
            for q in range(n_quantiles):
                mask = quintile_labels == q
                if mask.sum() > 0:
                    q_return = fwd_returns[mask[mask].index].mean()
                    quintile_returns[f"Q{q + 1}"].append(q_return)
                else:
                    quintile_returns[f"Q{q + 1}"].append(np.nan)

            return_dates.append(holding_date)

        # Assemble results
        result = pd.DataFrame(quintile_returns, index=return_dates)
        result.index.name = "date"

        # Add long-short spread: Q5 (top) minus Q1 (bottom)
        if not long_only:
            result["Long_Short"] = result[f"Q{n_quantiles}"] - result["Q1"]

        logger.info(
            "Quintile backtest complete: %d months, spread mean = %.4f",
            len(result),
            result.get("Long_Short", result[f"Q{n_quantiles}"]).mean(),
        )

        return result

    def run_markowitz_backtest(
        self,
        optimizer_class: type,
        factor_col: str = "momentum_12_1",
        lookback_months: int = 36,
        cluster_labels: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Run a walk-forward Markowitz-optimised backtest.

        Each month-end:
        1. Estimate covariance and expected returns from the trailing
           `lookback_months` of data (walk-forward, no future data).
        2. Run the MarkowitzOptimizer to construct optimal weights.
        3. Apply those weights to the following month's realised returns.

        This tests whether the full PCA + clustering + optimisation pipeline
        generates superior risk-adjusted returns compared to the naive quintile
        approach.

        Args:
            optimizer_class: The MarkowitzOptimizer class to instantiate each month.
            factor_col: Factor column name used for cluster tilting.
            lookback_months: Number of trailing months of data to use for
                covariance estimation. Default is 36 (3 years).
            cluster_labels: Pre-computed cluster assignments. If None, a simple
                equal-weight optimisation is performed without cluster tilting.

        Returns:
            pd.Series: Monthly portfolio returns from the walk-forward backtest.
                Index = DatetimeIndex at month-end frequency.

        Raises:
            ValueError: If lookback_months is less than 12.
        """
        if lookback_months < 12:
            raise ValueError(
                f"lookback_months must be >= 12, got {lookback_months}. "
                "Covariance estimation requires sufficient history."
            )

        logger.info(
            "Running Markowitz walk-forward backtest (lookback=%d months)",
            lookback_months,
        )

        dates = self.monthly_returns.index
        portfolio_returns = []
        return_dates = []

        # Factor scores pivoted to wide format
        factor_pivot = self.factor_data.pivot_table(
            index="date", columns="ticker", values=factor_col
        )

        for i in range(lookback_months, len(dates) - 1):
            formation_date = dates[i]
            holding_date = dates[i + 1]

            # Walk-forward: use only trailing data for estimation
            lookback_start = dates[max(0, i - lookback_months)]
            hist_returns = self.monthly_returns.loc[lookback_start:formation_date]

            # Remove stocks with too many missing values in the lookback window
            valid_stocks = hist_returns.columns[
                hist_returns.notna().sum() >= lookback_months * 0.7
            ]
            if len(valid_stocks) < 20:
                continue

            hist_returns = hist_returns[valid_stocks].fillna(0)

            # Get factor scores at formation date
            if formation_date in factor_pivot.index:
                scores = factor_pivot.loc[formation_date].reindex(valid_stocks)
            else:
                # Use nearest available factor date
                nearest_idx = factor_pivot.index.get_indexer(
                    [formation_date], method="ffill"
                )
                if nearest_idx[0] >= 0:
                    scores = factor_pivot.iloc[nearest_idx[0]].reindex(valid_stocks)
                else:
                    scores = pd.Series(0.0, index=valid_stocks)

            scores = scores.fillna(0)

            # Get cluster labels for valid stocks
            if cluster_labels is not None:
                cl = cluster_labels.reindex(valid_stocks).dropna().astype(int)
            else:
                # Assign all stocks to a single cluster
                cl = pd.Series(0, index=valid_stocks)

            try:
                # Instantiate optimizer with walk-forward data
                opt = optimizer_class(
                    returns=hist_returns,
                    factor_scores=scores,
                    cluster_labels=cl,
                )
                weights = opt.optimize_all_clusters()

                # Apply weights to next month's realised returns
                fwd = self.monthly_returns.loc[holding_date].reindex(
                    weights.index
                ).fillna(0)
                port_ret = (weights * fwd).sum()

                portfolio_returns.append(port_ret)
                return_dates.append(holding_date)

            except Exception as e:
                logger.warning(
                    "Optimisation failed at %s: %s", formation_date, str(e)
                )
                continue

        result = pd.Series(portfolio_returns, index=return_dates, name="markowitz")
        result.index.name = "date"

        logger.info(
            "Markowitz backtest complete: %d months, mean return = %.4f",
            len(result),
            result.mean() if len(result) > 0 else 0.0,
        )

        return result

    def compute_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Compute a comprehensive suite of performance metrics.

        Calculates the standard set of risk-adjusted performance metrics
        reported in professional factor research and fund tearsheets.

        Args:
            returns: Monthly return series to evaluate.
                Index = DatetimeIndex, values = monthly simple returns.

        Returns:
            dict: Performance metrics with keys:
                - 'annualised_return': Geometrically compounded annual return.
                - 'annualised_volatility': Annualised standard deviation.
                - 'sharpe_ratio': (Ann. return - risk-free rate) / ann. volatility.
                - 'max_drawdown': Maximum peak-to-trough decline.
                - 'calmar_ratio': Ann. return / |max drawdown|.
                - 'win_rate': Fraction of months with positive returns.
                - 'avg_monthly_return': Arithmetic mean of monthly returns.
                - 'best_month': Return of the best single month.
                - 'worst_month': Return of the worst single month.
                - 'n_months': Total number of months.

        Raises:
            ValueError: If returns series is empty.
        """
        if returns.empty:
            raise ValueError("Cannot compute metrics for empty return series.")

        returns = returns.dropna()
        n_months = len(returns)

        if n_months == 0:
            raise ValueError("Return series contains only NaN values.")

        # Annualised return (geometric compounding)
        cumulative = (1 + returns).prod()
        n_years = n_months / 12.0
        annualised_return = cumulative ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

        # Annualised volatility
        annualised_volatility = returns.std() * np.sqrt(12)

        # Sharpe ratio
        # Academic reference: Sharpe, W. F. (1966). Mutual Fund Performance.
        # *Journal of Business*, 39(1), 119–138.
        if annualised_volatility > 0:
            sharpe_ratio = (
                (annualised_return - RISK_FREE_RATE) / annualised_volatility
            )
        else:
            sharpe_ratio = 0.0

        # Maximum drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        if abs(max_drawdown) > 0:
            calmar_ratio = annualised_return / abs(max_drawdown)
        else:
            calmar_ratio = np.inf if annualised_return > 0 else 0.0

        # Win rate
        win_rate = (returns > 0).sum() / n_months

        metrics = {
            "annualised_return": round(annualised_return, 6),
            "annualised_volatility": round(annualised_volatility, 6),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "max_drawdown": round(max_drawdown, 6),
            "calmar_ratio": round(calmar_ratio, 4),
            "win_rate": round(win_rate, 4),
            "avg_monthly_return": round(returns.mean(), 6),
            "best_month": round(returns.max(), 6),
            "worst_month": round(returns.min(), 6),
            "n_months": n_months,
        }

        logger.info(
            "Metrics: Sharpe=%.2f, MaxDD=%.1f%%, AnnRet=%.1f%%",
            metrics["sharpe_ratio"],
            metrics["max_drawdown"] * 100,
            metrics["annualised_return"] * 100,
        )

        return metrics

    def compute_benchmark_relative_metrics(
        self, portfolio_returns: pd.Series
    ) -> Dict[str, float]:
        """Compute benchmark-relative (active) performance metrics.

        Args:
            portfolio_returns: Monthly portfolio return series.

        Returns:
            dict: Active metrics with keys:
                - 'active_return': Annualised portfolio return minus benchmark return.
                - 'tracking_error': Annualised std of active returns.
                - 'information_ratio': Active return / tracking error.
                - 'beta': Portfolio beta to benchmark.
                - 'alpha': Jensen's alpha (annualised).

        Raises:
            ValueError: If benchmark returns are not available.
        """
        if self.benchmark_monthly_returns.empty:
            raise ValueError("Benchmark returns not available.")

        # Align dates
        common_idx = portfolio_returns.index.intersection(
            self.benchmark_monthly_returns.index
        )
        if len(common_idx) < 12:
            logger.warning(
                "Only %d common months between portfolio and benchmark.",
                len(common_idx),
            )

        port = portfolio_returns.loc[common_idx]
        bench = self.benchmark_monthly_returns.loc[common_idx]

        # Active returns
        active = port - bench

        active_return_ann = active.mean() * 12
        tracking_error = active.std() * np.sqrt(12)

        if tracking_error > 0:
            information_ratio = active_return_ann / tracking_error
        else:
            information_ratio = 0.0

        # Beta and Alpha (CAPM)
        # Academic reference: Jensen, M. C. (1968). The performance of mutual
        # funds in the period 1945–1964. *Journal of Finance*, 23(2), 389–416.
        cov_matrix = np.cov(port.values, bench.values)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 1.0
        alpha_monthly = port.mean() - beta * bench.mean()
        alpha_ann = alpha_monthly * 12

        return {
            "active_return": round(active_return_ann, 6),
            "tracking_error": round(tracking_error, 6),
            "information_ratio": round(information_ratio, 4),
            "beta": round(beta, 4),
            "alpha": round(alpha_ann, 6),
        }

    def compute_turnover(
        self, factor_col: str, n_quantiles: int = 5
    ) -> pd.Series:
        """Compute monthly portfolio turnover for the top quintile.

        Turnover measures what fraction of the top-quintile portfolio changes
        each month. High turnover implies high transaction costs, which erode
        real-world alpha.

        Args:
            factor_col: Factor column to sort on.
            n_quantiles: Number of quantile buckets.

        Returns:
            pd.Series: Monthly turnover rates (0 to 1).
                Index = DatetimeIndex.
        """
        factor_pivot = self.factor_data.pivot_table(
            index="date", columns="ticker", values=factor_col
        )

        dates = sorted(factor_pivot.index)
        turnover_values = []
        turnover_dates = []
        prev_top_quintile = set()

        for date in dates:
            scores = factor_pivot.loc[date].dropna()
            if len(scores) < n_quantiles * 5:
                continue

            try:
                labels = pd.qcut(scores, q=n_quantiles, labels=False, duplicates="drop")
            except ValueError:
                continue

            top_quintile = set(labels[labels == n_quantiles - 1].index)

            if prev_top_quintile:
                # Turnover = fraction of new stocks entering the top quintile
                if len(top_quintile) > 0:
                    new_entries = top_quintile - prev_top_quintile
                    turnover = len(new_entries) / len(top_quintile)
                else:
                    turnover = 0.0
                turnover_values.append(turnover)
                turnover_dates.append(date)

            prev_top_quintile = top_quintile

        return pd.Series(turnover_values, index=turnover_dates, name="turnover")
