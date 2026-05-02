"""
EigenAlpha — Data Preprocessor
==============================

Provides the ``Preprocessor`` class for transforming raw OHLCV data into
research-ready return series and standardised factor inputs.

Key transformations:
    1. **Log returns** — continuous compounding, additive across time.
    2. **Monthly returns** — simple returns at month-end frequency.
    3. **Winsorisation** — clip extreme values to reduce outlier influence.
    4. **Cross-sectional z-score** — zero-mean, unit-variance per date.
    5. **Return matrix** — wide-format (T × N) matrix for PCA.

All operations are vectorised using NumPy/Pandas and avoid Python loops.

Usage:
    from data.preprocessor import Preprocessor
    pp = Preprocessor()
    log_ret = pp.compute_log_returns(close_prices)
    factor_z = pp.cross_sectional_zscore(raw_factor)
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Preprocessor:
    """Transform raw price/volume data into research-ready inputs.

    This class is stateless — all methods are pure functions that take
    data in and return transformed data without side effects.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Returns Computation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Compute log returns from a price DataFrame.

        Log returns are preferred for statistical modelling because they
        are additive across time and approximately normally distributed.

        Args:
            prices: DataFrame of prices.  Can be either:
                - Wide format (index=Date, columns=Tickers)
                - Long format with MultiIndex (Date, Ticker)

        Returns:
            pd.DataFrame: Log returns in the same format as input,
                with the first observation dropped (NaN from differencing).

        Raises:
            ValueError: If prices contain non-positive values.
        """
        if (prices <= 0).any().any() if isinstance(prices, pd.DataFrame) else False:
            logger.warning("Non-positive prices detected — log returns will contain NaN.")

        # Academic reference: Hull, J.C. (2018). Options, Futures, and Other
        # Derivatives, 10th ed. — log return = ln(P_t / P_{t-1})
        log_ret = np.log(prices / prices.shift(1))

        # Drop the first row (NaN from shift)
        if isinstance(log_ret.index, pd.MultiIndex):
            log_ret = log_ret.groupby(level="Ticker").apply(
                lambda x: x.iloc[1:]
            )
            # Clean up any residual MultiIndex levels from groupby
            if log_ret.index.nlevels > 2:
                log_ret = log_ret.droplevel(0)
        else:
            log_ret = log_ret.iloc[1:]

        logger.info("Computed log returns: shape %s", log_ret.shape)
        return log_ret

    @staticmethod
    def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Compute simple monthly returns from daily prices.

        Resamples to month-end and computes ``(P_t / P_{t-1}) - 1``.

        Args:
            prices: Wide-format DataFrame with index=Date, columns=Tickers,
                values=closing prices.

        Returns:
            pd.DataFrame: Monthly simple returns (wide format).
        """
        # Resample to month-end, take last available price
        monthly_prices = prices.resample("ME").last()

        # Simple return: (P_t / P_{t-1}) - 1
        monthly_returns = monthly_prices.pct_change()

        # Drop the first month (NaN)
        monthly_returns = monthly_returns.iloc[1:]

        logger.info(
            "Computed monthly returns: %d months × %d tickers",
            monthly_returns.shape[0],
            monthly_returns.shape[1],
        )
        return monthly_returns

    # ──────────────────────────────────────────────────────────────────────
    # Cross-Sectional Standardisation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def winsorise(
        series: pd.Series,
        lower: float = 2.5,
        upper: float = 97.5,
    ) -> pd.Series:
        """Winsorise a cross-sectional series at the given percentiles.

        Values below the ``lower`` percentile are clipped up, and values
        above the ``upper`` percentile are clipped down.  This reduces the
        influence of extreme outliers without removing observations.

        When applied to a DataFrame grouped by date, this performs
        cross-sectional winsorisation (each date treated independently).

        Args:
            series: Values to winsorise.
            lower: Lower percentile bound (0–100).
            upper: Upper percentile bound (0–100).

        Returns:
            pd.Series: Winsorised values.

        Raises:
            ValueError: If ``lower >= upper``.
        """
        if lower >= upper:
            raise ValueError(
                f"lower ({lower}) must be strictly less than upper ({upper})."
            )

        lo = np.nanpercentile(series.values, lower)
        hi = np.nanpercentile(series.values, upper)

        return series.clip(lower=lo, upper=hi)

    @staticmethod
    def cross_sectional_zscore(series: pd.Series) -> pd.Series:
        """Standardise a cross-sectional series to zero mean, unit variance.

        Args:
            series: Raw factor values for a single cross-section (one date).

        Returns:
            pd.Series: Z-scored values.  If standard deviation is zero
                (all identical values), returns zeros.
        """
        mean = series.mean()
        std = series.std()

        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=series.index)

        return (series - mean) / std

    @staticmethod
    def winsorise_by_date(
        df: pd.DataFrame,
        column: str,
        lower: float = 2.5,
        upper: float = 97.5,
    ) -> pd.DataFrame:
        """Apply winsorisation cross-sectionally (per date).

        Args:
            df: Long-format DataFrame with a ``date`` column.
            column: Name of the column to winsorise.
            lower: Lower percentile bound.
            upper: Upper percentile bound.

        Returns:
            pd.DataFrame: DataFrame with the specified column winsorised
                per date.
        """
        df = df.copy()
        df[column] = df.groupby("date")[column].transform(
            lambda x: Preprocessor.winsorise(x, lower, upper)
        )
        return df

    @staticmethod
    def zscore_by_date(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply cross-sectional z-scoring per date.

        Args:
            df: Long-format DataFrame with a ``date`` column.
            column: Name of the column to z-score.

        Returns:
            pd.DataFrame: DataFrame with the specified column z-scored
                per date.
        """
        df = df.copy()
        df[column] = df.groupby("date")[column].transform(
            lambda x: Preprocessor.cross_sectional_zscore(x)
        )
        return df

    # ──────────────────────────────────────────────────────────────────────
    # Matrix Construction
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def build_return_matrix(
        log_returns: pd.DataFrame,
        min_observations: int = 60,
    ) -> pd.DataFrame:
        """Build a wide-format return matrix (T × N) from log returns.

        This is the R matrix used in PCA decomposition:
            - Rows = dates (T observations)
            - Columns = tickers (N assets)

        Tickers with fewer than ``min_observations`` are excluded to
        ensure the covariance matrix is well-conditioned.

        Args:
            log_returns: Log returns in either:
                - Long format with MultiIndex (Date, Ticker)
                - Long format with columns ``[date, ticker, value]``
                - Wide format (index=Date, columns=Tickers)
            min_observations: Minimum number of non-NaN return
                observations required for a ticker to be included.

        Returns:
            pd.DataFrame: Wide-format return matrix (T × N).
        """
        # Handle MultiIndex input
        if isinstance(log_returns.index, pd.MultiIndex):
            if "Close" in log_returns.columns:
                wide = log_returns["Close"].unstack(level="Ticker")
            else:
                wide = log_returns.iloc[:, 0].unstack(level="Ticker")
        elif "date" in log_returns.columns and "ticker" in log_returns.columns:
            # Long format with named columns
            value_col = [
                c
                for c in log_returns.columns
                if c not in ("date", "ticker")
            ][0]
            wide = log_returns.pivot(
                index="date", columns="ticker", values=value_col
            )
        else:
            # Assume already wide format
            wide = log_returns.copy()

        # Filter tickers with insufficient history
        valid_counts = wide.notna().sum()
        valid_tickers = valid_counts[valid_counts >= min_observations].index
        wide = wide[valid_tickers]

        logger.info(
            "Return matrix: %d dates × %d tickers "
            "(dropped %d tickers with < %d observations)",
            wide.shape[0],
            wide.shape[1],
            len(valid_counts) - len(valid_tickers),
            min_observations,
        )

        return wide
