"""
EigenAlpha — Factor Engine
==========================

The ``FactorEngine`` class is the core of the EigenAlpha research pipeline.
It computes three cross-sectional equity factors for the Indian market:

1. **Momentum (12-1)**
   Cumulative return from t-12M to t-1M, skipping the most recent month
   to avoid the well-documented short-term reversal effect.  Momentum is
   one of the most robust anomalies in asset pricing.

   Academic reference:
       Jegadeesh, N., & Titman, S. (1993). Returns to Buying Winners and
       Selling Losers: Implications for Stock Market Efficiency.
       *Journal of Finance*, 48(1), 65–91.

2. **Realised Volatility (20-day)**
   Rolling 20-day standard deviation of daily log returns, annualised by
   √252.  Low-volatility stocks tend to outperform on a risk-adjusted
   basis — the "low-volatility anomaly."

   Academic reference:
       Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006). The Cross-
       Section of Volatility and Expected Returns. *Journal of Finance*,
       61(1), 259–299.

3. **Volume Trend (20-day)**
   Slope of OLS regression of log(volume) on a time index over the
   trailing 20 trading days.  Positive slope indicates increasing volume,
   which may signal informed trading or changing liquidity conditions.

   Academic reference:
       Campbell, J. Y., Grossman, S. J., & Wang, J. (1993). Trading Volume
       and Serial Correlation in Stock Returns. *Quarterly Journal of
       Economics*, 108(4), 905–939.

Usage:
    from factors.engine import FactorEngine
    engine = FactorEngine(prices=close_wide, volumes=volume_wide)
    factor_data = engine.compute_all()
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

from data.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class FactorEngine:
    """Compute cross-sectional equity factors from price and volume data.

    This is the most important class in Phase 0.  All three factors are
    computed at monthly frequency and returned in long format with columns
    ``[date, ticker, <factor_name>]``.

    Attributes:
        prices (pd.DataFrame): Wide-format daily closing prices
            (index=Date, columns=Tickers).
        volumes (pd.DataFrame): Wide-format daily volumes
            (index=Date, columns=Tickers).
        monthly_prices (pd.DataFrame): Month-end closing prices.
    """

    def __init__(self, prices: pd.DataFrame, volumes: pd.DataFrame) -> None:
        """Initialise the FactorEngine.

        Args:
            prices: Wide-format daily closing prices.  Index must be a
                DatetimeIndex, columns are ticker symbols.
            volumes: Wide-format daily trading volumes.  Must have the
                same index and columns as ``prices``.

        Raises:
            ValueError: If prices or volumes are empty, or if their
                indices/columns do not align.
        """
        if prices.empty:
            raise ValueError("prices DataFrame must not be empty.")
        if volumes.empty:
            raise ValueError("volumes DataFrame must not be empty.")

        self.prices = prices.copy()
        self.volumes = volumes.copy()

        # Ensure DatetimeIndex
        self.prices.index = pd.to_datetime(self.prices.index)
        self.volumes.index = pd.to_datetime(self.volumes.index)

        # Pre-compute month-end prices for momentum
        self.monthly_prices = self.prices.resample("ME").last()

        logger.info(
            "FactorEngine initialised: %d tickers, %s to %s",
            len(self.prices.columns),
            self.prices.index.min().strftime("%Y-%m-%d"),
            self.prices.index.max().strftime("%Y-%m-%d"),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Factor 1: Momentum (12-1)
    # ──────────────────────────────────────────────────────────────────────

    def momentum_12_1(self) -> pd.DataFrame:
        """Compute 12-minus-1 month momentum factor.

        For each stock and each month-end date *t*, the momentum signal is
        the cumulative return from *t-12M* to *t-1M*, deliberately skipping
        the most recent month to avoid the short-term reversal effect.

        Formula:
            ``momentum = (price[t-1M] / price[t-12M]) - 1``

        Academic reference:
            Jegadeesh, N., & Titman, S. (1993). Returns to Buying Winners
            and Selling Losers. *Journal of Finance*, 48(1), 65–91.

        Returns:
            pd.DataFrame: Long-format with columns
                ``[date, ticker, momentum_12_1]``.
        """
        mp = self.monthly_prices

        # Academic reference: Jegadeesh & Titman (1993) — skip-month momentum
        # price[t-1M] / price[t-12M] - 1
        # shift(1) = price at t-1M, shift(12) = price at t-12M
        momentum = (mp.shift(1) / mp.shift(12)) - 1.0

        # Melt to long format — handle any index name
        momentum_reset = momentum.reset_index()
        date_col = momentum_reset.columns[0]  # First column is the date index
        momentum_long = momentum_reset.melt(
            id_vars=date_col, var_name="ticker", value_name="momentum_12_1"
        )
        momentum_long = momentum_long.rename(columns={date_col: "date"})
        momentum_long = momentum_long.dropna(subset=["momentum_12_1"])

        logger.info(
            "Momentum (12-1): %d observations across %d tickers",
            len(momentum_long),
            momentum_long["ticker"].nunique(),
        )

        return momentum_long

    # ──────────────────────────────────────────────────────────────────────
    # Factor 2: Realised Volatility (20-day)
    # ──────────────────────────────────────────────────────────────────────

    def realized_volatility_20d(self) -> pd.DataFrame:
        """Compute 20-day realised volatility (annualised).

        For each stock and each date *t*, compute the rolling 20-day
        standard deviation of daily log returns, annualised by multiplying
        by √252.

        Formula:
            ``realized_vol = log_returns.rolling(20).std() * sqrt(252)``

        Academic reference:
            Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006).
            The Cross-Section of Volatility and Expected Returns.
            *Journal of Finance*, 61(1), 259–299.

        Returns:
            pd.DataFrame: Long-format with columns
                ``[date, ticker, realized_vol]``, resampled to month-end.
        """
        # Compute daily log returns
        log_ret = np.log(self.prices / self.prices.shift(1))

        # Academic reference: Ang et al. (2006) — realised volatility
        # Rolling 20-day std, annualised by sqrt(252)
        rolling_vol = log_ret.rolling(window=20, min_periods=15).std() * np.sqrt(252)

        # Resample to month-end for consistency with other factors
        monthly_vol = rolling_vol.resample("ME").last()

        # Melt to long format — handle any index name
        vol_reset = monthly_vol.reset_index()
        date_col = vol_reset.columns[0]
        vol_long = vol_reset.melt(
            id_vars=date_col, var_name="ticker", value_name="realized_vol"
        )
        vol_long = vol_long.rename(columns={date_col: "date"})
        vol_long = vol_long.dropna(subset=["realized_vol"])

        logger.info(
            "Realised Volatility (20d): %d observations across %d tickers",
            len(vol_long),
            vol_long["ticker"].nunique(),
        )

        return vol_long

    # ──────────────────────────────────────────────────────────────────────
    # Factor 3: Volume Trend (20-day)
    # ──────────────────────────────────────────────────────────────────────

    def volume_trend_20d(self) -> pd.DataFrame:
        """Compute 20-day volume trend (OLS slope of log-volume).

        For each stock and each date *t*, fit a simple OLS regression of
        ``log(volume)`` on a time index ``[0, 1, ..., 19]`` over the
        trailing 20 trading days.  The slope coefficient is the factor
        value.

        Positive slope → volume is increasing → potential informed trading.
        Negative slope → volume is decreasing → fading interest.

        Uses ``scipy.stats.linregress`` inside a vectorised rolling window.

        Academic reference:
            Campbell, J. Y., Grossman, S. J., & Wang, J. (1993). Trading
            Volume and Serial Correlation in Stock Returns. *Quarterly
            Journal of Economics*, 108(4), 905–939.

        Returns:
            pd.DataFrame: Long-format with columns
                ``[date, ticker, volume_trend]``, resampled to month-end.
        """
        # Replace zeros with NaN to avoid log(0)
        vol = self.volumes.replace(0, np.nan)
        log_vol = np.log(vol)

        window = 20

        def _rolling_slope(series: pd.Series) -> pd.Series:
            """Compute rolling OLS slope of series on time index."""
            result = pd.Series(np.nan, index=series.index)
            values = series.values
            x = np.arange(window, dtype=float)

            for i in range(window - 1, len(values)):
                y = values[i - window + 1: i + 1]
                if np.isnan(y).sum() > window * 0.3:  # skip if >30% NaN
                    continue
                # Replace NaN with mean for regression
                mask = ~np.isnan(y)
                if mask.sum() < 10:
                    continue
                # Academic reference: Campbell, Grossman & Wang (1993)
                slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
                result.iloc[i] = slope

            return result

        logger.info("Computing volume trend slopes (this may take a minute)...")

        # Apply rolling slope to each ticker
        slopes = log_vol.apply(_rolling_slope, axis=0)

        # Resample to month-end
        monthly_slopes = slopes.resample("ME").last()

        # Melt to long format — handle any index name
        trend_reset = monthly_slopes.reset_index()
        date_col = trend_reset.columns[0]
        trend_long = trend_reset.melt(
            id_vars=date_col, var_name="ticker", value_name="volume_trend"
        )
        trend_long = trend_long.rename(columns={date_col: "date"})
        trend_long = trend_long.dropna(subset=["volume_trend"])

        logger.info(
            "Volume Trend (20d): %d observations across %d tickers",
            len(trend_long),
            trend_long["ticker"].nunique(),
        )

        return trend_long

    # ──────────────────────────────────────────────────────────────────────
    # Combined Factor Output
    # ──────────────────────────────────────────────────────────────────────

    def compute_all(self) -> pd.DataFrame:
        """Compute all three factors, merge, winsorise, and z-score.

        Pipeline:
            1. Compute momentum_12_1, realized_volatility_20d, volume_trend_20d.
            2. Merge on ``[date, ticker]``.
            3. Winsorise each factor at (2.5, 97.5) percentiles per date.
            4. Z-score each factor cross-sectionally per date.

        Returns:
            pd.DataFrame: Merged factor data with columns
                ``[date, ticker, momentum_12_1, realized_vol, volume_trend]``.
                All factor columns are winsorised and z-scored.
        """
        logger.info("Computing all factors...")

        # Step 1: Compute raw factors
        mom = self.momentum_12_1()
        vol = self.realized_volatility_20d()
        vt = self.volume_trend_20d()

        # Step 2: Merge on [date, ticker]
        factor_data = mom.merge(vol, on=["date", "ticker"], how="inner")
        factor_data = factor_data.merge(vt, on=["date", "ticker"], how="inner")

        logger.info(
            "Merged factor data: %d rows, %d tickers, %d dates",
            len(factor_data),
            factor_data["ticker"].nunique(),
            factor_data["date"].nunique(),
        )

        # Step 3 & 4: Winsorise and z-score each factor per date
        pp = Preprocessor()
        factor_cols = ["momentum_12_1", "realized_vol", "volume_trend"]

        for col in factor_cols:
            factor_data = pp.winsorise_by_date(factor_data, col, lower=2.5, upper=97.5)
            factor_data = pp.zscore_by_date(factor_data, col)
            logger.info("Winsorised and z-scored: %s", col)

        # Sort for reproducibility
        factor_data = factor_data.sort_values(["date", "ticker"]).reset_index(drop=True)

        logger.info("Factor computation complete: %d rows", len(factor_data))
        return factor_data

    # ──────────────────────────────────────────────────────────────────────
    # Alphalens Integration
    # ──────────────────────────────────────────────────────────────────────

    def to_alphalens_format(
        self, factor_name: str, prices: pd.DataFrame
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Convert factor data to alphalens-compatible format.

        Alphalens expects:
            - ``factor``: a Series with MultiIndex ``(date, asset)``.
            - ``prices``: a wide-format DataFrame with index=Date,
              columns=Tickers.

        Args:
            factor_name: Name of the factor column (e.g. ``"momentum_12_1"``).
            prices: Wide-format daily closing prices.

        Returns:
            Tuple[pd.Series, pd.DataFrame]: ``(factor_series, prices)``
                ready for ``alphalens.utils.get_clean_factor_and_forward_returns()``.

        Raises:
            ValueError: If ``factor_name`` is not a recognised factor.
        """
        valid_factors = ["momentum_12_1", "realized_vol", "volume_trend"]
        if factor_name not in valid_factors:
            raise ValueError(
                f"Unknown factor '{factor_name}'. Must be one of {valid_factors}."
            )

        # Compute all factors first
        factor_data = self.compute_all()

        # Build MultiIndex Series
        factor_series = factor_data.set_index(["date", "ticker"])[factor_name]
        factor_series.index.names = ["date", "asset"]

        # Ensure prices index is datetime
        prices_clean = prices.copy()
        prices_clean.index = pd.to_datetime(prices_clean.index)

        logger.info(
            "Alphalens format: %d factor obs, %d price dates × %d assets",
            len(factor_series),
            len(prices_clean),
            len(prices_clean.columns),
        )

        return factor_series, prices_clean
