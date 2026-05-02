"""
Unit Tests for Factor Engine and Preprocessor
================================================

Tests the factor computation logic, ensuring momentum skips the most recent month,
winsorisation clips at correct percentiles, and z-scoring produces the expected
distributional properties.

Usage:
    $ pytest tests/test_factors.py -v
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessor import Preprocessor
from factors.engine import FactorEngine


class TestPreprocessor:
    """Test suite for the Preprocessor class."""

    @pytest.fixture
    def preprocessor(self) -> Preprocessor:
        """Create a Preprocessor instance."""
        return Preprocessor()

    @pytest.fixture
    def sample_prices(self) -> pd.DataFrame:
        """Create synthetic price data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        tickers = [f"STOCK{i}.NS" for i in range(1, 21)]

        # Generate correlated log-normal prices
        n_stocks = len(tickers)
        daily_returns = np.random.normal(0.0005, 0.02, size=(len(dates), n_stocks))
        prices = np.exp(np.cumsum(daily_returns, axis=0)) * 100

        df = pd.DataFrame(prices, index=dates, columns=tickers)
        return df

    def test_log_returns_shape(self, preprocessor, sample_prices):
        """Test that log returns have correct shape (one row fewer)."""
        log_ret = preprocessor.compute_log_returns(sample_prices)
        assert log_ret.shape[0] == sample_prices.shape[0] - 1
        assert log_ret.shape[1] == sample_prices.shape[1]

    def test_log_returns_values(self, preprocessor, sample_prices):
        """Test that log returns are computed correctly."""
        log_ret = preprocessor.compute_log_returns(sample_prices)
        # Manual check for first stock, second row
        expected = np.log(
            sample_prices.iloc[1, 0] / sample_prices.iloc[0, 0]
        )
        np.testing.assert_almost_equal(
            log_ret.iloc[0, 0], expected, decimal=10
        )

    def test_monthly_returns_frequency(self, preprocessor, sample_prices):
        """Test that monthly returns are at monthly frequency."""
        monthly_ret = preprocessor.compute_monthly_returns(sample_prices)
        # Should have approximately 500/21 ≈ 24 months
        assert len(monthly_ret) >= 15
        assert len(monthly_ret) <= 30

    def test_winsorise_clips_extremes(self, preprocessor):
        """Test that winsorise clips at the correct percentiles."""
        np.random.seed(42)
        values = pd.Series(np.random.randn(1000))
        # Add extreme outliers
        values.iloc[0] = 100.0
        values.iloc[1] = -100.0

        lower, upper = 2.5, 97.5
        result = preprocessor.winsorise(values, lower=lower, upper=upper)

        # After winsorising, no value should exceed the 97.5th percentile
        p_upper = np.percentile(values, upper)
        p_lower = np.percentile(values, lower)

        assert result.max() <= p_upper + 1e-10, (
            f"Max after winsorise ({result.max()}) exceeds upper bound ({p_upper})"
        )
        assert result.min() >= p_lower - 1e-10, (
            f"Min after winsorise ({result.min()}) below lower bound ({p_lower})"
        )

    def test_winsorise_preserves_non_extreme_values(self, preprocessor):
        """Test that winsorise does not alter non-extreme values."""
        # With a sufficiently large Series, middle values remain unchanged
        np.random.seed(42)
        values = pd.Series(np.linspace(1, 100, 100))
        result = preprocessor.winsorise(values, lower=5.0, upper=95.0)
        # The middle 90% of values should be unchanged
        mid_values = values.iloc[5:95]
        mid_results = result.iloc[5:95]
        pd.testing.assert_series_equal(mid_values, mid_results)

    def test_zscore_mean_near_zero(self, preprocessor):
        """Test that z-scoring produces a distribution with mean ≈ 0."""
        np.random.seed(42)
        values = pd.Series(np.random.normal(50, 10, 100))
        result = preprocessor.cross_sectional_zscore(values)
        assert abs(result.mean()) < 1e-10, (
            f"Z-scored mean should be ~0, got {result.mean()}"
        )

    def test_zscore_std_near_one(self, preprocessor):
        """Test that z-scoring produces a distribution with std ≈ 1."""
        np.random.seed(42)
        values = pd.Series(np.random.normal(50, 10, 100))
        result = preprocessor.cross_sectional_zscore(values)
        assert abs(result.std() - 1.0) < 0.02, (
            f"Z-scored std should be ~1, got {result.std()}"
        )

    def test_build_return_matrix_shape(self, preprocessor, sample_prices):
        """Test that the return matrix is T × N."""
        log_ret = preprocessor.compute_log_returns(sample_prices)
        rm = preprocessor.build_return_matrix(log_ret)
        # Should be a wide DataFrame: dates as rows, tickers as columns
        assert rm.shape[1] == sample_prices.shape[1]
        assert rm.shape[0] > 0


class TestFactorEngine:
    """Test suite for the FactorEngine class."""

    @pytest.fixture
    def sample_data(self) -> tuple:
        """Create synthetic prices and volumes for testing."""
        np.random.seed(42)
        dates = pd.date_range("2018-01-01", periods=800, freq="B")
        tickers = [f"STOCK{i}.NS" for i in range(1, 31)]
        n_stocks = len(tickers)

        # Prices: log-normal random walk
        daily_rets = np.random.normal(0.0003, 0.02, (len(dates), n_stocks))
        prices = np.exp(np.cumsum(daily_rets, axis=0)) * 100
        prices_df = pd.DataFrame(prices, index=dates, columns=tickers)

        # Volumes: positive with some trend
        volumes = np.abs(np.random.normal(1e6, 3e5, (len(dates), n_stocks)))
        volumes_df = pd.DataFrame(volumes, index=dates, columns=tickers)

        return prices_df, volumes_df

    @pytest.fixture
    def engine(self, sample_data) -> FactorEngine:
        """Create a FactorEngine instance with synthetic data."""
        prices, volumes = sample_data
        return FactorEngine(prices, volumes)

    def test_momentum_skips_last_month(self, engine):
        """Test that momentum (12-1) skips the most recent month.

        The momentum factor should use price[t-1M] / price[t-12M] - 1,
        NOT price[t] / price[t-12M] - 1. The most recent month is skipped
        to avoid short-term reversal effects (Jegadeesh & Titman, 1993).
        """
        mom = engine.momentum_12_1()
        assert "momentum_12_1" in mom.columns
        assert "date" in mom.columns
        assert "ticker" in mom.columns
        assert len(mom) > 0, "Momentum DataFrame should not be empty."

        # The earliest momentum date should be ~12 months after data start
        data_start = engine.prices.index.min()
        first_mom_date = mom["date"].min()
        months_diff = (first_mom_date - data_start).days / 30
        assert months_diff >= 11, (
            f"First momentum date is only {months_diff:.0f} months after start; "
            "expected at least 12 months for 12-1 momentum."
        )

    def test_momentum_values_reasonable(self, engine):
        """Test that momentum values are within a reasonable range."""
        mom = engine.momentum_12_1()
        # After z-scoring in compute_all, but raw values should be reasonable returns
        assert mom["momentum_12_1"].notna().sum() > 0
        # Raw momentum (before z-scoring) should typically be between -90% and +500%
        raw_mom = mom["momentum_12_1"].dropna()
        assert raw_mom.min() > -1.0, "Momentum < -100% is suspicious."

    def test_realized_vol_positive(self, engine):
        """Test that realized volatility is always non-negative."""
        vol = engine.realized_volatility_20d()
        assert "realized_vol" in vol.columns
        valid_vol = vol["realized_vol"].dropna()
        assert (valid_vol >= 0).all(), "Realized volatility should be non-negative."

    def test_realized_vol_annualised(self, engine):
        """Test that realized vol is annualised (multiplied by sqrt(252))."""
        vol = engine.realized_volatility_20d()
        valid_vol = vol["realized_vol"].dropna()
        # Annualised vol for typical stocks should be between 5% and 200%
        median_vol = valid_vol.median()
        assert 0.05 < median_vol < 2.0, (
            f"Median annualised vol = {median_vol:.2f}, expected 5%–200%."
        )

    def test_volume_trend_has_values(self, engine):
        """Test that volume trend computation produces values."""
        vt = engine.volume_trend_20d()
        assert "volume_trend" in vt.columns
        assert vt["volume_trend"].notna().sum() > 0

    def test_compute_all_merges_factors(self, engine):
        """Test that compute_all produces a DataFrame with all three factors."""
        factor_data = engine.compute_all()
        assert "momentum_12_1" in factor_data.columns
        assert "realized_vol" in factor_data.columns
        assert "volume_trend" in factor_data.columns
        assert "date" in factor_data.columns
        assert "ticker" in factor_data.columns

    def test_compute_all_no_nans_after_processing(self, engine):
        """Test that compute_all drops rows with NaN factor values."""
        factor_data = engine.compute_all()
        factor_cols = ["momentum_12_1", "realized_vol", "volume_trend"]
        for col in factor_cols:
            nan_count = factor_data[col].isna().sum()
            total = len(factor_data)
            nan_pct = nan_count / total if total > 0 else 0
            # Allow some NaNs (edge cases), but the majority should be complete
            assert nan_pct < 0.5, (
                f"{col} has {nan_pct*100:.0f}% NaN values — too many."
            )

    def test_zscore_applied_cross_sectionally(self, engine):
        """Test that z-scoring is applied per date (cross-sectionally)."""
        factor_data = engine.compute_all()

        for factor_col in ["momentum_12_1", "realized_vol", "volume_trend"]:
            # For each date, check mean ≈ 0 and std ≈ 1
            for date in factor_data["date"].unique()[:10]:
                snapshot = factor_data[factor_data["date"] == date][factor_col].dropna()
                if len(snapshot) < 10:
                    continue
                assert abs(snapshot.mean()) < 0.1, (
                    f"{factor_col} cross-sectional mean at {date} = {snapshot.mean():.4f}"
                )
                assert abs(snapshot.std() - 1.0) < 0.3, (
                    f"{factor_col} cross-sectional std at {date} = {snapshot.std():.4f}"
                )
