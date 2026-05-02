"""
Unit Tests for DataLoader
===========================

Tests the data loading pipeline to ensure correctness of the MultiIndex
DataFrame structure, date integrity, and price validity.

Usage:
    $ pytest tests/test_loader.py -v
"""

import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import DataLoader
from data.universe import NIFTY500_TICKERS


class TestDataLoader:
    """Test suite for the DataLoader class."""

    @pytest.fixture
    def loader(self) -> DataLoader:
        """Create a DataLoader instance."""
        return DataLoader()

    @pytest.fixture
    def sample_tickers(self) -> list:
        """A small set of liquid tickers for fast testing."""
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

    @pytest.fixture
    def sample_data(self, loader, sample_tickers) -> pd.DataFrame:
        """Fetch a small sample dataset for testing."""
        data = loader.fetch(
            sample_tickers,
            start="2023-01-01",
            end="2023-06-30",
        )
        return data

    def test_fetch_returns_dataframe(self, sample_data):
        """Test that fetch() returns a non-empty DataFrame."""
        assert isinstance(sample_data, pd.DataFrame)
        assert not sample_data.empty, "Fetched DataFrame should not be empty."

    def test_dataframe_has_correct_columns(self, sample_data):
        """Test that the DataFrame contains OHLCV columns."""
        if isinstance(sample_data.columns, pd.MultiIndex):
            level_0 = sample_data.columns.get_level_values(0).unique()
            expected = {"Close", "High", "Low", "Open", "Volume"}
            assert expected.issubset(set(level_0)), (
                f"Expected OHLCV columns, got: {list(level_0)}"
            )
        else:
            # Single ticker or wide-format
            assert len(sample_data.columns) > 0

    def test_no_future_dates(self, sample_data):
        """Test that no dates in the data are in the future."""
        today = pd.Timestamp.now().normalize()
        max_date = sample_data.index.max()
        if isinstance(max_date, pd.Timestamp):
            assert max_date <= today, (
                f"Data contains future date: {max_date} > {today}"
            )

    def test_positive_prices(self, sample_data):
        """Test that all price values are positive (non-negative)."""
        if isinstance(sample_data.columns, pd.MultiIndex):
            price_cols = ["Open", "High", "Low", "Close"]
            for col in price_cols:
                if col in sample_data.columns.get_level_values(0):
                    prices = sample_data[col]
                    # Allow NaN but no negative prices
                    valid_prices = prices.dropna()
                    if not valid_prices.empty:
                        assert (valid_prices >= 0).all().all(), (
                            f"Negative prices found in {col}"
                        )
        else:
            valid = sample_data.dropna()
            if not valid.empty:
                assert (valid >= 0).all().all(), "Negative prices found in data."

    def test_datetime_index(self, sample_data):
        """Test that the index contains a DatetimeIndex (or MultiIndex with Date level)."""
        if isinstance(sample_data.index, pd.MultiIndex):
            # MultiIndex (Date, Ticker) — check the Date level
            date_level = sample_data.index.get_level_values("Date")
            assert isinstance(date_level, pd.DatetimeIndex), (
                f"Expected Date level to be DatetimeIndex, got {type(date_level)}"
            )
        else:
            assert isinstance(sample_data.index, pd.DatetimeIndex), (
                f"Expected DatetimeIndex, got {type(sample_data.index)}"
            )

    def test_index_is_sorted(self, sample_data):
        """Test that the DatetimeIndex is sorted in ascending order."""
        assert sample_data.index.is_monotonic_increasing, (
            "DatetimeIndex should be sorted in ascending order."
        )

    def test_date_range_respected(self, sample_data):
        """Test date range is within the requested bounds."""
        if isinstance(sample_data.index, pd.MultiIndex):
            dates = sample_data.index.get_level_values("Date")
        else:
            dates = sample_data.index
        min_date = dates.min()
        max_date = dates.max()
        assert min_date >= pd.Timestamp("2022-12-01"), (
            f"Data starts too early: {min_date}"
        )
        assert max_date <= pd.Timestamp("2023-07-31"), (
            f"Data extends too far: {max_date}"
        )

    def test_no_all_nan_tickers(self, sample_data):
        """Test that no ticker has entirely NaN values."""
        if isinstance(sample_data.columns, pd.MultiIndex):
            close = sample_data["Close"] if "Close" in sample_data.columns.get_level_values(0) else sample_data
        else:
            close = sample_data

        all_nan_cols = close.columns[close.isna().all()]
        assert len(all_nan_cols) == 0, (
            f"Tickers with all NaN values: {list(all_nan_cols)}"
        )

    def test_volume_non_negative(self, sample_data):
        """Test that volume values are non-negative."""
        if isinstance(sample_data.columns, pd.MultiIndex):
            if "Volume" in sample_data.columns.get_level_values(0):
                volumes = sample_data["Volume"].dropna()
                if not volumes.empty:
                    assert (volumes >= 0).all().all(), "Negative volumes found."

    def test_save_and_load_roundtrip(self, loader, sample_data, tmp_path):
        """Test that save/load produces identical data."""
        filepath = str(tmp_path / "test_data.parquet")
        loader.save(sample_data, filepath)
        loaded = loader.load(filepath)
        pd.testing.assert_frame_equal(sample_data, loaded)

    def test_benchmark_fetch(self, loader):
        """Test that benchmark (Nifty 50) can be fetched."""
        benchmark = loader.get_benchmark(
            ticker="^NSEI",
            start="2023-01-01",
            end="2023-06-30",
        )
        assert benchmark is not None
        if isinstance(benchmark, pd.Series):
            assert not benchmark.empty, "Benchmark series should not be empty."
            assert (benchmark.dropna() > 0).all(), "Benchmark prices should be positive."

    def test_universe_tickers_format(self):
        """Test that NIFTY500_TICKERS are in correct .NS format."""
        for ticker in NIFTY500_TICKERS[:20]:
            assert ticker.endswith(".NS"), (
                f"Ticker '{ticker}' does not end with .NS"
            )
            assert len(ticker) > 3, f"Ticker '{ticker}' is too short."
