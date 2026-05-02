"""
EigenAlpha — Data Loader
========================

Provides the ``DataLoader`` class for downloading, caching, and serving
OHLCV equity data from Yahoo Finance (via the ``yfinance`` library).

The loader produces a MultiIndex DataFrame with levels ``(Date, Ticker)``
and columns ``[Open, High, Low, Close, Volume]``.  This format is the
canonical input to all downstream modules in the EigenAlpha pipeline.

Usage:
    from data.loader import DataLoader
    loader = DataLoader()
    df = loader.fetch(["RELIANCE.NS", "TCS.NS"], "2020-01-01", "2024-12-31")
    loader.save(df, "data_cache/prices.parquet")
    df = loader.load("data_cache/prices.parquet")
"""

import logging
import os
from typing import List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataLoader:
    """Download, cache, and serve OHLCV equity data from Yahoo Finance.

    This class wraps ``yfinance.download()`` to provide a clean, validated
    MultiIndex DataFrame suitable for quantitative research.

    Attributes:
        cache_dir (str): Directory for parquet cache files.
    """

    def __init__(self, cache_dir: str = "data_cache") -> None:
        """Initialise the DataLoader.

        Args:
            cache_dir: Directory path for caching downloaded data as
                parquet files.  Created automatically if it does not exist.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info("DataLoader initialised with cache_dir='%s'", self.cache_dir)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def fetch(
        self,
        tickers: List[str],
        start: str,
        end: str,
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """Download OHLCV data for a list of tickers from Yahoo Finance.

        Args:
            tickers: List of Yahoo Finance ticker symbols (e.g.
                ``["RELIANCE.NS", "TCS.NS"]``).
            start: Start date in ``YYYY-MM-DD`` format (inclusive).
            end: End date in ``YYYY-MM-DD`` format (inclusive).
            auto_adjust: If ``True``, adjust OHLC prices for dividends and
                splits.  Defaults to ``True``.

        Returns:
            pd.DataFrame: MultiIndex DataFrame with levels ``(Date, Ticker)``
                and columns ``[Open, High, Low, Close, Volume]``.

        Raises:
            ValueError: If ``tickers`` is empty or ``start >= end``.
        """
        if not tickers:
            raise ValueError("tickers list must not be empty.")
        if start >= end:
            raise ValueError(
                f"start date ({start}) must be before end date ({end})."
            )

        logger.info(
            "Downloading %d tickers from %s to %s ...", len(tickers), start, end
        )

        # yfinance returns a DataFrame with MultiIndex columns (Field, Ticker)
        # when multiple tickers are requested.
        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
            threads=True,
            progress=True,
        )

        if raw.empty:
            raise ValueError(
                "yfinance returned an empty DataFrame — check tickers / dates."
            )

        df = self._reshape_to_multiindex(raw, tickers)

        logger.info(
            "Download complete: %d rows, %d unique tickers, date range %s – %s",
            len(df),
            df.index.get_level_values("Ticker").nunique(),
            df.index.get_level_values("Date").min().strftime("%Y-%m-%d"),
            df.index.get_level_values("Date").max().strftime("%Y-%m-%d"),
        )

        return df

    def save(self, df: pd.DataFrame, path: Optional[str] = None) -> str:
        """Save a DataFrame to parquet format.

        Args:
            df: DataFrame to persist.
            path: Destination path.  If ``None``, defaults to
                ``<cache_dir>/prices.parquet``.

        Returns:
            str: The absolute path to the saved file.
        """
        if path is None:
            path = os.path.join(self.cache_dir, "prices.parquet")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        df.to_parquet(path, engine="pyarrow")
        logger.info("Saved DataFrame (%d rows) to %s", len(df), path)
        return path

    def load(self, path: Optional[str] = None) -> pd.DataFrame:
        """Load a DataFrame from parquet format.

        Args:
            path: Path to the parquet file.  If ``None``, defaults to
                ``<cache_dir>/prices.parquet``.

        Returns:
            pd.DataFrame: The loaded DataFrame.

        Raises:
            FileNotFoundError: If the parquet file does not exist.
        """
        if path is None:
            path = os.path.join(self.cache_dir, "prices.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cache file not found: {path}")
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info("Loaded DataFrame (%d rows) from %s", len(df), path)
        return df

    def get_benchmark(
        self,
        ticker: str = "^NSEI",
        start: str = "2014-01-01",
        end: str = "2024-12-31",
    ) -> pd.Series:
        """Fetch a benchmark index closing price series.

        Args:
            ticker: Yahoo Finance ticker for the benchmark index.
                Defaults to ``^NSEI`` (Nifty 50).
            start: Start date in ``YYYY-MM-DD`` format.
            end: End date in ``YYYY-MM-DD`` format.

        Returns:
            pd.Series: Daily closing prices indexed by date.
        """
        logger.info("Fetching benchmark %s from %s to %s", ticker, start, end)

        raw = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            raise ValueError(
                f"yfinance returned empty data for benchmark '{ticker}'."
            )

        # yfinance may return MultiIndex columns even for single ticker
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].iloc[:, 0]
        else:
            close = raw["Close"]

        close.name = ticker
        close.index.name = "Date"
        close.index = pd.to_datetime(close.index).tz_localize(None)

        logger.info(
            "Benchmark %s: %d observations, %s – %s",
            ticker,
            len(close),
            close.index.min().strftime("%Y-%m-%d"),
            close.index.max().strftime("%Y-%m-%d"),
        )

        return close

    # ──────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _reshape_to_multiindex(
        raw: pd.DataFrame, tickers: List[str]
    ) -> pd.DataFrame:
        """Reshape yfinance output to a (Date, Ticker) MultiIndex DataFrame.

        Args:
            raw: Raw DataFrame from ``yfinance.download()``.
            tickers: Original ticker list requested.

        Returns:
            pd.DataFrame: MultiIndex DataFrame with levels ``(Date, Ticker)``
                and columns ``[Open, High, Low, Close, Volume]``.
        """
        expected_cols = ["Open", "High", "Low", "Close", "Volume"]

        if isinstance(raw.columns, pd.MultiIndex):
            # Multiple tickers: columns are (Field, Ticker)
            frames = []
            available_tickers = raw.columns.get_level_values(1).unique()
            for ticker in tickers:
                if ticker not in available_tickers:
                    logger.warning("Ticker %s not found in download — skipping.", ticker)
                    continue
                sub = raw.xs(ticker, level=1, axis=1).copy()
                # Ensure only expected columns remain
                sub = sub[[c for c in expected_cols if c in sub.columns]]
                sub["Ticker"] = ticker
                frames.append(sub)

            if not frames:
                raise ValueError("No valid ticker data after reshaping.")

            combined = pd.concat(frames, axis=0)
            combined.index.name = "Date"
            combined.index = pd.to_datetime(combined.index).tz_localize(None)
            combined = combined.reset_index().set_index(["Date", "Ticker"])
            combined = combined.sort_index()

        else:
            # Single ticker: columns are plain field names
            ticker = tickers[0]
            combined = raw[[c for c in expected_cols if c in raw.columns]].copy()
            combined["Ticker"] = ticker
            combined.index.name = "Date"
            combined.index = pd.to_datetime(combined.index).tz_localize(None)
            combined = combined.reset_index().set_index(["Date", "Ticker"])
            combined = combined.sort_index()

        # Drop rows where Close is NaN (non-trading days for that stock)
        combined = combined.dropna(subset=["Close"])

        return combined
