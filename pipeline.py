"""
EigenAlpha — Master Pipeline Runner
======================================

This module executes the full EigenAlpha Phase 0 pipeline end-to-end. It
orchestrates data loading, preprocessing, factor computation, PCA decomposition,
clustering, portfolio optimisation, backtesting, and tearsheet generation.

The pipeline is designed to be idempotent: if intermediate data files exist
(e.g., cached parquet files from a previous run), they are reused rather than
re-downloaded. This significantly reduces runtime during iterative development.

Execution Order:
    1. Load data (or use cached parquet)
    2. Preprocess returns
    3. Compute covariance decomposition + PCA + clustering
    4. Compute factor signals + IC analysis
    5. Run quintile backtest for each factor
    6. Run Markowitz-optimised backtest
    7. Generate tearsheet
    8. Print summary metrics to console
    9. Save all outputs to outputs/ directory

Usage:
    $ python pipeline.py

    Or programmatically:
    >>> from pipeline import main
    >>> main()

Author: EigenAlpha Research
"""

import logging
import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

import numpy as np
import pandas as pd

# ─── Configure logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eigenalpha.pipeline")

# ─── Imports from project modules ───────────────────────────────────
from config import (
    START_DATE,
    END_DATE,
    RISK_FREE_RATE,
    N_CLUSTERS,
    TOP_QUINTILE,
    BOTTOM_QUINTILE,
    MIN_HISTORY_MONTHS,
    UNIVERSE_TICKERS,
)
from data.loader import DataLoader
from data.preprocessor import Preprocessor
from data.universe import NIFTY500_TICKERS
from factors.engine import FactorEngine
from factors.ic_analysis import InformationCoefficient
from pca.decompose import CovarianceDecomposer
from pca.cluster import MarketClusterer
from portfolio.optimizer import MarkowitzOptimizer
from portfolio.backtest import Backtester
from portfolio.tearsheet import TearsheetGenerator
from visualisation.eda import EDADashboard
from visualisation.pca_plots import PCAPlotter


# ─── Directory setup ────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data_cache"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def ensure_directories() -> None:
    """Create necessary output directories if they don't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "plots").mkdir(exist_ok=True)
    logger.info("Directories verified: %s, %s", DATA_DIR, OUTPUT_DIR)


def step_1_load_data(
    tickers: list, start: str, end: str
) -> tuple:
    """Step 1: Load OHLCV data and benchmark prices.

    Uses cached parquet files if available; otherwise downloads via yfinance.

    Args:
        tickers: List of ticker symbols in .NS format.
        start: Start date string (YYYY-MM-DD).
        end: End date string (YYYY-MM-DD).

    Returns:
        Tuple of (prices_wide, volumes_wide, benchmark_prices):
            - prices_wide: Wide-format adjusted close prices (dates × tickers).
            - volumes_wide: Wide-format volumes (dates × tickers).
            - benchmark_prices: Nifty 50 index prices as a Series.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Loading Data")
    logger.info("=" * 60)

    loader = DataLoader()
    cache_path = DATA_DIR / "nifty500_ohlcv.parquet"
    benchmark_cache = DATA_DIR / "nifty50_benchmark.parquet"

    # Load or download main data
    if cache_path.exists():
        logger.info("Loading cached data from %s", cache_path)
        raw_data = loader.load(str(cache_path))
    else:
        logger.info("Downloading data for %d tickers...", len(tickers))
        raw_data = loader.fetch(tickers, start, end)
        if raw_data is not None and not raw_data.empty:
            loader.save(raw_data, str(cache_path))
            logger.info("Data cached to %s", cache_path)
        else:
            logger.error("Data download failed or returned empty DataFrame.")
            sys.exit(1)

    # Extract close prices and volumes to wide format
    if isinstance(raw_data.index, pd.MultiIndex) and "Close" in raw_data.columns:
        # DataLoader format: (Date, Ticker) MultiIndex rows, flat columns
        prices_wide = raw_data["Close"].unstack(level="Ticker")
        if "Volume" in raw_data.columns:
            volumes_wide = raw_data["Volume"].unstack(level="Ticker")
        else:
            volumes_wide = pd.DataFrame()
    elif isinstance(raw_data.columns, pd.MultiIndex):
        # yfinance raw multi-ticker download format (Field, Ticker) columns
        if "Close" in raw_data.columns.get_level_values(0):
            prices_wide = raw_data["Close"]
            volumes_wide = raw_data["Volume"]
        elif "Adj Close" in raw_data.columns.get_level_values(0):
            prices_wide = raw_data["Adj Close"]
            volumes_wide = raw_data["Volume"]
        else:
            prices_wide = raw_data
            volumes_wide = pd.DataFrame()
    else:
        prices_wide = raw_data
        volumes_wide = pd.DataFrame()

    # Filter stocks with minimum history
    min_obs = MIN_HISTORY_MONTHS * 21  # Approximate trading days per month
    valid_stocks = prices_wide.columns[prices_wide.notna().sum() >= min_obs]
    prices_wide = prices_wide[valid_stocks]
    if not volumes_wide.empty:
        volumes_wide = volumes_wide[volumes_wide.columns.intersection(valid_stocks)]

    logger.info(
        "Loaded %d stocks with >= %d months of history. Date range: %s to %s",
        len(valid_stocks),
        MIN_HISTORY_MONTHS,
        prices_wide.index.min().strftime("%Y-%m-%d"),
        prices_wide.index.max().strftime("%Y-%m-%d"),
    )

    # Load benchmark
    if benchmark_cache.exists():
        logger.info("Loading cached benchmark from %s", benchmark_cache)
        benchmark_df = pd.read_parquet(str(benchmark_cache))
        benchmark_prices = benchmark_df.iloc[:, 0]
    else:
        logger.info("Downloading Nifty 50 benchmark...")
        benchmark_prices = loader.get_benchmark("^NSEI", start, end)
        if benchmark_prices is not None and not benchmark_prices.empty:
            benchmark_prices.to_frame().to_parquet(str(benchmark_cache))

    return prices_wide, volumes_wide, benchmark_prices


def step_2_preprocess(
    prices_wide: pd.DataFrame,
) -> tuple:
    """Step 2: Compute returns and build the return matrix.

    Args:
        prices_wide: Wide-format price matrix.

    Returns:
        Tuple of (log_returns, monthly_returns, return_matrix):
            - log_returns: Daily log returns (wide format).
            - monthly_returns: Monthly simple returns (wide format).
            - return_matrix: The R matrix (T × N) used for PCA.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing")
    logger.info("=" * 60)

    preprocessor = Preprocessor()

    log_returns = preprocessor.compute_log_returns(prices_wide)
    monthly_returns = preprocessor.compute_monthly_returns(prices_wide)
    return_matrix = preprocessor.build_return_matrix(log_returns)

    logger.info(
        "Return matrix shape: %s (T=%d dates × N=%d stocks)",
        return_matrix.shape,
        return_matrix.shape[0],
        return_matrix.shape[1],
    )

    return log_returns, monthly_returns, return_matrix


def step_3_pca_and_clustering(
    return_matrix: pd.DataFrame,
    n_clusters: int = N_CLUSTERS,
) -> tuple:
    """Step 3: Covariance decomposition, PCA, and K-Means clustering.

    Args:
        return_matrix: The R matrix (T × N).
        n_clusters: Number of K-Means clusters.

    Returns:
        Tuple of (decomposer, clusterer, cluster_labels):
            - decomposer: Fitted CovarianceDecomposer.
            - clusterer: Fitted MarketClusterer.
            - cluster_labels: Series mapping ticker → cluster_id.
    """
    logger.info("=" * 60)
    logger.info("STEP 3: PCA Decomposition & Clustering")
    logger.info("=" * 60)

    # Covariance decomposition
    decomposer = CovarianceDecomposer(return_matrix)
    decomposer.compute_covariance()
    eigenvalues, eigenvectors = decomposer.eigendecompose()

    logger.info(
        "Top 5 eigenvalues: %s",
        np.round(eigenvalues[:5], 4),
    )

    # PCA
    n_components = min(50, return_matrix.shape[1] - 1)
    decomposer.fit_pca(n_components=n_components)
    k = decomposer.select_components(variance_threshold=0.80)
    logger.info(
        "PCA: %d components explain 80%% of variance (out of %d)",
        k,
        n_components,
    )

    # K-Means clustering
    clusterer = MarketClusterer(decomposer, n_clusters=n_clusters)
    pc_scores = clusterer.get_stock_pc_scores(n_dims=3)
    cluster_labels = clusterer.fit_kmeans()

    # Log cluster sizes
    cluster_sizes = cluster_labels.value_counts().sort_index()
    for cid, size in cluster_sizes.items():
        logger.info("  Cluster %d: %d stocks", cid, size)

    return decomposer, clusterer, cluster_labels


def step_4_factors_and_ic(
    prices_wide: pd.DataFrame,
    volumes_wide: pd.DataFrame,
    monthly_returns: pd.DataFrame,
) -> tuple:
    """Step 4: Compute factor signals and IC analysis.

    Args:
        prices_wide: Wide-format price matrix.
        volumes_wide: Wide-format volume matrix.
        monthly_returns: Monthly simple returns (wide format).

    Returns:
        Tuple of (factor_data, ic_results, forward_returns):
            - factor_data: Merged, winsorised, z-scored factor DataFrame.
            - ic_results: Dict of IC analysis results per factor.
            - forward_returns: Forward 1-month returns.
    """
    logger.info("=" * 60)
    logger.info("STEP 4: Factor Computation & IC Analysis")
    logger.info("=" * 60)

    engine = FactorEngine(prices_wide, volumes_wide)
    factor_data = engine.compute_all()

    logger.info(
        "Factor data shape: %s (%d stock-months)",
        factor_data.shape,
        len(factor_data),
    )

    # Compute forward 1-month returns for IC analysis
    fwd_returns = monthly_returns.shift(-1)
    fwd_returns_long = fwd_returns.stack().reset_index()
    fwd_returns_long.columns = ["date", "ticker", "forward_return"]

    # Merge factor data with forward returns
    factor_with_fwd = factor_data.merge(
        fwd_returns_long, on=["date", "ticker"], how="inner"
    )

    # IC analysis for each factor
    ic_results = {}
    factor_cols = ["momentum_12_1", "realized_vol", "volume_trend"]

    for factor_col in factor_cols:
        if factor_col not in factor_with_fwd.columns:
            logger.warning("Factor column %s not found, skipping IC.", factor_col)
            continue

        ic_analyzer = InformationCoefficient(
            factor_data=factor_data,
            forward_returns=fwd_returns_long,
        )

        ic_series = ic_analyzer.compute_ic(factor_col)
        summary = ic_analyzer.ic_summary(factor_col)
        ic_decay = ic_analyzer.ic_decay(factor_col)

        ic_results[factor_col] = {
            "ic_series": ic_series,
            "summary": summary,
            "ic_decay": ic_decay,
            **summary,
        }

        logger.info(
            "  %s: Mean IC=%.4f, IR=%.4f, t-stat=%.2f, %%Positive=%.1f%%",
            factor_col,
            summary["mean_ic"],
            summary["ir"],
            summary["ic_t_stat"],
            summary["pct_positive"] * 100,
        )

    return factor_data, ic_results, factor_with_fwd


def step_5_quintile_backtest(
    prices_wide: pd.DataFrame,
    factor_data: pd.DataFrame,
    benchmark_prices: pd.Series,
) -> tuple:
    """Step 5: Run quintile backtests for each factor.

    Args:
        prices_wide: Wide-format price matrix.
        factor_data: Merged factor DataFrame.
        benchmark_prices: Benchmark price series.

    Returns:
        Tuple of (backtester, quintile_results):
            - backtester: Backtester instance (reused in step 6).
            - quintile_results: Dict mapping factor name → quintile returns DataFrame.
    """
    logger.info("=" * 60)
    logger.info("STEP 5: Quintile Backtests")
    logger.info("=" * 60)

    backtester = Backtester(prices_wide, factor_data, benchmark_prices)
    quintile_results = {}

    factor_cols = ["momentum_12_1", "realized_vol", "volume_trend"]
    for factor_col in factor_cols:
        if factor_col not in factor_data.columns:
            continue

        try:
            q_returns = backtester.run_quintile_backtest(factor_col)
            quintile_results[factor_col] = q_returns

            # Compute metrics for the long-short spread
            if "Long_Short" in q_returns.columns:
                ls_metrics = backtester.compute_metrics(q_returns["Long_Short"].dropna())
                logger.info(
                    "  %s L/S: Sharpe=%.2f, AnnRet=%.1f%%, MaxDD=%.1f%%",
                    factor_col,
                    ls_metrics["sharpe_ratio"],
                    ls_metrics["annualised_return"] * 100,
                    ls_metrics["max_drawdown"] * 100,
                )

            # Q5 metrics (long-only top quintile)
            q5_metrics = backtester.compute_metrics(q_returns["Q5"].dropna())
            logger.info(
                "  %s Q5 (long-only): Sharpe=%.2f, AnnRet=%.1f%%",
                factor_col,
                q5_metrics["sharpe_ratio"],
                q5_metrics["annualised_return"] * 100,
            )

        except Exception as e:
            logger.warning("Quintile backtest failed for %s: %s", factor_col, str(e))

    return backtester, quintile_results


def step_6_markowitz_backtest(
    backtester: "Backtester",
    cluster_labels: pd.Series,
) -> pd.Series:
    """Step 6: Run walk-forward Markowitz-optimised backtest.

    Args:
        backtester: Backtester instance from step 5.
        cluster_labels: K-Means cluster assignments.

    Returns:
        pd.Series: Monthly portfolio returns from the optimised strategy.
    """
    logger.info("=" * 60)
    logger.info("STEP 6: Markowitz Walk-Forward Backtest")
    logger.info("=" * 60)

    try:
        markowitz_returns = backtester.run_markowitz_backtest(
            optimizer_class=MarkowitzOptimizer,
            factor_col="momentum_12_1",
            lookback_months=36,
            cluster_labels=cluster_labels,
        )

        if len(markowitz_returns) > 0:
            metrics = backtester.compute_metrics(markowitz_returns)
            logger.info(
                "Markowitz portfolio: Sharpe=%.2f, AnnRet=%.1f%%, MaxDD=%.1f%%",
                metrics["sharpe_ratio"],
                metrics["annualised_return"] * 100,
                metrics["max_drawdown"] * 100,
            )
        else:
            logger.warning("Markowitz backtest produced no returns.")

        return markowitz_returns

    except Exception as e:
        logger.error("Markowitz backtest failed: %s", str(e))
        return pd.Series(dtype=float)


def step_7_generate_outputs(
    portfolio_returns: pd.Series,
    benchmark_prices: pd.Series,
    factor_data: pd.DataFrame,
    ic_results: dict,
    quintile_results: dict,
    backtester: "Backtester",
    decomposer: "CovarianceDecomposer",
    clusterer: "MarketClusterer",
    prices_wide: pd.DataFrame,
) -> None:
    """Step 7: Generate tearsheet, EDA plots, and PCA visualisations.

    Args:
        portfolio_returns: Strategy monthly returns.
        benchmark_prices: Benchmark price series.
        factor_data: Merged factor DataFrame.
        ic_results: IC analysis results.
        quintile_results: Quintile backtest results.
        backtester: Backtester instance.
        decomposer: Fitted CovarianceDecomposer.
        clusterer: Fitted MarketClusterer.
        prices_wide: Price matrix for EDA.
    """
    logger.info("=" * 60)
    logger.info("STEP 7: Generating Outputs")
    logger.info("=" * 60)

    # Compute metrics for the tearsheet
    metrics = {}
    if len(portfolio_returns) > 0:
        metrics = backtester.compute_metrics(portfolio_returns)
        try:
            relative_metrics = backtester.compute_benchmark_relative_metrics(
                portfolio_returns
            )
            metrics.update(relative_metrics)
        except Exception as e:
            logger.warning("Benchmark-relative metrics failed: %s", str(e))

    # Use the momentum quintile results for the tearsheet
    quintile_df = quintile_results.get("momentum_12_1", None)

    # Benchmark monthly returns
    bench_monthly = benchmark_prices.resample("ME").last().pct_change().dropna() \
        if benchmark_prices is not None and not benchmark_prices.empty else pd.Series(dtype=float)

    # Generate tearsheet
    try:
        tearsheet = TearsheetGenerator(
            portfolio_returns=portfolio_returns if len(portfolio_returns) > 0
                else pd.Series([0.0], index=[pd.Timestamp("2020-01-31")]),
            benchmark_returns=bench_monthly,
            factor_data=factor_data,
            ic_results=ic_results,
            quintile_returns=quintile_df,
            pca_decomposer=decomposer,
            clusterer=clusterer,
            metrics=metrics,
        )
        tearsheet_path = str(OUTPUT_DIR / "tearsheet.png")
        tearsheet.generate(tearsheet_path)
        logger.info("Tearsheet saved to %s", tearsheet_path)
    except Exception as e:
        logger.error("Tearsheet generation failed: %s", str(e))

    # EDA plots
    try:
        eda = EDADashboard(factor_data, prices_wide)
        eda.factor_distributions(save_path=str(OUTPUT_DIR / "plots" / "factor_distributions.png"))
        eda.factor_correlation_heatmap(save_path=str(OUTPUT_DIR / "plots" / "factor_correlations.png"))
        eda.autocorrelation_decay(save_path=str(OUTPUT_DIR / "plots" / "autocorrelation_decay.png"))
        eda.turnover_analysis(save_path=str(OUTPUT_DIR / "plots" / "turnover_analysis.png"))
        logger.info("EDA plots saved to %s", OUTPUT_DIR / "plots")
    except Exception as e:
        logger.error("EDA plot generation failed: %s", str(e))

    # PCA plots
    try:
        pca_plotter = PCAPlotter(decomposer, clusterer)
        pca_plotter.scree_chart(save_path=str(OUTPUT_DIR / "plots" / "pca_scree.png"))
        pca_plotter.cluster_scatter_2d(save_path=str(OUTPUT_DIR / "plots" / "cluster_scatter_2d.png"))
        pca_plotter.silhouette_plot(save_path=str(OUTPUT_DIR / "plots" / "silhouette_scores.png"))
        try:
            pca_plotter.eigenvalue_spectrum(save_path=str(OUTPUT_DIR / "plots" / "eigenvalue_spectrum.png"))
        except Exception:
            pass  # Optional - depends on eigendecomposition being available
        logger.info("PCA plots saved to %s", OUTPUT_DIR / "plots")
    except Exception as e:
        logger.error("PCA plot generation failed: %s", str(e))


def step_8_print_summary(
    backtester: "Backtester",
    portfolio_returns: pd.Series,
    quintile_results: dict,
    ic_results: dict,
) -> None:
    """Step 8: Print a formatted summary table to console.

    Args:
        backtester: Backtester instance for computing metrics.
        portfolio_returns: Strategy monthly returns.
        quintile_results: Quintile backtest results.
        ic_results: IC analysis results.
    """
    logger.info("=" * 60)
    logger.info("STEP 8: Performance Summary")
    logger.info("=" * 60)

    print("\n" + "=" * 70)
    print("  EigenAlpha Phase 0 - Performance Summary")
    print("=" * 70)

    # IC Summary Table
    print("\n+-------------------------------------------------------------+")
    print("|                  Information Coefficient (IC)               |")
    print("+------------------+----------+----------+----------+---------+")
    print("| Factor           | Mean IC  |   IR     |  t-stat  | %% Pos   |")
    print("+------------------+----------+----------+----------+---------+")
    for factor, data in ic_results.items():
        s = data.get("summary", data)
        print(
            f"| {factor:<16s} | {s.get('mean_ic', 0):>8.4f} | "
            f"{s.get('ir', 0):>8.4f} | {s.get('ic_t_stat', 0):>8.2f} | "
            f"{s.get('pct_positive', 0) * 100:>6.1f}% |"
        )
    print("+------------------+----------+----------+----------+---------+")

    # Quintile Backtest Summary
    print("\n+-------------------------------------------------------------+")
    print("|               Quintile Backtest (Long-Short)                |")
    print("+------------------+----------+----------+----------+---------+")
    print("| Factor           | Ann Ret  | Ann Vol  |  Sharpe  | Max DD  |")
    print("+------------------+----------+----------+----------+---------+")
    for factor, q_ret in quintile_results.items():
        if "Long_Short" in q_ret.columns:
            m = backtester.compute_metrics(q_ret["Long_Short"].dropna())
            print(
                f"| {factor:<16s} | {m['annualised_return'] * 100:>7.1f}% | "
                f"{m['annualised_volatility'] * 100:>7.1f}% | "
                f"{m['sharpe_ratio']:>8.2f} | "
                f"{m['max_drawdown'] * 100:>6.1f}% |"
            )
    print("+------------------+----------+----------+----------+---------+")

    # Markowitz Portfolio Summary
    if len(portfolio_returns) > 0:
        m = backtester.compute_metrics(portfolio_returns)
        print("\n+-------------------------------------------------------------+")
        print("|           Markowitz Optimised Portfolio                      |")
        print("+--------------------------------------------------------------+")
        print(f"| Annualised Return:      {m['annualised_return'] * 100:>8.2f}%                      |")
        print(f"| Annualised Volatility:  {m['annualised_volatility'] * 100:>8.2f}%                      |")
        print(f"| Sharpe Ratio:           {m['sharpe_ratio']:>8.2f}                        |")
        print(f"| Max Drawdown:           {m['max_drawdown'] * 100:>8.2f}%                      |")
        print(f"| Calmar Ratio:           {m['calmar_ratio']:>8.2f}                        |")
        print(f"| Win Rate:               {m['win_rate'] * 100:>8.1f}%                      |")
        print("+--------------------------------------------------------------+")

    print("\n" + "=" * 70)
    print(f"  All outputs saved to: {OUTPUT_DIR}")
    print("=" * 70 + "\n")


def _refresh_path() -> None:
    """Refresh the current process PATH from the system registry.

    When Node.js is freshly installed (e.g. via winget), the installer
    updates the Machine/User PATH in the registry, but the running
    Python process still has the old PATH.  This function reads the
    latest values so subprocess calls can find ``npm``.
    """
    if sys.platform == "win32":
        import winreg
        parts = []
        for root, sub in [
            (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
            (winreg.HKEY_CURRENT_USER, r"Environment"),
        ]:
            try:
                with winreg.OpenKey(root, sub) as key:
                    val, _ = winreg.QueryValueEx(key, "Path")
                    parts.append(val)
            except OSError:
                pass
        if parts:
            os.environ["PATH"] = ";".join(parts)
            logger.debug("PATH refreshed from registry.")


def _find_npm() -> str:
    """Return the npm command string that works on this machine."""
    import shutil
    # On Windows, prefer npm.cmd over npm.ps1 (which may be blocked by execution policy)
    if sys.platform == "win32":
        for candidate in [
            r"C:\Program Files\nodejs\npm.cmd",
            os.path.expandvars(r"%APPDATA%\npm\npm.cmd"),
        ]:
            if os.path.isfile(candidate):
                return candidate
    npm = shutil.which("npm")
    if npm:
        return npm
    return "npm"  # fall back — let the OS resolve it


def step_9_launch_dashboard() -> None:
    """Step 9: Launch the FastAPI backend and Next.js frontend for a live demo.

    This function:
        1. Installs frontend npm dependencies (if not already installed).
        2. Starts the FastAPI (uvicorn) backend on port 8000.
        3. Starts the Next.js dev server on port 3000.
        4. Waits for both servers to become ready.
        5. Opens the dashboard in the default browser.
        6. Blocks until Ctrl+C, then shuts down both servers.
    """
    logger.info("=" * 60)
    logger.info("STEP 9: Launching Dashboard")
    logger.info("=" * 60)

    # Refresh PATH so a freshly-installed Node.js is visible
    _refresh_path()

    frontend_dir = PROJECT_DIR / "frontend"
    if not frontend_dir.exists():
        logger.error("Frontend directory not found at %s", frontend_dir)
        return

    npm_cmd = _find_npm()
    logger.info("Using npm: %s", npm_cmd)

    # ── 9a. Install npm dependencies if needed ───────────────────────
    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        logger.info("Installing frontend dependencies (npm install)...")
        npm_install = subprocess.run(
            [npm_cmd, "install"],
            cwd=str(frontend_dir),
            shell=True,
            capture_output=True,
            text=True,
        )
        if npm_install.returncode != 0:
            logger.error("npm install failed:\n%s", npm_install.stderr)
            return
        logger.info("Frontend dependencies installed successfully.")
    else:
        logger.info("Frontend node_modules already present — skipping install.")

    # ── 9b. Start FastAPI backend ────────────────────────────────────
    logger.info("Starting FastAPI backend on http://localhost:8000 ...")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # ── 9c. Start Next.js frontend ───────────────────────────────────
    logger.info("Starting Next.js frontend on http://localhost:3000 ...")
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=str(frontend_dir),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # ── 9d. Wait for servers to become ready ─────────────────────────
    def _wait_for_server(url: str, name: str, timeout: int = 60) -> bool:
        """Poll a URL until it responds or timeout is reached."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                urlopen(url, timeout=2)
                logger.info("  [OK] %s is ready", name)
                return True
            except (URLError, OSError, Exception):
                time.sleep(1)
        logger.warning("  [FAIL] %s did not respond within %ds", name, timeout)
        return False

    _wait_for_server("http://127.0.0.1:8000/api/status", "FastAPI backend", timeout=30)
    _wait_for_server("http://127.0.0.1:3000", "Next.js frontend", timeout=90)

    # ── 9e. Open browser ─────────────────────────────────────────────
    dashboard_url = "http://localhost:3000"
    logger.info("Opening dashboard: %s", dashboard_url)
    webbrowser.open(dashboard_url)

    # ── 9f. Block until Ctrl+C ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  *  EigenAlpha Dashboard is live at http://localhost:3000")
    print("  *  API server running at        http://localhost:8000")
    print("  *  Press Ctrl+C to stop both servers and exit.")
    print("=" * 70 + "\n")

    try:
        # Stream subprocess output to console
        import threading

        def _stream_output(proc, prefix):
            """Read lines from a subprocess and print them."""
            try:
                for line in proc.stdout:
                    print(f"[{prefix}] {line}", end="")
            except (ValueError, OSError):
                pass

        t_back = threading.Thread(target=_stream_output, args=(backend_proc, "API"), daemon=True)
        t_front = threading.Thread(target=_stream_output, args=(frontend_proc, "WEB"), daemon=True)
        t_back.start()
        t_front.start()

        # Block main thread until a subprocess exits or Ctrl+C
        while True:
            if backend_proc.poll() is not None:
                logger.warning("Backend process exited (code %d)", backend_proc.returncode)
                break
            if frontend_proc.poll() is not None:
                logger.warning("Frontend process exited (code %d)", frontend_proc.returncode)
                break
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n")
        logger.info("Shutting down servers...")

    finally:
        # Graceful shutdown
        for name, proc in [("Backend", backend_proc), ("Frontend", frontend_proc)]:
            if proc.poll() is None:
                logger.info("  Stopping %s (PID %d)...", name, proc.pid)
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                logger.info("  %s stopped.", name)

        logger.info("All servers shut down. Goodbye!")


def main() -> None:
    """Execute the full EigenAlpha Phase 0 pipeline.

    This is the single entry point for running the complete research pipeline.
    It calls each step sequentially and handles errors gracefully, logging
    progress throughout.

    Returns:
        None. All outputs are saved to the outputs/ directory.
    """
    start_time = time.time()

    print("""
    ============================================================
    |                                                          |
    |   EIGEN ALPHA                                            |
    |                                                          |
    |   Phase 0 - Factor-Based Portfolio Construction Engine   |
    |   Indian Equities (Nifty 500 Universe)                   |
    |                                                          |
    ============================================================
    """)

    logger.info("EigenAlpha Phase 0 Pipeline - Starting")
    logger.info("Period: %s to %s", START_DATE, END_DATE)

    # Setup
    ensure_directories()
    tickers = UNIVERSE_TICKERS

    # Step 1: Load data
    prices_wide, volumes_wide, benchmark_prices = step_1_load_data(
        tickers, START_DATE, END_DATE
    )

    # Step 2: Preprocess
    log_returns, monthly_returns, return_matrix = step_2_preprocess(prices_wide)

    # Step 3: PCA & Clustering
    decomposer, clusterer, cluster_labels = step_3_pca_and_clustering(
        return_matrix, n_clusters=N_CLUSTERS
    )

    # Step 4: Factors & IC
    factor_data, ic_results, factor_with_fwd = step_4_factors_and_ic(
        prices_wide, volumes_wide, monthly_returns
    )

    # Step 5: Quintile backtests
    backtester, quintile_results = step_5_quintile_backtest(
        prices_wide, factor_data, benchmark_prices
    )

    # Step 6: Markowitz backtest
    markowitz_returns = step_6_markowitz_backtest(backtester, cluster_labels)

    # Step 7: Generate outputs
    step_7_generate_outputs(
        portfolio_returns=markowitz_returns,
        benchmark_prices=benchmark_prices,
        factor_data=factor_data,
        ic_results=ic_results,
        quintile_results=quintile_results,
        backtester=backtester,
        decomposer=decomposer,
        clusterer=clusterer,
        prices_wide=prices_wide,
    )

    # Step 8: Print summary
    step_8_print_summary(backtester, markowitz_returns, quintile_results, ic_results)

    # Step 9: Save raw results
    logger.info("Saving raw results...")
    factor_data.to_parquet(str(OUTPUT_DIR / "factor_data.parquet"))
    if len(markowitz_returns) > 0:
        markowitz_returns.to_frame().to_parquet(
            str(OUTPUT_DIR / "markowitz_returns.parquet")
        )
    for factor_name, q_ret in quintile_results.items():
        q_ret.to_parquet(
            str(OUTPUT_DIR / f"quintile_returns_{factor_name}.parquet")
        )

    elapsed = time.time() - start_time
    logger.info("EigenAlpha Phase 0 Pipeline — Completed in %.1f seconds", elapsed)

    # Step 10: Launch dashboard (FastAPI + Next.js)
    step_9_launch_dashboard()


if __name__ == "__main__":
    main()
