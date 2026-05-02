# EigenAlpha — Pipeline Walkthrough

**Version:** Phase 0  
**Last Updated:** 2025-04

This document walks through every step of the `pipeline.py` execution, describing inputs, outputs, configuration parameters, and the mathematical operations performed at each stage.

---

## Table of Contents

1. [Overview](#overview)
2. [Step 1 — Data Loading](#step-1--data-loading)
3. [Step 2 — Preprocessing](#step-2--preprocessing)
4. [Step 3 — PCA & Clustering](#step-3--pca--clustering)
5. [Step 4 — Factor Computation & IC Analysis](#step-4--factor-computation--ic-analysis)
6. [Step 5 — Quintile Backtests](#step-5--quintile-backtests)
7. [Step 6 — Markowitz Walk-Forward Backtest](#step-6--markowitz-walk-forward-backtest)
8. [Step 7 — Output Generation](#step-7--output-generation)
9. [Step 8 — Performance Summary](#step-8--performance-summary)
10. [Step 9 — Dashboard Launch](#step-9--dashboard-launch)
11. [Configuration Reference](#configuration-reference)
12. [Data Flow Diagram](#data-flow-diagram)

---

## Overview

The pipeline is invoked with:

```bash
python pipeline.py
```

It runs **9 sequential steps** and is designed to be **idempotent**: if cached data files exist from a previous run, they are reused to avoid re-downloading ~120 stocks × 10 years of daily data.

Total runtime: **~5–10 minutes** (first run with download) or **~2–3 minutes** (cached).

---

## Step 1 — Data Loading

**Module:** `data/loader.py` → `DataLoader`  
**Function:** `step_1_load_data(tickers, start, end)`

### What it does

1. Checks for a cached parquet file at `data_cache/nifty500_ohlcv.parquet`
2. If not found: downloads OHLCV data via `yfinance.download()` for all tickers
3. Reshapes the raw download into a `(Date, Ticker)` MultiIndex DataFrame
4. Extracts `Close` and `Volume` into wide-format DataFrames (dates × tickers)
5. Filters stocks requiring ≥ `MIN_HISTORY_MONTHS` (24) months of data
6. Downloads Nifty 50 benchmark prices (`^NSEI`)

### Input

| Parameter | Source | Value |
|-----------|--------|-------|
| `tickers` | `config.UNIVERSE_TICKERS` | ~120 NSE tickers in `.NS` format |
| `start` | `config.START_DATE` | `2014-01-01` |
| `end` | `config.END_DATE` | `2024-12-31` |

### Output

| Variable | Shape | Description |
|----------|-------|-------------|
| `prices_wide` | `(~2500, ~115)` | Daily adjusted close prices |
| `volumes_wide` | `(~2500, ~115)` | Daily trading volumes |
| `benchmark_prices` | `(~2500,)` | Nifty 50 daily close |

### Cache files

- `data_cache/nifty500_ohlcv.parquet` (~13 MB)
- `data_cache/nifty50_benchmark.parquet` (~45 KB)

---

## Step 2 — Preprocessing

**Module:** `data/preprocessor.py` → `Preprocessor`  
**Function:** `step_2_preprocess(prices_wide)`

### What it does

1. **Log returns:** `ln(Pt / Pt-1)` for each stock, each day
2. **Monthly returns:** resample to month-end, compute simple `pct_change()`
3. **Return matrix R:** wide-format matrix (T × N) used for PCA

### Mathematical operations

```
log_returns = ln(prices / prices.shift(1))
monthly_returns = month_end_prices.pct_change()
return_matrix = pivot(log_returns, index=dates, columns=tickers)
```

### Output

| Variable | Shape | Description |
|----------|-------|-------------|
| `log_returns` | `(~2499, ~115)` | Daily log returns |
| `monthly_returns` | `(~130, ~115)` | Monthly simple returns |
| `return_matrix` | `(~2499, ~115)` | R matrix for PCA (T × N) |

---

## Step 3 — PCA & Clustering

**Module:** `pca/decompose.py` → `CovarianceDecomposer`  
**Module:** `pca/cluster.py` → `MarketClusterer`  
**Function:** `step_3_pca_and_clustering(return_matrix, n_clusters)`

### What it does

1. **Covariance matrix:** Σ = R^T R / T (and pandas pairwise method)
2. **Eigendecomposition:** `np.linalg.eigh(Σ)` → eigenvalues λ, eigenvectors V
3. **PCA:** `sklearn.PCA(n_components=50).fit(R)` → explained variance ratios
4. **Component selection:** find minimum k where cumulative variance ≥ 80%
5. **Stock projection:** project each stock onto top-3 PCs → (N × 3) score matrix
6. **K-Means clustering:** `KMeans(k=8, n_init=20)` on PC scores

### Configuration

| Parameter | Config Key | Default |
|-----------|------------|---------|
| Max PCA components | `N_PCA_COMPONENTS` | 50 |
| Variance threshold | `VARIANCE_THRESHOLD` | 0.80 |
| Number of clusters | `N_CLUSTERS` | 8 |
| K-Means n_init | `KMEANS_N_INIT` | 20 |
| Random state | `KMEANS_RANDOM_STATE` | 42 |

### Output

| Variable | Type | Description |
|----------|------|-------------|
| `decomposer` | `CovarianceDecomposer` | Fitted PCA with eigenvalues |
| `clusterer` | `MarketClusterer` | Fitted K-Means with PC scores |
| `cluster_labels` | `pd.Series (N,)` | Ticker → cluster_id mapping |

---

## Step 4 — Factor Computation & IC Analysis

**Module:** `factors/engine.py` → `FactorEngine`  
**Module:** `factors/ic_analysis.py` → `InformationCoefficient`  
**Function:** `step_4_factors_and_ic(prices_wide, volumes_wide, monthly_returns)`

### What it does

1. **Momentum (12-1):** `(price[t-1M] / price[t-12M]) - 1` — skip last month
2. **Realised Vol (20D):** `std(log_returns, 20) × √252` — annualised
3. **Volume Trend (20D):** OLS slope of `log(volume)` over 20 days
4. **Merge:** inner join all three factors on `[date, ticker]`
5. **Winsorise:** clip at 2.5th/97.5th percentile per date
6. **Z-score:** standardise each factor cross-sectionally per date
7. **Forward returns:** shift monthly returns by -1 for IC computation
8. **IC analysis:** Spearman rank correlation per month per factor

### Output

| Variable | Shape | Description |
|----------|-------|-------------|
| `factor_data` | `(~12000, 5)` | `[date, ticker, mom, rvol, voltrd]` |
| `ic_results` | `dict` | Per-factor: mean IC, IR, t-stat, IC series |
| `factor_with_fwd` | `(~11000, 6)` | Factor data + forward returns |

---

## Step 5 — Quintile Backtests

**Module:** `portfolio/backtest.py` → `Backtester`  
**Function:** `step_5_quintile_backtest(prices_wide, factor_data, benchmark_prices)`

### What it does

For each factor (momentum, vol, volume trend):

1. Each month: rank all stocks by factor score
2. Assign to quintiles Q1 (bottom 20%) through Q5 (top 20%)
3. Compute equal-weighted return of each quintile next month
4. Long-short spread: Q5 − Q1

### Output

| Variable | Type | Description |
|----------|------|-------------|
| `backtester` | `Backtester` | Reused in step 6 |
| `quintile_results` | `dict` | Factor → DataFrame (Q1..Q5, Long_Short) |

---

## Step 6 — Markowitz Walk-Forward Backtest

**Module:** `portfolio/optimizer.py` → `MarkowitzOptimizer`  
**Function:** `step_6_markowitz_backtest(backtester, cluster_labels)`

### What it does

Walk-forward (no look-ahead bias):

1. Each month: use trailing 36 months of data for covariance estimation
2. Get factor scores at formation date
3. Get cluster labels for valid stocks
4. **Per cluster:** run `min w^T Σ w` (SLSQP) → intra-cluster optimal weights
5. **Across clusters:** weight proportional to mean factor score
6. Apply combined weights to next month's realised returns

### Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Lookback | 36 months | Function argument |
| Solver | SLSQP | `scipy.optimize.minimize` |
| Constraints | Long-only, fully invested | w ≥ 0, Σw = 1 |

---

## Step 7 — Output Generation

**Function:** `step_7_generate_outputs(...)`

### Generated files

| File | Description |
|------|-------------|
| `outputs/tearsheet.png` | 3×3 subplot research tearsheet |
| `outputs/plots/factor_distributions.png` | KDE + histogram per factor |
| `outputs/plots/factor_correlations.png` | Factor correlation heatmap |
| `outputs/plots/autocorrelation_decay.png` | Factor autocorrelation |
| `outputs/plots/turnover_analysis.png` | Monthly portfolio turnover |
| `outputs/plots/pca_scree.png` | Scree chart |
| `outputs/plots/cluster_scatter_2d.png` | PC1 vs PC2 clusters |
| `outputs/plots/silhouette_scores.png` | Silhouette analysis |

---

## Step 8 — Performance Summary

**Function:** `step_8_print_summary(...)`

Prints formatted ASCII tables to console:

```
══════════════════════════════════════════════════════════════════════
  EigenAlpha Phase 0 - Performance Summary
══════════════════════════════════════════════════════════════════════

+-------------------------------------------------------------+
|                  Information Coefficient (IC)               |
+------------------+----------+----------+----------+---------+
| Factor           | Mean IC  |   IR     |  t-stat  | % Pos   |
+------------------+----------+----------+----------+---------+
| momentum_12_1    |   0.0456 |   0.3214 |     2.87 |  62.4%  |
| realized_vol     |  -0.0312 |  -0.2145 |    -1.52 |  41.2%  |
| volume_trend     |   0.0234 |   0.1876 |     1.33 |  55.8%  |
+------------------+----------+----------+----------+---------+
```

---

## Step 9 — Dashboard Launch

**Function:** `step_9_launch_dashboard()`

1. Install frontend npm dependencies (if needed)
2. Start FastAPI backend on `http://localhost:8000`
3. Start Next.js frontend on `http://localhost:3000`
4. Wait for both servers to become ready
5. Open browser to `http://localhost:3000`
6. Block until Ctrl+C → graceful shutdown

---

## Configuration Reference

All configuration lives in `config.py`. Key parameters:

| Category | Parameter | Value | Description |
|----------|-----------|-------|-------------|
| **Dates** | `START_DATE` | `2014-01-01` | Backtest start |
| | `END_DATE` | `2024-12-31` | Backtest end |
| **Risk** | `RISK_FREE_RATE` | `0.065` | RBI repo rate proxy |
| | `MIN_HISTORY_MONTHS` | `24` | Minimum stock history |
| **Factors** | `MOMENTUM_LONG_WINDOW` | `252` | 12-month lookback (days) |
| | `MOMENTUM_SHORT_WINDOW` | `21` | 1-month skip (days) |
| | `VOL_WINDOW` | `20` | Volatility lookback |
| | `VOLUME_TREND_WINDOW` | `20` | Volume regression window |
| | `FACTOR_WINSOR_PERCENTILE` | `(2.5, 97.5)` | Winsorisation bounds |
| **PCA** | `N_PCA_COMPONENTS` | `50` | Max PCA components |
| | `VARIANCE_THRESHOLD` | `0.80` | Component selection |
| **Cluster** | `N_CLUSTERS` | `8` | K-Means clusters |
| | `KMEANS_N_INIT` | `20` | Random restarts |
| **Backtest** | `REBALANCE_FREQ` | `M` | Monthly rebalance |
| | `TOP_QUINTILE` | `0.20` | Top 20% |
| | `BOTTOM_QUINTILE` | `0.20` | Bottom 20% |
| **Optim** | `R_TARGET_ANNUALISED` | `0.12` | Target annual return |
| | `MAX_SINGLE_STOCK_WEIGHT` | `0.05` | 5% cap per stock |

---

## Data Flow Diagram

```
yfinance ──► DataLoader ──► MultiIndex OHLCV DataFrame
                                    │
                                    ▼
                          Preprocessor ──► Log-return matrix R (T×N)
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
               CovarianceDecomposer       FactorEngine
               Σ, eigenvalues, PCA        momentum, vol, vol_trend
                         │                     │
                         ▼                     ▼
               MarketClusterer          InformationCoefficient
               cluster_labels            IC, IR per factor
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                          MarkowitzOptimizer
                          optimal weights per cluster
                                    │
                                    ▼
                             Backtester
                          walk-forward returns
                                    │
                                    ▼
                         TearsheetGenerator
                         tearsheet.png + metrics
```

---

*See [MATH.md](MATH.md) for the mathematical derivations behind each operation.*
