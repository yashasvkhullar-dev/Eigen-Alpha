<div align="center">

# EigenAlpha

### Factor-Based Portfolio Construction for Indian Equities

**PCA · K-Means Clustering · Markowitz Optimisation · Walk-Forward Backtesting**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

## Overview

EigenAlpha is a systematic investment research platform that decomposes the Indian equity market (Nifty 500 universe) into latent risk factors using **Principal Component Analysis**, clusters stocks by return behaviour, applies academic alpha factors, and constructs optimal portfolios using **Markowitz mean-variance optimisation**.

Built as a rigorous, end-to-end quantitative research pipeline — from raw data to professional tearsheet — with every formula grounded in peer-reviewed finance literature.

### Key Features

- **PCA Decomposition** — Eigenvalue analysis of the covariance matrix to identify the dominant risk factors driving Indian equity returns
- **Behaviour-Based Clustering** — K-Means clustering in PC space reveals groups of stocks that move together, often cutting across traditional sector boundaries
- **Three Academic Alpha Factors** — Momentum (Jegadeesh & Titman, 1993), Realised Volatility (Ang et al., 2006), Volume Trend (Campbell et al., 1993)
- **IC/IR Factor Evaluation** — Spearman Information Coefficient, Information Ratio, t-statistics, and IC decay analysis
- **Cluster-Based Markowitz** — Per-cluster quadratic programming with factor-weighted inter-cluster allocation
- **Walk-Forward Backtesting** — No look-ahead bias, monthly rebalancing, with comprehensive performance metrics
- **Professional Tearsheet** — Publication-quality 3×3 subplot research summary
- **Live Dashboard** — FastAPI + Next.js full-stack dashboard for interactive analysis

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yashasvkhullar-dev/Eigen-Alpha.git
cd Eigen-Alpha

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python pipeline.py
```

The pipeline will:
1. Download ~120 Nifty 500 stocks (10 years of daily OHLCV data)
2. Compute log returns and build the return matrix
3. Run PCA and K-Means clustering
4. Compute alpha factors and IC analysis
5. Run quintile and walk-forward Markowitz backtests
6. Generate a professional tearsheet at `outputs/tearsheet.png`
7. Launch the dashboard at `http://localhost:3000`

---

## Project Structure

```
eigenalpha/
├── pipeline.py              # End-to-end orchestrator (run this)
├── config.py                # All hyperparameters and constants
├── api.py                   # FastAPI backend for dashboard
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Package metadata
├── Makefile                 # Developer commands
│
├── data/                    # Data loading and preprocessing
│   ├── loader.py            # yfinance downloader with parquet cache
│   ├── preprocessor.py      # Log returns, winsorisation, z-scoring
│   └── universe.py          # Nifty 500 ticker list + sector mapping
│
├── pca/                     # Eigendecomposition and clustering
│   ├── decompose.py         # Covariance → eigenvalues → PCA
│   └── cluster.py           # K-Means on PC scores, silhouette analysis
│
├── factors/                 # Alpha factor computation
│   ├── engine.py            # Momentum, volatility, volume trend
│   └── ic_analysis.py       # IC, IR, IC decay evaluation
│
├── portfolio/               # Portfolio construction and evaluation
│   ├── optimizer.py         # Cluster-based Markowitz (SLSQP)
│   ├── backtest.py          # Walk-forward + quintile backtests
│   └── tearsheet.py         # 3×3 professional research summary
│
├── visualisation/           # EDA and diagnostic plots
│   ├── eda.py               # Factor distributions, correlations
│   └── pca_plots.py         # Scree chart, cluster scatter
│
├── tests/                   # Unit tests (pytest)
│   ├── test_factors.py      # Factor computation validation
│   ├── test_loader.py       # Data pipeline tests
│   └── test_optimizer.py    # Optimisation constraint tests
│
├── notebooks/               # Jupyter notebooks for exploration
│   ├── 00_eda.ipynb
│   └── 01_full_pipeline.ipynb
│
├── frontend/                # Next.js dashboard
│   ├── app/                 # App router pages
│   └── components/          # React chart components
│
├── docs/                    # Documentation
│   ├── MATH.md              # Full mathematical derivations
│   ├── PIPELINE.md          # Step-by-step pipeline walkthrough
│   ├── SOURCES.md           # Academic references & data sources
│   ├── CHANGELOG.md         # Version history
│   ├── CONTRIBUTING.md      # Coding standards
│   └── ROADMAP.md           # Multi-phase development plan
│
└── outputs/                 # Generated outputs (gitignored)
    └── .gitkeep
```

---

## Pipeline Architecture

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

## Mathematical Framework

Every formula in the codebase traces back to a peer-reviewed source. Full derivations: [`docs/MATH.md`](docs/MATH.md)

### PCA & Eigendecomposition

The sample covariance matrix of log returns is decomposed as:

$$\Sigma = V \Lambda V^T$$

where $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_N)$ contains eigenvalues (sorted descending) and $V$ holds the corresponding eigenvectors. The top $k^*$ components (where cumulative variance $\geq 80\%$) capture the systematic risk structure.

### Alpha Factors

| Factor | Formula | Reference |
|--------|---------|-----------|
| **Momentum (12-1)** | $\text{MOM}_{i,t} = P_{i,t-1M} / P_{i,t-12M} - 1$ | Jegadeesh & Titman (1993) |
| **Realised Vol** | $\text{RVOL}_{i,t} = \sqrt{252} \cdot \sigma(\ell_{i})_{20D}$ | Ang et al. (2006) |
| **Volume Trend** | $\hat{\beta}_i$ from OLS: $\log V_{i,\tau} = \alpha + \beta \tau$ | Campbell et al. (1993) |

### Markowitz Optimisation

$$\min_{w} \; w^T \Sigma w \quad \text{s.t.} \quad w^T \mu \geq r^*, \; \mathbf{1}^T w = 1, \; w_i \geq 0$$

Solved per-cluster via SLSQP to avoid estimation error amplification in the full $N \times N$ covariance matrix.

---

## Configuration

All hyperparameters live in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `START_DATE` | `2014-01-01` | Backtest start |
| `END_DATE` | `2024-12-31` | Backtest end |
| `N_CLUSTERS` | `8` | K-Means clusters |
| `VARIANCE_THRESHOLD` | `0.80` | PCA component selection |
| `RISK_FREE_RATE` | `0.065` | RBI repo rate proxy |
| `MOMENTUM_LONG_WINDOW` | `252` | 12-month lookback (trading days) |
| `VOL_WINDOW` | `20` | Volatility lookback (trading days) |
| `MAX_SINGLE_STOCK_WEIGHT` | `0.05` | 5% per-stock cap |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## Dashboard

The integrated dashboard provides real-time visualisation of pipeline outputs:

```bash
# Start API + frontend
python pipeline.py  # launches both automatically

# Or manually:
uvicorn api:app --port 8000 &   # backend
cd frontend && npm run dev      # frontend at localhost:3000
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/MATH.md`](docs/MATH.md) | Full mathematical derivations for every formula |
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | Step-by-step pipeline walkthrough |
| [`docs/SOURCES.md`](docs/SOURCES.md) | Academic references, data sources, citation map |
| [`docs/CHANGELOG.md`](docs/CHANGELOG.md) | Version history |
| [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) | Coding standards and commit conventions |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Multi-phase development plan (3 years) |

---

## Known Limitations

- **Survivorship bias**: uses current Nifty 500 constituents; historically delisted stocks are excluded. Documented in [`data/universe.py`](data/universe.py).
- **No transaction costs**: monthly rebalancing costs are not modelled (planned for Phase 1).
- **Single country**: Indian equities only (NSE).

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for planned improvements: Black-Litterman, XGBoost alpha, HMM regime detection, and live paper trading via Zerodha Kite API.

---

## Academic References

Core papers (full list: [`docs/SOURCES.md`](docs/SOURCES.md)):

- Markowitz, H. (1952). *Portfolio Selection.* Journal of Finance
- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers.* Journal of Finance
- Connor, G. & Korajczyk, R. (1988). *Risk and Return in an Equilibrium APT.* JFE
- Ang, A. et al. (2006). *The Cross-Section of Volatility and Expected Returns.* Journal of Finance
- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management.* McGraw-Hill

---

## License

[MIT](LICENSE) — see [`LICENSE`](LICENSE) for details.
