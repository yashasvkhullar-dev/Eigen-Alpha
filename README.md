# EigenAlpha

**A factor-based portfolio construction engine for Indian equities.**

EigenAlpha is a systematic investment research platform that uses Principal Component Analysis to decompose the Nifty 500 covariance matrix into latent risk factors, clusters stocks by mathematical behaviour, scores them with academically grounded alpha factors, and constructs optimal portfolios using Markowitz mean-variance optimisation. It is designed as a multi-year research project — each phase builds directly on the last.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Phase](https://img.shields.io/badge/Phase-0%20%E2%80%94%20Foundation-teal)](CHANGELOG.md)
[![Status](https://img.shields.io/badge/Status-Active%20Research-orange)](CHANGELOG.md)

---

## Table of Contents

- [Background](#background)
- [Phase 0 Scope](#phase-0-scope)
- [Pipeline Architecture](#pipeline-architecture)
- [Mathematical Framework](#mathematical-framework)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Roadmap](#project-roadmap)
- [Literature](#literature)
- [Acknowledgements](#acknowledgements)

---

## Background

Professional quantitative equity research does not attempt to predict stock prices. Instead, it identifies systematic *risk factors* — persistent, measurable characteristics of stocks that are associated with above-market returns — and constructs portfolios to maximise exposure to those factors while controlling risk.

EigenAlpha implements this framework for the Indian equity market. The Nifty 500 universe is studied because it represents the full breadth of the Indian listed market, and because the application of systematic factor models to Indian equities remains an under-explored research area relative to US and European markets.

The name *EigenAlpha* reflects the core methodological contribution: using eigendecomposition of the asset return covariance matrix (PCA) to recover latent market structure, then building alpha-generating factor exposures on top of that structure.

---

## Phase 0 Scope

Phase 0 is the foundational implementation. It covers:

| Component | Description |
|---|---|
| Data pipeline | Nifty 500 OHLCV from yfinance, 10-year history, survivorship-bias documentation |
| Covariance + PCA | Σ = RᵀR/T, eigendecomposition, select top-k PCs (≥80% variance) |
| K-Means clustering | Cluster 500 stocks by return behaviour in PC space, not by sector |
| Factor engine | Momentum (12-1M), realised volatility (20D), volume trend (20D) |
| IC/IR analysis | Spearman rank IC vs forward returns, per factor, per month |
| Markowitz optimiser | min wᵀΣw, long-only, solved per cluster via scipy.optimize.SLSQP |
| Backtest engine | Walk-forward monthly rebalance, vs Nifty 500 benchmark |
| Tearsheet | Sharpe, Calmar, max drawdown, monthly heatmap, factor plots |

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAW DATA SOURCES                             │
│  yfinance (OHLCV, 10yr)          RBI macro (rates, CPI)             │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — DATA LAYER                               data/           │
│                                                                     │
│  DataLoader          Nifty 500 tickers (.NS format)                 │
│  ├── fetch()         yfinance.download(), auto_adjust=True          │
│  ├── MultiIndex      (Date, Ticker) × [O, H, L, C, V]              │
│  └── parquet cache   avoid re-downloading on reruns                 │
│                                                                     │
│  Preprocessor                                                       │
│  ├── log_returns()   ln(Pt / Pt-1)                                  │
│  ├── winsorise()     clip at 2.5th / 97.5th percentile per date     │
│  └── zscore()        cross-sectional: subtract mean, divide by std  │
│                                                                     │
│  Output: R matrix ∈ ℝ^(T×N)   rows=dates, cols=tickers             │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — COVARIANCE + PCA                         pca/            │
│                                                                     │
│  CovarianceDecomposer                                               │
│  ├── Σ = (1/T) RᵀR              sample covariance matrix (N×N)     │
│  ├── eigh(Σ) → λ, V             eigenvalues + eigenvectors          │
│  ├── PCA(n=50).fit(R)           scikit-learn PCA                    │
│  └── select k: cumvar ≥ 80%     typically k ≈ 8–15 components      │
│                                                                     │
│  Interpretation: each PC = a latent market risk factor              │
│  PC1 ≈ market beta, PC2 ≈ size, PC3 ≈ momentum...                  │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — CLUSTERING                               pca/            │
│                                                                     │
│  MarketClusterer                                                     │
│  ├── project stocks onto top-3 PCs   (N × 3) score matrix          │
│  ├── KMeans(k=8, n_init=20)          cluster by return behaviour    │
│  ├── silhouette_analysis()           validate k choice              │
│  └── cluster_vs_sector_heatmap()     show clusters ≠ GICS sectors  │
│                                                                     │
│  Key insight: stocks in the same cluster share risk exposure,       │
│  not business model. A bank and a commodity stock may cluster       │
│  together if they respond similarly to macro shocks.                │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4 — FACTOR ENGINE                            factors/        │
│                                                                     │
│  FactorEngine                                                       │
│  ├── momentum_12_1()     (Pt-1M / Pt-12M) - 1   per month-end      │
│  │   skips last month to avoid short-term reversal                  │
│  ├── realized_vol_20d()  std(log_ret, 20D) × √252  annualised      │
│  └── volume_trend_20d()  OLS slope of log(volume) over 20D         │
│                                                                     │
│  All factors: winsorised at 2.5/97.5 pct + cross-sectional z-score │
│                                                                     │
│  InformationCoefficient                                             │
│  ├── IC(t) = SpearmanCorr(factor(t), fwd_return(t))                │
│  ├── mean_IC, IC_std, IR = mean_IC / IC_std                        │
│  └── IC decay plot (lags 1M–6M) — measures signal persistence      │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 5 — OPTIMISATION + BACKTEST                  portfolio/      │
│                                                                     │
│  MarkowitzOptimizer                                                  │
│  ├── per cluster: min wᵀΣw   s.t. wᵀμ ≥ r*, Σw=1, w≥0            │
│  ├── cluster weights ∝ mean factor score of cluster                 │
│  └── efficient_frontier()  (return, volatility, Sharpe) curve      │
│                                                                     │
│  Backtester (walk-forward — no lookahead)                           │
│  ├── quintile_backtest()   Q5 vs Q1 long-short per factor          │
│  └── markowitz_backtest()  monthly rebalance, out-of-sample        │
│                                                                     │
│  TearsheetGenerator                                                 │
│  └── 3×3 subplot figure: cumulative returns, drawdown, IC plots,   │
│      PCA scatter, weights, monthly heatmap, performance table       │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

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

See [MATH.md](docs/MATH.md) for full derivations. Brief overview:

**Covariance matrix:**
$$\Sigma = \frac{1}{T} R^T R, \quad R \in \mathbb{R}^{T \times N}$$

**Eigendecomposition:**
$$\Sigma = V \Lambda V^T, \quad \Lambda = \text{diag}(\lambda_1, \ldots, \lambda_N), \quad \lambda_1 \geq \lambda_2 \geq \ldots$$

**Momentum factor (Jegadeesh & Titman, 1993):**
$$\text{MOM}_{i,t} = \frac{P_{i,t-1}}{P_{i,t-12}} - 1$$

**Realised volatility:**
$$\text{RVOL}_{i,t} = \sqrt{252} \cdot \text{std}\left(\log\frac{P_{i,\tau}}{P_{i,\tau-1}}\right)_{\tau=t-20}^{t}$$

**Markowitz optimisation:**
$$\min_{w} \; w^T \Sigma w \quad \text{s.t.} \quad w^T \mu \geq r^*, \; \mathbf{1}^T w = 1, \; w \geq 0$$

**Information Coefficient:**
$$\text{IC}_t = \rho_S\left(f_t, r_{t+1}\right), \quad \text{IR} = \frac{\overline{\text{IC}}}{\sigma_{\text{IC}}}$$

---

## Results

*Results populated after running `python pipeline.py`. Placeholder targets:*

| Metric | EigenAlpha Phase 0 | Nifty 500 Benchmark |
|---|---|---|
| Ann. Return | — | — |
| Ann. Volatility | — | — |
| Sharpe Ratio | — | — |
| Max Drawdown | — | — |
| Calmar Ratio | — | — |
| Momentum IC (mean) | — | — |
| Momentum IR | — | — |

---

## Installation

```bash
git clone https://github.com/yourusername/eigenalpha.git
cd eigenalpha
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Python 3.10+ required.**

---

## Usage

```bash
# Run the full pipeline
python pipeline.py

# Outputs saved to outputs/:
#   tearsheet.png        — full research tearsheet
#   metrics.json         — all performance metrics
#   factor_data.parquet  — computed factor scores
#   weights.csv          — final portfolio weights
```

```python
# Use individual modules
from data.loader import DataLoader
from factors.engine import FactorEngine
from pca.decompose import CovarianceDecomposer

loader = DataLoader()
prices, volumes = loader.fetch(tickers=NIFTY500_TICKERS, start="2014-01-01")

engine = FactorEngine(prices=prices, volumes=volumes)
factor_data = engine.compute_all()

decomposer = CovarianceDecomposer(return_matrix)
decomposer.fit_pca()
decomposer.plot_scree()
```

---

## Project Roadmap

| Phase | Semester | Key Addition |
|---|---|---|
| **0 — Foundation** | Sem 2 | PCA + clustering + 3 factors + Markowitz (this repo) |
| 1 — Factor expansion | Sem 3–4 | 10+ factors, Fama-MacBeth regression, Black-Litterman |
| 2 — ML alpha | Sem 5–6 | XGBoost with purged CV, HMM regime detection |
| 3 — Live trading | Year 3 | Zerodha Kite API, Kelly position sizing, NLP earnings |

See [ROADMAP.md](docs/ROADMAP.md) for detailed specifications.

---

## Literature

| Paper | Relevance |
|---|---|
| Markowitz (1952). Portfolio Selection. *J. Finance* | Mean-variance optimisation foundation |
| Fama & French (1993). Common risk factors in returns. *J. Financial Economics* | Factor model framework |
| Jegadeesh & Titman (1993). Returns to buying winners. *J. Finance* | Momentum factor justification |
| Connor & Korajczyk (1988). Risk and return in APT. *J. Financial Economics* | PCA as statistical factor model |
| Asness, Moskowitz & Pedersen (2013). Value and momentum everywhere. *J. Finance* | Cross-asset factor validation |
| López de Prado (2018). *Advances in Financial Machine Learning*. Wiley | Walk-forward CV, purged backtesting |

---

## Acknowledgements

EigenAlpha is a long-term research project maintained as part of B.Tech (AI/ML) studies at SIT Pune. It is designed to compound in depth and rigour each semester, with the goal of producing institutional-grade systematic research on Indian equities.

---

*Phase 0 — Initial Release. See [CHANGELOG.md](CHANGELOG.md) for version history.*
