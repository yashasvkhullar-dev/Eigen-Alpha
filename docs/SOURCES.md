# EigenAlpha — Sources & References

**Version:** Phase 0  
**Last Updated:** 2025-04

This document consolidates all academic papers, data sources, and software dependencies used in the EigenAlpha project. Every formula implemented in the codebase can be traced back to a specific reference below.

---

## Table of Contents

1. [Academic Papers](#academic-papers)
   - [Portfolio Theory & Optimisation](#portfolio-theory--optimisation)
   - [Factor Models & Asset Pricing](#factor-models--asset-pricing)
   - [Momentum & Return Anomalies](#momentum--return-anomalies)
   - [Volatility & Risk](#volatility--risk)
   - [Statistical Methods & PCA](#statistical-methods--pca)
   - [Machine Learning in Finance](#machine-learning-in-finance)
   - [Backtesting Methodology](#backtesting-methodology)
2. [Textbooks](#textbooks)
3. [Data Sources](#data-sources)
4. [Software Dependencies](#software-dependencies)
5. [Citation Map](#citation-map)

---

## Academic Papers

### Portfolio Theory & Optimisation

**Markowitz, H. (1952).** Portfolio Selection. *Journal of Finance*, 7(1), 77–91.  
→ Foundation of mean-variance optimisation. Defines the efficient frontier and the minimum-variance portfolio problem solved in `portfolio/optimizer.py`.

**Chopra, V. & Ziemba, W. (1993).** The effect of errors in means, variances, and covariances on optimal portfolio choice. *Journal of Portfolio Management*, 19(2), 6–11.  
→ Demonstrates that estimation errors in expected returns and covariances are amplified by the Markowitz optimiser, motivating our cluster-based approach.

**Ledoit, O. & Wolf, M. (2004).** Honey, I Shrunk the Sample Covariance Matrix. *Journal of Portfolio Management*, 30(4), 110–119.  
→ Proposes shrinkage estimators for covariance matrices. Referenced in `portfolio/optimizer.py` as motivation for per-cluster (smaller-dimensional) optimisation.

**DeMiguel, V., Garlappi, L. & Uppal, R. (2009).** Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio? *Review of Financial Studies*, 22(5), 1915–1953.  
→ Shows that naive equal-weight portfolios often outperform optimised portfolios out-of-sample due to estimation error. Motivates our regularisation via clustering.

---

### Factor Models & Asset Pricing

**Fama, E. & French, K. (1993).** Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56.  
→ The Fama-French 3-factor model. Defines the factor model framework used throughout `factors/engine.py`.

**Ross, S. (1976).** The arbitrage theory of capital asset pricing. *Journal of Economic Theory*, 13(3), 341–360.  
→ Arbitrage Pricing Theory (APT). Provides the theoretical link between PCA and latent factor models used in `pca/decompose.py`.

**Connor, G. & Korajczyk, R. (1988).** Risk and return in an equilibrium APT: Application of a new test methodology. *Journal of Financial Economics*, 21(2), 255–289.  
→ Uses PCA as a statistical factor model to estimate latent risk factors. Directly motivates our `CovarianceDecomposer` approach.

**Asness, C., Moskowitz, T. & Pedersen, L. (2013).** Value and Momentum Everywhere. *Journal of Finance*, 68(3), 929–985.  
→ Validates momentum and value factors across multiple asset classes and geographies. Supports applying momentum to Indian equities.

**Grinold, R. C. & Kahn, R. N. (2000).** *Active Portfolio Management*, 2nd ed. McGraw-Hill.  
→ Defines the IC/IR framework used in `factors/ic_analysis.py`. The fundamental law of active management: IR ≈ IC × √(breadth).

---

### Momentum & Return Anomalies

**Jegadeesh, N. & Titman, S. (1993).** Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency. *Journal of Finance*, 48(1), 65–91.  
→ The foundational momentum paper. Documents the 12-1 month momentum effect. Directly implemented in `FactorEngine.momentum_12_1()`.

**Jegadeesh, N. (1990).** Evidence of Predictable Behavior of Security Returns. *Journal of Finance*, 45(3), 881–898.  
→ Documents short-term reversal (1-month), which is why the momentum factor skips the most recent month.

**Chan, L., Jegadeesh, N. & Lakonishok, J. (1996).** Momentum Strategies. *Journal of Finance*, 51(5), 1681–1713.  
→ Extends momentum analysis to include earnings revision factors. Referenced in the Phase 1 roadmap.

---

### Volatility & Risk

**Ang, A., Hodrick, R. J., Xing, Y. & Zhang, X. (2006).** The Cross-Section of Volatility and Expected Returns. *Journal of Finance*, 61(1), 259–299.  
→ Documents the idiosyncratic volatility puzzle. Referenced in `FactorEngine.realized_volatility_20d()`.

**Blitz, D. & van Vliet, P. (2007).** The Volatility Effect: Lower Risk Without Lower Return. *Journal of Portfolio Management*, 34(1), 102–113.  
→ Documents the low-volatility anomaly: low-vol stocks deliver higher risk-adjusted returns than high-vol stocks.

**Baker, M., Bradley, B. & Wurgler, J. (2011).** Benchmarks as Limits to Arbitrage: Understanding the Low-Volatility Anomaly. *Financial Analysts Journal*, 67(1), 40–54.  
→ Explains the low-volatility anomaly via institutional constraints. Supports using realised volatility as a factor.

---

### Statistical Methods & PCA

**Laloux, L., Cizeau, P., Bouchaud, J.-P. & Potters, M. (1999).** Noise Dressing of Financial Correlation Matrices. *Physical Review Letters*, 83(7), 1467.  
→ Random Matrix Theory applied to financial correlations. Shows that most eigenvalues of large correlation matrices are noise. Referenced in `pca/decompose.py`.

**Marchenko, V. A. & Pastur, L. A. (1967).** Distribution of Eigenvalues for Some Sets of Random Matrices. *Mathematics of the USSR-Sbornik*, 1(4), 457–483.  
→ The Marchenko-Pastur distribution for eigenvalues of random matrices. Used to distinguish signal from noise in eigenvalue analysis.

**Rousseeuw, P. J. (1987).** Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.  
→ Defines the silhouette coefficient used in `MarketClusterer.silhouette_analysis()` to choose optimal k.

**Ahn, S. & Horenstein, A. R. (2013).** Eigenvalue Ratio Test for the Number of Factors. *Econometrica*, 81(3), 1203–1227.  
→ Statistical test for determining the number of factors in a panel. Referenced in `pca/cluster.py`.

---

### Volume & Liquidity

**Campbell, J. Y., Grossman, S. J. & Wang, J. (1993).** Trading Volume and Serial Correlation in Stock Returns. *Quarterly Journal of Economics*, 108(4), 905–939.  
→ Establishes the relationship between trading volume and return predictability. Referenced in `FactorEngine.volume_trend_20d()`.

---

### Machine Learning in Finance

**López de Prado, M. (2018).** *Advances in Financial Machine Learning*. Wiley.  
→ Walk-forward / purged cross-validation methodology. Referenced in backtesting design and planned for Phase 2 (XGBoost with purged CV).

---

### Backtesting Methodology

**Sharpe, W. F. (1966).** Mutual Fund Performance. *Journal of Business*, 39(1), 119–138.  
→ Defines the Sharpe ratio used in `Backtester.compute_metrics()`.

**Jensen, M. C. (1968).** The Performance of Mutual Funds in the Period 1945–1964. *Journal of Finance*, 23(2), 389–416.  
→ Defines Jensen's alpha (CAPM alpha). Used in `Backtester.compute_benchmark_relative_metrics()`.

**Elton, E. J., Gruber, M. J. & Blake, C. R. (1996).** Survivorship Bias and Mutual Fund Performance. *Review of Financial Studies*, 9(4), 1097–1120.  
→ Documents survivorship bias in performance studies. Referenced in the disclaimer in `data/universe.py`.

---

## Textbooks

| Book | Author(s) | Relevance |
|------|-----------|-----------|
| *Convex Optimization* | Boyd, S. & Vandenberghe, L. (2004) | QP formulation of Markowitz (free at stanford.edu/~boyd/cvxbook) |
| *Active Portfolio Management*, 2nd ed. | Grinold, R. C. & Kahn, R. N. (2000) | IC/IR framework, fundamental law of active management |
| *Quantitative Equity Portfolio Management* | Qian, E., Hua, R. & Sorensen, E. (2007) | Factor evaluation methodology |
| *Options, Futures, and Other Derivatives*, 10th ed. | Hull, J. C. (2018) | Log-return computation definition |
| *Advances in Financial Machine Learning* | López de Prado, M. (2018) | Walk-forward CV, purged backtesting |

---

## Data Sources

### Market Data

| Source | Data | Usage | Module |
|--------|------|-------|--------|
| **Yahoo Finance** (via `yfinance`) | OHLCV daily prices | Price data for ~120 Nifty 500 stocks | `data/loader.py` |
| **Yahoo Finance** (via `yfinance`) | `^NSEI` index | Nifty 50 benchmark prices | `data/loader.py` |

### Reference Data

| Source | Data | Usage |
|--------|------|-------|
| **NSE India** | Nifty 500 constituent list | Universe definition (`data/universe.py`) |
| **RBI** | Repo rate (6.5%) | Risk-free rate proxy (`config.py`) |

### Data Limitations

- **Survivorship bias:** current Nifty 500 constituents only; historical delistings excluded
- **Adjusted prices:** `auto_adjust=True` in yfinance corrects for dividends and splits
- **Missing data:** stocks with < 24 months of history are excluded
- **Frequency:** daily prices, resampled to monthly for factor computation

---

## Software Dependencies

### Core

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥ 2.0.0 | Data manipulation, time series |
| `numpy` | ≥ 1.26.0 | Numerical computation |
| `scipy` | ≥ 1.12.0 | SLSQP optimiser, linear regression, Spearman correlation |
| `scikit-learn` | ≥ 1.4.0 | PCA, K-Means, silhouette score |
| `yfinance` | ≥ 0.2.36 | Market data download |
| `pyarrow` | ≥ 15.0.0 | Parquet serialisation |

### Visualisation

| Package | Version | Purpose |
|---------|---------|---------|
| `matplotlib` | ≥ 3.8.0 | Tearsheet, scree chart, IC plots |
| `seaborn` | ≥ 0.13.0 | Heatmaps, styled plots |

### API & Frontend

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | ≥ 0.110.0 | REST API for dashboard |
| `uvicorn` | ≥ 0.29.0 | ASGI server |
| Next.js | 14.x | Frontend dashboard |
| React | 18.x | UI components |
| Recharts | 2.x | Chart components |
| Tailwind CSS | 3.x | Styling |

### Development

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | ≥ 8.0.0 | Unit testing |
| `pytest-cov` | ≥ 4.1.0 | Coverage reporting |
| `black` | ≥ 24.0.0 | Code formatting |
| `flake8` | ≥ 7.0.0 | Linting |

---

## Citation Map

Which paper is cited where in the codebase:

| Paper | Module | Function / Line |
|-------|--------|-----------------|
| Markowitz (1952) | `portfolio/optimizer.py` | `optimize_cluster()` |
| Jegadeesh & Titman (1993) | `factors/engine.py` | `momentum_12_1()` |
| Ang et al. (2006) | `factors/engine.py` | `realized_volatility_20d()` |
| Campbell, Grossman & Wang (1993) | `factors/engine.py` | `volume_trend_20d()` |
| Connor & Korajczyk (1988) | `pca/decompose.py` | `CovarianceDecomposer` class |
| Laloux et al. (1999) | `pca/decompose.py` | `compute_covariance()` |
| Marchenko & Pastur (1967) | `pca/decompose.py` | `eigendecompose()` |
| Rousseeuw (1987) | `pca/cluster.py` | `silhouette_analysis()` |
| Grinold & Kahn (2000) | `factors/ic_analysis.py` | `compute_ic()`, `ic_summary()` |
| Fama & French (1993) | `portfolio/backtest.py` | `run_quintile_backtest()` |
| Sharpe (1966) | `portfolio/backtest.py` | `compute_metrics()` |
| Jensen (1968) | `portfolio/backtest.py` | `compute_benchmark_relative_metrics()` |
| Elton, Gruber & Blake (1996) | `data/universe.py` | Survivorship bias disclaimer |
| López de Prado (2018) | `portfolio/backtest.py` | Walk-forward methodology |

---

*All citations follow APA format. For the mathematical derivations behind these references, see [MATH.md](MATH.md).*
