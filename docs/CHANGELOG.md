# Changelog

All notable changes to EigenAlpha are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`  
- MAJOR: complete phase change (new modelling paradigm)  
- MINOR: new factor, new model, or new module within a phase  
- PATCH: bug fixes, performance improvements, documentation updates

---

## [Unreleased]

*Features in development for Phase 1 (Semester 3–4):*
- Fama-MacBeth cross-sectional regression for factor premium estimation
- Black-Litterman portfolio construction (replaces basic Markowitz)
- Expansion from 3 to 10+ factors: value (P/B), quality (ROE), low-beta, earnings revision
- Transaction cost and slippage modelling in backtester
- alphalens tearsheet integration

---

## [0.1.0] — Phase 0 Initial Release — 2025-04

### Added

**Data layer (`data/`)**
- `DataLoader` class: fetches Nifty 500 OHLCV from yfinance with `auto_adjust=True`
- MultiIndex DataFrame output: `(Date, Ticker)` × `[Open, High, Low, Close, Volume]`
- Parquet serialisation for caching (avoids repeated API calls)
- Benchmark fetcher: Nifty 50 index (`^NSEI`) as performance reference
- Survivorship bias documentation: explicit module-level acknowledgement that the ticker list represents current Nifty 500 constituents and excludes historically delisted stocks
- `Preprocessor` class: log-return computation, winsorisation (2.5/97.5 pct), cross-sectional z-score
- Return matrix builder: pivots to wide format R ∈ ℝ^(T×N)

**PCA layer (`pca/`)**
- `CovarianceDecomposer` class: Σ computation, `np.linalg.eigh()` eigendecomposition
- Eigenvalue sorting (descending), eigenvector storage
- `sklearn.decomposition.PCA` integration with explained variance tracking
- Automatic component selection: minimum k such that cumulative EVR ≥ 80%
- Scree chart plotting: bar chart of per-component variance + cumulative line + 80% threshold marker
- `MarketClusterer` class: projects stocks onto top-3 PCs
- K-Means clustering (k=8, k-means++ init, n_init=20)
- Silhouette score analysis across k ∈ [3, 14] for optimal k selection
- 2D cluster scatter plot (PC1 vs PC2, coloured by cluster)
- Cluster vs GICS sector cross-tabulation heatmap

**Factor engine (`factors/`)**
- `FactorEngine` class with three academically grounded alpha factors:
  - `momentum_12_1()`: 12-1 month price momentum (Jegadeesh & Titman, 1993)
  - `realized_volatility_20d()`: 20-day annualised realised volatility
  - `volume_trend_20d()`: OLS slope of log-volume over trailing 20 days
- Automatic winsorisation and cross-sectional z-scoring per factor
- alphalens-compatible output format
- `InformationCoefficient` class:
  - Monthly Spearman IC computation per factor
  - IC summary statistics: mean IC, IC std, IR, t-statistic, % positive months
  - IC time-series bar chart with 6M rolling mean overlay
  - IC decay curve (lags 1M–6M) to measure signal persistence

**Portfolio layer (`portfolio/`)**
- `MarkowitzOptimizer` class:
  - Per-cluster quadratic programme: min wᵀΣw s.t. wᵀμ ≥ r*, Σw=1, w≥0
  - Solved via `scipy.optimize.minimize(method='SLSQP')`
  - Cluster-level weights proportional to mean composite factor score
  - Efficient frontier computation (100-point return-risk tradeoff curve)
- `Backtester` class:
  - Quintile backtest: Q5 vs Q1 long-short per factor, monthly rebalance
  - Markowitz backtest: walk-forward, no lookahead bias, monthly rebalance
  - Performance metrics: annualised return, annualised volatility, Sharpe ratio, max drawdown, Calmar ratio, win rate
- `TearsheetGenerator` class:
  - 3×3 subplot professional tearsheet (12×18 inches, 150 DPI)
  - Panels: cumulative returns, drawdown, monthly heatmap, Q1–Q5 quintile returns, IC timeseries (×2), scree chart, cluster scatter, portfolio weights

**Visualisation (`visualisation/`)**
- `EDADashboard` class: factor distributions (KDE + histogram), factor correlation heatmap, autocorrelation decay curves, monthly turnover analysis

**Infrastructure**
- `config.py`: centralised constants (tickers, dates, hyperparameters)
- `pipeline.py`: end-to-end runner with parquet cache check
- `requirements.txt`: pinned versions for reproducibility
- `tests/`: pytest unit tests for loader, factor engine, and optimizer
- Full Google-style docstrings on all public methods
- Python type hints throughout
- `logging` module integration (no bare print statements)

### Technical Notes

- Universe: Nifty 500 (current constituents, ~480 liquid tickers in `.NS` format)
- History: 2014-01-01 to 2024-12-31 (10 years, ~2500 daily observations)
- Rebalance frequency: monthly (month-end)
- Risk-free rate proxy: 6.5% (RBI repo rate, annualised)
- Minimum stock history: 24 months (stocks with less data excluded from backtest)

### Known Limitations

- **Survivorship bias**: the Nifty 500 constituent list reflects current membership. Stocks that were delisted, acquired, or demoted between 2014–2024 are not included. This biases backtested performance upward. Acknowledged and documented in `data/universe.py`. Addressed in Phase 1 via a historical constituent database.
- **Transaction costs**: not modelled. Monthly rebalancing of a 500-stock portfolio incurs meaningful costs in practice. Addressed in Phase 1.
- **Slippage**: not modelled. Assumes execution at month-end close. Addressed in Phase 1.
- **Estimation error**: Markowitz optimiser is susceptible to covariance estimation noise. Partially mitigated by cluster-level optimisation. Addressed in Phase 1 via Black-Litterman.
- **Single-country risk**: universe is limited to Indian equities (NSE). No international diversification.

---

## Version History Summary

| Version | Date | Phase | Description |
|---|---|---|---|
| 0.1.0 | 2025-04 | Phase 0 | Initial release: full pipeline from data to tearsheet |
| 0.2.0 | *planned* | Phase 1 | Factor expansion + Fama-MacBeth + Black-Litterman |
| 1.0.0 | *planned* | Phase 2 | ML alpha: XGBoost + HMM regime detection |
| 2.0.0 | *planned* | Phase 3 | Live paper trading via Zerodha Kite API |

---

*Maintained by the EigenAlpha Research project. All changes are committed to Git with descriptive messages referencing this changelog.*
