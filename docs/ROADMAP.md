# EigenAlpha — Research Roadmap

This document describes the planned evolution of EigenAlpha across three years of development. Each phase builds directly on the previous one — the codebase is extended, never replaced.

---

## Phase 0 — Foundation (Current)
**Semester 2 | v0.1.0**

The mathematical and engineering bedrock. Everything downstream depends on these modules being correct and well-tested.

**Delivered:**
- Data pipeline: Nifty 500 OHLCV, 10yr history, survivorship-bias documentation
- Covariance matrix + PCA eigendecomposition → latent risk factors
- K-Means clustering of stocks in PC space (behaviour-based, not sector-based)
- Factor engine: 3 alpha factors (momentum 12-1M, realised vol 20D, volume trend 20D)
- IC/IR analysis: Spearman IC, mean IC, IR, IC decay curves
- Markowitz optimiser: per-cluster QP via scipy.optimize.SLSQP
- Walk-forward backtest engine + professional tearsheet
- Full test suite, type hints, Google-style docstrings

**Academic grounding:** Markowitz (1952), Jegadeesh & Titman (1993), Connor & Korajczyk (1988)

---

## Phase 1 — Rigorous Factor Research
**Semester 3–4 | v0.2.x**

Expand the factor library, replace estimation-error-prone Markowitz with Black-Litterman, and add survivorship-bias-free data.

### 1.1 Factor expansion (10+ factors)

| Factor | Computation | Academic source |
|---|---|---|
| Value (P/B) | Book value / Market cap, cross-sectionally ranked | Fama & French (1993) |
| Quality (ROE) | Return on equity from quarterly BSE filings | Novy-Marx (2013) |
| Low beta | Rolling 252D beta to Nifty 50, negated | Frazzini & Pedersen (2014) |
| Earnings revision | % change in analyst EPS estimates month-over-month | Chan, Jegadeesh & Lakonishok (1996) |
| Carry (dividend yield) | Trailing 12M dividend / current price | Koijen et al. (2018) |
| Short-term reversal | 1M return, negated | Jegadeesh (1990) |
| Idiosyncratic vol | CAPM residual volatility over 60D | Ang et al. (2006) |

### 1.2 Fama-MacBeth regression

Replace IC analysis with a statistically rigorous cross-sectional factor premium estimation:

Step 1 (time-series): For each stock $i$, regress returns on factor exposures:
$$r_{i,t} = \alpha_i + \sum_k \beta_{i,k} F_{k,t} + \varepsilon_{i,t}$$

Step 2 (cross-section): Each period $t$, regress cross-sectional returns on betas:
$$r_{i,t} = \gamma_{0,t} + \sum_k \gamma_{k,t} \hat{\beta}_{i,k} + u_{i,t}$$

The time-series average $\bar{\gamma}_k$ is the factor premium. Standard errors are computed from the time-series of $\hat{\gamma}_{k,t}$ (Fama & MacBeth, 1973 — corrects for cross-sectional correlation).

**New module:** `factors/fama_macbeth.py` → class `FamaMacBeth`

### 1.3 Black-Litterman portfolio construction

Replace Markowitz with Black-Litterman (Black & Litterman, 1992):

1. **Prior:** market equilibrium returns implied by current market cap weights: $\Pi = \delta \Sigma w_{mkt}$
2. **Views:** our factor IC scores generate quantitative views: $P\mu = Q + \varepsilon$, $\varepsilon \sim N(0, \Omega)$
3. **Posterior:** Bayes update combining prior and views: $\mu_{BL} = [(\tau\Sigma)^{-1} + P^T\Omega^{-1}P]^{-1}[(\tau\Sigma)^{-1}\Pi + P^T\Omega^{-1}Q]$

**Advantage over Markowitz:** Black-Litterman starts from a stable equilibrium prior, so optimised weights are not dominated by estimation error. The factor scores feed naturally as "views" on return expectations.

**New module:** `portfolio/black_litterman.py` → class `BlackLittermanOptimizer`

### 1.4 Survivorship-bias-free universe

Replace static ticker list with historical Nifty 500 constituent data sourced from NSE. Include stocks that were delisted, demoted, or acquired during 2014–2024. This is methodologically necessary for a credible backtest.

**New file:** `data/historical_universe.py` with date-indexed constituent changes.

### 1.5 Transaction cost modelling

Add bid-ask spread and market impact costs to the backtest:

$$\text{TC}_t = \sum_i |w_{i,t} - w_{i,t-1}| \cdot c_i$$

where $c_i$ is the estimated one-way transaction cost for stock $i$ (spread + impact, estimated from average daily volume and volatility). This will reduce backtested returns by 1–2% annualised — producing a more honest performance estimate.

**Extended module:** `portfolio/backtest.py` with `transaction_costs` parameter.

---

## Phase 2 — ML Alpha Layer
**Semester 5–6 | v1.0.0**

Add machine learning models on top of the factor infrastructure. Do not replace the factor engine — extend it.

### 2.1 XGBoost quintile classifier with purged cross-validation

**Goal:** train a gradient-boosted tree to predict which quintile a stock will fall into next month, using factor exposures as features.

**Features:** all 10+ z-scored factor values, plus technical features (52W high proximity, bid-ask proxy, market cap rank).

**Target:** next-month return quintile (1–5), cast as a 5-class classification problem.

**Critical methodology — Combinatorial Purged Cross-Validation (de Prado, 2018):**
Standard k-fold CV leaks information through the time dimension (training on future data). Purged CV:
1. Removes training observations that overlap in time with the test set (purging)
2. Adds an embargo period after each test fold (prevents lookahead via autocorrelated features)

**New module:** `ml/xgboost_ranker.py` → class `XGBoostFactorRanker`

**New module:** `ml/purged_cv.py` → class `PurgedKFoldCV` (implements de Prado Chapter 7)

### 2.2 HMM regime detection

**Goal:** detect market regimes (bull, bear, sideways) and adapt factor weights per regime. Momentum works well in trending markets; low-volatility works better in choppy markets.

**Model:** Hidden Markov Model with 3 hidden states, Gaussian emissions on market returns + volatility:

$$P(\text{state}_t | \text{state}_{t-1}) = A \quad \text{(transition matrix)}$$
$$P(x_t | \text{state}_t) = \mathcal{N}(\mu_k, \sigma_k^2) \quad \text{(emission model)}$$

Fitted with the Baum-Welch (EM) algorithm via `hmmlearn`.

**New module:** `ml/regime.py` → class `RegimeDetector`

### 2.3 MLflow experiment tracking

Log all experiments to MLflow: feature sets, hyperparameters, validation IC, and out-of-sample Sharpe. This produces an auditable research history — every model version is reproducible.

### 2.4 React research dashboard

Build a local web dashboard using React + FastAPI:
- Factor weight sliders → live portfolio reallocation
- Regime indicator (bull/bear/sideways) with confidence
- Rolling IC chart per factor
- Portfolio allocation pie chart
- Backtest equity curve with benchmark comparison

**Purpose:** personal research tool for exploratory analysis. Not a public product at this stage.

---

## Phase 3 — Live Paper Trading + India-Specific Alpha
**Year 3 | v2.0.0**

The hardest and most valuable phase: move from backtesting to live market interaction.

### 3.1 Zerodha Kite API integration

- Connect to Kite API for real-time NSE data
- Execute monthly rebalances as paper trades (no real money)
- Log each order: ticker, direction, size, execution price, slippage vs VWAP
- Build a real-time P&L dashboard

**New module:** `execution/kite_connector.py` → class `KiteExecutor`

### 3.2 Kelly Criterion position sizing with volatility targeting

Replace equal cluster weights with Kelly-optimal sizing:

$$f_i^* = \frac{\mu_i - r_f}{\sigma_i^2}$$

where $\mu_i$ is the expected return (from factor model) and $\sigma_i^2$ is the realised volatility. Apply a half-Kelly constraint ($f_i = 0.5 f_i^*$) to reduce variance of outcomes.

Apply a volatility target overlay: scale overall portfolio leverage so that ex-ante portfolio volatility equals 15% annualised.

**Extended module:** `portfolio/position_sizer.py` → class `KellyVolTargetSizer`

### 3.3 NLP earnings revision factor

Scrape SEBI quarterly results filings (BSE/NSE corporate announcements). Apply a text classifier (FinBERT or a fine-tuned distilBERT) to generate a sentiment score per filing. Use the change in sentiment score quarter-over-quarter as an additional alpha factor.

This is **alternative data** — information that is not reflected in price and volume alone. It is India-specific (SEBI filings are Indian regulatory documents) and represents a genuine edge that global factors do not capture.

**New module:** `factors/nlp_earnings.py` → class `EarningsNLPFactor`

### 3.4 Live track record

By the end of Phase 3, EigenAlpha will have:
- 3 years of Git commit history (one coherent, deepening system)
- Real paper trading results with live Sharpe and drawdown numbers
- An auditable MLflow experiment log
- A factor model tested on real NSE market data, not just historical backtests
- An India-specific NLP factor unavailable to global competitors

This is the interview and startup artifact.

---

## Long-Term Vision (Year 4–5)

- Reinforcement learning portfolio allocator (replace the deterministic rebalancer)
- Multi-asset extension: Indian corporate bonds + INR currency + commodity futures
- Vectorised backtest rewrite in NumPy/Numba (10x speed improvement)
- SaaS product or proprietary trading desk infrastructure

---

## Git Branching Strategy

```
main          ← stable, tagged releases only
develop       ← integration branch
feature/xxx   ← individual features (e.g., feature/black-litterman)
phase-N       ← long-lived branch for each phase before merging to develop
```

Tag every release: `git tag -a v0.1.0 -m "Phase 0 initial release"`.

---

*The only rule that makes this roadmap work: never start a new project. Every semester, add one layer to EigenAlpha.*
