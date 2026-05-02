# EigenAlpha — Mathematical Documentation

**Version:** Phase 0  
**Author:** EigenAlpha Research  
**Status:** Active

This document derives and explains every mathematical operation in EigenAlpha Phase 0 from first principles. It is written to serve two purposes: as a reference for the implementer, and as a study guide for anyone who wants to understand *why* each formula is used, not just *how* to code it.

---

## Table of Contents

1. [Return Computation](#1-return-computation)
2. [Winsorisation and Z-Scoring](#2-winsorisation-and-z-scoring)
3. [Covariance Matrix](#3-covariance-matrix)
4. [Eigendecomposition and PCA](#4-eigendecomposition-and-pca)
5. [K-Means Clustering](#5-k-means-clustering)
6. [Alpha Factors](#6-alpha-factors)
7. [Information Coefficient](#7-information-coefficient)
8. [Markowitz Optimisation](#8-markowitz-optimisation)
9. [Backtest Metrics](#9-backtest-metrics)
10. [Why Cluster Before Optimising](#10-why-cluster-before-optimising)

---

## 1. Return Computation

### 1.1 Simple Returns

The simple return of asset $i$ from time $t-1$ to $t$ is:

$$r_{i,t} = \frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}} = \frac{P_{i,t}}{P_{i,t-1}} - 1$$

**Problem:** Simple returns are not additive across time. If an asset returns +50% then −50%, the compound result is −25%, not 0%.

### 1.2 Log Returns

The log return (continuously compounded return) is:

$$\ell_{i,t} = \ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right) = \ln(P_{i,t}) - \ln(P_{i,t-1})$$

**Why we use log returns for the covariance matrix:**

1. Log returns are *time-additive*: $\ell_{i,t \to t+2} = \ell_{i,t \to t+1} + \ell_{i,t+1 \to t+2}$
2. They are approximately normally distributed (log-normal price assumption)
3. The covariance matrix of log returns has cleaner statistical properties
4. For small returns, $\ell \approx r$ (the difference is negligible for daily returns)

### 1.3 The Return Matrix R

After computing daily log returns for all $N$ stocks over $T$ trading days:

$$R \in \mathbb{R}^{T \times N}, \quad R_{t,i} = \ell_{i,t}$$

This matrix is the fundamental input to PCA and covariance estimation.

---

## 2. Winsorisation and Z-Scoring

These are preprocessing steps applied to factor values, not raw returns. They are essential to ensure that extreme outliers do not dominate the cross-sectional analysis.

### 2.1 Winsorisation

For a cross-section of factor values $\{f_{i,t}\}_{i=1}^N$ at time $t$:

Let $q_L = \text{percentile}(f_t, 2.5)$ and $q_U = \text{percentile}(f_t, 97.5)$.

$$\tilde{f}_{i,t} = \begin{cases} q_L & \text{if } f_{i,t} < q_L \\ f_{i,t} & \text{if } q_L \leq f_{i,t} \leq q_U \\ q_U & \text{if } f_{i,t} > q_U \end{cases}$$

**Why:** A single stock with an extreme momentum value (e.g., +400% due to a news event) would distort the entire cross-sectional ranking if untreated. Winsorising clips it to the 97.5th percentile rather than removing it entirely.

### 2.2 Cross-Sectional Z-Score

After winsorising, standardise within each cross-section:

$$z_{i,t} = \frac{\tilde{f}_{i,t} - \bar{f}_t}{\sigma_{f,t}}$$

where $\bar{f}_t = \frac{1}{N}\sum_i \tilde{f}_{i,t}$ and $\sigma_{f,t} = \sqrt{\frac{1}{N-1}\sum_i (\tilde{f}_{i,t} - \bar{f}_t)^2}$.

**Why:** After z-scoring, the cross-section has mean 0 and standard deviation 1 at every point in time. This makes factor values comparable across time (avoids regime-specific level shifts) and across different factors (makes factors combinable on equal footing).

---

## 3. Covariance Matrix

### 3.1 Definition

The sample covariance matrix of asset returns is:

$$\Sigma = \frac{1}{T-1} \sum_{t=1}^{T} (r_t - \bar{r})(r_t - \bar{r})^T \in \mathbb{R}^{N \times N}$$

In matrix form, after demeaning the return matrix: $\tilde{R} = R - \mathbf{1}\bar{r}^T$,

$$\Sigma = \frac{1}{T-1} \tilde{R}^T \tilde{R}$$

**In pandas:** `returns.cov()` computes this correctly, handling missing values with pairwise-complete observations.

### 3.2 Interpretation

Entry $\Sigma_{ij}$ is the covariance between assets $i$ and $j$:
- Diagonal entries $\Sigma_{ii} = \sigma_i^2$ are each asset's variance
- Off-diagonal entries measure co-movement: positive = move together, negative = move in opposite directions
- The correlation matrix $C_{ij} = \Sigma_{ij} / (\sigma_i \sigma_j)$ normalises this to $[-1, 1]$

### 3.3 Properties Relevant to Optimisation

The covariance matrix is:
- **Symmetric:** $\Sigma = \Sigma^T$
- **Positive semi-definite:** $w^T \Sigma w \geq 0$ for all $w$ (portfolio variance cannot be negative)
- In practice with 500 assets and limited history: nearly **singular** (rank-deficient), which is why we cluster before optimising

---

## 4. Eigendecomposition and PCA

### 4.1 Eigendecomposition

For a symmetric matrix $\Sigma \in \mathbb{R}^{N \times N}$, the eigendecomposition is:

$$\Sigma = V \Lambda V^T$$

where:
- $V = [v_1 \; v_2 \; \ldots \; v_N]$ is an orthogonal matrix of **eigenvectors** ($V^T V = I$)
- $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_N)$ contains **eigenvalues**, sorted $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_N \geq 0$

Each eigenvector $v_k$ is a direction in the $N$-dimensional return space. Its corresponding eigenvalue $\lambda_k$ is the variance of the market "explained" by that direction.

**In numpy:** `np.linalg.eigh()` — use `eigh` not `eig` because it is specialised for symmetric/Hermitian matrices, is faster, and guarantees real eigenvalues.

### 4.2 Economic Interpretation

The first principal component $v_1$ is the direction of maximum variance in returns. For equity markets, this almost always represents the **market factor** (all stocks move together). The eigenvalue $\lambda_1$ measures how much of total market variance is attributable to this common factor.

- $v_1$: market factor (all elements positive, roughly proportional to market cap weight)
- $v_2$: typically a size factor (large-cap vs small-cap contrast)
- $v_3$: typically a value or momentum factor
- $v_k$ for large $k$: idiosyncratic, stock-specific noise

This is the theoretical link between PCA and the Arbitrage Pricing Theory (APT) of Ross (1976) — exploited by Connor & Korajczyk (1988).

### 4.3 Explained Variance and Component Selection

The fraction of total variance explained by the first $k$ components:

$$\text{EVR}(k) = \frac{\sum_{j=1}^{k} \lambda_j}{\sum_{j=1}^{N} \lambda_j}$$

Select $k^*$ such that $\text{EVR}(k^*) \geq 0.80$ (80% threshold — captures systematic risk, discards noise).

### 4.4 Projecting Stocks onto Principal Components

Each stock $i$ is represented by its $T$-length return series (a column of $R$). To express stock $i$ as a point in the $k^*$-dimensional PC space:

$$\text{score}_{i} = V_{[:, 1:k^*]}^T \cdot r_i \in \mathbb{R}^{k^*}$$

The $N \times k^*$ matrix of scores is the input to K-Means clustering. It says: "here is where each stock lives in the space of market risk factors."

---

## 5. K-Means Clustering

### 5.1 Objective

Given stock score vectors $\{s_i\}_{i=1}^{N}$ in $\mathbb{R}^{k^*}$, partition them into $K$ clusters $\{C_1, \ldots, C_K\}$ to minimise total within-cluster variance (inertia):

$$J = \sum_{k=1}^{K} \sum_{i \in C_k} \|s_i - \mu_k\|^2$$

where $\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} s_i$ is the centroid of cluster $k$.

### 5.2 Algorithm (Lloyd's Algorithm)

1. Initialise $K$ centroids (using k-means++ initialisation for stability)
2. **Assignment step:** assign each stock to the nearest centroid: $c_i = \arg\min_k \|s_i - \mu_k\|^2$
3. **Update step:** recompute centroids as the mean of assigned stocks
4. Repeat until assignments converge

### 5.3 Choosing K: Silhouette Score

For a given $K$, the silhouette score measures how well-separated the clusters are:

$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$

where $a_i$ = mean distance from stock $i$ to all other stocks in its cluster, $b_i$ = mean distance from stock $i$ to all stocks in the nearest other cluster.

Silhouette score $\in [-1, 1]$. Values above 0.3 indicate meaningful cluster structure.

### 5.4 Why This Matters

GICS sectors (Technology, Financials, Healthcare...) group stocks by business model. K-Means clusters them by *how their returns actually move*. A bank and a commodity exporter may cluster together if both are highly sensitive to the same macro PC (say, global risk appetite). This provides a more accurate picture of portfolio diversification than sector allocation.

---

## 6. Alpha Factors

### 6.1 Momentum (12-1 Month)

$$\text{MOM}_{i,t} = \frac{P_{i,t-1M}}{P_{i,t-12M}} - 1$$

**Economic rationale:** Jegadeesh & Titman (1993) documented that stocks with high returns over the past 12 months continue to outperform over the following 3–12 months. This is attributed to investor under-reaction to information and slow diffusion of news.

**Why skip the last month:** Short-term momentum (past 1 month) reverses (bid-ask bounce, mean-reversion effects). Skipping $t \to t-1M$ isolates the intermediate-horizon signal.

**Implementation note:** Compute on month-end prices. Use adjusted close prices (accounting for dividends and splits).

### 6.2 Realised Volatility (20-Day)

$$\text{RVOL}_{i,t} = \sqrt{252} \cdot \sigma\left(\ell_{i,\tau}\right)_{\tau=t-20}^{t}$$

where $\sigma(\cdot)$ is the sample standard deviation of daily log returns over the trailing 20 trading days, and $\sqrt{252}$ annualises it.

**Economic rationale:** The low-volatility anomaly (Blitz & van Vliet, 2007; Baker, Bradley & Wurgler, 2011) shows that low-volatility stocks generate risk-adjusted returns that exceed high-volatility stocks. This is attributed to institutional constraints (leverage aversion, benchmark-relative mandates) that cause overcrowding in high-volatility stocks.

**As a factor:** We use RVOL with a *negative* expected sign — lower volatility stocks are expected to have higher risk-adjusted returns. After z-scoring, a low RVOL score indicates a potentially attractive stock.

### 6.3 Volume Trend (20-Day)

Fit an OLS regression of $\log V_{i,\tau}$ on a time index over the trailing 20 trading days:

$$\log V_{i,\tau} = \alpha_i + \beta_i \cdot \tau + \varepsilon_{i,\tau}, \quad \tau \in [t-20, t]$$

The factor value is the slope $\hat{\beta}_i$.

$$\text{VOLTRD}_{i,t} = \hat{\beta}_i$$

**Economic rationale:** Rising volume tends to precede price continuation — it signals that new information is being incorporated, and that the prevailing price trend is well-supported. A positive volume trend alongside positive momentum is a stronger signal than momentum alone.

**Implementation:** Use `scipy.stats.linregress` inside a rolling window. Resample to month-end for consistency.

---

## 7. Information Coefficient

### 7.1 Definition

The Information Coefficient at time $t$ is the Spearman rank correlation between the factor values and the subsequent realised returns:

$$\text{IC}_t = \rho_S\left(f_t, r_{t+1M}\right)$$

where $\rho_S$ is the Spearman rank correlation:

$$\rho_S = 1 - \frac{6 \sum_i d_i^2}{N(N^2-1)}, \quad d_i = \text{rank}(f_{i,t}) - \text{rank}(r_{i,t+1M})$$

**Why Spearman, not Pearson?** Spearman correlation measures monotonic association (do higher-ranked factor stocks tend to have higher-ranked returns?) without assuming a linear or normally-distributed relationship. Factor scores predict relative ordering, not absolute returns.

### 7.2 Information Ratio

$$\text{IR} = \frac{\overline{\text{IC}}}{\sigma_{\text{IC}}}$$

where $\overline{\text{IC}} = \frac{1}{T}\sum_t \text{IC}_t$ and $\sigma_{\text{IC}} = \text{std}(\text{IC}_t)$.

**Interpretation:**
- $|\overline{\text{IC}}| > 0.05$: factor has practically useful predictive ability
- $\text{IR} > 0.5$: factor generates consistent alpha (low variance relative to mean)
- A high mean IC with high IC variance (low IR) indicates an inconsistent factor — sometimes powerful, sometimes useless

### 7.3 IC t-statistic

To test whether mean IC is statistically significantly different from zero:

$$t = \frac{\overline{\text{IC}}}{\sigma_{\text{IC}} / \sqrt{T}}$$

Under the null hypothesis of no predictability ($\text{IC} = 0$), this follows a $t$-distribution with $T-1$ degrees of freedom.

### 7.4 IC Decay

IC at lag $\ell$ measures how far into the future the factor retains predictive power:

$$\text{IC}_t(\ell) = \rho_S\left(f_t, r_{t+\ell M}\right)$$

Plotting $\overline{\text{IC}(\ell)}$ for $\ell = 1, 2, \ldots, 6$ shows the *signal decay curve*. A factor with IC that decays to zero after lag 2 supports monthly rebalancing; one that decays after lag 1 may need weekly rebalancing.

---

## 8. Markowitz Optimisation

### 8.1 The Mean-Variance Problem

Given $N$ assets with expected return vector $\mu \in \mathbb{R}^N$ and covariance matrix $\Sigma \in \mathbb{R}^{N \times N}$, find portfolio weights $w \in \mathbb{R}^N$ that solve:

$$\min_{w} \; w^T \Sigma w \quad \text{s.t.} \quad w^T \mu \geq r^*, \; \mathbf{1}^T w = 1, \; w_i \geq 0 \; \forall i$$

- $w^T \Sigma w$ is the portfolio variance (squared volatility)
- $r^*$ is the target return (a parameter the user chooses)
- $\mathbf{1}^T w = 1$ ensures weights sum to 100%
- $w_i \geq 0$ is the long-only constraint (no short selling)

### 8.2 Why This is a Quadratic Programme

The objective $w^T \Sigma w$ is a quadratic function of $w$ (positive semi-definite). The constraints are all linear in $w$. This is a **convex quadratic programme (QP)** — it has a unique global minimum, and the feasible set (satisfying all constraints) is a convex polytope.

`scipy.optimize.minimize(method='SLSQP')` implements Sequential Least Squares Programming, which handles this class of problems efficiently.

### 8.3 The Efficient Frontier

By varying $r^*$ from $\min(\mu)$ to $\max(\mu)$, we trace the **efficient frontier** — the set of portfolios with minimum variance for each level of expected return. Any portfolio *above* the frontier is unachievable; any portfolio *below* it is suboptimal (there exists a better-risk portfolio with the same return).

The **tangency portfolio** (maximum Sharpe ratio) is:

$$w^* = \arg\max_{w} \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}}$$

where $r_f$ is the risk-free rate. This is what institutional investors typically target.

### 8.4 Why We Optimise Per Cluster

Running Markowitz on all 500 stocks simultaneously fails in practice because:

1. The $500 \times 500$ covariance matrix is estimated from ~2500 monthly observations — statistically noisy
2. Inverting a near-singular matrix amplifies estimation errors, producing extreme weights (e.g., 80% in one stock)
3. With 500 assets and 500 constraints, the optimiser is numerically unstable

By running Markowitz within each cluster of ~60 stocks:
- Covariance estimates are more stable
- Constraints are binding at a scale where they make sense
- The per-cluster problem is $60 \times 60$ — well-conditioned

The inter-cluster allocation is determined by mean factor scores — clusters with stronger aggregate factor signals receive larger capital allocations.

---

## 9. Backtest Metrics

### 9.1 Annualised Return

From monthly returns $\{r_t\}_{t=1}^{T}$:

$$\hat{\mu}_{ann} = (1 + \bar{r})^{12} - 1$$

where $\bar{r} = \frac{1}{T}\sum_t r_t$ is the mean monthly return.

### 9.2 Annualised Volatility

$$\hat{\sigma}_{ann} = \sigma(r_t) \cdot \sqrt{12}$$

The $\sqrt{12}$ scaling assumes monthly returns are i.i.d. (independent, identically distributed) — a simplifying assumption that slightly overstates volatility if returns are autocorrelated.

### 9.3 Sharpe Ratio

$$\text{Sharpe} = \frac{\hat{\mu}_{ann} - r_f}{\hat{\sigma}_{ann}}$$

where $r_f = 0.065$ (RBI repo rate proxy). The Sharpe ratio measures return per unit of risk. Values above 1.0 are considered strong for a systematic equity strategy.

### 9.4 Maximum Drawdown

Given cumulative return series $W_t = \prod_{\tau=1}^t (1 + r_\tau)$:

$$\text{MDD} = \min_{t} \left(\frac{W_t}{\max_{\tau \leq t} W_\tau} - 1\right)$$

Maximum drawdown is the largest peak-to-trough loss experienced. It measures tail risk — a strategy may have good average returns but intolerable losses during crises.

### 9.5 Calmar Ratio

$$\text{Calmar} = \frac{\hat{\mu}_{ann}}{|\text{MDD}|}$$

The Calmar ratio measures return relative to worst-case loss. It is preferred over Sharpe by managers focused on drawdown risk. Values above 0.5 are considered acceptable; above 1.0 is strong.

---

## 10. Why Cluster Before Optimising

This section explains the methodological innovation that connects the PCA/clustering layer to the portfolio construction layer.

### 10.1 The Estimation Error Problem

Markowitz optimisation is mathematically elegant but practically fragile. Chopra & Ziemba (1993) showed that estimation errors in $\mu$ and $\Sigma$ are amplified by the optimiser — small errors in inputs produce wildly unstable weights.

With $N = 500$ assets, we must estimate:
- $N = 500$ expected returns
- $N(N+1)/2 = 125,250$ unique covariance terms

from only $T \approx 120$ months of history. The covariance matrix is **severely underdetermined** — we have far fewer observations than parameters. The matrix is nearly singular, and its inverse (needed implicitly by the optimiser) amplifies noise.

### 10.2 Dimensionality Reduction via PCA

PCA compresses the 500-dimensional return space into $k^* \approx 10$–15 dimensions that capture 80% of variance. This is a form of **regularisation** — we discard the high-frequency noise dimensions and work only with the systematic risk structure.

### 10.3 Clustering as Block Structure

K-Means partitions the 500 stocks into 8 clusters of ~60 stocks each. Within a cluster, stocks share common PC exposures — they move together for the same reasons. The within-cluster covariance matrix (60×60) is:

- Estimated from the same 120 months, but for only 60 assets → much lower estimation error
- Well-conditioned (invertible without amplifying noise)
- Economically coherent — we are optimising within a group of assets that share risk

### 10.4 Factor-Weighted Cluster Allocation

After optimising within each cluster, we must decide how much capital to allocate across clusters. We use the composite factor score:

$$w_{\text{cluster } k} \propto \bar{z}_k = \frac{1}{|C_k|} \sum_{i \in C_k} z_{i,t}$$

where $z_{i,t}$ is the composite (averaged) z-scored factor value. Clusters with higher average factor scores receive more capital. This connects the factor research (Stage 4) to the portfolio construction (Stage 5) in a principled way.

---

## References

Blitz, D. & van Vliet, P. (2007). The volatility effect. *Journal of Portfolio Management*, 34(1), 102–113.

Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press. (Free: stanford.edu/~boyd/cvxbook)

Chopra, V. & Ziemba, W. (1993). The effect of errors in means, variances, and covariances. *Journal of Portfolio Management*, 19(2), 6–11.

Connor, G. & Korajczyk, R. (1988). Risk and return in an equilibrium APT. *Journal of Financial Economics*, 21(2), 255–289.

Fama, E. & French, K. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56.

Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*, 48(1), 65–91.

López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77–91.

Ross, S. (1976). The arbitrage theory of capital asset pricing. *Journal of Economic Theory*, 13(3), 341–360.

---

*For implementation details, see the source code in each module. For academic context, follow the citations above.*
