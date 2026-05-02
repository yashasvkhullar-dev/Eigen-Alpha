"""
EigenAlpha — Markowitz Mean-Variance Optimiser
===============================================

The ``MarkowitzOptimizer`` class implements portfolio optimisation using the
classical Markowitz mean-variance framework, adapted for a cluster-based
portfolio construction approach:

1. **Intra-cluster optimisation**: Within each cluster, minimise portfolio
   variance subject to weight constraints (long-only, fully invested).
2. **Cross-cluster allocation**: Weight each cluster's sub-portfolio by its
   mean factor score, allocating more capital to clusters with stronger
   expected alpha signals.
3. **Efficient frontier**: Trace the full efficient frontier by varying
   the target return.

The optimiser uses ``scipy.optimize.minimize`` with the SLSQP method,
which handles equality and inequality constraints natively.

Academic references:
    Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*,
    7(1), 77–91.

    Ledoit, O., & Wolf, M. (2004). Honey, I Shrunk the Sample Covariance
    Matrix. *Journal of Portfolio Management*, 30(4), 110–119.

Usage:
    from portfolio.optimizer import MarkowitzOptimizer
    opt = MarkowitzOptimizer(returns, factor_scores, cluster_labels)
    weights = opt.optimize_all_clusters()
    frontier = opt.efficient_frontier()
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class MarkowitzOptimizer:
    """Mean-variance portfolio optimiser with cluster-based construction.

    Attributes:
        returns (pd.DataFrame): Wide-format monthly returns (T × N).
        factor_scores (pd.Series): Composite factor score per ticker.
        cluster_labels (pd.Series): Cluster assignment per ticker.
        n_clusters (int): Number of unique clusters.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        factor_scores: pd.Series,
        cluster_labels: pd.Series,
    ) -> None:
        """Initialise the MarkowitzOptimizer.

        Args:
            returns: Wide-format monthly returns with index=Date,
                columns=Tickers.
            factor_scores: Series indexed by ticker with a composite
                factor score (higher = better expected alpha).
            cluster_labels: Series indexed by ticker with cluster
                assignment (integer).

        Raises:
            ValueError: If inputs are empty or misaligned.
        """
        if returns.empty:
            raise ValueError("returns DataFrame must not be empty.")
        if factor_scores.empty:
            raise ValueError("factor_scores must not be empty.")
        if cluster_labels.empty:
            raise ValueError("cluster_labels must not be empty.")

        # Align tickers across all inputs
        common_tickers = (
            set(returns.columns) &
            set(factor_scores.index) &
            set(cluster_labels.index)
        )

        if len(common_tickers) < 5:
            raise ValueError(
                f"Insufficient common tickers across inputs: {len(common_tickers)}"
            )

        self.tickers = sorted(common_tickers)
        self.returns = returns[self.tickers].copy()
        self.factor_scores = factor_scores.loc[self.tickers].copy()
        self.cluster_labels = cluster_labels.loc[self.tickers].copy()
        self.n_clusters = self.cluster_labels.nunique()

        logger.info(
            "MarkowitzOptimizer: %d tickers, %d clusters, %d months",
            len(self.tickers),
            self.n_clusters,
            len(self.returns),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Intra-Cluster Optimisation
    # ──────────────────────────────────────────────────────────────────────

    def optimize_cluster(self, cluster_id: int) -> pd.Series:
        """Optimise a minimum-variance portfolio within a single cluster.

        Objective: minimise w^T Σ w (portfolio variance).

        Constraints:
            - sum(w) == 1 (fully invested)
            - w >= 0 (long-only)

        Args:
            cluster_id: Integer cluster ID to optimise.

        Returns:
            pd.Series: Optimal weights indexed by ticker.

        Raises:
            ValueError: If ``cluster_id`` is not found.
        """
        mask = self.cluster_labels == cluster_id
        tickers = self.cluster_labels[mask].index.tolist()

        if len(tickers) < 2:
            # Single stock or empty cluster: equal weight
            return pd.Series(1.0, index=tickers)

        ret = self.returns[tickers].dropna(how="all")
        cov = ret.cov().values
        n = len(tickers)

        # Initial guess: equal weight
        w0 = np.ones(n) / n

        # Academic reference: Markowitz (1952) — minimum-variance portfolio
        def objective(w: np.ndarray) -> float:
            """Portfolio variance: w^T Σ w."""
            return w @ cov @ w

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # sum(w) = 1
        ]

        # Bounds: long-only
        bounds = [(0.0, 1.0)] * n

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not result.success:
            logger.warning(
                "Optimisation did not converge for cluster %d: %s. "
                "Using equal weights.",
                cluster_id,
                result.message,
            )
            weights = np.ones(n) / n
        else:
            weights = result.x

        # Clean tiny weights
        weights = np.maximum(weights, 0)
        weights /= weights.sum()

        return pd.Series(weights, index=tickers, name=f"cluster_{cluster_id}")

    # ──────────────────────────────────────────────────────────────────────
    # Cross-Cluster Allocation
    # ──────────────────────────────────────────────────────────────────────

    def optimize_all_clusters(self) -> pd.Series:
        """Optimise across all clusters and allocate by factor score.

        Process:
            1. Run ``optimize_cluster()`` for each cluster.
            2. Compute mean factor score per cluster.
            3. Weight each cluster proportional to its mean factor score
               (shifted to be non-negative).
            4. Combine intra-cluster weights with cluster-level weights.
            5. Normalise final portfolio to sum to 1.

        Returns:
            pd.Series: Final portfolio weights indexed by ticker,
                summing to 1.0.
        """
        logger.info("Optimising all %d clusters...", self.n_clusters)

        cluster_ids = sorted(self.cluster_labels.unique())
        cluster_weights = {}
        cluster_portfolios = {}

        for cid in cluster_ids:
            # Intra-cluster optimisation
            cluster_portfolios[cid] = self.optimize_cluster(cid)

            # Mean factor score for this cluster
            mask = self.cluster_labels == cid
            mean_score = self.factor_scores[mask].mean()
            cluster_weights[cid] = mean_score

        # Shift scores to be non-negative, then normalise
        cw = pd.Series(cluster_weights)
        cw = cw - cw.min() + 0.01  # Ensure all positive
        cw = cw / cw.sum()

        logger.info("Cluster allocations: %s", cw.round(4).to_dict())

        # Combine: final_weight[stock] = cluster_weight * intra_weight
        all_weights = pd.Series(0.0, index=self.tickers, dtype=float)

        for cid in cluster_ids:
            intra = cluster_portfolios[cid]
            for ticker in intra.index:
                all_weights[ticker] = cw[cid] * intra[ticker]

        # Normalise
        all_weights = all_weights / all_weights.sum()

        n_nonzero = (all_weights > 1e-6).sum()
        logger.info(
            "Final portfolio: %d non-zero weights (of %d tickers). "
            "Max weight: %.4f, Min non-zero: %.6f",
            n_nonzero,
            len(all_weights),
            all_weights.max(),
            all_weights[all_weights > 1e-6].min(),
        )

        return all_weights

    # ──────────────────────────────────────────────────────────────────────
    # Efficient Frontier
    # ──────────────────────────────────────────────────────────────────────

    def efficient_frontier(self, n_points: int = 100) -> pd.DataFrame:
        """Compute the efficient frontier by varying target return.

        Traces a series of portfolios along the efficient frontier from
        the minimum-variance portfolio to the maximum-return portfolio.

        Args:
            n_points: Number of points on the frontier.  Defaults to 100.

        Returns:
            pd.DataFrame: DataFrame with columns
                ``[return, volatility, sharpe]``.  Each row is a
                portfolio on the frontier.
        """
        ret = self.returns.dropna(how="all")
        mu = ret.mean().values
        cov = ret.cov().values
        n = len(mu)

        r_min = mu.min()
        r_max = mu.max()
        targets = np.linspace(r_min, r_max, n_points)

        frontier = []

        for r_target in targets:
            w0 = np.ones(n) / n

            def objective(w: np.ndarray) -> float:
                return w @ cov @ w

            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "ineq", "fun": lambda w, rt=r_target: w @ mu - rt},
            ]

            bounds = [(0.0, 1.0)] * n

            result = minimize(
                objective,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-10},
            )

            if result.success:
                w = result.x
                port_ret = w @ mu * 12  # Annualise
                port_vol = np.sqrt(w @ cov @ w) * np.sqrt(12)  # Annualise
                sharpe = (port_ret - 0.065) / port_vol if port_vol > 0 else 0
                frontier.append(
                    {"return": port_ret, "volatility": port_vol, "sharpe": sharpe}
                )

        frontier_df = pd.DataFrame(frontier)

        if not frontier_df.empty:
            best_idx = frontier_df["sharpe"].idxmax()
            logger.info(
                "Efficient frontier: %d points. "
                "Max Sharpe: %.3f at (ret=%.3f, vol=%.3f)",
                len(frontier_df),
                frontier_df.loc[best_idx, "sharpe"],
                frontier_df.loc[best_idx, "return"],
                frontier_df.loc[best_idx, "volatility"],
            )

        return frontier_df
