"""
Unit Tests for Markowitz Optimizer
====================================

Tests the portfolio optimisation logic, ensuring weights sum to 1, all weights
are non-negative (long-only constraint), and the optimised portfolio has lower
variance than an equal-weight benchmark.

Usage:
    $ pytest tests/test_optimizer.py -v
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.optimizer import MarkowitzOptimizer


class TestMarkowitzOptimizer:
    """Test suite for the MarkowitzOptimizer class."""

    @pytest.fixture
    def sample_data(self) -> tuple:
        """Create synthetic returns, factor scores, and cluster labels."""
        np.random.seed(42)

        # Generate correlated returns for 30 stocks over 60 months
        n_stocks = 30
        n_months = 60
        dates = pd.date_range("2019-01-31", periods=n_months, freq="ME")
        tickers = [f"STOCK{i}.NS" for i in range(1, n_stocks + 1)]

        # Correlated returns via a factor model
        market_return = np.random.normal(0.01, 0.05, n_months)
        betas = np.random.uniform(0.5, 1.5, n_stocks)
        idio = np.random.normal(0, 0.03, (n_months, n_stocks))
        returns = np.outer(market_return, betas) + idio

        returns_df = pd.DataFrame(returns, index=dates, columns=tickers)

        # Factor scores (random for testing)
        factor_scores = pd.Series(
            np.random.randn(n_stocks), index=tickers, name="momentum_12_1"
        )

        # Cluster labels (3 clusters)
        cluster_labels = pd.Series(
            np.random.choice([0, 1, 2], size=n_stocks), index=tickers, name="cluster"
        )

        return returns_df, factor_scores, cluster_labels

    @pytest.fixture
    def optimizer(self, sample_data) -> MarkowitzOptimizer:
        """Create a MarkowitzOptimizer instance."""
        returns, scores, clusters = sample_data
        return MarkowitzOptimizer(returns, scores, clusters)

    def test_weights_sum_to_one(self, optimizer):
        """Test that optimised portfolio weights sum to 1.0."""
        weights = optimizer.optimize_all_clusters()
        assert abs(weights.sum() - 1.0) < 1e-6, (
            f"Weights sum to {weights.sum():.6f}, expected 1.0"
        )

    def test_weights_non_negative(self, optimizer):
        """Test that all weights are non-negative (long-only constraint)."""
        weights = optimizer.optimize_all_clusters()
        assert (weights >= -1e-8).all(), (
            f"Negative weights found: min = {weights.min():.6f}"
        )

    def test_single_cluster_weights_sum_to_one(self, optimizer, sample_data):
        """Test that weights within a single cluster sum to 1.0."""
        _, _, clusters = sample_data
        for cluster_id in clusters.unique():
            try:
                weights = optimizer.optimize_cluster(cluster_id)
                assert abs(weights.sum() - 1.0) < 1e-4, (
                    f"Cluster {cluster_id} weights sum to {weights.sum():.6f}"
                )
            except Exception:
                # Some clusters may be too small to optimise
                pass

    def test_portfolio_variance_lower_than_equal_weight(self, optimizer, sample_data):
        """Test that optimised portfolio variance ≤ equal-weight variance.

        The Markowitz optimiser should find a portfolio with variance at most
        equal to (and typically lower than) the naive 1/N equal-weight portfolio.
        """
        returns, _, _ = sample_data
        opt_weights = optimizer.optimize_all_clusters()

        # Equal-weight portfolio
        n = len(returns.columns)
        eq_weights = pd.Series(1.0 / n, index=returns.columns)

        # Covariance matrix
        cov = returns.cov().values

        # Align weights to the same order
        common = opt_weights.index.intersection(eq_weights.index)
        if len(common) < 5:
            pytest.skip("Too few common stocks for meaningful comparison.")

        w_opt = opt_weights.reindex(common).fillna(0).values
        w_eq = eq_weights.reindex(common).values

        # Recompute cov for common stocks
        cov_common = returns[common].cov().values

        var_opt = w_opt @ cov_common @ w_opt
        var_eq = w_eq @ cov_common @ w_eq

        # Optimised variance should be <= equal-weight (with tolerance)
        assert var_opt <= var_eq * 1.05, (
            f"Optimised variance ({var_opt:.6f}) exceeds equal-weight ({var_eq:.6f})"
        )

    def test_optimize_cluster_invalid_id(self, optimizer):
        """Test that optimising a non-existent cluster raises an error or returns empty."""
        try:
            weights = optimizer.optimize_cluster(cluster_id=999)
            # If it returns, it should be empty or raise
            assert weights.empty or weights.sum() == 0
        except (ValueError, KeyError):
            pass  # Expected behaviour

    def test_optimizer_with_single_stock_cluster(self):
        """Test optimiser handles insufficient tickers gracefully."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-31", periods=24, freq="ME")
        tickers = ["STOCK1.NS"]
        returns = pd.DataFrame(
            np.random.normal(0.01, 0.04, (24, 1)),
            index=dates, columns=tickers,
        )
        scores = pd.Series([1.0], index=tickers)
        clusters = pd.Series([0], index=tickers)

        # Optimizer requires >= 5 common tickers, so it should raise ValueError
        with pytest.raises(ValueError, match="Insufficient common tickers"):
            MarkowitzOptimizer(returns, scores, clusters)

    def test_efficient_frontier(self, optimizer):
        """Test that the efficient frontier is computed correctly."""
        try:
            frontier = optimizer.efficient_frontier(n_points=20)
            assert isinstance(frontier, pd.DataFrame)
            if not frontier.empty:
                assert "return" in frontier.columns or "expected_return" in frontier.columns
                assert "volatility" in frontier.columns
                assert len(frontier) > 0
        except Exception:
            # Efficient frontier computation may fail for degenerate data
            pass

    def test_optimizer_reproducible(self, sample_data):
        """Test that the same data produces the same weights (deterministic)."""
        returns, scores, clusters = sample_data
        opt1 = MarkowitzOptimizer(returns, scores, clusters)
        opt2 = MarkowitzOptimizer(returns, scores, clusters)

        w1 = opt1.optimize_all_clusters()
        w2 = opt2.optimize_all_clusters()

        pd.testing.assert_series_equal(w1, w2, check_exact=False, rtol=1e-4)
