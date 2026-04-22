# Contributing to EigenAlpha

EigenAlpha is a long-term personal research project. This document describes the coding standards, commit conventions, and documentation requirements that keep the codebase coherent across multiple years of development.

---

## Code Standards

### Python style
- PEP 8 throughout. Use `black` for formatting (`black .` before every commit).
- Type hints on all public method signatures.
- No bare `print()` statements — use `logging.getLogger(__name__)`.
- Maximum line length: 100 characters.

### Docstrings (Google style)
Every public class and method requires a docstring in this format:

```python
def compute_ic(self, factor_col: str) -> pd.Series:
    """Compute monthly Spearman IC between a factor and forward returns.

    The Information Coefficient measures the rank correlation between
    factor values and subsequent realised returns. A mean IC > 0.05
    is typically considered practically useful.

    Args:
        factor_col: Name of the factor column in self.factor_data.
            Must be one of: 'momentum_12_1', 'realized_vol', 'volume_trend'.

    Returns:
        pd.Series: Monthly IC values, indexed by date (month-end).
            Values in [-1, 1]; positive = factor predicts returns correctly.

    Raises:
        ValueError: If factor_col is not found in self.factor_data.

    References:
        Grinold, R. & Kahn, R. (1999). Active Portfolio Management.
        McGraw-Hill. Chapter 6.
    """
```

### Academic citations
Any line implementing a formula from a paper must have a citation comment above it:

```python
# Jegadeesh & Titman (1993): momentum is computed from t-12M to t-1M,
# skipping the most recent month to avoid short-term reversal bias.
momentum = (prices.shift(21) / prices.shift(252)) - 1
```

---

## Commit Message Convention

Format: `type(scope): short description`

Types:
- `feat`: new feature or module
- `fix`: bug fix
- `docs`: documentation only
- `refactor`: code restructuring without behaviour change
- `test`: adding or modifying tests
- `perf`: performance improvement
- `data`: changes to data pipeline or universe

Examples:
```
feat(factors): add quality factor using ROE from BSE filings
fix(backtest): correct lookahead bias in walk-forward split
docs(math): add Black-Litterman derivation to MATH.md
refactor(optimizer): extract cluster weight computation to separate method
perf(loader): add parquet caching to avoid repeated yfinance API calls
```

**Every commit must reference the CHANGELOG.md entry it corresponds to.**

---

## Testing

Run tests before every commit:
```bash
pytest tests/ -v
```

Test requirements:
- New module → new test file
- New method → new test function
- Bug fix → regression test added
- Minimum 80% line coverage on `factors/` and `portfolio/` modules

---

## Adding a New Factor

1. Add the computation method to `factors/engine.py` as a new method on `FactorEngine`
2. Add academic citation in the docstring
3. Add the factor to `compute_all()` merge
4. Add a test in `tests/test_factors.py` verifying the formula
5. Add an EDA visualisation in `visualisation/eda.py`
6. Add IC analysis in the pipeline
7. Update `CHANGELOG.md` under `[Unreleased]`
8. Update `docs/MATH.md` with the factor's mathematical definition

---

## File Naming

- Classes: `UpperCamelCase` (`FactorEngine`, `MarkowitzOptimizer`)
- Functions/methods: `snake_case` (`compute_ic`, `plot_scree`)
- Files/modules: `snake_case` (`ic_analysis.py`, `black_litterman.py`)
- Constants: `UPPER_SNAKE_CASE` (`RISK_FREE_RATE`, `NIFTY500_TICKERS`)

---

## Output Directory

All outputs (plots, metrics, parquet files) go to `outputs/`. This directory is in `.gitignore` — never commit raw data or generated figures to the repository. The code generates them.

---

## Reproducibility

Every model and backtest must be reproducible from the code alone:
- Set `random_state=42` on all stochastic operations (K-Means, train/test splits)
- Log the Python version and library versions at the start of `pipeline.py`
- Store hyperparameters in `config.py`, not hardcoded in methods
