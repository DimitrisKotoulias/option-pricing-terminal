# Notebooks

> **Status: roadmap / outline — no notebooks are implemented yet.**
> This folder is a placeholder describing the analyses planned on top of the
> `optpricing` package. The library, tests, dashboard and validation script in the
> repository root are the finished work; these notebooks are future additions.

When implemented, install the toolkit with the extras first:

```bash
pip install -e ".[viz,data]"
```

Planned analyses:

| Notebook | Contents |
|----------|----------|
| `01_theory_overview` | The three models, assumptions, and where each is used. |
| `02_black_scholes_deep_dive` | d1/d2 intuition, price vs. spot/vol/time. |
| `03_monte_carlo_simulation` | Antithetic variance reduction; convergence & correct confidence intervals. |
| `04_binomial_tree_convergence` | Log-log convergence of CRR to Black-Scholes. |
| `05_greeks_analysis` | First/second-order Greeks; analytical vs. finite-difference; 3-D surfaces. |
| `06_implied_volatility_surface` | Implied vol from a cached option chain; smile/skew. |
| `07_market_validation` | Model prices vs. market (SPX/NDX/RUT); MAE/RMSE by moneyness; parity checks. |
| `08_final_presentation` | Consolidated results and conclusions. |

The market-data analyses (06, 07) will use `optpricing.data.market_data_fetcher`,
which caches option chains to `data/raw/` so they stay reproducible even when the data
provider is unavailable. Until then, see `scripts/market_validation.py` for a working
end-to-end market-validation run.
