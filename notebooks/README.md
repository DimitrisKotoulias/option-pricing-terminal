# Notebooks

Analysis notebooks that build on the `optpricing` package. Install the toolkit
with the extras first:

```bash
pip install -e ".[viz,data]"
```

Planned notebooks (outline for the analysis phase):

| Notebook | Contents |
|----------|----------|
| `01_theory_overview` | The three models, assumptions, and where each is used. |
| `02_black_scholes_deep_dive` | d1/d2 intuition, price vs. spot/vol/time. |
| `03_monte_carlo_simulation` | Antithetic variance reduction; convergence & correct confidence intervals. |
| `04_binomial_tree_convergence` | Log-log convergence of CRR → Black-Scholes. |
| `05_greeks_analysis` | First/second-order Greeks; analytical vs. finite-difference; 3-D surfaces. |
| `06_implied_volatility_surface` | Implied vol from a cached option chain; smile/skew. |
| `07_market_validation` | Model prices vs. market (SPY/AAPL/QQQ); MAE/RMSE by moneyness; parity checks. |
| `08_final_presentation` | Consolidated results and conclusions. |

The market-data notebooks (06, 07) use `optpricing.data.market_data_fetcher`,
which caches option chains to `data/raw/` so the notebooks stay reproducible even
when the data provider is unavailable.
