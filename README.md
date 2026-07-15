# European Options Pricing Toolkit

Black-Scholes, Monte Carlo, and Binomial Tree pricing for European options behind a
single interface — with a full Greeks suite (analytical **and** numerically verified),
implied-volatility solvers, and put-call parity checks.

## Features

- **Three pricing models** with a shared `OptionPricingModel` interface:
  - `BlackScholesModel` — closed form, with continuous dividend yield.
  - `MonteCarloOptionPricer` — risk-neutral simulation with **antithetic variates**
    and *statistically correct* standard errors (pair-mean estimator).
  - `BinomialTreeModel` — vectorised Cox-Ross-Rubinstein; optional American exercise.
- **Greeks** — first order (delta, gamma, vega, theta, rho) and second order
  (vanna, vomma, charm), each cross-checked against finite differences.
- **Implied volatility** — Brent's method and Newton-Raphson, with no-arbitrage guards.
- **Put-call parity** — verification, implied forward, and arbitrage description.
- **Tested** — 37 tests, 94% coverage, including cross-model consistency and IV round-trips.

## Install

```bash
pip install -e .            # core library
pip install -e ".[dev]"     # + pytest / coverage / black
pip install -e ".[viz,app,data]"   # + plotting, Streamlit app, live market data
```

## Quick start

```python
from optpricing import BlackScholesModel, MonteCarloOptionPricer, BinomialTreeModel, Greeks

bs = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2)
bs.call_price()          # 10.4506

mc = MonteCarloOptionPricer(100, 100, 1, 0.05, 0.2, n_simulations=500_000, seed=42)
price, std_err = mc.call_price(return_std_error=True)   # 10.4557 ± 0.0148

BinomialTreeModel(100, 100, 1, 0.05, 0.2, n_steps=1000).call_price()   # 10.4486

Greeks(100, 100, 1, 0.05, 0.2).get_all_greeks("call")
# {'delta': 0.6368, 'gamma': 0.0188, 'vega': 0.3752, 'theta': -0.0176, 'rho': 0.5323, ...}
```

All three models agree on an at-the-money call to within ~0.005 (Black-Scholes 10.4506,
Binomial 10.4486, Monte Carlo 10.4557 ± 0.0148).

## Project structure

```
src/optpricing/
  pricing/    base_model, black_scholes, monte_carlo, binomial_tree
  analytics/  greeks, greeks_numerical, implied_volatility, put_call_parity
  data/       market_data_fetcher   (yfinance + data/raw cache)
  utils/      visualization
tests/        one module per component (pytest)
app/          streamlit_dashboard.py
docs/         mathematical_background.md
notebooks/    analysis outline (see notebooks/README.md)
```

## Testing

```bash
pytest --cov=optpricing --cov-report=term-missing
```

## Dashboard

```bash
pip install -e ".[app]"
streamlit run app/streamlit_dashboard.py
```

## Roadmap (follow-up phases)

- Fill the analysis notebooks (convergence, volatility surface, market validation).
- Live validation against SPY / AAPL / QQQ option chains via the cached fetcher.
- Stochastic-volatility (Heston) and jump-diffusion (Merton) extensions.

## Math

See [`docs/mathematical_background.md`](docs/mathematical_background.md) for derivations,
scaling conventions, and the variance-reduction note.

## License

MIT — see [LICENSE](LICENSE).
