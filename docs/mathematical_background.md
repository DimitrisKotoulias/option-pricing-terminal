# Mathematical Background

## 1. Black-Scholes-Merton model

### Assumptions
1. The underlying follows geometric Brownian motion with constant volatility.
2. Constant risk-free rate and (optionally) a continuous dividend yield.
3. No arbitrage, continuous trading, no transaction costs.
4. European exercise only.

### PDE and closed form
Under the risk-neutral measure the stock price satisfies `dS = (r - q) S dt + σ S dW`.
Constructing a riskless hedged portfolio and applying Itô's lemma gives the
Black-Scholes PDE:

```
∂V/∂t + ½ σ² S² ∂²V/∂S² + (r - q) S ∂V/∂S - r V = 0
```

with closed-form European solutions (dividend yield `q`):

```
C = S e^(-qT) N(d1) - K e^(-rT) N(d2)
P = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)

d1 = [ln(S/K) + (r - q + σ²/2) T] / (σ √T)
d2 = d1 - σ √T
```

Implemented in `optpricing/pricing/black_scholes.py`.

## 2. Monte Carlo

Terminal price under the risk-neutral measure:

```
S_T = S₀ · exp[(r - q - σ²/2) T + σ √T · Z],   Z ~ N(0, 1)
C₀ = e^(-rT) · E[max(S_T - K, 0)]
```

Only the terminal value is needed for European payoffs, so paths are not simulated.

### Variance reduction — antithetic variates
Each draw `Z` is paired with `-Z`. For a monotonic payoff this reduces estimator
variance. **Important:** the paired samples are *not* independent, so the standard
error must be computed from the `M = N/2` pair means
`½(payoff(Z) + payoff(-Z))`, **not** from `std(all)/√N`, which ignores the
negative correlation within each pair. See `MonteCarloOptionPricer._standard_error`.

## 3. Binomial tree (Cox-Ross-Rubinstein)

Discretise `[0, T]` into `n` steps of `Δt = T/n`:

```
u = e^(σ √Δt),   d = 1/u,   p = (e^((r - q) Δt) - d) / (u - d)
```

Terminal payoffs are discounted by backward induction. As `n → ∞` the discrete
price converges to Black-Scholes (via the CLT). Setting `american=True` adds an
early-exercise check `max(continuation, intrinsic)` at each node.
Implemented in `optpricing/pricing/binomial_tree.py`.

## 4. The Greeks

| Greek | Definition | Notes |
|-------|------------|-------|
| Delta | ∂V/∂S | Hedge ratio |
| Gamma | ∂²V/∂S² | Same for call & put |
| Vega  | ∂V/∂σ | Same for call & put; quoted per 1% |
| Theta | ∂V/∂t | Time decay; quoted per day |
| Rho   | ∂V/∂r | Quoted per 1% |
| Vanna | ∂²V/∂S∂σ | Second order |
| Vomma | ∂²V/∂σ² | Second order (volga) |
| Charm | ∂²V/∂S∂t | Delta decay |

Second-order Greeks are built on the **raw** (unscaled) vega so their scaling is
internally consistent. Every first-order Greek is cross-checked against a central
finite-difference approximation in `greeks_numerical.py` (see `tests/test_greeks.py`).

## 5. Put-call parity

```
C - P = S e^(-qT) - K e^(-rT)
```

Proved by replication: a call plus `K e^(-rT)` cash has the same expiry payoff as a
put plus `S e^(-qT)` of stock. Deviations imply arbitrage. See
`optpricing/analytics/put_call_parity.py`.

## 6. Implied volatility

Given a market price, solve `BS(S, K, T, r, σ) = price` for `σ`. No closed form
exists, so we use Brent's method (bracketed, robust) and Newton-Raphson (uses vega,
fast). Prices outside the no-arbitrage bounds have no solution and return `None`.
See `optpricing/analytics/implied_volatility.py`.

## References
- Hull, J. *Options, Futures, and Other Derivatives.*
- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.*
- Cox, J., Ross, S. & Rubinstein, M. (1979). *Option Pricing: A Simplified Approach.*
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering.* (variance reduction)
