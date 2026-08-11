"""optpricing: European options pricing and validation toolkit.

Three pricing models (Black-Scholes, Monte Carlo, Binomial Tree) behind a common
interface, plus analytical/numerical Greeks, implied volatility solvers and
put-call parity checks.
"""

from optpricing.analytics.greeks import Greeks
from optpricing.analytics.greeks_numerical import GreeksNumerical
from optpricing.analytics.implied_volatility import ImpliedVolatilityCalculator
from optpricing.analytics.put_call_parity import PutCallParity
from optpricing.analytics.volatility_surface import VolatilitySurface
from optpricing.pricing.binomial_tree import BinomialTreeModel
from optpricing.pricing.black_scholes import BlackScholesModel
from optpricing.pricing.monte_carlo import MonteCarloOptionPricer

__version__ = "0.1.0"

__all__ = [
    "BlackScholesModel",
    "MonteCarloOptionPricer",
    "BinomialTreeModel",
    "Greeks",
    "GreeksNumerical",
    "ImpliedVolatilityCalculator",
    "PutCallParity",
    "VolatilitySurface",
]
