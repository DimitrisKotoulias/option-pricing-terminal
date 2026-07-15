"""Analytical Black-Scholes-Merton pricing for European options."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from optpricing.pricing.base_model import OptionPricingModel


class BlackScholesModel(OptionPricingModel):
    """Closed-form Black-Scholes-Merton model with a continuous dividend yield.

    Assumptions: constant volatility and interest rate, log-normally distributed
    returns, frictionless markets and European exercise only.
    """

    def d1(self):
        return (
            np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T
        ) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def call_price(self):
        d1, d2 = self.d1(), self.d2()
        return self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(
            -self.r * self.T
        ) * norm.cdf(d2)

    def put_price(self):
        d1, d2 = self.d1(), self.d2()
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(
            -self.q * self.T
        ) * norm.cdf(-d1)
