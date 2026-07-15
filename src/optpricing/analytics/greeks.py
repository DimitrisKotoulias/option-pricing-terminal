"""Analytical option Greeks (first and second order)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from optpricing.pricing.black_scholes import BlackScholesModel


class Greeks(BlackScholesModel):
    """Closed-form option sensitivities derived from Black-Scholes.

    Presentation scaling (matching how trading desks usually quote Greeks):

    * ``vega`` and ``rho`` are per **1 percentage-point** move (divided by 100).
    * ``theta`` is per **calendar day** (divided by 365).
    * Second-order Greeks (``vanna``, ``vomma``) are built on the *raw*,
      unscaled vega (:meth:`_vega_raw`) so they stay internally consistent;
      only :meth:`vega` applies the /100 presentation scale. This avoids the
      subtle bug of computing volga from an already-divided vega.
    """

    # -- shared raw quantity -------------------------------------------------
    def _vega_raw(self):
        """dPrice/dSigma per unit (1.00) change in volatility (unscaled)."""
        return self.S * np.exp(-self.q * self.T) * norm.pdf(self.d1()) * np.sqrt(self.T)

    # -- first-order ---------------------------------------------------------
    def delta_call(self):
        return np.exp(-self.q * self.T) * norm.cdf(self.d1())

    def delta_put(self):
        return -np.exp(-self.q * self.T) * norm.cdf(-self.d1())

    def gamma(self):
        """Identical for calls and puts."""
        return (
            np.exp(-self.q * self.T)
            * norm.pdf(self.d1())
            / (self.S * self.sigma * np.sqrt(self.T))
        )

    def vega(self):
        """Per 1% change in volatility (identical for calls and puts)."""
        return self._vega_raw() / 100.0

    def theta_call(self):
        d1, d2 = self.d1(), self.d2()
        term1 = -(self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * self.sigma) / (
            2 * np.sqrt(self.T)
        )
        term2 = self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(d1)
        term3 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        return (term1 + term2 + term3) / 365.0

    def theta_put(self):
        d1, d2 = self.d1(), self.d2()
        term1 = -(self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * self.sigma) / (
            2 * np.sqrt(self.T)
        )
        term2 = -self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)
        term3 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
        return (term1 + term2 + term3) / 365.0

    def rho_call(self):
        return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2()) / 100.0

    def rho_put(self):
        return (
            -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2()) / 100.0
        )

    # -- second-order --------------------------------------------------------
    def vanna(self):
        """dDelta/dSigma = dVega/dSpot (per unit vol); identical for call/put."""
        return -np.exp(-self.q * self.T) * norm.pdf(self.d1()) * self.d2() / self.sigma

    def vomma(self):
        """dVega/dSigma (volga), built on raw vega for internal consistency."""
        return self._vega_raw() * self.d1() * self.d2() / self.sigma

    def charm_call(self):
        """dDelta/dTime for a call, per calendar day."""
        return self._charm(is_call=True)

    def charm_put(self):
        """dDelta/dTime for a put, per calendar day."""
        return self._charm(is_call=False)

    def _charm(self, is_call):
        d1, d2 = self.d1(), self.d2()
        sqrt_t = np.sqrt(self.T)
        common = (
            np.exp(-self.q * self.T)
            * norm.pdf(d1)
            * (2 * (self.r - self.q) * self.T - d2 * self.sigma * sqrt_t)
            / (2 * self.T * self.sigma * sqrt_t)
        )
        if is_call:
            first = self.q * np.exp(-self.q * self.T) * norm.cdf(d1)
        else:
            first = -self.q * np.exp(-self.q * self.T) * norm.cdf(-d1)
        return (first - common) / 365.0

    # -- convenience ---------------------------------------------------------
    def get_all_greeks(self, option_type="call"):
        """Return a dict of all Greeks for the requested option type."""
        shared = {
            "gamma": self.gamma(),
            "vega": self.vega(),
            "vanna": self.vanna(),
            "vomma": self.vomma(),
        }
        if option_type == "call":
            return {
                "delta": self.delta_call(),
                "theta": self.theta_call(),
                "rho": self.rho_call(),
                "charm": self.charm_call(),
                **shared,
            }
        return {
            "delta": self.delta_put(),
            "theta": self.theta_put(),
            "rho": self.rho_put(),
            "charm": self.charm_put(),
            **shared,
        }
