"""Finite-difference Greeks used to verify the analytical formulas.

Every helper re-prices the model with bumped inputs and compares against the
closed-form :class:`~optpricing.analytics.greeks.Greeks`. The dividend yield
``q`` is threaded through every bump so the verification also holds for
dividend-paying underlyings (the naive version that dropped ``q`` silently
disagreed with the analytical Greeks whenever ``q != 0``).

The returned values are **raw** (unscaled) derivatives. To compare with the
desk-scaled analytical Greeks, divide vega/rho by 100 and theta by 365.
"""

from __future__ import annotations


class GreeksNumerical:
    """Central finite-difference approximations of the option Greeks."""

    @staticmethod
    def _price(model_class, S, K, T, r, sigma, q, option_type):
        model = model_class(S, K, T, r, sigma, q)
        return model.call_price() if option_type == "call" else model.put_price()

    @classmethod
    def delta(cls, model_class, S, K, T, r, sigma, q=0.0, h=1e-2, option_type="call"):
        f = cls._price
        up = f(model_class, S + h, K, T, r, sigma, q, option_type)
        down = f(model_class, S - h, K, T, r, sigma, q, option_type)
        return (up - down) / (2 * h)

    @classmethod
    def gamma(cls, model_class, S, K, T, r, sigma, q=0.0, h=1e-2, option_type="call"):
        f = cls._price
        up = f(model_class, S + h, K, T, r, sigma, q, option_type)
        mid = f(model_class, S, K, T, r, sigma, q, option_type)
        down = f(model_class, S - h, K, T, r, sigma, q, option_type)
        return (up - 2 * mid + down) / (h**2)

    @classmethod
    def vega(cls, model_class, S, K, T, r, sigma, q=0.0, h=1e-4, option_type="call"):
        """Raw dPrice/dSigma. Divide by 100 to compare with analytical vega."""
        f = cls._price
        up = f(model_class, S, K, T, r, sigma + h, q, option_type)
        down = f(model_class, S, K, T, r, sigma - h, q, option_type)
        return (up - down) / (2 * h)

    @classmethod
    def theta(cls, model_class, S, K, T, r, sigma, q=0.0, h=1e-4, option_type="call"):
        """Raw calendar theta = -dPrice/d(time-to-maturity).

        Divide by 365 to compare with the analytical per-day theta.
        """
        f = cls._price
        up = f(model_class, S, K, T + h, r, sigma, q, option_type)
        down = f(model_class, S, K, T - h, r, sigma, q, option_type)
        return -(up - down) / (2 * h)

    @classmethod
    def rho(cls, model_class, S, K, T, r, sigma, q=0.0, h=1e-4, option_type="call"):
        """Raw dPrice/dr. Divide by 100 to compare with analytical rho."""
        f = cls._price
        up = f(model_class, S, K, T, r + h, sigma, q, option_type)
        down = f(model_class, S, K, T, r - h, sigma, q, option_type)
        return (up - down) / (2 * h)

    # -- second-order --------------------------------------------------------
    @classmethod
    def vanna(
        cls, model_class, S, K, T, r, sigma, q=0.0, hs=1e-2, hv=1e-4, option_type="call"
    ):
        """Raw d^2Price/(dS dSigma) via a 4-point mixed-partial stencil.

        Matches the *raw* analytical ``Greeks.vanna()`` directly (both unscaled).
        Identical for calls and puts.
        """
        f = cls._price
        pp = f(model_class, S + hs, K, T, r, sigma + hv, q, option_type)
        pm = f(model_class, S + hs, K, T, r, sigma - hv, q, option_type)
        mp = f(model_class, S - hs, K, T, r, sigma + hv, q, option_type)
        mm = f(model_class, S - hs, K, T, r, sigma - hv, q, option_type)
        return (pp - pm - mp + mm) / (4 * hs * hv)

    @classmethod
    def vomma(cls, model_class, S, K, T, r, sigma, q=0.0, h=1e-3, option_type="call"):
        """Raw d^2Price/dSigma^2 (volga) via a central second difference.

        Uses a larger step than the first-order Greeks for second-derivative
        stability (as :meth:`gamma` does), but small in volatility terms.
        Matches raw ``Greeks.vomma()``.
        """
        f = cls._price
        up = f(model_class, S, K, T, r, sigma + h, q, option_type)
        mid = f(model_class, S, K, T, r, sigma, q, option_type)
        down = f(model_class, S, K, T, r, sigma - h, q, option_type)
        return (up - 2 * mid + down) / (h**2)

    @classmethod
    def charm(cls, model_class, S, K, T, r, sigma, q=0.0, h=1e-4, option_type="call"):
        """Raw charm = -d(delta)/dT. Divide by 365 to compare with analytical charm.

        Delta at each bumped maturity is itself a central difference in spot.
        """
        d_up = cls.delta(model_class, S, K, T + h, r, sigma, q, option_type=option_type)
        d_down = cls.delta(
            model_class, S, K, T - h, r, sigma, q, option_type=option_type
        )
        return -(d_up - d_down) / (2 * h)
