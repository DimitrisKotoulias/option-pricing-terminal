"""Abstract base class shared by every option pricing model."""

from __future__ import annotations

from abc import ABC, abstractmethod


class OptionPricingModel(ABC):
    """Common interface and shared input validation for pricing models.

    Parameters
    ----------
    S : float
        Current price of the underlying asset (spot).
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously-compounded annualised risk-free rate.
    sigma : float
        Annualised volatility of the underlying.
    q : float, optional
        Continuous dividend yield (annualised). Defaults to ``0.0``.

    Notes
    -----
    Validation runs in ``__init__`` so that *every* subclass (Black-Scholes,
    Monte Carlo, Binomial Tree) rejects invalid inputs on construction, rather
    than only the models that remember to call :meth:`validate_inputs`.
    """

    def __init__(self, S, K, T, r, sigma, q=0.0):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)
        self.validate_inputs()

    def validate_inputs(self):
        """Raise ``ValueError`` if any shared input is out of range."""
        if self.S <= 0 or self.K <= 0:
            raise ValueError("Spot and strike prices must be positive.")
        if self.T <= 0:
            raise ValueError("Time to maturity must be positive.")
        if self.sigma <= 0:
            raise ValueError("Volatility must be positive.")

    @abstractmethod
    def call_price(self):
        """Return the price of a European call option."""

    @abstractmethod
    def put_price(self):
        """Return the price of a European put option."""

    def __repr__(self):
        return (
            f"{type(self).__name__}(S={self.S}, K={self.K}, T={self.T}, "
            f"r={self.r}, sigma={self.sigma}, q={self.q})"
        )
