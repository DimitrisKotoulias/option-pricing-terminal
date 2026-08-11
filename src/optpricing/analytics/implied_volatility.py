"""Implied-volatility solvers (Brent's method and Newton-Raphson)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from optpricing.pricing.black_scholes import BlackScholesModel


class ImpliedVolatilityCalculator:
    """Recover the Black-Scholes volatility implied by a market option price.

    The inverse problem ``BS(S, K, T, r, sigma) = market_price`` has no
    closed-form solution, so it is solved numerically. Prices outside the
    no-arbitrage bounds have no solution and return ``None`` rather than raising.
    """

    @staticmethod
    def _within_no_arbitrage_bounds(market_price, S, K, T, r, option_type, q):
        disc_k = K * np.exp(-r * T)
        disc_s = S * np.exp(-q * T)
        if option_type == "call":
            lower, upper = max(disc_s - disc_k, 0.0), disc_s
        else:
            lower, upper = max(disc_k - disc_s, 0.0), disc_k
        return lower - 1e-10 <= market_price <= upper + 1e-10

    @classmethod
    def calculate(cls, market_price, S, K, T, r, option_type="call", q=0.0):
        """Solve for implied volatility with Brent's method.

        Returns ``None`` when the price is non-positive, the maturity has
        elapsed, or the price lies outside the no-arbitrage bounds (all cases
        where no real implied volatility exists).
        """
        if market_price <= 0 or T <= 0:
            return None
        if not cls._within_no_arbitrage_bounds(
            market_price, S, K, T, r, option_type, q
        ):
            return None

        def objective(sigma):
            model = BlackScholesModel(S, K, T, r, sigma, q)
            price = model.call_price() if option_type == "call" else model.put_price()
            return price - market_price

        try:
            return brentq(objective, 1e-6, 5.0, xtol=1e-8, maxiter=200)
        except ValueError:
            return None

    @staticmethod
    def newton_raphson(
        market_price,
        S,
        K,
        T,
        r,
        option_type="call",
        q=0.0,
        initial_guess=0.3,
        max_iter=100,
        tol=1e-8,
    ):
        """Solve for implied volatility with Newton-Raphson using vega.

        Returns ``None`` if it fails to converge (e.g. vega collapses to zero).
        """
        from optpricing.analytics.greeks import Greeks

        if market_price <= 0 or T <= 0:
            return None

        sigma = initial_guess
        diff = None
        for _ in range(max_iter):
            model = Greeks(S, K, T, r, sigma, q)
            price = model.call_price() if option_type == "call" else model.put_price()
            # Greeks.vega() is per 1%; multiply by 100 to recover raw dPrice/dSigma.
            vega = model.vega() * 100.0
            diff = price - market_price
            if abs(diff) < tol:
                return sigma
            if vega < 1e-12:
                break
            sigma -= diff / vega
            if sigma <= 0:
                sigma = 1e-4
        return sigma if diff is not None and abs(diff) < 1e-4 else None

    @classmethod
    def calculate_vectorized(
        cls,
        prices,
        S,
        strikes,
        T,
        r,
        option_type="call",
        q=0.0,
        initial_guess=0.3,
        max_iter=50,
    ):
        """Solve implied volatility for a whole strike vector at once.

        A NumPy Newton-Raphson (BS price + raw vega) run in lock-step across all
        strikes -- far faster than looping the scalar solver for the many
        (strike x expiry) points a volatility surface needs. Entries whose price
        is outside the no-arbitrage bounds, or that fail to converge, come back
        as ``NaN`` (so callers can mask them) rather than raising.

        Parameters
        ----------
        prices, strikes : array-like
            Observed option prices and their strikes (broadcast together).
        S, T, r, q : float
            Spot, maturity (years), risk-free rate and dividend yield -- shared
            across the vector (one expiry per call).
        option_type : {"call", "put"}
        """
        prices = np.asarray(prices, dtype=float)
        strikes = np.asarray(strikes, dtype=float)
        prices, strikes = np.broadcast_arrays(prices, strikes)
        result = np.full(prices.shape, np.nan)

        if T <= 0:
            return result

        disc_k = strikes * np.exp(-r * T)
        disc_s = S * np.exp(-q * T)
        if option_type == "call":
            lower, upper = np.maximum(disc_s - disc_k, 0.0), np.full_like(
                disc_k, disc_s
            )
        else:
            lower, upper = np.maximum(disc_k - disc_s, 0.0), disc_k
        valid = (
            (prices > 0)
            & (strikes > 0)
            & (prices >= lower - 1e-10)
            & (prices <= upper + 1e-10)
        )
        if not valid.any():
            return result

        sqrt_t = np.sqrt(T)
        sigma = np.full(prices.shape, float(initial_guess))
        for _ in range(max_iter):
            d1 = (np.log(S / strikes) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
            d2 = d1 - sigma * sqrt_t
            if option_type == "call":
                price = disc_s * norm.cdf(d1) - disc_k * norm.cdf(d2)
            else:
                price = disc_k * norm.cdf(-d2) - disc_s * norm.cdf(-d1)
            vega = disc_s * norm.pdf(d1) * sqrt_t
            diff = price - prices
            # Deep-OTM strikes underflow vega to 0.0; only divide where it is
            # safe so ``diff / vega`` never evaluates 0/0 (which np.where would
            # still compute for every element, emitting a divide-by-zero
            # RuntimeWarning even though the result is masked out).
            safe = valid & (vega > 1e-12)
            step = np.divide(diff, vega, out=np.zeros_like(diff), where=safe)
            sigma = np.clip(sigma - step, 1e-6, 5.0)

        # Accept only strikes whose final residual is acceptably small.
        d1 = (np.log(S / strikes) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        if option_type == "call":
            final_price = disc_s * norm.cdf(d1) - disc_k * norm.cdf(d2)
        else:
            final_price = disc_k * norm.cdf(-d2) - disc_s * norm.cdf(-d1)
        converged = valid & (np.abs(final_price - prices) < 1e-4)
        result[converged] = sigma[converged]
        return result
