"""Put-call parity verification and arbitrage detection."""

from __future__ import annotations

import numpy as np


class PutCallParity:
    """European put-call parity: ``C - P = S*e^{-qT} - K*e^{-rT}``."""

    @staticmethod
    def verify(call_price, put_price, S, K, r, T, q=0.0, tolerance=0.01):
        """Return the two sides of the parity identity and their difference."""
        lhs = call_price - put_price
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        difference = lhs - rhs
        return {
            "lhs": lhs,
            "rhs": rhs,
            "difference": difference,
            "parity_holds": abs(difference) < tolerance,
        }

    @staticmethod
    def implied_forward_price(call_price, put_price, K, r, T):
        """Forward price implied by parity: ``F = (C - P)*e^{rT} + K``."""
        return (call_price - put_price) * np.exp(r * T) + K

    @staticmethod
    def detect_arbitrage(call_price, put_price, S, K, r, T, q=0.0, tolerance=0.01):
        """Flag a parity violation and describe the offsetting strategy."""
        check = PutCallParity.verify(call_price, put_price, S, K, r, T, q, tolerance)
        if check["parity_holds"]:
            return {"arbitrage": False}

        if check["difference"] > 0:
            # C - P too rich relative to the forward: sell the call side.
            strategy = "Sell call, buy put, buy stock, borrow PV(K)"
        else:
            strategy = "Buy call, sell put, short stock, lend PV(K)"
        return {
            "arbitrage": True,
            "strategy": strategy,
            "expected_profit": abs(check["difference"]),
        }
