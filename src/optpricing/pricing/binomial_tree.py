"""Cox-Ross-Rubinstein binomial tree pricing for options."""

from __future__ import annotations

import numpy as np

from optpricing.pricing.base_model import OptionPricingModel


class BinomialTreeModel(OptionPricingModel):
    """Cox-Ross-Rubinstein (CRR) binomial tree.

    Prices European options by default. Set ``american=True`` to allow early
    exercise at every node, which demonstrates the model's flexibility relative
    to the closed-form Black-Scholes solution (whose price it converges to as
    ``n_steps`` grows for European options).
    """

    def __init__(self, S, K, T, r, sigma, q=0.0, n_steps=500, american=False):
        super().__init__(S, K, T, r, sigma, q)
        self.n_steps = int(n_steps)
        self.american = bool(american)

    def _tree_parameters(self):
        dt = self.T / self.n_steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((self.r - self.q) * dt) - d) / (u - d)
        return dt, u, d, p

    def _payoff(self, stock, option_type):
        if option_type == "call":
            return np.maximum(stock - self.K, 0.0)
        return np.maximum(self.K - stock, 0.0)

    def _price(self, option_type):
        dt, u, d, p = self._tree_parameters()

        # Terminal layer (vectorised): node j has had j up-moves.
        j = np.arange(self.n_steps + 1)
        stock = self.S * (u**j) * (d ** (self.n_steps - j))
        values = self._payoff(stock, option_type)

        # Backward induction.
        discount = np.exp(-self.r * dt)
        for i in range(self.n_steps - 1, -1, -1):
            values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
            if self.american:
                j = np.arange(i + 1)
                stock = self.S * (u**j) * (d ** (i - j))
                values = np.maximum(values, self._payoff(stock, option_type))
        return values[0]

    def call_price(self):
        return self._price("call")

    def put_price(self):
        return self._price("put")

    def convergence_to_bs(self, bs_price, step_range=None, option_type="call"):
        """Tabulate the binomial price converging to a Black-Scholes reference.

        Restores ``n_steps`` afterwards so the instance is left unchanged.
        """
        if step_range is None:
            step_range = [10, 50, 100, 500, 1_000, 5_000]

        original_n = self.n_steps
        results = []
        try:
            for n in step_range:
                self.n_steps = n
                price = self._price(option_type)
                results.append(
                    {
                        "n_steps": n,
                        "binomial_price": price,
                        "bs_price": bs_price,
                        "absolute_error": abs(price - bs_price),
                    }
                )
        finally:
            self.n_steps = original_n
        return results
