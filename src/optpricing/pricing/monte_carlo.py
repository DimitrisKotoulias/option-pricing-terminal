"""Monte Carlo pricing for European options with variance reduction."""

from __future__ import annotations

import numpy as np

from optpricing.pricing.base_model import OptionPricingModel


class MonteCarloOptionPricer(OptionPricingModel):
    """Risk-neutral Monte Carlo pricer using simulated terminal GBM prices.

    Only the terminal underlying price is required for European payoffs, so no
    full path simulation is needed. Antithetic variates are supported for
    variance reduction.
    """

    def __init__(self, S, K, T, r, sigma, q=0.0, n_simulations=100_000, seed=None):
        super().__init__(S, K, T, r, sigma, q)
        self.n_simulations = int(n_simulations)
        if self.n_simulations <= 0:
            raise ValueError("Number of simulations must be positive.")
        # ``default_rng`` seeds correctly for seed=0 (``if seed:`` would skip it)
        # and avoids mutating NumPy's global random state.
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _simulate_terminal_prices(self, antithetic=True):
        """Simulate terminal underlying prices under the risk-neutral measure."""
        if antithetic:
            half = self.n_simulations // 2
            z = self.rng.standard_normal(half)
            z = np.concatenate([z, -z])
            if self.n_simulations % 2:
                # Odd count: the antithetic pairs cover 2*half draws; add one
                # independent draw so exactly ``n_simulations`` samples are used.
                z = np.concatenate([z, self.rng.standard_normal(1)])
        else:
            z = self.rng.standard_normal(self.n_simulations)

        drift = (self.r - self.q - 0.5 * self.sigma**2) * self.T
        diffusion = self.sigma * np.sqrt(self.T) * z
        return self.S * np.exp(drift + diffusion)

    def _price(self, option_type, antithetic=True, return_std_error=False):
        terminal = self._simulate_terminal_prices(antithetic=antithetic)
        if option_type == "call":
            payoffs = np.maximum(terminal - self.K, 0.0)
        else:
            payoffs = np.maximum(self.K - terminal, 0.0)

        discounted = np.exp(-self.r * self.T) * payoffs
        price = discounted.mean()

        if not return_std_error:
            return price
        return price, self._standard_error(discounted, antithetic)

    @staticmethod
    def _standard_error(discounted, antithetic):
        """Return the standard error of the discounted-payoff estimator.

        With antithetic variates the draws are **not** i.i.d.: each pair
        ``(Z, -Z)`` is negatively correlated, so ``std(all) / sqrt(N)`` is not a
        valid standard error (it ignores that correlation). The correct
        estimator treats each antithetic *pair mean* as one i.i.d. observation,
        giving ``M = N / 2`` effective samples.
        """
        if antithetic:
            half = len(discounted) // 2
            pair_means = 0.5 * (discounted[:half] + discounted[half : 2 * half])
            # An odd sample count leaves one unpaired independent draw at the
            # end; count it as one more i.i.d. observation.
            leftover = discounted[2 * half :]
            obs = (
                np.concatenate([pair_means, leftover]) if leftover.size else pair_means
            )
            return obs.std(ddof=1) / np.sqrt(len(obs))
        return discounted.std(ddof=1) / np.sqrt(len(discounted))

    def call_price(self, antithetic=True, return_std_error=False):
        return self._price("call", antithetic, return_std_error)

    def put_price(self, antithetic=True, return_std_error=False):
        return self._price("put", antithetic, return_std_error)

    def convergence_analysis(self, simulation_counts=None, option_type="call"):
        """Price the option across increasing simulation counts.

        Returns a list of dicts with the price, standard error and 95%
        confidence interval, illustrating how quickly Monte Carlo converges to
        the analytical Black-Scholes value.
        """
        if simulation_counts is None:
            simulation_counts = [100, 1_000, 10_000, 100_000, 1_000_000]

        original_n = self.n_simulations
        results = []
        try:
            for n in simulation_counts:
                self.n_simulations = n
                price, std_err = self._price(
                    option_type, antithetic=True, return_std_error=True
                )
                results.append(
                    {
                        "n_simulations": n,
                        "price": price,
                        "std_error": std_err,
                        "ci_95_low": price - 1.96 * std_err,
                        "ci_95_high": price + 1.96 * std_err,
                    }
                )
        finally:
            self.n_simulations = original_n
        return results

    def generate_paths(self, n_paths: int = 100) -> np.ndarray:
        """Generate n_paths simulated daily price paths using GBM.

        Returns a 2D array of shape (steps + 1, n_paths) where steps is the number
        of daily steps (252 per year).
        """
        # Draw from a dedicated local generator (seeded identically to the
        # instance) so path generation is reproducible per call and never
        # advances the pricing RNG (``self.rng``) state.
        rng = np.random.default_rng(self.seed)
        steps = int(self.T * 252.0)
        if steps < 1:
            steps = 1
        dt = self.T / steps
        paths = np.zeros((steps + 1, n_paths))
        paths[0] = self.S

        # We simulate using the risk-neutral drift
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)

        # We can draw the increments
        for t in range(1, steps + 1):
            z = rng.standard_normal(n_paths)
            paths[t] = paths[t - 1] * np.exp(drift + diffusion * z)

        return paths
