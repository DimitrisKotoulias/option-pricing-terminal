"""Matplotlib plotting helpers for options analytics.

Matplotlib is an optional dependency, imported lazily so that importing the
core toolkit never requires it. Install with::

    pip install -e ".[viz]"
"""

from __future__ import annotations

import numpy as np


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            'Plotting requires matplotlib. Install with: pip install -e ".[viz]"'
        ) from exc
    return plt


def plot_payoff(S_range, K, option_type="call", premium=0.0, position="long"):
    """Plot the P&L of a single option position at expiration."""
    plt = _pyplot()
    if option_type == "call":
        payoff = np.maximum(S_range - K, 0.0)
    else:
        payoff = np.maximum(K - S_range, 0.0)
    pnl = payoff - premium if position == "long" else premium - payoff

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(S_range, pnl, lw=2.5, color="navy")
    ax.axhline(0, color="black", ls="--", alpha=0.3)
    ax.axvline(K, color="red", ls="--", alpha=0.5, label=f"Strike = {K}")
    ax.fill_between(S_range, pnl, 0, where=(pnl > 0), alpha=0.2, color="green")
    ax.fill_between(S_range, pnl, 0, where=(pnl < 0), alpha=0.2, color="red")
    ax.set_xlabel("Underlying price at expiration")
    ax.set_ylabel("Profit / Loss")
    ax.set_title(f"{position.capitalize()} {option_type} P&L")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_convergence(results, x_key, y_key="absolute_error", title="Convergence"):
    """Log-log convergence plot from a list of result dicts.

    Works directly with the output of ``BinomialTreeModel.convergence_to_bs``
    (``x_key="n_steps"``) or ``MonteCarloOptionPricer.convergence_analysis``.
    """
    plt = _pyplot()
    x = [row[x_key] for row in results]
    y = [row[y_key] for row in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    return fig


def plot_volatility_smile(strikes, implied_vols, spot):
    """Plot implied volatility against moneyness (K / S) for one maturity."""
    plt = _pyplot()
    moneyness = np.asarray(strikes) / spot

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(moneyness, implied_vols, marker="o", lw=2)
    ax.axvline(1.0, color="red", ls="--", alpha=0.5, label="ATM")
    ax.set_xlabel("Moneyness (K / S)")
    ax.set_ylabel("Implied volatility")
    ax.set_title("Volatility smile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig
