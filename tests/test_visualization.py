"""Smoke tests for the optional matplotlib helpers.

These are public API (documented in the README) but nothing inside the repo
imports them, so without these tests a broken helper would only surface for a
user of the library. Headless ``Agg`` backend: no display, no blocking.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - must follow the backend switch
from matplotlib.figure import Figure  # noqa: E402

from optpricing import BinomialTreeModel, BlackScholesModel  # noqa: E402
from optpricing.utils import visualization as viz  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("position", ["long", "short"])
def test_plot_payoff_returns_figure(option_type, position):
    S_range = np.linspace(50, 150, 101)
    fig = viz.plot_payoff(S_range, 100, option_type, premium=5.0, position=position)
    assert isinstance(fig, Figure)
    # One P&L line plus the two zero/strike guides.
    ax = fig.axes[0]
    assert len(ax.lines) >= 1
    assert ax.get_ylabel() == "Profit / Loss"


def test_plot_payoff_pnl_is_break_even_at_strike_plus_premium():
    S_range = np.linspace(50, 150, 101)
    fig = viz.plot_payoff(S_range, 100, "call", premium=5.0)
    x, y = fig.axes[0].lines[0].get_data()
    assert y[0] == pytest.approx(-5.0)  # deep OTM long call loses the premium
    assert np.interp(105.0, x, y) == pytest.approx(0.0, abs=1e-9)


def test_plot_convergence_accepts_binomial_results():
    bs = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2)
    results = BinomialTreeModel(S=100, K=100, T=1, r=0.05, sigma=0.2).convergence_to_bs(
        bs.call_price(), step_range=[10, 100, 500]
    )
    fig = viz.plot_convergence(results, x_key="n_steps")
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"


def test_plot_volatility_smile_plots_moneyness():
    strikes = np.array([80.0, 100.0, 120.0])
    fig = viz.plot_volatility_smile(strikes, [0.25, 0.20, 0.23], spot=100.0)
    assert isinstance(fig, Figure)
    x, _ = fig.axes[0].lines[0].get_data()
    assert list(x) == pytest.approx([0.8, 1.0, 1.2])
