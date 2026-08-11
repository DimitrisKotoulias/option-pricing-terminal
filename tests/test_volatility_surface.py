"""Volatility-surface tests using synthetic Black-Scholes chains (no network)."""

import numpy as np
import pandas as pd
import pytest

from optpricing import BlackScholesModel, VolatilitySurface


def _synthetic_chain(spot, strikes, T, r, sigma, q=0.0):
    """Build call/put frames priced by BS at a known sigma.

    ``(bid+ask)/2`` is set to exactly the model price so the surface should
    recover ``sigma``. Column names mirror yfinance option chains.
    """
    calls, puts = [], []
    for K in strikes:
        model = BlackScholesModel(spot, K, T, r, sigma, q)
        c, p = model.call_price(), model.put_price()
        calls.append({"strike": K, "bid": c * 0.99, "ask": c * 1.01, "lastPrice": c})
        puts.append({"strike": K, "bid": p * 0.99, "ask": p * 1.01, "lastPrice": p})
    return pd.DataFrame(calls), pd.DataFrame(puts)


def _records(spot=100.0, sigma=0.20, r=0.03, q=0.0, maturities=(0.25, 0.5, 1.0)):
    strikes = np.arange(85.0, 116.0, 5.0)  # 0.85..1.15 moneyness
    recs = []
    for T in maturities:
        calls, puts = _synthetic_chain(spot, strikes, T, r, sigma, q)
        recs.append(
            {"expiry": f"T={T}", "T": T, "calls": calls, "puts": puts, "spot": spot}
        )
    return recs


def test_surface_recovers_constant_sigma():
    surf = VolatilitySurface.from_chains(_records(sigma=0.20), r=0.03, q=0.0)
    # Every solved point should be ~20% since sigma is flat across the chains.
    assert np.allclose(surf.raw_iv, 20.0, atol=0.5)
    assert np.isfinite(surf.iv_mesh).any()
    assert np.nanmean(surf.iv_mesh) == pytest.approx(20.0, abs=0.5)


def test_surface_mesh_shape_and_axes():
    surf = VolatilitySurface.from_chains(_records(), r=0.03, grid_size=30)
    assert surf.iv_mesh.shape == (len(surf.maturity_axis), len(surf.moneyness_axis))
    assert surf.moneyness_axis.min() >= 0.7 and surf.moneyness_axis.max() <= 1.3
    assert surf.maturity_axis.min() == pytest.approx(0.25)
    assert surf.maturity_axis.max() == pytest.approx(1.0)


def test_surface_callable_rate_is_accepted():
    # r as a callable r(T) (e.g. risk_free_rate_for) must be honoured per expiry.
    surf = VolatilitySurface.from_chains(_records(r=0.03), r=lambda T: 0.03, q=0.0)
    assert np.allclose(surf.raw_iv, 20.0, atol=0.5)


def test_single_expiry_falls_back_to_polyfit():
    surf = VolatilitySurface.from_chains(_records(maturities=(0.5,)), r=0.03)
    assert surf.maturity_axis.size == 1
    assert np.allclose(surf.raw_iv, 20.0, atol=0.5)


def test_polyfit_smoothing_mode():
    surf = VolatilitySurface.from_chains(_records(), r=0.03, smoothing="polyfit")
    assert surf.iv_mesh.shape == (len(surf.maturity_axis), len(surf.moneyness_axis))
    assert np.nanmean(surf.iv_mesh) == pytest.approx(20.0, abs=0.6)


def test_empty_records_raise():
    with pytest.raises(ValueError):
        VolatilitySurface.from_chains([], r=0.03)


def _degenerate_cloud():
    """The point geometry real chains produce, which broke cubic interpolation.

    Dozens of scattered strikes packed into a razor-thin near-expiry slice, with
    the next maturity an order of magnitude further out. That gives the
    Clough-Tocher triangulation near-degenerate triangles and wild gradients: on
    the live AAPL chain it produced a mesh spanning -8473% to +7734% IV out of
    quotes that topped out at 112%.
    """
    rng = np.random.default_rng(0)
    m, t, iv = [], [], []
    near = np.linspace(0.89, 1.13, 77)
    m += near.tolist()
    t += [0.0027] * near.size
    iv += (
        20 + 120 * np.abs(near - 1.0) ** 1.5 * rng.uniform(0.3, 3.0, near.size)
    ).tolist()
    for maturity, level in ((0.044, 55.0), (0.351, 35.0), (1.099, 31.0)):
        k = np.linspace(0.70, 1.29, 32)
        m += k.tolist()
        t += [maturity] * k.size
        iv += (level + 180 * (1 - k) ** 2).tolist()
    return (np.asarray(a, dtype=float) for a in (m, t, iv))


def test_mesh_never_leaves_the_observed_iv_band():
    """Regression: the smoothed mesh must not invent volatilities.

    Guards both interpolation paths against overshoot -- an interpolated surface
    that prints an IV no quote ever showed is not a market surface.
    """
    raw_m, raw_t, raw_iv = _degenerate_cloud()
    lo, hi = np.percentile(raw_iv, [2, 98])
    band_lo, band_hi = np.clip(raw_iv, lo, hi).min(), np.clip(raw_iv, lo, hi).max()

    for smoothing in ("griddata", "polyfit"):
        _, _, mesh = VolatilitySurface._build_grid(raw_m, raw_t, raw_iv, 40, smoothing)
        assert np.nanmin(mesh) >= band_lo - 1e-9, smoothing
        assert np.nanmax(mesh) <= band_hi + 1e-9, smoothing


def test_surface_mesh_stays_within_raw_iv_range_end_to_end():
    """Same invariant through the public entry point, on uneven maturities."""
    surf = VolatilitySurface.from_chains(
        _records(maturities=(0.003, 0.05, 0.4, 1.1)), r=0.03
    )
    assert np.nanmax(surf.iv_mesh) <= surf.raw_iv.max() + 1e-9
    assert np.nanmin(surf.iv_mesh) >= 0.0
