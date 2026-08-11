import numpy as np
import pytest

from optpricing import BlackScholesModel
from optpricing import ImpliedVolatilityCalculator as IV


def test_iv_round_trip_call():
    true_sigma = 0.28
    price = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=true_sigma).call_price()
    iv = IV.calculate(price, 100, 100, 1, 0.05, "call")
    assert iv == pytest.approx(true_sigma, abs=1e-4)


def test_iv_round_trip_put_with_dividend():
    true_sigma = 0.19
    price = BlackScholesModel(
        S=100, K=90, T=0.75, r=0.03, sigma=true_sigma, q=0.02
    ).put_price()
    iv = IV.calculate(price, 100, 90, 0.75, 0.03, "put", q=0.02)
    assert iv == pytest.approx(true_sigma, abs=1e-4)


def test_newton_matches_brent():
    price = BlackScholesModel(S=100, K=110, T=2, r=0.04, sigma=0.22).call_price()
    brent = IV.calculate(price, 100, 110, 2, 0.04, "call")
    newton = IV.newton_raphson(price, 100, 110, 2, 0.04, "call")
    assert newton == pytest.approx(brent, abs=1e-4)


def test_iv_above_upper_bound_returns_none():
    # A call cannot be worth more than the (dividend-discounted) spot.
    assert IV.calculate(1e6, 100, 100, 1, 0.05, "call") is None


def test_iv_zero_price_returns_none():
    assert IV.calculate(0.0, 100, 100, 1, 0.05, "call") is None


def test_calculate_vectorized_matches_scalar():
    S, T, r, sigma = 100.0, 1.0, 0.05, 0.25
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    prices = np.array(
        [BlackScholesModel(S, K, T, r, sigma).call_price() for K in strikes]
    )
    ivs = IV.calculate_vectorized(prices, S, strikes, T, r, "call")
    assert np.allclose(ivs, sigma, atol=1e-4)
    # And each entry agrees with the scalar Brent solver.
    for K, price in zip(strikes, prices):
        assert IV.calculate(price, S, K, T, r, "call") == pytest.approx(sigma, abs=1e-4)


def test_calculate_vectorized_put_with_dividend():
    S, T, r, sigma, q = 100.0, 0.75, 0.03, 0.19, 0.02
    strikes = np.array([85.0, 100.0, 115.0])
    prices = np.array(
        [BlackScholesModel(S, K, T, r, sigma, q).put_price() for K in strikes]
    )
    ivs = IV.calculate_vectorized(prices, S, strikes, T, r, "put", q)
    assert np.allclose(ivs, sigma, atol=1e-4)


def test_calculate_vectorized_out_of_bounds_returns_nan():
    strikes = np.array([100.0, 100.0])
    prices = np.array([1e6, -5.0])  # above spot (impossible) and non-positive
    ivs = IV.calculate_vectorized(prices, 100, strikes, 1.0, 0.05, "call")
    assert np.isnan(ivs).all()
