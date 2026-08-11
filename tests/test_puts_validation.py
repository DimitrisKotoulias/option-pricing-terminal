"""Put-side pricing coverage: parity, price bounds, and American early exercise.

Complements ``test_put_call_parity.py`` (which checks the parity helper itself)
by exercising the put *price* across regimes rather than re-testing parity keys.
"""

import numpy as np
import pytest

from optpricing import BinomialTreeModel, BlackScholesModel, PutCallParity

PARAMS = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, q=0.02)


def test_bs_put_and_call_satisfy_parity():
    bs = BlackScholesModel(**PARAMS)
    res = PutCallParity.verify(
        bs.call_price(),
        bs.put_price(),
        PARAMS["S"],
        PARAMS["K"],
        PARAMS["r"],
        PARAMS["T"],
        PARAMS["q"],
    )
    assert res["parity_holds"]
    assert res["difference"] == pytest.approx(0.0, abs=1e-9)


def test_deep_otm_put_is_nearly_worthless():
    # Spot far above strike: the put is almost certain to expire worthless.
    p = BlackScholesModel(
        S=200.0, K=100.0, T=1.0, r=0.05, sigma=0.25, q=0.02
    ).put_price()
    assert 0.0 <= p
    assert p == pytest.approx(0.0, abs=0.05)


def test_deep_itm_put_near_discounted_intrinsic():
    # Spot far below strike: value approaches the European lower bound
    # K e^{-rT} - S e^{-qT}, and never drops below it.
    S, K, T, r, q = 10.0, 100.0, 1.0, 0.05, 0.0
    p = BlackScholesModel(S=S, K=K, T=T, r=r, sigma=0.2, q=q).put_price()
    lower_bound = K * np.exp(-r * T) - S * np.exp(-q * T)
    assert p == pytest.approx(lower_bound, abs=1e-2)
    assert p >= lower_bound - 1e-9


def test_european_put_binomial_matches_black_scholes():
    bs_put = BlackScholesModel(**PARAMS).put_price()
    tree_put = BinomialTreeModel(**PARAMS, n_steps=2000, american=False).put_price()
    assert tree_put == pytest.approx(bs_put, abs=1e-2)


def test_american_put_has_positive_early_exercise_premium():
    # An ITM put with positive rates: early exercise is worth something.
    args = dict(S=90.0, K=100.0, T=1.0, r=0.08, sigma=0.30, q=0.0, n_steps=500)
    american = BinomialTreeModel(**args, american=True).put_price()
    european = BinomialTreeModel(**args, american=False).put_price()
    assert american >= european
    assert american > european


def test_american_put_equals_european_without_early_exercise_incentive():
    # With r = q = 0 there is no incentive to exercise a put early, so the
    # American and European prices coincide.
    args = dict(S=100.0, K=100.0, T=1.0, r=0.0, sigma=0.20, q=0.0, n_steps=500)
    american = BinomialTreeModel(**args, american=True).put_price()
    european = BinomialTreeModel(**args, american=False).put_price()
    assert american == pytest.approx(european, abs=1e-6)
