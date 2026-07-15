import numpy as np
import pytest

from optpricing import BlackScholesModel


def test_call_price_hull_textbook_value():
    # Hull, "Options, Futures and Other Derivatives": S=42, K=40, T=0.5,
    # r=0.10, sigma=0.20 -> call ~ 4.76.
    model = BlackScholesModel(S=42, K=40, T=0.5, r=0.10, sigma=0.20)
    assert model.call_price() == pytest.approx(4.759, abs=1e-2)


def test_put_price_hull_textbook_value():
    model = BlackScholesModel(S=42, K=40, T=0.5, r=0.10, sigma=0.20)
    assert model.put_price() == pytest.approx(0.81, abs=1e-2)


def test_put_call_parity_identity_no_dividends():
    model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2)
    lhs = model.call_price() - model.put_price()
    rhs = 100 - 100 * np.exp(-0.05)
    assert lhs == pytest.approx(rhs, abs=1e-10)


def test_put_call_parity_with_dividend_yield():
    model = BlackScholesModel(S=100, K=95, T=1.5, r=0.04, sigma=0.25, q=0.03)
    lhs = model.call_price() - model.put_price()
    rhs = 100 * np.exp(-0.03 * 1.5) - 95 * np.exp(-0.04 * 1.5)
    assert lhs == pytest.approx(rhs, abs=1e-10)


def test_deep_itm_call_approaches_intrinsic():
    model = BlackScholesModel(S=200, K=100, T=0.01, r=0.05, sigma=0.2)
    intrinsic = 200 - 100 * np.exp(-0.05 * 0.01)
    assert model.call_price() == pytest.approx(intrinsic, abs=0.5)


def test_deep_otm_call_near_zero():
    model = BlackScholesModel(S=50, K=150, T=0.25, r=0.05, sigma=0.2)
    assert model.call_price() < 1e-2


def test_call_price_increases_with_volatility():
    low = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.10).call_price()
    high = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.40).call_price()
    assert high > low


@pytest.mark.parametrize(
    "bad",
    [
        dict(S=-100, K=100, T=1, r=0.05, sigma=0.2),
        dict(S=100, K=0, T=1, r=0.05, sigma=0.2),
        dict(S=100, K=100, T=0, r=0.05, sigma=0.2),
        dict(S=100, K=100, T=1, r=0.05, sigma=-0.2),
    ],
)
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        BlackScholesModel(**bad)
