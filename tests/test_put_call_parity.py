import numpy as np
import pytest

from optpricing import BlackScholesModel, PutCallParity


def test_parity_holds_for_bs_prices():
    m = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2)
    result = PutCallParity.verify(m.call_price(), m.put_price(), 100, 100, 0.05, 1)
    assert result["parity_holds"]
    assert result["difference"] == pytest.approx(0.0, abs=1e-8)


def test_implied_forward_price():
    m = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2)
    fwd = PutCallParity.implied_forward_price(
        m.call_price(), m.put_price(), 100, 0.05, 1
    )
    assert fwd == pytest.approx(100 * np.exp(0.05), abs=1e-6)


def test_arbitrage_detected_when_parity_violated():
    m = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2)
    # Inflate the call by $5 to break parity.
    res = PutCallParity.detect_arbitrage(
        m.call_price() + 5, m.put_price(), 100, 100, 0.05, 1
    )
    assert res["arbitrage"]
    assert res["expected_profit"] == pytest.approx(5.0, abs=1e-6)


def test_no_arbitrage_when_parity_holds():
    m = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2)
    res = PutCallParity.detect_arbitrage(
        m.call_price(), m.put_price(), 100, 100, 0.05, 1
    )
    assert res["arbitrage"] is False
