import pytest

from optpricing import BlackScholesModel, Greeks, GreeksNumerical

PARAMS = dict(S=100, K=105, T=1.0, r=0.05, sigma=0.25, q=0.02)


def test_delta_call_matches_numerical():
    g = Greeks(**PARAMS)
    num = GreeksNumerical.delta(BlackScholesModel, **PARAMS, option_type="call")
    assert g.delta_call() == pytest.approx(num, abs=1e-4)


def test_delta_put_matches_numerical():
    g = Greeks(**PARAMS)
    num = GreeksNumerical.delta(BlackScholesModel, **PARAMS, option_type="put")
    assert g.delta_put() == pytest.approx(num, abs=1e-4)


def test_gamma_matches_numerical():
    g = Greeks(**PARAMS)
    num = GreeksNumerical.gamma(BlackScholesModel, **PARAMS)
    assert g.gamma() == pytest.approx(num, abs=1e-4)


def test_vega_matches_numerical():
    g = Greeks(**PARAMS)
    num = GreeksNumerical.vega(BlackScholesModel, **PARAMS) / 100.0
    assert g.vega() == pytest.approx(num, abs=1e-4)


def test_theta_call_matches_numerical():
    g = Greeks(**PARAMS)
    num = GreeksNumerical.theta(BlackScholesModel, **PARAMS, option_type="call") / 365.0
    assert g.theta_call() == pytest.approx(num, abs=1e-4)


def test_rho_call_matches_numerical():
    g = Greeks(**PARAMS)
    num = GreeksNumerical.rho(BlackScholesModel, **PARAMS, option_type="call") / 100.0
    assert g.rho_call() == pytest.approx(num, abs=1e-4)


def test_theta_put_matches_numerical():
    g = Greeks(**PARAMS)
    num = GreeksNumerical.theta(BlackScholesModel, **PARAMS, option_type="put") / 365.0
    assert g.theta_put() == pytest.approx(num, abs=1e-4)


def test_rho_put_matches_numerical():
    g = Greeks(**PARAMS)
    num = GreeksNumerical.rho(BlackScholesModel, **PARAMS, option_type="put") / 100.0
    assert g.rho_put() == pytest.approx(num, abs=1e-4)


def test_vanna_matches_numerical():
    # vanna is raw (unscaled) on both sides and identical for call/put.
    g = Greeks(**PARAMS)
    num = GreeksNumerical.vanna(BlackScholesModel, **PARAMS)
    assert g.vanna() == pytest.approx(num, abs=1e-3)


def test_vomma_matches_numerical():
    g = Greeks(**PARAMS)
    num = GreeksNumerical.vomma(BlackScholesModel, **PARAMS)
    assert g.vomma() == pytest.approx(num, abs=1e-2)


def test_charm_call_matches_numerical():
    g = Greeks(**PARAMS)
    num = GreeksNumerical.charm(BlackScholesModel, **PARAMS, option_type="call") / 365.0
    assert g.charm_call() == pytest.approx(num, abs=1e-4)


def test_charm_put_matches_numerical():
    g = Greeks(**PARAMS)
    num = GreeksNumerical.charm(BlackScholesModel, **PARAMS, option_type="put") / 365.0
    assert g.charm_put() == pytest.approx(num, abs=1e-4)


def test_gamma_positive_and_shared():
    # Gamma is identical for calls and puts and strictly positive.
    assert Greeks(**PARAMS).gamma() > 0


def test_get_all_greeks_contains_expected_keys():
    call_greeks = Greeks(**PARAMS).get_all_greeks("call")
    for key in ("delta", "gamma", "vega", "theta", "rho", "vanna", "vomma", "charm"):
        assert key in call_greeks
