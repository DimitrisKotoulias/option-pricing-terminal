import pytest

from optpricing import BinomialTreeModel, BlackScholesModel


def test_binomial_converges_to_bs_call():
    bs = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2)
    bt = BinomialTreeModel(S=100, K=100, T=1, r=0.05, sigma=0.2, n_steps=2000)
    assert bt.call_price() == pytest.approx(bs.call_price(), abs=1e-2)


def test_binomial_converges_to_bs_put():
    bs = BlackScholesModel(S=100, K=110, T=1, r=0.05, sigma=0.3)
    bt = BinomialTreeModel(S=100, K=110, T=1, r=0.05, sigma=0.3, n_steps=2000)
    assert bt.put_price() == pytest.approx(bs.put_price(), abs=1e-2)


def test_convergence_error_decreases_with_steps():
    bs = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2)
    bt = BinomialTreeModel(S=100, K=100, T=1, r=0.05, sigma=0.2)
    results = bt.convergence_to_bs(bs.call_price(), step_range=[10, 100, 1000])
    errors = [row["absolute_error"] for row in results]
    assert errors[-1] < errors[0]
    # The instance's own n_steps is restored after the sweep.
    assert bt.n_steps == 500


def test_american_put_at_least_european():
    euro = BinomialTreeModel(
        S=100, K=110, T=1, r=0.05, sigma=0.3, n_steps=500
    ).put_price()
    amer = BinomialTreeModel(
        S=100, K=110, T=1, r=0.05, sigma=0.3, n_steps=500, american=True
    ).put_price()
    # Early exercise can only add value to an American option.
    assert amer >= euro - 1e-9
