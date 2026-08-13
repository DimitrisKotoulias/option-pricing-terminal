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


def test_american_call_has_positive_early_exercise_premium():
    """A dividend-paying underlying makes early exercise of a call worthwhile.

    The call side of the early-exercise branch had no test at all, so a
    regression there (e.g. comparing against the put payoff) would have priced
    silently wrong instead of failing.
    """
    args = dict(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0.08, n_steps=500)
    american = BinomialTreeModel(**args, american=True).call_price()
    european = BinomialTreeModel(**args, american=False).call_price()
    assert american > european


def test_american_call_equals_european_without_dividends():
    # With q = 0 it is never optimal to exercise an American call early, so the
    # two prices must coincide exactly -- not merely be close.
    args = dict(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0.0, n_steps=500)
    american = BinomialTreeModel(**args, american=True).call_price()
    european = BinomialTreeModel(**args, american=False).call_price()
    assert american == pytest.approx(european, abs=1e-12)


def test_non_positive_n_steps_raises():
    # Guard against dt = T / n_steps blowing up (ZeroDivisionError) or nonsense.
    with pytest.raises(ValueError):
        BinomialTreeModel(S=100, K=100, T=1, r=0.05, sigma=0.2, n_steps=0)
    with pytest.raises(ValueError):
        BinomialTreeModel(S=100, K=100, T=1, r=0.05, sigma=0.2, n_steps=-5)


def test_arbitrage_violation_raises_instead_of_wrong_price():
    # Small vol vs large drift with few steps pushes the risk-neutral prob out
    # of [0, 1]. Previously this silently returned an absurd price (~8640 vs a
    # ~39 Black-Scholes reference); it must now raise instead.
    bt = BinomialTreeModel(S=100, K=100, T=1, r=0.5, sigma=0.05, n_steps=10)
    with pytest.raises(ValueError, match="no-arbitrage"):
        bt.call_price()


def test_arbitrage_condition_restored_by_more_steps():
    # The very same params converge cleanly once dt is small enough that the
    # per-step vol move outruns the per-step drift.
    bs = BlackScholesModel(S=100, K=100, T=1, r=0.5, sigma=0.05)
    bt = BinomialTreeModel(S=100, K=100, T=1, r=0.5, sigma=0.05, n_steps=5000)
    assert bt.call_price() == pytest.approx(bs.call_price(), abs=1e-2)
