import numpy as np
from optpricing import BlackScholesModel, MonteCarloOptionPricer

BS = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2)


def test_mc_call_within_confidence_interval():
    bs_price = BS.call_price()
    mc = MonteCarloOptionPricer(100, 100, 1, 0.05, 0.2, n_simulations=200_000, seed=42)
    price, std_err = mc.call_price(return_std_error=True)
    assert abs(price - bs_price) < 4 * std_err


def test_mc_put_within_confidence_interval():
    bs_price = BS.put_price()
    mc = MonteCarloOptionPricer(100, 100, 1, 0.05, 0.2, n_simulations=200_000, seed=7)
    price, std_err = mc.put_price(return_std_error=True)
    assert abs(price - bs_price) < 4 * std_err


def test_seed_zero_is_reproducible():
    # ``if seed:`` would have skipped seeding for seed=0; default_rng does not.
    a = MonteCarloOptionPricer(100, 100, 1, 0.05, 0.2, n_simulations=10_000, seed=0)
    b = MonteCarloOptionPricer(100, 100, 1, 0.05, 0.2, n_simulations=10_000, seed=0)
    assert a.call_price() == b.call_price()


def test_antithetic_reduces_standard_error():
    kw = dict(S=100, K=100, T=1, r=0.05, sigma=0.2, n_simulations=100_000)
    _, se_plain = MonteCarloOptionPricer(**kw, seed=1).call_price(
        antithetic=False, return_std_error=True
    )
    _, se_anti = MonteCarloOptionPricer(**kw, seed=1).call_price(
        antithetic=True, return_std_error=True
    )
    assert se_anti < se_plain


def test_convergence_analysis_shrinks_std_error():
    mc = MonteCarloOptionPricer(100, 100, 1, 0.05, 0.2, n_simulations=500, seed=3)
    results = mc.convergence_analysis(simulation_counts=[1_000, 10_000, 100_000])
    assert results[-1]["std_error"] < results[0]["std_error"]
    # The instance's own n_simulations is restored after the sweep.
    assert mc.n_simulations == 500


def test_generate_paths():
    mc = MonteCarloOptionPricer(100, 100, 1, 0.05, 0.2, n_simulations=100, seed=42)
    paths = mc.generate_paths(n_paths=10)
    assert paths.shape == (253, 10)  # 252 steps + 1 starting point
    assert np.all(paths[0] == 100)

