"""Live-data helper tests with yfinance monkeypatched out (no network)."""

import types

import numpy as np
import pandas as pd
import pytest

from optpricing.data import market_data_fetcher as mdf


# --- risk-free curve interpolation ----------------------------------------- #
def test_risk_free_rate_for_interpolates_and_clamps():
    curve = [(0.25, 0.05), (5.0, 0.04), (10.0, 0.045)]
    # Exact tenor hits.
    assert mdf.risk_free_rate_for(0.25, curve) == pytest.approx(0.05)
    assert mdf.risk_free_rate_for(10.0, curve) == pytest.approx(0.045)
    # Clamp outside the range to the nearest endpoint.
    assert mdf.risk_free_rate_for(0.0, curve) == pytest.approx(0.05)
    assert mdf.risk_free_rate_for(30.0, curve) == pytest.approx(0.045)
    # Linear interpolation between the first two tenors.
    mid = mdf.risk_free_rate_for((0.25 + 5.0) / 2, curve)
    assert 0.04 < mid < 0.05


def test_risk_free_rate_for_empty_curve_uses_fallback():
    assert mdf.risk_free_rate_for(1.0, []) == mdf.RISK_FREE_FALLBACK


# --- dividend yield fallback / normalisation -------------------------------- #
def test_dividend_yield_falls_back_when_offline(monkeypatch):
    def boom():
        raise RuntimeError("offline")

    monkeypatch.setattr(mdf, "_import_yfinance", boom)
    assert mdf.fetch_dividend_yield("^SPX") == 0.013
    assert mdf.fetch_dividend_yield("UNKNOWN") == 0.0


def test_dividend_yield_normalises_percentage(monkeypatch):
    class FakeTicker:
        def __init__(self, _):
            pass

        @property
        def info(self):
            return {"dividendYield": 1.3}  # provider returned a percentage

    monkeypatch.setattr(
        mdf, "_import_yfinance", lambda: types.SimpleNamespace(Ticker=FakeTicker)
    )
    assert mdf.fetch_dividend_yield("AAPL") == pytest.approx(0.013)


# --- market snapshot composition ------------------------------------------- #
def test_market_snapshot_shape(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"Close": np.linspace(100.0, 110.0, 10)}, index=idx)
    monkeypatch.setattr(mdf, "fetch_10y_historical_data", lambda *a, **k: df)
    monkeypatch.setattr(mdf, "risk_free_rate_for", lambda T, curve=None: 0.04)
    monkeypatch.setattr(mdf, "fetch_dividend_yield", lambda t: 0.01)

    snap = mdf.fetch_market_snapshot("^SPX", T=1.0)
    assert set(snap) == {"spot", "r", "q", "hist_vol", "as_of"}
    assert snap["spot"] == pytest.approx(110.0)
    assert snap["r"] == 0.04
    assert snap["q"] == 0.01
    assert snap["as_of"] == "2020-01-10"


# --- pure helpers ----------------------------------------------------------- #
def test_years_to_expiry_floored_at_one_day():
    today = pd.Timestamp("2026-01-01")
    assert mdf.years_to_expiry("2027-01-01", today=today) == pytest.approx(365 / 365.0)
    # A same-day (0-DTE) expiry is floored rather than going to zero.
    assert mdf.years_to_expiry("2026-01-01", today=today) == pytest.approx(1 / 365.0)


def test_years_to_expiry_rejects_past_expiry():
    """A dead expiry must be rejected, not resurrected as a 1-day option.

    The old ``max(days, 1)`` floor mapped *every* stale date to 1/365, so a
    surface built from six long-expired cached chains collapsed into a single
    degenerate 1-day slice and was still presented as a term structure.
    """
    today = pd.Timestamp("2026-01-01")
    assert mdf.years_to_expiry("2025-01-01", today=today) is None
    assert mdf.years_to_expiry("2025-12-31", today=today) is None


def test_fetch_option_surface_raises_when_every_expiry_expired(monkeypatch):
    """Stale cache -> explicit error, so the dashboard shows its warning banner."""
    monkeypatch.setattr(
        mdf, "_import_yfinance", lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    )
    monkeypatch.setattr(mdf, "_cached_expiries", lambda t: ["2020-01-01", "2020-02-01"])
    monkeypatch.setattr(
        mdf, "fetch_option_chain", lambda *a, **k: pytest.fail("must not be fetched")
    )
    with pytest.raises(ValueError, match="expiries have passed"):
        mdf.fetch_option_surface("^SPX")


def test_estimate_spot_from_chain_picks_atm():
    # Call/put mids cross near strike 100 -> spot proxy should be ~100.
    calls = pd.DataFrame(
        {
            "strike": [90, 100, 110],
            "bid": [12, 5, 1],
            "ask": [13, 6, 2],
            "lastPrice": [12.5, 5.5, 1.5],
        }
    )
    puts = pd.DataFrame(
        {
            "strike": [90, 100, 110],
            "bid": [1, 5, 12],
            "ask": [2, 6, 13],
            "lastPrice": [1.5, 5.5, 12.5],
        }
    )
    assert mdf._estimate_spot_from_chain(calls, puts) == pytest.approx(100.0)
