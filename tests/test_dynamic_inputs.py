"""Market-input fetch tests with yfinance monkeypatched out (no network)."""

import types

import pandas as pd
import pytest

from optpricing.data import market_data_fetcher as mdf


def _fake_yf(close_values):
    class FakeTicker:
        def __init__(self, _):
            pass

        def history(self, period="1d"):
            return pd.DataFrame({"Close": list(close_values)})

    return types.SimpleNamespace(Ticker=FakeTicker)


def test_fetch_risk_free_rate_parses_irx(monkeypatch):
    # ^IRX is quoted in percent, so a 4.25 close -> 0.0425 rate.
    monkeypatch.setattr(mdf, "_import_yfinance", lambda: _fake_yf([4.25]))
    r = mdf.fetch_risk_free_rate()
    assert r == pytest.approx(0.0425)
    assert 0.0 < r < 0.15


def test_fetch_risk_free_rate_empty_uses_fallback(monkeypatch):
    monkeypatch.setattr(mdf, "_import_yfinance", lambda: _fake_yf([]))
    assert mdf.fetch_risk_free_rate() == mdf.RISK_FREE_FALLBACK


def test_fetch_dividend_yield_index_fallback(monkeypatch):
    def boom():
        raise RuntimeError("offline")

    monkeypatch.setattr(mdf, "_import_yfinance", boom)
    q = mdf.fetch_dividend_yield("^SPX")
    assert 0.0 <= q < 0.10
    assert q == pytest.approx(0.013)  # the ^SPX index fallback constant
