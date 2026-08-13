"""Historical-data fetch tests with yfinance monkeypatched out (no network)."""

import os
import time
import types

import numpy as np
import pandas as pd
import pytest

from optpricing.data import market_data_fetcher as mdf


def _fake_yf(df):
    class FakeTicker:
        def __init__(self, _):
            pass

        def history(self, period="10y"):
            return df

    return types.SimpleNamespace(Ticker=FakeTicker)


def _frame(close_start, close_end, periods=2500, start="2015-01-01"):
    idx = pd.date_range(start, periods=periods, freq="D")
    df = pd.DataFrame(
        {"Close": np.linspace(close_start, close_end, periods)}, index=idx
    )
    df.index.name = "Date"  # yfinance names its index "Date"; the cache relies on it
    return df


def _age_cache(cache_file, hours):
    """Backdate a cache file's mtime by ``hours``."""
    old = time.time() - hours * 3600.0
    os.utime(cache_file, (old, old))


def test_fetch_10y_data_live_then_caches(monkeypatch, tmp_path):
    df = _frame(100.0, 200.0)
    monkeypatch.setattr(mdf, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(mdf, "_import_yfinance", lambda: _fake_yf(df))

    out = mdf.fetch_10y_historical_data("^SPX", use_cache=False)
    assert not out.empty
    assert len(out) > 2000
    assert "Close" in out.columns

    # The live fetch wrote a cache file; the second call re-reads it from disk
    # without touching yfinance at all.
    def boom():
        raise RuntimeError("no network")

    monkeypatch.setattr(mdf, "_import_yfinance", boom)
    cached = mdf.fetch_10y_historical_data("^SPX", use_cache=True)
    assert len(cached) == len(out)
    assert "Close" in cached.columns


# --- cache expiry ----------------------------------------------------------- #
# Without a max age the on-disk CSV was served forever: a year-old file kept
# feeding last year's spot (and realised vol) into the dashboard's "Sync from
# market", which reported it as freshly synced.


def test_cache_used_when_fresh(monkeypatch, tmp_path):
    """A cache younger than max_age_hours must not touch the provider."""
    monkeypatch.setattr(mdf, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(mdf, "_import_yfinance", lambda: _fake_yf(_frame(100.0, 200.0)))
    first = mdf.fetch_10y_historical_data("^SPX", use_cache=False)

    def boom():
        raise AssertionError("provider must not be queried for a fresh cache")

    monkeypatch.setattr(mdf, "_import_yfinance", boom)
    out = mdf.fetch_10y_historical_data("^SPX", max_age_hours=24)
    assert out["Close"].iloc[-1] == pytest.approx(first["Close"].iloc[-1])


def test_cache_bypassed_when_stale(monkeypatch, tmp_path):
    """A cache older than max_age_hours triggers a live refetch."""
    monkeypatch.setattr(mdf, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(mdf, "_import_yfinance", lambda: _fake_yf(_frame(100.0, 200.0)))
    mdf.fetch_10y_historical_data("^SPX", use_cache=False)
    _age_cache(tmp_path / "^SPX_10y_historical.csv", hours=48)

    # New data on the wire: the stale cache must not shadow it.
    monkeypatch.setattr(mdf, "_import_yfinance", lambda: _fake_yf(_frame(300.0, 400.0)))
    out = mdf.fetch_10y_historical_data("^SPX", max_age_hours=24)
    assert out["Close"].iloc[-1] == pytest.approx(400.0)


def test_stale_cache_returned_when_fetch_fails(monkeypatch, tmp_path):
    """Offline-first: a failed refetch falls back to the stale copy, not an error."""
    monkeypatch.setattr(mdf, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(mdf, "_import_yfinance", lambda: _fake_yf(_frame(100.0, 200.0)))
    mdf.fetch_10y_historical_data("^SPX", use_cache=False)
    _age_cache(tmp_path / "^SPX_10y_historical.csv", hours=48)

    def boom():
        raise RuntimeError("no network")

    monkeypatch.setattr(mdf, "_import_yfinance", boom)
    out = mdf.fetch_10y_historical_data("^SPX", max_age_hours=24)
    assert out["Close"].iloc[-1] == pytest.approx(200.0)


def test_fetch_raises_when_no_cache_to_fall_back_on(monkeypatch, tmp_path):
    """With nothing cached, the provider's own error must survive intact."""
    monkeypatch.setattr(mdf, "RAW_DATA_DIR", tmp_path)

    def boom():
        raise ImportError("yfinance is required for live data")

    monkeypatch.setattr(mdf, "_import_yfinance", boom)
    with pytest.raises(ImportError, match="yfinance is required"):
        mdf.fetch_10y_historical_data("^SPX", max_age_hours=24)


def test_max_age_none_preserves_behaviour(monkeypatch, tmp_path):
    """The default keeps serving any cache, however old (regression guard)."""
    monkeypatch.setattr(mdf, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(mdf, "_import_yfinance", lambda: _fake_yf(_frame(100.0, 200.0)))
    mdf.fetch_10y_historical_data("^SPX", use_cache=False)
    _age_cache(tmp_path / "^SPX_10y_historical.csv", hours=24 * 365)

    def boom():
        raise AssertionError("default must not re-fetch")

    monkeypatch.setattr(mdf, "_import_yfinance", boom)
    out = mdf.fetch_10y_historical_data("^SPX")
    assert out["Close"].iloc[-1] == pytest.approx(200.0)


def test_snapshot_forwards_max_age(monkeypatch, tmp_path):
    """The snapshot's spot/vol honour the age bound, not just the fetcher."""
    monkeypatch.setattr(mdf, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(mdf, "_import_yfinance", lambda: _fake_yf(_frame(100.0, 200.0)))
    mdf.fetch_10y_historical_data("^SPX", use_cache=False)
    _age_cache(tmp_path / "^SPX_10y_historical.csv", hours=48)

    monkeypatch.setattr(mdf, "_import_yfinance", lambda: _fake_yf(_frame(300.0, 400.0)))
    monkeypatch.setattr(mdf, "risk_free_rate_for", lambda T, curve=None: 0.04)
    monkeypatch.setattr(mdf, "fetch_dividend_yield", lambda t: 0.01)

    snap = mdf.fetch_market_snapshot("^SPX", max_age_hours=24)
    assert snap["spot"] == pytest.approx(400.0)
