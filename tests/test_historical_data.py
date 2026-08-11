"""Historical-data fetch tests with yfinance monkeypatched out (no network)."""

import types

import numpy as np
import pandas as pd

from optpricing.data import market_data_fetcher as mdf


def _fake_yf(df):
    class FakeTicker:
        def __init__(self, _):
            pass

        def history(self, period="10y"):
            return df

    return types.SimpleNamespace(Ticker=FakeTicker)


def test_fetch_10y_data_live_then_caches(monkeypatch, tmp_path):
    idx = pd.date_range("2015-01-01", periods=2500, freq="D")
    df = pd.DataFrame({"Close": np.linspace(100.0, 200.0, 2500)}, index=idx)
    df.index.name = "Date"  # yfinance names its index "Date"; the cache relies on it
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
