"""Fetch and cache real option-chain data for market validation.

This is a convenience layer for the market-validation notebooks. Third-party
market data (via ``yfinance``) is inherently flaky and rate-limited, so every
fetch is cached to ``data/raw/`` as CSV and re-loaded from disk on subsequent
calls, keeping notebooks reproducible offline.

Requires the optional ``data`` extra::

    pip install -e ".[data]"
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# repo_root/data/raw  (this file lives at src/optpricing/data/market_data_fetcher.py)
RAW_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
EXPORTS_DIR = Path(__file__).resolve().parents[3] / "data" / "exports"


def _import_yfinance():
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            'yfinance is required for live data. Install it with: pip install -e ".[data]"'
        ) from exc
    return yf


def _cache_path(ticker, expiry, kind):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DATA_DIR / f"{ticker}_{expiry}_{kind}.csv"


def fetch_option_chain(ticker, expiry=None, use_cache=True):
    """Return ``(calls, puts, spot, expiry)`` for a ticker.

    Results are cached to ``data/raw/`` so re-running is reproducible and does
    not hammer the data provider.

    Parameters
    ----------
    ticker : str
        Underlying symbol, e.g. ``"SPY"``.
    expiry : str, optional
        Expiration date ``YYYY-MM-DD``. Defaults to the nearest listed expiry.
    use_cache : bool
        Load from ``data/raw`` when a cached copy exists.
    """
    yf = _import_yfinance()
    tk = yf.Ticker(ticker)
    if expiry is None:
        if not tk.options:
            raise ValueError(
                f"No listed option chain available for {ticker!r} on Yahoo Finance."
            )
        expiry = tk.options[0]

    calls_cache = _cache_path(ticker, expiry, "calls")
    puts_cache = _cache_path(ticker, expiry, "puts")

    if use_cache and calls_cache.exists() and puts_cache.exists():
        calls = pd.read_csv(calls_cache)
        puts = pd.read_csv(puts_cache)
    else:
        chain = tk.option_chain(expiry)
        calls, puts = chain.calls, chain.puts
        calls.to_csv(calls_cache, index=False)
        puts.to_csv(puts_cache, index=False)

    spot = float(tk.history(period="1d")["Close"].iloc[-1])
    return calls, puts, spot, expiry


def historical_volatility(ticker, period="1y"):
    """Annualised historical volatility from daily close-to-close log returns."""
    yf = _import_yfinance()
    close = yf.Ticker(ticker).history(period=period)["Close"]
    log_returns = np.log(close / close.shift(1)).dropna()
    return float(log_returns.std() * np.sqrt(252))


def fetch_risk_free_rate() -> float:
    """Fetch 3-Month US Treasury yield (^IRX) from yfinance as risk-free rate proxy."""
    yf = _import_yfinance()
    irx = yf.Ticker("^IRX")
    history = irx.history(period="1d")
    if history.empty:
        return 0.043  # fallback
    return float(history["Close"].iloc[-1]) / 100.0


def fetch_dividend_yield(ticker: str) -> float:
    """Fetch dividend yield for a given ticker or return fallback."""
    if ticker == "^SPX":
        return 0.013
    if ticker == "^NDX":
        return 0.007
    if ticker == "^RUT":
        return 0.014
    yf = _import_yfinance()
    tk = yf.Ticker(ticker)
    info = tk.info
    yield_val = info.get("dividendYield", None)
    if yield_val is None:
        yield_val = info.get("trailingAnnualDividendYield", 0.0)
    return float(yield_val)


def fetch_10y_historical_data(
    ticker: str, use_cache: bool = True, period: str = "10y"
) -> pd.DataFrame:
    """Fetch historical daily data for a ticker and cache it.

    ``period`` accepts anything ``yfinance``'s ``Ticker.history`` supports
    (e.g. ``"10y"``, ``"5y"``, ``"max"``) and defaults to ``"10y"`` to keep
    this function's original behaviour and on-disk cache filename for
    existing callers.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = RAW_DATA_DIR / f"{ticker}_{period}_historical.csv"
    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["Date"])
        df.set_index("Date", inplace=True)
        return df
    yf = _import_yfinance()
    tk = yf.Ticker(ticker)
    df = tk.history(period=period)
    if not df.empty:
        df.to_csv(cache_file)
    return df


def export_indices_to_excel(
    tickers: list[str] | None = None,
    period: str = "max",
    output_path: str | Path | None = None,
    use_cache: bool = True,
) -> Path:
    """Fetch historical daily data for each ticker and write it to a single
    multi-sheet ``.xlsx`` workbook (one sheet per ticker).

    Requires the optional ``data`` extra (``pip install -e ".[data]"``), which
    provides both ``yfinance`` (fetching) and ``openpyxl`` (writing).
    """
    if tickers is None:
        tickers = ["^SPX", "^NDX", "^RUT"]
    if output_path is None:
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = EXPORTS_DIR / "indices_historical_data.xlsx"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    data_by_ticker: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = fetch_10y_historical_data(ticker, use_cache=use_cache, period=period)
        if df.empty:
            warnings.warn(f"No historical data returned for {ticker!r}; sheet skipped.")
            continue
        df = df.copy()
        # Excel/openpyxl cannot write timezone-aware datetimes. A fresh yfinance
        # fetch gives a proper tz-aware DatetimeIndex, but reloading from the CSV
        # cache can come back as a plain object Index of individually tz-aware
        # Timestamps (mixed UTC offsets across DST transitions don't collapse
        # into a single DatetimeIndex tz on re-parse) -- strip tz per-element so
        # both shapes are handled.
        df.index = pd.Index(
            ts.tz_localize(None) if isinstance(ts, pd.Timestamp) and ts.tzinfo is not None else ts
            for ts in df.index
        )
        data_by_ticker[ticker] = df

    if not data_by_ticker:
        raise ValueError(f"No historical data available for any of {tickers}; workbook not written.")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for ticker, df in data_by_ticker.items():
            df.to_excel(writer, sheet_name=ticker)

    return output_path


