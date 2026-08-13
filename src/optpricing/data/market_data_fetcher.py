"""Fetch and cache real option-chain data for market validation.

This is a convenience layer for the market-validation notebooks. Third-party
market data (via ``yfinance``) is inherently flaky and rate-limited, so every
fetch is cached to ``data/raw/`` as CSV and re-loaded from disk on subsequent
calls, keeping notebooks reproducible offline.

Requires the optional ``data`` extra::

    pip install -e ".[data]"
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from optpricing.data.quotes import mid_prices

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


def _read_cached_chain(ticker, expiry):
    """Return ``(calls, puts, spot)`` from disk, or ``None`` if incomplete.

    ``spot`` is read from the ``_meta`` sidecar when present; if the sidecar is
    missing (older caches predate it) it is estimated offline from the chain
    itself via at-the-money put-call parity, so the cached data stays usable
    without any network access.
    """
    calls_cache = _cache_path(ticker, expiry, "calls")
    puts_cache = _cache_path(ticker, expiry, "puts")
    if not (calls_cache.exists() and puts_cache.exists()):
        return None
    calls = pd.read_csv(calls_cache)
    puts = pd.read_csv(puts_cache)
    meta_cache = _cache_path(ticker, expiry, "meta")
    if meta_cache.exists():
        spot = float(pd.read_csv(meta_cache)["spot"].iloc[0])
    else:
        spot = _estimate_spot_from_chain(calls, puts)
    return calls, puts, spot


def _side(merged, suffix):
    """One leg of a call/put merge, with the suffix stripped off the quote columns."""
    cols = ["bid", "ask", "lastPrice"]
    return merged[[c + suffix for c in cols]].rename(
        columns={c + suffix: c for c in cols}
    )


def _estimate_spot_from_chain(calls, puts):
    """Offline spot proxy: the strike where call and put mids are closest.

    From put-call parity ``C - P = S - K e^{-rT}``, the call/put price gap is
    smallest near ``K ≈ S``, so the at-the-money strike is a reasonable stand-in
    for spot when no cached/live quote is available. Falls back to the median
    strike if prices are unusable.
    """
    try:
        merged = calls.merge(puts, on="strike", suffixes=("_c", "_p"))
        # Rebuild one-sided quote frames so the shared mid-price rule applies to
        # each leg (the merge suffixes the columns it expects).
        c_mid = mid_prices(_side(merged, "_c"))
        p_mid = mid_prices(_side(merged, "_p"))
        gap = np.abs(c_mid - p_mid)
        if len(gap) and np.isfinite(gap).any():
            return float(merged["strike"].iloc[int(np.nanargmin(gap))])
    except Exception:  # pragma: no cover - defensive; any odd frame -> median
        pass
    return float(pd.concat([calls["strike"], puts["strike"]]).median())


def fetch_option_chain(ticker, expiry=None, use_cache=True):
    """Return ``(calls, puts, spot, expiry)`` for a ticker.

    Results are cached to ``data/raw/`` so re-running is reproducible and does
    not hammer the data provider. When ``expiry`` is given and a full cached copy
    (calls, puts and the ``_meta`` spot sidecar) exists, this returns entirely
    from disk without importing ``yfinance`` or touching the network.

    Parameters
    ----------
    ticker : str
        Underlying symbol, e.g. ``"SPY"``.
    expiry : str, optional
        Expiration date ``YYYY-MM-DD``. Defaults to the nearest listed expiry.
    use_cache : bool
        Load from ``data/raw`` when a cached copy exists.
    """
    # 1) Fully-offline fast path: explicit expiry + complete cache -> no yfinance.
    if use_cache and expiry is not None:
        cached = _read_cached_chain(ticker, expiry)
        if cached is not None:
            calls, puts, spot = cached
            return calls, puts, spot, expiry

    # 2) Resolve a default expiry: prefer the live listing, fall back to any
    #    already-cached expiry so the nearest-expiry chain still loads offline.
    #    Already-expired dates are skipped in both cases -- the live listing can
    #    still carry yesterday's expiry, and the on-disk cache is full of them.
    yf = None
    if expiry is None:
        try:
            yf = _import_yfinance()
            listed = list(yf.Ticker(ticker).options)
        except Exception:
            # Missing ``data`` extra, no network, or a throttled provider: an
            # empty listing falls through to the cached expiries below.
            listed = []
        candidates = [e for e in listed if years_to_expiry(e) is not None] or [
            e for e in _cached_expiries(ticker) if years_to_expiry(e) is not None
        ]
        if not candidates:
            raise ValueError(
                f"No unexpired option chain available for {ticker!r} (live or cached)."
            )
        expiry = candidates[0]

    # 3) Serve from cache when a copy exists.
    if use_cache:
        cached = _read_cached_chain(ticker, expiry)
        if cached is not None:
            calls, puts, spot = cached
            return calls, puts, spot, expiry

    # 4) Live fetch, populating the cache (spot sidecar included).
    if yf is None:
        yf = _import_yfinance()
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)
    calls, puts = chain.calls, chain.puts
    try:
        spot = float(tk.history(period="1d")["Close"].iloc[-1])
    except (IndexError, KeyError, ValueError):
        # Empty/failed history (holiday, throttle, delisting) shouldn't sink the
        # whole live fetch -- fall back to the ATM put-call-parity spot proxy.
        spot = _estimate_spot_from_chain(calls, puts)
    calls.to_csv(_cache_path(ticker, expiry, "calls"), index=False)
    puts.to_csv(_cache_path(ticker, expiry, "puts"), index=False)
    pd.DataFrame({"spot": [spot]}).to_csv(
        _cache_path(ticker, expiry, "meta"), index=False
    )
    return calls, puts, spot, expiry


def _cached_expiries(ticker):
    """List expiries for ``ticker`` that already have cached chains on disk."""
    if not RAW_DATA_DIR.exists():
        return []
    expiries = sorted(
        p.name[len(ticker) + 1 : -len("_calls.csv")]
        for p in RAW_DATA_DIR.glob(f"{ticker}_*_calls.csv")
    )
    return expiries


def years_to_expiry(expiry, today=None):
    """Years from ``today`` to an ``YYYY-MM-DD`` expiry, or ``None`` if it passed.

    A same-day (0-DTE) expiry is floored at one day so it still prices, but an
    expiry that is already in the past returns ``None`` and must be dropped by
    the caller. The previous ``max(days, 1)`` floored *every* stale date to one
    day, which silently resurrected expired cached chains: a surface built from
    six long-dead expiries collapsed into a single degenerate 1-day slice and was
    still reported as a term structure.
    """
    if today is None:
        today = pd.Timestamp.now().normalize()
    days = (pd.Timestamp(expiry).normalize() - today).days
    if days < 0:
        return None
    return max(days, 1) / 365.0


def fetch_option_surface(ticker, n_expiries=6, use_cache=True):
    """Return option chains across several expiries for a volatility surface.

    Produces a list of records, one per expiry, each holding the calls/puts
    frames, the spot and the maturity ``T`` in years::

        [{"expiry": "2026-08-15", "T": 0.083, "calls": df, "puts": df,
          "spot": 5000.0}, ...]

    Expiries are sampled *evenly across the whole listed term structure* (not
    just the nearest ``n_expiries``), so the surface spans short- to long-dated
    maturities and reads smoothly. Falls back to any already-cached expiries on
    disk when offline. Each per-expiry chain reuses :func:`fetch_option_chain`,
    inheriting its cache.

    Expiries that have already passed are dropped rather than floored to one day
    (see :func:`years_to_expiry`), so a stale on-disk cache raises here instead
    of yielding a plausible-looking but meaningless surface.
    """

    def _spread(all_exp):
        """Pick ``n_expiries`` unexpired expiries evenly spread across the listing.

        Expired dates are dropped *before* the even spread is computed: spreading
        first and dropping afterwards silently spent one of the ``n_expiries``
        slots on a dead date, so a chain with stale entries came back short.
        """
        all_exp = [e for e in all_exp if years_to_expiry(e) is not None]
        if len(all_exp) <= n_expiries:
            return all_exp
        idx = np.linspace(0, len(all_exp) - 1, n_expiries)
        picks = sorted({int(round(i)) for i in idx})
        return [all_exp[i] for i in picks]

    available = []
    expiries = []
    try:
        yf = _import_yfinance()
        available = list(yf.Ticker(ticker).options)
        expiries = _spread(available) if available else []
    except Exception:
        # Missing ``data`` extra / network / throttling -> fall back to disk.
        expiries = []
    if not expiries:
        available = _cached_expiries(ticker)
        expiries = _spread(available)
    if not expiries:
        stale = [e for e in available if years_to_expiry(e) is None]
        detail = (
            f" (all {len(stale)} available expiries have passed; the on-disk "
            "cache is stale and needs a live refresh)"
            if stale
            else ""
        )
        raise ValueError(
            f"No option expiries available for {ticker!r} (live or cached).{detail}"
        )

    records = []
    for expiry in expiries:
        T = years_to_expiry(expiry)
        try:
            calls, puts, spot, expiry = fetch_option_chain(
                ticker, expiry=expiry, use_cache=use_cache
            )
        except Exception:  # pragma: no cover - skip an expiry that fails to load
            continue
        records.append(
            {
                "expiry": expiry,
                "T": T,
                "calls": calls,
                "puts": puts,
                "spot": spot,
            }
        )
    if not records:
        raise ValueError(f"Could not load any option chain for {ticker!r}.")
    return records


def historical_volatility(ticker, period="1y"):
    """Annualised historical volatility from daily close-to-close log returns.

    **Live-only**: this always queries the provider and raises when it is
    unreachable, which is what the validation scripts want (a report must not
    quietly quote last month's vol). For the offline-safe, cache-backed variant
    over a trailing window use :func:`market_sigma_estimate`.
    """
    yf = _import_yfinance()
    close = yf.Ticker(ticker).history(period=period)["Close"]
    log_returns = np.log(close / close.shift(1)).dropna()
    return float(log_returns.std() * np.sqrt(252))


RISK_FREE_FALLBACK = 0.043

# Treasury yield tickers on Yahoo Finance mapped to their tenor in years. Each
# is quoted as a percentage (e.g. ^TNX close of 4.2 -> 0.042), so divide by 100.
_TREASURY_TENORS = {"^IRX": 0.25, "^FVX": 5.0, "^TNX": 10.0, "^TYX": 30.0}

# Canonical underlyings the toolkit pulls data for. yfinance symbols; the display
# map carries the friendly names. Stocks (AAPL, MSFT) are included because -- unlike
# spot indices -- they expose full option chains on Yahoo, so the vol-surface and
# smile tabs work for them. Add a new underlying in one place here and every
# selector/export picks it up.
SUPPORTED_UNDERLYINGS = ["^SPX", "^NDX", "^RUT", "AAPL", "MSFT"]

UNDERLYING_DISPLAY_NAMES = {
    "^SPX": "S&P 500 (SPX)",
    "^NDX": "Nasdaq-100 (NDX)",
    "^RUT": "Russell 2000 (RUT)",
    "AAPL": "Apple (AAPL)",
    "MSFT": "Microsoft (MSFT)",
}

# Dividend-yield fallback used only when a live yield is unavailable. Spot indices
# don't expose a dividendYield via yfinance; stocks do (computed live), so they
# don't need an entry here (absent -> 0.0 fallback).
_INDEX_DIV_FALLBACK = {"^SPX": 0.013, "^NDX": 0.007, "^RUT": 0.014}


def fetch_risk_free_rate() -> float:
    """Fetch 3-Month US Treasury yield (^IRX) from yfinance as risk-free rate proxy.

    A single short-tenor proxy, for callers that price one near-term maturity.
    Anything that spans maturities (the surface, the dashboard) should use
    :func:`risk_free_rate_for`, which interpolates the whole curve at ``T``.
    """
    try:
        yf = _import_yfinance()
        history = yf.Ticker("^IRX").history(period="1d")
    except Exception:  # network / missing dependency -> fallback
        return RISK_FREE_FALLBACK
    if history.empty:
        return RISK_FREE_FALLBACK  # fallback
    return float(history["Close"].iloc[-1]) / 100.0


def fetch_risk_free_curve() -> list[tuple[float, float]]:
    """Fetch a coarse US Treasury yield curve as ``[(tenor_years, rate), ...]``.

    Pulls the 13-week, 5-, 10- and 30-year constant-maturity yields (^IRX/^FVX/
    ^TNX/^TYX). Tenors that fail to load are skipped; the result is sorted by
    tenor and may be empty if the provider is unreachable.
    """
    yf = _import_yfinance()
    points: list[tuple[float, float]] = []
    for sym, tenor in _TREASURY_TENORS.items():
        try:
            hist = yf.Ticker(sym).history(period="1d")
            if not hist.empty:
                points.append((tenor, float(hist["Close"].iloc[-1]) / 100.0))
        except Exception:  # pragma: no cover - one flaky tenor shouldn't sink the curve
            continue
    points.sort()
    return points


def risk_free_rate_for(
    T: float, curve: list[tuple[float, float]] | None = None
) -> float:
    """Risk-free rate at maturity ``T`` by linear interpolation of the curve.

    ``curve`` may be passed in (e.g. cached, or in tests) to avoid a network
    call; otherwise it is fetched. Outside the curve's tenor range the nearest
    endpoint is used (``numpy.interp`` clamps). Falls back to
    :data:`RISK_FREE_FALLBACK` when no curve is available.
    """
    if curve is None:
        try:
            curve = fetch_risk_free_curve()
        except Exception:
            # Missing ``data`` extra or an unreachable provider: an empty curve
            # means the constant fallback below is used instead.
            curve = []
    if not curve:
        return RISK_FREE_FALLBACK
    tenors = [t for t, _ in curve]
    rates = [r for _, r in curve]
    return float(np.interp(T, tenors, rates))


# Any yield above this (25%) is treated as implausible-as-a-fraction: it must be
# a percentage-form figure (or bad data), so it is rescaled / rejected.
_MAX_PLAUSIBLE_DIV_YIELD = 0.25


def fetch_dividend_yield(ticker: str) -> float:
    """Dividend yield for ``ticker``: live from yfinance, else a fallback.

    yfinance reports the yield inconsistently -- as a fraction (``0.013``) in
    some versions and a percentage (``1.3``) in others -- and the old
    ``value > 1.0`` heuristic silently mis-scaled sub-1% percentage-form yields
    by 100x (``0.5`` meaning 0.5% was left as 50%). To avoid the unit ambiguity
    we first compute the yield directly from the absolute annual dividend
    (``trailingAnnualDividendRate``, in $/share) over the quoted price, which is
    unit-unambiguous. Only if that is unavailable do we fall back to the reported
    yield field, rescaling percentage-form values and accepting only a sane band
    (``0 < y < 25%``). When nothing plausible is live, falls back to the known
    index constants (or 0.0).
    """
    try:
        yf = _import_yfinance()
        info = yf.Ticker(ticker).info

        # 1) Unit-unambiguous: absolute annual dividend / price.
        rate = info.get("trailingAnnualDividendRate")
        price = (
            info.get("regularMarketPrice")
            or info.get("currentPrice")
            or info.get("previousClose")
        )
        if rate and price:
            computed = float(rate) / float(price)
            if 0 < computed < _MAX_PLAUSIBLE_DIV_YIELD:
                return computed

        # 2) Fall back to the reported yield field, normalising units.
        yield_val = info.get("dividendYield")
        if yield_val is None:
            yield_val = info.get("trailingAnnualDividendYield")
        if yield_val:
            yield_val = float(yield_val)
            if yield_val > _MAX_PLAUSIBLE_DIV_YIELD:
                # Too large to be a fraction -> percentage form; rescale.
                warnings.warn(
                    f"dividendYield for {ticker} looks percentage-form "
                    f"({yield_val}); rescaling by /100.",
                    stacklevel=2,
                )
                yield_val /= 100.0
            if 0 < yield_val < _MAX_PLAUSIBLE_DIV_YIELD:
                return yield_val
    except Exception:  # pragma: no cover - offline / missing info -> fallback
        pass
    return _INDEX_DIV_FALLBACK.get(ticker, 0.0)


def market_sigma_estimate(
    ticker,
    window: int = 30,
    use_cache: bool = True,
    max_age_hours: float | None = None,
):
    """Annualised realised volatility over the last ``window`` trading days.

    Reuses the cached historical series (:func:`fetch_10y_historical_data`) so it
    works offline, and returns ``None`` when there isn't enough data.
    ``max_age_hours`` is forwarded to bound how stale that cache may be.

    The live-only sibling :func:`historical_volatility` computes the same
    quantity straight from the provider; prefer this one anywhere the result
    must still be available without a network.
    """
    df = fetch_10y_historical_data(
        ticker, use_cache=use_cache, max_age_hours=max_age_hours
    )
    if df.empty or "Close" not in df.columns:
        return None
    returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    if len(returns) < window:
        return None
    return float(returns.tail(window).std() * np.sqrt(252))


def fetch_market_snapshot(
    ticker,
    T: float = 1.0,
    use_cache: bool = True,
    max_age_hours: float | None = None,
) -> dict:
    """Compose a live market snapshot to auto-fill model inputs.

    Returns ``{"spot", "r", "q", "hist_vol", "as_of"}`` where ``spot`` is the
    latest cached close, ``r`` the interpolated risk-free rate at ``T``, ``q``
    the dividend yield and ``hist_vol`` a realised-vol estimate. Every component
    degrades gracefully (``None`` / fallback) so a partial snapshot is still
    returned when the provider is flaky.

    ``spot`` and ``hist_vol`` both come from the historical series, so
    ``max_age_hours`` (forwarded to :func:`fetch_10y_historical_data`) is what
    stops a months-old CSV from being presented as a fresh sync. ``as_of``
    always reports the date actually used.
    """
    try:
        df = fetch_10y_historical_data(
            ticker, use_cache=use_cache, max_age_hours=max_age_hours
        )
    except Exception:  # missing dep / network / no cache -> partial snapshot
        df = pd.DataFrame()
    if not df.empty and "Close" in df.columns:
        spot = float(df["Close"].iloc[-1])
        as_of = str(pd.Timestamp(df.index[-1]).date())
    else:
        spot, as_of = None, None
    try:
        hist_vol = market_sigma_estimate(
            ticker, use_cache=use_cache, max_age_hours=max_age_hours
        )
    except Exception:  # same failures as the spot read -> partial snapshot
        hist_vol = None
    return {
        "spot": spot,
        "r": risk_free_rate_for(T),
        "q": fetch_dividend_yield(ticker),
        "hist_vol": hist_vol,
        "as_of": as_of,
    }


def _read_historical_cache(cache_file: Path) -> pd.DataFrame:
    """Load a cached historical CSV back into a Date-indexed frame."""
    df = pd.read_csv(cache_file, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    return df


def fetch_10y_historical_data(
    ticker: str,
    use_cache: bool = True,
    period: str = "10y",
    max_age_hours: float | None = None,
) -> pd.DataFrame:
    """Fetch historical daily data for a ticker and cache it.

    ``period`` accepts anything ``yfinance``'s ``Ticker.history`` supports
    (e.g. ``"10y"``, ``"5y"``, ``"max"``) and defaults to ``"10y"`` to keep
    this function's original behaviour and on-disk cache filename for
    existing callers.

    ``max_age_hours`` bounds how old the on-disk cache may be before a live
    refetch is attempted:

    * ``None`` (default) -- any cached copy is served, however old. This is the
      historical behaviour and what the offline notebooks/tests rely on.
    * a number -- the cache is used only while it is younger than that many
      hours; otherwise the provider is queried and the cache rewritten.

    A live fetch that fails (or comes back empty) falls back to the stale cache
    rather than propagating, so the offline-first guarantee holds: the network
    is a bonus, not a requirement. With no cache at all there is nothing to fall
    back to, and the original error (e.g. the "install the ``data`` extra"
    ``ImportError``) is re-raised unchanged.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = RAW_DATA_DIR / f"{ticker}_{period}_historical.csv"
    fresh = cache_file.exists() and (
        max_age_hours is None
        or (time.time() - cache_file.stat().st_mtime) < max_age_hours * 3600.0
    )
    if use_cache and fresh:
        return _read_historical_cache(cache_file)

    try:
        df = _import_yfinance().Ticker(ticker).history(period=period)
    except Exception:  # missing ``data`` extra / network / throttling
        if use_cache and cache_file.exists():
            return _read_historical_cache(cache_file)  # stale beats nothing
        raise  # nothing to serve -> let the original error speak
    if df.empty:
        if use_cache and cache_file.exists():
            return _read_historical_cache(cache_file)
        return df
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
        tickers = list(SUPPORTED_UNDERLYINGS)
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
            (
                ts.tz_localize(None)
                if isinstance(ts, pd.Timestamp) and ts.tzinfo is not None
                else ts
            )
            for ts in df.index
        )
        data_by_ticker[ticker] = df

    if not data_by_ticker:
        raise ValueError(
            f"No historical data available for any of {tickers}; workbook not written."
        )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for ticker, df in data_by_ticker.items():
            df.to_excel(writer, sheet_name=ticker)

    return output_path
