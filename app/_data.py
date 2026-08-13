"""Streamlit-cached market-data wrappers, plus small chain/display helpers.

The library stays pure (fetch + on-disk CSV cache); Streamlit's in-memory cache
lives here so dragging sliders never re-reads disk or re-hits the network within
a session.
"""

from __future__ import annotations

import datetime as dt

import streamlit as st

from optpricing.data.market_data_fetcher import (
    UNDERLYING_DISPLAY_NAMES,
    fetch_10y_historical_data,
    fetch_dividend_yield,
    fetch_market_snapshot,
    fetch_option_chain,
    fetch_option_surface,
    fetch_risk_free_curve,
)
from optpricing.data.quotes import mid_prices  # noqa: F401 - re-exported for the tabs

# Streamlit's cache has no expiry unless one is given, so without a ttl the very
# first read of a session was pinned for the whole session -- pressing "Sync from
# market" a second time just replayed the memoized values. 15 minutes is short
# enough that an intraday quote can't go stale unnoticed and long enough that
# dragging a slider never re-hits the network. The Sync button additionally
# clears these caches outright, so it always means "now".
_TTL = 900

# How old the on-disk CSV cache may be before a snapshot refetches it (hours).
_SNAPSHOT_MAX_AGE_HOURS = 12


def underlying_label(ticker: str) -> str:
    """Friendly label for an underlying ticker (falls back to the raw symbol)."""
    return UNDERLYING_DISPLAY_NAMES.get(ticker, ticker)


def staleness_days(as_of: str | None) -> int | None:
    """Calendar days between an ``as_of`` date string and today, or ``None``.

    Drives the sync badge: anything past a long weekend is older than the last
    close, i.e. cached rather than live.
    """
    try:
        return (dt.date.today() - dt.date.fromisoformat(as_of)).days
    except (TypeError, ValueError):
        return None


@st.cache_data(show_spinner=False, ttl=_TTL)
def cached_option_chain(ticker):
    return fetch_option_chain(ticker, use_cache=True)


@st.cache_data(show_spinner=False, ttl=_TTL)
def cached_option_surface(ticker, n_expiries):
    return fetch_option_surface(ticker, n_expiries=n_expiries, use_cache=True)


@st.cache_data(show_spinner=False, ttl=_TTL)
def cached_historical(ticker):
    return fetch_10y_historical_data(ticker, use_cache=True)


@st.cache_data(show_spinner=False, ttl=_TTL)
def cached_risk_free_curve():
    return fetch_risk_free_curve()


@st.cache_data(show_spinner=False, ttl=_TTL)
def cached_dividend_yield(ticker):
    return fetch_dividend_yield(ticker)


@st.cache_data(show_spinner=False, ttl=_TTL)
def cached_market_snapshot(ticker, max_age_hours=_SNAPSHOT_MAX_AGE_HOURS):
    return fetch_market_snapshot(ticker, max_age_hours=max_age_hours)


def clear_market_caches() -> None:
    """Drop every memoized market read so the next call genuinely refetches."""
    for fn in (
        cached_market_snapshot,
        cached_risk_free_curve,
        cached_dividend_yield,
        cached_historical,
    ):
        fn.clear()
