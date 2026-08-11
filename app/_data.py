"""Streamlit-cached market-data wrappers, plus small chain/display helpers.

The library stays pure (fetch + on-disk CSV cache); Streamlit's in-memory cache
lives here so dragging sliders never re-reads disk or re-hits the network within
a session.
"""

from __future__ import annotations

import numpy as np
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


def underlying_label(ticker: str) -> str:
    """Friendly label for an underlying ticker (falls back to the raw symbol)."""
    return UNDERLYING_DISPLAY_NAMES.get(ticker, ticker)


def mid_prices(df):
    """Mid quote per row, falling back to lastPrice when the spread is empty."""
    mid = (df["bid"] + df["ask"]) / 2.0
    return np.where(mid > 0, mid, df["lastPrice"])


@st.cache_data(show_spinner=False)
def cached_option_chain(ticker):
    return fetch_option_chain(ticker, use_cache=True)


@st.cache_data(show_spinner=False)
def cached_option_surface(ticker, n_expiries):
    return fetch_option_surface(ticker, n_expiries=n_expiries, use_cache=True)


@st.cache_data(show_spinner=False)
def cached_historical(ticker):
    return fetch_10y_historical_data(ticker, use_cache=True)


@st.cache_data(show_spinner=False)
def cached_risk_free_curve():
    return fetch_risk_free_curve()


@st.cache_data(show_spinner=False)
def cached_dividend_yield(ticker):
    return fetch_dividend_yield(ticker)


@st.cache_data(show_spinner=False)
def cached_market_snapshot(ticker):
    return fetch_market_snapshot(ticker)
