"""Pure quote-frame helpers shared by every consumer of an option chain.

Deliberately dependency-free (numpy only, no ``yfinance``) so the analytics layer
and the Streamlit app can both import it without pulling in the optional ``data``
extra.
"""

from __future__ import annotations

import numpy as np


def mid_prices(df):
    """Mid quote per row: ``(bid + ask) / 2``, falling back to ``lastPrice``.

    A chain row with no live two-sided market quotes ``bid = ask = 0``, which
    would price as a free option; the last traded price is the only usable
    figure there. Returns a NumPy array aligned with ``df``'s rows.
    """
    mid = (df["bid"] + df["ask"]) / 2.0
    return np.where(mid > 0, mid, df["lastPrice"])
