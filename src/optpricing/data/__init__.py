"""Market-data helpers (optional; requires the ``data`` extra).

Unlike :mod:`optpricing.pricing` and :mod:`optpricing.analytics`, this
subpackage deliberately does **not** re-export its members at the package level.
Import them explicitly, e.g.::

    from optpricing.data.market_data_fetcher import fetch_option_chain

This keeps ``import optpricing`` from eagerly pulling in the optional ``data``
dependency (``yfinance``), so the core pricing library has no heavy imports.
"""
