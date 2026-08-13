"""Smoke tests for the dashboard modules.

Only the pure helpers are exercised -- the panels themselves need a live
Streamlit script run to render. That is still enough to catch the failures that
actually happen here: an import that no longer resolves, a renamed theme
constant, or a helper whose contract drifted from its callers.

Skipped wholesale when the ``app`` extra is not installed (the library supports
Python 3.9, streamlit>=1.58 does not).
"""

import datetime as dt

import pandas as pd
import pytest

pytest.importorskip("streamlit", reason="dashboard needs the 'app' extra")
pytest.importorskip("plotly", reason="dashboard needs the 'app' extra")

import _data  # noqa: E402
import _theme  # noqa: E402


def test_app_modules_import():
    """The flat imports inside app/ resolve (this is what conftest.py buys)."""
    import _panels
    import _tabs

    assert callable(_panels.historical_and_export_panel)
    assert callable(_tabs.vol_surface_tab)


def test_underlying_label_falls_back_to_symbol():
    assert _data.underlying_label("^SPX") == "S&P 500 (SPX)"
    assert _data.underlying_label("NOT_A_TICKER") == "NOT_A_TICKER"


def test_mid_prices_is_the_shared_library_helper():
    from optpricing.data.quotes import mid_prices

    assert _data.mid_prices is mid_prices
    df = pd.DataFrame({"bid": [4.0, 0.0], "ask": [6.0, 0.0], "lastPrice": [5.5, 2.25]})
    assert list(_data.mid_prices(df)) == [5.0, 2.25]


def test_every_cached_wrapper_has_a_ttl():
    """No ttl meant the first read of a session was pinned for the whole session,
    which is why pressing "Sync from market" a second time changed nothing."""
    wrappers = [
        _data.cached_option_chain,
        _data.cached_option_surface,
        _data.cached_historical,
        _data.cached_risk_free_curve,
        _data.cached_dividend_yield,
        _data.cached_market_snapshot,
    ]
    for fn in wrappers:
        assert hasattr(fn, "clear"), f"{fn.__name__} is not a cached wrapper"
    assert _data._TTL > 0


def test_clear_market_caches_runs():
    # The Sync button's escape hatch from the ttl; must not raise when nothing
    # has been cached yet.
    _data.clear_market_caches()


def test_theme_helpers_render_html():
    card = _theme.card("Title", [{"sub": "Call", "value": "$1.00"}])
    assert "Title" in card and "$1.00" in card
    assert "<div" in card

    tile = _theme.kpi("Spot", "$100.00", accent=_theme.ACCENT)
    assert "Spot" in tile and "$100.00" in tile

    contours = _theme.mesh_contours(0.0, 1.0, 0.0, 2.0, n=10)
    assert contours["x"]["size"] == pytest.approx(0.1)
    assert contours["y"]["size"] == pytest.approx(0.2)


def test_mesh_contours_survives_a_degenerate_axis():
    # A single-point axis would otherwise divide by zero.
    contours = _theme.mesh_contours(1.0, 1.0, 0.0, 1.0)
    assert contours["x"]["size"] == 1


def test_staleness_days_drives_the_freshness_badge():
    """The badge must be able to flag data older than the last close."""
    today = dt.date.today()
    assert _data.staleness_days(str(today)) == 0
    assert _data.staleness_days(str(today - dt.timedelta(days=30))) == 30
    # A missing or malformed as_of degrades to "unknown", never to a crash.
    assert _data.staleness_days(None) is None
    assert _data.staleness_days("not-a-date") is None
