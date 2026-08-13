"""Interactive Streamlit dashboard for the options pricing toolkit.

Run with::

    pip install -e ".[app,data]"
    streamlit run app/streamlit_dashboard.py

This entry script owns the page layout, inputs and market-sync callbacks. The
design system lives in ``_theme.py``, cached data wrappers in ``_data.py`` and
the visualization panels in ``_tabs.py``.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from optpricing import (
    BinomialTreeModel,
    BlackScholesModel,
    Greeks,
    ImpliedVolatilityCalculator,
    MonteCarloOptionPricer,
    PutCallParity,
)
from optpricing.data.market_data_fetcher import (
    SUPPORTED_UNDERLYINGS,
    risk_free_rate_for,
)

from _data import (
    cached_market_snapshot,
    cached_risk_free_curve,
    clear_market_caches,
    staleness_days,
    underlying_label,
)
from _panels import historical_and_export_panel
from _tabs import (
    mc_paths_tab,
    sensitivity_surface_tab,
    smile_tab,
    vol_surface_tab,
)
from _theme import (
    ACCENT,
    CALL,
    CRIT,
    GOOD,
    INK_2,
    PUT,
    card,
    inject_css,
    kpi,
    play_entrance_count_up,
    reset_stagger,
)

st.set_page_config(page_title="Option Pricing Terminal", layout="wide")

# Imported modules don't re-execute on a Streamlit rerun, so the entrance-stagger
# counters in _theme must be reset once per run here.
reset_stagger()
inject_css()

# --------------------------------------------------------------------------- #
# Header                                                                       #
# --------------------------------------------------------------------------- #
st.html("""
    <div class="hdr-eyebrow">Option Pricing Terminal</div>
    <h1 class="hdr-title">European Options Pricing &amp; Analysis Terminal</h1>
    <div class="hdr-sub">Black-Scholes, Monte Carlo and Binomial Tree valuation
    with a full Greeks suite, implied-volatility solvers and put-call parity
    diagnostics.</div>
    <div class="hdr-rule"></div>
    """)

# KPI strip placeholder — rendered here (top of page) but filled at the end of
# the script once prices, Greeks and the parity check have all been computed.
kpi_slot = st.container()

# --------------------------------------------------------------------------- #
# Input state & live-market sync                                              #
# --------------------------------------------------------------------------- #
_INPUT_DEFAULTS = {
    "S_in": 100.0,
    "K_in": 100.0,
    "T_in": 1.0,
    "r_in": 5.0,
    "sigma_in": 20.0,
    "q_in": 0.0,
}
for _key, _val in _INPUT_DEFAULTS.items():
    st.session_state.setdefault(_key, _val)


def _mc_label(n: int) -> str:
    """Compact simulation count, e.g. ``500k`` or ``1M``."""
    return f"{n // 1000:,}k" if n < 1_000_000 else f"{n // 1_000_000}M"


def _apply_curve_rate() -> None:
    """Set the risk-free slider from the synced yield curve at the current T."""
    curve = st.session_state.get("mkt_curve")
    if curve:
        rate_pct = risk_free_rate_for(st.session_state["T_in"], curve) * 100.0
        st.session_state["r_in"] = float(np.clip(rate_pct, 0.0, 15.0))


def _request_sync() -> None:
    """Button callback: drop the memoized reads and flag a sync for this rerun.

    Two things have to happen here rather than in ``_sync_from_market``. The
    caches must be cleared *before* the refetch or the button would replay the
    same memoized snapshot forever; and the fetch itself has to run in the main
    script body, because ``on_click`` callbacks execute before the rerun paints,
    so a ``st.spinner`` in here would never be visible during a slow (10 years
    of history) refetch.
    """
    clear_market_caches()
    st.session_state["_force_sync"] = True


def _sync_from_market() -> None:
    """Fill spot / rate / dividend / vol from a live market snapshot."""
    ticker = st.session_state.get("underlying_ticker", "^SPX")
    snap = cached_market_snapshot(ticker)
    st.session_state["mkt_curve"] = cached_risk_free_curve()
    st.session_state["mkt_asof"] = snap.get("as_of")
    st.session_state["mkt_ticker"] = ticker
    if snap.get("spot") is not None:
        st.session_state["S_in"] = round(float(snap["spot"]), 2)
        # Strike follows spot to ATM: a stale strike (e.g. 100 vs SPX ~7500)
        # makes the put worthless and every put Greek ~0.
        st.session_state["K_in"] = round(float(snap["spot"]), 2)
    if snap.get("hist_vol") is not None:
        st.session_state["sigma_in"] = float(
            np.clip(snap["hist_vol"] * 100.0, 5.0, 100.0)
        )
    st.session_state["q_in"] = float(np.clip((snap.get("q") or 0.0) * 100.0, 0.0, 10.0))
    _apply_curve_rate()


def _reset_iv_price() -> None:
    """Snap the IV solver's observed price back to the current model call price.

    The Observed Option Price box is seeded once and then owned by the user, so
    it does not auto-track the model inputs. This callback provides an explicit
    re-sync, mirroring the ``_sync_from_market`` pattern.
    """
    st.session_state["iv_market_price"] = float(
        round(st.session_state.get("_bs_call", 0.0), 4)
    )


# --------------------------------------------------------------------------- #
# 3-Column Layout Grid                                                        #
# --------------------------------------------------------------------------- #
col_left, col_center, col_right = st.columns([1.1, 2.0, 0.9])

# --------------------------------------------------------------------------- #
# Left Column: Inputs & Solver                                                #
# --------------------------------------------------------------------------- #
with col_left:
    st.markdown('<div class="section-label">Market Sync</div>', unsafe_allow_html=True)
    st.selectbox(
        "Live underlying",
        SUPPORTED_UNDERLYINGS,
        format_func=underlying_label,
        key="underlying_ticker",
    )
    st.button("Sync from market", on_click=_request_sync, width="stretch")
    if st.session_state.pop("_force_sync", False):
        with st.spinner("Fetching current market data…"):
            _sync_from_market()
    if st.session_state.get("mkt_asof"):
        _ticker = st.session_state.get("mkt_ticker", "")
        _age = staleness_days(st.session_state["mkt_asof"])
        # Anything past a long weekend is older than the last close, i.e. the
        # provider was unreachable and this is cached data. Say so, loudly --
        # silently presenting a months-old spot as "synced" is the whole bug.
        if _age is not None and _age > 2:
            st.markdown(
                f'<div class="sync-caption stale">synced {_ticker} · as of '
                f'{st.session_state["mkt_asof"]} ({_age} days ago) · cached</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="sync-caption">synced {_ticker} '
                f'· as of {st.session_state["mkt_asof"]} · source: yfinance</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-label" style="margin-top:1.0rem;">1. Contract Details</div>',
        unsafe_allow_html=True,
    )
    S = st.number_input("Spot price (S)", min_value=0.01, step=1.0, key="S_in")
    K = st.number_input("Strike price (K)", min_value=0.01, step=1.0, key="K_in")
    T = st.slider(
        "Time to maturity (years)",
        0.01,
        3.0,
        step=0.01,
        key="T_in",
        on_change=_apply_curve_rate,
    )

    st.markdown(
        '<div class="section-label" style="margin-top:1.0rem;">2. Market Details</div>',
        unsafe_allow_html=True,
    )
    r = st.slider("Risk-free rate (%)", 0.0, 15.0, step=0.1, key="r_in") / 100
    sigma = st.slider("Volatility (%)", 5.0, 100.0, step=1.0, key="sigma_in") / 100
    q = st.slider("Dividend yield (%)", 0.0, 10.0, step=0.1, key="q_in") / 100

    bs = BlackScholesModel(S, K, T, r, sigma, q)
    bs_call, bs_put = bs.call_price(), bs.put_price()
    # Expose the current model call price to the IV-solver re-sync callback.
    st.session_state["_bs_call"] = bs_call

    st.markdown(
        '<div class="section-label" style="margin-top:1.2rem;">3. Solvers</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Implied Volatility Solver", expanded=True):
        # Fully reactive (no button gate that hides the result): the IV recomputes
        # on every change, so flipping call<->put at the same observed price
        # immediately shows the different implied vol each side gives.
        #
        # The box is seeded with the model call price once, then becomes
        # user-owned input. Seeding via setdefault rather than ``value=`` keeps
        # Streamlit from warning about a widget default colliding with the
        # re-sync callback's session-state write.
        st.session_state.setdefault("iv_market_price", float(round(bs_call, 4)))
        market_price = st.number_input(
            "Observed Option Price",
            min_value=0.0,
            step=0.1,
            key="iv_market_price",
        )
        st.button(
            "Use model price",
            on_click=_reset_iv_price,
            width="stretch",
            help="Reset the observed price to the current model call price.",
        )
        iv_type = st.radio(
            "Option Type", ["call", "put"], horizontal=True, key="iv_solver_type"
        )
        solved_iv = ImpliedVolatilityCalculator.calculate(
            market_price, S, K, T, r, iv_type, q
        )
        if solved_iv is None:
            # Two distinct failures used to share one (often untrue) message: a
            # price outside the no-arbitrage bounds genuinely has no implied vol,
            # while a price inside them that the solver misses is a range issue.
            in_bounds = ImpliedVolatilityCalculator.within_no_arbitrage_bounds(
                market_price, S, K, T, r, iv_type, q
            )
            reason = (
                "Price implies a volatility beyond the solver's range."
                if in_bounds
                else "No arbitrage-free solution for this price."
            )
            st.markdown(
                '<div class="status status-bad"><span class="status-dot"></span>'
                f"{reason}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                card(
                    "Implied Volatility Solver Result",
                    [
                        {
                            "sub": f"Solved IV ({iv_type})",
                            "value": f"{solved_iv * 100:.2f}%",
                        },
                        {"sub": "Model Input σ", "value": f"{sigma * 100:.2f}%"},
                    ],
                    accent=GOOD,
                ),
                unsafe_allow_html=True,
            )

# --------------------------------------------------------------------------- #
# Center Column: Prices, Greeks, & Plots                                      #
# --------------------------------------------------------------------------- #
with col_center:
    st.markdown(
        '<div class="section-label">4. Valuation Engines</div>', unsafe_allow_html=True
    )

    # Monte Carlo path count — more paths shrink the standard error (~1/sqrt(N)).
    mc_sims = st.select_slider(
        "Monte Carlo simulations",
        options=[
            100_000,
            500_000,
            1_000_000,
            2_000_000,
            5_000_000,
            10_000_000,
            25_000_000,
        ],
        value=1_000_000,
        format_func=_mc_label,
        key="mc_sims",
    )
    mc_label = _mc_label(mc_sims)

    bt = BinomialTreeModel(S, K, T, r, sigma, q, n_steps=500)
    mc = MonteCarloOptionPricer(S, K, T, r, sigma, q, n_simulations=mc_sims, seed=1)
    # One simulation for both sides: pricing them separately doubled the runtime
    # and peak memory on every slider move (~1.6 GB / ~12 s at 25M paths) and
    # gave the two legs different samples, so they didn't respect parity.
    (mc_call, mc_call_se), (mc_put, mc_put_se) = mc.call_and_put_price(
        return_std_error=True
    )
    bt_call, bt_put = bt.call_price(), bt.put_price()

    p_col1, p_col2, p_col3 = st.columns(3)
    p_col1.markdown(
        card(
            "Black-Scholes",
            [
                {"sub": "Call", "value": f"${bs_call:.4f}"},
                {"sub": "Put", "value": f"${bs_put:.4f}"},
            ],
            accent=CALL,
        ),
        unsafe_allow_html=True,
    )
    p_col2.markdown(
        card(
            f"Monte Carlo ({mc_label})",
            [
                {
                    "sub": "Call",
                    "value": f"${mc_call:.4f}",
                    "se": f"± {mc_call_se:.4f}",
                },
                {"sub": "Put", "value": f"${mc_put:.4f}", "se": f"± {mc_put_se:.4f}"},
            ],
            accent=PUT,
        ),
        unsafe_allow_html=True,
    )
    p_col3.markdown(
        card(
            "Binomial Tree (500)",
            [
                {"sub": "Call", "value": f"${bt_call:.4f}"},
                {"sub": "Put", "value": f"${bt_put:.4f}"},
            ],
            accent=ACCENT,
        ),
        unsafe_allow_html=True,
    )

    g = Greeks(S, K, T, r, sigma, q)
    greeks_type = st.radio(
        "Option Type", ["call", "put"], horizontal=True, key="greeks_type"
    )
    gd = g.get_all_greeks(greeks_type)
    type_label = greeks_type.capitalize()
    st.markdown(
        f'<div class="section-label" style="margin-top:1.2rem;">'
        f"5. First-Order Greeks Cluster ({type_label})</div>",
        unsafe_allow_html=True,
    )
    g_col1, g_col2, g_col3, g_col4, g_col5 = st.columns(5)
    g_col1.markdown(
        card("Delta", [{"sub": type_label, "value": f"{gd['delta']:.4f}"}]),
        unsafe_allow_html=True,
    )
    g_col2.markdown(
        card("Gamma", [{"sub": "Value", "value": f"{gd['gamma']:.4f}"}]),
        unsafe_allow_html=True,
    )
    g_col3.markdown(
        card("Vega", [{"sub": "Value", "value": f"{gd['vega']:.4f}"}]),
        unsafe_allow_html=True,
    )
    g_col4.markdown(
        card("Theta", [{"sub": type_label, "value": f"{gd['theta']:.4f}"}]),
        unsafe_allow_html=True,
    )
    g_col5.markdown(
        card("Rho", [{"sub": type_label, "value": f"{gd['rho']:.4f}"}]),
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label" style="margin-top:1.2rem;">6. Visualizations</div>',
        unsafe_allow_html=True,
    )
    tab_surface, tab_vol_surface, tab_mc_paths, tab_smile = st.tabs(
        [
            "Sensitivity Surface (3D)",
            "Volatility Surface (3D)",
            "Monte Carlo Price Paths",
            "Volatility Smile / Skew",
        ]
    )
    with tab_surface:
        sensitivity_surface_tab(S, K, T, r, sigma, q)
    with tab_vol_surface:
        vol_surface_tab()
    with tab_mc_paths:
        mc_paths_tab(S, K, T, r, sigma, q, mc_sims, mc_label)
    with tab_smile:
        smile_tab()

# --------------------------------------------------------------------------- #
# Right Column: Diagnostics & Historical Volatility                            #
# --------------------------------------------------------------------------- #
with col_right:
    st.markdown(
        '<div class="section-label">7. Put-Call Parity Diagnostic</div>',
        unsafe_allow_html=True,
    )
    res = PutCallParity.verify(bs_call, bs_put, S, K, r, T, q)
    st.markdown(
        card(
            "Parity Check Details",
            [
                {"sub": "C - P", "value": f"${res['lhs']:.4f}"},
                {"sub": "Forward Side", "value": f"${res['rhs']:.4f}"},
            ],
            accent=GOOD if res["parity_holds"] else CRIT,
        ),
        unsafe_allow_html=True,
    )
    if res["parity_holds"]:
        st.markdown(
            f'<div class="status status-good"><span class="status-dot"></span>Parity Holds (diff: {res["difference"]:.6f})</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="status status-bad"><span class="status-dot"></span>Parity Violated (diff: {res["difference"]:.4f})</div>',
            unsafe_allow_html=True,
        )

    historical_and_export_panel()

# --------------------------------------------------------------------------- #
# Fill the KPI strip now that prices, Greeks and the parity check are known.   #
# --------------------------------------------------------------------------- #
with kpi_slot:
    parity_ok = res["parity_holds"]
    st.html(
        '<div class="kpi-strip">'
        + kpi("Spot", f"${S:,.2f}", accent=INK_2)
        + kpi("BS Call", f"${bs_call:.2f}", accent=CALL)
        + kpi("BS Put", f"${bs_put:.2f}", accent=PUT)
        + kpi(f"Delta ({type_label})", f"{gd['delta']:.3f}", accent=ACCENT)
        + kpi("Volatility σ", f"{sigma * 100:.1f}%", accent=ACCENT)
        + kpi(
            "Put-Call Parity",
            "HOLDS" if parity_ok else "VIOLATED",
            accent=GOOD if parity_ok else CRIT,
            tone="pos" if parity_ok else "neg",
        )
        + "</div>"
    )

# --------------------------------------------------------------------------- #
# One-shot entrance flourish: count the metric cards up on first load only,   #
# so dragging sliders afterwards stays instant and doesn't fight the user.    #
# --------------------------------------------------------------------------- #
if not st.session_state.get("_entrance_played"):
    play_entrance_count_up()
    st.session_state["_entrance_played"] = True
