"""Right-column diagnostic panels for the Streamlit dashboard.

The historical-volatility chart and the Excel export control.
"""

from __future__ import annotations

import html

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from optpricing.data.market_data_fetcher import (
    SUPPORTED_UNDERLYINGS,
    export_indices_to_excel,
)

from _data import cached_historical, underlying_label
from _theme import CALL, PUT, card, style_fig


def historical_and_export_panel() -> None:
    hist_ticker = st.selectbox(
        "Select Ticker for Volatility Analysis:",
        SUPPORTED_UNDERLYINGS,
        format_func=underlying_label,
        key="hist_ticker",
    )
    hist_skeleton = st.empty()
    hist_skeleton.markdown(
        '<div class="skeleton" style="height:280px;"></div>', unsafe_allow_html=True
    )
    try:
        # .copy() so the per-rerun derived columns don't mutate the cached frame.
        hist_df = cached_historical(hist_ticker).copy()
        hist_skeleton.empty()
        if not hist_df.empty:
            hist_df["Returns"] = np.log(hist_df["Close"] / hist_df["Close"].shift(1))
            hist_df["Vol30"] = hist_df["Returns"].rolling(30).std() * np.sqrt(252) * 100
            hist_df["Vol252"] = (
                hist_df["Returns"].rolling(252).std() * np.sqrt(252) * 100
            )

            recent_30 = hist_df["Vol30"].iloc[-1]
            recent_252 = hist_df["Vol252"].iloc[-1]
            avg_10y = hist_df["Vol30"].mean()

            # Vol252 is NaN until a full year of returns exists, so a short
            # history must not reach the card formatted as "nan%".
            def pct(v):
                return f"{v:.2f}%" if not np.isnan(v) else "n/a"

            st.markdown(
                card(
                    "Volatility Statistics",
                    [
                        {"sub": "30D Rolling", "value": pct(recent_30)},
                        {"sub": "252D Rolling", "value": pct(recent_252)},
                        {"sub": "10Y Average", "value": pct(avg_10y)},
                    ],
                    accent=CALL,
                ),
                unsafe_allow_html=True,
            )

            plot_df = hist_df.dropna(subset=["Vol30"]).tail(500)
            fig_hist = go.Figure()
            fig_hist.add_scatter(
                x=plot_df.index,
                y=plot_df["Vol30"],
                mode="lines",
                line=dict(color=PUT, width=1.5),
                name="30D Vol",
            )
            fig_hist.add_scatter(
                x=plot_df.index,
                y=plot_df["Vol252"],
                mode="lines",
                line=dict(color=CALL, width=1.5, dash="dash"),
                name="252D Vol",
            )
            fig_hist.update_layout(
                title="Historical Volatility (Last 500 Trading Days)",
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(style_fig(fig_hist, height=220))
    except Exception as e:
        hist_skeleton.empty()
        st.warning(f"Could not load historical volatility: {e}")

    st.markdown(
        '<div class="section-label" style="margin-top:1.2rem;">8. Historical Data Export</div>',
        unsafe_allow_html=True,
    )
    if st.button("Export all indices to Excel (Max History)"):
        with st.spinner("Fetching maximum available history for all indices..."):
            try:
                xlsx_path = export_indices_to_excel(period="max", use_cache=True)
                st.session_state["export_xlsx_bytes"] = xlsx_path.read_bytes()
                st.session_state["export_error"] = None
            except Exception as e:
                st.session_state["export_error"] = str(e)
                st.session_state["export_xlsx_bytes"] = None

    if st.session_state.get("export_error"):
        # Escaped: this block renders with unsafe_allow_html, and the message is
        # whatever the provider/openpyxl raised -- angle brackets in it would
        # otherwise break out of the div.
        _err = html.escape(str(st.session_state["export_error"]))
        st.markdown(
            '<div class="status status-bad"><span class="status-dot"></span>'
            f"Export failed: {_err}</div>",
            unsafe_allow_html=True,
        )

    if st.session_state.get("export_xlsx_bytes"):
        index_codes = ", ".join(underlying_label(t) for t in SUPPORTED_UNDERLYINGS)
        st.markdown(
            '<div class="status status-good"><span class="status-dot"></span>'
            f"Workbook ready — {index_codes}, max available history.</div>",
            unsafe_allow_html=True,
        )
        st.download_button(
            label="Download indices_historical_data.xlsx",
            data=st.session_state["export_xlsx_bytes"],
            file_name="indices_historical_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_indices_xlsx",
        )
