"""Visualization tab bodies for the Streamlit dashboard.

Each function renders one panel directly into the current Streamlit container,
taking the model primitives it needs as explicit arguments.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale

from optpricing import (
    Greeks,
    ImpliedVolatilityCalculator,
    MonteCarloOptionPricer,
    VolatilitySurface,
)
from optpricing.data.market_data_fetcher import (
    SUPPORTED_UNDERLYINGS,
    risk_free_rate_for,
    years_to_expiry,
)

from _data import (
    cached_dividend_yield,
    cached_option_chain,
    cached_option_surface,
    cached_risk_free_curve,
    mid_prices,
    underlying_label,
)
from _theme import (
    BG_PAGE,
    CALL,
    CRIT,
    FONT,
    INK_2,
    INK_MUTED,
    PUT,
    RAINBOW,
    SCENE_GRID,
    mesh_contours,
    style_fig,
)


def sensitivity_surface_tab(S, K, T, r, sigma, q) -> None:
    surf_type = st.radio(
        "Option Type", ["call", "put"], horizontal=True, key="surf_greek_type"
    )
    labels = {
        "delta": "Delta",
        "gamma": "Gamma",
        "vega": "Vega",
        "theta": "Theta",
        "rho": "Rho",
    }
    greek = st.selectbox(
        "Sensitivity Surface target Z-axis:",
        list(labels),
        format_func=labels.get,
        key="surf_greek_select",
    )
    s_axis = np.linspace(S * 0.5, S * 1.5, 30)
    t_axis = np.linspace(0.05, max(T * 2, 0.1), 30)
    s_mesh, t_mesh = np.meshgrid(s_axis, t_axis)
    surface = np.array(
        [
            [
                Greeks(s, K, t, r, sigma, q).get_all_greeks(surf_type)[greek]
                for s in s_axis
            ]
            for t in t_axis
        ]
    )
    fig_s = go.Figure(
        go.Surface(
            x=s_mesh,
            y=t_mesh,
            z=surface,
            colorscale=RAINBOW,
            contours=mesh_contours(
                s_axis.min(), s_axis.max(), t_axis.min(), t_axis.max()
            ),
            colorbar=dict(
                title=labels[greek],
                outlinewidth=0,
                tickfont=dict(size=11, color=INK_2),
            ),
        )
    )
    fig_s.update_layout(
        title=f"{labels[greek]} across spot and maturity",
        scene=dict(
            xaxis=dict(
                title="Spot",
                backgroundcolor=BG_PAGE,
                gridcolor=SCENE_GRID,
                showgrid=True,
            ),
            yaxis=dict(
                title="Maturity",
                backgroundcolor=BG_PAGE,
                gridcolor=SCENE_GRID,
                showgrid=True,
            ),
            zaxis=dict(
                title=labels[greek],
                backgroundcolor=BG_PAGE,
                gridcolor=SCENE_GRID,
                showgrid=True,
            ),
        ),
    )
    st.plotly_chart(style_fig(fig_s, height=450))


def vol_surface_tab() -> None:
    vs_ticker = st.selectbox(
        "Underlying for IV surface:",
        SUPPORTED_UNDERLYINGS,
        format_func=underlying_label,
        key="vol_surf_ticker",
    )
    vs_n = st.slider("Expiries to include", 2, 12, 6, 1, key="vol_surf_n")
    vs_skeleton = st.empty()
    vs_skeleton.markdown(
        '<div class="skeleton" style="height:480px;"></div>', unsafe_allow_html=True
    )
    try:
        vs_records = cached_option_surface(vs_ticker, vs_n)
        vs_curve = cached_risk_free_curve()
        vs_q = cached_dividend_yield(vs_ticker)
        surface = VolatilitySurface.from_chains(
            vs_records,
            r=lambda t_exp: risk_free_rate_for(t_exp, vs_curve),
            q=vs_q,
        )
        vs_skeleton.empty()
        fig_vs = go.Figure(
            go.Surface(
                x=surface.moneyness_axis,
                y=surface.maturity_axis,
                z=surface.iv_mesh,
                colorscale=RAINBOW,
                opacity=1.0,
                contours=mesh_contours(
                    surface.moneyness_axis.min(),
                    surface.moneyness_axis.max(),
                    surface.maturity_axis.min(),
                    surface.maturity_axis.max(),
                ),
                colorbar=dict(
                    title="IV %",
                    outlinewidth=0,
                    tickfont=dict(size=11, color=INK_2),
                ),
            )
        )
        # Clip the raw-point overlay to the same 2-98pct band as the mesh so a
        # couple of short-dated outliers don't spike above the smooth surface.
        vs_lo, vs_hi = np.percentile(surface.raw_iv, [2, 98])
        fig_vs.add_scatter3d(
            x=surface.raw_moneyness,
            y=surface.raw_maturity,
            z=np.clip(surface.raw_iv, vs_lo, vs_hi),
            mode="markers",
            marker=dict(size=1.6, color="rgba(15,23,42,0.35)"),
            name="Market IV",
        )
        fig_vs.update_layout(
            title=f"Market Implied Volatility Surface ({vs_ticker})",
            scene=dict(
                xaxis=dict(
                    title="Moneyness (K/S)",
                    backgroundcolor=BG_PAGE,
                    gridcolor=SCENE_GRID,
                    showgrid=True,
                ),
                yaxis=dict(
                    title="Maturity (yrs)",
                    backgroundcolor=BG_PAGE,
                    gridcolor=SCENE_GRID,
                    showgrid=True,
                ),
                zaxis=dict(
                    title="Implied Vol (%)",
                    backgroundcolor=BG_PAGE,
                    gridcolor=SCENE_GRID,
                    showgrid=True,
                ),
            ),
        )
        st.plotly_chart(style_fig(fig_vs, height=480))
        st.caption(
            f"{len(vs_records)} expiries · spot {surface.spot:.2f} · "
            "OTM-side quotes · griddata-smoothed"
        )
    except Exception as e:
        vs_skeleton.empty()
        st.warning(f"Could not build volatility surface: {e}")


def mc_paths_tab(S, K, T, r, sigma, q, mc_sims, mc_label) -> None:
    # Number of sample paths drawn scales with the simulations slider, so the
    # chart visibly densifies as you raise the count (drawing millions of
    # lines is impossible, so display is capped while prices use the full N).
    _MC_PATH_COUNTS = {
        100_000: 60,
        500_000: 120,
        1_000_000: 200,
        2_000_000: 300,
        5_000_000: 450,
        10_000_000: 600,
        25_000_000: 800,
    }
    n_paths = _MC_PATH_COUNTS.get(mc_sims, 200)
    mc_paths_gen = MonteCarloOptionPricer(
        S, K, T, r, sigma, q, n_simulations=n_paths, seed=42
    )
    paths = mc_paths_gen.generate_paths(n_paths)
    steps = len(paths) - 1
    time_grid = np.linspace(0, T, steps + 1)

    fig_mc = go.Figure()
    # Colour each path by its terminal price on the rainbow scale, so the
    # fan of paths reads like a heatmap of where the underlying lands.
    terminal = paths[-1, :]
    t_min, t_max = float(terminal.min()), float(terminal.max())
    norm = (terminal - t_min) / (t_max - t_min + 1e-12)
    path_colors = sample_colorscale(RAINBOW, norm)
    for i in range(n_paths):
        rgba = path_colors[i].replace("rgb(", "rgba(").replace(")", ", 0.45)")
        fig_mc.add_scatter(
            x=time_grid,
            y=paths[:, i],
            mode="lines",
            line=dict(width=1.0, color=rgba),
            showlegend=False,
        )
    # Invisible marker trace purely to render the rainbow colourbar legend.
    fig_mc.add_scatter(
        x=[time_grid[0]],
        y=[paths[0, 0]],
        mode="markers",
        marker=dict(
            size=0.1,
            color=[t_min],
            colorscale=RAINBOW,
            cmin=t_min,
            cmax=t_max,
            colorbar=dict(
                title="Terminal $",
                outlinewidth=0,
                tickfont=dict(size=11, color=INK_2),
            ),
        ),
        hoverinfo="skip",
        showlegend=False,
    )

    fig_mc.add_hline(
        y=K,
        line_dash="dash",
        line_color=CRIT,
        line_width=1.5,
        annotation_text=f"Strike K={K}",
        annotation_position="top left",
        annotation_font=dict(color=INK_2, family=FONT),
    )

    fig_mc.add_hline(
        y=S,
        line_dash="dot",
        line_color=CALL,
        line_width=1.0,
        annotation_text=f"Spot S={S}",
        annotation_position="bottom left",
        annotation_font=dict(color=INK_2, family=FONT),
    )

    fig_mc.update_layout(
        title=f"Monte Carlo Asset Price Paths (GBM) — {n_paths} of {mc_label} sampled",
        xaxis_title="Time to Maturity (years)",
        yaxis_title="Underlying Asset Price ($)",
        showlegend=False,
    )
    fig_mc = style_fig(fig_mc, height=450)
    # Dotted square grid on both axes.
    fig_mc.update_xaxes(griddash="dot", gridcolor="rgba(15,23,42,0.18)")
    fig_mc.update_yaxes(griddash="dot", gridcolor="rgba(15,23,42,0.18)")
    st.plotly_chart(fig_mc)


def smile_tab() -> None:
    smile_ticker = st.selectbox(
        "Select ticker for Volatility Smile:",
        SUPPORTED_UNDERLYINGS,
        format_func=underlying_label,
        key="smile_ticker",
    )
    smile_skeleton = st.empty()
    smile_skeleton.markdown(
        '<div class="skeleton" style="height:450px;"></div>', unsafe_allow_html=True
    )
    try:
        calls_df, puts_df, smile_spot, smile_expiry = cached_option_chain(smile_ticker)
        smile_skeleton.empty()

        # Solve IV at the chain's *own* time-to-expiry and a market rate/yield
        # (not the model sliders), which is what makes the smile correct. Shared
        # with the surface path so an expired cached chain is rejected here too
        # rather than being silently treated as a 1-day option.
        T_smile = years_to_expiry(smile_expiry)
        if T_smile is None:
            raise ValueError(
                f"cached chain for expiry {smile_expiry} has already expired — "
                "refresh the data cache with a live fetch"
            )
        r_smile = risk_free_rate_for(T_smile, cached_risk_free_curve())
        q_smile = cached_dividend_yield(smile_ticker)

        calls_df = calls_df.copy()
        puts_df = puts_df.copy()
        calls_df["mid"] = mid_prices(calls_df)
        puts_df["mid"] = mid_prices(puts_df)

        common = set(calls_df["strike"]).intersection(set(puts_df["strike"]))
        if common:
            calls_df = calls_df[calls_df["strike"].isin(common)]
            puts_df = puts_df[puts_df["strike"].isin(common)]
            strike_dist = (calls_df["strike"] - smile_spot).abs()
            top_s = calls_df.loc[strike_dist.nsmallest(7).index, "strike"].tolist()

            c_filtered = calls_df[calls_df["strike"].isin(top_s)].sort_values("strike")
            p_filtered = puts_df[puts_df["strike"].isin(top_s)].sort_values("strike")

            smile_strikes = c_filtered["strike"].to_numpy()
            solved_call_ivs = (
                ImpliedVolatilityCalculator.calculate_vectorized(
                    c_filtered["mid"].to_numpy(),
                    smile_spot,
                    smile_strikes,
                    T_smile,
                    r_smile,
                    "call",
                    q_smile,
                )
                * 100
            )
            solved_put_ivs = (
                ImpliedVolatilityCalculator.calculate_vectorized(
                    p_filtered["mid"].to_numpy(),
                    smile_spot,
                    p_filtered["strike"].to_numpy(),
                    T_smile,
                    r_smile,
                    "put",
                    q_smile,
                )
                * 100
            )

            fig_smile = go.Figure()
            fig_smile.add_scatter(
                x=smile_strikes,
                y=solved_call_ivs,
                mode="lines+markers",
                name="Call Implied Vol",
                line=dict(color=CALL),
            )
            fig_smile.add_scatter(
                x=p_filtered["strike"].to_numpy(),
                y=solved_put_ivs,
                mode="lines+markers",
                name="Put Implied Vol",
                line=dict(color=PUT),
            )
            fig_smile.add_vline(
                x=smile_spot,
                line_dash="dot",
                line_color=INK_MUTED,
                annotation_text=f"Spot: {smile_spot:.2f}",
            )
            fig_smile.update_layout(
                title=f"Solved Implied Volatility Smile/Skew ({smile_ticker})",
                xaxis_title="Strike Price ($)",
                yaxis_title="Implied Volatility (%)",
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(style_fig(fig_smile, height=450))
            st.caption(
                f"expiry {smile_expiry} · T={T_smile:.3f}y · r={r_smile * 100:.2f}% "
                f"· q={q_smile * 100:.2f}% (market, not model sliders)"
            )
        else:
            st.info("No common strikes found for volatility smile.")
    except Exception as e:
        smile_skeleton.empty()
        st.warning(f"Could not load volatility smile: {e}")
