"""Interactive Streamlit dashboard for the options pricing toolkit.

Run with::

    pip install -e ".[app,data]"
    streamlit run app/streamlit_dashboard.py
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from optpricing import (
    BinomialTreeModel,
    BlackScholesModel,
    Greeks,
    ImpliedVolatilityCalculator,
    MonteCarloOptionPricer,
    PutCallParity,
)
from optpricing.data.market_data_fetcher import (
    fetch_10y_historical_data,
    fetch_risk_free_rate,
    fetch_dividend_yield,
    fetch_option_chain,
    export_indices_to_excel,
)

# --------------------------------------------------------------------------- #
# Design system                                                               #
# --------------------------------------------------------------------------- #
BG_PAGE = "#f4f6fb"
BG_SURFACE = "rgba(255, 255, 255, 0.72)"
BG_PANEL = "#eef1f7"
BORDER = "rgba(15, 23, 42, 0.10)"
INK = "#101828"
INK_2 = "#3f4657"
INK_MUTED = "#5b6472"
GRID = "rgba(15, 23, 42, 0.07)"
CALL = "#2f6fed"   # categorical slot 1 — blue
PUT = "#0d9488"    # categorical slot 2 — teal
ACCENT = "#2563eb"
GOOD = "#047857"
CRIT = "#b91c1c"
FONT = 'Inter, system-ui, -apple-system, sans-serif'
MONO_FONT = '"JetBrains Mono", monospace'

# Single-hue blue ramp for continuous-magnitude surfaces (light -> dark, for a white scene)
BLUE_SEQ = [
    [0.0, "#eaf2ff"],
    [0.25, "#a9c9f5"],
    [0.5, "#5b93e0"],
    [0.75, "#2a63c4"],
    [1.0, "#123a7a"],
]

st.set_page_config(page_title="Options Pricing Toolkit", layout="wide")


def inject_css() -> None:
    st.html(
        f"""
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet" />
        <style>
        /* strip default Streamlit chrome for a clean, professional canvas */
        #MainMenu, header[data-testid="stHeader"], footer,
        [data-testid="stToolbar"], [data-testid="stDecoration"] {{
            display: none !important;
        }}

        /* ---- motion primitives ------------------------------------------- */
        @keyframes card-in {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes fade-scale-in {{
            from {{ opacity: 0; transform: scale(.98); }}
            to   {{ opacity: 1; transform: scale(1); }}
        }}
        @keyframes rule-shimmer {{
            0%, 100% {{ background-position: 0% 50%; }}
            50%      {{ background-position: 100% 50%; }}
        }}
        @keyframes pulse-ring {{
            0%   {{ transform: scale(.6); opacity: .55; }}
            70%  {{ transform: scale(1.9); opacity: 0; }}
            100% {{ transform: scale(1.9); opacity: 0; }}
        }}
        @keyframes shimmer-sweep {{
            0%   {{ background-position: -400px 0; }}
            100% {{ background-position: 400px 0; }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{
                animation-duration: .001ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: .001ms !important;
            }}
        }}

        html, body, [data-testid="stAppViewContainer"], .stApp {{
            background: {BG_PAGE};
            color: {INK};
            font-family: {FONT};
        }}
        [data-testid="stAppViewContainer"] .main .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 100%;
            padding-left: 2rem;
            padding-right: 2rem;
        }}

        /* sidebar / inputs styling */
        [data-testid="stSidebar"] {{
            background: {BG_PANEL};
            border-right: 1px solid {BORDER};
        }}
        [data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}

        /* header block */
        .hdr-eyebrow {{
            font-size: .72rem; font-weight: 600; letter-spacing: .22em;
            text-transform: uppercase; color: {ACCENT}; margin-bottom: .35rem;
        }}
        .hdr-title {{
            font-size: 1.8rem; font-weight: 700; line-height: 1.1;
            color: {INK}; margin: 0;
            font-family: {FONT};
        }}
        .hdr-sub {{
            font-size: .9rem; color: {INK_2}; margin-top: .4rem;
        }}
        .hdr-rule {{
            height: 2px; margin: 1.0rem 0 1.0rem; border-radius: 2px;
            background: linear-gradient(90deg,
                {BORDER} 0%, {CALL} 22%, {ACCENT} 45%, {PUT} 68%, {BORDER} 100%);
            background-size: 220% 100%;
            animation: rule-shimmer 7s ease-in-out infinite;
            opacity: .55;
        }}
        .section-label {{
            font-size: .72rem; font-weight: 600; letter-spacing: .18em;
            text-transform: uppercase; color: {INK_MUTED};
            margin: .2rem 0 .9rem;
        }}

        /* glassmorphic card */
        .card {{
            background: {BG_SURFACE};
            backdrop-filter: blur(12px);
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 1.0rem 1.1rem;
            height: 100%;
            box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 4px 10px rgba(15,23,42,.06);
            animation: card-in .45s cubic-bezier(.16,1,.3,1) both;
            animation-delay: calc(var(--card-i, 0) * 45ms);
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }}
        .card:hover {{
            transform: translateY(-2px);
            border-color: color-mix(in srgb, var(--accent, {ACCENT}) 45%, {BORDER});
            box-shadow: 0 1px 2px rgba(15,23,42,.05),
                        0 10px 24px rgba(15,23,42,.10);
        }}
        .card-accent {{ border-top: 2px solid var(--accent, {ACCENT}); }}
        .card-label {{
            font-size: .72rem; font-weight: 600; letter-spacing: .12em;
            text-transform: uppercase; color: {INK_MUTED}; margin-bottom: .45rem;
        }}
        .card-row {{ display: flex; align-items: baseline; gap: .9rem; }}
        .card-metric {{ flex: 1; }}
        .card-metric + .card-metric {{
            border-left: 1px solid {BORDER}; padding-left: .9rem;
        }}
        .card-sub {{
            font-size: .68rem; font-weight: 600; letter-spacing: .1em;
            text-transform: uppercase; color: {INK_MUTED}; margin-bottom: .2rem;
        }}
        .card-value {{
            font-size: 1.5rem; font-weight: 650; color: {INK};
            font-family: {MONO_FONT};
            font-variant-numeric: tabular-nums; line-height: 1.15;
        }}
        .card-se {{
            font-size: .74rem; color: {INK_2}; margin-top: .2rem;
            font-family: {MONO_FONT};
            font-variant-numeric: tabular-nums;
        }}

        /* tabs styling */
        [data-testid="stTabs"] [role="tablist"] {{
            gap: .4rem; border-bottom: 1px solid {BORDER};
        }}
        [data-testid="stTabs"] [role="tab"] {{
            color: {INK_MUTED}; font-weight: 600; font-size: .88rem;
            padding: .5rem .2rem;
        }}
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
            color: {INK};
        }}
        [data-testid="stTabs"] [role="tab"][aria-selected="true"]::after {{
            background: {ACCENT} !important;
        }}

        /* controls */
        [data-testid="stSidebar"] label p, label p {{
            font-size: .8rem; color: {INK_2}; font-weight: 500;
        }}
        [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
            background: {ACCENT};
        }}
        .stButton > button {{
            background: {ACCENT}; color: #fff; border: none; font-weight: 600;
            border-radius: 6px; padding: .5rem 1.1rem;
            font-family: {FONT};
            width: 100%;
            transition: background-color .15s ease, transform .1s ease, box-shadow .15s ease;
        }}
        .stButton > button:hover {{
            background: #1d4ed8; color: #fff;
            box-shadow: 0 4px 12px rgba(37,99,235,.28);
        }}
        .stButton > button:active {{ transform: scale(.98); }}

        /* status banners */
        .status {{
            border-radius: 8px; padding: .75rem 1.0rem; font-weight: 600;
            font-size: .88rem; display: flex; align-items: center; gap: .6rem;
            margin-top: 10px;
        }}
        .status-good {{
            background: rgba(4,120,87,.10); border: 1px solid {GOOD};
            color: {GOOD};
        }}
        .status-bad {{
            background: rgba(185,28,28,.10); border: 1px solid {CRIT};
            color: {CRIT};
        }}
        .status-dot {{
            position: relative;
            width: 8px; height: 8px; border-radius: 50%; background: currentColor;
        }}
        .status-dot::after {{
            content: ""; position: absolute; inset: -4px; border-radius: 50%;
            border: 1px solid currentColor;
            animation: pulse-ring 1.8s ease-out infinite;
        }}

        /* skeleton placeholder shown while market data loads */
        .skeleton {{
            border-radius: 8px;
            background: linear-gradient(90deg,
                {BG_PANEL} 0%, rgba(255,255,255,.9) 50%, {BG_PANEL} 100%);
            background-size: 800px 100%;
            animation: shimmer-sweep 1.4s linear infinite;
            border: 1px solid {BORDER};
        }}

        /* tab panels + charts fade in whenever they become visible */
        [data-testid="stTabs"] [role="tabpanel"] {{
            animation: fade-scale-in .3s ease both;
        }}
        [data-testid="stPlotlyChart"] {{
            animation: fade-scale-in .5s cubic-bezier(.16,1,.3,1) both;
        }}
        </style>
        """
    )


def style_fig(fig: go.Figure, height: int | None = None) -> go.Figure:
    """Apply the shared professional theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK_2, size=13),
        title=dict(font=dict(family=FONT, color=INK, size=15), x=0, xanchor="left"),
        margin=dict(l=10, r=10, t=48, b=10),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom",
            y=1.02, xanchor="right", x=1, font=dict(size=12),
        ),
        colorway=[CALL, PUT],
        hoverlabel=dict(
            bgcolor="#ffffff", font_color=INK, font_family=FONT, bordercolor=BORDER
        ),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER)
    if height:
        fig.update_layout(height=height)
    return fig


_card_sequence = 0  # reset to 0 every Streamlit rerun (module re-executes top-to-bottom)


def card(label: str, metrics: list[dict], accent: str = ACCENT) -> str:
    """Render a metric card with one or more value columns."""
    global _card_sequence
    stagger_index = _card_sequence
    _card_sequence += 1

    cells = "".join(
        f'<div class="card-metric">'
        f'<div class="card-sub">{m["sub"]}</div>'
        f'<div class="card-value">{m["value"]}</div>'
        + (f'<div class="card-se">{m["se"]}</div>' if m.get("se") else "")
        + "</div>"
        for m in metrics
    )
    return (
        f'<div class="card card-accent" style="--accent:{accent}; --card-i:{stagger_index};">'
        f'<div class="card-label">{label}</div>'
        f'<div class="card-row">{cells}</div></div>'
    )


def play_entrance_count_up() -> None:
    """One-shot count-up animation for `.card-value` numbers on first load.

    Runs via a tiny script component (`window.parent.document` reaches out of
    the component's iframe into the app DOM — the standard escape hatch for
    injecting real JS into Streamlit, since `st.html`/`st.markdown` strip
    <script> tags). Gated to the first script run per session via
    `st.session_state` so it never re-fires and fights the user while they
    are dragging sliders; if the iframe ever can't reach the parent document
    the try/catch just leaves the numbers as-is, so it fails safe.
    """
    components.html(
        r"""
        <script>
        (function() {
            try {
                if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
                    return;
                }
                var doc = window.parent.document;
                var els = doc.querySelectorAll('.card-value');
                els.forEach(function(el) {
                    var raw = el.textContent.trim();
                    var m = raw.match(/^([^0-9\-]*)(-?[0-9]+\.?[0-9]*)(.*)$/);
                    if (!m) return;
                    var prefix = m[1], numStr = m[2], suffix = m[3];
                    var target = parseFloat(numStr);
                    if (isNaN(target)) return;
                    var decimals = (numStr.split('.')[1] || '').length;
                    var duration = 650;
                    var startTime = performance.now();
                    function tick(now) {
                        var p = Math.min((now - startTime) / duration, 1);
                        var eased = 1 - Math.pow(1 - p, 3);
                        el.textContent = prefix + (target * eased).toFixed(decimals) + suffix;
                        if (p < 1) { requestAnimationFrame(tick); }
                        else { el.textContent = raw; }
                    }
                    requestAnimationFrame(tick);
                });
            } catch (e) { /* parent DOM unreachable — leave static values */ }
        })();
        </script>
        """,
        height=0,
        width=0,
    )


inject_css()

# --------------------------------------------------------------------------- #
# Header                                                                       #
# --------------------------------------------------------------------------- #
st.html(
    """
    <div class="hdr-eyebrow">QUANT_TERMINAL v2.0</div>
    <h1 class="hdr-title">European Options Pricing &amp; Analysis Terminal</h1>
    <div class="hdr-sub">Black-Scholes, Monte Carlo and Binomial Tree valuation
    with a full Greeks suite, implied-volatility solvers and put-call parity
    diagnostics.</div>
    <div class="hdr-rule"></div>
    """
)

# --------------------------------------------------------------------------- #
# 3-Column Layout Grid                                                        #
# --------------------------------------------------------------------------- #
col_left, col_center, col_right = st.columns([1.1, 2.0, 0.9])

# --------------------------------------------------------------------------- #
# Left Column: Inputs & Solver                                                #
# --------------------------------------------------------------------------- #
with col_left:
    st.markdown('<div class="section-label">1. Contract Details</div>', unsafe_allow_html=True)
    S = st.number_input("Spot price (S)", value=100.0, min_value=0.01, step=1.0)
    K = st.number_input("Strike price (K)", value=100.0, min_value=0.01, step=1.0)
    T = st.slider("Time to maturity (years)", 0.01, 3.0, 1.0, 0.01)

    st.markdown('<div class="section-label" style="margin-top:1.2rem;">2. Market Details</div>', unsafe_allow_html=True)
    r = st.slider("Risk-free rate (%)", 0.0, 15.0, 5.0, 0.1) / 100
    sigma = st.slider("Volatility (%)", 5.0, 100.0, 20.0, 1.0) / 100
    q = st.slider("Dividend yield (%)", 0.0, 10.0, 0.0, 0.1) / 100

    bs = BlackScholesModel(S, K, T, r, sigma, q)
    bs_call, bs_put = bs.call_price(), bs.put_price()

    st.markdown('<div class="section-label" style="margin-top:1.2rem;">3. Solvers</div>', unsafe_allow_html=True)
    with st.expander("Implied Volatility Solver", expanded=True):
        market_price = st.number_input("Observed Option Price", value=float(bs_call), min_value=0.0, step=0.1)
        iv_type = st.radio("Option Type", ["call", "put"], horizontal=True, key="iv_solver_type")
        if st.button("Solve Implied Volatility"):
            solved_iv = ImpliedVolatilityCalculator.calculate(market_price, S, K, T, r, iv_type, q)
            if solved_iv is None:
                st.markdown(
                    '<div class="status status-bad"><span class="status-dot"></span>No arbitrage-free solution.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    card("Implied Volatility Solver Result",
                         [
                             {"sub": "Solved IV", "value": f"{solved_iv * 100:.2f}%"},
                             {"sub": "Model Input σ", "value": f"{sigma * 100:.2f}%"}
                         ],
                         accent=GOOD),
                    unsafe_allow_html=True,
                )

# --------------------------------------------------------------------------- #
# Center Column: Prices, Greeks, & Plots                                      #
# --------------------------------------------------------------------------- #
with col_center:
    st.markdown('<div class="section-label">4. Valuation Engines</div>', unsafe_allow_html=True)
    
    # Pricing Models
    bt = BinomialTreeModel(S, K, T, r, sigma, q, n_steps=500)
    mc = MonteCarloOptionPricer(S, K, T, r, sigma, q, n_simulations=100_000, seed=1)
    mc_call, mc_call_se = mc.call_price(return_std_error=True)
    mc_put, mc_put_se = mc.put_price(return_std_error=True)
    bt_call, bt_put = bt.call_price(), bt.put_price()

    p_col1, p_col2, p_col3 = st.columns(3)
    p_col1.markdown(
        card("Black-Scholes",
             [
                 {"sub": "Call", "value": f"${bs_call:.4f}"},
                 {"sub": "Put", "value": f"${bs_put:.4f}"}
             ],
             accent=CALL),
        unsafe_allow_html=True,
    )
    p_col2.markdown(
        card("Monte Carlo (100k)",
             [
                 {"sub": "Call", "value": f"${mc_call:.4f}", "se": f"± {mc_call_se:.4f}"},
                 {"sub": "Put", "value": f"${mc_put:.4f}", "se": f"± {mc_put_se:.4f}"}
             ],
             accent=PUT),
        unsafe_allow_html=True,
    )
    p_col3.markdown(
        card("Binomial Tree (500)",
             [
                 {"sub": "Call", "value": f"${bt_call:.4f}"},
                 {"sub": "Put", "value": f"${bt_put:.4f}"}
             ],
             accent=ACCENT),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label" style="margin-top:1.2rem;">5. First-Order Greeks Cluster (Call)</div>', unsafe_allow_html=True)
    g = Greeks(S, K, T, r, sigma, q)
    g_col1, g_col2, g_col3, g_col4, g_col5 = st.columns(5)
    g_col1.markdown(card("Delta", [{"sub": "Call", "value": f"{g.delta_call():.4f}"}]), unsafe_allow_html=True)
    g_col2.markdown(card("Gamma", [{"sub": "Value", "value": f"{g.gamma():.4f}"}]), unsafe_allow_html=True)
    g_col3.markdown(card("Vega", [{"sub": "Value", "value": f"{g.vega():.4f}"}]), unsafe_allow_html=True)
    g_col4.markdown(card("Theta", [{"sub": "Call", "value": f"{g.theta_call():.4f}"}]), unsafe_allow_html=True)
    g_col5.markdown(card("Rho", [{"sub": "Call", "value": f"{g.rho_call():.4f}"}]), unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1.2rem;">6. Advanced Visualizations</div>', unsafe_allow_html=True)
    # Placeholder tabs for Task 8
    tab_surface, tab_mc_paths, tab_smile = st.tabs([
        "Sensitivity Surface (3D)",
        "Monte Carlo Price Paths",
        "Volatility Smile / Skew"
    ])
    with tab_surface:
        labels = {
            "delta_call": "Delta", "gamma": "Gamma",
            "vega": "Vega", "theta_call": "Theta",
        }
        greek = st.selectbox(
            "Sensitivity Surface target Z-axis:", list(labels), format_func=labels.get, key="surf_greek_select"
        )
        s_axis = np.linspace(S * 0.5, S * 1.5, 30)
        t_axis = np.linspace(0.05, max(T * 2, 0.1), 30)
        s_mesh, t_mesh = np.meshgrid(s_axis, t_axis)
        surface = np.array(
            [[getattr(Greeks(s, K, t, r, sigma, q), greek)() for s in s_axis]
             for t in t_axis]
        )
        fig_s = go.Figure(
            go.Surface(
                x=s_mesh, y=t_mesh, z=surface, colorscale=BLUE_SEQ,
                colorbar=dict(title=labels[greek], outlinewidth=0, tickfont=dict(size=11, color=INK_2)),
            )
        )
        fig_s.update_layout(
            title=f"{labels[greek]} across spot and maturity",
            scene=dict(
                xaxis=dict(title="Spot", backgroundcolor=BG_PAGE, gridcolor=GRID),
                yaxis=dict(title="Maturity", backgroundcolor=BG_PAGE, gridcolor=GRID),
                zaxis=dict(title=labels[greek], backgroundcolor=BG_PAGE, gridcolor=GRID),
            ),
        )
        st.plotly_chart(style_fig(fig_s, height=450), use_container_width=True)

    with tab_mc_paths:
        mc_paths_gen = MonteCarloOptionPricer(S, K, T, r, sigma, q, n_simulations=100, seed=42)
        paths = mc_paths_gen.generate_paths(100)
        steps = len(paths) - 1
        time_grid = np.linspace(0, T, steps + 1)
        
        fig_mc = go.Figure()
        # Add paths with high transparency
        for i in range(100):
            fig_mc.add_scatter(x=time_grid, y=paths[:, i], mode="lines",
                               line=dict(width=1.0, color="rgba(37, 99, 235, 0.16)"),
                               showlegend=False)
                               
        # Add strike line
        fig_mc.add_hline(y=K, line_dash="dash", line_color=CRIT, line_width=1.5,
                         annotation_text=f"Strike K={K}", annotation_position="top left",
                         annotation_font=dict(color=INK_2, family=FONT))
        
        # Add spot line
        fig_mc.add_hline(y=S, line_dash="dot", line_color=CALL, line_width=1.0,
                         annotation_text=f"Spot S={S}", annotation_position="bottom left",
                         annotation_font=dict(color=INK_2, family=FONT))
                         
        fig_mc.update_layout(
            title="Monte Carlo Asset Price Path Simulations (GBM)",
            xaxis_title="Time to Maturity (years)",
            yaxis_title="Underlying Asset Price ($)",
            showlegend=False
        )
        st.plotly_chart(style_fig(fig_mc, height=450), use_container_width=True)

    with tab_smile:
        smile_ticker = st.selectbox("Select ticker for Volatility Smile:", ["^SPX", "^NDX", "^RUT"], key="smile_ticker")
        smile_skeleton = st.empty()
        smile_skeleton.markdown(
            '<div class="skeleton" style="height:450px;"></div>', unsafe_allow_html=True
        )
        try:
            calls_df, puts_df, smile_spot, smile_expiry = fetch_option_chain(smile_ticker, use_cache=True)
            smile_skeleton.empty()

            calls_df = calls_df.copy()
            puts_df = puts_df.copy()
            calls_df["mid"] = (calls_df["bid"] + calls_df["ask"]) / 2
            calls_df["mid"] = np.where(calls_df["mid"] > 0, calls_df["mid"], calls_df["lastPrice"])
            puts_df["mid"] = (puts_df["bid"] + puts_df["ask"]) / 2
            puts_df["mid"] = np.where(puts_df["mid"] > 0, puts_df["mid"], puts_df["lastPrice"])
            
            common = set(calls_df["strike"]).intersection(set(puts_df["strike"]))
            if common:
                calls_df = calls_df[calls_df["strike"].isin(common)]
                puts_df = puts_df[puts_df["strike"].isin(common)]
                calls_df["dist"] = (calls_df["strike"] - smile_spot).abs()
                top_s = calls_df.nsmallest(7, "dist")["strike"].tolist()
                
                c_filtered = calls_df[calls_df["strike"].isin(top_s)].sort_values("strike")
                p_filtered = puts_df[puts_df["strike"].isin(top_s)].sort_values("strike")
                
                smile_strikes = c_filtered["strike"].tolist()
                solved_call_ivs = []
                solved_put_ivs = []
                
                for _, c_row in c_filtered.iterrows():
                    k_strk = float(c_row["strike"])
                    c_iv = ImpliedVolatilityCalculator.calculate(float(c_row["mid"]), smile_spot, k_strk, T, r, "call", q)
                    solved_call_ivs.append(c_iv * 100 if c_iv else np.nan)
                    
                for _, p_row in p_filtered.iterrows():
                    k_strk = float(p_row["strike"])
                    p_iv = ImpliedVolatilityCalculator.calculate(float(p_row["mid"]), smile_spot, k_strk, T, r, "put", q)
                    solved_put_ivs.append(p_iv * 100 if p_iv else np.nan)
                    
                fig_smile = go.Figure()
                fig_smile.add_scatter(x=smile_strikes, y=solved_call_ivs, mode="lines+markers", name="Call Implied Vol", line=dict(color=CALL))
                fig_smile.add_scatter(x=smile_strikes, y=solved_put_ivs, mode="lines+markers", name="Put Implied Vol", line=dict(color=PUT))
                fig_smile.add_vline(x=smile_spot, line_dash="dot", line_color=INK_MUTED, annotation_text=f"Spot: {smile_spot:.2f}")
                fig_smile.update_layout(
                    title=f"Solved Implied Volatility Smile/Skew ({smile_ticker})",
                    xaxis_title="Strike Price ($)",
                    yaxis_title="Implied Volatility (%)",
                    legend=dict(orientation="h", y=1.1)
                )
                st.plotly_chart(style_fig(fig_smile, height=450), use_container_width=True)
            else:
                st.info("No common strikes found for volatility smile.")
        except Exception as e:
            smile_skeleton.empty()
            st.warning(f"Could not load volatility smile: {e}")

# --------------------------------------------------------------------------- #
# Right Column: Diagnostics & Historical Volatility                            #
# --------------------------------------------------------------------------- #
with col_right:
    st.markdown('<div class="section-label">7. Put-Call Parity Diagnostic</div>', unsafe_allow_html=True)
    res = PutCallParity.verify(bs_call, bs_put, S, K, r, T, q)
    st.markdown(
        card("Parity Check Details",
             [
                 {"sub": "C - P", "value": f"${res['lhs']:.4f}"},
                 {"sub": "Forward Side", "value": f"${res['rhs']:.4f}"}
             ],
             accent=GOOD if res["parity_holds"] else CRIT),
        unsafe_allow_html=True,
    )
    if res["parity_holds"]:
        st.markdown(
            '<div class="status status-good"><span class="status-dot"></span>Parity Holds (diff: {:.6f})</div>'.format(res['difference']),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status status-bad"><span class="status-dot"></span>Parity Violated (diff: {:.4f})</div>'.format(res['difference']),
            unsafe_allow_html=True,
        )

    hist_ticker = st.selectbox("Select Ticker for Volatility Analysis:", ["^SPX", "^NDX", "^RUT"], key="hist_ticker")
    hist_skeleton = st.empty()
    hist_skeleton.markdown(
        '<div class="skeleton" style="height:280px;"></div>', unsafe_allow_html=True
    )
    try:
        hist_df = fetch_10y_historical_data(hist_ticker, use_cache=True)
        hist_skeleton.empty()
        if not hist_df.empty:
            hist_df["Returns"] = np.log(hist_df["Close"] / hist_df["Close"].shift(1))
            hist_df["Vol30"] = hist_df["Returns"].rolling(30).std() * np.sqrt(252) * 100
            hist_df["Vol252"] = hist_df["Returns"].rolling(252).std() * np.sqrt(252) * 100
            
            recent_30 = hist_df["Vol30"].iloc[-1]
            recent_252 = hist_df["Vol252"].iloc[-1]
            avg_10y = hist_df["Vol30"].mean()
            
            st.markdown(
                card("Volatility Statistics",
                     [
                         {"sub": "30D Rolling", "value": f"{recent_30:.2f}%"},
                         {"sub": "252D Rolling", "value": f"{recent_252:.2f}%"},
                         {"sub": "10Y Average", "value": f"{avg_10y:.2f}%"}
                     ],
                     accent=CALL),
                unsafe_allow_html=True,
            )
            
            plot_df = hist_df.dropna(subset=["Vol30"]).tail(500)
            fig_hist = go.Figure()
            fig_hist.add_scatter(x=plot_df.index, y=plot_df["Vol30"], mode="lines", 
                                 line=dict(color=PUT, width=1.5), name="30D Vol")
            fig_hist.add_scatter(x=plot_df.index, y=plot_df["Vol252"], mode="lines", 
                                 line=dict(color=CALL, width=1.5, dash="dash"), name="252D Vol")
            fig_hist.update_layout(
                title=f"Historical Volatility (Last 500 Trading Days)",
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(style_fig(fig_hist, height=220), use_container_width=True)
    except Exception as e:
        hist_skeleton.empty()
        st.warning(f"Could not load historical volatility: {e}")

    st.markdown('<div class="section-label" style="margin-top:1.2rem;">8. Historical Data Export</div>', unsafe_allow_html=True)
    if st.button("Export ^SPX / ^NDX / ^RUT to Excel (Max History)"):
        with st.spinner("Fetching maximum available history for all three indices..."):
            try:
                xlsx_path = export_indices_to_excel(period="max", use_cache=True)
                st.session_state["export_xlsx_bytes"] = xlsx_path.read_bytes()
                st.session_state["export_error"] = None
            except Exception as e:
                st.session_state["export_error"] = str(e)
                st.session_state["export_xlsx_bytes"] = None

    if st.session_state.get("export_error"):
        st.markdown(
            f'<div class="status status-bad"><span class="status-dot"></span>Export failed: {st.session_state["export_error"]}</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.get("export_xlsx_bytes"):
        st.markdown(
            '<div class="status status-good"><span class="status-dot"></span>Workbook ready — ^SPX, ^NDX, ^RUT, max available history.</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            label="Download indices_historical_data.xlsx",
            data=st.session_state["export_xlsx_bytes"],
            file_name="indices_historical_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_indices_xlsx",
        )

# --------------------------------------------------------------------------- #
# One-shot entrance flourish: count the metric cards up on first load only,   #
# so dragging sliders afterwards stays instant and doesn't fight the user.    #
# --------------------------------------------------------------------------- #
if not st.session_state.get("_entrance_played"):
    play_entrance_count_up()
    st.session_state["_entrance_played"] = True
