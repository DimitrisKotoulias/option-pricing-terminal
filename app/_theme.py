"""Design system + Plotly/card/KPI helpers for the Streamlit dashboard.

Pure presentation: colours, the injected CSS, the shared Plotly theme, and the
metric-card / KPI-tile HTML builders.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

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
CALL = "#2f6fed"  # categorical slot 1 — blue
PUT = "#0d9488"  # categorical slot 2 — teal
ACCENT = "#2563eb"
GOOD = "#047857"
WARN = "#b45309"
CRIT = "#b91c1c"
FONT = "Inter, system-ui, -apple-system, sans-serif"
MONO_FONT = '"JetBrains Mono", monospace'

# Rainbow heatmap ramp for the 3D surfaces and Monte Carlo paths. "Jet" is the
# classic MATLAB rainbow (dark blue -> cyan -> green -> yellow -> red).
RAINBOW = "Jet"

# Visible gridlines on the 3D scene walls (the dotted back-panel squares of a
# MATLAB surf() plot).
SCENE_GRID = "rgba(15,23,42,0.16)"


def mesh_contours(x_min, x_max, y_min, y_max, n=24, color="rgba(15,23,42,0.35)"):
    """Wireframe of ~``n`` x ``n`` squares over a 3D surface (MATLAB surf() look).

    ``contours`` at a fixed *step* (rather than Plotly's default level count)
    draws an evenly-spaced grid of lines along the surface, so the mesh reads as
    regular little squares regardless of each axis's data range.
    """
    x_size = (x_max - x_min) / n if x_max > x_min else 1
    y_size = (y_max - y_min) / n if y_max > y_min else 1
    return dict(
        x=dict(show=True, color=color, width=1, start=x_min, end=x_max, size=x_size),
        y=dict(show=True, color=color, width=1, start=y_min, end=y_max, size=y_size),
    )


def inject_css() -> None:
    st.html(f"""
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet" />
        <style>
        /* strip default Streamlit chrome */
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
            margin: .2rem 0 .8rem;
        }}
        /* tighter vertical rhythm between stacked Streamlit blocks */
        [data-testid="stVerticalBlock"] {{ gap: .7rem; }}

        /* KPI strip — headline numbers directly under the header */
        .kpi-strip {{
            display: flex; gap: .6rem; flex-wrap: wrap;
            margin: .1rem 0 1.4rem;
        }}
        .kpi {{
            flex: 1 1 0; min-width: 118px;
            background: {BG_SURFACE};
            backdrop-filter: blur(12px);
            border: 1px solid {BORDER};
            border-top: 2px solid var(--accent, {ACCENT});
            border-radius: 8px; padding: .6rem .8rem;
            box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 4px 10px rgba(15,23,42,.05);
            animation: card-in .45s cubic-bezier(.16,1,.3,1) both;
            animation-delay: calc(var(--card-i, 0) * 40ms);
            transition: transform .18s ease, box-shadow .18s ease;
        }}
        .kpi:hover {{
            transform: translateY(-2px);
            box-shadow: 0 1px 2px rgba(15,23,42,.05), 0 10px 22px rgba(15,23,42,.10);
        }}
        .kpi-label {{
            font-size: .62rem; font-weight: 600; letter-spacing: .14em;
            text-transform: uppercase; color: {INK_MUTED};
        }}
        .kpi-value {{
            font-size: 1.18rem; font-weight: 650; color: {INK};
            font-family: {MONO_FONT}; font-variant-numeric: tabular-nums;
            margin-top: .18rem; line-height: 1.1;
        }}
        .kpi-value.pos {{ color: {GOOD}; }}
        .kpi-value.neg {{ color: {CRIT}; }}
        .sync-caption {{
            font-size: .68rem; color: {INK_MUTED}; margin: .1rem 0 .6rem;
            font-family: {MONO_FONT};
        }}
        /* Data older than the last close: amber, so cached values can never
           pass for a fresh sync. */
        .sync-caption.stale {{ color: {WARN}; font-weight: 600; }}

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
        label p {{
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
        """)


def style_fig(fig: go.Figure, height: int | None = None) -> go.Figure:
    """Apply the shared Plotly theme to a figure."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK_2, size=13),
        title=dict(font=dict(family=FONT, color=INK, size=15), x=0, xanchor="left"),
        margin=dict(l=10, r=10, t=48, b=10),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
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


# Entrance-animation stagger counters. Reset once per Streamlit run via
# ``reset_stagger`` from the entry script (imported modules don't re-execute on
# rerun, so the counters would otherwise grow unbounded across reruns).
_card_sequence = 0
_kpi_sequence = 0


def reset_stagger() -> None:
    """Reset the card/KPI entrance-stagger counters."""
    global _card_sequence, _kpi_sequence
    _card_sequence = 0
    _kpi_sequence = 0


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


def kpi(label: str, value: str, accent: str = ACCENT, tone: str = "") -> str:
    """Render one compact headline tile for the KPI strip."""
    global _kpi_sequence
    stagger_index = _kpi_sequence
    _kpi_sequence += 1
    tone_cls = f" {tone}" if tone else ""
    return (
        f'<div class="kpi" style="--accent:{accent}; --card-i:{stagger_index};">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value{tone_cls}">{value}</div></div>'
    )


def play_entrance_count_up() -> None:
    """One-shot count-up animation for `.card-value` numbers on first load.

    Runs inside a zero-size iframe (`window.parent.document` reaches out of
    that iframe into the app DOM — the standard escape hatch for injecting
    real JS into Streamlit, since `st.html`/`st.markdown` strip <script>
    tags). Gated to the first script run per session via `st.session_state`
    so it never re-fires and fights the user while they are dragging sliders;
    if the iframe ever can't reach the parent document the try/catch just
    leaves the numbers as-is, so it fails safe.
    """
    st.iframe(
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
        # 1px rather than 0: st.iframe rejects non-positive dimensions. The
        # element only renders on the first run of a session, so it never
        # accumulates page height.
        width=1,
        height=1,
    )
