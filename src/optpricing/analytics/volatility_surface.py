"""Build a market implied-volatility surface from multi-expiry option chains.

Given real option chains across several expiries (see
:func:`optpricing.data.market_data_fetcher.fetch_option_surface`), this solves
the Black-Scholes implied volatility for each liquid strike and assembles a
smoothed ``moneyness x maturity`` surface ready for a 3D plot.

Two design choices keep the surface clean:

* **OTM side only** -- calls for strikes at/above spot, puts below -- which is
  the standard market convention and avoids the wide bid/ask of deep in-the-money
  quotes.
* **Per-expiry true maturity** -- each expiry uses its own time to expiry ``T``,
  fixing the common bug of solving every strike at one blanket maturity.

Smoothing is either a 2D ``scipy.interpolate.griddata`` interpolation (default)
or a per-expiry degree-2 ``numpy.polyfit`` smile fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import griddata

from optpricing.analytics.implied_volatility import ImpliedVolatilityCalculator


def _mid_prices(df):
    """Mid price per row: ``(bid+ask)/2``, falling back to ``lastPrice``."""
    mid = (df["bid"] + df["ask"]) / 2.0
    return np.where(mid > 0, mid, df["lastPrice"])


@dataclass
class VolatilitySurface:
    """A smoothed implied-volatility surface plus the raw solved points.

    All volatilities are in **percent**. ``iv_mesh`` has shape
    ``(len(maturity_axis), len(moneyness_axis))``.
    """

    moneyness_axis: np.ndarray
    maturity_axis: np.ndarray
    iv_mesh: np.ndarray
    raw_moneyness: np.ndarray
    raw_maturity: np.ndarray
    raw_iv: np.ndarray
    spot: float

    @classmethod
    def from_chains(
        cls,
        records,
        r,
        q: float = 0.0,
        moneyness_range: tuple[float, float] = (0.7, 1.3),
        grid_size: int = 40,
        smoothing: str = "griddata",
    ) -> "VolatilitySurface":
        """Assemble a surface from :func:`fetch_option_surface` records.

        Parameters
        ----------
        records : list of dict
            Each with ``expiry``, ``T``, ``calls``, ``puts`` and ``spot``.
        r : float or callable
            Risk-free rate. A callable ``r(T)`` is evaluated per expiry (e.g.
            ``risk_free_rate_for``); a float is applied to every expiry.
        q : float
            Continuous dividend yield.
        moneyness_range : (float, float)
            Keep only strikes with ``K/S`` inside this window (near-ATM, liquid).
        grid_size : int
            Resolution of the smoothed mesh along each axis.
        smoothing : {"griddata", "polyfit"}
            2D interpolation, or per-expiry degree-2 smile fit.
        """
        rate_for = r if callable(r) else (lambda _T: float(r))

        m_lo, m_hi = moneyness_range
        raw_m, raw_t, raw_iv = [], [], []
        spot = None

        for rec in records:
            spot = float(rec["spot"])
            if spot <= 0:
                continue
            T = float(rec["T"])
            r_t = rate_for(T)

            for df, opt_type, keep in (
                (rec["calls"], "call", lambda k: k >= spot),
                (rec["puts"], "put", lambda k: k < spot),
            ):
                if df is None or len(df) == 0:
                    continue
                frame = df.copy()
                frame["mid"] = _mid_prices(frame)
                moneyness = frame["strike"] / spot
                sel = keep(frame["strike"]) & (moneyness >= m_lo) & (moneyness <= m_hi)
                sel = sel & (frame["mid"] > 0)
                frame = frame[sel]
                if frame.empty:
                    continue

                ivs = ImpliedVolatilityCalculator.calculate_vectorized(
                    frame["mid"].to_numpy(),
                    spot,
                    frame["strike"].to_numpy(),
                    T,
                    r_t,
                    opt_type,
                    q,
                )
                ok = ~np.isnan(ivs)
                if not ok.any():
                    continue
                strikes = frame["strike"].to_numpy()[ok]
                iv_pct = ivs[ok] * 100.0
                # Robust per-expiry wing-noise trim (median +/- k*MAD): a handful
                # of illiquid deep-OTM quotes otherwise spike the surface. Keeps
                # genuine smile curvature while cutting outliers for a smooth mesh.
                med = np.median(iv_pct)
                mad = np.median(np.abs(iv_pct - med))
                if mad > 0:
                    keep = np.abs(iv_pct - med) <= 4.0 * 1.4826 * mad
                    strikes, iv_pct = strikes[keep], iv_pct[keep]
                if iv_pct.size == 0:
                    continue
                raw_m.extend((strikes / spot).tolist())
                raw_t.extend([T] * iv_pct.size)
                raw_iv.extend(iv_pct.tolist())

        raw_m = np.asarray(raw_m, dtype=float)
        raw_t = np.asarray(raw_t, dtype=float)
        raw_iv = np.asarray(raw_iv, dtype=float)

        if raw_m.size == 0:
            raise ValueError("No implied volatilities could be solved for the surface.")

        moneyness_axis, maturity_axis, iv_mesh = cls._build_grid(
            raw_m, raw_t, raw_iv, grid_size, smoothing
        )
        return cls(
            moneyness_axis=moneyness_axis,
            maturity_axis=maturity_axis,
            iv_mesh=iv_mesh,
            raw_moneyness=raw_m,
            raw_maturity=raw_t,
            raw_iv=raw_iv,
            spot=float(spot),
        )

    @staticmethod
    def _build_grid(raw_m, raw_t, raw_iv, grid_size, smoothing):
        # Winsorize to the 2nd-98th percentile so a few residual short-dated
        # needles are capped rather than towering over the smoothed mesh. The
        # raw scatter points keep their true (unclipped) values.
        if raw_iv.size >= 5:
            lo, hi = np.percentile(raw_iv, [2, 98])
            raw_iv = np.clip(raw_iv, lo, hi)
        moneyness_axis = np.linspace(raw_m.min(), raw_m.max(), grid_size)
        uniq_t = np.unique(raw_t)

        # A single expiry (or an explicit request) -> per-expiry polynomial smile:
        # 2D triangulation needs >= 2 distinct maturities, so fall back here.
        if smoothing == "polyfit" or uniq_t.size < 2:
            maturity_axis = np.sort(uniq_t)
            rows = []
            for t in maturity_axis:
                mask = raw_t == t
                m, v = raw_m[mask], raw_iv[mask]
                if m.size >= 3:
                    coeffs = np.polyfit(m, v, min(2, m.size - 1))
                    rows.append(np.polyval(coeffs, moneyness_axis))
                elif m.size >= 1:
                    rows.append(np.full(grid_size, float(np.mean(v))))
                else:  # pragma: no cover - defensive
                    rows.append(np.full(grid_size, np.nan))
            # A degree-2 fit extrapolates hard in the wings (observed 292% out of
            # a 143% chain), so hold it inside the observed band too. NaNs kept.
            return (
                moneyness_axis,
                maturity_axis,
                np.clip(np.asarray(rows), float(raw_iv.min()), float(raw_iv.max())),
            )

        maturity_axis = np.linspace(raw_t.min(), raw_t.max(), grid_size)
        mesh_m, mesh_t = np.meshgrid(moneyness_axis, maturity_axis)
        points = np.column_stack([raw_m, raw_t])

        # The mesh must never leave the band spanned by the observed quotes: an
        # interpolated surface that invents volatilities the market never printed
        # is not a market surface. Real chains pack dozens of scattered strikes
        # into a razor-thin near-expiry slice sitting an order of magnitude
        # closer in T than the next maturity, which gives the cubic
        # (Clough-Tocher) triangulation near-degenerate triangles and wild
        # gradients -- observed peaks of ~7700% IV against raw quotes topping out
        # at 112%. Cubic is still tried first for its smoother look, but it is
        # only accepted when it stays inside the band; otherwise the piecewise-
        # linear interpolant (a convex combination within each triangle, so it
        # cannot overshoot) is used.
        band_lo, band_hi = float(raw_iv.min()), float(raw_iv.max())

        iv_mesh = None
        for method in ("cubic", "linear", "nearest"):
            try:
                grid = griddata(points, raw_iv, (mesh_m, mesh_t), method=method)
            except Exception:  # pragma: no cover - Qhull edge cases -> next method
                continue
            if grid is None or np.isnan(grid).all():
                continue
            if np.isnan(grid).any():  # fill hull gaps with nearest neighbour
                nn = griddata(points, raw_iv, (mesh_m, mesh_t), method="nearest")
                grid = np.where(np.isnan(grid), nn, grid)
            if method == "cubic" and not (
                np.nanmin(grid) >= band_lo - 1e-9 and np.nanmax(grid) <= band_hi + 1e-9
            ):
                continue  # overshot the observed quotes -> fall back to linear
            iv_mesh = grid
            break
        if iv_mesh is None:  # pragma: no cover - defensive fallback
            iv_mesh = np.full(mesh_m.shape, float(np.nanmean(raw_iv)))
        # Belt-and-braces: whichever interpolant ran, keep the mesh inside the
        # observed band (NaNs preserved).
        iv_mesh = np.clip(iv_mesh, band_lo, band_hi)
        return moneyness_axis, maturity_axis, iv_mesh
