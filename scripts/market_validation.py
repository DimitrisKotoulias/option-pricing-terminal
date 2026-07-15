"""Live market validation: fetch real index option chains from Yahoo Finance
and check the toolkit's Black-Scholes prices / implied vols / put-call parity.

Run: python scripts/market_validation.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from optpricing import BlackScholesModel, ImpliedVolatilityCalculator, PutCallParity
from optpricing.data.market_data_fetcher import (
    fetch_option_chain,
    historical_volatility,
    fetch_risk_free_rate,
    fetch_dividend_yield,
)

TICKERS = ["^SPX", "^NDX", "^RUT"]   # S&P 500, Nasdaq-100, Russell 2000 indices
DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"


def year_fraction(expiry: str) -> float:
    exp = dt.datetime.strptime(expiry, "%Y-%m-%d").date()
    return max((exp - dt.date.today()).days, 1) / 365.0


def expiries_near(ticker: str, target_days: int = 30) -> list[str]:
    """Listed expiries sorted by closeness to `target_days` (avoids 0-DTE noise)."""
    import yfinance as yf

    opts = list(yf.Ticker(ticker).options)
    return sorted(opts, key=lambda e: abs((year_fraction(e) * 365) - target_days))


def near_atm_chains(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find 5 near-ATM strikes where both call and put have bids and asks, and return filtered DataFrames."""
    calls = calls.copy()
    puts = puts.copy()

    # Calculate mid price
    calls["mid"] = (calls["bid"] + calls["ask"]) / 2
    calls["mid"] = np.where(calls["mid"] > 0, calls["mid"], calls["lastPrice"])
    puts["mid"] = (puts["bid"] + puts["ask"]) / 2
    puts["mid"] = np.where(puts["mid"] > 0, puts["mid"], puts["lastPrice"])

    # Filter for valid positive prices
    calls = calls[calls["mid"] > 0]
    puts = puts[puts["mid"] > 0]

    # Find common strikes
    common_strikes = set(calls["strike"]).intersection(set(puts["strike"]))
    if not common_strikes:
        return pd.DataFrame(), pd.DataFrame()

    calls = calls[calls["strike"].isin(common_strikes)].copy()
    puts = puts[puts["strike"].isin(common_strikes)].copy()

    # Sort by distance from spot
    calls["dist"] = (calls["strike"] - spot).abs()
    top_strikes = calls.nsmallest(5, "dist")["strike"].tolist()

    calls_filtered = calls[calls["strike"].isin(top_strikes)].sort_values("strike")
    puts_filtered = puts[puts["strike"].isin(top_strikes)].sort_values("strike")

    return calls_filtered, puts_filtered


def validate(ticker: str, target_days: int = 30, use_cache: bool = True) -> str:
    print(f"\n{'='*72}\n{ticker}\n{'='*72}")
    near_calls = pd.DataFrame()
    near_puts = pd.DataFrame()
    expiry = None
    spot = None
    try:
        for cand in expiries_near(ticker, target_days=target_days)[:6]:
            calls, puts, spot, expiry = fetch_option_chain(
                ticker, expiry=cand, use_cache=use_cache
            )
            near_calls, near_puts = near_atm_chains(calls, puts, spot)
            if not near_calls.empty:
                break
    except Exception as e:  # noqa: BLE001
        print(f"  fetch failed: {e!r}")
        return f"### {ticker}\nFetch failed: {e!r}\n"
        
    if near_calls.empty:
        print("  no live quotes on any near-term expiry — skipping")
        return f"### {ticker}\nNo live quotes found.\n"

    T = year_fraction(expiry)
    hv = historical_volatility(ticker, period="6mo")
    r = fetch_risk_free_rate()
    q = fetch_dividend_yield(ticker)
    
    print(f"  spot        = {spot:,.2f}")
    print(f"  expiry      = {expiry}  (T = {T:.4f} yr)")
    print(f"  hist. vol   = {hv*100:.2f}%   r = {r*100:.1f}%   q = {q*100:.1f}%")

    rows = []
    strikes = []
    call_ivs = []
    put_ivs = []
    
    for (_, c), (_, p) in zip(near_calls.iterrows(), near_puts.iterrows()):
        K = float(c["strike"])
        strikes.append(K)
        
        c_mkt = float(c["mid"])
        p_mkt = float(p["mid"])
        
        # Call pricing & IV solving
        bs_call_hv = BlackScholesModel(spot, K, T, r, hv, q).call_price()
        c_iv = ImpliedVolatilityCalculator.calculate(c_mkt, spot, K, T, r, "call", q)
        c_iv_mkt = float(c.get("impliedVolatility", np.nan))
        c_rt = BlackScholesModel(spot, K, T, r, c_iv, q).call_price() if c_iv is not None else np.nan
        call_ivs.append(c_iv if c_iv is not None else np.nan)
        
        # Put pricing & IV solving
        bs_put_hv = BlackScholesModel(spot, K, T, r, hv, q).put_price()
        p_iv = ImpliedVolatilityCalculator.calculate(p_mkt, spot, K, T, r, "put", q)
        p_iv_mkt = float(p.get("impliedVolatility", np.nan))
        p_rt = BlackScholesModel(spot, K, T, r, p_iv, q).put_price() if p_iv is not None else np.nan
        put_ivs.append(p_iv if p_iv is not None else np.nan)
        
        # Put-Call Parity
        parity = PutCallParity.verify(c_mkt, p_mkt, spot, K, r, T, q)
        
        rows.append(
            {
                "strike": K,
                "call_mkt": c_mkt,
                "call_our_IV": None if c_iv is None else round(c_iv * 100, 2),
                "call_y_IV": round(c_iv_mkt * 100, 2) if c_iv_mkt == c_iv_mkt else None,
                "put_mkt": p_mkt,
                "put_our_IV": None if p_iv is None else round(p_iv * 100, 2),
                "put_y_IV": round(p_iv_mkt * 100, 2) if p_iv_mkt == p_iv_mkt else None,
                "parity_diff": round(parity["difference"], 4),
                "parity_holds": parity["parity_holds"]
            }
        )
        
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    print("\n" + df.to_string(index=False))

    # Plot Volatility Smile
    plt.figure(figsize=(8, 4))
    plt.plot(strikes, [v * 100 if v == v else np.nan for v in call_ivs], 'o-', label="Call Solved IV", color='#3b82f6')
    plt.plot(strikes, [v * 100 if v == v else np.nan for v in put_ivs], 's--', label="Put Solved IV", color='#44e2cd')
    plt.axvline(x=spot, color='gray', linestyle=':', label=f"Spot ({spot:.1f})")
    plt.title(f"Implied Volatility Smile for {ticker} (Expiry: {expiry})")
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility (%)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    smile_path = DOCS_DIR / f"volatility_smile_{ticker}.png"
    plt.savefig(smile_path)
    plt.close()
    print(f"\n  Saved volatility smile plot to {smile_path}")

    # Build report section
    report = f"### {ticker} Validation\n"
    report += f"- **Spot**: {spot:,.2f}\n"
    report += f"- **Expiry**: {expiry} (T = {T:.4f} yr)\n"
    report += f"- **Parameters**: $r = {r*100:.2f}\\%$, $q = {q*100:.2f}\\%$, Historical Vol (6m) = {hv*100:.2f}%\n\n"
    report += df.to_markdown(index=False) + "\n\n"
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Options Pricing Toolkit Market Validation")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=TICKERS,
        help="List of tickers to validate (e.g., ^SPX ^NDX ^RUT)",
    )
    parser.add_argument(
        "--target-days",
        type=int,
        default=30,
        help="Target option maturity in days (default: 30)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass local cache for option chains",
    )
    args = parser.parse_args()

    use_cache = not args.no_cache
    full_report = "# Market Validation Report\n\nGenerated on " + str(dt.date.today()) + "\n\n"
    
    for tk in args.tickers:
        full_report += validate(tk, target_days=args.target_days, use_cache=use_cache)
    
    report_file = DOCS_DIR / "validation_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"\nSaved market validation report to {report_file}")
