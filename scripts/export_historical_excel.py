"""Export historical daily data for market indices to a single Excel workbook.

Run with::

    pip install -e ".[data]"
    python scripts/export_historical_excel.py
"""

from __future__ import annotations

from pathlib import Path

from optpricing.data.market_data_fetcher import export_indices_to_excel

TICKERS = ["^SPX", "^NDX", "^RUT"]  # S&P 500, Nasdaq-100, Russell 2000 indices


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Export historical daily data for market indices to a "
        "single multi-sheet Excel workbook."
    )
    parser.add_argument(
        "--tickers", nargs="+", default=TICKERS,
        help="Tickers to export (default: %(default)s)",
    )
    parser.add_argument(
        "--period", default="max",
        help="yfinance history period, e.g. 'max', '10y', '5y' (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .xlsx path (default: data/exports/indices_historical_data.xlsx)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Bypass the on-disk CSV cache and re-fetch from Yahoo Finance",
    )
    args = parser.parse_args()

    output_path = export_indices_to_excel(
        tickers=args.tickers,
        period=args.period,
        output_path=args.output,
        use_cache=not args.no_cache,
    )
    print(f"Wrote historical data for {', '.join(args.tickers)} to {output_path}")


if __name__ == "__main__":
    main()
