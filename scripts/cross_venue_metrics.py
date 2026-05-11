"""Compute the formal cross-venue metrics table.

Four headline numbers per venue, computed on the same UTC window:

    kyle_lambda          (60-second bins)
    round_trip_rate      (60-second window, 1% qty tolerance)
    top_n_size_share     (top 10 distinct sizes)
    buy_sell_ratio       (volume-weighted)

All venues are clipped to the challenge window so the comparison is
apples-to-apples. The output is written to two places:

    notebooks/figures/provisional/cross_venue_metrics.csv
    stdout                          (rendered as a fixed-width table)

Usage:
    uv run python scripts/cross_venue_metrics.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from challenge.analysis.manipulation import (
    buy_sell_ratio,
    kyle_lambda,
    round_trip_rate,
    top_n_size_share,
)
from challenge.sources.csv_loader import load_trades

CHALLENGE_TRADES = Path("data/eth-btc-trades.csv")
KRAKEN_CACHE = Path("cache/kraken_xethxxbt_20250901_20250904.parquet")
BINANCE_CACHE = Path("cache/binance_ethbtc_20250901_20250904.parquet")
OUT_CSV = Path("notebooks/figures/provisional/cross_venue_metrics.csv")


def _clip(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df.loc[(df["ts"] >= start) & (df["ts"] <= end)].reset_index(drop=True)


def main() -> int:
    challenge = load_trades(CHALLENGE_TRADES)
    kraken = pd.read_parquet(KRAKEN_CACHE)
    binance = pd.read_parquet(BINANCE_CACHE)

    start, end = challenge["ts"].min(), challenge["ts"].max()
    print(f"window: {start}  ->  {end}")

    venues = {
        "challenge": challenge,
        "kraken":    _clip(kraken, start, end),
        "binance":   _clip(binance, start, end),
    }

    rows: list[dict] = []
    for name, df in venues.items():
        rows.append(
            {
                "venue":             name,
                "n_trades":          len(df),
                "total_volume":      float(df["qty"].sum()),
                "kyle_lambda":       kyle_lambda(df, bin_seconds=60),
                "round_trip_rate":   round_trip_rate(df, window_seconds=60, qty_tolerance=0.01),
                "top10_size_share":  top_n_size_share(df, n=10),
                "buy_sell_ratio":    buy_sell_ratio(df),
            }
        )

    table = pd.DataFrame(rows).set_index("venue")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV)

    pd.set_option("display.float_format", lambda x: f"{x:.6g}")
    print()
    print(table.to_string())
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
