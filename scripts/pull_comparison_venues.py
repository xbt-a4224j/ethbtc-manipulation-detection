"""Pull Kraken + Binance ETH/BTC trades over the challenge window.

The window is hardcoded to match the supplied dataset (2025-09-01 to
2025-09-04 UTC, with a one-hour pad on the end). Results are cached to
`cache/*.parquet` via `challenge.io.cache.cached_parquet`; subsequent runs
hit the cache and return immediately.

Usage:
    uv run python scripts/pull_comparison_venues.py            # both venues
    uv run python scripts/pull_comparison_venues.py kraken     # only Kraken
    uv run python scripts/pull_comparison_venues.py binance    # only Binance
    uv run python scripts/pull_comparison_venues.py --refresh  # ignore cache
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

import httpx

from challenge.io.cache import cached_parquet
from challenge.sources.comparison_venues import (
    fetch_binance_eth_btc,
    fetch_kraken_eth_btc,
)

START = datetime(2025, 9, 1, 0, 0, tzinfo=timezone.utc)
END = datetime(2025, 9, 4, 0, 0, tzinfo=timezone.utc)


def _pull_kraken(refresh: bool) -> None:
    name = f"kraken_xethxxbt_{START:%Y%m%d}_{END:%Y%m%d}"
    print(f"[kraken] pulling {START:%Y-%m-%d} -> {END:%Y-%m-%d} (cache: {name})")
    t0 = time.time()
    with httpx.Client(timeout=30.0) as client:
        df = cached_parquet(
            name,
            lambda: fetch_kraken_eth_btc(start=START, end=END, client=client),
            overwrite=refresh,
        )
    print(f"[kraken] {len(df):>7,} trades in {time.time() - t0:5.1f}s")
    if len(df) > 0:
        print(f"[kraken] window  : {df['ts'].min()} -> {df['ts'].max()}")
        print(f"[kraken] sides   : {dict(df['side'].value_counts())}")


def _pull_binance(refresh: bool) -> None:
    name = f"binance_ethbtc_{START:%Y%m%d}_{END:%Y%m%d}"
    print(f"[binance] pulling {START:%Y-%m-%d} -> {END:%Y-%m-%d} (cache: {name})")
    t0 = time.time()
    with httpx.Client(timeout=30.0) as client:
        df = cached_parquet(
            name,
            lambda: fetch_binance_eth_btc(start=START, end=END, client=client),
            overwrite=refresh,
        )
    print(f"[binance] {len(df):>7,} trades in {time.time() - t0:5.1f}s")
    if len(df) > 0:
        print(f"[binance] window : {df['ts'].min()} -> {df['ts'].max()}")
        print(f"[binance] sides  : {dict(df['side'].value_counts())}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "venue",
        nargs="?",
        choices=("kraken", "binance", "both"),
        default="both",
    )
    parser.add_argument("--refresh", action="store_true", help="ignore cache and re-pull")
    args = parser.parse_args(argv)

    if args.venue in ("kraken", "both"):
        _pull_kraken(args.refresh)
    if args.venue in ("binance", "both"):
        _pull_binance(args.refresh)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
