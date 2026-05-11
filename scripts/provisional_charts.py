"""Produce a battery of provisional charts to build intuition.

These charts are exploratory, not formal report deliverables. They sit
across the challenge dataset alone, plus a quick cross-venue overlay
where it's cheap. Outputs land in `notebooks/figures/provisional/`.

Usage:
    uv run python scripts/provisional_charts.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from challenge.analysis.manipulation import inter_arrival_seconds
from challenge.analysis.plot import (
    apply_default_style,
    plot_book_depth_profile,
    plot_cumulative_volume_by_side,
    plot_imbalance_timeseries,
    plot_inter_arrival_distribution,
    plot_per_hour_volume_breakdown,
    plot_price_trace_with_trades,
    plot_spread_timeseries,
    plot_trade_size_distribution,
)
from challenge.sources.csv_loader import load_orderbooks, load_trades, top_of_book

CHALLENGE_TRADES = Path("data/eth-btc-trades.csv")
CHALLENGE_BOOK = Path("data/eth-btc-orderbooks.csv")
KRAKEN_CACHE = Path("cache/kraken_xethxxbt_20250901_20250904.parquet")
BINANCE_CACHE = Path("cache/binance_ethbtc_20250901_20250904.parquet")
OUT = Path("notebooks/figures/provisional")


def _load_cached(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def main() -> int:
    apply_default_style()
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"loading challenge data: {CHALLENGE_TRADES}")
    challenge = load_trades(CHALLENGE_TRADES)
    print(f"  {len(challenge):,} trades, {challenge['ts'].min()} -> {challenge['ts'].max()}")

    print(f"loading challenge book:  {CHALLENGE_BOOK}")
    long_book = load_orderbooks(CHALLENGE_BOOK)
    top = top_of_book(long_book)
    print(f"  {long_book['ts'].nunique()} snapshots ({len(long_book)} rows)")

    kraken = _load_cached(KRAKEN_CACHE)
    binance = _load_cached(BINANCE_CACHE)
    print(f"comparison venues: kraken={'OK' if kraken is not None else 'MISSING'}, "
          f"binance={'OK' if binance is not None else 'MISSING'}")

    # ---- challenge-only exploratory charts ---------------------------------

    saved: list[Path] = []

    fig = plot_price_trace_with_trades(
        challenge, title="Challenge venue price trace with buy/sell overlay (2025-09-01..2025-09-04)"
    )
    saved.append(_save(fig, OUT / "01_price_trace_with_trades.png"))

    fig = plot_cumulative_volume_by_side(
        challenge, title="Challenge venue: cumulative buy vs sell volume"
    )
    saved.append(_save(fig, OUT / "02_cumulative_volume_by_side.png"))

    fig = plot_per_hour_volume_breakdown(
        challenge, title="Challenge venue: per-hour volume by side (log y)", log_y=True,
    )
    saved.append(_save(fig, OUT / "03_per_hour_volume.png"))

    fig = plot_spread_timeseries(
        top, title="Challenge venue: top-of-book spread (bps) over time"
    )
    saved.append(_save(fig, OUT / "04_spread_timeseries.png"))

    fig = plot_imbalance_timeseries(
        top, title="Challenge venue: top-of-book imbalance (bid_qty / total) over time"
    )
    saved.append(_save(fig, OUT / "05_imbalance_timeseries.png"))

    fig = plot_book_depth_profile(
        long_book,
        title="Challenge venue: cumulative depth profile (median snapshot)",
    )
    saved.append(_save(fig, OUT / "06_depth_profile_median_snapshot.png"))

    # ---- provisional cross-venue overlays (uncalibrated, just for shape) ----

    if kraken is not None and binance is not None:
        all_trades = {"challenge": challenge, "kraken": kraken, "binance": binance}
        fig = plot_trade_size_distribution(
            all_trades, title="Trade-size distribution by venue (provisional)"
        )
        saved.append(_save(fig, OUT / "07_trade_size_distribution.png"))

        ia_by_venue = {name: inter_arrival_seconds(df) for name, df in all_trades.items()}
        fig = plot_inter_arrival_distribution(
            ia_by_venue, title="Inter-arrival distribution by venue (provisional)"
        )
        saved.append(_save(fig, OUT / "08_inter_arrival_distribution.png"))

        fig = plot_inter_arrival_distribution(
            ia_by_venue,
            title="Inter-arrival distribution by venue — per-venue panels (provisional)",
            layout="subplots",
        )
        saved.append(_save(fig, OUT / "08b_inter_arrival_per_venue.png"))

    print(f"\nwrote {len(saved)} chart(s) to {OUT}/")
    for p in saved:
        print(f"  {p}")
    return 0


def _save(fig, path: Path) -> Path:
    fig.savefig(path)
    fig.clear()
    return path


if __name__ == "__main__":
    import sys

    sys.exit(main())
