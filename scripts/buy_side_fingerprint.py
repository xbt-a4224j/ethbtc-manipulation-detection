"""Investigate the buy-side anomaly in the challenge dataset.

The challenge venue shows 720 buys vs 125 sells with ~99.99% buy-volume
share. This script characterizes the buys by:

  - rounded size concentration (do the same handful of sizes recur?)
  - exact-size concentration (do trades repeat byte-for-byte?)
  - per-side qty distribution (is the sell side a different population?)
  - clock-fingerprint (do timestamps cluster on suspicious modular
    boundaries — second-of-minute, minute-of-hour, etc.)
  - same-second flurries and price-stamp recurrence (a venue without
    trade IDs leaks "this came from one bot" via timing + price stamp)

Each section prints a small, diff-friendly summary. The full numbers
will be re-derived in the notebook; this script's job is to inform the
hypothesis before the investigator's voice version is written up.

Usage:
    uv run python scripts/buy_side_fingerprint.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from challenge.analysis.plot import SIDE_COLORS, apply_default_style
from challenge.sources.csv_loader import load_trades

CHALLENGE_TRADES = Path("data/eth-btc-trades.csv")
KRAKEN_CACHE = Path("cache/kraken_xethxxbt_20250901_20250904.parquet")
BINANCE_CACHE = Path("cache/binance_ethbtc_20250901_20250904.parquet")
OUT_DIR = Path("notebooks/figures/provisional")


def _section(title: str) -> None:
    bar = "-" * len(title)
    print(f"\n{title}\n{bar}")


def _clip(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df.loc[(df["ts"] >= start) & (df["ts"] <= end)].reset_index(drop=True)


def main() -> int:
    challenge = load_trades(CHALLENGE_TRADES)
    buys = challenge.loc[challenge["side"] == "buy"].reset_index(drop=True)
    sells = challenge.loc[challenge["side"] == "sell"].reset_index(drop=True)

    _section("Counts and volume by side")
    print(f"buys:  n={len(buys):>5,}  total_qty={buys['qty'].sum():>14,.4f}  "
          f"mean={buys['qty'].mean():>10,.4f}  median={buys['qty'].median():>10,.4f}")
    print(f"sells: n={len(sells):>5,}  total_qty={sells['qty'].sum():>14,.4f}  "
          f"mean={sells['qty'].mean():>10,.4f}  median={sells['qty'].median():>10,.4f}")
    ratio_n = len(buys) / max(len(sells), 1)
    ratio_v = buys["qty"].sum() / max(sells["qty"].sum(), 1e-12)
    print(f"buy/sell count ratio:  {ratio_n:.2f}x")
    print(f"buy/sell volume ratio: {ratio_v:,.0f}x")

    _section("Top exact (raw) buy sizes")
    exact = buys["qty"].value_counts().head(15)
    total_buy_qty = buys["qty"].sum()
    cum_pct = 0.0
    for size, count in exact.items():
        share = (size * count) / total_buy_qty
        cum_pct += share
        print(f"  qty={size:>14,.6f}  n={count:>4d}  vol_share={share:6.2%}  cum={cum_pct:6.2%}")
    print(f"top 15 exact sizes account for {len(exact.index.unique())} distinct values, "
          f"covering {cum_pct:.1%} of buy volume")

    _section("Distinct exact buy sizes (full count)")
    distinct_exact = buys["qty"].nunique()
    print(f"buys: {distinct_exact} distinct exact sizes across {len(buys)} trades "
          f"({distinct_exact/len(buys):.1%} unique-rate)")
    distinct_sells = sells["qty"].nunique()
    print(f"sells: {distinct_sells} distinct exact sizes across {len(sells)} trades "
          f"({distinct_sells/max(len(sells),1):.1%} unique-rate)")

    _section("Top-N rounded buy-size concentration (qty share)")
    for nd in (4, 6, 8):
        rounded = buys["qty"].round(nd)
        total = rounded.sum()
        for n in (5, 10, 25):
            top = rounded.groupby(rounded).sum().sort_values(ascending=False).head(n).sum()
            print(f"  decimals={nd}  top-{n:>2d} sizes share = {top/total:6.2%}")

    _section("Round-number / fixed-grid signature")
    is_round_4dp = buys["qty"].apply(lambda x: float(round(x, 4)) == float(x))
    is_round_2dp = buys["qty"].apply(lambda x: float(round(x, 2)) == float(x))
    is_round_int = buys["qty"].apply(lambda x: float(round(x, 0)) == float(x))
    print(f"buys with qty exactly representable at 4dp: {is_round_4dp.mean():.1%}")
    print(f"buys with qty exactly representable at 2dp: {is_round_2dp.mean():.1%}")
    print(f"buys with integer qty:                       {is_round_int.mean():.1%}")

    _section("Are buy sizes drawn from a different population than sells?")
    buy_q = buys["qty"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
    sell_q = sells["qty"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
    print("            p05            p25            p50            p75            p95")
    print(f"buys:   "
          f"{buy_q[0.05]:>12,.4f}   {buy_q[0.25]:>12,.4f}   "
          f"{buy_q[0.5]:>12,.4f}   {buy_q[0.75]:>12,.4f}   {buy_q[0.95]:>12,.4f}")
    print(f"sells:  "
          f"{sell_q[0.05]:>12,.4f}   {sell_q[0.25]:>12,.4f}   "
          f"{sell_q[0.5]:>12,.4f}   {sell_q[0.75]:>12,.4f}   {sell_q[0.95]:>12,.4f}")

    _section("Repeated (timestamp, price, qty, side) tuples on the buy side")
    # Without trade IDs, identical (ts, price, qty) tuples on the same side
    # are a strong "same source" signal — venues do not hand out the same
    # nanosecond stamp to two unrelated trades.
    buy_tuples = buys.groupby(["ts", "price", "qty"]).size().sort_values(ascending=False)
    n_groups = len(buy_tuples)
    n_dup_groups = int((buy_tuples > 1).sum())
    n_dup_rows = int(buy_tuples[buy_tuples > 1].sum())
    print(f"distinct (ts, price, qty) groups on the buy side: {n_groups:,}")
    print(f"  groups with 2+ identical rows:   {n_dup_groups}")
    print(f"  rows in those groups:            {n_dup_rows}")
    print("top 10 most-duplicated (ts, price, qty) groups:")
    for (ts, px, q), cnt in buy_tuples.head(10).items():
        if cnt < 2:
            break
        print(f"  ts={ts}  price={px}  qty={q:.6f}  n={cnt}")

    _section("Same-second buy clusters (trades within the same wall-clock second)")
    buy_sec = buys.assign(sec=buys["ts"].dt.floor("s")).groupby("sec").size()
    print(f"distinct buy-bearing seconds:    {len(buy_sec):,}")
    print(f"  seconds with 2+ buys:          {int((buy_sec >= 2).sum())}")
    print(f"  seconds with 5+ buys:          {int((buy_sec >= 5).sum())}")
    print(f"  max buys in one second:        {int(buy_sec.max())}")

    _section("Clock-modular fingerprint (do buys cluster on grid boundaries?)")
    # Bots often fire on a fixed cadence — every 10s, every minute on the
    # second, etc. These show up as a spike at second-of-minute == 0 (or
    # any other narrow modular value).
    sec_of_min = buys["ts"].dt.second.value_counts().sort_index()
    top_sec = sec_of_min.sort_values(ascending=False).head(5)
    print("top 5 second-of-minute values for buys (out of 60):")
    for s, n in top_sec.items():
        print(f"  second={int(s):>2d}  n={int(n):>4d}  share={n/len(buys):6.2%}")
    expected = len(buys) / 60.0
    spike_ratio = float(top_sec.iloc[0]) / expected
    print(f"uniform expectation per second-bin: {expected:.2f}  "
          f"(observed top bin is {spike_ratio:.2f}x uniform)")

    _section("Cross-venue check: do these exact buy sizes appear on Kraken / Binance?")
    kraken = pd.read_parquet(KRAKEN_CACHE)
    binance = pd.read_parquet(BINANCE_CACHE)
    start, end = challenge["ts"].min(), challenge["ts"].max()
    kraken_w = _clip(kraken, start, end)
    binance_w = _clip(binance, start, end)
    top_buy_sizes = exact.head(5).index.tolist()
    print("for each top buy size, count of *exact* matches on each venue:")
    for size in top_buy_sizes:
        ch_n = int((buys["qty"] == size).sum())
        kr_n = int((kraken_w["qty"] == size).sum())
        bn_n = int((binance_w["qty"] == size).sum())
        print(f"  qty={size:>14,.6f}   challenge={ch_n:>4d}   kraken={kr_n:>4d}   binance={bn_n:>4d}")

    _section("Hourly buy-flow regularity")
    # If a bot is dripping at a fixed cadence, hourly buy counts should be
    # remarkably flat compared to legitimate venues.
    hourly = buys.set_index("ts").resample("1h").size()
    hourly_v = buys.set_index("ts")["qty"].resample("1h").sum()
    print(f"hours covered:                    {len(hourly)}")
    print(f"hourly buy-count mean / std:      "
          f"{hourly.mean():.2f} / {hourly.std():.2f}  (cv={hourly.std()/hourly.mean():.3f})")
    print(f"hourly buy-volume mean / std:     "
          f"{hourly_v.mean():.4f} / {hourly_v.std():.4f}  (cv={hourly_v.std()/hourly_v.mean():.3f})")
    # Compare against challenge sells and against legitimate venues
    sell_hour = sells.set_index("ts").resample("1h").size()
    if len(sell_hour) and sell_hour.mean() > 0:
        print(f"sells: hourly count mean/std:    "
              f"{sell_hour.mean():.2f} / {sell_hour.std():.2f}  "
              f"(cv={sell_hour.std()/sell_hour.mean():.3f})")
    kr_hourly = kraken_w.set_index("ts").resample("1h").size()
    bn_hourly = binance_w.set_index("ts").resample("1h").size()
    print(f"kraken: hourly trade-count cv:    {kr_hourly.std()/kr_hourly.mean():.3f}")
    print(f"binance: hourly trade-count cv:   {bn_hourly.std()/bn_hourly.mean():.3f}")

    # Save a per-size buy table for the notebook to import directly.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_path = OUT_DIR / "buy_side_size_table.csv"
    rounded6 = buys["qty"].round(6)
    by_size = (
        rounded6.groupby(rounded6).agg(["count", "sum"])
        .rename(columns={"count": "n_trades", "sum": "total_qty"})
        .sort_values("total_qty", ascending=False)
    )
    by_size["vol_share"] = by_size["total_qty"] / by_size["total_qty"].sum()
    by_size.to_csv(table_path)
    print(f"\nwrote per-size buy table -> {table_path}  ({len(by_size)} distinct sizes)")

    # ------------------------------------------------------------------
    # Visual: per-side size distribution + hourly buy-flow regularity.
    # The two-panel layout is the cleanest way to show "buy and sell are
    # different populations" alongside "buy flow is suspiciously flat".
    # ------------------------------------------------------------------
    apply_default_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    pos_buy = buys["qty"][buys["qty"] > 0]
    pos_sell = sells["qty"][sells["qty"] > 0]
    edges = np.logspace(
        np.log10(min(pos_buy.min(), pos_sell.min(), 1e-6)),
        np.log10(max(pos_buy.max(), pos_sell.max())),
        50,
    )
    ax1.hist(
        pos_buy, bins=edges, histtype="stepfilled", color=SIDE_COLORS["buy"],
        alpha=0.35, edgecolor=SIDE_COLORS["buy"], linewidth=1.4,
        label=f"buy  (n={len(pos_buy)}, median={pos_buy.median():.3g})",
    )
    ax1.hist(
        pos_sell, bins=edges, histtype="stepfilled", color=SIDE_COLORS["sell"],
        alpha=0.35, edgecolor=SIDE_COLORS["sell"], linewidth=1.4,
        label=f"sell (n={len(pos_sell)}, median={pos_sell.median():.3g})",
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("trade size (base asset)")
    ax1.set_ylabel("count")
    ax1.set_title("Challenge venue: buy vs sell size distribution")
    ax1.legend(frameon=False)
    ax1.grid(True, which="both", alpha=0.3)

    hourly_buys = buys.set_index("ts").resample("1h").size()
    hourly_kraken = kraken_w.set_index("ts").resample("1h").size()
    hourly_binance = binance_w.set_index("ts").resample("1h").size()

    def _norm(s: pd.Series) -> pd.Series:
        m = s.mean()
        return s / m if m else s

    ax2.plot(_norm(hourly_buys).index, _norm(hourly_buys).values,
             color="#222222", linewidth=1.6,
             label=f"challenge buys (cv={hourly_buys.std()/hourly_buys.mean():.2f})")
    ax2.plot(_norm(hourly_kraken).index, _norm(hourly_kraken).values,
             color="#1f6feb", linewidth=1.0, alpha=0.7,
             label=f"kraken trades  (cv={hourly_kraken.std()/hourly_kraken.mean():.2f})")
    ax2.plot(_norm(hourly_binance).index, _norm(hourly_binance).values,
             color="#a33333", linewidth=1.0, alpha=0.7,
             label=f"binance trades (cv={hourly_binance.std()/hourly_binance.mean():.2f})")
    ax2.axhline(1.0, color="#888888", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("hourly count / mean")
    ax2.set_title("Hourly trade-count regularity (each venue normalized to its own mean)")
    ax2.legend(frameon=False)
    ax2.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig_path = OUT_DIR / "09_buy_side_fingerprint.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"wrote fingerprint chart  -> {fig_path}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
