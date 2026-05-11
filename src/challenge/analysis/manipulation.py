"""
Manipulation-detection primitives.

Pure, vectorized, testable. Each function takes a normalized trades DataFrame
with the canonical schema:

    ts:    pd.Timestamp (UTC, tz-aware)
    price: float
    qty:   float (positive)
    side:  str   ("buy" or "sell")

The detection helpers return either a scalar metric or a per-bin Series.
Loud failures on schema mismatch — silent corruption is the worst outcome here.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

CANONICAL_COLS: Final[frozenset[str]] = frozenset({"ts", "price", "qty", "side"})


def _validate_schema(trades: pd.DataFrame) -> None:
    missing = CANONICAL_COLS - set(trades.columns)
    if missing:
        raise ValueError(
            f"trades DataFrame missing columns {sorted(missing)}; "
            f"got {sorted(trades.columns)}"
        )


def kyle_lambda(
    trades: pd.DataFrame,
    *,
    bin_seconds: int = 60,
) -> float:
    """Estimate Kyle's lambda — the OLS slope of price change on signed flow.

    Bins trades by `bin_seconds`, computes per-bin signed volume
    (sum(qty) for buys minus sum(qty) for sells) and per-bin price change
    (last price minus first price), then regresses dP on signed_volume.
    The slope is lambda — price impact per unit of signed flow.

    Returns 0.0 when signed volume has no variance (e.g. perfectly balanced
    flow with no informational content). A healthy market with directional
    flow will produce a positive, non-trivial lambda; wash-like flow
    collapses toward zero.

    See primer module 07 for the derivation and interpretation.
    """
    _validate_schema(trades)
    if len(trades) == 0:
        return 0.0

    df = trades.loc[:, ["ts", "price", "qty", "side"]].copy()
    df["bin"] = df["ts"].dt.floor(f"{bin_seconds}s")

    sign = df["side"].str.lower().map({"buy": 1, "sell": -1})
    if sign.isna().any():
        bad = df.loc[sign.isna(), "side"].unique().tolist()
        raise ValueError(f"side column must contain only 'buy' or 'sell'; saw {bad}")
    df["signed_qty"] = df["qty"].astype(float) * sign

    by_bin = df.groupby("bin").agg(
        signed_volume=("signed_qty", "sum"),
        first_price=("price", "first"),
        last_price=("price", "last"),
    )
    by_bin["dP"] = by_bin["last_price"] - by_bin["first_price"]

    x = by_bin["signed_volume"].to_numpy(dtype=float)
    y = by_bin["dP"].to_numpy(dtype=float)

    if x.size < 2 or float(np.var(x)) < 1e-12:
        return 0.0

    cov_xy = float(np.cov(x, y, bias=True)[0, 1])
    var_x = float(np.var(x))
    return cov_xy / var_x


def round_trip_rate(
    trades: pd.DataFrame,
    *,
    window_seconds: int = 60,
    qty_tolerance: float = 0.01,
) -> float:
    """Fraction of executed volume that round-trips within `window_seconds`.

    Greedy nearest-time pairing: walk trades chronologically; for each
    unpaired trade, find the next opposite-side trade with quantity within
    `qty_tolerance` (relative) and timestamp within the window. Mark both
    paired. Return paired_volume / total_volume.

    The greedy pairing is a deliberate simplification (see primer 02);
    optimal pairing would be combinatorial and isn't worth the cost.
    """
    _validate_schema(trades)
    if qty_tolerance < 0:
        raise ValueError(f"qty_tolerance must be >= 0; got {qty_tolerance}")
    if window_seconds <= 0:
        raise ValueError(f"window_seconds must be > 0; got {window_seconds}")

    n = len(trades)
    if n < 2:
        return 0.0

    df = trades.sort_values("ts").reset_index(drop=True)
    sides = df["side"].str.lower().to_numpy()
    qtys = df["qty"].astype(float).to_numpy()
    ts = df["ts"].to_numpy()
    paired = np.zeros(n, dtype=bool)
    deadline_delta = np.timedelta64(window_seconds, "s")

    paired_qty = 0.0
    for i in range(n):
        if paired[i]:
            continue
        if qtys[i] <= 0:
            continue
        target_side = "sell" if sides[i] == "buy" else "buy"
        deadline = ts[i] + deadline_delta
        for j in range(i + 1, n):
            if ts[j] > deadline:
                break
            if paired[j]:
                continue
            if sides[j] != target_side:
                continue
            if abs(qtys[j] - qtys[i]) / qtys[i] > qty_tolerance:
                continue
            paired[i] = True
            paired[j] = True
            paired_qty += qtys[i] + qtys[j]
            break

    total_qty = float(qtys.sum())
    return paired_qty / total_qty if total_qty > 0 else 0.0


def top_n_size_share(
    trades: pd.DataFrame,
    *,
    n: int = 10,
    rounding_decimals: int = 6,
) -> float:
    """Share of total volume concentrated in the top-n distinct trade sizes.

    Sizes are rounded to `rounding_decimals` to collapse floating-point
    noise (some venues report sizes with 8-decimal precision but the bot
    is using 4). Wash bots ship from a small fixed set; legitimate flow
    is more dispersed.
    """
    _validate_schema(trades)
    if n <= 0:
        raise ValueError(f"n must be > 0; got {n}")

    sizes = trades["qty"].astype(float).round(rounding_decimals)
    total = float(sizes.sum())
    if total <= 0:
        return 0.0
    by_size = sizes.groupby(sizes).sum().sort_values(ascending=False)
    return float(by_size.head(n).sum() / total)


def inter_arrival_seconds(trades: pd.DataFrame) -> pd.Series:
    """Per-trade inter-arrival times in seconds.

    Returns a Series of length len(trades) - 1, with the leading NaN from
    .diff() dropped. Useful directly for plotting; aggregate via
    .quantile() for headline summary stats.

    Sorts defensively — raw venue dumps occasionally arrive out of order
    after pagination merges.
    """
    _validate_schema(trades)
    if len(trades) < 2:
        return pd.Series([], name="inter_arrival_s", dtype=float)

    sorted_ts = trades["ts"].sort_values().reset_index(drop=True)
    deltas = sorted_ts.diff().dt.total_seconds()
    return deltas.iloc[1:].rename("inter_arrival_s").reset_index(drop=True)


def buy_sell_ratio(trades: pd.DataFrame) -> float:
    """Buy-side volume share across the supplied window.

    Returns a float in [0, 1]. 0.5 = balanced. Useful as a sanity check
    against Kyle's lambda: balanced flow with collapsed lambda is
    suggestive; balanced flow with healthy lambda is normal market-making.

    Returns 0.5 (the neutral value) when total volume is zero — this
    avoids a zero-divide and is the right semantic default for an empty
    window.
    """
    _validate_schema(trades)
    if len(trades) == 0:
        return 0.5
    qty = trades["qty"].astype(float)
    total_volume = float(qty.sum())
    if total_volume <= 0:
        return 0.5
    side = trades["side"].astype(str).str.lower()
    buy_volume = float(qty[side == "buy"].sum())
    return buy_volume / total_volume
