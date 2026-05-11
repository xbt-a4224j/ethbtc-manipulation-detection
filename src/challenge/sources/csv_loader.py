"""
Loaders for the supplied challenge CSVs.

Both functions normalize the raw CSV columns into the canonical schemas
used by the analysis layer.

Canonical schemas:

    Trades:
        ts:    pd.Timestamp (UTC, tz-aware)
        price: float
        qty:   float (positive)
        side:  str ("buy" or "sell")

    Orderbooks:
        ts:    pd.Timestamp (UTC, tz-aware)
        side:  str ("bid" or "ask")
        price: float
        qty:   float
        level: int (0-indexed; 0 = top of book)

The column-detection logic is best-effort against typical CEX dump
conventions. Inspect the actual CSV first; if the column names don't
match what's detected, supply explicit overrides via the `column_map`
arguments.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

_TRADES_TS_CANDIDATES: tuple[str, ...] = (
    "ts", "timestamp", "time", "datetime", "date", "exchange_time",
)
_TRADES_PRICE_CANDIDATES: tuple[str, ...] = ("price", "px", "rate", "p")
_TRADES_QTY_CANDIDATES: tuple[str, ...] = (
    "qty", "quantity", "size", "amount", "volume", "q",
)
_TRADES_SIDE_CANDIDATES: tuple[str, ...] = (
    "side", "type", "direction", "aggressor", "taker_side",
)

_SIDE_NORMALIZATION: dict[str, str] = {
    "buy": "buy", "b": "buy", "bid": "buy", "true": "buy", "1": "buy",
    "sell": "sell", "s": "sell", "ask": "sell", "false": "sell", "0": "sell",
    "long": "buy", "short": "sell",
}


def _detect_column(columns: list[str], candidates: tuple[str, ...], purpose: str) -> str:
    lower_to_actual = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_to_actual:
            return lower_to_actual[cand]
    raise ValueError(
        f"could not detect a {purpose} column; tried {list(candidates)} "
        f"against {sorted(columns)}"
    )


def load_trades(
    path: Path | str,
    *,
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load a trades CSV and return the canonical trades DataFrame.

    Args:
        path: path to the CSV file.
        column_map: optional explicit override for column detection. Keys are
            the canonical names ("ts", "price", "qty", "side"); values are
            the raw column names in the file. Anything omitted from the map
            falls back to auto-detection.

    Validates after loading: side values normalize cleanly, no negative
    quantities, timestamps parse to UTC. Raises loud on any violation.
    """
    raw = pd.read_csv(path)
    overrides = column_map or {}

    ts_col = overrides.get("ts") or _detect_column(list(raw.columns), _TRADES_TS_CANDIDATES, "timestamp")
    price_col = overrides.get("price") or _detect_column(list(raw.columns), _TRADES_PRICE_CANDIDATES, "price")
    qty_col = overrides.get("qty") or _detect_column(list(raw.columns), _TRADES_QTY_CANDIDATES, "quantity")
    side_col = overrides.get("side") or _detect_column(list(raw.columns), _TRADES_SIDE_CANDIDATES, "side")

    ts_parsed = pd.to_datetime(raw[ts_col], utc=True, errors="coerce")
    if ts_parsed.isna().any():
        bad_count = int(ts_parsed.isna().sum())
        raise ValueError(f"{bad_count} timestamp(s) failed to parse in column {ts_col!r}")

    side_norm = raw[side_col].astype(str).str.lower().str.strip().map(_SIDE_NORMALIZATION)
    if side_norm.isna().any():
        bad = raw[side_col][side_norm.isna()].astype(str).unique().tolist()
        raise ValueError(f"unrecognized side values in column {side_col!r}: {bad[:5]}")

    qty_parsed = raw[qty_col].astype(float)
    if (qty_parsed < 0).any():
        bad_count = int((qty_parsed < 0).sum())
        raise ValueError(f"{bad_count} negative quantit(ies) in column {qty_col!r}")

    df = pd.DataFrame(
        {
            "ts": ts_parsed,
            "price": raw[price_col].astype(float),
            "qty": qty_parsed,
            "side": side_norm,
        }
    )
    return df.sort_values("ts").reset_index(drop=True)


def load_orderbooks(
    path: Path | str,
    *,
    ts_column: str = "timestamp",
    asks_column: str = "asks",
    bids_column: str = "bids",
) -> pd.DataFrame:
    """Load the supplied orderbook CSV and reshape into long canonical form.

    Source format:
        timestamp, asks, bids

    where `asks` and `bids` each hold a Python-literal string encoding
    a list of `{'price': float, 'size': float}` dicts. Each snapshot
    typically has up to 50 levels per side.

    Levels are normalized to 0-indexed where 0 is the top of book — the
    *lowest*-priced ask and the *highest*-priced bid. The function
    re-sorts within each snapshot so a misordered source still produces
    a correctly-indexed long table.

    Returns a DataFrame with columns: ts, side, level, price, qty.
    """
    raw = pd.read_csv(path)
    for needed in (ts_column, asks_column, bids_column):
        if needed not in raw.columns:
            raise ValueError(
                f"column {needed!r} not found; have {sorted(raw.columns)}"
            )

    ts_parsed = pd.to_datetime(raw[ts_column], utc=True, errors="coerce")
    if ts_parsed.isna().any():
        bad_count = int(ts_parsed.isna().sum())
        raise ValueError(
            f"{bad_count} timestamp(s) failed to parse in column {ts_column!r}"
        )

    rows: list[dict] = []
    for snap_idx in range(len(raw)):
        ts = ts_parsed.iloc[snap_idx]
        asks_raw = raw[asks_column].iloc[snap_idx]
        bids_raw = raw[bids_column].iloc[snap_idx]
        try:
            asks = ast.literal_eval(asks_raw) if isinstance(asks_raw, str) else asks_raw
            bids = ast.literal_eval(bids_raw) if isinstance(bids_raw, str) else bids_raw
        except (ValueError, SyntaxError) as e:
            raise ValueError(
                f"failed to parse book at snapshot {snap_idx} (ts={ts}): {e}"
            ) from e

        if not isinstance(asks, list) or not isinstance(bids, list):
            raise ValueError(
                f"snapshot {snap_idx} (ts={ts}) does not encode list-of-dicts on both sides"
            )

        # Normalize: asks ascending, bids descending. Level 0 = top of book.
        asks_sorted = sorted(asks, key=lambda x: float(x["price"]))
        bids_sorted = sorted(bids, key=lambda x: -float(x["price"]))

        for level, entry in enumerate(asks_sorted):
            rows.append(
                {
                    "ts": ts,
                    "side": "ask",
                    "level": level,
                    "price": float(entry["price"]),
                    "qty": float(entry["size"]),
                }
            )
        for level, entry in enumerate(bids_sorted):
            rows.append(
                {
                    "ts": ts,
                    "side": "bid",
                    "level": level,
                    "price": float(entry["price"]),
                    "qty": float(entry["size"]),
                }
            )

    if not rows:
        return pd.DataFrame(
            {
                "ts": pd.Series(dtype="datetime64[ns, UTC]"),
                "side": pd.Series(dtype=object),
                "level": pd.Series(dtype=int),
                "price": pd.Series(dtype=float),
                "qty": pd.Series(dtype=float),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["ts", "side", "level"])
        .reset_index(drop=True)
    )


def top_of_book(long_book: pd.DataFrame) -> pd.DataFrame:
    """Build a per-snapshot top-of-book DataFrame from the long-format book.

    Returns a DataFrame indexed by snapshot timestamp with columns:
        bid_price, ask_price, bid_qty, ask_qty, mid, spread_bps, imbalance.

    The mid/spread/imbalance computations live here so every microstructure
    metric in the report shares one consistent definition.
    """
    required = {"ts", "side", "level", "price", "qty"}
    missing = required - set(long_book.columns)
    if missing:
        raise ValueError(f"long_book missing columns {sorted(missing)}")

    top = (
        long_book.loc[long_book["level"] == 0]
        .pivot(index="ts", columns="side", values=["price", "qty"])
    )
    if top.empty:
        return pd.DataFrame(
            columns=["bid_price", "ask_price", "bid_qty", "ask_qty", "mid", "spread_bps", "imbalance"]
        )

    top.columns = [f"{side}_{field}" for field, side in top.columns]
    expected = {"bid_price", "ask_price", "bid_qty", "ask_qty"}
    missing_after = expected - set(top.columns)
    if missing_after:
        raise ValueError(
            f"long_book level-0 rows missing one side of book; missing columns after pivot: {sorted(missing_after)}"
        )

    top["mid"] = (top["bid_price"] + top["ask_price"]) / 2
    top["spread_bps"] = (top["ask_price"] - top["bid_price"]) / top["mid"] * 10_000
    denom = top["bid_qty"] + top["ask_qty"]
    top["imbalance"] = (top["bid_qty"] / denom).where(denom > 0)
    return top
