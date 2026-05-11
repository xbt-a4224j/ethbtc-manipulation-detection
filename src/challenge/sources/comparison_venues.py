"""
Comparison-venue adapters: Kraken and Binance public REST.

Both venues are pulled over the *same UTC window* as the supplied challenge
data so every quantitative claim in the report can be calibrated against
named legitimate venues. Mismatched windows invalidate the comparison —
see CLAUDE.md "Validation moves" and primer module 09.

Both endpoints are unauthenticated for the trade-history calls used here.
Rate limits are real; cache aggressively via `io.cache.cached_parquet`.

Each function returns a DataFrame normalized to the canonical trades schema:

    ts:    pd.Timestamp (UTC, tz-aware)
    price: float
    qty:   float
    side:  str ("buy" or "sell")

See primer module 05 for endpoint detail, side-encoding gotchas, and
pagination semantics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Final

import httpx
import pandas as pd

KRAKEN_BASE: Final[str] = "https://api.kraken.com/0/public"
KRAKEN_PAIR: Final[str] = "XETHXXBT"

BINANCE_BASE: Final[str] = "https://data-api.binance.vision"
# data-api.binance.vision is Binance's read-only public-data subdomain. Same
# /api/v3/* schema as api.binance.com but reachable from US IPs (api.binance.com
# returns HTTP 451 to US clients). Swap to api.binance.com if running from
# elsewhere and needing the production endpoint.
BINANCE_SYMBOL: Final[str] = "ETHBTC"
BINANCE_CHUNK_MS: Final[int] = 60 * 60 * 1000  # 1 hour — endpoint maximum window


def _to_utc_ts(dt: datetime) -> pd.Timestamp:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return pd.Timestamp(dt).tz_convert("UTC")


def fetch_kraken_eth_btc(
    *,
    start: datetime,
    end: datetime,
    client: httpx.Client | None = None,
    max_pages: int = 10_000,
) -> pd.DataFrame:
    """Pull Kraken ETH/BTC public trades across [start, end). UTC, normalized.

    Endpoint: GET https://api.kraken.com/0/public/Trades
    Pair: XETHXXBT
    Pagination: nanosecond `since` cursor returned in `result["last"]`.
    Side mapping: 'b' -> 'buy', 's' -> 'sell' (taker side).

    Returns a DataFrame with the canonical trades schema, sorted by ts,
    trimmed to `[start, end)`. The `max_pages` argument is a defensive
    cap to avoid infinite loops on a misbehaving cursor.
    """
    owned_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        start_ts = _to_utc_ts(start)
        end_ts = _to_utc_ts(end)
        since_ns = int(start_ts.timestamp() * 1_000_000_000)
        end_seconds = end_ts.timestamp()
        rows: list[list] = []

        for _ in range(max_pages):
            r = client.get(
                f"{KRAKEN_BASE}/Trades",
                params={"pair": KRAKEN_PAIR, "since": str(since_ns)},
            )
            r.raise_for_status()
            body = r.json()
            if body.get("error"):
                raise RuntimeError(f"kraken error: {body['error']}")

            result = body.get("result", {})
            trades = result.get(KRAKEN_PAIR, [])
            if not trades:
                break
            rows.extend(trades)

            new_since = int(result["last"])
            # No forward progress: stop, otherwise we infinite-loop.
            if new_since <= since_ns:
                break
            since_ns = new_since

            latest_ts = trades[-1][2]
            if latest_ts >= end_seconds:
                break
        else:
            raise RuntimeError(
                f"kraken pagination exceeded max_pages={max_pages}; widen or fix the loop"
            )

        if not rows:
            return _empty_canonical_trades()

        df = pd.DataFrame(
            rows,
            columns=["price", "qty", "ts_s", "side_raw", "ordertype", "misc", "trade_id"],
        )
        df["ts"] = pd.to_datetime(df["ts_s"].astype(float), unit="s", utc=True)
        df["price"] = df["price"].astype(float)
        df["qty"] = df["qty"].astype(float)
        side_norm = df["side_raw"].map({"b": "buy", "s": "sell"})
        if side_norm.isna().any():
            bad = df["side_raw"][side_norm.isna()].unique().tolist()
            raise RuntimeError(f"unrecognized kraken side values: {bad}")
        df["side"] = side_norm

        df = df.loc[(df["ts"] >= start_ts) & (df["ts"] < end_ts)]
        return df.loc[:, ["ts", "price", "qty", "side"]].sort_values("ts").reset_index(drop=True)
    finally:
        if owned_client:
            client.close()


def fetch_binance_eth_btc(
    *,
    start: datetime,
    end: datetime,
    client: httpx.Client | None = None,
    max_iterations: int = 100_000,
) -> pd.DataFrame:
    """Pull Binance ETHBTC aggregated trades across [start, end). UTC, normalized.

    Endpoint: GET https://api.binance.com/api/v3/aggTrades
    Symbol: ETHBTC
    Window params: startTime/endTime in ms; max 1-hour span; max 1000 trades.
    Side mapping: `m` (isBuyerMaker) True -> taker is seller -> 'sell'.

    Walks the time window in 1-hour chunks. When a chunk hits the 1000-trade
    ceiling, advances cursor to one ms past the last trade and re-pulls.
    """
    owned_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        start_ts = _to_utc_ts(start)
        end_ts = _to_utc_ts(end)
        cursor_ms = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000)

        all_rows: list[dict] = []
        iterations = 0
        while cursor_ms < end_ms:
            iterations += 1
            if iterations > max_iterations:
                raise RuntimeError(
                    f"binance pagination exceeded max_iterations={max_iterations}"
                )

            chunk_end = min(cursor_ms + BINANCE_CHUNK_MS - 1, end_ms - 1)
            r = client.get(
                f"{BINANCE_BASE}/api/v3/aggTrades",
                params={
                    "symbol": BINANCE_SYMBOL,
                    "startTime": cursor_ms,
                    "endTime": chunk_end,
                    "limit": 1000,
                },
            )
            r.raise_for_status()
            chunk = r.json()
            if not isinstance(chunk, list):
                raise RuntimeError(f"binance unexpected response shape: {type(chunk).__name__}")

            if not chunk:
                cursor_ms = chunk_end + 1
                continue

            all_rows.extend(chunk)
            if len(chunk) >= 1000:
                # Window had more than the limit; re-pull from one past the last trade.
                cursor_ms = int(chunk[-1]["T"]) + 1
            else:
                cursor_ms = chunk_end + 1

        if not all_rows:
            return _empty_canonical_trades()

        df = pd.DataFrame(all_rows)
        df["ts"] = pd.to_datetime(df["T"].astype("int64"), unit="ms", utc=True)
        df["price"] = df["p"].astype(float)
        df["qty"] = df["q"].astype(float)
        df["side"] = df["m"].map({True: "sell", False: "buy"})
        if df["side"].isna().any():
            raise RuntimeError("binance returned a row with non-boolean isBuyerMaker")

        df = df.loc[(df["ts"] >= start_ts) & (df["ts"] < end_ts)]
        return df.loc[:, ["ts", "price", "qty", "side"]].sort_values("ts").reset_index(drop=True)
    finally:
        if owned_client:
            client.close()


def _empty_canonical_trades() -> pd.DataFrame:
    """Empty DataFrame with the canonical trades schema and correct dtypes."""
    return pd.DataFrame(
        {
            "ts": pd.Series(dtype="datetime64[ns, UTC]"),
            "price": pd.Series(dtype=float),
            "qty": pd.Series(dtype=float),
            "side": pd.Series(dtype=object),
        }
    )
