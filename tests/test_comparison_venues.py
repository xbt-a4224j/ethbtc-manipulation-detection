"""Tests for comparison-venue HTTP adapters.

All tests use httpx.MockTransport to avoid hitting real APIs. Each test
locks in a specific contract: pagination, side normalization, window
trimming, error surfacing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import httpx
import pandas as pd
import pytest

from challenge.sources.comparison_venues import (
    fetch_binance_eth_btc,
    fetch_kraken_eth_btc,
)


# ---------- Mock helpers ------------------------------------------------------


def _kraken_response(trades: list[list], last_cursor_ns: int) -> dict:
    """Build a well-formed Kraken Trades response payload."""
    return {
        "error": [],
        "result": {
            "XETHXXBT": trades,
            "last": str(last_cursor_ns),
        },
    }


def _kraken_handler(pages: list[dict]) -> Callable[[httpx.Request], httpx.Response]:
    """Closure that returns successive page payloads, then empty pages."""
    iterator = iter(pages)

    def handler(request: httpx.Request) -> httpx.Response:
        try:
            page = next(iterator)
        except StopIteration:
            page = _kraken_response([], 0)
        return httpx.Response(200, json=page)

    return handler


def _binance_handler(chunks: list[list]) -> Callable[[httpx.Request], httpx.Response]:
    """Closure that returns successive chunk payloads, then empty chunks."""
    iterator = iter(chunks)

    def handler(request: httpx.Request) -> httpx.Response:
        try:
            chunk = next(iterator)
        except StopIteration:
            chunk = []
        return httpx.Response(200, json=chunk)

    return handler


def _client_with(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------- Kraken adapter ----------------------------------------------------


class TestKrakenAdapter:
    def test_single_page_normalizes_to_canonical_schema(self) -> None:
        """A single page of trades produces the canonical (ts, price, qty, side) schema."""
        pages = [
            _kraken_response(
                trades=[
                    ["0.0500", "1.0", 1735689600.0, "b", "l", "", 1],   # 2025-01-01 00:00:00 UTC
                    ["0.0501", "2.0", 1735689601.5, "s", "l", "", 2],
                ],
                last_cursor_ns=1735689601_500000000,
            ),
        ]
        with _client_with(_kraken_handler(pages)) as client:
            df = fetch_kraken_eth_btc(
                start=datetime(2025, 1, 1, tzinfo=timezone.utc),
                end=datetime(2025, 1, 2, tzinfo=timezone.utc),
                client=client,
            )
        assert list(df.columns) == ["ts", "price", "qty", "side"]
        assert df["side"].tolist() == ["buy", "sell"]
        assert df["price"].tolist() == [0.0500, 0.0501]
        assert df["qty"].tolist() == [1.0, 2.0]

    def test_multi_page_pagination_accumulates_and_advances_cursor(self) -> None:
        """Pages are concatenated; cursor advances across calls."""
        pages = [
            _kraken_response(
                [["0.05", "1.0", 1735689600.0, "b", "l", "", 1]],
                last_cursor_ns=1735689600_500000000,
            ),
            _kraken_response(
                [["0.05", "1.0", 1735689601.0, "s", "l", "", 2]],
                last_cursor_ns=1735689601_500000000,
            ),
            _kraken_response([], last_cursor_ns=0),
        ]
        with _client_with(_kraken_handler(pages)) as client:
            df = fetch_kraken_eth_btc(
                start=datetime(2025, 1, 1, tzinfo=timezone.utc),
                end=datetime(2025, 1, 2, tzinfo=timezone.utc),
                client=client,
            )
        assert len(df) == 2

    def test_window_trims_overshoot_from_pagination(self) -> None:
        """Trades returned past `end` are trimmed; pagination stops once it crosses."""
        pages = [
            _kraken_response(
                trades=[
                    ["0.05", "1.0", 1735689600.0, "b", "l", "", 1],   # in window
                    ["0.05", "1.0", 1735776000.0, "s", "l", "", 2],   # past end (Jan 2)
                ],
                last_cursor_ns=1735776000_000000000,
            ),
        ]
        with _client_with(_kraken_handler(pages)) as client:
            df = fetch_kraken_eth_btc(
                start=datetime(2025, 1, 1, tzinfo=timezone.utc),
                end=datetime(2025, 1, 2, tzinfo=timezone.utc),
                client=client,
            )
        assert len(df) == 1

    def test_kraken_error_field_raises(self) -> None:
        """Non-empty `error` field surfaces as RuntimeError naming the error."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"error": ["EAPI:Rate limit exceeded"], "result": {}})

        with _client_with(handler) as client:
            with pytest.raises(RuntimeError, match="kraken error"):
                fetch_kraken_eth_btc(
                    start=datetime(2025, 1, 1, tzinfo=timezone.utc),
                    end=datetime(2025, 1, 2, tzinfo=timezone.utc),
                    client=client,
                )

    def test_unknown_side_value_raises(self) -> None:
        pages = [
            _kraken_response(
                [["0.05", "1.0", 1735689600.0, "x", "l", "", 1]],
                last_cursor_ns=1735689600_000000001,
            ),
        ]
        with _client_with(_kraken_handler(pages)) as client:
            with pytest.raises(RuntimeError, match="kraken side"):
                fetch_kraken_eth_btc(
                    start=datetime(2025, 1, 1, tzinfo=timezone.utc),
                    end=datetime(2025, 1, 2, tzinfo=timezone.utc),
                    client=client,
                )

    def test_empty_response_returns_empty_canonical_frame(self) -> None:
        with _client_with(_kraken_handler([])) as client:
            df = fetch_kraken_eth_btc(
                start=datetime(2025, 1, 1, tzinfo=timezone.utc),
                end=datetime(2025, 1, 2, tzinfo=timezone.utc),
                client=client,
            )
        assert list(df.columns) == ["ts", "price", "qty", "side"]
        assert len(df) == 0

    def test_naive_datetime_inputs_assumed_utc(self) -> None:
        """Tz-naive datetime inputs are interpreted as UTC, not silently rejected."""
        pages = [
            _kraken_response(
                [["0.05", "1.0", 1735689600.0, "b", "l", "", 1]],
                last_cursor_ns=1735689600_000000001,
            ),
        ]
        with _client_with(_kraken_handler(pages)) as client:
            df = fetch_kraken_eth_btc(
                start=datetime(2025, 1, 1),
                end=datetime(2025, 1, 2),
                client=client,
            )
        assert len(df) == 1


# ---------- Binance adapter ---------------------------------------------------


class TestBinanceAdapter:
    def test_single_chunk_normalizes_to_canonical_schema(self) -> None:
        """A single chunk normalizes (ts, price, qty, side) and flips isBuyerMaker correctly."""
        chunks = [
            [
                {"a": 1, "p": "0.0500", "q": "1.0", "f": 1, "l": 1, "T": 1735689600000, "m": False, "M": True},
                {"a": 2, "p": "0.0501", "q": "2.0", "f": 2, "l": 2, "T": 1735689601500, "m": True, "M": True},
            ],
        ]
        with _client_with(_binance_handler(chunks)) as client:
            df = fetch_binance_eth_btc(
                start=datetime(2025, 1, 1, tzinfo=timezone.utc),
                end=datetime(2025, 1, 1, 1, tzinfo=timezone.utc),
                client=client,
            )
        assert list(df.columns) == ["ts", "price", "qty", "side"]
        # m=False -> buyer is taker -> "buy"; m=True -> seller is taker -> "sell"
        assert df["side"].tolist() == ["buy", "sell"]

    def test_chunked_window_walks_in_one_hour_blocks(self) -> None:
        """A multi-hour request walks the time window in 1-hour chunks."""
        # Three 1-hour chunks expected for a 3-hour window.
        chunks = [
            [{"a": 1, "p": "0.05", "q": "1.0", "f": 1, "l": 1, "T": 1735689600000, "m": False, "M": True}],
            [{"a": 2, "p": "0.05", "q": "1.0", "f": 2, "l": 2, "T": 1735693200000, "m": True,  "M": True}],
            [{"a": 3, "p": "0.05", "q": "1.0", "f": 3, "l": 3, "T": 1735696800000, "m": False, "M": True}],
        ]
        with _client_with(_binance_handler(chunks)) as client:
            df = fetch_binance_eth_btc(
                start=datetime(2025, 1, 1, 0, tzinfo=timezone.utc),
                end=datetime(2025, 1, 1, 3, tzinfo=timezone.utc),
                client=client,
            )
        assert len(df) == 3

    def test_thousand_trade_chunk_triggers_re_pull_at_last_ts_plus_one(self) -> None:
        """A full 1000-trade chunk causes the cursor to advance to last_T + 1, not chunk_end."""
        # First chunk: 1000 trades all in second 0; second chunk: 1 trade at second 1; third: empty.
        full_chunk = [
            {"a": i, "p": "0.05", "q": "1.0", "f": i, "l": i, "T": 1735689600000, "m": False, "M": True}
            for i in range(1000)
        ]
        small_chunk = [
            {"a": 1001, "p": "0.05", "q": "1.0", "f": 1001, "l": 1001, "T": 1735689601000, "m": False, "M": True}
        ]
        chunks = [full_chunk, small_chunk]
        with _client_with(_binance_handler(chunks)) as client:
            df = fetch_binance_eth_btc(
                start=datetime(2025, 1, 1, 0, tzinfo=timezone.utc),
                end=datetime(2025, 1, 1, 0, 1, tzinfo=timezone.utc),
                client=client,
            )
        assert len(df) == 1001

    def test_window_trimming_excludes_trades_past_end(self) -> None:
        """Trades at or past `end` are excluded from the result."""
        chunks = [
            [
                {"a": 1, "p": "0.05", "q": "1.0", "f": 1, "l": 1, "T": 1735689600000, "m": False, "M": True},
                # ts past end:
                {"a": 2, "p": "0.05", "q": "1.0", "f": 2, "l": 2, "T": 1735693200000, "m": True,  "M": True},
            ],
        ]
        with _client_with(_binance_handler(chunks)) as client:
            df = fetch_binance_eth_btc(
                start=datetime(2025, 1, 1, 0, tzinfo=timezone.utc),
                end=datetime(2025, 1, 1, 1, tzinfo=timezone.utc),
                client=client,
            )
        assert len(df) == 1
        assert df["ts"].iloc[0] == pd.Timestamp("2025-01-01 00:00:00", tz="UTC")

    def test_unexpected_response_shape_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"error": "not a list"})
        with _client_with(handler) as client:
            with pytest.raises(RuntimeError, match="unexpected response"):
                fetch_binance_eth_btc(
                    start=datetime(2025, 1, 1, 0, tzinfo=timezone.utc),
                    end=datetime(2025, 1, 1, 1, tzinfo=timezone.utc),
                    client=client,
                )

    def test_empty_window_returns_empty_canonical_frame(self) -> None:
        with _client_with(_binance_handler([])) as client:
            df = fetch_binance_eth_btc(
                start=datetime(2025, 1, 1, 0, tzinfo=timezone.utc),
                end=datetime(2025, 1, 1, 1, tzinfo=timezone.utc),
                client=client,
            )
        assert list(df.columns) == ["ts", "price", "qty", "side"]
        assert len(df) == 0
