"""Tests for the CSV loaders.

Synthetic CSV fixtures written to tmp_path; no committed test data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from challenge.sources.csv_loader import load_orderbooks, load_trades, top_of_book


# ---------- load_trades -------------------------------------------------------


class TestLoadTrades:
    def test_canonical_columns_load_directly(self, tmp_path: Path) -> None:
        """A CSV with canonical column names loads with no overrides."""
        path = tmp_path / "trades.csv"
        path.write_text(
            "ts,price,qty,side\n"
            "2025-07-01T00:00:00Z,1.0,10.0,buy\n"
            "2025-07-01T00:00:01Z,1.01,5.0,sell\n"
        )
        df = load_trades(path)
        assert list(df.columns) == ["ts", "price", "qty", "side"]
        assert len(df) == 2
        assert df["ts"].dt.tz is not None
        assert df["side"].tolist() == ["buy", "sell"]

    def test_alias_column_names_detected(self, tmp_path: Path) -> None:
        """Common alias column names ('timestamp', 'size', 'type') auto-resolve."""
        path = tmp_path / "trades_aliases.csv"
        path.write_text(
            "timestamp,price,size,type\n"
            "2025-07-01T00:00:00Z,1.0,10.0,b\n"
            "2025-07-01T00:00:01Z,1.01,5.0,s\n"
        )
        df = load_trades(path)
        assert df["side"].tolist() == ["buy", "sell"]

    def test_explicit_column_map_overrides_detection(self, tmp_path: Path) -> None:
        """`column_map` lets the caller force-name columns the detector won't find."""
        path = tmp_path / "trades_custom.csv"
        path.write_text(
            "exec_time,exec_price,exec_qty,exec_side\n"
            "2025-07-01T00:00:00Z,1.0,10.0,buy\n"
        )
        df = load_trades(
            path,
            column_map={"ts": "exec_time", "price": "exec_price", "qty": "exec_qty", "side": "exec_side"},
        )
        assert len(df) == 1
        assert df["price"].iloc[0] == 1.0

    def test_unknown_side_value_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "trades_bad_side.csv"
        path.write_text("ts,price,qty,side\n2025-07-01T00:00:00Z,1.0,10.0,maker\n")
        with pytest.raises(ValueError, match="unrecognized side"):
            load_trades(path)

    def test_negative_qty_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "trades_neg.csv"
        path.write_text("ts,price,qty,side\n2025-07-01T00:00:00Z,1.0,-5.0,buy\n")
        with pytest.raises(ValueError, match="negative quantit"):
            load_trades(path)

    def test_unparseable_timestamp_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "trades_bad_ts.csv"
        path.write_text("ts,price,qty,side\nnot-a-date,1.0,10.0,buy\n")
        with pytest.raises(ValueError, match="failed to parse"):
            load_trades(path)

    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "trades_missing.csv"
        path.write_text("ts,price,qty\n2025-07-01T00:00:00Z,1.0,10.0\n")
        with pytest.raises(ValueError, match="side"):
            load_trades(path)

    def test_output_is_sorted_by_ts(self, tmp_path: Path) -> None:
        path = tmp_path / "trades_unsorted.csv"
        path.write_text(
            "ts,price,qty,side\n"
            "2025-07-01T00:00:02Z,1.0,10.0,buy\n"
            "2025-07-01T00:00:00Z,1.0,10.0,sell\n"
            "2025-07-01T00:00:01Z,1.0,10.0,buy\n"
        )
        df = load_trades(path)
        assert df["ts"].is_monotonic_increasing


# ---------- load_orderbooks ---------------------------------------------------


def _write_listdict_book(path: Path) -> None:
    """Two snapshots, 3 levels per side, in the supplied dataset's encoding.

    Source columns: timestamp, asks, bids
    Each side is a Python-literal list of {'price': float, 'size': float}.
    Asks are intentionally written in ascending order; bids in descending
    order; the loader's reordering should leave them alone.
    """
    rows = [
        {
            "timestamp": "2025-07-01T00:00:00Z",
            "asks": "[{'price': 0.0501, 'size': 8}, {'price': 0.0502, 'size': 12}, {'price': 0.0503, 'size': 18}]",
            "bids": "[{'price': 0.0500, 'size': 10}, {'price': 0.0499, 'size': 15}, {'price': 0.0498, 'size': 20}]",
        },
        {
            "timestamp": "2025-07-01T00:00:30Z",
            "asks": "[{'price': 0.0502, 'size': 9}, {'price': 0.0503, 'size': 13}, {'price': 0.0504, 'size': 19}]",
            "bids": "[{'price': 0.0501, 'size': 11}, {'price': 0.0500, 'size': 14}, {'price': 0.0499, 'size': 21}]",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


class TestLoadOrderbooks:
    def test_listdict_format_reshapes_to_long_canonical(self, tmp_path: Path) -> None:
        path = tmp_path / "ob.csv"
        _write_listdict_book(path)
        df = load_orderbooks(path)
        assert set(df.columns) == {"ts", "side", "level", "price", "qty"}
        # 2 snapshots * 2 sides * 3 levels = 12 rows
        assert len(df) == 12
        assert sorted(df["level"].unique().tolist()) == [0, 1, 2]
        assert sorted(df["side"].unique().tolist()) == ["ask", "bid"]

    def test_top_of_book_is_lowest_ask_and_highest_bid(self, tmp_path: Path) -> None:
        path = tmp_path / "ob.csv"
        _write_listdict_book(path)
        df = load_orderbooks(path)
        first = df[df["ts"] == df["ts"].min()]
        top_ask = first[(first["side"] == "ask") & (first["level"] == 0)].iloc[0]
        top_bid = first[(first["side"] == "bid") & (first["level"] == 0)].iloc[0]
        assert top_ask["price"] == pytest.approx(0.0501)
        assert top_bid["price"] == pytest.approx(0.0500)

    def test_resorts_misordered_input(self, tmp_path: Path) -> None:
        """Even if the source lists levels in scrambled order, level 0 is still top of book."""
        path = tmp_path / "ob_scrambled.csv"
        rows = [
            {
                "timestamp": "2025-07-01T00:00:00Z",
                "asks": "[{'price': 0.0503, 'size': 18}, {'price': 0.0501, 'size': 8}, {'price': 0.0502, 'size': 12}]",
                "bids": "[{'price': 0.0498, 'size': 20}, {'price': 0.0500, 'size': 10}, {'price': 0.0499, 'size': 15}]",
            }
        ]
        pd.DataFrame(rows).to_csv(path, index=False)
        df = load_orderbooks(path)
        top_ask = df[(df["side"] == "ask") & (df["level"] == 0)].iloc[0]
        top_bid = df[(df["side"] == "bid") & (df["level"] == 0)].iloc[0]
        assert top_ask["price"] == pytest.approx(0.0501)
        assert top_bid["price"] == pytest.approx(0.0500)

    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "ob_missing.csv"
        path.write_text("timestamp,asks\n2025-07-01T00:00:00Z,\"[]\"\n")
        with pytest.raises(ValueError, match="bids"):
            load_orderbooks(path)

    def test_unparseable_book_string_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "ob_bad.csv"
        rows = [{"timestamp": "2025-07-01T00:00:00Z", "asks": "not-a-list", "bids": "[]"}]
        pd.DataFrame(rows).to_csv(path, index=False)
        with pytest.raises(ValueError, match="failed to parse book"):
            load_orderbooks(path)

    def test_non_list_payload_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "ob_dict.csv"
        rows = [{"timestamp": "2025-07-01T00:00:00Z", "asks": "{'price': 1, 'size': 1}", "bids": "[]"}]
        pd.DataFrame(rows).to_csv(path, index=False)
        with pytest.raises(ValueError, match="list-of-dicts"):
            load_orderbooks(path)


# ---------- top_of_book -------------------------------------------------------


class TestTopOfBook:
    def test_computes_mid_spread_and_imbalance(self, tmp_path: Path) -> None:
        path = tmp_path / "ob.csv"
        _write_listdict_book(path)
        long_book = load_orderbooks(path)
        top = top_of_book(long_book)
        assert {"bid_price", "ask_price", "bid_qty", "ask_qty", "mid", "spread_bps", "imbalance"} <= set(top.columns)
        first = top.iloc[0]
        assert first["mid"] == pytest.approx((0.0500 + 0.0501) / 2)
        # Spread = (0.0501 - 0.0500) / mid * 10000 ~ 19.8 bps
        assert first["spread_bps"] == pytest.approx((0.0001 / first["mid"]) * 10_000)
        # Imbalance = bid_qty / (bid_qty + ask_qty) = 10 / 18
        assert first["imbalance"] == pytest.approx(10 / 18)

    def test_rejects_missing_columns(self) -> None:
        df = pd.DataFrame({"ts": [pd.Timestamp("2025-01-01", tz="UTC")]})
        with pytest.raises(ValueError, match="missing"):
            top_of_book(df)

    def test_empty_book_returns_empty_dataframe_with_expected_columns(self) -> None:
        empty = pd.DataFrame(
            {
                "ts": pd.Series(dtype="datetime64[ns, UTC]"),
                "side": pd.Series(dtype=object),
                "level": pd.Series(dtype=int),
                "price": pd.Series(dtype=float),
                "qty": pd.Series(dtype=float),
            }
        )
        top = top_of_book(empty)
        assert len(top) == 0
        assert "spread_bps" in top.columns
