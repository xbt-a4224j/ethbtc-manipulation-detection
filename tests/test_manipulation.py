"""Tests for manipulation-detection primitives.

The contract every test locks in is named in the test's docstring. Together
they document the analysis layer's behavior on the canonical trades schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from challenge.analysis.manipulation import (
    buy_sell_ratio,
    inter_arrival_seconds,
    kyle_lambda,
    round_trip_rate,
    top_n_size_share,
)


# ---------- Fixtures ----------------------------------------------------------


def _make_trades(
    *,
    n: int = 100,
    start: str = "2025-07-01",
    freq: str = "1s",
    price: float | list[float] = 1.0,
    qty: float | list[float] = 10.0,
    side: str | list[str] | None = None,
) -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    if n == 0:
        return pd.DataFrame(
            {
                "ts": ts,
                "price": pd.Series([], dtype=float),
                "qty": pd.Series([], dtype=float),
                "side": pd.Series([], dtype=object),
            }
        )
    if side is None:
        side = (["buy", "sell"] * ((n + 1) // 2))[:n]
    elif isinstance(side, str):
        side = [side] * n
    return pd.DataFrame(
        {
            "ts": ts,
            "price": [price] * n if isinstance(price, float) else price,
            "qty": [qty] * n if isinstance(qty, float) else qty,
            "side": side,
        }
    )


@pytest.fixture
def balanced_wash_trades() -> pd.DataFrame:
    """Alternating buy/sell at constant price — the wash-flow archetype."""
    return _make_trades(n=120)


@pytest.fixture
def directional_trades() -> pd.DataFrame:
    """Five 10-second bins where signed flow and price change are perfectly correlated.

    Per-bin (signed_volume, dP) targets: (+10, +0.005), (+5, +0.002),
    (0, 0), (-5, -0.002), (-10, -0.005). The OLS slope of dP on
    signed_volume is positive by construction.
    """
    targets = [
        (10, 0.005),
        (5, 0.002),
        (0, 0.000),
        (-5, -0.002),
        (-10, -0.005),
    ]
    rows = []
    base_price = 1.0
    for bin_idx, (signed_vol, d_price) in enumerate(targets):
        n_buy = (signed_vol + 10) // 2
        n_sell = 10 - n_buy
        sides = ["buy"] * n_buy + ["sell"] * n_sell
        for t_idx in range(10):
            ts = pd.Timestamp("2025-07-01", tz="UTC") + pd.Timedelta(
                seconds=bin_idx * 10 + t_idx
            )
            price = base_price + d_price * (t_idx / 9)
            rows.append(
                {"ts": ts, "price": price, "qty": 1.0, "side": sides[t_idx]}
            )
        base_price += d_price
    return pd.DataFrame(rows)


# ---------- kyle_lambda -------------------------------------------------------


class TestKyleLambda:
    def test_balanced_wash_flow_is_zero(self, balanced_wash_trades: pd.DataFrame) -> None:
        """Perfectly balanced flow with zero price movement collapses lambda to 0."""
        assert kyle_lambda(balanced_wash_trades, bin_seconds=10) == 0.0

    def test_directional_flow_is_positive(self, directional_trades: pd.DataFrame) -> None:
        """Net buy pressure with rising prices produces a positive lambda."""
        assert kyle_lambda(directional_trades, bin_seconds=10) > 0.0

    def test_empty_input_returns_zero(self) -> None:
        """An empty trades DataFrame returns 0.0 (no signed-flow variance to regress)."""
        empty = _make_trades(n=0)
        assert kyle_lambda(empty) == 0.0

    def test_rejects_invalid_side_values(self) -> None:
        """Side values other than 'buy'/'sell' raise ValueError naming the bad input."""
        bad = _make_trades(n=2, side=["buy", "trade"])
        with pytest.raises(ValueError, match="side"):
            kyle_lambda(bad)

    def test_rejects_missing_columns(self) -> None:
        """Missing canonical columns raise ValueError listing the missing ones."""
        df = pd.DataFrame({"ts": [pd.Timestamp("2025-01-01", tz="UTC")], "price": [1.0]})
        with pytest.raises(ValueError, match="missing columns"):
            kyle_lambda(df)


# ---------- round_trip_rate ---------------------------------------------------


class TestRoundTripRate:
    def test_balanced_pairs_within_window_fully_paired(self) -> None:
        """Each buy-sell adjacent pair within the window pairs fully — rate = 1.0."""
        trades = _make_trades(n=10, qty=5.0)
        rate = round_trip_rate(trades, window_seconds=60, qty_tolerance=0.0)
        assert rate == pytest.approx(1.0)

    def test_unpairable_qty_mismatch_yields_zero(self) -> None:
        """Mismatched quantities outside tolerance never pair — rate = 0."""
        trades = pd.DataFrame(
            {
                "ts": pd.date_range("2025-07-01", periods=4, freq="1s", tz="UTC"),
                "price": [1.0] * 4,
                "qty": [10.0, 20.0, 10.0, 20.0],  # buy 10, sell 20, buy 10, sell 20
                "side": ["buy", "sell", "buy", "sell"],
            }
        )
        rate = round_trip_rate(trades, window_seconds=60, qty_tolerance=0.01)
        assert rate == 0.0

    def test_window_boundary_excludes_late_pairs(self) -> None:
        """A counterpart trade past the deadline does not pair."""
        trades = pd.DataFrame(
            {
                "ts": pd.to_datetime(
                    ["2025-07-01 00:00:00", "2025-07-01 00:02:00"], utc=True
                ),
                "price": [1.0, 1.0],
                "qty": [10.0, 10.0],
                "side": ["buy", "sell"],
            }
        )
        # 60s window: the sell is at +120s, outside the window.
        assert round_trip_rate(trades, window_seconds=60) == 0.0
        # 180s window: in range, pairs.
        assert round_trip_rate(trades, window_seconds=180) == pytest.approx(1.0)

    def test_tolerance_admits_near_matches(self) -> None:
        """Quantities within `qty_tolerance` (relative) are treated as a match."""
        trades = pd.DataFrame(
            {
                "ts": pd.date_range("2025-07-01", periods=2, freq="1s", tz="UTC"),
                "price": [1.0, 1.0],
                "qty": [10.0, 10.05],  # 0.5% mismatch
                "side": ["buy", "sell"],
            }
        )
        assert round_trip_rate(trades, window_seconds=60, qty_tolerance=0.0) == 0.0
        assert round_trip_rate(trades, window_seconds=60, qty_tolerance=0.01) == pytest.approx(1.0)

    def test_empty_input_returns_zero(self) -> None:
        assert round_trip_rate(_make_trades(n=0)) == 0.0
        assert round_trip_rate(_make_trades(n=1)) == 0.0

    def test_rejects_invalid_window(self) -> None:
        with pytest.raises(ValueError, match="window_seconds"):
            round_trip_rate(_make_trades(n=4), window_seconds=0)

    def test_rejects_negative_tolerance(self) -> None:
        with pytest.raises(ValueError, match="qty_tolerance"):
            round_trip_rate(_make_trades(n=4), qty_tolerance=-0.01)

    def test_greedy_pairing_doesnt_double_count(self) -> None:
        """Once a trade is paired it cannot pair again — paired_qty is bounded."""
        # Three buys followed by one sell of matching qty: only one pair forms.
        trades = pd.DataFrame(
            {
                "ts": pd.date_range("2025-07-01", periods=4, freq="1s", tz="UTC"),
                "price": [1.0] * 4,
                "qty": [10.0] * 4,
                "side": ["buy", "buy", "buy", "sell"],
            }
        )
        # Total qty is 40. One pair contributes 20 paired. Rate = 0.5.
        assert round_trip_rate(trades, window_seconds=60) == pytest.approx(0.5)


# ---------- top_n_size_share --------------------------------------------------


class TestTopNSizeShare:
    def test_all_same_size_yields_one(self) -> None:
        """If every trade is the same size, top-1 captures 100% of volume."""
        trades = _make_trades(n=50, qty=7.5)
        assert top_n_size_share(trades, n=1) == pytest.approx(1.0)

    def test_uniformly_diverse_sizes_yields_n_over_total(self) -> None:
        """N distinct sizes, each appearing once, with N <= total trades:
        top-N captures n/total of distinct-count, but here volume share."""
        trades = pd.DataFrame(
            {
                "ts": pd.date_range("2025-07-01", periods=10, freq="1s", tz="UTC"),
                "price": [1.0] * 10,
                "qty": list(range(1, 11)),  # 1, 2, ..., 10 — all distinct
                "side": ["buy"] * 10,
            }
        )
        # Total volume = 55. Top-3 sizes by qty are [10, 9, 8] = 27. Share = 27/55.
        assert top_n_size_share(trades, n=3) == pytest.approx(27 / 55)

    def test_rounding_collapses_floating_noise(self) -> None:
        """Sizes differing only at sub-rounding precision are treated as one bucket."""
        trades = pd.DataFrame(
            {
                "ts": pd.date_range("2025-07-01", periods=4, freq="1s", tz="UTC"),
                "price": [1.0] * 4,
                "qty": [1.0000001, 1.0000002, 1.0000003, 1.0000004],
                "side": ["buy"] * 4,
            }
        )
        # All collapse to 1.000000 at rounding_decimals=6 -> top-1 = 100%.
        assert top_n_size_share(trades, n=1, rounding_decimals=6) == pytest.approx(1.0)

    def test_zero_volume_returns_zero(self) -> None:
        trades = _make_trades(n=4, qty=0.0)
        assert top_n_size_share(trades, n=10) == 0.0

    def test_rejects_invalid_n(self) -> None:
        with pytest.raises(ValueError, match="n"):
            top_n_size_share(_make_trades(n=4), n=0)


# ---------- inter_arrival_seconds ---------------------------------------------


class TestInterArrivalSeconds:
    def test_uniform_spacing_yields_uniform_intervals(self) -> None:
        """Trades 1s apart produce inter-arrivals all equal to 1.0."""
        trades = _make_trades(n=10, freq="1s")
        ia = inter_arrival_seconds(trades)
        assert len(ia) == 9
        assert (ia == 1.0).all()

    def test_drops_leading_nan(self) -> None:
        """The leading NaN from .diff() is dropped — output length is n - 1."""
        trades = _make_trades(n=5)
        ia = inter_arrival_seconds(trades)
        assert ia.notna().all()

    def test_sorts_defensively(self) -> None:
        """Out-of-order timestamps are sorted before differencing."""
        ts = pd.to_datetime(
            ["2025-07-01 00:00:02", "2025-07-01 00:00:00", "2025-07-01 00:00:05"],
            utc=True,
        )
        trades = pd.DataFrame(
            {"ts": ts, "price": [1.0] * 3, "qty": [1.0] * 3, "side": ["buy"] * 3}
        )
        ia = inter_arrival_seconds(trades)
        assert list(ia) == [2.0, 3.0]

    def test_empty_or_singleton_returns_empty(self) -> None:
        assert len(inter_arrival_seconds(_make_trades(n=0))) == 0
        assert len(inter_arrival_seconds(_make_trades(n=1))) == 0

    def test_returns_named_series(self) -> None:
        """The Series carries the 'inter_arrival_s' name for downstream plotting."""
        ia = inter_arrival_seconds(_make_trades(n=5))
        assert ia.name == "inter_arrival_s"


# ---------- buy_sell_ratio ----------------------------------------------------


class TestBuySellRatio:
    def test_balanced_yields_half(self) -> None:
        """Equal buy and sell volume yields 0.5."""
        trades = _make_trades(n=10)  # alternating
        assert buy_sell_ratio(trades) == pytest.approx(0.5)

    def test_all_buy_yields_one(self) -> None:
        trades = _make_trades(n=10, side="buy")
        assert buy_sell_ratio(trades) == 1.0

    def test_all_sell_yields_zero(self) -> None:
        trades = _make_trades(n=10, side="sell")
        assert buy_sell_ratio(trades) == 0.0

    def test_uneven_volume_weighted_by_qty_not_count(self) -> None:
        """The ratio is volume-weighted, not trade-count-weighted."""
        trades = pd.DataFrame(
            {
                "ts": pd.date_range("2025-07-01", periods=2, freq="1s", tz="UTC"),
                "price": [1.0, 1.0],
                "qty": [100.0, 1.0],
                "side": ["buy", "sell"],
            }
        )
        assert buy_sell_ratio(trades) == pytest.approx(100 / 101)

    def test_zero_volume_returns_neutral_default(self) -> None:
        """Empty / zero-volume input returns 0.5 instead of div-by-zero."""
        assert buy_sell_ratio(_make_trades(n=0)) == 0.5
        assert buy_sell_ratio(_make_trades(n=4, qty=0.0)) == 0.5

    def test_case_insensitive_side_matching(self) -> None:
        """Side values are matched case-insensitively."""
        trades = pd.DataFrame(
            {
                "ts": pd.date_range("2025-07-01", periods=2, freq="1s", tz="UTC"),
                "price": [1.0, 1.0],
                "qty": [10.0, 10.0],
                "side": ["BUY", "Sell"],
            }
        )
        assert buy_sell_ratio(trades) == 0.5
