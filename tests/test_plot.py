"""Smoke tests for plot helpers.

Visual correctness can't be unit-tested, but we can verify each helper
returns a Figure, doesn't crash on empty edge cases, and produces axes
with the expected labels. Use the matplotlib non-interactive backend
to keep tests headless.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402  must precede any matplotlib import in tests

import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from challenge.analysis.plot import (
    apply_default_style,
    plot_book_depth_profile,
    plot_cumulative_volume_by_side,
    plot_imbalance_timeseries,
    plot_inter_arrival_distribution,
    plot_kyle_lambda_comparison,
    plot_orderbook_depth_comparison,
    plot_per_hour_volume_breakdown,
    plot_price_trace_with_trades,
    plot_spread_timeseries,
    plot_trade_size_distribution,
    plot_volume_overview,
)


@pytest.fixture(autouse=True)
def _close_all_figures():
    yield
    plt.close("all")


def _trades(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.date_range("2025-07-01", periods=n, freq="1s", tz="UTC"),
            "price": [1.0] * n,
            "qty": [10.0] * n,
            "side": (["buy", "sell"] * ((n + 1) // 2))[:n],
        }
    )


class TestPlotVolumeOverview:
    def test_returns_figure(self) -> None:
        idx = pd.date_range("2025-07-01", periods=5, freq="h", tz="UTC")
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        fig = plot_volume_overview({"challenge": s, "kraken": s * 2})
        assert isinstance(fig, mfigure.Figure)
        assert fig.axes[0].get_ylabel() == "volume (base asset)"


class TestPlotTradeSizeDistribution:
    def test_returns_figure(self) -> None:
        fig = plot_trade_size_distribution({"challenge": _trades(), "kraken": _trades()})
        assert isinstance(fig, mfigure.Figure)
        assert fig.axes[0].get_xscale() == "log"

    def test_handles_empty_volume_gracefully(self) -> None:
        empty = _trades(n=3).assign(qty=0.0)
        fig = plot_trade_size_distribution({"challenge": empty})
        assert isinstance(fig, mfigure.Figure)


class TestPlotInterArrivalDistribution:
    def test_returns_figure(self) -> None:
        ia = pd.Series([0.1, 0.5, 1.0, 2.0, 5.0], name="inter_arrival_s")
        fig = plot_inter_arrival_distribution({"challenge": ia, "kraken": ia * 2})
        assert isinstance(fig, mfigure.Figure)
        assert fig.axes[0].get_xscale() == "log"

    def test_handles_all_zero_inter_arrivals_gracefully(self) -> None:
        fig = plot_inter_arrival_distribution({"challenge": pd.Series([0.0, 0.0])})
        assert isinstance(fig, mfigure.Figure)

    def test_subplots_layout_has_one_axis_per_venue(self) -> None:
        ia = pd.Series([0.1, 0.5, 1.0, 2.0, 5.0], name="inter_arrival_s")
        fig = plot_inter_arrival_distribution(
            {"challenge": ia, "kraken": ia * 2, "binance": ia * 0.1},
            layout="subplots",
        )
        assert isinstance(fig, mfigure.Figure)
        assert len(fig.axes) == 3
        # All x-axes log-scaled.
        assert all(a.get_xscale() == "log" for a in fig.axes)

    def test_rejects_unknown_layout(self) -> None:
        ia = pd.Series([0.1, 0.5, 1.0])
        with pytest.raises(ValueError, match="layout"):
            plot_inter_arrival_distribution({"challenge": ia}, layout="grid")


class TestPlotKyleLambdaComparison:
    def test_returns_figure_with_correct_bars(self) -> None:
        fig = plot_kyle_lambda_comparison({"challenge": 0.001, "kraken": 0.014, "binance": 0.012})
        assert isinstance(fig, mfigure.Figure)
        # One bar per venue.
        assert len(fig.axes[0].patches) == 3


class TestPlotOrderbookDepthComparison:
    def test_returns_figure(self) -> None:
        idx = pd.date_range("2025-07-01", periods=5, freq="min", tz="UTC")
        top = pd.DataFrame(
            {"bid_qty": [10.0] * 5, "ask_qty": [12.0] * 5},
            index=idx,
        )
        fig = plot_orderbook_depth_comparison({"challenge": top, "kraken": top * 1.5})
        assert isinstance(fig, mfigure.Figure)
        assert fig.axes[0].get_ylabel() == "top-of-book depth (base asset)"


# ---------- Provisional / single-venue exploratory charts -------------------


class TestApplyDefaultStyle:
    def test_sets_known_rcparam(self) -> None:
        """Calling the helper sets at least one known matplotlib rcParam."""
        apply_default_style()
        import matplotlib as mpl
        assert mpl.rcParams["axes.titleweight"] == "bold"


def _top(n: int = 8) -> pd.DataFrame:
    idx = pd.date_range("2025-09-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "bid_price": [0.05] * n,
            "ask_price": [0.0501] * n,
            "bid_qty": [10.0] * n,
            "ask_qty": [12.0] * n,
            "mid": [(0.05 + 0.0501) / 2] * n,
            "spread_bps": [20.0] * n,
            "imbalance": [10 / 22] * n,
        },
        index=idx,
    )


class TestPriceTraceWithTrades:
    def test_returns_figure_with_two_legend_entries(self) -> None:
        fig = plot_price_trace_with_trades(_trades(n=20))
        assert isinstance(fig, mfigure.Figure)
        assert len(fig.axes[0].get_legend().get_texts()) == 2

    def test_handles_single_side_input(self) -> None:
        df = _trades(n=10).assign(side="buy")
        fig = plot_price_trace_with_trades(df)
        assert isinstance(fig, mfigure.Figure)


class TestCumulativeVolumeBySide:
    def test_returns_figure(self) -> None:
        fig = plot_cumulative_volume_by_side(_trades(n=20))
        assert isinstance(fig, mfigure.Figure)
        assert fig.axes[0].get_ylabel() == "cumulative volume (base asset)"


class TestPerHourVolumeBreakdown:
    def test_returns_figure(self) -> None:
        fig = plot_per_hour_volume_breakdown(_trades(n=120))
        assert isinstance(fig, mfigure.Figure)

    def test_handles_empty_input(self) -> None:
        empty = pd.DataFrame(
            {
                "ts": pd.Series(dtype="datetime64[ns, UTC]"),
                "price": pd.Series(dtype=float),
                "qty": pd.Series(dtype=float),
                "side": pd.Series(dtype=object),
            }
        )
        fig = plot_per_hour_volume_breakdown(empty)
        assert isinstance(fig, mfigure.Figure)


class TestSpreadTimeseries:
    def test_returns_figure(self) -> None:
        fig = plot_spread_timeseries(_top())
        assert isinstance(fig, mfigure.Figure)
        assert fig.axes[0].get_ylabel() == "spread (bps)"


class TestImbalanceTimeseries:
    def test_returns_figure_with_y_limits(self) -> None:
        fig = plot_imbalance_timeseries(_top())
        assert isinstance(fig, mfigure.Figure)
        assert fig.axes[0].get_ylim() == (0.0, 1.0)


class TestBookDepthProfile:
    def test_returns_figure_for_a_simple_book(self) -> None:
        ts = pd.Timestamp("2025-09-01 12:00", tz="UTC")
        long_book = pd.DataFrame(
            [
                {"ts": ts, "side": "bid", "level": 0, "price": 0.0500, "qty": 10.0},
                {"ts": ts, "side": "bid", "level": 1, "price": 0.0499, "qty": 15.0},
                {"ts": ts, "side": "ask", "level": 0, "price": 0.0501, "qty": 8.0},
                {"ts": ts, "side": "ask", "level": 1, "price": 0.0502, "qty": 12.0},
            ]
        )
        fig = plot_book_depth_profile(long_book)
        assert isinstance(fig, mfigure.Figure)

    def test_raises_on_missing_snapshot(self) -> None:
        ts = pd.Timestamp("2025-09-01", tz="UTC")
        long_book = pd.DataFrame(
            [{"ts": ts, "side": "bid", "level": 0, "price": 0.05, "qty": 1.0},
             {"ts": ts, "side": "ask", "level": 0, "price": 0.06, "qty": 1.0}]
        )
        with pytest.raises(ValueError, match="no snapshot"):
            plot_book_depth_profile(long_book, snapshot_ts=pd.Timestamp("2099-01-01", tz="UTC"))
