"""
Chart helpers for the report.

Five charts per primer module 06. Each function takes the relevant venue-keyed
mapping and returns a `matplotlib.figure.Figure`. Visual identity rules
(colors, weights, layout) live here so the notebook stays narrative.

The default color map is a deliberate choice: the challenge venue is near-black
(it's the subject); comparison venues are blue and red (distinguishable but
neutrally weighted). Override via the `colors` argument when needed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_COLORS: Final[dict[str, str]] = {
    "challenge": "#222222",
    "kraken": "#1f6feb",
    "binance": "#a33333",
}
SIDE_COLORS: Final[dict[str, str]] = {
    "buy": "#1f7a3a",   # green for buy aggressor
    "sell": "#a33333",  # red for sell aggressor
}
_FALLBACK_PALETTE: Final[tuple[str, ...]] = ("#666666", "#888888", "#aaaaaa")


def _color_for(name: str, colors: Mapping[str, str] | None, idx: int) -> str:
    if colors is not None and name in colors:
        return colors[name]
    if name in DEFAULT_COLORS:
        return DEFAULT_COLORS[name]
    return _FALLBACK_PALETTE[idx % len(_FALLBACK_PALETTE)]


def apply_default_style() -> None:
    """Apply consistent matplotlib defaults for the report's visual identity.

    Call once at the top of any notebook or script. Settings match the
    style guide in primer module 06 (Inter font, light grids, no legend
    frame, consistent type sizes).
    """
    plt.rcParams.update(
        {
            "figure.figsize": (11, 4.5),
            "figure.dpi": 100,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.3,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        }
    )


def plot_volume_overview(
    venue_volumes: Mapping[str, pd.Series],
    *,
    colors: Mapping[str, str] | None = None,
    title: str = "Volume by venue across the comparison window",
) -> mfigure.Figure:
    """Per-bin volume across venues over the same window."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (name, series) in enumerate(venue_volumes.items()):
        ax.plot(
            series.index, series.values,
            label=name, color=_color_for(name, colors, i), linewidth=1.4,
        )
    ax.set_ylabel("volume (base asset)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_trade_size_distribution(
    venue_trades: Mapping[str, pd.DataFrame],
    *,
    colors: Mapping[str, str] | None = None,
    bins: int = 60,
    density: bool = True,
    title: str = "Trade-size distribution by venue",
) -> mfigure.Figure:
    """Per-venue distribution of trade sizes, log-scaled x-axis.

    `density=True` (default) normalizes each histogram so its area is 1 —
    lets you compare *shape* across venues with very different trade
    counts. Set `density=False` to plot raw counts when the goal is to
    show the absolute volume disparity.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))
    all_sizes = pd.concat([df["qty"].astype(float) for df in venue_trades.values()])
    positive = all_sizes[all_sizes > 0]
    if len(positive) == 0:
        ax.set_title(title + " (no positive sizes)")
        return fig
    edges = np.logspace(np.log10(positive.min()), np.log10(positive.max()), bins)
    for i, (name, df) in enumerate(venue_trades.items()):
        sizes = df["qty"].astype(float)
        ax.hist(
            sizes[sizes > 0], bins=edges, histtype="step", density=density,
            label=name, color=_color_for(name, colors, i), linewidth=1.4,
        )
    ax.set_xscale("log")
    ax.set_xlabel("trade size (base asset)")
    ax.set_ylabel("density" if density else "count")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_inter_arrival_distribution(
    venue_inter_arrivals: Mapping[str, pd.Series],
    *,
    colors: Mapping[str, str] | None = None,
    bins: int = 80,
    density: bool = True,
    title: str = "Inter-arrival distribution by venue",
    layout: str = "overlay",
) -> mfigure.Figure:
    """Per-venue inter-arrival time distribution on a log x-axis.

    `density=True` (default) normalizes each histogram so its area is 1.
    Without normalization, a venue with 100x more trades drowns out the
    others' shapes. The shape is what matters for spotting bot signatures
    (sharp narrow peaks at fixed intervals).

    `layout="overlay"` plots all venues on a single axis (good for direct
    shape comparison). `layout="subplots"` gives each venue its own
    horizontal panel sharing x-axis — useful when one venue's distribution
    sits at a very different scale (e.g. the challenge venue's slow cadence
    against Binance's sub-second cadence) and the overlay flattens it.
    """
    if layout not in ("overlay", "subplots"):
        raise ValueError(f"layout must be 'overlay' or 'subplots'; got {layout!r}")

    all_pos = pd.concat([s[s > 0] for s in venue_inter_arrivals.values()])

    if layout == "overlay":
        fig, ax = plt.subplots(figsize=(11, 4.5))
        if len(all_pos) == 0:
            ax.set_title(title + " (no positive inter-arrivals)")
            return fig
        edges = np.logspace(np.log10(max(all_pos.min(), 1e-6)), np.log10(all_pos.max()), bins)
        for i, (name, ia) in enumerate(venue_inter_arrivals.items()):
            ax.hist(
                ia[ia > 0], bins=edges, histtype="step", density=density,
                label=name, color=_color_for(name, colors, i), linewidth=1.4,
            )
        ax.set_xscale("log")
        ax.set_xlabel("inter-arrival (s)")
        ax.set_ylabel("density" if density else "count")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        return fig

    n = len(venue_inter_arrivals)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.4 * n + 0.6), sharex=True)
    if n == 1:
        axes = [axes]
    if len(all_pos) == 0:
        axes[0].set_title(title + " (no positive inter-arrivals)")
        return fig
    edges = np.logspace(np.log10(max(all_pos.min(), 1e-6)), np.log10(all_pos.max()), bins)
    for i, (name, ia) in enumerate(venue_inter_arrivals.items()):
        ax = axes[i]
        positive = ia[ia > 0]
        color = _color_for(name, colors, i)
        ax.hist(
            positive, bins=edges, histtype="stepfilled", density=density,
            color=color, alpha=0.25, edgecolor=color, linewidth=1.4,
        )
        ax.set_xscale("log")
        ax.set_ylabel("density" if density else "count")
        ax.grid(True, which="both", alpha=0.3)
        if len(positive):
            median = float(positive.median())
            ax.axvline(median, color=color, linestyle="--", linewidth=0.8, alpha=0.8)
            label = (
                f"{name}  (n={len(positive):,}, median={median:.3g}s, "
                f"p95={float(positive.quantile(0.95)):.3g}s)"
            )
        else:
            label = f"{name}  (no positive inter-arrivals)"
        ax.text(
            0.01, 0.92, label,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold", color=color,
        )
    axes[-1].set_xlabel("inter-arrival (s)")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout()
    return fig


def plot_kyle_lambda_comparison(
    lambdas: Mapping[str, float],
    *,
    colors: Mapping[str, str] | None = None,
    title: str = "Kyle's lambda by venue (60-second bins)",
) -> mfigure.Figure:
    """Bar chart of Kyle's lambda across venues for the same window."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(lambdas.keys())
    values = [lambdas[n] for n in names]
    bar_colors = [_color_for(n, colors, i) for i, n in enumerate(names)]
    ax.bar(names, values, color=bar_colors)
    ax.set_ylabel("lambda (price per unit signed flow)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_orderbook_depth_comparison(
    venue_top_of_book: Mapping[str, pd.DataFrame],
    *,
    colors: Mapping[str, str] | None = None,
    title: str = "Top-of-book depth by venue",
) -> mfigure.Figure:
    """Time series of top-of-book size across venues.

    Each input DataFrame must have a DatetimeIndex and the columns
    `bid_qty` and `ask_qty`. Plots bid_qty + ask_qty as a single
    "total top-of-book depth" line per venue.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (name, top) in enumerate(venue_top_of_book.items()):
        depth = top["bid_qty"].astype(float) + top["ask_qty"].astype(float)
        ax.plot(
            top.index, depth.values,
            label=name, color=_color_for(name, colors, i), linewidth=1.0, alpha=0.9,
        )
    ax.set_ylabel("top-of-book depth (base asset)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Provisional / single-venue exploratory charts.
#
# The five anchors above govern the report. The charts below are for
# building intuition before the formal cross-venue calibration. They
# expect canonical-schema trade and orderbook frames; outputs are saved
# to disk by `scripts/provisional_charts.py`.
# ---------------------------------------------------------------------------


def plot_price_trace_with_trades(
    trades: pd.DataFrame,
    *,
    title: str = "Price trace with buy/sell overlay",
    max_marker_size: float = 100.0,
) -> mfigure.Figure:
    """Price line plus buy/sell scatter with marker size proportional to qty.

    Marker sizes are normalized so the largest trade in the window maps
    to `max_marker_size`. Use to visually inspect price action vs flow
    direction — wash regimes show flat price with dense buy+sell scatter;
    informed regimes show price moving in the direction of dominant flow.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    df = trades.sort_values("ts")
    ax.plot(df["ts"], df["price"], color="#888888", linewidth=0.8, alpha=0.8)

    qty_max = float(df["qty"].max()) if len(df) else 1.0
    sizes = (df["qty"].astype(float) / qty_max) * max_marker_size
    sides = df["side"].str.lower()
    for side in ("buy", "sell"):
        mask = sides == side
        ax.scatter(
            df.loc[mask, "ts"], df.loc[mask, "price"],
            s=sizes[mask], color=SIDE_COLORS[side], alpha=0.6,
            edgecolors="none", label=f"{side} (n={int(mask.sum())})",
        )
    ax.set_ylabel("price")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_cumulative_volume_by_side(
    trades: pd.DataFrame,
    *,
    title: str = "Cumulative volume by side",
) -> mfigure.Figure:
    """Two lines: cumulative buy volume and cumulative sell volume over time.

    The vertical gap at any point is the running net imbalance. Useful
    for seeing *when* an imbalance accumulated — sustained slope on one
    side suggests a regime, a kink suggests an event.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))
    df = trades.sort_values("ts").copy()
    df["side_lc"] = df["side"].str.lower()
    for side in ("buy", "sell"):
        mask = df["side_lc"] == side
        cum = df.loc[mask, "qty"].astype(float).cumsum()
        ax.plot(
            df.loc[mask, "ts"], cum.values,
            label=f"cumulative {side} (total {cum.iloc[-1] if len(cum) else 0:.4f})",
            color=SIDE_COLORS[side], linewidth=1.6,
        )
    ax.set_ylabel("cumulative volume (base asset)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_per_hour_volume_breakdown(
    trades: pd.DataFrame,
    *,
    title: str = "Per-hour volume by side",
    log_y: bool = False,
) -> mfigure.Figure:
    """Side-by-side bar of per-hour buy and sell volume.

    Hourly bins on the x-axis. Buy and sell bars sit side-by-side rather
    than stacked, so a tiny side stays visible even when the other side
    is many orders of magnitude larger. Set `log_y=True` to plot a log
    y-axis when the disparity spans 1000x+.
    """
    fig, ax = plt.subplots(figsize=(12, 4.5))
    if len(trades) == 0:
        ax.set_title(title + " (empty)")
        return fig

    df = trades.copy()
    df["hour"] = df["ts"].dt.floor("h")
    df["side_lc"] = df["side"].str.lower()
    pivot = (
        df.groupby(["hour", "side_lc"])["qty"].sum().unstack(fill_value=0.0)
    )
    for side in ("buy", "sell"):
        if side not in pivot.columns:
            pivot[side] = 0.0

    half_width = pd.Timedelta(minutes=22)  # ~half of an hour, with a small gap
    ax.bar(
        pivot.index, pivot["buy"], width=half_width,
        color=SIDE_COLORS["buy"], label="buy", align="edge",
    )
    ax.bar(
        pivot.index - half_width, pivot["sell"], width=half_width,
        color=SIDE_COLORS["sell"], label="sell", align="edge",
    )
    ax.set_ylabel("volume (base asset)" + (" — log scale" if log_y else ""))
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_spread_timeseries(
    top: pd.DataFrame,
    *,
    title: str = "Top-of-book spread over time",
) -> mfigure.Figure:
    """Line chart of top-of-book spread (bps) over time."""
    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.plot(top.index, top["spread_bps"].values, color="#222222", linewidth=1.0)
    ax.fill_between(top.index, 0, top["spread_bps"].values, color="#222222", alpha=0.10)
    ax.set_ylabel("spread (bps)")
    ax.set_title(title)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_imbalance_timeseries(
    top: pd.DataFrame,
    *,
    title: str = "Top-of-book imbalance over time",
) -> mfigure.Figure:
    """Line chart of bid-share imbalance over time, with the 0.5 reference line.

    Imbalance = bid_qty / (bid_qty + ask_qty). Sustained values away
    from 0.5 mean the book is structurally leaning one way.
    """
    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.plot(top.index, top["imbalance"].values, color="#1f6feb", linewidth=1.0)
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("bid share (bid_qty / total)")
    ax.set_title(title)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_book_depth_profile(
    long_book: pd.DataFrame,
    *,
    snapshot_ts: pd.Timestamp | None = None,
    title: str | None = None,
) -> mfigure.Figure:
    """Cumulative depth profile for a single snapshot.

    X-axis is distance from mid in bps; y-axis is cumulative size on each
    side from top-of-book outward. If `snapshot_ts` is None, picks the
    snapshot closest to the median timestamp.
    """
    if snapshot_ts is None:
        snapshot_ts = long_book["ts"].quantile(0.5, interpolation="nearest")
    snap = long_book.loc[long_book["ts"] == snapshot_ts].copy()
    if snap.empty:
        raise ValueError(f"no snapshot at ts={snapshot_ts}")

    bids = snap.loc[snap["side"] == "bid"].sort_values("level")
    asks = snap.loc[snap["side"] == "ask"].sort_values("level")
    if bids.empty or asks.empty:
        raise ValueError(f"snapshot at {snapshot_ts} missing one side of book")

    mid = (bids["price"].iloc[0] + asks["price"].iloc[0]) / 2
    bid_dist_bps = (mid - bids["price"]) / mid * 10_000
    ask_dist_bps = (asks["price"] - mid) / mid * 10_000

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.step(-bid_dist_bps.values, bids["qty"].cumsum().values,
            where="post", color=SIDE_COLORS["buy"], label="bids (cumulative)", linewidth=1.4)
    ax.step(ask_dist_bps.values, asks["qty"].cumsum().values,
            where="post", color=SIDE_COLORS["sell"], label="asks (cumulative)", linewidth=1.4)
    ax.axvline(0, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_xlabel("distance from mid (bps); negative = bid side")
    ax.set_ylabel("cumulative size (base asset)")
    ax.set_title(title or f"Order book depth profile at {snapshot_ts}")
    ax.legend()
    fig.tight_layout()
    return fig
