"""
Anomaly-scoring primitives.

Statistical helpers for claiming "this slice is unusual". Any anomaly claim
in the report should compose at least one of these against a comparison
venue's baseline distribution — see primer module 08.
"""

from __future__ import annotations

import pandas as pd


def volume_zscore(volume: pd.Series, *, window: int = 24) -> pd.Series:
    """Per-bin z-score of volume against a trailing rolling baseline.

    Args:
        volume: per-bin volume Series, indexed by bin timestamp (UTC,
            tz-aware DatetimeIndex). Bin width is implicit in the index
            spacing; this function doesn't care whether the bins are
            minutes or hours.
        window: number of trailing bins for the rolling mean and std.
            Choose long enough to capture diurnal seasonality if the
            series hasn't been pre-deseasonalized. `24*7` is the default
            you want for hourly ETH/BTC volume.

    The baseline is the prior `window` observations *excluding* the
    current one — surveillance semantics ("how unusual is this hour
    compared to the prior hours"). The first `window` values are NaN
    (no full prior baseline yet); bins where the rolling std is zero
    return NaN.
    """
    if not isinstance(volume.index, pd.DatetimeIndex):
        raise TypeError(
            f"volume Series must have a DatetimeIndex; got {type(volume.index).__name__}"
        )
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window}")

    prior = volume.shift(1)
    rolling_mean = prior.rolling(window=window, min_periods=window).mean()
    rolling_std = prior.rolling(window=window, min_periods=window).std(ddof=0)
    rolling_std = rolling_std.where(rolling_std > 0)
    return ((volume - rolling_mean) / rolling_std).rename("volume_z")


def burst_score(trades: pd.DataFrame, *, window_seconds: int = 60) -> pd.Series:
    """Per-trade burstiness: count of trades in the trailing window.

    For each trade, returns the count of trades (including itself) in the
    `window_seconds` immediately before its timestamp. High burst values
    clustered into short intervals are a coordination signal.

    Computed via pandas time-based rolling on the trade timestamps.
    Returns a Series aligned positionally with the input (ignoring the
    input's index).
    """
    if "ts" not in trades.columns:
        raise ValueError("trades must include a 'ts' column")
    if window_seconds <= 0:
        raise ValueError(f"window_seconds must be > 0; got {window_seconds}")

    if len(trades) == 0:
        return pd.Series([], name="burst", dtype=int)

    ts = trades["ts"].sort_values().reset_index(drop=True)
    counts = (
        pd.Series(1, index=pd.DatetimeIndex(ts))
        .rolling(f"{window_seconds}s", center=False)
        .sum()
        .astype(int)
    )
    return counts.rename("burst").reset_index(drop=True)
