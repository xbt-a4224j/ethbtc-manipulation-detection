"""Tests for anomaly-scoring primitives."""

from __future__ import annotations

import pandas as pd
import pytest

from challenge.analysis.anomaly import burst_score, volume_zscore


# ---------- volume_zscore -----------------------------------------------------


class TestVolumeZscore:
    def test_constant_baseline_yields_nan_throughout(self) -> None:
        """Constant volume across the prior window has zero std — output is NaN."""
        idx = pd.date_range("2025-07-01", periods=10, freq="h", tz="UTC")
        vol = pd.Series([5.0] * 10, index=idx)
        z = volume_zscore(vol, window=3)
        assert z.iloc[:3].isna().all()  # warmup (need 3 prior obs)
        assert z.iloc[3:].isna().all()  # std == 0 -> NaN by design

    def test_spike_against_steady_baseline_is_strongly_positive(self) -> None:
        """A spike against a non-degenerate baseline (excluding the spike itself) yields a large z."""
        idx = pd.date_range("2025-07-01", periods=10, freq="h", tz="UTC")
        vol = pd.Series([5.0, 6.0, 5.5, 5.2, 5.8, 5.4, 5.6, 5.3, 5.7, 50.0], index=idx)
        z = volume_zscore(vol, window=5)
        # Baseline at position 9 = prior 5 = [5.4, 5.6, 5.3, 5.7, 5.8].
        # That's a tight baseline; spike of 50 is many sigmas out.
        assert z.iloc[-1] > 50.0

    def test_warmup_returns_nan_until_window_prior_observations_exist(self) -> None:
        """The first `window` positions are NaN — no full prior baseline yet."""
        idx = pd.date_range("2025-07-01", periods=6, freq="h", tz="UTC")
        vol = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
        z = volume_zscore(vol, window=3)
        # Position i needs prior `window` observations, so first valid is i=window.
        assert z.iloc[:3].isna().all()
        assert z.iloc[3:].notna().all()

    def test_rejects_non_datetime_index(self) -> None:
        vol = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            volume_zscore(vol, window=2)

    def test_rejects_invalid_window(self) -> None:
        idx = pd.date_range("2025-07-01", periods=3, freq="h", tz="UTC")
        vol = pd.Series([1.0, 2.0, 3.0], index=idx)
        with pytest.raises(ValueError, match="window"):
            volume_zscore(vol, window=1)

    def test_returns_named_series(self) -> None:
        idx = pd.date_range("2025-07-01", periods=5, freq="h", tz="UTC")
        z = volume_zscore(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx), window=2)
        assert z.name == "volume_z"


# ---------- burst_score -------------------------------------------------------


class TestBurstScore:
    def test_uniform_one_per_second_with_60s_window_yields_arithmetic_growth(self) -> None:
        """One trade per second, 60s window: count grows 1,2,3,... up to 60, then plateaus."""
        ts = pd.date_range("2025-07-01", periods=120, freq="1s", tz="UTC")
        trades = pd.DataFrame({"ts": ts, "price": [1.0] * 120, "qty": [1.0] * 120, "side": ["buy"] * 120})
        burst = burst_score(trades, window_seconds=60)
        assert burst.iloc[0] == 1
        assert burst.iloc[59] == 60
        assert (burst.iloc[60:] == 60).all()

    def test_isolated_trade_yields_one(self) -> None:
        """A single trade has burst-score 1 (counts itself)."""
        trades = pd.DataFrame(
            {
                "ts": [pd.Timestamp("2025-07-01", tz="UTC")],
                "price": [1.0],
                "qty": [1.0],
                "side": ["buy"],
            }
        )
        assert list(burst_score(trades)) == [1]

    def test_empty_returns_empty(self) -> None:
        empty = pd.DataFrame({"ts": pd.Series(dtype="datetime64[ns, UTC]"),
                              "price": pd.Series(dtype=float),
                              "qty": pd.Series(dtype=float),
                              "side": pd.Series(dtype=object)})
        out = burst_score(empty)
        assert len(out) == 0
        assert out.name == "burst"

    def test_rejects_missing_ts(self) -> None:
        df = pd.DataFrame({"price": [1.0], "qty": [1.0], "side": ["buy"]})
        with pytest.raises(ValueError, match="ts"):
            burst_score(df)

    def test_rejects_invalid_window(self) -> None:
        trades = pd.DataFrame(
            {
                "ts": pd.date_range("2025-07-01", periods=2, freq="1s", tz="UTC"),
                "price": [1.0, 1.0],
                "qty": [1.0, 1.0],
                "side": ["buy", "sell"],
            }
        )
        with pytest.raises(ValueError, match="window_seconds"):
            burst_score(trades, window_seconds=0)

    def test_handles_duplicate_timestamps(self) -> None:
        """Multiple trades at the same timestamp — each counts toward the burst at that ts."""
        ts = pd.to_datetime(
            ["2025-07-01 00:00:00", "2025-07-01 00:00:00", "2025-07-01 00:00:30"],
            utc=True,
        )
        trades = pd.DataFrame(
            {"ts": ts, "price": [1.0] * 3, "qty": [1.0] * 3, "side": ["buy"] * 3}
        )
        burst = burst_score(trades, window_seconds=60)
        # Two trades at the first ts: burst counts grow 1, 2 over the duplicates,
        # then 3 at the +30s trade since all three fall in the window.
        assert list(burst) == [1, 2, 3]
