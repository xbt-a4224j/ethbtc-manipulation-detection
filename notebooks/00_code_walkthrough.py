# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # Code walkthrough — hands-on tour through everything we built
#
# **How to use this file:**
# - Open in Cursor or VS Code. Each `# %%` block is a runnable cell — shift+enter
#   to execute, output appears inline. (Same convention as Jupyter, no ipynb needed.)
# - PyCharm Pro / JupyterLab: open the paired `00_code_walkthrough.ipynb` for the
#   rendered-notebook view (markdown cells render, plots inline, Mermaid diagrams
#   render in the architecture cell).
# - Or run end-to-end: `uv run python notebooks/00_code_walkthrough.py`.
# - Or paste blocks into `uv run python -i` for a REPL session.
#
# The setup cell below chdirs to the repo root so relative paths like
# `data/eth-btc-trades.csv` resolve the same way no matter which mode you use.
#
# **Goal:** by the end you should be able to explain, on a call, what every
# function in `src/challenge/` does, the math/algorithm behind it, why it's
# shaped that way, and what could go wrong. Each section follows the same
# four-part shape:
#
#   1. **Why this exists** — the problem the function solves.
#   2. **The source, inlined** — `show_source(fn)`, no flipping to source files.
#   3. **Line-by-line walkthrough** — what each block does and why.
#   4. **Live demo + interpretation** — what the output means on this dataset.
#
# Most sections also have an "Experiment" prompt at the end — modify the cell
# and re-run to build intuition.

# %%
# Setup. Run this once. The pandas display options keep wide tables readable.
from __future__ import annotations

import inspect
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Anchor CWD to the repo root so the relative paths below
# ("data/...", "cache/...") resolve the same way in `uv run python ...`,
# in `jupyter`, and in PyCharm/Cursor cell-runners (which default the
# kernel CWD to the notebook's own directory). Walks up until it finds
# pyproject.toml, then chdirs there.
_p = Path.cwd().resolve()
while _p != _p.parent and not (_p / "pyproject.toml").exists():
    _p = _p.parent
if (_p / "pyproject.toml").exists():
    os.chdir(_p)
print(f"working directory: {Path.cwd()}")

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 12)
warnings.filterwarnings("ignore", category=FutureWarning)


def show_source(obj) -> None:
    """Print the source of a function or class with line numbers.

    Used throughout this walkthrough so the function under discussion is
    visible right next to the explanation, no source-file flipping.
    """
    src = inspect.getsource(obj)
    width = len(str(src.count("\n") + 1))
    for i, line in enumerate(src.rstrip("\n").splitlines(), start=1):
        print(f"{i:>{width}}  {line}")


# %% [markdown]
# ## Architecture — who imports who, who calls who
#
# Three concentric layers. `sources/` and `io/` produce data; `analysis/`
# consumes data and produces metrics or figures; scripts and the notebook
# at the outside compose those into investigations.
#
# ```mermaid
# flowchart LR
#   subgraph data[Data + venues]
#     CSV1[data/eth-btc-trades.csv]
#     CSV2[data/eth-btc-orderbooks.csv]
#     KEX[Kraken REST]
#     BEX[Binance REST]
#     PARQ[(cache/&ast;.parquet)]
#   end
#
#   subgraph sources[challenge.sources]
#     LT[csv_loader.load_trades]
#     LO[csv_loader.load_orderbooks]
#     TOB[csv_loader.top_of_book]
#     FK[comparison_venues.fetch_kraken_eth_btc]
#     FB[comparison_venues.fetch_binance_eth_btc]
#   end
#
#   subgraph io[challenge.io]
#     CACHE[cache.cached_parquet]
#   end
#
#   subgraph analysis[challenge.analysis]
#     subgraph manip[manipulation.py]
#       KL[kyle_lambda]
#       RTR[round_trip_rate]
#       TNS[top_n_size_share]
#       IA[inter_arrival_seconds]
#       BSR[buy_sell_ratio]
#     end
#     subgraph anom[anomaly.py]
#       VZ[volume_zscore]
#       BS[burst_score]
#     end
#     PLOT[plot.&ast; helpers]
#   end
#
#   subgraph entrypoints[Scripts + notebook]
#     PULL[scripts/pull_comparison_venues.py]
#     PROV[scripts/provisional_charts.py]
#     CV[scripts/cross_venue_metrics.py]
#     BSF[scripts/buy_side_fingerprint.py]
#     NB[notebooks/00_code_walkthrough.py]
#   end
#
#   CSV1 --> LT
#   CSV2 --> LO
#   LO --> TOB
#   KEX --> FK
#   BEX --> FB
#   FK --> CACHE
#   FB --> CACHE
#   CACHE --> PARQ
#
#   LT -.canonical trades.-> manip
#   LT -.canonical trades.-> anom
#   PARQ -.canonical trades.-> manip
#   PARQ -.canonical trades.-> anom
#
#   PULL --> FK
#   PULL --> FB
#   PULL --> CACHE
#   PROV --> LT
#   PROV --> LO
#   PROV --> TOB
#   PROV --> PLOT
#   PROV --> IA
#   CV --> LT
#   CV --> KL
#   CV --> RTR
#   CV --> TNS
#   CV --> BSR
#   BSF --> LT
#   BSF --> PLOT
#   NB --> LT
#   NB --> LO
#   NB --> TOB
#   NB --> manip
#   NB --> anom
#   NB --> PLOT
# ```
#
# **Reading the diagram:**
# - Solid arrows = direct function calls / imports.
# - Dotted arrows = data flow ("emits a DataFrame in the canonical schema").
# - `cache.cached_parquet` is a read-through cache: on cache miss it calls
#   the wrapped `fetch_*` function; on hit it skips the network entirely.
# - Every analysis function takes the *same* canonical trades DataFrame.
#   That uniform input is why a single helper like `venue_summary` (later
#   in this notebook) can compute the cross-venue table in one call.
#
# **If your renderer doesn't show the Mermaid graph:** copy the block
# above (the lines between the triple-backticks) into mermaid.live for a
# rendered SVG. PyCharm Pro, VS Code Jupyter, and JupyterLab with the
# Mermaid extension all render it inline.

# %% [markdown]
# ## 1. The canonical schema — the contract every function speaks
#
# Every analysis function expects a trades DataFrame with exactly these columns:
#
#     ts:    pd.Timestamp (UTC, tz-aware)
#     price: float
#     qty:   float (positive)
#     side:  str ("buy" or "sell")
#
# Source adapters (`csv_loader`, `comparison_venues`) are responsible for
# producing this shape. Analysis functions assume it and raise loudly if
# anything is missing. This separation is the single biggest reason the
# codebase stays simple — `kyle_lambda` doesn't care whether its input came
# from a CSV, Kraken, or Binance.
#
# **Why "loud failures" matter.** Silent type coercion on a market-data
# pipeline is the worst possible outcome: a wrong dtype propagates through
# every downstream metric, and the resulting numbers look plausible. So the
# adapters validate aggressively and `_validate_schema` in `manipulation.py`
# raises a clean `ValueError` if the trades frame is missing a column.

# %%
from challenge.analysis.manipulation import _validate_schema  # noqa: PLC2701
show_source(_validate_schema)

# %% [markdown]
# **Walking through `_validate_schema`:**
# - `CANONICAL_COLS` is a `frozenset` of the four required column names —
#   immutable so it can't be accidentally mutated by a caller.
# - Set difference: `CANONICAL_COLS - set(trades.columns)` is what's missing
#   on the input. Empty set = good.
# - The error message names *both* the missing columns and what we got, so
#   debugging an upstream adapter mistake takes one error line, not five.
#
# Notice we don't validate dtypes here — the adapters guarantee them. This
# is a deliberate trust boundary: the adapter is the only place we coerce.

# %% [markdown]
# ## 2. `csv_loader.load_trades` — column auto-detection + validation
#
# **Why this exists.** Every CEX exports trade history under different
# column names (`timestamp` / `time` / `exchange_time`, `size` / `qty` /
# `amount`, `B`/`S` / `buy`/`sell` / `True`/`False`). Hard-coding column
# names locks the loader to one source. Auto-detecting against a small
# whitelist plus an explicit `column_map` override is enough to handle
# every real CEX dump in practice.

# %%
from challenge.sources.csv_loader import load_trades
show_source(load_trades)

# %% [markdown]
# **Walking through `load_trades`:**
#
# - **`raw = pd.read_csv(path)`** — pandas does the heavy parsing once. We
#   then *map* its columns into canonical names; we do not re-read.
#
# - **`overrides = column_map or {}`** — the explicit-override pattern. Pass
#   `column_map={"ts": "exchange_time"}` to skip auto-detection for any
#   single column. Anything not overridden falls through to detection.
#
# - **`_detect_column(...)`** does case-insensitive matching against an
#   ordered candidate list (`_TRADES_TS_CANDIDATES = ("ts", "timestamp",
#   "time", ...)`). First match wins. The candidate lists are seeded from
#   what we've seen on Kraken, Binance, Coinbase, Bybit, Deribit dumps.
#
# - **`pd.to_datetime(..., utc=True, errors="coerce")`** — coerces unparseable
#   timestamps to `NaT`, then we count and raise. We deliberately do *not*
#   silently drop them: a single unparseable timestamp probably indicates a
#   schema change that the analyst needs to know about.
#
# - **`_SIDE_NORMALIZATION`** is the side-encoding lookup. It handles the
#   string variants (`buy`/`b`/`bid`), boolean strings (`true`/`1`), and
#   the rare `long`/`short` convention. Anything else triggers a loud
#   failure with a sample of the bad values.
#
# - **`(qty_parsed < 0).any()`** — negative quantities indicate either a
#   broken adapter or a sign-encoded side. Either way, refuse and raise.
#
# - Finally we assemble the canonical DataFrame, sort by `ts`, and reset
#   index. The sort is defensive: pagination merges sometimes drop trades
#   slightly out of order, and downstream code (`inter_arrival_seconds`,
#   the round-trip pairing loop) assumes monotone time.
#
# **Why `_detect_column` raises rather than returning `None`:** failing fast
# at load time, with a useful error, is much cheaper than letting a `None`
# leak through and surface as a `KeyError` ten cells later.

# %%
challenge = load_trades("data/eth-btc-trades.csv")
print(f"shape: {challenge.shape}")
print(f"dtypes:\n{challenge.dtypes}")
print()
print(challenge.head(3))

# %% [markdown]
# **Notice:** the CSV has columns `timestamp,price,size,side`. The loader's
# auto-detection mapped them to canonical names. If detection fails on some
# future CSV, pass `column_map={"ts": "exec_time", ...}`.
#
# **Experiment:** open `data/eth-btc-trades.csv` in your editor. Confirm the
# raw column names. Try renaming `side` to `direction` in a copy of the CSV
# and reload — does auto-detection still find it?

# %% [markdown]
# ## 3. `kyle_lambda` — the price-impact regression in 30 lines
#
# **The economics.** Kyle (1985) modeled a market with informed traders who
# know the asset's true value and uninformed "noise" traders who don't.
# A market maker sees aggregate signed order flow and adjusts price.
# **Lambda (λ) is the slope** of price change on signed flow: how much
# price moves per unit of net buying. High lambda = each trade carries
# information; low lambda = flow is noise; near-zero or negative lambda is
# either a wash-dominated market or a price-trend going against the flow.
#
# **The specific estimator we use.** Per fixed-width time bin (default 60s):
#
#   * **signed flow** *q*<sub>t</sub> = Σ qty<sub>buy</sub> − Σ qty<sub>sell</sub>
#   * **price change** Δp<sub>t</sub> = last_price<sub>t</sub> − first_price<sub>t</sub>
#   * **λ** = OLS slope of Δp on q  =  Cov(q, Δp) / Var(q)
#
# That's exactly the closed-form OLS slope for a univariate regression.
# We could also use `np.polyfit` or `scipy.stats.linregress`, but with no
# scipy in the deps, the two-line cov/var version is the cleanest.
#
# **Why bin?** Tick-by-tick the noise dominates the signal — every trade is
# a separate observation of a tiny price move. Aggregating into bins
# averages the noise; bin width is a hyperparameter we sweep over.
#
# **What "robust" looks like.** A real lambda finding shows the same sign
# and order of magnitude across bin choices (15s, 30s, 60s, 120s, 300s).
# If the sign flips with bin width, the result is fragile — probably noise.

# %%
from challenge.analysis.manipulation import kyle_lambda
show_source(kyle_lambda)

# %% [markdown]
# **Walking through `kyle_lambda`:**
#
# - **`if len(trades) == 0: return 0.0`** — semantic default. An empty
#   window has no informational content; returning 0 is the right neutral.
#
# - **`df["bin"] = df["ts"].dt.floor(f"{bin_seconds}s")`** — anchor every
#   trade to the *start* of its bin. Two trades 23 seconds apart end up in
#   the same 60-second bin iff they cross the same minute-boundary multiple.
#   `floor` is the canonical pandas idiom for fixed-width time binning.
#
# - **`sign = df["side"].str.lower().map({"buy": 1, "sell": -1})`** —
#   converts side to ±1 so signed_qty is just qty × sign. The `.str.lower()`
#   handles any upstream case inconsistency. Anything unmapped becomes NaN
#   and triggers the loud failure on the next line.
#
# - **`groupby("bin").agg(signed_volume=..., first_price=..., last_price=...)`**
#   — single pass to compute all three per-bin quantities. `first` and
#   `last` use insertion order, which is sorted by ts thanks to the upstream
#   adapter. **This is why the sort in the loader matters.**
#
# - **`x.size < 2 or float(np.var(x)) < 1e-12`** — guard against the OLS
#   collapse cases. With one bin you can't fit a slope; with zero variance
#   in x, the regression is undefined (division by zero). Returning 0 in
#   either case is consistent with the "no informational flow" semantics.
#
# - **`cov_xy / var_x`** — the closed-form slope. We use `np.cov(bias=True)`
#   (population covariance, divide by N) and `np.var` (also population) so
#   the N's cancel; bias=False would still work but introduces an unnecessary
#   N/(N-1) factor that doesn't change the slope.
#
# **Why no intercept?** A Kyle regression is conventionally fit *with* an
# intercept and we report the slope. Mathematically the slope = Cov/Var
# whether or not you include an intercept (the intercept absorbs the
# means). So we skip the intercept arithmetic and go directly to the slope.
#
# **Alternative we rejected:** `scipy.stats.linregress` would give us the
# slope plus a t-stat, p-value, and stderr in one call. But scipy is a
# 70MB transitive dep we don't otherwise need. The slope alone is
# sufficient for the report; if we wanted CIs we'd bootstrap manually.

# %%
# Default 60-second bins.
lam = kyle_lambda(challenge, bin_seconds=60)
print(f"challenge venue lambda (60s bins): {lam:.6e}")

# %% [markdown]
# **The bin-size sweep.** No single bin choice is canonical. Sweep across a
# reasonable range; if the result is *monotonic* (same sign, similar order
# of magnitude), the finding is robust. If it flips sign or changes 100x,
# you have a noise problem.

# %%
print("bin_seconds | lambda")
for bs in (15, 30, 60, 120, 300):
    print(f"  {bs:>6}     | {kyle_lambda(challenge, bin_seconds=bs):+.4e}")

# %% [markdown]
# **What you should see:** all five values are negative (price moves *opposite*
# the signed flow direction here — buy pressure correlates with falling price,
# the structural impossibility we flagged in chat). Magnitude is small;
# all values within an order of magnitude of each other — the finding is robust.
#
# **Experiment:** force a "balanced wash" subset and verify lambda collapses
# to ~0. Take 100 trades, alternate buy/sell, constant price, recompute.

# %%
n = 100
wash = pd.DataFrame({
    "ts":    pd.date_range("2025-09-01", periods=n, freq="1s", tz="UTC"),
    "price": [1.0] * n,
    "qty":   [10.0] * n,
    "side":  (["buy", "sell"] * 50)[:n],
})
print(f"synthetic wash lambda: {kyle_lambda(wash, bin_seconds=10):.6f}")

# %% [markdown]
# ## 4. `round_trip_rate` — greedy nearest-time pairing
#
# **The economics.** A wash account places a buy, then a sell of approximately
# the same qty within a short window. Same beneficial owner on both sides;
# net P&L ≈ zero (modulo fees and slippage); volume metric ticks up.
# Detection: for each unpaired trade, find the next opposite-side trade with
# matching qty inside the window. `paired_volume / total_volume` is the rate.
#
# **Why "greedy" and not "optimal"?** The optimal pairing problem (maximum
# matching on a weighted bipartite graph) is polynomial but expensive — and
# *the question we're trying to answer doesn't need the optimum*. We just
# need a rate that's monotonic in wash intensity. Greedy nearest-time
# pairing has that property and runs in O(n²) worst case, which is fine
# for the scales we encounter (≤ ~10⁵ trades per venue per window).

# %%
from challenge.analysis.manipulation import round_trip_rate
show_source(round_trip_rate)

# %% [markdown]
# **Walking through `round_trip_rate`:**
#
# - **`df = trades.sort_values("ts").reset_index(drop=True)`** — defensive
#   sort. The pairing inner loop assumes index `i+1, i+2, ...` is monotone
#   in time so it can `break` early when the deadline passes.
#
# - **NumPy column extraction** (`sides`, `qtys`, `ts`, `paired`) — we drop
#   to numpy arrays before the inner loop. The hot path is a Python double
#   loop; pandas access in the loop body would be 10–100x slower.
#
# - **`paired = np.zeros(n, dtype=bool)`** — flag array tracking which
#   trades have already been consumed by a pairing.
#
# - **Outer loop** walks every trade. If already paired, skip.
#
# - **`target_side = "sell" if sides[i] == "buy" else "buy"`** — we only
#   pair against the opposite side. A wash is buy→sell or sell→buy.
#
# - **Inner loop** scans forward looking for a match. The `if ts[j] >
#   deadline: break` short-circuit is what keeps the algorithm fast in
#   practice — you only scan as far as the time window before bailing.
#
# - **`abs(qtys[j] - qtys[i]) / qtys[i] > qty_tolerance`** — relative qty
#   tolerance. Default 1% accommodates rounding artifacts (a buy of 0.5
#   matched against a sell of 0.4995). Set to 0.0 for exact-only pairing.
#
# - **`paired_qty += qtys[i] + qtys[j]`** — both legs count toward paired
#   volume. So a venue where every trade round-trips would have rate = 1.0,
#   not 2.0.
#
# - **`break` after a match** — greedy: take the first matching forward
#   trade, don't keep looking. Optimal pairing might delay this match and
#   take a later one; we don't.
#
# **Edge cases handled:**
# - `n < 2` → 0.0 (can't round-trip a single trade).
# - `qtys[i] <= 0` → skip (can't compute relative tolerance).
# - `total_qty == 0` → 0.0 (no volume, no rate).
#
# **Why a relative tolerance and not an absolute one?** ETHBTC trade sizes
# span six orders of magnitude. An absolute tolerance of 0.001 would make
# every micro-trade pair with everything and miss every macro-trade pair.

# %%
# Three windows × three tolerances. The challenge venue's pattern across the grid:
print(f"{'window':>8} | tol=0.0     tol=0.01    tol=0.05")
for ws in (15, 60, 300):
    row = [round_trip_rate(challenge, window_seconds=ws, qty_tolerance=t)
           for t in (0.0, 0.01, 0.05)]
    print(f"{ws:>5}s   | {row[0]:.4f}      {row[1]:.4f}      {row[2]:.4f}")

# %% [markdown]
# **What you should see:** the rates are all very low. Why? Because the buys
# (mean size ~234) and sells (mean size ~0.008) have nothing close to matching
# quantities. Round-trip pairing requires *opposite-side qty match*; this
# venue's sides are different orders of magnitude, so almost nothing pairs.
#
# That tells us: round_trip_rate doesn't pick up *this* venue's anomaly —
# it would catch wash where both sides are small and matched, not asymmetric
# manufactured volume. The headline finding here is the buy_sell_ratio +
# kyle_lambda combination, not round_trip_rate.
#
# That's a real lesson: **no single signal is sufficient**. Different patterns
# show up on different metrics.

# %% [markdown]
# ## 5. `top_n_size_share` — size concentration
#
# **The economics.** Wash bots ship from a small fixed set of sizes — the
# qty parameter is hard-coded into the bot's order generator. Legitimate
# flow is dispersed: hundreds of independent traders sending hundreds of
# different sizes. The volume share concentrated in the top-N most common
# sizes is therefore much higher on a wash venue than on a clean one.
#
# **The rounding parameter is doing real work.** Some venues report sizes
# at 8-decimal precision but the bot generates them at 4. Without rounding,
# 0.50000001 and 0.50000000 look different to the size-share calculation.
# Default `rounding_decimals=6` — collapses the 7th–8th decimal of float
# noise but preserves 6-decimal intent. Tighten to 4 if you're seeing
# fragmented sizes that should be the same.

# %%
from challenge.analysis.manipulation import top_n_size_share
show_source(top_n_size_share)

# %% [markdown]
# **Walking through `top_n_size_share`:**
#
# - **`sizes = trades["qty"].astype(float).round(rounding_decimals)`** — coerce
#   to float (in case the loader produced Decimal or object dtype) and round
#   in one step. The `.round(6)` on a float Series is a vectorized numpy call.
#
# - **`by_size = sizes.groupby(sizes).sum().sort_values(ascending=False)`** —
#   `groupby` on the Series itself groups identical (rounded) sizes. `.sum()`
#   gives total qty per size. `.sort_values` puts the heaviest sizes first.
#
# - **`return float(by_size.head(n).sum() / total)`** — top-N qty divided by
#   total qty. Cast to native float so downstream code can compare against
#   plain numbers without numpy-scalar surprises.
#
# **Edge cases handled:**
# - `n <= 0` → loud `ValueError` (a top-zero-sizes share is a logic error).
# - `total <= 0` → 0.0 (no volume, no concentration).
#
# **Alternative we rejected:** the entropy of the size distribution would
# also detect concentration. But entropy is harder to interpret on a call
# ("the entropy is 4.2") whereas "the top 10 sizes account for 82% of
# volume" is concrete. Same information, plainer language.

# %%
print(f"{'n':>4} | top-n size share (challenge)")
for n in (1, 3, 5, 10, 25, 50):
    print(f"{n:>4} | {top_n_size_share(challenge, n=n):.4f}")

# %% [markdown]
# **What you should see:** even at n=1, the share is small. The 720 buys aren't
# a few repeated sizes — they're varied (645 distinct exact sizes / 720 trades
# = 89.6% unique). So this venue isn't fingerprint-style bot wash; the bot is
# producing varied sizes via a generator function. Different surface area
# than what `top_n_size_share` is tuned for.

# %% [markdown]
# ## 6. `buy_sell_ratio` — count-weighted vs. volume-weighted
#
# **Why two versions matter.** Buy/sell *count* and buy/sell *volume* can
# disagree wildly. If 700 small buys (avg 0.01) are matched by 100 enormous
# sells (avg 70), the count ratio is 7:1 buy-heavy but the volume ratio is
# balanced. The challenge venue is the opposite: 720 buys average ~234, 125
# sells average ~0.008 — so the count ratio is mildly skewed (5.76×) but the
# volume ratio is **175,000×** buy-heavy. The gap between the two is the
# diagnostic.
#
# **The function returns volume-weighted by default**, because volume is what
# the venue reports as its "trading activity" and what the wash regime is
# trying to inflate. The count version is computed inline in the demo cell.

# %%
from challenge.analysis.manipulation import buy_sell_ratio
show_source(buy_sell_ratio)

# %% [markdown]
# **Walking through `buy_sell_ratio`:**
#
# - **`if len(trades) == 0: return 0.5`** — neutral default for an empty
#   window. The 0.5 comes from "balanced market" semantics; a 0.0 default
#   would falsely look like extreme sell-pressure.
#
# - **`qty = trades["qty"].astype(float)`** — defensive cast; some adapters
#   land a Series with object dtype.
#
# - **`if total_volume <= 0: return 0.5`** — same neutral semantics; also
#   guards against a divide-by-zero in the next line.
#
# - **`side = trades["side"].astype(str).str.lower()`** — normalize for the
#   filter below. We don't validate side values here because the loader
#   already did; if something snuck through, the buy_volume sum will just
#   be a subset of total and we'll return a number in [0,1].
#
# - **`return buy_volume / total_volume`** — share, not ratio. 0.5 = balanced,
#   1.0 = all buys, 0.0 = all sells. Some literature uses "buy/sell" as a
#   ratio (1.0 = balanced, infinite at all-buy); we use the share form because
#   it's bounded and easier to plot.
#
# **Why no symmetry test?** We could verify `buy_sell_ratio(df) +
# buy_sell_ratio(df, count_only_sells=True) == 1.0`, but there's no
# `count_only_sells` flag — the function is single-purpose. The unit test
# checks the share-form invariant directly.

# %%
n_buy   = (challenge["side"] == "buy").sum()
n_sell  = (challenge["side"] == "sell").sum()
n_total = len(challenge)

bsr_count  = n_buy / n_total
bsr_volume = buy_sell_ratio(challenge)

print(f"buy share by COUNT:  {bsr_count:.6f}  ({n_buy} buys / {n_total} total)")
print(f"buy share by VOLUME: {bsr_volume:.6f}  ({challenge.loc[challenge.side=='buy','qty'].sum():.2f} buy qty)")
print(f"gap = {bsr_volume - bsr_count:.6f} — the size asymmetry doing the work")

# %% [markdown]
# **What you should see:** count-weighted buy share ≈ 0.85, volume-weighted
# ≈ 0.99999+. The 14-percentage-point gap is the headline anomaly: buys aren't
# just more frequent, they're orders of magnitude larger.
#
# This is the structural "tell" no single primer-listed signal captured cleanly.
# Pair it with kyle_lambda < 0 and you have a real finding.

# %% [markdown]
# ## 7. `inter_arrival_seconds` — the cadence fingerprint
#
# **The economics.** Bots fire on a clock; humans fire on news. A bot
# generating one buy every ~5 minutes produces a sharp peak in the
# inter-arrival distribution at ~300 seconds. Real markets produce an
# exponential-ish distribution with most arrivals sub-second and a long
# tail. The shape is enough to fingerprint bot activity, even when the
# per-trade sizes are varied (which is why this metric caught the
# challenge venue's anomaly when `top_n_size_share` didn't).

# %%
from challenge.analysis.manipulation import inter_arrival_seconds
show_source(inter_arrival_seconds)

# %% [markdown]
# **Walking through `inter_arrival_seconds`:**
#
# - **`if len(trades) < 2`** — need at least two timestamps for a difference.
#   Returning an empty Series (rather than NaN-padded) keeps downstream code
#   straightforward.
#
# - **`sorted_ts = trades["ts"].sort_values().reset_index(drop=True)`** —
#   defensive sort, again. The loader sorts, but if a caller filters or
#   concatenates trades from multiple sources, the order can break.
#
# - **`deltas = sorted_ts.diff().dt.total_seconds()`** — `diff()` on a
#   datetime Series gives a Timedelta Series; `.dt.total_seconds()` flattens
#   to float seconds. `diff()` produces NaN for the first row.
#
# - **`return deltas.iloc[1:].rename(...).reset_index(drop=True)`** — drop
#   the leading NaN, rename for plot legends, reset index.
#
# **Returns are length n-1.** That's standard for a difference operation.
# Don't try to pair the output positionally with the input — the i-th
# delta is the gap *between* trade i and trade i+1.
#
# **Aggregation idioms** (used downstream): `.median()` for the headline
# cadence number; `.quantile([0.05, 0.5, 0.95])` for the distribution shape;
# `.value_counts(bins=...)` for histogram-bucketed counts. The notebook's
# IA chart uses `np.logspace` log-binning to render the multi-decade range.

# %%
ia = inter_arrival_seconds(challenge)
print("inter-arrival (seconds) summary:")
print(ia.describe(percentiles=[0.05, 0.5, 0.95]).round(2))

# %% [markdown]
# **What you should see:** median ~5 minutes (300s); p95 ~10 minutes; max
# tens of minutes. Compare to Kraken's ~0.21s median and Binance's ~0.008s
# median (computed in the cross-venue table below) — the challenge venue
# is multiple orders of magnitude slower.

# %% [markdown]
# ## 8. `anomaly.volume_zscore` and `burst_score`
#
# These two together cover "is *this* slice unusual?" — the surveillance
# question. `volume_zscore` answers it for time-binned volume; `burst_score`
# answers it for trade-count clusters at the per-trade level.

# %%
from challenge.analysis.anomaly import burst_score, volume_zscore
show_source(volume_zscore)

# %% [markdown]
# **Walking through `volume_zscore`:**
#
# - **DatetimeIndex check + `window >= 2` check** — defensive type guards.
#   A non-datetime index would silently roll across calendar weeks the
#   wrong way; window=1 has zero baseline observations.
#
# - **`prior = volume.shift(1)`** — the key surveillance trick. The baseline
#   for time t is the prior `window` observations *not including t itself*.
#   Without `shift(1)`, t's own value contaminates its baseline and the
#   z-score regresses toward zero whenever t is large.
#
# - **`prior.rolling(window=window, min_periods=window).mean()`** — rolling
#   mean. `min_periods=window` means the first `window` outputs are NaN
#   (no full baseline yet). We don't return partial-window means because
#   they have wildly inflated variance.
#
# - **`.std(ddof=0)`** — population std, not sample std. This is a
#   convention choice; either works for relative-anomaly-ranking purposes.
#   Population std is slightly less noisy at small `window`.
#
# - **`rolling_std.where(rolling_std > 0)`** — guards against zero-variance
#   windows (constant volume). NaN propagates to the final z-score, which
#   is the right answer ("undefined").
#
# - **Final z** = (current − rolling_mean) / rolling_std. The output Series
#   is renamed `volume_z` for cleaner column display in plots.
#
# **What "window=24" means for hourly bins.** Each hour is z-scored against
# the prior 24 hours — one full diurnal cycle. For a longer baseline use
# `window=24*7`. The default is conservative because the challenge dataset
# is only 72 hours total; longer windows would have too few full-baseline
# observations to be useful.
#
# **Caveat noted in the dataset reality check:** with only 3 days of data,
# a `window=24` baseline still overlaps the same diurnal phase, so the
# z-scores are exploratory only. Don't put them in the report without
# calibration against a longer comparison-venue baseline.

# %%
hourly_vol = challenge.set_index("ts")["qty"].resample("h").sum()
z = volume_zscore(hourly_vol, window=24)
print("hourly volume z-score — top 5 most-anomalous hours:")
print(z.dropna().nlargest(5).round(2))

# %%
show_source(burst_score)

# %% [markdown]
# **Walking through `burst_score`:**
#
# - **Schema check + `window_seconds > 0`** — fail loud on bad input.
#
# - **`ts = trades["ts"].sort_values().reset_index(drop=True)`** — same
#   defensive sort; the rolling time-window depends on monotone time.
#
# - **`pd.Series(1, index=pd.DatetimeIndex(ts))`** — clever idiom: a Series
#   of ones indexed by trade timestamps. Now `.rolling("60s").sum()` over
#   that Series is "count of trades in the trailing 60s for each row".
#
# - **`.rolling(f"{window_seconds}s", center=False)`** — pandas time-based
#   rolling. `center=False` means the window is *trailing* (left-closed,
#   includes current). Each output is "count of timestamps in the window
#   ending at this row's timestamp."
#
# - **`.astype(int)`** — counts are integers; pandas defaults to float for
#   rolling sums.
#
# - **`reset_index(drop=True)`** — drop the timestamp index and return a
#   plain integer Series, positionally aligned with the (sorted) input.
#
# **Why per-trade rather than per-bin?** A burst metric is most useful at
# the single-trade granularity — you want to ask "is *this* trade part of
# a coordinated cluster?" If you wanted per-bin counts, just do a 60s
# resample.

# %%
burst = burst_score(challenge, window_seconds=60)
print(f"burst (trades in trailing 60s) — p50: {burst.median()}, p95: {burst.quantile(0.95)}, max: {burst.max()}")

# %% [markdown]
# ## 9. `load_orderbooks` and `top_of_book` — long-format book + top metrics
#
# **Why long format.** The supplied CSV stores each snapshot as one row
# with the full ladder embedded as a Python-literal list-of-dicts. That
# format is fine for storage but useless for analysis — you can't filter,
# group, or pivot it. The first job of `load_orderbooks` is to *explode*
# each snapshot into one-row-per-(ts, side, level) and normalize the level
# numbering so level 0 is always the top of book.
#
# **Why a separate `top_of_book` helper.** The mid/spread/imbalance
# definitions are small but error-prone (off-by-one mid, wrong sign on
# imbalance). Defining them in one place means every microstructure metric
# in the report agrees on the same definitions.

# %%
from challenge.sources.csv_loader import load_orderbooks, top_of_book
show_source(load_orderbooks)

# %% [markdown]
# **Walking through `load_orderbooks`:**
#
# - **Column existence check** — fail loud if the CSV doesn't have
#   `timestamp`/`asks`/`bids`. Cheaper than letting `KeyError` surface
#   later.
#
# - **Timestamp parsing** with `errors="coerce"` and a count-and-raise on
#   any NaT. Same idiom as `load_trades`.
#
# - **Per-snapshot loop** — yes it's a Python for-loop, not a vectorized
#   pandas operation. We have ~200 snapshots so it's fine. Vectorizing the
#   ast.literal_eval call wouldn't help; it's the bottleneck per snapshot
#   regardless.
#
# - **`ast.literal_eval(asks_raw)`** — `eval`'s safe cousin. Parses Python
#   literal strings (lists, dicts, strings, numbers, tuples, sets, bools,
#   None) and *only* those — no function calls, no attribute access, no
#   imports. Safe to call on untrusted input. We catch `ValueError` /
#   `SyntaxError` and re-raise with the snapshot index for debugging.
#
# - **The sort step** — `asks_sorted = sorted(asks, key=lambda x:
#   float(x["price"]))` (ascending) and `bids_sorted = sorted(bids, key=lambda
#   x: -float(x["price"]))` (descending). This is *normalization*: the
#   source CSV claims "level 0 is top of book" but doesn't guarantee the
#   levels are sorted consistently. We sort defensively so level 0 is
#   always the lowest ask and the highest bid.
#
# - **The row emission** — we emit one row per (snapshot, side, level) into
#   a plain Python list, then build the DataFrame at the end. Building one
#   DataFrame is much faster than `pd.concat`-ing per snapshot.
#
# - **Empty case** returns the canonical schema with correct dtypes — so
#   downstream code (which expects datetime64[ns, UTC] etc.) doesn't break.
#
# **Why `ast.literal_eval` and not `json.loads`?** Python literal lists use
# single quotes by default; JSON requires double quotes. The challenge CSV
# has single quotes; rather than string-replace them (which can corrupt
# nested strings) we just use `ast.literal_eval`.

# %%
long_book = load_orderbooks("data/eth-btc-orderbooks.csv")
print(f"long-format rows: {len(long_book)} ({long_book['ts'].nunique()} snapshots × ~{len(long_book) // long_book['ts'].nunique()} per snapshot)")
print(long_book.head(3))

# %%
show_source(top_of_book)

# %% [markdown]
# **Walking through `top_of_book`:**
#
# - **Column existence check** — same defensive validation idiom.
#
# - **`long_book.loc[long_book["level"] == 0]`** — filter to top-of-book
#   rows only. Levels >= 1 are deeper in the book; they don't belong here.
#
# - **`.pivot(index="ts", columns="side", values=["price", "qty"])`** —
#   pivot from long to wide. Result has a MultiIndex on columns:
#   `(price, ask), (price, bid), (qty, ask), (qty, bid)`. Each row is one
#   snapshot.
#
# - **`top.columns = [f"{side}_{field}" for field, side in top.columns]`** —
#   flatten the MultiIndex into simple `bid_price` / `ask_price` etc.
#   strings. Note the `for field, side` order: the MultiIndex is `(price,
#   bid)` so the unpacking yields `field=price, side=bid` and the joined
#   string is `bid_price`. Subtle, but it matches the rest of the codebase.
#
# - **Sanity check** — if any of `bid_price/ask_price/bid_qty/ask_qty` is
#   missing after the pivot, one side of the book was empty in the level-0
#   rows. That's a data integrity error; raise with a useful message.
#
# - **Mid, spread, imbalance** — the three canonical microstructure
#   metrics. Mid = midpoint of best bid/ask. Spread (in basis points) =
#   (ask − bid) / mid × 10⁴. Imbalance = bid_qty / (bid_qty + ask_qty),
#   so 0.5 is balanced, > 0.5 is bid-heavy.
#
# - **`.where(denom > 0)`** — guards against snapshots with zero qty on
#   both sides (degenerate book). NaN-propagates instead of dividing by
#   zero. The chart helpers handle NaN gracefully.

# %%
top = top_of_book(long_book)
print("top-of-book metrics summary:")
print(top[["spread_bps", "imbalance", "bid_qty", "ask_qty"]].describe().round(4))

# %% [markdown]
# **What you should see:** spreads averaging ~90 bps (vs. expected 1–4 bps
# on Kraken/Binance). Imbalance heavily ask-tilted (mean ~0.29 — only 29%
# of top-of-book qty is bids). These are the headline microstructure
# observations.
#
# **Experiment:** pick a single snapshot in the middle of the window and look
# at its full depth profile.

# %%
mid_ts = long_book["ts"].quantile(0.5, interpolation="nearest")
snap = long_book.loc[long_book["ts"] == mid_ts]
print(f"snapshot at {mid_ts}")
print(f"  bid levels: {(snap['side']=='bid').sum()}, ask levels: {(snap['side']=='ask').sum()}")
print(f"  total bid qty: {snap.loc[snap.side=='bid','qty'].sum():.4f}")
print(f"  total ask qty: {snap.loc[snap.side=='ask','qty'].sum():.4f}")

# %% [markdown]
# ## 10. `cached_parquet` — the read-through cache
#
# **Why this exists.** Comparison-venue pulls (Kraken, Binance) take
# ~60 seconds per venue and burn rate-limit headroom. We need to iterate
# on the analysis dozens of times. Solution: write the result to a local
# parquet file the first time; on subsequent calls, read parquet directly
# and skip the network. Parquet preserves dtypes (datetimes stay
# datetimes), compresses well (zstd ~5× CSV), and reads ~50× faster than
# CSV at our scale.

# %%
from challenge.io.cache import cached_parquet
show_source(cached_parquet)

# %% [markdown]
# **Walking through `cached_parquet`:**
#
# - **`name`** — filename stem, no extension. Use a descriptive name
#   including source and date range so the cache is self-documenting.
#
# - **`build`** — zero-arg callable producing the DataFrame on cache miss.
#   Why a callable instead of a DataFrame? Because we want lazy evaluation
#   — on a cache hit we never call `build`, so we never hit the network.
#   Pass the function as `lambda: fetch_kraken_eth_btc(...)`.
#
# - **`overwrite=True`** — force a rebuild. Use this when the underlying
#   data has changed but the cache key hasn't (e.g. you pulled the same
#   window twice and want fresh data).
#
# - **`compression="zstd"`** — modern default. `snappy` is faster at
#   compression but bigger; `gzip` is smaller but slower. zstd hits the
#   sweet spot at our scale.
#
# - **`index=False`** — we always reset the index before saving. The
#   loaders all return a 0..n-1 RangeIndex which is content-free; saving
#   it is a waste of bytes.
#
# **Alternative we rejected:** a richer cache with TTL, hashing the
# `build` function, key compositing across multiple parameters. We don't
# need any of that — the cache key is the human-readable name, and we
# manually `--refresh` when we need to re-pull.

# %% [markdown]
# ## 11. `fetch_kraken_eth_btc` and `fetch_binance_eth_btc` — REST adapters
#
# **The job.** Both functions: walk a public REST endpoint, paginate over
# the requested window, normalize the response into the canonical trades
# schema. Both endpoints are unauthenticated for trade history. Both have
# pagination limits (Kraken: nanosecond `since` cursor; Binance: 1-hour
# windows × 1000 trades per call). The two implementations look different
# because the endpoints are different — but they hit the same canonical
# output.

# %%
from challenge.sources.comparison_venues import (
    fetch_binance_eth_btc,
    fetch_kraken_eth_btc,
)
show_source(fetch_kraken_eth_btc)

# %% [markdown]
# **Walking through `fetch_kraken_eth_btc`:**
#
# - **`owned_client = client is None; client = client or httpx.Client(...)`** —
#   the dependency-injection-with-default pattern. Caller can pass a shared
#   client (better connection pooling) or accept the default. We track
#   ownership so the `finally` block only closes a client we created.
#
# - **`since_ns = int(start_ts.timestamp() * 1_000_000_000)`** — Kraken's
#   `since` parameter is *nanoseconds* since epoch (not ms or s). Get this
#   wrong and the endpoint silently returns trades from the wrong era.
#
# - **`for _ in range(max_pages)`** — bounded loop with a defensive cap.
#   Without it, a misbehaving cursor (Kraken sometimes returns the same
#   `last` value twice) would infinite-loop.
#
# - **`r.raise_for_status(); body = r.json()`** — separate HTTP error
#   handling from JSON-payload error handling. Kraken returns HTTP 200 on
#   logical errors; the error is in the JSON body's `"error"` key. Both
#   need to be checked.
#
# - **`new_since = int(result["last"])`** — Kraken's pagination cursor.
#   Each response returns the latest trade's nanosecond timestamp; pass it
#   back as `since` to get the next page.
#
# - **`if new_since <= since_ns: break`** — defensive halt. If the cursor
#   doesn't advance, we'd infinite-loop; better to break early and accept
#   the trades we got.
#
# - **`if latest_ts >= end_seconds: break`** — Kraken returns trades in
#   *seconds* in the row payload, not nanoseconds. So we compare against
#   `end_seconds`, not `since_ns`. (This off-by-unit is a real Kraken
#   gotcha.)
#
# - **The `for/else` block** — Python's underused `else` clause on a `for`
#   loop runs *only if the loop exhausts without break*. So hitting
#   `max_pages` raises rather than silently truncating. If the user
#   requests a window so large that it needs more than 10,000 pages,
#   that's a sign their cap should be raised, not that we should silently
#   drop data.
#
# - **The DataFrame assembly** — Kraken's row format is a positional list
#   `[price, qty, ts_s, side_raw, ordertype, misc, trade_id]`. Map to
#   columns by position, then coerce types and the side encoding (`b`/`s`).
#
# - **`df = df.loc[(df["ts"] >= start_ts) & (df["ts"] < end_ts)]`** — the
#   `since` cursor is inclusive of trades *after* the cursor, but we want
#   exact `[start, end)` bounds. Trim post-pagination.
#
# - **`finally: if owned_client: client.close()`** — only close the client
#   if we created it. If the caller passed one in, they own its lifetime.

# %%
show_source(fetch_binance_eth_btc)

# %% [markdown]
# **Walking through `fetch_binance_eth_btc`:**
#
# - **`BINANCE_BASE = "https://data-api.binance.vision"`** — Binance has
#   *two* public hosts. `api.binance.com` is the production endpoint but
#   returns HTTP 451 to US IPs. `data-api.binance.vision` is their
#   read-only public-data subdomain that serves the same `/api/v3/*`
#   schema and *is* reachable from the US. If you're outside the US you
#   can swap to `api.binance.com`.
#
# - **`cursor_ms = ...; end_ms = ...`** — Binance times are *milliseconds*
#   since epoch (not nanoseconds; not seconds). Different from Kraken.
#
# - **`while cursor_ms < end_ms`** — loop until we've covered the window.
#
# - **`chunk_end = min(cursor_ms + BINANCE_CHUNK_MS - 1, end_ms - 1)`** —
#   Binance allows up to a 1-hour `(startTime, endTime)` window per call.
#   We chunk the requested window into 1-hour pieces.
#
# - **`if not chunk: cursor_ms = chunk_end + 1; continue`** — empty hour:
#   advance to the next hour and try again. Some hours have no trades; we
#   shouldn't get stuck.
#
# - **`if len(chunk) >= 1000: cursor_ms = int(chunk[-1]["T"]) + 1`** — the
#   1000-trade ceiling case. We may have gotten only the *first 1000* of
#   the trades in this hour. Advance the cursor to one ms past the last
#   trade we got and re-pull, *without* moving past the chunk_end.
#
# - **`else: cursor_ms = chunk_end + 1`** — fewer than 1000 trades means we
#   got the whole hour; jump to the next hour.
#
# - **`df["side"] = df["m"].map({True: "sell", False: "buy"})`** — Binance
#   encodes side as `isBuyerMaker`: True means the maker was the buyer, so
#   the *taker* (the aggressor, which is what we want as "side") was the
#   seller. False means the taker was the buyer. Easy to invert by accident.
#
# - **Final trim and sort** — same as Kraken: `[start, end)`, sorted,
#   reset index.
#
# **Why the two adapters look so different.** Endpoints differ. Kraken's
# `since` cursor is nanoseconds and returns one big page-by-page stream;
# Binance windows in milliseconds with a 1000-row ceiling per chunk. Each
# adapter is a thin wrapper around the venue's actual semantics; we don't
# pretend the endpoints are the same.

# %%
KRAKEN_PARQ  = Path("cache/kraken_xethxxbt_20250901_20250904.parquet")
BINANCE_PARQ = Path("cache/binance_ethbtc_20250901_20250904.parquet")

kraken  = pd.read_parquet(KRAKEN_PARQ)  if KRAKEN_PARQ.exists()  else None
binance = pd.read_parquet(BINANCE_PARQ) if BINANCE_PARQ.exists() else None

if kraken is None or binance is None:
    print("Cache missing — run: uv run python scripts/pull_comparison_venues.py")
else:
    print(f"kraken:  {len(kraken):>7,} trades, {kraken['ts'].min()} -> {kraken['ts'].max()}")
    print(f"binance: {len(binance):>7,} trades, {binance['ts'].min()} -> {binance['ts'].max()}")

# %% [markdown]
# **Experiment:** delete one of the parquet files and re-run the pull script
# with `--refresh` — watch it re-fetch (Kraken takes ~60s, Binance ~30s).
# Then run again — instant cache hit.

# %% [markdown]
# ## 12. The cross-venue summary table — the report's anchor exhibit
#
# This is the punchline of the whole codebase. Every detection helper is a
# pure function on the canonical trades frame, so the cross-venue table
# falls out as a one-liner over the three venues.

# %%
def venue_summary(name: str, df: pd.DataFrame) -> dict:
    """Compute the headline metrics for one venue."""
    return {
        "venue": name,
        "trades": len(df),
        "bsr_count":  float((df["side"] == "buy").mean()),
        "bsr_volume": buy_sell_ratio(df),
        "kyle_lambda_60s": kyle_lambda(df, bin_seconds=60),
        "round_trip_60s_pct1": round_trip_rate(df, window_seconds=60, qty_tolerance=0.01),
        "top_10_size_share":   top_n_size_share(df, n=10),
        "ia_median_seconds":   inter_arrival_seconds(df).median(),
    }

if kraken is not None and binance is not None:
    rows = [
        venue_summary("challenge", challenge),
        venue_summary("kraken",    kraken),
        venue_summary("binance",   binance),
    ]
    table = pd.DataFrame(rows).set_index("venue")
    # Round for display; preserve precision in the underlying frame.
    print(table.round(6).T.to_string())

# %% [markdown]
# **How to read this table:**
#
# - `bsr_count` vs `bsr_volume`: gap on challenge is the headline.
#   Kraken/Binance have nearly-identical count and volume buy shares (mean
#   trade sizes are stable across sides). Challenge has a 14-pp gap.
# - `kyle_lambda_60s`: positive on healthy venues, near-zero or negative on
#   a wash-dominated venue. Compare ratios.
# - `round_trip_60s_pct1`: the challenge venue's value is much *lower* than
#   the comparators because its qty asymmetry breaks pairing — that's a
#   useful sanity check, not a finding.
# - `top_10_size_share`: sizes on the challenge venue aren't fingerprint-style
#   repeated. Healthy venues have specific concentration patterns too;
#   compare.
# - `ia_median_seconds`: order-of-magnitude difference. Challenge is
#   human-scale cadence; comparators are bot-scale.
#
# That's the report's headline data table. Every claim in the report's
# findings section is going to cite a row of it.

# %% [markdown]
# ## 13. Plot helpers — calling them, plus the source for the most interesting one
#
# All helpers live in `src/challenge/analysis/plot.py`. Each takes a
# venue-keyed mapping and returns a matplotlib `Figure`. Apply the shared
# style once at the top, then call any helper.

# %%
import matplotlib.pyplot as plt

from challenge.analysis.plot import (
    apply_default_style,
    plot_inter_arrival_distribution,
    plot_per_hour_volume_breakdown,
)

apply_default_style()

fig = plot_per_hour_volume_breakdown(
    challenge,
    title="Per-hour volume by side (log y) — challenge venue",
    log_y=True,
)
plt.show()  # in interactive cell mode this renders inline

# %% [markdown]
# **The most interesting plot helper** is the inter-arrival distribution
# with optional per-venue subplots, because it solves a real layout problem:
# Binance's sub-millisecond mass squashes the challenge venue's slow cadence
# in an overlay. Splitting into subplots makes both visible.

# %%
show_source(plot_inter_arrival_distribution)

# %% [markdown]
# **Walking through `plot_inter_arrival_distribution`:**
#
# - **`layout` parameter** with two valid values (`"overlay"` or
#   `"subplots"`), validated up front with a `ValueError` on anything else.
#   Validating arguments at function entry catches typos before they
#   produce a wrong-but-plausible chart.
#
# - **`all_pos = pd.concat([s[s > 0] for s in venue_inter_arrivals.values()])`**
#   — concatenate all positive inter-arrivals across venues. Used to compute
#   shared log-scale bin edges so the histograms are directly comparable
#   (same bins on every panel).
#
# - **`np.logspace(np.log10(...), np.log10(...), bins)`** — log-spaced bin
#   edges, the right choice when the data spans multiple decades (here:
#   sub-millisecond to ~10 minutes). Linear bins would put 99% of the mass
#   in the leftmost bucket.
#
# - **Overlay branch**: one axis, one histogram per venue with
#   `histtype="step"` (just the outline, no fill — necessary for visibility
#   when overlaid). Density-normalized so each venue integrates to 1.0,
#   regardless of trade count. Without normalization, Binance's 144k
#   inter-arrivals drown out the challenge venue's 753.
#
# - **Subplots branch**: one panel per venue. `sharex=True` so the
#   log-scale x-axis is identical on every panel — the eye can compare
#   "where does this venue's mass live" across panels. Each panel shows
#   the venue's median as a vertical dashed line and a text label with
#   n/median/p95 stats.
#
# - **`histtype="stepfilled"`** in subplots — the panels don't overlap so
#   we can use a filled histogram with low alpha, which reads better than
#   pure step lines at small density.
#
# - **`fig.suptitle(...)` + `fig.tight_layout()`** at the end — suptitle
#   sits above all panels; tight_layout adjusts spacing so labels don't
#   collide.
#
# **Why two layouts in one function instead of two functions?** Because
# the bin computation and shared scale are identical; only the axis layout
# differs. DRY pays off when you want to add a feature (say, marking p95
# on every panel) — you do it once.

# %%
ia_by_venue = {name: inter_arrival_seconds(df)
               for name, df in {"challenge": challenge,
                                "kraken": kraken,
                                "binance": binance}.items()}
fig = plot_inter_arrival_distribution(
    ia_by_venue,
    title="Inter-arrival distribution by venue",
    layout="subplots",
)
plt.show()

# %% [markdown]
# ## 14. Tests as live documentation
#
# Every test in `tests/test_*.py` has a docstring naming the contract it
# locks in. Reading the tests is often the fastest way to learn what a
# function guarantees.

# %%
import subprocess

print("First 30 test names — each is a contract we depend on:")
result = subprocess.run(
    ["uv", "run", "pytest", "--co", "-q"], capture_output=True, text=True
)
print("\n".join(result.stdout.splitlines()[:30]))

# %% [markdown]
# **Experiment:** open `tests/test_manipulation.py` and read the
# `TestKyleLambda` class. Each method is named after a contract:
# `test_balanced_wash_flow_is_zero`, `test_directional_flow_is_positive`,
# `test_rejects_invalid_side_values`, etc. Together they fully document
# what `kyle_lambda` promises.

# %% [markdown]
# ## 15. Where to go from here
#
# - **Read the primer modules** — 02 (wash signals), 03 (volume / time
#   anomalies), 05 (comparison venues), 07 (Kyle's lambda). They expand
#   the math and economics you've now seen the code for.
# - **Refine the cross-venue table** — try volume-weighted vs count-weighted
#   in the table; compute multiple bin sizes for kyle_lambda; add CIs to
#   buy_sell_ratio.
# - **Start the report draft** — primer 06 covers voice and structure.
#
# The functions you've now seen in action are the entire surface of the
# analysis layer. Anything in the report will be a composition of these
# pieces.
