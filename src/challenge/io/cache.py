"""
Local read-through parquet cache for replay.

Comparison-venue pulls are slow and rate-limited. Cache once, iterate on the cache.
Parquet preserves types, compresses well, and reads orders of magnitude faster than CSV.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path("cache")


def cached_parquet(
    name: str,
    build: Callable[[], pd.DataFrame],
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    overwrite: bool = False,
    compression: str = "zstd",
) -> pd.DataFrame:
    """Read from local parquet if present, else build, save, return.

    Args:
        name: filename stem (no extension). Use a descriptive name including
            the source and date range, e.g. "kraken_ethbtc_2024w42".
        build: zero-arg callable that produces the DataFrame on cache miss.
        cache_dir: where parquet files live. Default ./cache/.
        overwrite: force re-build even if cache exists.
        compression: parquet compression codec. zstd is the modern default.
    """
    path = cache_dir / f"{name}.parquet"
    if path.exists() and not overwrite:
        return pd.read_parquet(path)
    df = build()
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression=compression, index=False)
    return df
