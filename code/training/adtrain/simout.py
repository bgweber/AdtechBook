"""Read the simulator's output, across runs.

A live run writes its logs and truth to `out/sim`, and also keeps a copy by
start day in `out/sim/runs/start=NNNN`, since each run continues the market
where the last one stopped. These helpers read a table from every run, or
from `out/sim` when there are no per-run copies (an older run, or an
offline one). The truth tables are for checking models, never for training
them.
"""
from __future__ import annotations

from pathlib import Path


def table(sim_out, kind: str, name: str, columns=None):
    """One of the simulator's tables, from every run, or None.

    `kind` is "logs" or "truth". Rows repeated across runs are dropped.
    """
    if not sim_out:
        return None
    try:
        import pandas as pd
    except ImportError:
        return None
    root = Path(sim_out)
    files = sorted(root.glob(f"runs/start=*/{kind}/{name}.parquet"))
    if not files and (root / kind / f"{name}.parquet").exists():
        files = [root / kind / f"{name}.parquet"]
    if not files:
        return None
    try:
        parts = [pd.read_parquet(f, columns=columns) for f in files]
    except Exception:
        return None
    return pd.concat(parts, ignore_index=True).drop_duplicates()
