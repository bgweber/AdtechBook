"""Read the bidder's ledger: gzip JSON lines, partitioned by simulated day.

    <ledger>/<stream>/day=0003/part-0000.jsonl.gz

The bidder may still be writing the newest part, so a truncated gzip
stream or a half-written last line is read up to where it ends.
"""
from __future__ import annotations

import gzip
import json
import re
import zlib
from pathlib import Path

DAY_RE = re.compile(r"day=(\d+)$")


def days(ledger: str | Path, stream: str = "bids") -> list[int]:
    root = Path(ledger) / stream
    out = []
    for d in root.glob("day=*"):
        m = DAY_RE.search(d.name)
        if m and d.is_dir():
            out.append(int(m.group(1)))
    return sorted(out)


def read(ledger: str | Path, stream: str, day_list=None):
    """Yield records from a stream, in day and part order."""
    root = Path(ledger) / stream
    for day in (day_list if day_list is not None else days(ledger, stream)):
        for part in sorted((root / f"day={day:04d}").glob("part-*.jsonl.gz")):
            yield from _read_part(part)


def _read_part(path: Path):
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    return  # a half-written last line
    except (EOFError, OSError, zlib.error):
        return  # a part that's still open
