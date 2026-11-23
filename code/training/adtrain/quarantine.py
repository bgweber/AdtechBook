"""Quarantining bad data (Chapter 24).

When bad data reaches the ledger, such as a day of false install postbacks
from a misconfigured MMP, it doesn't go away: every dataset whose window
includes it learns from it again, and the device history built from it
carries it forward. A quarantine entry marks the bad records, by source,
time range, and field values, in `out/datasets/quarantine.json`. Every
dataset build reads the list and skips what it marks, and records which
entries it applied in its manifest. The bidder's device profiles are
recounted from the postbacks with the same list.

    python -m adtrain quarantine inspect --day 41
    python -m adtrain quarantine add --day 41 --where match_type=probabilistic \\
        --reason "MMP probabilistic matching turned on by mistake"
    python -m adtrain quarantine list
    python -m adtrain quarantine remove q-0001
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import ledger

DAY_MS = 86_400_000
HOUR_MS = 3_600_000
SOURCES = ("postbacks",)


def path(datasets_dir) -> Path:
    return Path(datasets_dir) / "quarantine.json"


def load(datasets_dir) -> list[dict]:
    p = path(datasets_dir)
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("entries", [])


def _save(datasets_dir, entries) -> None:
    p = path(datasets_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"entries": entries}, indent=1))
    tmp.replace(p)


def add(datasets_dir, source: str, from_ms: int, to_ms: int, where: dict | None = None,
        reason: str = "") -> dict:
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}: one of {SOURCES}")
    if to_ms <= from_ms:
        raise ValueError("the time range is empty")
    entries = load(datasets_dir)
    n = max([int(e["id"].split("-")[1]) for e in entries] + [0]) + 1
    e = {"id": f"q-{n:04d}", "source": source, "from_ms": int(from_ms), "to_ms": int(to_ms),
         "where": {str(k): str(v) for k, v in (where or {}).items()}, "reason": reason,
         "added": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    entries.append(e)
    _save(datasets_dir, entries)
    return e


def remove(datasets_dir, entry_id: str) -> bool:
    entries = load(datasets_dir)
    keep = [e for e in entries if e["id"] != entry_id]
    _save(datasets_dir, keep)
    return len(keep) < len(entries)


def matcher(entries):
    """A function that says which quarantine entry, if any, marks a
    postback record, by its received time and fields."""
    pb = [e for e in entries if e.get("source") == "postbacks"]
    if not pb:
        return lambda p: None

    def match(p):
        t = int(p.get("received_ts_ms") or 0)
        f = p.get("fields") or {}
        for e in pb:
            if e["from_ms"] <= t < e["to_ms"] and all(
                    str(f.get(k, "")) == v for k, v in e.get("where", {}).items()):
                return e["id"]
        return None
    return match


def device_history(ledger_dir, datasets_dir) -> dict:
    """Each device's installs and purchases credited to us, from every
    postback received, deduplicated, without the quarantined ones: what the
    bidder's profiles count, recounted (Chapter 24, step 4)."""
    match = matcher(load(datasets_dir))
    seen, skipped = set(), Counter()
    counts = defaultdict(lambda: [0, 0])
    for p in ledger.read(ledger_dir, "postbacks"):
        f = p.get("fields") or {}
        pid = f.get("postback_id") or p.get("postback_id")
        if p.get("duplicate") or pid in seen:
            continue
        seen.add(pid)
        q = match(p)
        if q:
            skipped[q] += 1
            continue
        ifa = (f.get("advertising_id") or "").lower()
        if not ifa:
            continue
        if f.get("event_name") == "install":
            counts[ifa][0] += 1
        elif f.get("event_name") == "af_purchase":
            counts[ifa][1] += 1
    return {"devices": dict(counts), "skipped": dict(skipped)}


def inspect(ledger_dir, day: int, field: str = "match_type", days_before: int = 3) -> dict:
    """Install postbacks received on a day, by hour and by the values of one
    field, against the days before: where a bad batch of labels shows up
    and what marks it."""
    days = [d for d in ledger.days(ledger_dir, "postbacks") if day - days_before <= d <= day + 1]
    by_day = defaultdict(Counter)
    hours = defaultdict(Counter)
    seen = set()
    for p in ledger.read(ledger_dir, "postbacks", days):
        f = p.get("fields") or {}
        pid = f.get("postback_id")
        if f.get("event_name") != "install" or pid in seen:
            continue
        seen.add(pid)
        t = int(p.get("received_ts_ms") or 0)
        d = t // DAY_MS
        if not day - days_before <= d <= day:
            continue
        v = str(f.get(field) or "(none)")
        by_day[d][v] += 1
        if d == day:
            hours[(t % DAY_MS) // HOUR_MS][v] += 1
    return {"day": day, "field": field,
            "by_day": {int(d): dict(c) for d, c in sorted(by_day.items())},
            "by_hour": {int(h): dict(c) for h, c in sorted(hours.items())}}


def print_inspect(r: dict) -> None:
    print(f"install postbacks received, by {r['field']}:")
    vals = sorted({v for c in r["by_day"].values() for v in c})
    print("  day   " + "".join(f"{v:>16}" for v in vals))
    for d, c in r["by_day"].items():
        print(f"  {d:<5} " + "".join(f"{c.get(v, 0):>16,}" for v in vals))
    print(f"day {r['day']} by hour received:")
    for h, c in r["by_hour"].items():
        print(f"  {h:02d}:00 " + "".join(f"{c.get(v, 0):>16,}" for v in vals))


def print_list(entries) -> None:
    if not entries:
        print("no quarantine entries")
    for e in entries:
        w = ", ".join(f"{k}={v}" for k, v in e.get("where", {}).items()) or "every record"
        print(f"{e['id']}  {e['source']}  {_clock(e['from_ms'])} to {_clock(e['to_ms'])}  "
              f"{w}  {e.get('reason', '')}")


def _clock(ms: int) -> str:
    m = (ms % DAY_MS) // 60_000
    return f"day {ms // DAY_MS} {m // 60:02d}:{m % 60:02d}"
