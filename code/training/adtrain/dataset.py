"""Click examples from the ledger (Chapter 9).

One example is one shown impression: the FeatureVector and ad the bidder
logged when it bid, joined on the request ID, and a label of 1 if the
first click came within the click window. Impressions whose window hasn't
closed yet aren't final, so they're left out.

    billing notices (impressions, deduplicated)
      join bids (FeatureVector + ad), on request_id
      left join clicks (first per impression), on request_id
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import ledger

DAY_MS = 86_400_000


@dataclass
class Example:
    request_id: str
    ts_ms: int  # when the impression was shown
    day: int
    features: dict
    ad: dict
    logged_p: float  # what the bidder predicted when it bid
    label: int
    logged_model: str = ""  # model version, empty for the prior


@dataclass
class Dataset:
    examples: list = field(default_factory=list)
    data_end_ms: int = 0
    click_ts: dict = field(default_factory=dict)  # request ID -> first click
    click_ids: dict = field(default_factory=dict)  # our click ID -> request ID
    counts: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def build(ledger_dir, click_window_s: int = 3600) -> Dataset:
    ds = Dataset()
    c = ds.counts
    # Impressions and clicks first: they're small, and they say which
    # bids to keep, so only bids that were shown are held in memory.
    # The logs may still be growing, so the label cut-off comes from the
    # logs the labels depend on, read in this order: the newest
    # impression read, since every click read after it is for an
    # impression at or before its own time.
    imps, end = {}, 0
    for r in ledger.read(ledger_dir, "notices"):
        end = max(end, r.get("ts_ms", 0))
        if r.get("type") != "billing" or r.get("duplicate"):
            continue
        imps.setdefault(r["request_id"], r)  # the first billing counts
    clicks = {}
    for r in ledger.read(ledger_dir, "clicks"):
        if r.get("duplicate"):
            continue
        rid = r["request_id"]
        if rid not in clicks or r["ts_ms"] < clicks[rid]:
            clicks[rid] = r["ts_ms"]
        ds.click_ids[r.get("event_id", "c" + rid)] = rid
    bids, dup_bids, n_bids = {}, 0, 0
    for r in ledger.read(ledger_dir, "bids"):
        d = r.get("decision") or {}
        if d.get("action") != "bid":
            continue
        n_bids += 1
        rid = r["request_id"]
        if rid not in imps:
            continue
        if not r.get("features") or not r.get("ad"):
            c["shown_bids_without_features"] = c.get(
                "shown_bids_without_features", 0) + 1
            continue
        if rid in bids:
            dup_bids += 1
        bids[rid] = r
    c["bids"] = n_bids
    c["impressions"], c["clicks"] = len(imps), len(clicks)
    if dup_bids:
        ds.warnings.append(
            f"{dup_bids:,} request IDs were bid on more than once, which "
            "happens when the same simulated day is run twice into one "
            "ledger. Continue later runs at a new day (--start-day), or "
            "clear out/ledger to start over.")
    ds.data_end_ms = end
    ds.click_ts = clicks
    window = click_window_s * 1000
    unjoined = not_final = late = 0
    for rid, n in imps.items():
        b = bids.get(rid)
        if b is None:
            unjoined += 1
            continue
        t = n["ts_ms"]
        if t + window > end:
            not_final += 1
            continue
        ct = clicks.get(rid)
        label = int(ct is not None and t - 1000 <= ct <= t + window)
        if ct is not None and not label:
            late += 1
        d = b["decision"]
        ds.examples.append(Example(
            request_id=rid, ts_ms=t, day=t // DAY_MS, features=b["features"],
            ad=b["ad"], logged_p=float(d.get("p_click") or 0.0), label=label,
            logged_model=d.get("model", "")))
    ds.examples.sort(key=lambda e: e.ts_ms)
    c.update(examples=len(ds.examples), unjoined_impressions=unjoined,
             not_final=not_final, clicks_outside_window=late,
             positives=sum(e.label for e in ds.examples))
    return ds
