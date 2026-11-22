"""Segment click rates, the fallback when the click model can't be used
(Chapter 12).

The observed click rate for each placement type, country, and exchange,
from the same impressions the model trains on, each shrunk toward a
broader segment, as in Chapter 10:

    placement|country|exchange -> placement|country -> placement -> overall

Published in the bundle next to the model. Much worse than the model at
telling impressions apart, but rarely badly miscalibrated.
"""
from __future__ import annotations

from collections import defaultdict


def compute(examples, k: float = 200.0) -> dict:
    seg, pc, pl = (defaultdict(lambda: [0, 0]) for _ in range(3))
    total = [0, 0]
    for e in examples:
        s = e.features.get("strings") or {}
        p, c, x = s.get("placement", ""), s.get("country", ""), s.get("exchange", "")
        for acc in (seg[f"{p}|{c}|{x}"], pc[f"{p}|{c}"], pl[p], total):
            acc[0] += 1
            acc[1] += e.label
    overall = total[1] / total[0] if total[0] else 0.0
    shrink = lambda n, y, parent: (y + k * parent) / (n + k)
    pl_r = {p: shrink(n, y, overall) for p, (n, y) in pl.items()}
    pc_r = {key: shrink(n, y, pl_r[key.split("|")[0]]) for key, (n, y) in pc.items()}
    seg_r = {key: shrink(n, y, pc_r["|".join(key.split("|")[:2])])
             for key, (n, y) in seg.items()}
    return {"k": k, "overall": overall, "placement": pl_r,
            "placement_country": pc_r, "segments": seg_r}
