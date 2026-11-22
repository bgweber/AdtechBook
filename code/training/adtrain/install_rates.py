"""Click-to-install rates by segment, for the first CPI campaigns
(Chapters 8 and 10).

Install postbacks from the MMP are joined to our clicks on the click ID,
in batch. For each segment (campaign, country, and placement type), the
rate is the share of clicks followed by an install within a day, shrunk
toward its parent's rate:

    rate = (installs + k * parent rate) / (clicks + k)

with segment -> campaign -> all campaigns -> a prior. With few installs,
the rates stay near their parents, which is the point: labels are slow and
sparse, and a segment with 40 clicks and one install says little.
"""
from __future__ import annotations

from collections import defaultdict

from . import ledger

INSTALL_WINDOW_MS = 24 * 3_600_000  # count installs within a day of the click
MATURE_MS = 6 * 3_600_000  # most installs that come at all come by then


def compute(ledger_dir, ds, prior: float = 0.03, k: float = 200.0,
            truth=None) -> tuple[dict, dict]:
    """Returns (rates for the bundle, a readout for the report)."""
    pb_end, installs, dups, unmatched = 0, {}, 0, 0
    for r in ledger.read(ledger_dir, "postbacks"):
        pb_end = max(pb_end, r.get("received_ts_ms", 0))
        f = r.get("fields") or {}
        if r.get("duplicate"):
            dups += 1
            continue
        # the MMP's vocabulary, translated here (Chapter 8)
        if f.get("event_name") != "install" or f.get("touch_type") != "click":
            continue
        rid = ds.click_ids.get(f.get("click_id", ""))
        if rid is None:
            unmatched += 1
            continue
        installs.setdefault(rid, int(float(f.get("event_ts_ms") or 0)))
    end = max(ds.data_end_ms, pb_end)

    # clicked impressions whose install window has mostly passed
    seg = defaultdict(lambda: [0, 0])
    camp = defaultdict(lambda: [0, 0])
    total = [0, 0]
    meta = {}
    for e in ds.examples:
        ct = ds.click_ts.get(e.request_id)
        if not e.label or ct is None or ct + MATURE_MS > end:
            continue
        it = installs.get(e.request_id)
        y = int(it is not None and ct <= it <= ct + INSTALL_WINDOW_MS)
        s = e.features.get("strings") or {}
        c = e.ad.get("campaign", "")
        key = f"{c}|{s.get('country', '')}|{s.get('placement', '')}"
        for acc in (seg[key], camp[c], total):
            acc[0] += 1
            acc[1] += y
        meta[c] = e.ad.get("goal", "")

    overall = (total[1] + k * prior) / (total[0] + k)
    camp_rate = {c: (i + k * overall) / (n + k) for c, (n, i) in camp.items()}
    seg_rate = {s: (i + k * camp_rate[s.split("|")[0]]) / (n + k)
                for s, (n, i) in seg.items()}
    rates = {"prior": prior, "k": k, "overall": overall,
             "campaigns": camp_rate, "segments": seg_rate,
             "install_window_hours": INSTALL_WINDOW_MS / 3_600_000}

    true_cti = _true_rates(truth, ds) if truth is not None else {}
    rows = []
    for c, (n, i) in sorted(camp.items(), key=lambda x: -x[1][0]):
        rows.append({"campaign": c, "goal": meta.get(c, ""), "clicks": n,
                     "installs": i, "raw": i / n if n else None,
                     "smoothed": camp_rate[c], "true": true_cti.get(c)})
    top = sorted(seg.items(), key=lambda x: -x[1][0])[:12]
    readout = {
        "overall": {"clicks": total[0], "installs": total[1],
                    "raw": total[1] / total[0] if total[0] else None,
                    "smoothed": overall},
        "prior": prior, "k": k, "campaigns": rows,
        "segments": [{"segment": s, "clicks": n, "installs": i,
                      "smoothed": seg_rate[s]} for s, (n, i) in top],
        "postbacks": {"click_installs_matched": len(installs),
                      "unmatched": unmatched, "duplicates": dups},
    }
    return rates, readout


def _true_rates(truth: dict, ds) -> dict:
    """Mean true click-to-install probability of our clicked impressions,
    by campaign, from the simulator (for checking only)."""
    acc = defaultdict(lambda: [0.0, 0])
    for e in ds.examples:
        p = truth.get("i" + e.request_id)
        if e.label and p is not None:
            a = acc[e.ad.get("campaign", "")]
            a[0] += p
            a[1] += 1
    return {c: s / n for c, (s, n) in acc.items() if n >= 20}
