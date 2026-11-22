"""The feature parity check (Chapter 8), and a trace of one request.

For every bid whose raw context the bidder logged, recompute the
FeatureVector with the second implementation in `features_ref`, compare
it with the one the bidder logged, and report mismatches by feature.

    python -m adtrain parity --ledger out/ledger
    python -m adtrain trace <request_id> --ledger out/ledger
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone

from . import features_ref, ledger


def check(ledger_dir, max_examples: int = 3) -> dict:
    raw = {r["request_id"]: r for r in ledger.read(ledger_dir, "raw")}
    stats = defaultdict(lambda: {"compared": 0, "mismatched": 0, "examples": []})
    rows = mismatched_rows = 0
    by_exchange = defaultdict(lambda: [0, 0])
    by_day = defaultdict(lambda: [0, 0])
    if raw:
        for b in ledger.read(ledger_dir, "bids"):
            r = raw.get(b.get("request_id"))
            if r is None or not b.get("features"):
                continue
            n = features_ref.normalize(r["request"], r["exchange"])
            ref = features_ref.flatten(features_ref.build(n, r["profile"], r["ts_ms"]))
            got = features_ref.flatten(b["features"])
            rows += 1
            bad = False
            for k in sorted(set(ref) | set(got)):
                st = stats[k]
                st["compared"] += 1
                a, c = got.get(k), ref.get(k)
                if _same(a, c):
                    continue
                st["mismatched"] += 1
                bad = True
                if len(st["examples"]) < max_examples:
                    st["examples"].append({"request_id": b["request_id"],
                                           "exchange": r["exchange"],
                                           "logged": _show(a), "recomputed": _show(c)})
            mismatched_rows += bad
            by_exchange[r["exchange"]][0] += 1
            by_exchange[r["exchange"]][1] += bad
            by_day[r["day"]][0] += 1
            by_day[r["day"]][1] += bad
    feats = [{"feature": k, **v, "rate": v["mismatched"] / v["compared"]}
             for k, v in stats.items()]
    feats.sort(key=lambda f: (-f["mismatched"], f["feature"]))
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": rows, "mismatched_rows": mismatched_rows,
        "passed": rows > 0 and mismatched_rows == 0,
        "features": feats,
        "exchanges": [{"exchange": e, "rows": n, "mismatched": m}
                      for e, (n, m) in sorted(by_exchange.items())],
        "days": [{"day": d, "rows": n, "mismatched": m}
                 for d, (n, m) in sorted(by_day.items())],
        "note": ("No raw context logged yet: the bidder logs it for 5% of "
                 "bids. Run the simulator in bid mode first." if not rows else ""),
    }


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-9
    return a == b


def _show(v):
    """A short, JSON-friendly version of a value for the report."""
    if v is None:
        return "(missing)"
    if isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, tuple) for x in v):
        return {"values": list(v[0])[:5], "age_seconds": list(v[1])[:5]}
    if isinstance(v, tuple):
        return list(v)[:5]
    return v


def trace(ledger_dir, request_id: str) -> dict:
    """Everything the logs know about one request, from bid to label."""
    out = {"request_id": request_id}
    for b in ledger.read(ledger_dir, "bids"):
        if b.get("request_id") == request_id:
            out["bid"] = b
    out["notices"] = [n for n in ledger.read(ledger_dir, "notices")
                      if n.get("request_id") == request_id]
    out["clicks"] = [c for c in ledger.read(ledger_dir, "clicks")
                     if c.get("request_id") == request_id]
    ids = {c.get("event_id") for c in out["clicks"]}
    out["postbacks"] = [p for p in ledger.read(ledger_dir, "postbacks")
                        if (p.get("fields") or {}).get("click_id") in ids]
    out["raw_context_logged"] = any(r.get("request_id") == request_id
                                    for r in ledger.read(ledger_dir, "raw"))
    return out


def print_trace(t: dict) -> None:
    b = t.get("bid")
    if b is None:
        print(f"{t['request_id']}: not in the bid ledger")
        return
    d = b["decision"]
    print(f"request {t['request_id']} on {b['request']['exchange']}, "
          f"day {b['day']}, policy {b['policy_version']}")
    print(f"  decision: {d['action']}{' (' + d['reason'] + ')' if d.get('reason') else ''}, "
          f"campaign {d['campaign_id']}, creative {d['creative_id']}, "
          f"p_click {d.get('p_click', 0):.4f} from {d.get('p_click_from', '?')}, "
          f"bid ${d.get('bid', 0):.4f} CPM")
    if b.get("features"):
        fv = b["features"]
        print(f"  FeatureVector: {len(fv.get('strings', {}))} strings, "
              f"{len(fv.get('numeric', {}))} numerics, "
              f"{len(fv.get('sets', {}) or {})} sets; ad {b.get('ad')}")
    for n in t["notices"]:
        print(f"  notice: {n['type']}{' (duplicate)' if n.get('duplicate') else ''} "
              f"at {n['ts_ms']}, price {n.get('price', '')}")
    shown = any(n["type"] == "billing" and not n.get("duplicate") for n in t["notices"])
    for c in t["clicks"]:
        print(f"  click: {c['event_id']} at {c['ts_ms']}"
              f"{' (duplicate)' if c.get('duplicate') else ''}")
    for p in t["postbacks"]:
        f = p.get("fields", {})
        print(f"  postback: {f.get('event_name')} via {f.get('touch_type')}, "
              f"received at {p['received_ts_ms']}")
    if shown:
        imp = min(n["ts_ms"] for n in t["notices"]
                  if n["type"] == "billing" and not n.get("duplicate"))
        label = int(any(not c.get("duplicate") and imp - 1000 <= c["ts_ms"] <= imp + 3_600_000
                        for c in t["clicks"]))
        print(f"  training row: an impression with label {label} "
              f"({'clicked within an hour' if label else 'no click within an hour'}), "
              "features as logged above")
    else:
        print("  training row: none, since the ad was never shown and billed")
    print(f"  raw context logged: {'yes' if t['raw_context_logged'] else 'no (5% sample)'}")


def write(report: dict, path) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=1) + "\n")
