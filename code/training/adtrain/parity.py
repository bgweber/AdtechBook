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


def check(ledger_dir, max_examples: int = 3, days=None) -> dict:
    """days limits the check to some days of the ledger, as the bidder's
    daily parity job does (Chapter 24); None checks every day."""
    raw = {r["request_id"]: r for r in ledger.read(ledger_dir, "raw", days)}
    stats = defaultdict(lambda: {"compared": 0, "mismatched": 0, "examples": []})
    rows = mismatched_rows = 0
    by_exchange = defaultdict(lambda: [0, 0])
    by_day = defaultdict(lambda: [0, 0])
    if raw:
        for b in ledger.read(ledger_dir, "bids", days):
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
        "checked_days": days,
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
    if d.get("action") == "bid" or d.get("advertiser_value"):
        line = (f"  pricing: advertiser value ${d.get('advertiser_value', 0):.4f}, "
                f"media value ${d.get('media_value', 0):.4f}, "
                f"explore x{d.get('explore_mult', 1):.3f}")
        if d.get("p_install_click") is not None:
            line += (f", install rate {d['p_install_click']:.4f} "
                     f"from {d.get('install_rate_from', '?')}")
        print(line)
    if d.get("p_install"):
        # walk pricing, term by term (Chapter 21)
        print(f"  installs per impression: {d.get('p_click', 0):.4f} x {d.get('p_install_click', 0):.4f}"
              f" + (1 - {d.get('p_click', 0):.4f}) x {d.get('p_install_view', 0):.6f} = {d['p_install']:.6f}")
        if d.get("value_per_install"):
            print(f"  D7 revenue per install: ${d['value_per_install']:.2f} ({d.get('value_from', '?')})")
        print(f"  target {d.get('target')}, calibration multiplier {d.get('kappa', 1):.3f}, "
              f"advertiser value ${d.get('advertiser_value', 0):.4f}, media value ${d.get('media_value', 0):.4f}"
              f" after the {20}% margin")
    if d.get("shade_from"):
        print(f"  shading: {d['shade_from']}, {d.get('shade', 0):.3f} of the media value"
              + (f", predicted win rate {d['win_prob']:.3f}" if d.get("win_prob") else ""))
    if d.get("candidates"):
        print(f"  internal auction ({'explored: picked at random' if d.get('auction_explored') else 'most expected surplus wins'}):")
        for c in d["candidates"]:
            mark = "*" if c["campaign_id"] == d.get("campaign_id") else " "
            print(f"   {mark} campaign {c['campaign_id']:>3} {c['goal']:<5} value ${c['value']:.3f}  bid ${c['bid']:.3f}"
                  + (f"  win {c['win_prob']:.3f}" if c.get("win_prob") else "")
                  + ("  below the floor" if c.get("below_floor") else f"  score {c['score']:.4f}"))
    if d.get("creative_from"):
        print(f"  creative {d.get('creative_id')}: {d['creative_from']}, selection probability "
              f"{d.get('selection_prob', 0):.4f} (campaign and creative)")
    if d.get("conversion_model"):
        print(f"  conversion models: {d['conversion_model']} (click, install after a click "
              f"{d.get('p_install_click', 0):.4f}, install after a view "
              f"{d.get('p_install_view', 0):.6f})")
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


def find(ledger_dir, day=None, goal=None, reason=None, explored=False,
         clicked=False, limit=5) -> list[dict]:
    """A few bids that match, to pick a request ID for `trace`."""
    clicks = None
    if clicked:
        clicks = {c.get("request_id") for c in ledger.read(ledger_dir, "clicks")}
    days = [day] if day is not None else None
    out = []
    for b in ledger.read(ledger_dir, "bids", days):
        d = b.get("decision") or {}
        if goal and (b.get("ad") or {}).get("goal", "").lower() != goal.lower():
            continue
        if reason and d.get("reason") != reason:
            continue
        if explored and d.get("explore_mult", 1) == 1:
            continue
        if clicks is not None and b.get("request_id") not in clicks:
            continue
        out.append(b)
        if len(out) >= limit:
            break
    return out


def print_found(bids: list[dict]) -> None:
    if not bids:
        print("no matching bids")
    for b in bids:
        d = b["decision"]
        print(f"{b['request_id']}  day {b['day']}  {d['action']}"
              f"{' (' + d['reason'] + ')' if d.get('reason') else ''}  "
              f"campaign {d.get('campaign_id')}  bid ${d.get('bid', 0):.4f}")

