"""Install and purchase labels: how late they arrive, and what they mean.

Chapter 13. Install labels come from the MMP's postbacks, hours or days
after the impression, so a day's labels keep growing after the day ends.
This report reads the bidder's ledger and measures:

- delays: hours from each impression to the postback that labels it, as
  a cumulative share, for click-through and view-through installs;
- maturity: for each day of impressions, the installs received so far and
  the share that had arrived by the end of that day and each day after;
- purchases: purchase postbacks, revenue, and delay from the impression;
- by placement: impressions and attributed installs, and, when the
  simulator's truth is available, how many of those installs the ads
  actually caused, and how many installs the ads caused that the MMP
  credited to another network.

    python -m adtrain labels --ledger out/ledger --sim-out out/sim
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import ledger, simout

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
DELAY_HOURS = (1, 2, 4, 8, 12, 24, 48, 72, 96, 168)


def report(ledger_dir, sim_out=None, max_days_after: int = 6) -> dict:
    imp_ts = {}  # request ID -> billing time, our impressions
    for n in ledger.read(ledger_dir, "notices"):
        if n.get("type") == "billing" and not n.get("duplicate"):
            rid = n.get("request_id")
            if rid and rid not in imp_ts:
                imp_ts[rid] = int(n["ts_ms"])
    info = {}  # request ID -> (placement, exchange, campaign)
    for b in ledger.read(ledger_dir, "bids"):
        rid = b.get("request_id")
        if rid in imp_ts:
            d = b.get("decision") or {}
            info[rid] = (b["request"].get("placement", "?"),
                         b["request"].get("exchange", "?"),
                         d.get("campaign_id"), int(b.get("day", 0)))
    seen = set()
    installs, purchases = [], []
    latest_ms = 0
    for p in ledger.read(ledger_dir, "postbacks"):
        f = p.get("fields") or {}
        key = f.get("postback_id") or p.get("postback_id")
        if key in seen:
            continue  # the MMP sends some postbacks twice
        seen.add(key)
        recv = int(p["received_ts_ms"])
        latest_ms = max(latest_ms, recv)
        rid = _request_id(f)
        if rid not in imp_ts:
            continue
        row = {"rid": rid, "recv": recv, "imp": imp_ts[rid],
               "event_ts": _int(f.get("event_ts_ms")),
               "touch": f.get("touch_type") or "?",
               "revenue": _float(f.get("revenue_usd"))}
        if f.get("event_name") == "install":
            installs.append(row)
        elif f.get("event_name") == "af_purchase":
            purchases.append(row)
    out = {"impressions": len(imp_ts), "installs": len(installs),
           "purchases": len(purchases),
           "purchase_revenue": round(sum(r["revenue"] for r in purchases), 2),
           "latest_postback_ms": latest_ms}
    out["delays"] = _delays(installs)
    out["purchase_delays"] = _delays(purchases)
    out["maturity"] = _maturity(installs, imp_ts, info, latest_ms, max_days_after)
    out["by_placement"] = _by_placement(installs, imp_ts, info, sim_out)
    return out


def _request_id(f: dict) -> str | None:
    """Our request ID, from the click or impression the MMP credited."""
    for k, prefix in (("click_id", "c"), ("impression_id", "i")):
        v = f.get(k) or ""
        if v.startswith(prefix) and len(v) > 1:
            return v[1:]
    return None


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _delays(rows) -> dict:
    """Cumulative share received by each number of hours after the
    impression, overall and by touch type."""
    out = {"hours": list(DELAY_HOURS)}
    groups = {"all": rows}
    for t in sorted({r["touch"] for r in rows}):
        groups[t] = [r for r in rows if r["touch"] == t]
    for name, g in groups.items():
        h = np.array([(r["recv"] - r["imp"]) / HOUR_MS for r in g], float)
        out[name] = {
            "n": int(len(h)),
            "median_hours": round(float(np.median(h)), 1) if len(h) else None,
            "share_by_hour": [round(float((h <= x).mean()), 3) if len(h) else None
                              for x in DELAY_HOURS],
        }
    return out


def _maturity(installs, imp_ts, info, latest_ms, max_after) -> list[dict]:
    """For each day of impressions: installs so far, and the share that had
    arrived by the end of that day, the next day, and so on."""
    imps_by_day = defaultdict(int)
    for rid in imp_ts:
        imps_by_day[info[rid][3] if rid in info else imp_ts[rid] // DAY_MS] += 1
    by_day = defaultdict(list)
    for r in installs:
        day = info[r["rid"]][3] if r["rid"] in info else r["imp"] // DAY_MS
        by_day[day].append(r["recv"] // DAY_MS)
    rows = []
    for day in sorted(imps_by_day):
        recv = np.array(by_day.get(day, []), int)
        complete = latest_ms // DAY_MS  # days fully observed so far
        shares = []
        for k in range(max_after + 1):
            if day + k > complete:
                shares.append(None)  # that day hasn't happened yet
            elif len(recv):
                shares.append(round(float((recv <= day + k).mean()), 3))
            else:
                shares.append(None)
        rows.append({"day": int(day), "impressions": imps_by_day[day],
                     "installs_so_far": int(len(recv)),
                     "share_by_end_of_day": shares})
    return rows


def _by_placement(installs, imp_ts, info, sim_out) -> list[dict]:
    rows = defaultdict(lambda: defaultdict(int))
    for rid in imp_ts:
        rows[info.get(rid, ("?",))[0]]["impressions"] += 1
    for r in installs:
        pl = info.get(r["rid"], ("?",))[0]
        rows[pl]["attributed"] += 1
        rows[pl]["attributed_" + r["touch"]] += 1
    truth = simout.table(sim_out, "truth", "installs")
    has_truth = truth is not None and len(truth) > 0
    if has_truth:
        ours = {"i" + rid for rid in imp_ts}
        for t in truth.itertuples(index=False):
            caused_by_us = t.source == "ad" and t.impression_id in ours
            credited = t.credited_to == "us" and not t.rejected
            if credited:
                ref = t.click_id if t.touch_type == "click" else t.impression_ref
                rid = (ref or "")[1:]
                if rid not in imp_ts:
                    continue
                pl = info.get(rid, ("?",))[0]
                rows[pl]["true_credited"] += 1
                rows[pl]["true_credited_" + t.touch_type] += 1
                if caused_by_us:
                    rows[pl]["credited_and_caused"] += 1
                    rows[pl]["credited_and_caused_" + t.touch_type] += 1
            if caused_by_us:
                pl = info.get(t.impression_id[1:], ("?",))[0]
                rows[pl]["caused"] += 1
                if not credited:
                    rows[pl]["caused_not_credited"] += 1
    out = []
    for pl in sorted(rows, key=lambda k: -rows[k]["impressions"]):
        r = rows[pl]
        row = {"placement": pl, "impressions": r["impressions"],
               "attributed": r["attributed"],
               "attributed_click": r["attributed_click"],
               "attributed_view": r["attributed_view"]}
        if has_truth:
            cr = r["true_credited"]
            row.update({
                "caused": r["caused"],
                "credited_caused_share": _share(r["credited_and_caused"], cr),
                "view_caused_share": _share(r["credited_and_caused_view"],
                                            r["true_credited_view"]),
                "click_caused_share": _share(r["credited_and_caused_click"],
                                             r["true_credited_click"]),
                "caused_not_credited": r["caused_not_credited"],
            })
        out.append(row)
    return out


def _share(a, b):
    return round(a / b, 3) if b else None


def print_report(r: dict) -> None:
    print(f"{r['impressions']:,} impressions, {r['installs']:,} attributed "
          f"installs, {r['purchases']:,} purchases (${r['purchase_revenue']:,.2f})")
    d = r["delays"]
    print("share of installs received by hours after the impression:")
    print("  " + " ".join(f"{h:>5}h" for h in d["hours"]))
    for k in ("all", "click", "view"):
        if k in d and d[k]["n"]:
            print("  " + " ".join(f"{s:>6.0%}" for s in d[k]["share_by_hour"])
                  + f"  {k} (n={d[k]['n']}, median {d[k]['median_hours']}h)")
    print("installs by day of impression, share received by the end of day +k:")
    for m in r["maturity"]:
        s = " ".join("    -" if x is None else f"{x:>5.0%}"
                     for x in m["share_by_end_of_day"])
        print(f"  day {m['day']}: {m['installs_so_far']:>4} so far  {s}")
    print("by placement:")
    for p in r["by_placement"]:
        line = (f"  {p['placement']:<15} {p['impressions']:>7,} imps "
                f"{p['attributed']:>4} attributed "
                f"({p['attributed_click']} click, {p['attributed_view']} view)")
        if "caused" in p:
            line += (f"; caused {p['caused']}, credited and caused "
                     f"{_pct(p['credited_caused_share'])} (views "
                     f"{_pct(p['view_caused_share'])}), caused but credited "
                     f"elsewhere {p['caused_not_credited']}")
        print(line)


def _pct(x):
    return "-" if x is None else f"{x:.0%}"
