"""Incrementality from the ghost-bid holdout (Chapter 23).

Devices in the holdout (5%, by a salted hash of the device ID) get a
ghost bid: the bidder runs the whole bid path and logs the bid, but never
sends it. For each advertiser app:

    ITT        = installs per treated device that got a bid
               - installs per holdout device that got a ghost bid
    pi         = share of treated devices with at least one impression
    lift       = ITT / pi, per exposed device
    incremental installs = lift x exposed devices
    iCPI       = spend / incremental installs
    incrementality = incremental installs / attributed installs

An install counts if the device installed the app after its first (real or
ghost) bid for it, from any source: the installs the MMP credited to us,
plus the installs the advertiser shares from its full install data (other
networks and organic). Advertisers who don't share can't be measured this
way: their holdout devices' installs are invisible. With the simulator's
truth, the report also shows how many of our attributed installs our ads
actually caused.

    python -m adtrain lift --ledger out/ledger --sim-out out/sim
"""
from __future__ import annotations

import math
import zlib
from collections import defaultdict

import numpy as np

from . import ledger, simout

Z = 1.96


def _ifa(s):
    return (s or "").lower()


def measure(ledger_dir, sim_out, days: int = 28, holdout_share: float | None = None,
            app: str | None = None, log=print) -> dict:
    """holdout_share < 0.05 keeps only that share of devices in the holdout
    (by a second hash), to see how the interval widens with a smaller one."""
    all_days = ledger.days(ledger_dir, "bids")
    use = all_days[-days:]
    if not use:
        raise RuntimeError("the ledger has no bids")
    camp = simout.table(sim_out, "logs", "campaigns")
    meta = simout.table(sim_out, "logs", "app_metadata")
    if camp is None or meta is None:
        raise RuntimeError("needs the simulator's campaign and app logs (--sim-out)")
    bundle = dict(zip(meta.app_id, meta.bundle))
    camp_app = {int(c): bundle[a] for c, a in zip(camp.campaign_id, camp.app_id)}
    # first real or ghost bid by (app, device), and the group
    first, group = {}, {}
    rid_key = {}
    for b in ledger.read(ledger_dir, "bids", use):
        d = b.get("decision") or {}
        act = d.get("action")
        if act not in ("bid", "ghost"):
            continue
        a = camp_app.get(d.get("campaign_id"))
        ifa = _ifa((b.get("request") or {}).get("ifa"))
        if not a or not ifa:
            continue
        k = (a, ifa)
        g = "control" if act == "ghost" else "treatment"
        if k not in first:
            first[k], group[k] = int(b["ts_ms"]), g
        if act == "bid":
            rid_key[b["request_id"]] = k
    # impressions and spend
    shown, spend = set(), defaultdict(float)
    for n in ledger.read(ledger_dir, "notices", use + [use[-1] + 1]):
        if n.get("type") != "billing" or n.get("duplicate"):
            continue
        k = rid_key.get(n["request_id"])
        if k:
            shown.add(k)
            spend[k[0]] += float(n.get("price") or 0) / 1000
    # installs: credited to us, from postbacks; everything else, from the
    # advertiser's shared install data
    inst, attributed, attributed_at = {}, defaultdict(int), {}
    seen = set()
    for p in ledger.read(ledger_dir, "postbacks"):
        f = p.get("fields") or {}
        if f.get("event_name") != "install" or f.get("postback_id") in seen:
            continue
        seen.add(f.get("postback_id"))
        k = (f.get("app_bundle"), _ifa(f.get("advertising_id")))
        ts = int(float(f.get("event_ts_ms") or 0))
        inst[k] = min(inst.get(k, ts), ts)
        attributed_at[k] = min(attributed_at.get(k, ts), ts)
    na = simout.table(sim_out, "logs", "nonattributed")
    sharing = set()
    if na is not None:
        na = na[na.event_name == "install"].drop_duplicates(["advertising_id", "app_bundle", "event_ts_ms"])
        sharing = set(na.app_bundle)
        for a, ifa, ts in zip(na.app_bundle, na.advertising_id, na.event_ts_ms):
            k = (a, _ifa(ifa))
            inst[k] = min(inst.get(k, int(ts)), int(ts))
    end_ms = (use[-1] + 1) * 86_400_000
    # truth: for each device and app, the time of an install an ad caused;
    # counted below on the same treated devices and window as the estimate
    caused_at = {}
    ti = simout.table(sim_out, "truth", "installs", ["app_id", "caused", "ts_ms", "device_id"])
    if ti is not None:
        ti = ti[ti.caused.astype(bool)]
        for a, dev, ts in zip(ti.app_id, ti.device_id, ti.ts_ms):
            k = (bundle.get(a), _ifa(dev))
            caused_at[k] = min(caused_at.get(k, ts), ts)
    caused = defaultdict(int) if caused_at else None
    rows = defaultdict(lambda: {"treatment": [0, 0], "control": [0, 0], "exposed": 0})
    for k, t0 in first.items():
        a, ifa = k
        if app and a != app:
            continue
        if a not in sharing:
            continue
        ti_ = inst.get(k)
        if ti_ is not None and ti_ < t0:
            continue  # installed before we ever bid: can't install again
        g = group[k]
        if g == "control" and holdout_share is not None and holdout_share < 0.05:
            if zlib.crc32(("shrink:" + ifa).encode()) % 10000 >= holdout_share / 0.05 * 10000:
                continue
        y = int(ti_ is not None and t0 <= ti_ < end_ms + 7 * 86_400_000)
        rows[a][g][0] += 1
        rows[a][g][1] += y
        if g == "treatment" and k in shown:
            rows[a]["exposed"] += 1
            # one attributed install per device, in the same window
            ta = attributed_at.get(k)
            if ta is not None and t0 <= ta < end_ms + 7 * 86_400_000:
                attributed[a] += 1
        if g == "treatment" and caused is not None:
            tc = caused_at.get(k)
            if tc is not None and t0 <= tc < end_ms + 7 * 86_400_000:
                caused[a] += 1
    out = {"days": [use[0], use[-1]], "holdout_share": holdout_share or 0.05,
           "apps": [], "not_measurable": sorted({a for a, _ in first} - sharing)}
    tot = {"treatment": [0, 0], "control": [0, 0], "exposed": 0}
    for a in sorted(rows):
        r = rows[a]
        for g in ("treatment", "control"):
            tot[g][0] += r[g][0]
            tot[g][1] += r[g][1]
        tot["exposed"] += r["exposed"]
        out["apps"].append(_estimate(a, r, spend[a], attributed[a],
                                     caused.get(a, 0) if caused is not None else None))
    out["all"] = _estimate("all measured apps", tot, sum(spend[a] for a in rows),
                           sum(attributed[a] for a in rows),
                           sum(caused.get(a, 0) for a in rows) if caused is not None else None)
    log(f"lift: days {use[0]}-{use[-1]}, {len(rows)} apps with shared install data")
    return out


def _estimate(name, r, spend, attributed, caused) -> dict:
    nt, yt = r["treatment"]
    nc, yc = r["control"]
    e = {"app": name, "treated_devices": nt, "holdout_devices": nc, "treated_installs": yt,
         "holdout_installs": yc, "exposed_devices": r["exposed"], "spend": round(spend, 2),
         "attributed_installs": attributed}
    if nt == 0 or nc == 0:
        return e
    pt, pc = yt / nt, yc / nc
    itt = pt - pc
    # Agresti-Caffo: the interval from rates with one install and one
    # non-install added to each group. A holdout with no installs would
    # otherwise look certain, however few devices it has.
    at, ac = (yt + 1) / (nt + 2), (yc + 1) / (nc + 2)
    se = math.sqrt(at * (1 - at) / (nt + 2) + ac * (1 - ac) / (nc + 2))
    mid = at - ac
    pi = r["exposed"] / nt if nt else 0
    e.update({"rate_treated": round(pt, 5), "rate_holdout": round(pc, 5), "itt": round(itt, 5),
              "itt_ci": [round(mid - Z * se, 5), round(mid + Z * se, 5)], "exposed_share": round(pi, 3)})
    if pi > 0:
        lift = itt / pi
        lo, hi = (mid - Z * se) / pi, (mid + Z * se) / pi
        inc = lift * r["exposed"]
        inc_ci = [lo * r["exposed"], hi * r["exposed"]]
        e.update({"lift_per_exposed": round(lift, 5), "incremental_installs": round(inc, 1),
                  "incremental_ci": [round(inc_ci[0], 1), round(inc_ci[1], 1)]})
        if attributed:
            e["incrementality"] = round(inc / attributed, 3)
            e["incrementality_ci"] = [round(inc_ci[0] / attributed, 3), round(inc_ci[1] / attributed, 3)]
            e["attributed_cpi"] = round(spend / attributed, 2)
        e["icpi"] = round(spend / inc, 2) if inc > 0 else None
        e["icpi_ci"] = [round(spend / inc_ci[1], 2) if inc_ci[1] > 0 else None,
                        round(spend / inc_ci[0], 2) if inc_ci[0] > 0 else None]  # None: unbounded
    if caused is not None:
        e["true_caused_installs"] = caused
        if attributed:
            e["true_incrementality"] = round(caused / attributed, 3)
        e["true_icpi"] = round(spend / caused, 2) if caused else None
    return e


def print_report(r: dict) -> None:
    print(f"\nincrementality, days {r['days'][0]}-{r['days'][1]}, holdout {100 * r['holdout_share']:.1f}% of devices")
    hdr = (f"  {'app':<28}{'treated':>8}{'holdout':>8}{'rate T':>8}{'rate C':>8}{'exposed':>8}"
           f"{'attrib':>7}{'increm.':>9}{'95% CI':>16}{'ratio':>7}{'true':>7}")
    print(hdr)
    print(f"  {'':<28}{'':>8}{'':>8}{'':>8}{'':>8}{'':>8}{'':>7}{'':>9}{'':>16}{'':>7}  (true: installs our ads caused / attributed)")
    for e in r["apps"] + [r["all"]]:
        if "itt" not in e:
            print(f"  {e['app']:<28}{e['treated_devices']:>8}{e['holdout_devices']:>8}   not enough devices")
            continue
        ci = e.get("incremental_ci") or [0, 0]
        print(f"  {e['app']:<28}{e['treated_devices']:>8}{e['holdout_devices']:>8}{100 * e['rate_treated']:>7.2f}%"
              f"{100 * e['rate_holdout']:>7.2f}%{e['exposed_devices']:>8}{e['attributed_installs']:>7}"
              f"{e.get('incremental_installs', 0):>9.1f}{f'{ci[0]:.0f} to {ci[1]:.0f}':>16}"
              f"{(e.get('incrementality') or 0):>7.2f}{(e.get('true_incrementality') or 0):>7.2f}")
    if r["not_measurable"]:
        print(f"  not measurable (the advertiser doesn't share its install data): {', '.join(r['not_measurable'])}")
