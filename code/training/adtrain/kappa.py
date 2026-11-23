"""Calibration multipliers by campaign (Chapter 21).

    kappa_c = (O_c + k * kappa_all) / (E_c + k)

O_c is the installs that arrived for campaign c's recent billed
impressions, and E_c the installs the bidder predicted for the same
impressions, when it priced them. Only impressions whose install labels
have matured count, recent days weigh more, small campaigns shrink toward
the overall multiplier, and every multiplier is bounded to 0.5-2.

    python -m adtrain kappa --ledger out/ledger --models out/models
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from . import ledger

K = 20.0          # predicted installs of evidence a campaign needs to move off the overall
DECAY = 0.8       # weight per day of age
MATURE_DAYS = 3   # an impression's installs count once it's this many days old
WINDOW_DAYS = 10
BOUNDS = (0.5, 2.0)


def compute(ledger_dir, log=print) -> dict:
    days = ledger.days(ledger_dir, "bids")
    if not days:
        raise RuntimeError("the ledger has no bids yet")
    last = max(days)
    use = [d for d in days if last - WINDOW_DAYS <= d <= last - MATURE_DAYS]
    if not use:
        raise RuntimeError(f"no day is {MATURE_DAYS} days old yet: play more days")
    pred = {}
    for b in ledger.read(ledger_dir, "bids", use):
        d = b.get("decision") or {}
        if d.get("action") != "bid" or (d.get("p_install_click") or 0) <= 0:
            continue
        pc, pic, piv = d.get("p_click") or 0, d.get("p_install_click") or 0, d.get("p_install_view") or 0
        pred[b["request_id"]] = (d.get("campaign_id"), (b.get("ad") or {}).get("goal", ""),
                                 d.get("p_install") or pc * pic + (1 - pc) * piv, int(b.get("day", 0)))
    billed = set()
    for n in ledger.read(ledger_dir, "notices", use + [max(use) + 1]):
        if n.get("type") == "billing" and not n.get("duplicate"):
            billed.add(n["request_id"])
    got = set()
    seen = set()
    for p in ledger.read(ledger_dir, "postbacks"):
        f = p.get("fields") or {}
        pid = f.get("postback_id") or p.get("postback_id")
        if pid in seen or f.get("event_name") != "install":
            continue
        seen.add(pid)
        for k in ("click_id", "impression_id"):
            v = f.get(k) or ""
            if len(v) > 1:
                got.add(v[1:])
                break
    O, E = defaultdict(float), defaultdict(float)
    goal = {}
    for rid, (cid, g, p, day) in pred.items():
        if rid not in billed or str(g).upper() == "CPC":
            continue
        w = DECAY ** (last - day)
        E[cid] += w * p
        O[cid] += w * (rid in got)
        goal[cid] = g
    o_all, e_all = sum(O.values()), sum(E.values())
    k_all = (o_all + K) / (e_all + K)
    lo, hi = BOUNDS
    by = {}
    detail = []
    for cid in sorted(E):
        k = (O[cid] + K * k_all) / (E[cid] + K)
        by[str(cid)] = round(min(hi, max(lo, k)), 3)
        detail.append({"campaign_id": cid, "goal": goal[cid], "installs": round(O[cid], 1),
                       "predicted": round(E[cid], 1), "raw_ratio": round(O[cid] / E[cid], 3) if E[cid] else None,
                       "kappa": by[str(cid)], "bounded": not (lo <= k <= hi)})
    log(f"calibration from days {use[0]}-{use[-1]}: {o_all:.1f} installs against {e_all:.1f} "
        f"predicted (weighted), overall kappa {k_all:.3f}")
    return {"days": [use[0], use[-1]], "overall": round(min(hi, max(lo, k_all)), 3),
            "by_campaign": by, "detail": detail, "k": K, "decay": DECAY, "bounds": list(BOUNDS)}


def publish(models_dir, c: dict) -> str:
    d = Path(models_dir) / "calibration"
    d.mkdir(parents=True, exist_ok=True)
    n = len(list(d.glob("kappa-*.json"))) + 1
    c["version"] = f"kappa-{n:04d}"
    (d / f"{c['version']}.json").write_text(json.dumps(c, indent=1))
    return c["version"]


def print_calibration(c: dict) -> None:
    print(f"\n{c.get('version', '')}: days {c['days'][0]}-{c['days'][1]}, overall {c['overall']}")
    print(f"  {'campaign':>8} {'goal':<6}{'installs':>10}{'predicted':>11}{'ratio':>8}{'kappa':>8}")
    for x in c["detail"]:
        print(f"  {x['campaign_id']:>8} {x['goal']:<6}{x['installs']:>10.1f}{x['predicted']:>11.1f}"
              f"{(x['raw_ratio'] or 0):>8.2f}{x['kappa']:>8.2f}" + ("  (bounded)" if x["bounded"] else ""))
