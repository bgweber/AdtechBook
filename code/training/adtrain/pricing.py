"""The unified pricing equation, from the bid ledger (Chapter 33).

explain: one logged bid, term by term: each goal's value, calibration, and
weight; the winner's curse correction; the incrementality tilt; the pacing
multiplier; the margin; shading; exploration; and the floor and cap. Every
term comes from the ledger, so the product of the terms is checked against
the bid that went out.

frontier: the consistency check. Replay a day's bids for one campaign at
pacing multipliers from 0.3 to 2, as the pacing forecast does (Chapter 22),
and forecast at each its spend and its predicted installs and revenue. A
budget and a target are consistent when some multiplier spends the budget
without missing the target; when none does, the check says so, with the
frontier between spend and cost per install (or ROAS) to choose from.

    python -m adtrain explain --campaign 17
    python -m adtrain frontier --campaign 17 --budget 40 --target-cpi 1.5
"""
from __future__ import annotations


import numpy as np

from . import evaluate, ledger, simout
from .pacing import GAMMAS


def _bids(ledger_dir, day):
    days = ledger.days(ledger_dir, "bids")
    if not days:
        raise RuntimeError("the ledger has no bids yet")
    return (days[-1] if day is None else int(day)), days


def explain(ledger_dir, request_id=None, campaign_id=None, day=None) -> dict:
    day, _ = _bids(ledger_dir, day)
    pick = None
    for b in ledger.read(ledger_dir, "bids", [day]):
        d = b.get("decision") or {}
        if d.get("action") != "bid":
            continue
        if request_id and b["request_id"] != request_id:
            continue
        if campaign_id is not None and d.get("campaign_id") != int(campaign_id):
            continue
        if not d.get("p_install") and not request_id and campaign_id is None:
            continue  # by default, an install campaign's bid: more terms
        pick = b  # the day's last match
        if request_id:
            break
    if pick is None:
        raise RuntimeError("no such bid on day %d" % day)
    d = pick["decision"]
    fv = pick.get("features") or {}
    floor = float((fv.get("numeric") or {}).get("bidfloor") or 0)
    steps = []
    goals = d.get("goals") or []
    k = float(d.get("kappa") or 1)
    if goals:
        v = sum(g["weight"] * g["kappa"] * g["value"] for g in goals)
        for g in goals:
            steps.append({"term": f"{g['goal']} goal value", "value": g["value"],
                          "detail": f"target {g['target']}, weight {g['weight']:.2f}, calibration {g['kappa']:.3f}"})
        steps.append({"term": "weighted value", "value": v,
                      "detail": " + ".join(f"{g['weight']:.2f} x {g['kappa']:.3f} x {g['value']:.2f}" for g in goals)})
    else:
        ck, tk = float(d.get("curse_kappa") or 1), float(d.get("tilt") or 1)
        v = float(d["advertiser_value"]) / (ck * tk)
        goal = (pick.get("ad") or {}).get("goal", "").upper()
        steps.append({"term": f"{goal} goal value", "value": v / k,
                      "detail": f"target {d.get('target', '')}"})
        steps.append({"term": "calibration", "value": v, "detail": f"x {k:.3f}"})
    if d.get("curse_kappa"):
        v *= float(d["curse_kappa"])
        steps.append({"term": "winner's curse", "value": v, "detail": f"x {d['curse_kappa']:.3f}"})
    if d.get("tilt"):
        v *= float(d["tilt"])
        steps.append({"term": "incrementality tilt", "value": v,
                      "detail": f"x {d['tilt']:.3f}, from a ratio of {d.get('uplift', 0):.3f}"})
    g, ob = float(d.get("pacing") or 1), float(d.get("overbid") or 1)
    if g != 1 or ob != 1:
        v *= g * ob
        steps.append({"term": "pacing", "value": v,
                      "detail": f"x {g:.3f}" + (f", overbid x {ob:.2f}" if ob != 1 else "")})
    m = d.get("margin")
    if m is None:  # logged since bidder 0.18: before, from the media value
        m = 1 - float(d["media_value"]) / max(float(d["advertiser_value"]) * g * ob, 1e-12)
    m = float(m)
    v *= 1 - m
    steps.append({"term": "margin", "value": v, "detail": f"x (1 - {m:.2f}): the media value"})
    shaded = v * float(d.get("shade") or 0)
    steps.append({"term": "shading", "value": shaded,
                  "detail": f"x {float(d.get('shade') or 0):.3f}, {d.get('shade_from', '')}"
                            + (f", win rate {d['win_prob']:.2f}" if d.get("win_prob") else "")})
    em = float(d.get("explore_mult") or 1)
    if em != 1:
        steps.append({"term": "exploration", "value": shaded * em, "detail": f"x {em:.3f}"})
    steps.append({"term": "bid", "value": float(d["bid"]),
                  "detail": f"after the floor ({floor}) and cap"})
    for s in steps:
        s["value"] = round(float(s["value"]), 4)
    return {"day": day, "request_id": pick["request_id"], "campaign_id": d.get("campaign_id"),
            "goal": (pick.get("ad") or {}).get("goal", "").upper(),
            "predictions": {k_: d.get(k_) for k_ in ("p_click", "p_install", "value_per_install", "uplift")
                            if d.get(k_) is not None},
            "media_value_logged": round(float(d["media_value"]), 4),
            "media_value_from_terms": round(v, 4), "steps": steps}


def frontier(ledger_dir, campaign_id, budget=None, target_cpi=None, target_roas=None,
             day=None, sim_out=None) -> dict:
    """The consistency check for one campaign, from a day's bids."""
    day, _ = _bids(ledger_dir, day)
    cid = int(campaign_id)
    camp = simout.table(sim_out, "logs", "campaigns") if sim_out else None
    goal, target = None, None
    if camp is not None:
        row = camp[camp.campaign_id == cid]
        if len(row):
            goal, target = str(row.goal.iloc[0]).upper(), float(row.target.iloc[0])
            if budget is None:
                budget = float(row.daily_budget.iloc[0])
    if target_cpi is None and target_roas is None and target is not None:
        if goal == "CPI":
            target_cpi = target
        elif goal in ("ROAS", "AD_ROAS"):
            target_roas = target
    rows, all_rows, skipped = [], [], 0
    for b in ledger.read(ledger_dir, "bids", [day]):
        d = b.get("decision") or {}
        if str(cid) in (d.get("filtered") or {}) and d["filtered"][str(cid)] == "budget_spent":
            skipped += 1
        if d.get("action") != "bid":
            continue
        r = {"request_id": b["request_id"], "exchange": (b.get("request") or {}).get("exchange"),
             "bid": float(d.get("bid") or 0)}
        all_rows.append(r)
        if d.get("campaign_id") == cid:
            r.update(gamma=float(d.get("pacing") or 1),
                     installs=float(d.get("p_install") or 0) * float(d.get("kappa") or 1),
                     value=float(d.get("value_per_install") or 0))
            rows.append(r)
    if not rows:
        raise RuntimeError(f"campaign {cid} has no bids on day {day}")
    wins, billed = set(), set()
    for n in ledger.read(ledger_dir, "notices", [day, day + 1]):
        if n.get("duplicate"):
            continue
        if n.get("type") == "win":
            wins.add(n["request_id"])
        elif n.get("type") == "billing":
            billed.add(n["request_id"])
    # the win curve by exchange, from every bid of the day
    curves = evaluate._win_curve(all_rows, np.array([r["request_id"] in wins for r in all_rows], float))
    clear = {}
    for ex in {r["exchange"] for r in rows}:
        w = [r["request_id"] for r in all_rows if r["exchange"] == ex and r["request_id"] in wins]
        clear[ex] = sum(x in billed for x in w) / max(1, len(w))
    base = np.array([r["bid"] / r["gamma"] for r in rows])
    ex = [r["exchange"] for r in rows]
    clr = np.array([clear[e] for e in ex])
    inst = np.array([r["installs"] for r in rows])
    val = np.array([r["value"] for r in rows])
    scale = (len(rows) + skipped) / len(rows)
    out = []
    for g in GAMMAS:
        imps = evaluate._p_win(curves, ex, base * g) * clr
        spend = float((imps * base * g).sum()) / 1000 * scale
        n_inst = float((imps * inst).sum()) * scale
        rev = float((imps * inst * val).sum()) * scale
        out.append({"gamma": round(float(g), 3), "spend": round(spend, 2), "installs": round(n_inst, 2),
                    "cpi": round(spend / n_inst, 3) if n_inst > 0 else None,
                    "roas": round(rev / spend, 3) if spend > 0 and rev > 0 else None})
    meets_budget = [p for p in out if budget and p["spend"] >= budget]
    def meets_target(p):
        if target_cpi is not None:
            return p["cpi"] is not None and p["cpi"] <= target_cpi
        if target_roas is not None:
            return p["roas"] is not None and p["roas"] >= target_roas
        return True
    ok = [p for p in meets_budget if meets_target(p)]
    best_target = [p for p in out if meets_target(p)]
    res = {"campaign_id": cid, "day": day, "goal": goal, "budget": budget,
           "target_cpi": target_cpi, "target_roas": target_roas, "bids": len(rows),
           "skipped_for_budget": skipped, "frontier": out, "consistent": bool(ok)}
    if ok:
        res["verdict"] = (f"consistent: pacing multipliers from {ok[0]['gamma']} to {ok[-1]['gamma']} spend "
                          f"the ${budget:.2f} budget within the target")
    elif not meets_budget:
        top = out[-1]
        res["verdict"] = (f"inconsistent: even at the highest multiplier, {top['gamma']}, the forecast spend is "
                          f"${top['spend']:.2f}, below the ${budget:.2f} budget: the campaign will underspend")
    else:
        need = meets_budget[0]
        res["verdict"] = (f"inconsistent: spending the ${budget:.2f} budget needs a multiplier of {need['gamma']}, "
                          + (f"at a forecast CPI of ${need['cpi']:.2f}, above the ${target_cpi:.2f} target"
                             if target_cpi is not None else
                             f"at a forecast ROAS of {need['roas']}, below the {target_roas} target")
                          + (f"; within the target, it can spend at most ${best_target[-1]['spend']:.2f} a day"
                             if best_target else "; no multiplier meets the target"))
    nxt = day + 1
    if nxt in _bids(ledger_dir, None)[1]:
        res["next_day"] = _actual(ledger_dir, cid, nxt, out)
    return res


def _actual(ledger_dir, cid, day, out) -> dict:
    """What the campaign did the next day, against the forecast at the
    same spend."""
    rids = set()
    for b in ledger.read(ledger_dir, "bids", [day]):
        d = b.get("decision") or {}
        if d.get("action") == "bid" and d.get("campaign_id") == cid:
            rids.add(b["request_id"])
    spend = 0.0
    for n in ledger.read(ledger_dir, "notices", [day, day + 1]):
        if n.get("type") == "billing" and not n.get("duplicate") and n["request_id"] in rids:
            spend += float(n.get("price") or 0) / 1000
    installs, seen = 0, set()
    for p in ledger.read(ledger_dir, "postbacks"):
        f = p.get("fields") or {}
        if f.get("event_name") != "install" or f.get("postback_id") in seen:
            continue
        seen.add(f.get("postback_id"))
        t = (f.get("click_id") or f.get("impression_id") or "")[1:]
        if t in rids:
            installs += 1
    sp = [p["spend"] for p in out]
    fc_inst = float(np.interp(spend, sp, [p["installs"] for p in out]))
    return {"day": day, "spend": round(spend, 2), "installs": installs,
            "cpi": round(spend / installs, 3) if installs else None,
            "forecast_installs_at_that_spend": round(fc_inst, 1),
            "forecast_cpi_at_that_spend": round(spend / fc_inst, 3) if fc_inst > 0 else None}


def print_explain(r: dict) -> None:
    print(f"bid {r['request_id']}, campaign {r['campaign_id']} ({r['goal']}), day {r['day']}: "
          + ", ".join(f"{k} {v}" for k, v in r["predictions"].items()))
    for s in r["steps"]:
        print(f"  {s['term']:<22}{s['value']:>10.4f}  {s['detail']}")
    print(f"  media value from the terms {r['media_value_from_terms']}, logged {r['media_value_logged']}")


def print_frontier(r: dict) -> None:
    print(f"consistency check, campaign {r['campaign_id']} ({r['goal']}), from day {r['day']}'s {r['bids']:,} bids: "
          f"budget ${r['budget']}, target " + (f"CPI ${r['target_cpi']}" if r['target_cpi'] is not None
                                                 else f"ROAS {r['target_roas']}"))
    print(f"  {'gamma':>6}{'spend':>9}{'installs':>10}{'CPI':>8}{'ROAS':>8}")
    for p in r["frontier"][::3]:
        print(f"  {p['gamma']:>6}{p['spend']:>9.2f}{p['installs']:>10.2f}{(p['cpi'] or 0):>8.2f}{(p['roas'] or 0):>8.3f}")
    print("  " + r["verdict"])
    n = r.get("next_day")
    if n:
        print(f"  day {n['day']}, actual: spend ${n['spend']:.2f}, {n['installs']} installs credited, CPI "
              f"{n['cpi']}; forecast at that spend: {n['forecast_installs_at_that_spend']} installs, "
              f"CPI {n['forecast_cpi_at_that_spend']}")
