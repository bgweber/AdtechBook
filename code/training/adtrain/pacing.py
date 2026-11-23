"""The pacing forecast (Chapter 22): each campaign's starting multiplier
for tomorrow, from replaying today's bids at different values of gamma.

For one campaign, the forecast spend at gamma is

    S(gamma) = sum over its priced requests of W(b * gamma) * b * gamma * clear

where b is the bid without today's pacing multiplier, W the win rate at
that bid level (from the logged wins, by exchange), and clear the share of
wins the exchange billed. Requests the campaign skipped once its budget ran
out are counted too, at the average of the ones it priced. The starting
gamma is the one whose forecast spend matches the daily budget, within the
controller's bounds; the controller corrects the forecast's error during
the day.

    python -m adtrain pace --ledger out/ledger
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import evaluate, ledger

GAMMAS = np.exp(np.linspace(math.log(0.3), math.log(2.0), 31))


def forecast(ledger_dir, budgets: dict, day: int | None = None, log=print) -> dict:
    days = ledger.days(ledger_dir, "bids")
    if not days:
        raise RuntimeError("the ledger has no bids yet")
    day = days[-1] if day is None else day
    rows, skipped = [], defaultdict(int)
    for b in ledger.read(ledger_dir, "bids", [day]):
        d = b.get("decision") or {}
        for cid, why in (d.get("filtered") or {}).items():
            if why == "budget_spent":
                skipped[int(cid)] += 1
        if d.get("action") != "bid":
            continue
        rows.append({"request_id": b["request_id"], "campaign": d.get("campaign_id"),
                     "exchange": (b.get("request") or {}).get("exchange"),
                     "bid": float(d.get("bid") or 0),
                     "gamma": float(d.get("pacing") or 1)})
    wins, billed = set(), set()
    for n in ledger.read(ledger_dir, "notices", [day, day + 1]):
        if n.get("duplicate"):
            continue
        if n.get("type") == "win":
            wins.add(n["request_id"])
        elif n.get("type") == "billing":
            billed.add(n["request_id"])
    if not rows:
        raise RuntimeError(f"no bids on day {day}")
    won = np.array([r["request_id"] in wins for r in rows], float)
    curves = evaluate._win_curve(rows, won)
    clear = {}
    for ex in {r["exchange"] for r in rows}:
        w = [r["request_id"] for r in rows if r["exchange"] == ex and r["request_id"] in wins]
        clear[ex] = sum(x in billed for x in w) / max(1, len(w))
    by = defaultdict(list)
    for r in rows:
        by[r["campaign"]].append(r)
    out = {}
    for cid, rs in sorted(by.items()):
        budget = float(budgets.get(str(cid), budgets.get(cid, 0)) or 0)
        if budget <= 0:
            continue
        base = np.array([r["bid"] / r["gamma"] for r in rs])
        ex = [r["exchange"] for r in rs]
        clr = np.array([clear[e] for e in ex])
        scale = (len(rs) + skipped.get(cid, 0)) / len(rs)
        spend = np.array([float((evaluate._p_win(curves, ex, base * g) * base * g * clr).sum())
                          / 1000 * scale for g in GAMMAS])
        i = int(np.searchsorted(spend, budget))  # spend rises with gamma
        g0 = float(GAMMAS[min(i, len(GAMMAS) - 1)])
        out[str(cid)] = {"gamma0": round(g0, 3), "budget": round(budget, 2),
                         "forecast_at_1": round(float(np.interp(0.0, np.log(GAMMAS), spend)), 2),
                         "forecast_at_gamma0": round(float(spend[min(i, len(GAMMAS) - 1)]), 2),
                         "requests": len(rs), "skipped_for_budget": skipped.get(cid, 0)}
    log(f"pacing forecast for day {day + 1} from day {day}: {len(out)} campaigns")
    return {"from_day": day, "day": day + 1, "by_campaign": out}


def publish(models_dir, f: dict) -> str:
    d = Path(models_dir) / "pacing"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"forecast-day-{f['day']:04d}.json"
    p.write_text(json.dumps(f, indent=1))
    return str(p)


def print_forecast(f: dict) -> None:
    print(f"\npacing forecast for day {f['day']}, from day {f['from_day']}'s bids")
    print(f"  {'campaign':>8}{'budget':>9}{'at gamma 1':>12}{'gamma0':>8}{'at gamma0':>11}")
    for cid, x in f["by_campaign"].items():
        print(f"  {cid:>8}{x['budget']:>9.2f}{x['forecast_at_1']:>12.2f}{x['gamma0']:>8.2f}{x['forecast_at_gamma0']:>11.2f}")
