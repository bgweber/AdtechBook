"""Pacing with reinforcement learning, in a simulator built from logged
auctions (Chapter 34).

The simulator replays one campaign's logged day, interval by interval (96
quarter hours). For each of its logged bids, the bid without the day's
pacing multiplier is the base; at a multiplier gamma the bid is base x
gamma, it wins with the day's win curve for its exchange at that bid, pays
its bid (first price) times the exchange's billed share of wins, and wins
the campaign's advertiser value for the impression. Requests the campaign
skipped once its budget ran out, or throttled, are filled in from its bids
in the same hour. The budget is a hard stop, as in the bidder.

Three pacing policies run in it, each setting gamma every interval:

    PI     the bidder's controller (Chapter 22): spend against a plan shaped
           like the day before's traffic, with the same gains and bounds
    MPC    re-plans every interval: the gamma that would spend the budget
           left on the rest of the day, by the forecast (the day before)
    Q      tabular Q-learning, trained in the simulator on randomized
           versions of the day before (traffic and competition +-20%,
           random surges), with the reward of the chapter: value won, minus
           a penalty for spending ahead of plan and for budget left over

They're compared on a test day, normally the newest, with the simulator
checked first by replaying the multipliers the bidder really used that day.

    python -m adtrain rlpace --campaign 16
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from . import evaluate, ledger, simout

DAY_MS = 86_400_000
STEPS = 96
STEP_MS = DAY_MS // STEPS
G_MIN, G_MAX = 0.3, 2.0
KP, KI, MAX_STEP, SLACK = 0.5, 0.05, 0.2, 0.02  # the bidder's PI controller (ops/pacing.go)
LG = np.arange(math.log(G_MIN) - 0.25, math.log(G_MAX) + 0.25 + 1e-9, 0.05)  # log gamma grid, padded
ACTIONS = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])  # changes to log gamma, for Q


def _win_curves(ledger_dir, day):
    rows, wins, billed = [], set(), set()
    for b in ledger.read(ledger_dir, "bids", [day]):
        d = b.get("decision") or {}
        if d.get("action") == "bid":
            rows.append({"request_id": b["request_id"], "exchange": (b.get("request") or {}).get("exchange"),
                         "bid": float(d.get("bid") or 0)})
    for n in ledger.read(ledger_dir, "notices", [day, day + 1]):
        if n.get("duplicate"):
            continue
        if n.get("type") == "win":
            wins.add(n["request_id"])
        elif n.get("type") == "billing":
            billed.add(n["request_id"])
    curves = evaluate._win_curve(rows, np.array([r["request_id"] in wins for r in rows], float))
    clear = {}
    for ex in {r["exchange"] for r in rows}:
        w = [r["request_id"] for r in rows if r["exchange"] == ex and r["request_id"] in wins]
        clear[ex] = sum(x in billed for x in w) / max(1, len(w))
    return curves, clear, billed


def load_day(ledger_dir, cid, day, budget):
    """One campaign's day, as tables of expected spend and value by
    interval and log gamma."""
    curves, clear, billed = _win_curves(ledger_dir, day)
    bids, skipped, gam = [], np.zeros(STEPS, int), defaultdict(list)
    for b in ledger.read(ledger_dir, "bids", [day]):
        d = b.get("decision") or {}
        t = min(STEPS - 1, (int(b["ts_ms"]) - day * DAY_MS) // STEP_MS)
        why = (d.get("filtered") or {}).get(str(cid))
        if why in ("budget_spent", "throttled"):
            skipped[t] += 1
        if d.get("action") != "bid" or d.get("campaign_id") != cid:
            continue
        g = float(d.get("pacing") or 1) * float(d.get("overbid") or 1)
        gam[t].append(float(d.get("pacing") or 1))
        bids.append((t, (b.get("request") or {}).get("exchange"), float(d["bid"]) / g,
                     float(d.get("advertiser_value") or 0) / 1000, b["request_id"]))
    if not bids:
        raise RuntimeError(f"campaign {cid} has no bids on day {day}")
    t = np.array([x[0] for x in bids])
    ex = np.array([x[1] for x in bids], dtype=object)
    base = np.array([x[2] for x in bids])
    val = np.array([x[3] for x in bids])
    clr = np.array([clear.get(e, 1.0) for e in ex])
    # p(win) at each log gamma on the grid, by exchange
    G = np.exp(LG)
    W = np.zeros((len(base), len(LG)))
    for e in set(ex):
        m = ex == e
        if e in curves:
            x, y = curves[e]
            W[m] = np.interp(np.log(np.maximum(base[m, None] * G[None, :], 1e-6)), x, y)
    spend = np.zeros((STEPS, len(LG)))
    value = np.zeros((STEPS, len(LG)))
    np.add.at(spend, t, W * clr[:, None] * base[:, None] * G[None, :] / 1000)
    np.add.at(value, t, W * clr[:, None] * val[:, None])
    # fill in the requests skipped for budget or throttling, from the same
    # hour's bids
    n_bids = np.bincount(t, minlength=STEPS)
    hour = np.arange(STEPS) // 4
    for h in range(24):
        idx = np.flatnonzero(hour == h)
        nb, ns = n_bids[idx].sum(), skipped[idx].sum()
        if ns and nb:
            per_s, per_v = spend[idx].sum(0) / nb, value[idx].sum(0) / nb
            spend[idx] += np.outer(skipped[idx], per_s)
            value[idx] += np.outer(skipped[idx], per_v)
    # what the bidder actually spent, and its multiplier by interval
    actual = 0.0
    rids = set(x[4] for x in bids)
    for n in ledger.read(ledger_dir, "notices", [day, day + 1]):
        if n.get("type") == "billing" and not n.get("duplicate") and n["request_id"] in rids:
            actual += float(n.get("price") or 0) / 1000
    logged_g = np.array([np.mean(gam[i]) if gam[i] else np.nan for i in range(STEPS)])
    logged_g = _ffill(logged_g)
    return {"day": day, "budget": budget, "spend": spend, "value": value, "requests": n_bids + skipped,
            "actual_spend": round(actual, 2), "logged_gamma": logged_g, "bids": int(len(base)),
            "skipped": int(skipped.sum())}


def _ffill(a):
    out, last = a.copy(), 1.0
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        last = out[i]
    return out


def _at(table, t, lg):
    """Interpolate a row of a table at log gamma lg."""
    return float(np.interp(lg, LG, table[t]))


def run(env, policy, plan_share, gamma0=1.0):
    """One day: returns the trajectory and the totals."""
    B = env["budget"]
    spent, won, lg, integ = 0.0, 0.0, 0.0, 0.0
    traj = []
    state = {"lg": math.log(min(gamma0, 1.0)), "integral": 0.0, "lg0": math.log(gamma0)}
    for t in range(STEPS):
        lg = policy(t, spent, state, env, plan_share)
        lg = min(math.log(G_MAX), max(math.log(G_MIN), lg))
        state["lg"] = lg
        s, v = _at(env["spend"], t, lg), _at(env["value"], t, lg)
        if spent + s > B:  # the budget runs out inside this interval
            f = max(0.0, (B - spent) / s) if s > 0 else 0.0
            s, v = s * f, v * f
        spent += s
        won += v
        traj.append({"t": t, "gamma": round(math.exp(lg), 3), "spend": round(spent, 3),
                     "plan": round(B * plan_share[t], 3)})
    g = np.log([p["gamma"] for p in traj])
    track = float(np.mean([abs(p["spend"] - p["plan"]) for p in traj])) / B
    out_at = next((p["t"] for p in traj if p["spend"] >= B * 0.999), None)
    return {"spend": round(spent, 2), "value": round(won, 2), "value_per_dollar": round(won / spent, 3) if spent else None,
            "delivery": round(spent / B, 3), "tracking_error": round(track, 3),
            "volatility": round(float(np.mean(np.abs(np.diff(g)))), 4),
            "budget_out_at": None if out_at is None else f"{out_at // 4:02d}:{15 * (out_at % 4):02d}",
            "trajectory": traj}


def pi_policy(t, spent, st, env, plan_share):
    """The bidder's PI controller, every interval after the first."""
    if t == 0:
        return st["lg"]
    B = env["budget"]
    plan = B * plan_share[t - 1]
    d = SLACK * B
    if plan < 0.01 * B:  # hold still through the first 1% of the plan
        return st["lg"]
    e = math.log((plan + d) / (spent + d))
    st["integral"] += e
    target = st["lg0"] + KP * e + KI * st["integral"]
    return st["lg"] + max(-MAX_STEP, min(MAX_STEP, target - st["lg"]))


def mpc_policy(forecast):
    """Re-plan every interval: the gamma whose forecast spend on the rest
    of the day equals the budget left."""
    rest = np.cumsum(forecast["spend"][::-1], axis=0)[::-1]  # spend from t to the end, by gamma

    def policy(t, spent, st, env, plan_share):
        left = env["budget"] - spent
        r = rest[t]
        if left <= 0:
            return math.log(G_MIN)
        i = int(np.searchsorted(r, left))  # spend rises with gamma
        return float(LG[min(i, len(LG) - 1)])
    return policy


def _state(t, spent, lg, B, plan_share):
    plan = B * plan_share[max(0, t - 1)] if t else 0.0
    r = math.log((spent + 0.02 * B) / (plan + 0.02 * B))
    ahead = int(np.digitize(r, [-0.3, -0.1, 0.1, 0.3]))
    gb = int(np.digitize(lg, [-0.9, -0.4, 0.0, 0.4]))
    return t // 12, ahead, gb


def _variant(base, rng):
    """A randomized day: traffic and competition up to 20% either way, and
    a surge of 1 to 3 times the traffic for three hours, half the time."""
    scale = np.full(STEPS, rng.uniform(0.8, 1.2))
    if rng.random() < 0.5:
        s = rng.integers(28, 76)  # 07:00 to 19:00
        scale[s:s + 12] *= rng.uniform(1.0, 3.0)
    k = int(round(rng.uniform(-0.2, 0.2) / 0.05))  # competition: the win curve shifts in log price
    sp = np.roll(base["spend"], k, axis=1) * math.exp(-0.05 * k)
    va = np.roll(base["value"], k, axis=1)
    return {"budget": base["budget"], "spend": sp * scale[:, None], "value": va * scale[:, None]}


BETA_AHEAD, BETA_LEFT = 2.0, 2.0  # $ of value per $ of overspend against plan, and per $ left at the end


def train_q(train_env, plan_share, episodes=10000, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((8, 5, 5, len(ACTIONS)))
    B = train_env["budget"]
    for ep in range(episodes):
        env = _variant(train_env, rng)
        eps = max(0.05, 1 - ep / (0.7 * episodes))
        spent, lg = 0.0, 0.0
        s0 = _state(0, spent, lg, B, plan_share)
        for t in range(STEPS):
            a = rng.integers(len(ACTIONS)) if rng.random() < eps else int(np.argmax(Q[s0]))
            lg = min(math.log(G_MAX), max(math.log(G_MIN), lg + ACTIONS[a]))
            sp, v = _at(env["spend"], t, lg), _at(env["value"], t, lg)
            if spent + sp > B:
                f = max(0.0, (B - spent) / sp) if sp > 0 else 0.0
                sp, v = sp * f, v * f
            spent += sp
            r = v - BETA_AHEAD * max(0.0, spent - B * plan_share[t]) / STEPS
            if t == STEPS - 1:
                r -= BETA_LEFT * (B - spent)
            s1 = _state(t + 1, spent, lg, B, plan_share)
            nxt = 0.0 if t == STEPS - 1 else Q[s1].max()
            Q[s0][a] += 0.1 * (r + nxt - Q[s0][a])
            s0 = s1
    return Q


def q_policy(Q, plan_share_unused=None):
    def policy(t, spent, st, env, plan_share):
        s = _state(t, spent, st["lg"], env["budget"], plan_share)
        return st["lg"] + ACTIONS[int(np.argmax(Q[s]))]
    return policy


def compare(ledger_dir, campaign_id, sim_out=None, test_day=None, train_day=None,
            budget=None, episodes=10000, log=print) -> dict:
    days = ledger.days(ledger_dir, "bids")
    test_day = days[-1] if test_day is None else int(test_day)
    train_day = test_day - 1 if train_day is None else int(train_day)
    cid = int(campaign_id)
    if budget is None:
        camp = simout.table(sim_out, "logs", "campaigns") if sim_out else None
        if camp is None or not (camp.campaign_id == cid).any():
            raise RuntimeError("give the campaign's daily budget (--budget)")
        budget = float(camp[camp.campaign_id == cid].daily_budget.iloc[0])
    train = load_day(ledger_dir, cid, train_day, budget)
    test = load_day(ledger_dir, cid, test_day, budget)
    log(f"rlpace: campaign {cid}, budget ${budget:.2f}; trained on day {train_day} "
        f"({train['bids']:,} bids), tested on day {test_day} ({test['bids']:,} bids)")
    # the plan: shaped like the training day's requests, as the bidder's is
    req = train["requests"].astype(float)
    plan_share = np.cumsum(req) / max(req.sum(), 1)
    # check the simulator: the multipliers the bidder used, replayed
    logged = test["logged_gamma"]
    check = run(test, lambda t, s, st, e, p: math.log(logged[t]), plan_share)
    res = {"campaign_id": cid, "budget": round(budget, 2), "train_day": train_day, "test_day": test_day,
           "check": {"actual_spend": test["actual_spend"], "replayed_spend": check["spend"]},
           "policies": {}}
    Q = train_q(train, plan_share, episodes)
    for name, pol in (("PI", pi_policy), ("MPC", mpc_policy(train)), ("Q-learning", q_policy(Q))):
        res["policies"][name] = run(test, pol, plan_share)
    log(f"  the simulator, replaying the bidder's multipliers on day {test_day}: spend "
        f"${check['spend']:.2f}, against ${test['actual_spend']:.2f} actually spent")
    return res


def print_compare(r: dict) -> None:
    c = r["check"]
    print(f"campaign {r['campaign_id']}, budget ${r['budget']}: trained on day {r['train_day']}, "
          f"tested on day {r['test_day']}")
    print(f"  simulator check: the bidder's own multipliers replayed spend ${c['replayed_spend']:.2f}, "
          f"against ${c['actual_spend']:.2f} actually spent")
    print(f"  {'policy':<11}{'spend':>8}{'value':>9}{'value/$':>9}{'delivery':>10}{'tracking':>10}"
          f"{'volatility':>12}  budget out")
    for name, p in r["policies"].items():
        print(f"  {name:<11}{p['spend']:>8.2f}{p['value']:>9.2f}{p['value_per_dollar'] or 0:>9.3f}"
              f"{p['delivery']:>10.3f}{p['tracking_error']:>10.3f}{p['volatility']:>12.4f}  "
              f"{p['budget_out_at'] or '-'}")
    print("  gamma by hour:")
    for name, p in r["policies"].items():
        g = [p["trajectory"][h * 4]["gamma"] for h in range(0, 24, 2)]
        print(f"    {name:<11}" + " ".join(f"{x:4.2f}" for x in g))
