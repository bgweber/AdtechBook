"""Offline evaluation from the ledger (Chapter 16).

Three tools, each answering a question metrics on a test set can't:

- replay: how would a challenger's bids differ from the champion's on the
  same requests, and what would that do to wins and spend?
- ope: what would a different campaign selection policy have achieved,
  estimated from the logged selection probabilities?
- periods: what happened, day by day, while each model was serving?

    python -m adtrain replay --day 12 --b gbm-0003
    python -m adtrain ope --policy uniform
    python -m adtrain periods
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import datasets, gbm, ledger, simout

# The crawl policy's pricing (code/dsp/internal/policy/crawl.go).
MARGIN, SHADE, CAP_CPM = 0.2, 0.7, 50.0


# ------------------------------------------------------------------ replay

def _logged_bids(ledger_dir, day):
    """The day's priced bids: what replay can rescore. Requests the bidder
    didn't price, or priced below the floor without logging features,
    are counted but can't be replayed."""
    rows, skipped = [], Counter()
    for b in ledger.read(ledger_dir, "bids", [day]):
        d = b.get("decision") or {}
        act = d.get("action")
        if act != "bid":
            if act == "nobid" and d.get("reason") == "below_floor":
                skipped["below the floor (no FeatureVector logged)"] += 1
            continue
        if not b.get("features") or not b.get("ad"):
            skipped["bid without a FeatureVector"] += 1
            continue
        req = b.get("request") or {}
        rows.append({
            "request_id": b["request_id"], "exchange": req.get("exchange"),
            "placement": req.get("placement"), "floor": float(req.get("bidfloor") or 0),
            "goal": (b["ad"].get("goal") or "").upper(), "campaign_id": d.get("campaign_id"),
            "p_click": float(d.get("p_click") or 0),
            "p_install": float(d.get("p_install_click") or 0),
            "value": float(d.get("advertiser_value") or 0),
            "explore": float(d.get("explore_mult") or 1), "bid": float(d.get("bid") or 0),
            # walk pricing terms (Chapter 21), when the bidder logged them
            "walk": bool(d.get("p_install")), "target": float(d.get("target") or 0),
            "kappa": float(d.get("kappa") or 1), "vpi": float(d.get("value_per_install") or 1),
            "conversion_model": d.get("conversion_model") or "",
            "features": b["features"], "ad": b["ad"]})
    return rows, skipped


def _predict(models_dir, version, rows):
    """A bundle's click and install-after-click predictions for logged
    requests, from the inputs the bidder logged (its device history)."""
    bdir = Path(models_dir) / "conversion" / version
    spec = json.loads((bdir / "model.json").read_text())
    X = np.array([gbm.encode(gbm.raw_values(r["features"], r["ad"]), spec["vocab"])
                  for r in rows], float)
    boost = gbm.load_boosters(bdir)
    if boost:
        return {t: boost[t].predict(X) for t in ("click", "install_click", "install_view") if t in boost}
    # bundles from before adtrain 0.6 have only LightGBM's JSON dump
    return {t: DumpModel(bdir / f"{t}.json").predict(X)
            for t in ("click", "install_click", "install_view") if (bdir / f"{t}.json").exists()}


def _goal_value(r, pc, pic, piv):
    """crawl.go's GoalValue, times the logged calibration multiplier."""
    p_inst = pc * pic + (1 - pc) * piv
    g = r["goal"]
    v = {"CPI": r["target"] * p_inst, "CPA": r["target"] * p_inst * 0.2,
         "ROAS": p_inst * r["vpi"] / r["target"] if r["target"] else 0.0}.get(g, 0.0)
    return v * 1000 * r["kappa"]


def _price(rows, pc, pi, pv=None):
    """Reprice each request as crawl.go does, from new predictions. Walk
    pricing (Chapter 21) is recomputed from its logged terms; otherwise the
    value is linear in the predictions, and the logged value scales."""
    out = np.zeros(len(rows))
    for i, r in enumerate(rows):
        if r.get("walk") and r["goal"] != "CPC" and pv is not None:
            value = _goal_value(r, pc[i], pi[i], pv[i])
        else:
            old = r["p_click"] * (r["p_install"] if r["goal"] != "CPC" else 1)
            new = pc[i] * (pi[i] if r["goal"] != "CPC" else 1)
            value = r["value"] * new / old if old > 0 else 0.0
        media = value * (1 - MARGIN)
        if media < r["floor"]:
            continue  # no bid
        bid = min(media * SHADE * r["explore"], CAP_CPM)
        if r["floor"] > 0:
            bid = max(bid, r["floor"])
        out[i] = round(bid * 1e4) / 1e4
    return out


def _win_curve(rows, won):
    """Observed win rate by exchange and bid level, from the logged bids,
    made non-decreasing in the bid. Exploration multipliers spread the
    bids, which is what makes the curve possible to estimate at all."""
    curves = {}
    df = pd.DataFrame({"ex": [r["exchange"] for r in rows],
                       "lb": np.log([max(r["bid"], 1e-4) for r in rows]), "won": won})
    for ex, g in df.groupby("ex"):
        g = g.sort_values("lb")
        bins = np.array_split(np.arange(len(g)), max(1, min(20, len(g) // 200)))
        bins = [b for b in bins if len(b)]
        x = np.array([g.lb.iloc[b].mean() for b in bins])
        y = _pav([g.won.iloc[b].mean() for b in bins], [len(b) for b in bins])
        curves[ex] = (x, y)
    return curves


def _pav(y, w):
    """Pool adjacent violators: the closest non-decreasing fit, which keeps
    the weighted mean, so the curve still matches the wins it came from."""
    blocks = []  # [value, weight, count]
    for v, n in zip(y, w):
        blocks.append([v, n, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, n2, c2 = blocks.pop()
            v1, n1, c1 = blocks.pop()
            blocks.append([(v1 * n1 + v2 * n2) / (n1 + n2), n1 + n2, c1 + c2])
    return np.array([v for v, _, c in blocks for _ in range(c)])


def _p_win(curves, exchanges, bids):
    p = np.zeros(len(bids))
    for i, (ex, b) in enumerate(zip(exchanges, bids)):
        if b > 0 and ex in curves:
            x, y = curves[ex]
            p[i] = np.interp(math.log(b), x, y)
    return p


def replay(ledger_dir, models_dir, day: int, b: str, a: str = "logged",
           sim_out=None, log=print) -> dict:
    """Rescore a logged day through two versions and compare the bids.
    a and b are bundle versions, or "logged" for what the bidder did."""
    for v in (a, b):
        if v != "logged" and not re.fullmatch(r"[a-z0-9_-]{1,64}", v or ""):
            raise ValueError(f"bad version {v!r}")
    rows, skipped = _logged_bids(ledger_dir, day)
    if not rows:
        raise RuntimeError(f"no priced bids with FeatureVectors on day {day}")
    logged_pc = np.array([r["p_click"] for r in rows])
    logged_pi = np.array([r["p_install"] for r in rows])
    preds, bids = {}, {}
    for side, v in (("a", a), ("b", b)):
        if v == "logged":
            preds[side] = (logged_pc, logged_pi)
            bids[side] = np.array([r["bid"] for r in rows])
        else:
            p = _predict(models_dir, v, rows)
            pi = np.where([r["goal"] == "CPC" for r in rows], logged_pi, p["install_click"])
            preds[side] = (p["click"], pi)
            bids[side] = _price(rows, p["click"], pi, p.get("install_view"))
    # A check that replay prices the way the bidder does: rescoring with the
    # bundle that actually served reproduces the logged bids.
    served = Counter(r["conversion_model"] for r in rows).most_common(1)[0][0]
    check = None
    for side, v in (("a", a), ("b", b)):
        if v not in ("logged", "") and v == served:
            same = [r["conversion_model"] == v for r in rows]
            diff = np.abs(bids[side] - np.array([r["bid"] for r in rows]))[same]
            check = {"version": v, "rows": int(sum(same)),
                     "reproduced": int((diff <= 1e-4).sum())}
    # Outcomes. Observed: did the logged bid win the auction (a win
    # notice), and was the ad then shown and billed (a billing notice)?
    rid = [r["request_id"] for r in rows]
    wins, billed = set(), set()
    for n in ledger.read(ledger_dir, "notices", [day]):
        if n.get("duplicate"):
            continue
        if n.get("type") == "win":
            wins.add(n["request_id"])
        elif n.get("type") == "billing":
            billed.add(n["request_id"])
    won = np.array([x in wins for x in rid], float)
    curves = _win_curve(rows, won)
    ex = [r["exchange"] for r in rows]
    # we pay for wins that render: the billed share of wins, by exchange
    shown = {e: (sum(1 for x, y in zip(rid, ex) if y == e and x in billed) /
                 max(1, sum(1 for x, y in zip(rid, ex) if y == e and x in wins)))
             for e in set(ex)}
    render = np.array([shown[e] for e in ex])
    # The simulator's truth: the highest other bid in each auction, which a
    # real DSP never sees. With it, replay knows exactly what a different
    # bid would have won (if nobody else changed their bids).
    hi = None
    auc = simout.table(sim_out, "truth", "auctions", ["request_id", "true_highest_other"]) \
        if sim_out else None
    if auc is not None:
        hi = auc.drop_duplicates("request_id").set_index("request_id") \
            .true_highest_other.reindex(rid).to_numpy(float)
    res = {"day": day, "a": a, "b": b, "served": served or "click model and priors",
           "rows": len(rows), "not_replayable": dict(skipped), "reproduce_check": check,
           "segments": []}
    res["logged_wins"] = int(won.sum())
    if hi is not None:
        logged_bid = np.array([r["bid"] for r in rows])
        ok = ~np.isnan(hi)
        res["truth_check"] = {"rows": int(ok.sum()),
                              "agree": int(((logged_bid > hi) == (won > 0))[ok].sum())}
    groups = [("all", np.ones(len(rows), bool))] + \
        [(f"exchange {e}", np.array([x == e for x in ex])) for e in sorted(set(ex))] + \
        [(f"goal {g}", np.array([r["goal"] == g for r in rows]))
         for g in sorted({r["goal"] for r in rows})]
    for name, m in groups:
        seg = {"segment": name, "requests": int(m.sum())}
        for side in ("a", "b"):
            bd, rr = bids[side][m], render[m]
            pw = _p_win(curves, [e for e, k in zip(ex, m) if k], bd)
            s = {"bids": int((bd > 0).sum()), "mean_bid": _r(bd[bd > 0].mean() if (bd > 0).any() else 0),
                 "est_wins": _r(pw.sum(), 1), "est_spend": _r((pw * bd * rr).sum() / 1000, 2),
                 "mean_p_click": _r(preds[side][0][m].mean(), 5)}
            if hi is not None:
                h = hi[m]
                w = (bd > 0) & ~np.isnan(h) & (bd > np.nan_to_num(h))
                s["true_wins"] = int(w.sum())
                s["true_spend"] = _r((bd * rr)[w].sum() / 1000, 2)
            seg[side] = s
        both = (bids["a"][m] > 0) & (bids["b"][m] > 0)
        if both.any():
            ratio = bids["b"][m][both] / bids["a"][m][both]
            seg["bid_ratio_quantiles"] = {q: _r(np.quantile(ratio, q / 100), 3)
                                          for q in (10, 25, 50, 75, 90)}
        if hi is not None and name == "all":
            h = np.nan_to_num(hi)
            wa, wb = (bids["a"] > h) & (bids["a"] > 0), (bids["b"] > h) & (bids["b"] > 0)
            gain, lose = wb & ~wa, wa & ~wb
            seg["changing_hands"] = {
                "won_by_b_only": int(gain.sum()), "won_by_a_only": int(lose.sum()),
                # on what B newly wins, how much higher B's value is than A's:
                # those impressions are where B's errors would show up
                "b_over_a_value_on_b_only": _r(float(
                    (preds["b"][0][gain] * preds["b"][1][gain]).sum() /
                    max((preds["a"][0][gain] * preds["a"][1][gain]).sum(), 1e-12)), 3)
                if gain.any() else None}
        res["segments"].append(seg)
    return res


def print_replay(r: dict) -> None:
    print(f"\nreplay of day {r['day']}: A = {r['a']}, B = {r['b']} "
          f"(served that day: {r['served']}), {r['rows']:,} priced bids")
    for k, v in r["not_replayable"].items():
        print(f"  not replayable: {v:,} requests {k}")
    c = r.get("reproduce_check")
    if c:
        print(f"  check: rescoring with {c['version']} reproduces {c['reproduced']:,} "
              f"of {c['rows']:,} logged bids to $0.0001")
    t = r.get("truth_check")
    if t:
        print(f"  check: the simulator's highest other bid explains {t['agree']:,} of "
              f"{t['rows']:,} logged wins and losses")
    print("\n  est: from the win rates the logs show at each bid; true: from the "
          "simulator's highest other bid; $: spend, true when the simulator's truth is there")
    print(f"\n  {'segment':<22}{'bids A':>9}{'bids B':>9}{'mean A':>9}{'mean B':>9}"
          f"{'est wins A':>12}{'est wins B':>12}{'true A':>9}{'true B':>9}"
          f"{'$ A':>10}{'$ B':>10}")
    for s in r["segments"]:
        a, b = s["a"], s["b"]
        print(f"  {s['segment']:<22}{a['bids']:>9,}{b['bids']:>9,}{a['mean_bid']:>9.3f}"
              f"{b['mean_bid']:>9.3f}{a['est_wins']:>12,.0f}{b['est_wins']:>12,.0f}"
              f"{a.get('true_wins', 0):>9,}{b.get('true_wins', 0):>9,}"
              f"{a.get('true_spend', a['est_spend']):>10,.2f}"
              f"{b.get('true_spend', b['est_spend']):>10,.2f}")
    s = r["segments"][0]
    if s.get("bid_ratio_quantiles"):
        print("\n  B's bid / A's bid, quantiles: " + ", ".join(
            f"p{k} {v}" for k, v in s["bid_ratio_quantiles"].items()))
    ch = s.get("changing_hands")
    if ch:
        print(f"  changing hands (truth): B alone wins {ch['won_by_b_only']:,}, "
              f"A alone wins {ch['won_by_a_only']:,}; on B's new wins, B values "
              f"them {ch['b_over_a_value_on_b_only']}x what A does")


class DumpModel:
    """LightGBM's dump_model trees evaluated with numpy, for bundles that
    have no model text file. The same decision rules as the Go evaluator."""

    def __init__(self, path):
        d = json.loads(Path(path).read_text())
        self.trees = [self._flatten(t["tree_structure"]) for t in d["tree_info"]]

    @staticmethod
    def _flatten(root):
        nodes = []

        def walk(n):
            i = len(nodes)
            nodes.append(None)
            if "leaf_value" in n:
                nodes[i] = ("leaf", n["leaf_value"])
                return i
            left, right = walk(n["left_child"]), walk(n["right_child"])
            if n["decision_type"] == "==":
                cats = np.array([int(c) for c in str(n["threshold"]).split("||")])
                nodes[i] = ("cat", n["split_feature"], cats, left, right)
            else:
                nodes[i] = ("num", n["split_feature"], float(n["threshold"]),
                            n.get("missing_type", "None"), bool(n["default_left"]), left, right)
            return i
        walk(root)
        return nodes

    def predict(self, X):
        z = np.zeros(len(X))
        for nodes in self.trees:
            at = np.zeros(len(X), int)
            out = np.zeros(len(X))
            live = np.ones(len(X), bool)
            while live.any():
                for node in np.unique(at[live]):
                    m = live & (at == node)
                    nd = nodes[node]
                    if nd[0] == "leaf":
                        out[m] = nd[1]
                        live[m] = False
                        continue
                    v = X[m, nd[1]]
                    if nd[0] == "cat":
                        left = ~np.isnan(v) & (v >= 0) & np.isin(np.nan_to_num(v, nan=-1).astype(int), nd[2])
                        at[m] = np.where(left, nd[3], nd[4])
                    else:
                        _, f, thr, mt, dl, l, r = nd
                        nan = np.isnan(v)
                        vv = np.where(nan & (mt != "NaN"), 0.0, v)
                        default = (mt == "NaN") & nan | (mt == "Zero") & (vv == 0)
                        left = np.where(default, dl, vv <= thr)
                        at[m] = np.where(left, l, r)
            z += out
        return 1 / (1 + np.exp(-z))


# --------------------------------------------------------------------- ope

POLICIES = ("uniform", "cpi-first")


def _target_prob(policy, d) -> float | None:
    """The probability a policy would have made the logged choice."""
    elig, goals, ncr = d.get("eligible"), d.get("eligible_goals"), d.get("n_creatives")
    if not elig or not ncr or d.get("campaign_id") not in elig:
        return None
    if policy == "uniform":
        return 1 / len(elig) / ncr
    if policy == "cpi-first":
        w = [3.0 if g == "CPI" else 1.0 for g in goals]
        return w[elig.index(d["campaign_id"])] / sum(w) / ncr
    raise ValueError(policy)


def _rewards(ledger_dir, dataset_dir=None):
    """Clicks and installs per request ID, from the newest as-of dataset."""
    d = dataset_dir or newest_dataset(Path(ledger_dir).parent / "datasets")
    if d is None:
        raise RuntimeError("build a dataset first (adtrain dataset)")
    df, m = datasets.read(d)
    df = df[["request_id", "day", "click", "install", "install_final"]]
    return df, m.get("version")


def newest_dataset(root, leak=False):
    root = Path(root) / "conversions"
    for p in sorted(root.glob("conversions-*"), reverse=True):
        try:
            m = json.loads((p / "manifest.json").read_text())
        except OSError:
            continue
        if bool(m["settings"].get("leak")) == leak:
            return p
    return None


def ope(ledger_dir, policy: str, reward: str = "click", days=None,
        dataset_dir=None, log=print) -> dict:
    """Estimate a selection policy's reward per decision from logs, with
    inverse propensity scoring (IPS) and its self-normalized version."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    rew, dver = _rewards(ledger_dir, dataset_dir)
    final_col = "install_final" if reward == "install" else None
    r_by_id = dict(zip(rew.request_id, rew[reward]))
    final_days = set(rew.day[rew[final_col]].unique()) if final_col else set(rew.day.unique())
    by_logging = defaultdict(lambda: {"w": [], "r": []})
    skipped = 0
    for day in (days or ledger.days(ledger_dir, "bids")):
        if day not in final_days:
            continue
        for b in ledger.read(ledger_dir, "bids", [day]):
            d = b.get("decision") or {}
            if d.get("campaign_id", -1) < 0 or not d.get("selection_prob"):
                continue
            p = _target_prob(policy, d)
            if p is None:
                skipped += 1
                continue
            g = by_logging[(d.get("selection") or "budget", day)]
            g["w"].append(p / d["selection_prob"])
            g["r"].append(r_by_id.get(b["request_id"], 0))
    out = {"policy": policy, "reward": reward, "dataset": dver,
           "decisions_without_eligible_set": skipped, "by_logging_policy": []}
    pooled = defaultdict(lambda: {"w": [], "r": [], "days": []})
    for (lp, day), g in sorted(by_logging.items()):
        pooled[lp]["w"] += g["w"]
        pooled[lp]["r"] += g["r"]
        pooled[lp]["days"].append(int(day))
    for lp, g in pooled.items():
        w, r = np.array(g["w"]), np.array(g["r"], float)
        n = len(w)
        ips = (w * r).mean()
        ips_se = (w * r).std(ddof=1) / math.sqrt(n)
        snips = (w * r).sum() / w.sum()
        snips_se = math.sqrt(((w * (r - snips)) ** 2).sum()) / w.sum()
        out["by_logging_policy"].append({
            "logging_policy": lp, "days": g["days"], "decisions": n,
            "logged_value": _r(r.mean(), 5),
            "ips": _r(ips, 5), "ips_ci": [_r(ips - 1.96 * ips_se, 5), _r(ips + 1.96 * ips_se, 5)],
            "snips": _r(snips, 5),
            "snips_ci": [_r(snips - 1.96 * snips_se, 5), _r(snips + 1.96 * snips_se, 5)],
            "effective_sample_size": int(w.sum() ** 2 / (w ** 2).sum()),
            "max_weight": _r(w.max(), 2),
            # when the logging policy IS the target, the honest comparison
            "is_target": lp == policy})
    return out


def print_ope(o: dict) -> None:
    unit = "clicks" if o["reward"] == "click" else "installs"
    print(f"\nselection policy '{o['policy']}': {unit} per decision, "
          f"labels from {o['dataset']}")
    for g in o["by_logging_policy"]:
        days = g["days"]
        span = f"days {days[0]}-{days[-1]}" if days else "no days"
        if g["is_target"]:
            print(f"  ran live ({span}, {g['decisions']:,} decisions): "
                  f"{g['logged_value']:.5f}")
        else:
            print(f"  estimated from '{g['logging_policy']}' logs ({span}, "
                  f"{g['decisions']:,} decisions; logged {g['logged_value']:.5f}):")
            print(f"    IPS   {g['ips']:.5f}  95% CI {g['ips_ci'][0]:.5f}-{g['ips_ci'][1]:.5f}")
            print(f"    SNIPS {g['snips']:.5f}  95% CI {g['snips_ci'][0]:.5f}-{g['snips_ci'][1]:.5f}")
            print(f"    effective sample size {g['effective_sample_size']:,}, "
                  f"largest weight {g['max_weight']}")
    if o["decisions_without_eligible_set"]:
        print(f"  {o['decisions_without_eligible_set']:,} decisions from before the bidder "
              "logged its eligible campaigns (bidder 0.7) were skipped")


# ----------------------------------------------------------------- periods

def periods(ledger_dir, sim_out=None, dataset_dir=None) -> dict:
    """Day by day: which conversion model served, and what it bought."""
    d = dataset_dir or newest_dataset(Path(ledger_dir).parent / "datasets")
    if d is None:
        raise RuntimeError("build a dataset first (adtrain dataset)")
    df, m = datasets.read(d)
    camp = simout.table(sim_out, "logs", "campaigns") if sim_out else None
    payout = dict(zip(camp.campaign_id, camp.target)) if camp is not None else {}
    df = df[df.goal.str.upper() == "CPI"].copy()
    df["payout"] = df.campaign_id.map(payout)
    days = []
    for day, g in df.groupby("day"):
        served = g.logged_model.replace("", "priors").value_counts().index[0]
        spend = g.price_cpm.sum() / 1000
        rev = float((g.install * g.payout).sum()) if payout else None
        days.append({"day": int(day), "serving": served, "impressions": len(g),
                     "spend": _r(spend, 2), "clicks": int(g.click.sum()),
                     "installs": int(g.install.sum()), "payouts": _r(rev, 2),
                     "margin": _r(rev - spend, 2) if rev is not None else None,
                     "installs_final": bool(g.install_final.all())})
    # contiguous runs of days with the same model
    runs = []
    for x in days:
        if runs and runs[-1]["serving"] == x["serving"] and runs[-1]["last"] == x["day"] - 1:
            r = runs[-1]
            r["last"] = x["day"]
            for k in ("impressions", "clicks", "installs"):
                r[k] += x[k]
            for k in ("spend", "payouts", "margin"):
                if x[k] is not None:
                    r[k] = _r(r[k] + x[k], 2)
            r["installs_final"] &= x["installs_final"]
        else:
            runs.append({"first": x["day"], "last": x["day"], **{k: x[k] for k in x if k != "day"}})
    return {"dataset": m.get("version"), "goal": "CPI", "days": days, "periods": runs}


def print_periods(p: dict) -> None:
    print(f"\nCPI campaigns by serving period, from {p['dataset']}")
    print(f"  {'days':<9}{'serving':<12}{'imps':>9}{'spend':>10}{'clicks':>8}{'installs':>10}"
          f"{'payouts':>10}{'margin':>10}{'cost/inst':>11}")
    for r in p["periods"]:
        cpi = r["spend"] / r["installs"] if r["installs"] else float("nan")
        print(f"  {str(r['first']) + '-' + str(r['last']):<9}{r['serving']:<12}"
              f"{r['impressions']:>9,}{r['spend']:>10,.2f}{r['clicks']:>8,}{r['installs']:>10,}"
              f"{(r['payouts'] or 0):>10,.2f}{(r['margin'] or 0):>10,.2f}{cpi:>11.2f}"
              + ("" if r["installs_final"] else "  (installs still arriving)"))


def _r(x, k: int = 4):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else round(x, k)


# ------------------------------------------------------------ experiments

def abtest(ledger_dir, experiment: str, sim_out=None, dataset_dir=None,
           goal: str = "CPI", reps: int = 400, seed: int = 0) -> dict:
    """A/B test readout (Chapter 19): cost per install by arm, the
    difference with a confidence interval from resampling devices, and,
    with the simulator, the same difference from its true install
    probabilities, which carry no outcome noise."""
    d = dataset_dir or newest_dataset(Path(ledger_dir).parent / "datasets")
    if d is None:
        raise RuntimeError("build a dataset first (adtrain dataset)")
    df, m = datasets.read(d)
    if "experiment" not in df.columns:
        raise RuntimeError(f"{m.get('version')} has no experiment columns: build a new dataset")
    df = df[(df.experiment == experiment) & df.arm.isin(["control", "treatment"])]
    if goal:
        df = df[df.goal.str.upper() == goal]
    if df.empty:
        raise RuntimeError(f"no {goal} impressions from {experiment} in {m.get('version')}")
    # Ad ROAS (Chapter 30): the outcome is D3 ad revenue per dollar, so an
    # impression counts once its install, if any, has final D3 revenue
    roas = goal == "AD_ROAS"
    done = df.install_final & ((df.install == 0) | df.d3_final) if roas else df.install_final
    final = done.mean()
    df = df[done]
    if df.empty:
        raise RuntimeError(f"no {experiment} impression's installs are final yet: play more days")
    df = df.assign(spend=df.price_cpm / 1000)
    if roas:
        df = df.assign(revenue=df.d3_revenue.where(df.install == 1, 0.0))
    tt = simout.table(sim_out, "truth", "impressions",
                      ["impression_id", "p_click", "p_install_click", "p_install_view"]
                      + (["expected_d3_ad_revenue"] if roas else [])) \
        if sim_out else None
    if tt is not None:
        t = tt.set_index("impression_id").reindex("i" + df.request_id)
        pc = t.p_click.to_numpy(float)
        df = df.assign(expected=np.nan_to_num(pc * t.p_install_click.to_numpy(float) +
                                              (1 - pc) * t.p_install_view.to_numpy(float)))
        if roas:
            df = df.assign(expected_revenue=df.expected * np.nan_to_num(
                t.expected_d3_ad_revenue.to_numpy(float)))
    cols = ["spend", "install", "click"] + (["expected"] if "expected" in df else [])
    if roas:
        cols += ["revenue"] + (["expected_revenue"] if "expected_revenue" in df else [])
    if "sample_weight" in df and (df.sample_weight != 1).any():
        # a sampled dataset (--sample-rate): each kept row stands for its weight
        df = df.assign(**{c: df[c] * df.sample_weight for c in cols})
    arms, dev = {}, {}
    for arm, g in df.groupby("arm"):
        s = g[cols].sum()
        n_imps = int(round(g.sample_weight.sum())) if "sample_weight" in g else len(g)
        arms[arm] = {"devices": int(g.device_hash.nunique()), "impressions": n_imps,
                     "spend": _r(s.spend, 2), "clicks": int(s.click), "installs": int(s.install),
                     "cpi": _r(s.spend / s.install, 3) if s.install else None}
        if "expected" in s:
            arms[arm]["expected_installs"] = _r(s.expected, 1)
            arms[arm]["true_cpi"] = _r(s.spend / s.expected, 3) if s.expected else None
        if roas:
            arms[arm]["d3_revenue"] = _r(s.revenue, 2)
            arms[arm]["roas"] = _r(s.revenue / s.spend, 3) if s.spend else None
            if "expected_revenue" in s:
                arms[arm]["true_roas"] = _r(s.expected_revenue / s.spend, 3) if s.spend else None
        dev[arm] = g.groupby("device_hash")[cols].sum().to_numpy()
    out = {"experiment": experiment, "dataset": m.get("version"), "goal": goal,
           "share_final": _r(final, 3), "arms": arms}
    if set(dev) == {"control", "treatment"}:
        rng = np.random.default_rng(seed)
        idx = {c: cols.index(c) for c in cols}

        def ratio(a, b, col):
            """treatment CPI over control CPI; for Ad ROAS, treatment ROAS
            over control ROAS"""
            r = (a[:, 0].sum() / a[:, idx[col]].sum()) / (b[:, 0].sum() / b[:, idx[col]].sum())
            return 1 / r if roas else r
        pairs = ((("revenue", "measured"), ("expected_revenue", "true")) if roas
                 else (("install", "measured"), ("expected", "true")))
        for col, key in pairs:
            if col not in idx:
                continue
            point = ratio(dev["treatment"], dev["control"], col)
            boot = []
            for _ in range(reps):
                a = dev["treatment"][rng.integers(0, len(dev["treatment"]), len(dev["treatment"]))]
                b = dev["control"][rng.integers(0, len(dev["control"]), len(dev["control"]))]
                with np.errstate(divide="ignore", invalid="ignore"):
                    boot.append(ratio(a, b, col))
            boot = np.array(boot)
            boot = boot[np.isfinite(boot)]
            out[key] = {"roas_ratio" if roas else "cpi_ratio": _r(point, 3),
                        "ci": [_r(np.quantile(boot, 0.025), 3), _r(np.quantile(boot, 0.975), 3)]}
    return out


def print_abtest(r: dict) -> None:
    print(f"\nA/B test {r['experiment']}, {r['goal']} campaigns, from {r['dataset']} "
          f"({100 * r['share_final']:.0f}% of impressions have final installs; only those count)")
    print(f"  {'arm':<11}{'devices':>9}{'imps':>9}{'spend':>10}{'installs':>10}{'CPI':>8}"
          f"{'expected':>10}{'true CPI':>10}")
    for arm in ("control", "treatment"):
        a = r["arms"].get(arm)
        if a:
            print(f"  {arm:<11}{a['devices']:>9,}{a['impressions']:>9,}{a['spend']:>10,.2f}"
                  f"{a['installs']:>10,}{(a['cpi'] or 0):>8.2f}"
                  f"{a.get('expected_installs', 0):>10,.1f}{(a.get('true_cpi') or 0):>10.2f}")
    if r["goal"] == "AD_ROAS":
        for arm in ("control", "treatment"):
            a = r["arms"].get(arm)
            if a:
                print(f"  {arm:<11}D3 ad revenue ${a['d3_revenue']:,.2f}, ROAS {a['roas']}"
                      + (f", true ROAS {a['true_roas']}" if a.get("true_roas") is not None else ""))
    for key, label in (("measured", "measured from postbacks"), ("true", "from the true rates and values")):
        if key in r:
            x = r[key]
            name, v = (("ROAS", x["roas_ratio"]) if "roas_ratio" in x else ("CPI", x["cpi_ratio"]))
            print(f"  treatment {name} / control {name}, {label}: {v} "
                  f"(95% CI {x['ci'][0]}-{x['ci'][1]})")


def _experiment_days(models_dir, experiment):
    """The simulated days an experiment ran, from its record, so readouts
    don't read the whole ledger."""
    if not re.fullmatch(r"exp-\d{4}", experiment or ""):
        raise ValueError(f"bad experiment {experiment!r}")
    f = Path(models_dir) / "experiments" / f"{experiment}.json"
    try:
        e = json.loads(f.read_text())
    except (OSError, ValueError):
        return None
    first = int(e.get("started_ms") or 0) // 86_400_000
    last = int(e.get("ended_ms") or 0) // 86_400_000 if e.get("ended_ms") else None
    return [d for d in range(first, (last if last is not None else first + 365) + 1)]


def shadow(ledger_dir, models_dir, experiment: str, days=None) -> dict:
    """Shadow parity (Chapter 19): rescore the logged FeatureVectors with
    the challenger offline and compare with what it predicted live."""
    rows, live = [], []
    version = None
    if days is None:
        days = _experiment_days(models_dir, experiment) or ledger.days(ledger_dir, "bids")
    have = set(ledger.days(ledger_dir, "bids"))
    for day in [d for d in days if d in have]:
        for b in ledger.read(ledger_dir, "bids", [day]):
            d = b.get("decision") or {}
            s = d.get("shadow")
            if d.get("experiment") != experiment or not s or not b.get("features") or not b.get("ad"):
                continue
            version = s["model"]
            rows.append({"features": b["features"], "ad": b["ad"]})
            live.append((s["p_click"], s.get("p_install_click") or 0.0))
    if not rows:
        raise RuntimeError(f"no shadow scores logged for {experiment}")
    off = _predict(models_dir, version, rows)
    lc, li = np.array(live).T
    # The bidder doesn't trust a click prediction outside 0.01% to 50%: it
    # prices with a fallback instead (Chapter 12), and logs that. Those
    # rows can't match; count them apart.
    out_of_bounds = (off["click"] < 1e-4) | (off["click"] > 0.5)
    dc = np.abs(off["click"] - lc)[~out_of_bounds]
    if not len(dc):
        dc = np.array([0.0])
    has = li > 0
    di = np.abs(off["install_click"][has] - li[has]) if has.any() else np.array([0.0])
    return {"experiment": experiment, "model": version, "scored": len(rows),
            "click_out_of_bounds": int(out_of_bounds.sum()),
            "click_max_diff": float(dc.max()), "install_max_diff": float(di.max()),
            "click_match": int((dc <= 1e-9).sum()), "click_rows": int((~out_of_bounds).sum()), "install_match": int((di <= 1e-9).sum()),
            "install_rows": int(has.sum())}


def shadetest(ledger_dir, models_dir, experiment: str) -> dict:
    """A shading A/B test (Chapter 20), from the ledger: by arm, the bids,
    the win rate, spend, and surplus, the media value of the impressions
    bought minus what we paid for them."""
    days = _experiment_days(models_dir, experiment) or ledger.days(ledger_dir, "bids")
    have = set(ledger.days(ledger_dir, "bids"))
    days = [d for d in days if d in have]
    arms = {a: {"bids": 0, "wins": 0, "billed": 0, "spend": 0.0, "value": 0.0, "shade": 0.0,
                "win_prob": 0.0, "model_shaded": 0} for a in ("control", "treatment")}
    media = {}
    for day in days:
        for b in ledger.read(ledger_dir, "bids", [day]):
            d = b.get("decision") or {}
            if d.get("experiment") != experiment or d.get("action") != "bid" or d.get("arm") not in arms:
                continue
            a = arms[d["arm"]]
            a["bids"] += 1
            a["shade"] += d.get("shade") or 0
            if d.get("shade_from") == "model":
                a["model_shaded"] += 1
                a["win_prob"] += d.get("win_prob") or 0
            media[b["request_id"]] = (d["arm"], float(d.get("media_value") or 0))
    for day in days + [max(days) + 1] if days else []:
        for n in ledger.read(ledger_dir, "notices", [day]):
            m = media.get(n.get("request_id"))
            if not m or n.get("duplicate"):
                continue
            a = arms[m[0]]
            if n.get("type") == "win":
                a["wins"] += 1
            elif n.get("type") == "billing":
                a["billed"] += 1
                a["spend"] += float(n.get("price") or 0) / 1000
                a["value"] += m[1] / 1000
    out = {"experiment": experiment, "days": days, "arms": {}}
    for k, a in arms.items():
        if not a["bids"]:
            continue
        out["arms"][k] = {
            "bids": a["bids"], "wins": a["wins"], "win_rate": _r(a["wins"] / a["bids"], 4),
            "predicted_win_rate": _r(a["win_prob"] / a["model_shaded"], 4) if a["model_shaded"] else None,
            "mean_shade": _r(a["shade"] / a["bids"], 3), "impressions": a["billed"],
            "spend": _r(a["spend"], 2), "media_value": _r(a["value"], 2),
            "surplus": _r(a["value"] - a["spend"], 2),
            "surplus_per_dollar": _r((a["value"] - a["spend"]) / a["spend"], 3) if a["spend"] else None}
    return out


def print_shadetest(r: dict) -> None:
    d = r["days"]
    print(f"\nshading test {r['experiment']}, days {d[0]}-{d[-1]}" if d else f"\n{r['experiment']}: no days")
    print(f"  {'arm':<11}{'bids':>9}{'win rate':>10}{'predicted':>11}{'shade':>7}{'imps':>8}"
          f"{'spend':>10}{'value':>10}{'surplus':>10}")
    for k in ("control", "treatment"):
        a = r["arms"].get(k)
        if a:
            pw = f"{a['predicted_win_rate']:.3f}" if a["predicted_win_rate"] is not None else "-"
            print(f"  {k:<11}{a['bids']:>9,}{a['win_rate']:>10.3f}{pw:>11}{a['mean_shade']:>7.2f}"
                  f"{a['impressions']:>8,}{a['spend']:>10,.2f}{a['media_value']:>10,.2f}{a['surplus']:>10,.2f}")
