"""Personalized creative selection on the Part IV dataset (Chapter 36).

Replays the impressions from the first validation day on, in time order. For
each one, a policy picks one of its campaign's creatives, and the outcome is
drawn from that creative's true rates on that device, from the simulated
market's creative truth (the snapshot's creative_truth table, written by
`python -m adsim.creative_truth`). The market gives each device a hidden
preference over creative styles, so the best creative differs by device.

Policies:

- random;
- per-campaign Thompson sampling: a Beta posterior on each creative's install
  rate, the same for every device (Chapter 21);
- personalized Thompson sampling: a Bayesian logistic model of an install
  over the creative's content (its style and format, what a content
  embedding sees), a per-creative intercept, and the device's creative
  history (its click and install rates by style), refitted every 2,000
  impressions with a Laplace approximation, and sampled once per refit;
- best fixed: each campaign's best creative in hindsight;
- oracle: each impression's best creative, the upper bound.

Every policy starts from the training days' logged impressions, and
outcomes use common random numbers, so policies that pick the same creative
get the same outcome. One creative per campaign is new: it's held back until
the first test day, the cold-start check. Install feedback is immediate here;
the delays of Chapter 29 are left out.

    python -m adtrain creatives --snapshot out/snapshot
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__

SEED = 7
REFIT = 2_000
PRIOR = 1.0  # the Gaussian prior's precision on every weight
SMOOTH = 5.0  # impressions' worth of the device's own rate in its style rates
POLICIES = ("random", "per-campaign Thompson", "personalized Thompson", "best fixed (hindsight)", "oracle")
DAY = 86_400_000


class History:
    """Each device's impressions, clicks, and installs, by creative style."""

    def __init__(self, n_dev, n_style):
        self.imp = np.zeros((n_dev, n_style))
        self.clk = np.zeros((n_dev, n_style))
        self.ins = np.zeros((n_dev, n_style))

    def copy(self):
        h = History(0, 0)
        h.imp, h.clk, h.ins = self.imp.copy(), self.clk.copy(), self.ins.copy()
        return h

    def add(self, d, s, c, i):
        self.imp[d, s] += 1
        self.clk[d, s] += c
        self.ins[d, s] += i

    def lift(self, d):
        """Log ratio of the device's click and install rates in each style to
        its overall rate, smoothed toward zero."""
        out = []
        for num in (self.clk, self.ins):
            tot = num[d].sum(-1, keepdims=True) + 0.5
            n = self.imp[d].sum(-1, keepdims=True) + 1
            rate = tot / n
            r = (num[d] + SMOOTH * rate) / (self.imp[d] + SMOOTH)
            out.append(np.log(r / rate))
        return np.concatenate(out, -1)


class Features:
    """x(device, creative): a per-creative intercept, the creative's style
    and format, and the device's lift for the creative's style, by clicks
    and installs."""

    def __init__(self, creatives):
        self.cid = {c: i for i, c in enumerate(creatives.creative_id)}
        self.styles = sorted(set(creatives["style"]))
        self.formats = sorted(set(creatives["format"]))
        self.style = np.array([self.styles.index(s) for s in creatives["style"]])
        self.fmt = np.array([self.formats.index(f) for f in creatives["format"]])
        self.n_c, self.n_s, self.n_f = len(creatives), len(self.styles), len(self.formats)
        self.dim = 1 + self.n_c + self.n_s + self.n_f + 2 * self.n_s

    def x(self, ci, lift):
        """ci: creative indices (n,), lift: device lifts (n, 2 * styles)."""
        n = len(ci)
        X = np.zeros((n, self.dim))
        X[:, 0] = 1
        X[np.arange(n), 1 + ci] = 1
        o = 1 + self.n_c
        X[np.arange(n), o + self.style[ci]] = 1
        o += self.n_s
        X[np.arange(n), o + self.fmt[ci]] = 1
        o += self.n_f
        s = self.style[ci]
        X[np.arange(n), o + s] = lift[np.arange(n), s]  # the click lift for this style
        X[np.arange(n), o + self.n_s + s] = lift[np.arange(n), self.n_s + s]  # the install lift
        return X


def fit_logistic(X, y, w0=None, iters=8):
    """MAP logistic regression with a Gaussian prior, by Newton's method;
    returns the weights and the Hessian, the Laplace posterior's precision."""
    d = X.shape[1]
    w = np.zeros(d) if w0 is None else w0.copy()
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + PRIOR * w
        H = (X * (p * (1 - p))[:, None]).T @ X + PRIOR * np.eye(d)
        w -= np.linalg.solve(H, g)
    p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
    H = (X * (p * (1 - p))[:, None]).T @ X + PRIOR * np.eye(d)
    return w, H


def run(snapshot_dir="out/snapshot", out_dir=None, log=print) -> dict:
    from . import snapshot
    s = snapshot.load(snapshot_dir)
    if "creative_truth" not in s:
        raise SystemExit("this snapshot has no creative truth: run `python -m adsim.creative_truth --snapshot "
                         f"{snapshot_dir}`, or download the Part IV dataset again")
    imp, cr, ct = s["impressions"], s["creatives"].sort_values("creative_id").reset_index(drop=True), s["creative_truth"]
    t0 = time.time()
    F = Features(cr)
    devs = {d: i for i, d in enumerate(sorted(set(imp.device)))}
    imp = imp.sort_values("imp_ts_ms", kind="stable").reset_index(drop=True)
    first = int(s["manifest"]["creative_truth"]["from_day"])
    test_day = min(int(d) for d, sp in s["manifest"]["split"].items() if sp == "test")
    by_camp = cr.groupby("campaign_id").creative_id.apply(sorted).to_dict()
    new = {c: max(v) for c, v in by_camp.items()}  # the new creative of each campaign, from the first test day
    new_set = set(new.values())
    # the training days: everyone's warm start
    warm = imp[imp.day < first]
    warm = warm[~warm.creative_id.isin(new_set)]
    H0 = History(len(devs), F.n_s)
    wd = warm.device.map(devs).to_numpy()
    wc = warm.creative_id.map(F.cid).to_numpy()
    np.add.at(H0.imp, (wd, F.style[wc]), 1)
    np.add.at(H0.clk, (wd, F.style[wc]), warm.click.to_numpy())
    np.add.at(H0.ins, (wd, F.style[wc]), warm.install.to_numpy())
    beta0 = np.ones((F.n_c, 2))
    np.add.at(beta0[:, 0], wc, warm.install.to_numpy())
    np.add.at(beta0[:, 1], wc, 1 - warm.install.to_numpy())
    # the personalized model's training rows from the warm days: features as of the end of the warm days
    Xw = F.x(wc, H0.lift(wd))
    yw = warm.install.to_numpy(float)
    # the replay
    rep_imp = imp[imp.day >= first].reset_index(drop=True)
    P = ct.pivot(index="request_id", columns="creative_id", values="p_install")
    Pc = ct.pivot(index="request_id", columns="creative_id", values="p_click")
    rng = np.random.default_rng(SEED)
    n = len(rep_imp)
    dev = rep_imp.device.map(devs).to_numpy()
    camp = rep_imp.campaign_id.to_numpy()
    tday = rep_imp.day.to_numpy()
    avail = [[c for c in by_camp[k] if c != new[k] or day >= test_day] for k, day in zip(camp, tday)]
    pi = P.reindex(rep_imp.request_id)
    pc = Pc.reindex(rep_imp.request_id)
    U = rng.random((n, 2))  # common random numbers: a click and an install draw per impression
    # best fixed in hindsight: each campaign's creative with the most installs over the replay
    mean_p = pi.groupby(camp).mean()
    best_fixed = {}
    for k in set(camp):
        old = [c for c in by_camp[k] if c != new[k]]
        best_fixed[(k, False)] = max(old, key=lambda c: mean_p.loc[k, c])
        best_fixed[(k, True)] = max(by_camp[k], key=lambda c: mean_p.loc[k, c])
    chosen = {p: np.zeros(n, int) for p in POLICIES}
    exp_inst = {p: np.zeros(n) for p in POLICIES}
    hist = {p: H0.copy() for p in ("per-campaign Thompson", "personalized Thompson")}
    beta = beta0.copy()
    Xs, ys = [Xw], [yw]
    w, H = fit_logistic(Xw, yw)
    w_draw = rng.multivariate_normal(w, np.linalg.inv(H))
    fits = 1
    for i in range(n):
        a = avail[i]
        pa = pi.iloc[i][a].to_numpy(float)
        ca = pc.iloc[i][a].to_numpy(float)
        ci = np.array([F.cid[c] for c in a])
        picks = {
            "random": int(rng.integers(len(a))),
            "oracle": int(pa.argmax()),
            "best fixed (hindsight)": a.index(best_fixed[(camp[i], tday[i] >= test_day)]),
            "per-campaign Thompson": int(rng.beta(beta[ci, 0], beta[ci, 1]).argmax()),
        }
        Xa = F.x(ci, hist["personalized Thompson"].lift(np.full(len(a), dev[i])))
        picks["personalized Thompson"] = int((Xa @ w_draw).argmax())
        for p, j in picks.items():
            chosen[p][i], exp_inst[p][i] = a[j], pa[j]
        # outcomes for the two learning policies, from the shared draws
        for p in ("per-campaign Thompson", "personalized Thompson"):
            j = picks[p]
            clk, ins = float(U[i, 0] < ca[j]), float(U[i, 1] < pa[j])
            hist[p].add(dev[i], F.style[ci[j]], clk, ins)
            if p == "per-campaign Thompson":
                beta[ci[j], 0 if ins else 1] += 1
            else:
                Xs.append(Xa[j:j + 1])
                ys.append(np.array([ins]))
        if (i + 1) % REFIT == 0:
            w, H = fit_logistic(np.vstack(Xs), np.concatenate(ys), w)
            w_draw = rng.multivariate_normal(w, np.linalg.inv(H))
            fits += 1
            log(f"creatives: {i + 1:,} of {n:,} impressions")
    secs = round(time.time() - t0, 1)
    # the report
    test = tday >= test_day
    newc = np.array([c in new_set for c in chosen["oracle"]])
    rep = {"model": "creatives", "adtrain_version": __version__, "impressions": int(n), "fits": fits,
           "seconds": secs, "first_day": first, "new_creatives_from_day": test_day,
           "creatives": int(F.n_c), "campaigns": int(len(by_camp)), "styles": F.styles, "policies": {}}
    rnd, ora = exp_inst["random"], exp_inst["oracle"]
    for p in POLICIES:
        e = exp_inst[p]
        row = {"installs": round(float(e.sum()), 1),
               "vs_random": round(float(e.sum() / rnd.sum() - 1), 4),
               "gain_captured": round(float((e.sum() - rnd.sum()) / (ora.sum() - rnd.sum())), 3),
               "before_test": round(float(e[~test].sum()), 1), "test": round(float(e[test].sum()), 1),
               "new_creative_share_test": round(float(np.isin(chosen[p][test], list(new_set)).mean()), 3)}
        rep["policies"][p] = row
    rep["oracle_new_creative_share_test"] = round(float(newc[test].mean()), 3)
    # by day
    rep["by_day"] = {int(d): {p: round(float(exp_inst[p][tday == d].sum()), 1) for p in POLICIES}
                     for d in sorted(set(tday))}
    # agreement with the oracle, by how much creative history the device had
    hist_n = H0.imp[dev].sum(1)
    seg = {"devices with under 20 impressions before the replay": hist_n < 20,
           "20 to 100": (hist_n >= 20) & (hist_n < 100), "100 or more": hist_n >= 100}
    rep["by_history"] = {k: {"impressions": int(m.sum()),
                             **{p: round(float(exp_inst[p][m].sum() / max(ora[m].sum(), 1e-9)), 3)
                                for p in POLICIES if p != "oracle"}} for k, m in seg.items() if m.sum() > 500}
    out = Path(out_dir or Path(snapshot_dir).parent / "creatives")
    out.mkdir(parents=True, exist_ok=True)
    names = (["intercept"] + [f"creative {c}" for c in cr.creative_id] + [f"style {x}" for x in F.styles]
             + [f"format {x}" for x in F.formats] + [f"click lift x {x}" for x in F.styles]
             + [f"install lift x {x}" for x in F.styles])
    pd.DataFrame({"feature": names, "weight": w, "sd": np.sqrt(np.diag(np.linalg.inv(H)))}) \
        .to_csv(out / "personalized_weights.csv", index=False)
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    rep["out_dir"] = str(out)
    return rep


def print_report(r: dict) -> None:
    print(f"Creative selection: {r['impressions']:,} impressions replayed from day {r['first_day']}, {r['creatives']} "
          f"creatives in {r['campaigns']} campaigns; one new creative per campaign from day {r['new_creatives_from_day']}; "
          f"{r['fits']} refits in {r['seconds']} s")
    print(f"  expected installs, from the true rates of the creatives each policy chose:")
    print(f"    {'policy':<26}{'installs':>10}{'vs random':>11}{'of the oracle gain':>20}{'new creatives, test days':>27}")
    for p, row in r["policies"].items():
        print(f"    {p:<26}{row['installs']:>10,.1f}{row['vs_random']:>+11.1%}{row['gain_captured']:>20.0%}"
              f"{row['new_creative_share_test']:>27.0%}")
    print(f"  the oracle gives the new creatives {r['oracle_new_creative_share_test']:.0%} of the test days' impressions")
    print("  share of the oracle's installs, by the device's creative history before the replay:")
    for k, row in r["by_history"].items():
        print(f"    {k:<52}{row['impressions']:>8,}  " + ", ".join(
            f"{p} {row[p]:.0%}" for p in ("random", "per-campaign Thompson", "personalized Thompson")))
    print(f"  wrote the personalized model's weights and the report to {r.get('out_dir')}")
