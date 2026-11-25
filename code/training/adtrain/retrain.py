"""Retraining through a market shift, with delayed labels (Chapter 29).

Replays the days around a shift in the Part IV shift dataset: at 12:00 UTC
on its shift day, a new hypercasual hit takes over the genre's audience, and
the genre's games get 40% of their installs from then on. Every model is
LightGBM on the same inputs, predicting an install from an impression, and
each is trained only on what it could have known when it was trained: the
installs whose postbacks had arrived by then.

- daily: retrained at midnight on the last 7 days, on impressions whose
  install window had closed (as in walk);
- daily, with kappa: the daily model times a calibration multiplier per
  advertised app, updated every hour from the installs reported so far,
  corrected for the installs still to come, and bounded per update;
- every 6 hours, naive: retrained every 6 hours on every impression, with
  installs not reported yet counted as no install;
- every 6 hours, delay-weighted: the same rows, with each reported install
  counted as 1/F(e) installs, where F(e) is the share of installs reported
  by the time its impression had waited e: importance weights by elapsed
  time, a delayed feedback correction in the spirit of ES-DFM. With install
  rates of a few percent, weighting the positives alone is close to
  unbiased.

Each is scored against the simulator's true install rates, hour by hour:

    python -m adtrain retrain --snapshot out/snapshot-shift
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, gbm

HOUR, DAY = 3_600_000, 86_400_000
WINDOW_DAYS = 7
WAIT_MS = DAY  # the install window, closed for the daily model's rows
KAPPA_LOOKBACK_MS = DAY
KAPPA_PRIOR = 5.0  # installs' worth of prior toward 1
KAPPA_STEP = 0.2  # the most ln kappa can move in an hour
KAPPA_BOUNDS = (0.25, 4.0)
SEED = 7
ARMS = ("daily", "daily, with kappa", "every 6 hours, naive", "every 6 hours, delay-weighted")
MAX_WEIGHT = 5.0  # the most one reported install counts for


def _features(imp):
    from .pretrain import _tree_X  # the Part III inputs, as LightGBM sees them
    return _tree_X(imp, {}, ())


def _fit(X, y, names, cats, w=None):
    import lightgbm as lgb
    ds = lgb.Dataset(X, y, weight=w, feature_name=names, categorical_feature=cats, free_raw_data=False)
    return lgb.train({**gbm.PARAMS, "objective": "binary", "seed": SEED, "deterministic": True,
                      "num_threads": 1, "verbose": -1}, ds, num_boost_round=gbm.ROUNDS)


DELAY_HORIZON_MS = 3 * DAY  # impressions this old have had their installs reported


def delay_cdf(imp, before_ms):
    """F(t): the share of installs reported within t of the impression, from
    the impressions at least DELAY_HORIZON_MS older than a time, so that
    their late installs have been reported too."""
    m = ((imp.install.to_numpy() == 1) & (imp.install_received_ts_ms.to_numpy() < before_ms)
         & (imp.imp_ts_ms.to_numpy() < before_ms - DELAY_HORIZON_MS))
    d = np.sort((imp.install_received_ts_ms.to_numpy() - imp.imp_ts_ms.to_numpy())[m])
    d = np.maximum(d, 0)

    def F(t):
        return np.searchsorted(d, np.asarray(t), side="right") / max(len(d), 1)
    return F, d


def run(snapshot_dir="out/snapshot-shift", out_dir=None, shift_day=None, log=print) -> dict:
    from . import snapshot
    s = snapshot.load(snapshot_dir)
    imp, truth, apps = s["impressions"], s["truth"], s["apps"]
    if "install_received_ts_ms" not in imp:
        raise SystemExit("this snapshot has no install arrival times: use the Part IV shift dataset "
                         "(part4-shift.zip), or build one with adtrain 0.17 or later")
    imp = imp.reset_index(drop=True)
    tr = truth.set_index("request_id").reindex(imp.request_id)
    p_true = (tr.p_click * tr.p_install_click + (1 - tr.p_click) * tr.p_install_view).to_numpy(float)
    genre = imp.adv_bundle.map(dict(zip(apps.bundle, apps.genre))).fillna("").to_numpy()
    last_day = int(imp.day.max())
    shift_day = int(shift_day if shift_day is not None else last_day - 3)
    shift = shift_day * DAY + 12 * HOUR
    t_imp = imp.imp_ts_ms.to_numpy()
    recv = imp.install_received_ts_ms.to_numpy()
    inst = imp.install.to_numpy() == 1
    start, end = (shift_day - 1) * DAY, (last_day + 1) * DAY  # the replay: from the day before the shift
    log(f"retrain: {len(imp):,} impressions, days 0 to {last_day}; shift at day {shift_day} 12:00; "
        f"replaying from day {shift_day - 1}")
    X, names = _features(imp)
    cats = [c for c in gbm.CATEGORICAL if c in names]
    F, delays = delay_cdf(imp, start)
    hc = genre == "hypercasual"

    def label_at(T, rows):
        return (inst[rows] & (recv[rows] <= T)).astype(float)

    pred = {a: np.full(len(imp), np.nan) for a in ARMS}
    fits, t0 = 0, time.time()
    # daily: at midnight, on the last 7 days with closed windows
    daily = {}  # each day's model's predictions on that day and the day before, for kappa
    for T in range(start, end, DAY):
        rows = np.flatnonzero((t_imp >= T - WINDOW_DAYS * DAY) & (t_imp < T - WAIT_MS))
        b = _fit(X[rows], label_at(T, rows), names, cats)
        fits += 1
        now = np.flatnonzero((t_imp >= T) & (t_imp < T + DAY))
        pred["daily"][now] = b.predict(X[now])
        near = np.flatnonzero((t_imp >= T - KAPPA_LOOKBACK_MS) & (t_imp < T + DAY))
        daily[T] = pd.Series(b.predict(X[near]), index=near)
    log(f"retrain: daily models, {fits} fits")
    # every 6 hours, naive and delay-weighted
    for T in range(start, end, 6 * HOUR):
        rows = np.flatnonzero((t_imp >= T - WINDOW_DAYS * DAY) & (t_imp < T))
        y = label_at(T, rows)
        b = _fit(X[rows], y, names, cats)
        w = np.where(y > 0, np.minimum(1 / np.maximum(F(T - t_imp[rows]), 1e-9), MAX_WEIGHT), 1.0)
        bw = _fit(X[rows], y, names, cats, w)
        fits += 2
        now = np.flatnonzero((t_imp >= T) & (t_imp < T + 6 * HOUR))
        pred["every 6 hours, naive"][now] = b.predict(X[now])
        pred["every 6 hours, delay-weighted"][now] = bw.predict(X[now])
    log(f"retrain: every 6 hours, {fits} fits in all, {time.time() - t0:.0f} s")
    # kappa per advertised app, every hour, on top of the daily model: the
    # installs reported for the last day's impressions, over what the model
    # in use predicts for them times the share reported so far; it starts
    # again at 1 when the model is retrained
    kap = pd.Series(1.0, index=sorted(set(imp.adv_bundle)))
    adv = imp.adv_bundle.to_numpy()
    kappa_trace = []
    for T in range(start, end, HOUR):
        model = daily[T - (T - start) % DAY]
        if T % DAY == start % DAY:
            kap[:] = 1.0
        rows = np.flatnonzero((t_imp >= T - KAPPA_LOOKBACK_MS) & (t_imp < T))
        if len(rows):
            df = pd.DataFrame({"app": adv[rows], "obs": label_at(T, rows),
                               "exp": model.reindex(rows).to_numpy() * F(T - t_imp[rows])})
            g = df.groupby("app")[["obs", "exp"]].sum()
            target = (g.obs + KAPPA_PRIOR) / (g.exp + KAPPA_PRIOR)
            cur = kap.reindex(target.index)
            step = np.clip(np.log(target) - np.log(cur), -KAPPA_STEP, KAPPA_STEP)
            kap.loc[target.index] = np.clip(np.exp(np.log(cur) + step), *KAPPA_BOUNDS)
        now = np.flatnonzero((t_imp >= T) & (t_imp < T + HOUR))
        pred["daily, with kappa"][now] = pred["daily"][now] * kap.reindex(adv[now]).to_numpy()
        hc_apps = sorted(set(adv[hc]))
        kappa_trace.append({"hour": (T - shift) / HOUR,
                            "hypercasual": round(float(np.exp(np.log(kap.reindex(hc_apps)).mean())), 3)})

    # calibration against the truth: the labels count the installs credited
    # to us, a share of the installs the ads cause, so a well-calibrated model
    # predicts the true rate times that share, measured on the week before
    # the replay
    wk = (t_imp >= start - WINDOW_DAYS * DAY) & (t_imp < start)
    credit = {g_: float(inst[wk & gm].sum() / p_true[wk & gm].sum()) for g_, gm in (("hypercasual", hc), ("other genres", ~hc))}
    phases = [("day before the shift", shift - DAY, shift), ("first 6 hours", shift, shift + 6 * HOUR),
              ("6 to 24 hours", shift + 6 * HOUR, shift + DAY), ("second day", shift + DAY, shift + 2 * DAY),
              ("third day", shift + 2 * DAY, shift + 3 * DAY)]
    rep = {"model": "retrain", "adtrain_version": __version__, "shift_day": shift_day,
           "impressions_replayed": int(((t_imp >= start) & (t_imp < end)).sum()), "fits": fits,
           "seconds": round(time.time() - t0, 1),
           "install_delay_hours": {q: round(float(np.percentile(delays, q)) / HOUR, 1) for q in (50, 90, 99)},
           "share_reported_within_6h": round(float(F(6 * HOUR)), 3),
           "credited_share": {k: round(v, 3) for k, v in credit.items()}, "phases": {}}
    for grp, gm in (("hypercasual", hc), ("other genres", ~hc)):
        rep["phases"][grp] = []
        for name, a, b in phases:
            m = gm & (t_imp >= a) & (t_imp < b)
            if m.sum() < 200:
                continue
            row = {"phase": name, "impressions": int(m.sum()), "true_rate": round(float(p_true[m].mean()), 5)}
            for arm in ARMS:
                row[arm] = round(float(np.nansum(pred[arm][m]) / (p_true[m].sum() * credit[grp])), 3)
            rep["phases"][grp].append(row)
    hours = []
    for T in range(start, end, HOUR):
        for grp, gm in (("hypercasual", hc), ("other genres", ~hc)):
            m = gm & (t_imp >= T) & (t_imp < T + HOUR)
            if m.sum() < 30:
                continue
            hours.append({"hour": (T - shift) / HOUR, "group": grp, "impressions": int(m.sum()),
                          "true_credited": float(p_true[m].sum() * credit[grp]),
                          **{arm: float(np.nansum(pred[arm][m])) for arm in ARMS}})
    out = Path(out_dir or Path(snapshot_dir).parent / "retrain")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hours).to_csv(out / "calibration_by_hour.csv", index=False)
    pd.DataFrame(kappa_trace).to_csv(out / "kappa_by_hour.csv", index=False)
    rep["kappa_hypercasual"] = {f"{int(k['hour'])}h": k["hypercasual"] for k in kappa_trace
                                if k["hour"] in (-1, 6, 24, 48)}
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    rep["out_dir"] = str(out)
    return rep


def print_report(r: dict) -> None:
    d = r["install_delay_hours"]
    print(f"Retraining through a market shift: {r['impressions_replayed']:,} impressions replayed from the day "
          f"before the shift (day {r['shift_day']}, 12:00), {r['fits']} LightGBM fits in {r['seconds']} s")
    print(f"  install postbacks arrive {d[50]} h after the impression at the median, {d[90]} h at the 90th "
          f"percentile; {r['share_reported_within_6h']:.0%} within 6 hours")
    c = r["credited_share"]
    print(f"  calibration: predicted installs over the true installs times the share credited to us "
          f"(hypercasual {c['hypercasual']:.2f}, other genres {c['other genres']:.2f}, the week before); "
          f"1.00 is right")
    arms = ARMS
    short = ("daily", "+ kappa", "6h naive", "6h weighted")
    for grp, rows in r["phases"].items():
        print(f"    {grp}:")
        print(f"      {'phase':<22}{'impressions':>12}{'true rate':>11}" + "".join(f"{s_:>12}" for s_ in short))
        for row in rows:
            print(f"      {row['phase']:<22}{row['impressions']:>12,}{row['true_rate']:>11.4f}"
                  + "".join(f"{row[a]:>12.2f}" for a in arms))
    k = r["kappa_hypercasual"]
    print("  kappa for the hypercasual apps (geometric mean), hours after the shift: "
          + ", ".join(f"{h} {v:.2f}" for h, v in k.items()))
    print(f"  wrote calibration by hour, kappa by hour, and the report to {r.get('out_dir')}")
