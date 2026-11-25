"""Uplift models from a lift study (Chapter 32).

During a lift study, the bidder bids as usual for half the devices and
only ghost-bids for the other half (and for the 5% holdout of Chapter 23),
so whether a device sees our ads is random. The unit is a device and an
advertiser app on one day, from our first (real or ghost) bid for the app
on the device that day. Its label is an install of the app, from any
source, within WINDOW_DAYS of that bid: the installs the MMP credited to us, plus the rest
from the advertiser's shared install data. Three models of the label:

    mu1      the install rate of devices we bid on    (LightGBM, treated rows)
    mu0      the install rate of held-out devices     (LightGBM, control rows)
    mu_attr  the rate of installs credited to us      (LightGBM, treated rows)
    TO    the uplift itself, from the transformed outcome
          Z = y (T - e) / (e (1 - e)), whose mean is the uplift (LightGBM
          regression on every row)

The T-learner's uplift is mu1 - mu0: the installs a bid causes. Pricing
needs it per install we get credit for, since that's what the conversion
models predict and the advertiser pays for: the incrementality ratio
r = (mu1 - mu0) / mu_attr. The attributed model mu_attr also ranks the rows
for comparison, as the conversion models would.

Evaluation is on devices the models never saw (a third, by device):
Qini curves on the randomized rows, and, with the simulator's truth, the
installs our ads really caused. The bundle (the three models, served natively
by the bidder) goes to models/uplift/upl-NNNN; the bidder tests it with a
tilt A/B test rather than deploying it.

    python -m adtrain uplift --ledger out/ledger --models out/models --sim-out out/sim
"""
from __future__ import annotations

import json
import math
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import __version__, gbm, ledger, simout

DAY_MS = 86_400_000
WINDOW_DAYS = 1  # an install counts within this long of the day's first bid
EVAL_SHARE = 1 / 3  # devices held out for evaluation
PARAMS = {"learning_rate": 0.05, "num_leaves": 5, "min_data_in_leaf": 200,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
          "lambda_l2": 1.0, "verbose": -1, "seed": 0, "deterministic": True,
          "num_threads": 1, "max_cat_to_onehot": 4, "cat_smooth": 10}
ROUNDS = 150
# A few inputs only. The randomized rows have a few hundred installs, and a
# model with every input learns noise; the signals of lift in Chapter 32
# that the bidder has are how much the device plays (its request rate), how
# often we've shown it ads, and the app, supply, and country. The rest are
# left out (missing), so the bidder's trees never split on them.
INPUTS = ("adv_app", "placement", "country", "exchange", "our_imps_today",
          "device_prior_installs")
EXCLUDE = tuple(k for k in gbm.NUMERIC if k not in INPUTS)
EXCLUDE_CAT = tuple(k for k in gbm.CATEGORICAL if k not in INPUTS)


def _ifa(s):
    return (s or "").lower()


def study_days(models_dir):
    """The simulated days of the newest lift study, from its record, or
    None: its start to its end (or the newest day, if it's running)."""
    recs = []
    for f in sorted((Path(models_dir) / "experiments").glob("exp-*.json")):
        try:
            e = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if (e.get("config") or {}).get("kind") == "study":
            recs.append(e)
    if not recs:
        return None
    e = recs[-1]
    first = int(e.get("started_ms") or 0) // DAY_MS
    last = int(e["ended_ms"]) // DAY_MS if e.get("ended_ms") else None
    return first, last


def load(ledger_dir, sim_out, log=print, days=None) -> dict:
    """The study's rows: one per device, app, and day, with group, label,
    and inputs, for apps whose advertisers share their install data."""
    all_days = ledger.days(ledger_dir, "bids")
    if not all_days:
        raise RuntimeError("the ledger has no bids")
    last_ms = (max(all_days) + 1) * DAY_MS
    if days is not None:
        lo, hi = days
        all_days = [d for d in all_days if d >= lo and (hi is None or d <= hi)]
    window = WINDOW_DAYS * DAY_MS
    camp = simout.table(sim_out, "logs", "campaigns")
    meta = simout.table(sim_out, "logs", "app_metadata")
    na = simout.table(sim_out, "logs", "nonattributed")
    if camp is None or meta is None or na is None:
        raise RuntimeError("needs the simulator's campaign, app, and shared install logs (--sim-out)")
    bundle = dict(zip(meta.app_id, meta.bundle))
    camp_app = {int(c): bundle[a] for c, a in zip(camp.campaign_id, camp.app_id)}
    na = na[na.event_name == "install"]
    sharing = set(na.app_bundle)
    first, rids = {}, defaultdict(list)
    for b in ledger.read(ledger_dir, "bids", all_days):
        d = b.get("decision") or {}
        act = d.get("action")
        if act not in ("bid", "ghost") or not b.get("features"):
            continue
        a = camp_app.get(d.get("campaign_id"))
        ifa = _ifa((b.get("request") or {}).get("ifa"))
        if not a or not ifa or a not in sharing:
            continue
        k = (a, ifa, int(b["ts_ms"]) // DAY_MS)
        if k in first:
            t0 = first[k][0]
            if act == "bid" and int(b["ts_ms"]) < t0 + window:
                rids[k].append(b["request_id"])
            continue
        first[k] = (int(b["ts_ms"]), act == "bid", b["features"], b.get("ad") or {})
        if act == "bid":
            rids[k].append(b["request_id"])
    # installs from any source, and the ones credited to us
    inst, credited, seen = {}, {}, set()
    for p in ledger.read(ledger_dir, "postbacks"):
        f = p.get("fields") or {}
        if f.get("event_name") != "install" or f.get("postback_id") in seen:
            continue
        seen.add(f.get("postback_id"))
        k = (f.get("app_bundle"), _ifa(f.get("advertising_id")))
        ts = int(float(f.get("event_ts_ms") or 0))
        inst[k] = min(inst.get(k, ts), ts)
        credited[k] = min(credited.get(k, ts), ts)
    for a, ifa, ts in zip(na.app_bundle, na.advertising_id, na.event_ts_ms):
        k = (a, _ifa(ifa))
        inst[k] = min(inst.get(k, int(ts)), int(ts))
    # the truth: installs our ads caused
    caused = {}
    ti = simout.table(sim_out, "truth", "installs", ["app_id", "caused", "ts_ms", "device_id"])
    if ti is not None:
        ti = ti[ti.caused.astype(bool)]
        for a, dev, ts in zip(ti.app_id, ti.device_id, ti.ts_ms):
            k = (bundle.get(a), _ifa(dev))
            caused[k] = min(caused.get(k, int(ts)), int(ts))
    # the truth: each impression's lift, the chance it causes an install
    lift = {}
    tl = simout.table(sim_out, "truth", "impressions", ["impression_id", "lift"])
    if tl is not None:
        lift = dict(zip(tl.impression_id, tl.lift.astype(float)))
    rows = defaultdict(list)
    for key, (t0, treated, fv, ad) in first.items():
        if t0 + window > last_ms:
            continue  # its window isn't over yet
        k = key[:2]
        ti_ = inst.get(k)
        if ti_ is not None and ti_ < t0:
            continue  # installed before this bid: can't install again
        within = lambda t: t is not None and t0 <= t < t0 + window
        rows["app"].append(k[0])
        rows["ifa"].append(k[1])
        rows["day"].append(t0 // DAY_MS)
        rows["t"].append(int(treated))
        rows["y"].append(int(within(ti_)))
        rows["y_attr"].append(int(within(credited.get(k))))
        rows["caused"].append(int(within(caused.get(k))))
        # the true uplift of bidding on the pair: the lift of the
        # impressions our bids won in its window
        rows["true_lift"].append(sum(lift.get("i" + r, 0.0) for r in rids.get(key, ())))
        rows["raw"].append(gbm.raw_values(fv, ad))
    if not rows:
        raise RuntimeError("no study rows yet: run a lift study, and wait a day for its labels")
    out = {k: np.array(v) if k != "raw" else v for k, v in rows.items()}
    out["truth"] = ti is not None
    out["last_day"] = max(all_days)
    log(f"uplift: {len(out['t']):,} device-app-days on {len(sharing)} apps, days "
        f"{int(out['day'].min())}-{int(out['day'].max())}: {int(out['t'].sum()):,} bid on, "
        f"{int((1 - out['t']).sum()):,} held out; installs {int(out['y'][out['t'] == 1].sum())} "
        f"and {int(out['y'][out['t'] == 0].sum())}")
    return out


SERVED_SAMPLE = 20_000


R_MAX = 1.5  # our ads can cause installs another network gets credit for


def ratio(p1, p0, pa):
    """The incrementality ratio: installs caused per install credited to us."""
    return np.clip((p1 - p0) / np.maximum(pa, 1e-9), 0, R_MAX)


def served_mean(ledger_dir, day, vocab, m1, m0, ma) -> float:
    """The average incremental share over a sample of a day's real bids
    for install campaigns."""
    raws = []
    for b in ledger.read(ledger_dir, "bids", [day]):
        d = b.get("decision") or {}
        if d.get("action") != "bid" or not b.get("features"):
            continue
        if ((b.get("ad") or {}).get("goal") or "").upper() not in ("CPI", "ROAS", "AD_ROAS"):
            continue
        if zlib.crc32(b["request_id"].encode()) % 4 == 0:
            raws.append(gbm.raw_values(b["features"], b.get("ad") or {}))
        if len(raws) >= SERVED_SAMPLE:
            break
    if not raws:
        return 0.5
    X = _matrix(raws, vocab)
    return float(ratio(m1.predict(X), m0.predict(X), ma.predict(X)).mean())


def _matrix(raws, vocab):
    X = np.array([gbm.encode(r, vocab) for r in raws], float)
    for k in EXCLUDE:
        X[:, len(gbm.CATEGORICAL) + gbm.NUMERIC.index(k)] = np.nan
    for k in EXCLUDE_CAT:
        X[:, gbm.CATEGORICAL.index(k)] = np.nan
    dv = gbm.derived([r["device_requests"] for r in raws], [r["seconds_since_first_seen"] for r in raws])
    return np.hstack([X, dv])


def _names():
    return list(gbm.CATEGORICAL) + list(gbm.NUMERIC) + list(gbm.DERIVED)


def _fit(X, y, objective="binary"):
    import lightgbm as lgb
    ds = lgb.Dataset(X, y, feature_name=_names(), categorical_feature=list(gbm.CATEGORICAL),
                     free_raw_data=False)
    return lgb.train({**PARAMS, "objective": objective}, ds, num_boost_round=ROUNDS)


def qini(score, y, t, points=10):
    """The Qini curve: rank by score, and for the top share phi, the
    incremental installs Y_T - Y_C N_T / N_C, as a share of the total."""
    o = np.argsort(-score, kind="stable")
    y, t = y[o].astype(float), t[o].astype(float)
    yt, yc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    nt, nc = np.cumsum(t), np.cumsum(1 - t)
    q = yt - yc * nt / np.maximum(nc, 1)
    total = q[-1] if q[-1] > 0 else 1.0
    n = len(y)
    curve = [round(float(q[max(1, int(round(n * f))) - 1] / total), 3) for f in np.arange(1, points + 1) / points]
    # the Qini coefficient: the area between the curve and random targeting
    fr = np.arange(1, n + 1) / n
    coef = float(np.mean(q / total - fr))
    return curve, round(coef, 3)


def _share_top(score, x, frac=1 / 3):
    o = np.argsort(-score, kind="stable")[: max(1, int(len(score) * frac))]
    return round(float(x[o].sum() / max(x.sum(), 1e-9)), 3)


def train(ledger_dir, models_dir, sim_out, log=print) -> tuple[str, dict]:
    days = study_days(models_dir)
    if days is None:
        raise RuntimeError("no lift study yet: start one on the Experiments tab (Chapter 32)")
    data = load(ledger_dir, sim_out, log, days)
    raws, t, y = data["raw"], data["t"], data["y"]
    ev = np.array([zlib.crc32(("uplift:" + i).encode()) % 1000 < EVAL_SHARE * 1000 for i in data["ifa"]])
    tr = ~ev
    vocab = gbm.build_vocab([r for r, m in zip(raws, tr) if m], min_count=10)
    X = _matrix(raws, vocab)
    m1 = _fit(X[tr & (t == 1)], y[tr & (t == 1)])
    m0 = _fit(X[tr & (t == 0)], y[tr & (t == 0)])
    e = float(t[tr].mean())
    z = y * (t - e) / (e * (1 - e))
    mto = _fit(X[tr], z[tr], "regression")
    mat = _fit(X[tr & (t == 1)], data["y_attr"][tr & (t == 1)])
    p1, p0 = m1.predict(X), m0.predict(X)
    u_t, u_to, p_attr = p1 - p0, mto.predict(X), mat.predict(X)
    r = ratio(p1, p0, p_attr)
    # the tilt divides by the average share over the bids the bidder makes,
    # not over these rows (one per device, app, and day: heavy players
    # make most of the bids), so the average bid stays about where it was
    mean_r = served_mean(ledger_dir, data["last_day"], vocab, m1, m0, mat)
    rng = np.random.default_rng(0)
    rankers = {"uplift (T-learner)": u_t, "uplift (transformed outcome)": u_to,
               "attributed install model": p_attr, "random": rng.random(len(y))}
    rep = {"model": "uplift", "adtrain_version": __version__, "window_days": WINDOW_DAYS,
           "days": [int(data["day"].min()), int(data["day"].max())],
           "rows": int(len(y)), "train_rows": int(tr.sum()), "eval_rows": int(ev.sum()),
           "treated_share": round(e, 3),
           "installs": {"treated": int(y[t == 1].sum()), "control": int(y[t == 0].sum()),
                        "treated_rows": int((t == 1).sum()), "control_rows": int((t == 0).sum())},
           "mean_incremental_share": round(mean_r, 4),
           "rows_incremental_share": round(float(r[t == 1].mean()), 4), "qini": {}}
    for name, s in rankers.items():
        curve, coef = qini(s[ev], y[ev], t[ev])
        q = {"curve": curve, "coefficient": coef, "top_third": curve[2] if len(curve) > 2 else None}
        if data["truth"]:
            tm = ev & (t == 1)
            # among devices we bid on: the share of the installs our ads
            # caused, and of the installs credited to us, in its top third
            q["true_caused_top_third"] = _share_top(s[tm], data["caused"][tm])
            q["credited_top_third"] = _share_top(s[tm], data["y_attr"][tm])
        rep["qini"][name] = q
    # by how much the device plays: measured from the randomized rows,
    # predicted, and true
    rpd = X[:, -1]
    ok = np.isfinite(rpd)
    edges = np.quantile(rpd[ok], [0.2, 0.4, 0.6, 0.8]) if ok.sum() > 10 else []
    fifth = np.where(ok, np.digitize(np.nan_to_num(rpd), edges), -1)
    rep["by_requests_per_day"] = []
    for k in range(5):
        m = fifth == k
        rt = y[m & (t == 1)].mean() if (m & (t == 1)).any() else float("nan")
        rc = y[m & (t == 0)].mean() if (m & (t == 0)).any() else float("nan")
        ra = data["y_attr"][m & (t == 1)].mean() if (m & (t == 1)).any() else float("nan")
        row = {"fifth": k + 1, "rows": int(m.sum()), "rate_treated": _r(rt), "rate_control": _r(rc),
               "rate_credited": _r(ra),
               "measured_share": _r((rt - rc) / ra if ra > 0 else float("nan")),
               # the share of all its installs, from any source, our ads cause
               "measured_share_all": _r((rt - rc) / rt if rt > 0 else float("nan")),
               "predicted_share": _r(r[m & (t == 1)].mean() if (m & (t == 1)).any() else float("nan"))}
        if data["truth"]:
            mt = m & (t == 1)
            row["true_share"] = _r(data["caused"][mt].sum() / max(data["y_attr"][mt].sum(), 1))
            row["true_share_all"] = _r(data["caused"][mt].sum() / max(data["y"][mt].sum(), 1))
        rep["by_requests_per_day"].append(row)
    if data["truth"]:
        tm = t == 1
        rep["true_incremental_share"] = _r(data["caused"][tm].sum() / max(data["y_attr"][tm].sum(), 1))
        rep["true_credited_share"] = _r(
            ((data["caused"] == 1) & (data["y_attr"] == 1))[tm].sum() / max(data["y_attr"][tm].sum(), 1))
    for name, q in rep["qini"].items():
        log(f"  {name}: Qini {q['coefficient']}, top third {q['top_third']} of incremental installs"
            + (f"; true: {q['true_caused_top_third']} of caused, {q['credited_top_third']} of credited"
               if "true_caused_top_third" in q else ""))
    return _publish(models_dir, {"mu1": m1, "mu0": m0, "mu_attr": mat}, vocab, rep, data, X, log)


def _publish(models_dir, models, vocab, rep, data, X, log):
    root = Path(models_dir) / "uplift"
    root.mkdir(parents=True, exist_ok=True)
    n = max([int(p.name[4:]) for p in root.glob("upl-[0-9]*") if p.name[4:].isdigit()], default=0) + 1
    version = f"upl-{n:04d}"
    d = root / version
    d.mkdir()
    spec = {"model": "uplift", "version": version, "adtrain_version": __version__,
            "categorical": list(gbm.CATEGORICAL), "numeric": list(gbm.NUMERIC),
            "zero_if_missing": list(gbm.ZERO_IF_MISSING), "other": gbm.OTHER,
            "vocab": vocab, "targets": sorted(models), "derived": list(gbm.DERIVED),
            "uplift_mean": rep["mean_incremental_share"]}
    (d / "model.json").write_text(json.dumps(spec))
    for name, b in models.items():
        (d / f"{name}.json").write_text(json.dumps(b.dump_model()))
        b.save_model(str(d / f"{name}.txt"))
    # golden examples for the bidder's parity check: the logged inputs and
    # each model's prediction
    rng = np.random.default_rng(0)
    pick = rng.choice(len(data["t"]), size=min(200, len(data["t"])), replace=False)
    with open(d / "golden.jsonl", "w") as f:
        for i in sorted(pick):
            raw = data["raw"][i]
            fv = {"strings": {k: raw[k] for k in gbm.CATEGORICAL if k not in gbm_ad_keys()},
                  "numeric": {k: raw[k] for k in gbm.NUMERIC if not (isinstance(raw[k], float) and math.isnan(raw[k]))}}
            for k in gbm.ZERO_IF_MISSING:
                if not fv["numeric"].get(k):
                    fv["numeric"].pop(k, None)
            ad = {k: raw[k] for k in gbm_ad_keys()}
            preds = {name: float(b.predict(X[i:i + 1])[0]) for name, b in models.items()}
            f.write(json.dumps({"features": fv, "ad": ad, "predictions": preds}) + "\n")
    rep["version"] = version
    (d / "report.json").write_text(json.dumps(rep, indent=1))
    log(f"published {d}: mean incremental share {rep['mean_incremental_share']}")
    return str(d), rep


def gbm_ad_keys():
    return ("campaign", "creative", "creative_format", "adv_app", "goal")


def print_report(r: dict) -> None:
    i = r["installs"]
    print(f"{r['version']}: days {r['days'][0]}-{r['days'][1]}, {r['rows']:,} device-app-days "
          f"({r['eval_rows']:,} for evaluation); installs within {r['window_days']} days: "
          f"{i['treated']} of {i['treated_rows']:,} bid on, {i['control']} of {i['control_rows']:,} held out")
    print(f"  incrementality ratio, installs caused per install credited to us: {r['rows_incremental_share']} "
          "predicted on the study's rows"
          + (f" (true {r['true_incremental_share']}; of the installs credited to us, "
             f"{r['true_credited_share']} were caused)" if "true_incremental_share" in r else "")
          + f"; {r['mean_incremental_share']} over the bids we make, which the tilt divides by")
    print("  Qini on held-out devices (share of incremental installs in the top 10%, 20%, ... 100%):")
    for name, q in r["qini"].items():
        print(f"    {name:30s} {' '.join(f'{x:5.2f}' for x in q['curve'])}  coefficient {q['coefficient']}"
              + (f"; top third: {q['true_caused_top_third']} of caused installs, "
                 f"{q['credited_top_third']} of credited" if "true_caused_top_third" in q else ""))
    print("  by requests per day, fifths: install rate bid on / held out / credited; share of all installs "
          "caused, measured and true; ratio to credited, measured, predicted, true")
    for b in r["by_requests_per_day"]:
        print(f"    {b['fifth']}  {b['rows']:6,} rows  {b['rate_treated']} / {b['rate_control']} / {b['rate_credited']}  "
              f"{b['measured_share_all']} {b.get('true_share_all', '-')}  "
              f"{b['measured_share']}  {b['predicted_share']}  {b.get('true_share', '-')}")


def _r(x, k: int = 4):
    return None if x is None or not np.isfinite(x) else round(float(x), k)
