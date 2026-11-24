"""A win rate model for bid shading (Chapter 20).

The label is the win notice: did our bid win the auction? The bid is an
input, as ln(bid) and ln(bid / floor), with a monotone constraint so the
win rate can only rise with the bid. Other inputs describe the request the
way every bidder sees it: supply, geography, time, and device. Nothing
about our campaign or ad goes in: it doesn't change what others bid.

Exchanges that report the minimum bid to win tell us the highest
competing price exactly (or rounded up), not just whether we beat it. Each
report becomes a few extra rows at bids above and below it, labeled by
which side of it they fall on, with a small weight. That spreads the
training bids beyond the narrow band the fixed shade produces.

    python -m adtrain win --ledger out/ledger --models out/models
"""
from __future__ import annotations

import json
import math
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import __version__, ledger, metrics

CATEGORICAL = ("exchange", "placement", "country", "os", "os_major", "make", "model",
               "connection_type", "publisher_app")
NUMERIC = ("bidfloor", "hour_of_day", "rewarded", "interstitial", "device_known",
           "device_requests", "seconds_since_first_seen", "seconds_since_last_seen",
           "ln_bid", "ln_bid_over_floor")
BID_FEATURES = ("ln_bid", "ln_bid_over_floor")
OTHER = "__other__"
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
          "min_data_in_leaf": 100, "feature_fraction": 0.9, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "seed": 0,
          "deterministic": True, "num_threads": 1, "max_cat_to_onehot": 4,
          "cat_smooth": 10, "monotone_constraints_method": "advanced"}
ROUNDS = 200
MTW_MULTS = (0.6, 0.8, 1.25, 1.6)   # pseudo-bids around each reported price
MTW_WEIGHT = 0.5                     # total weight of one report's rows (lambda)


def bid_inputs(fv: dict, bid: float) -> dict:
    """The inputs for one request at one bid. Shared with the Go bidder,
    which fills the two bid features for each candidate bid."""
    s, n = fv.get("strings") or {}, fv.get("numeric") or {}
    out = {k: str(s.get(k) or "") for k in CATEGORICAL}
    for k in NUMERIC:
        out[k] = float(n[k]) if k in n else math.nan
    floor = float(n.get("bidfloor") or 0)
    out["ln_bid"] = math.log(max(bid, 1e-4))
    out["ln_bid_over_floor"] = math.log(max(bid, 1e-4) / floor) if floor > 0 else math.nan
    return out


def _sampled(rid: str, rate: float) -> bool:
    """Keep a bid with chance rate, by a hash of its request ID: the same
    bids every time, and its notices and installs come with it."""
    return rate >= 1.0 or zlib.crc32(rid.encode()) / 2**32 < rate


def load(ledger_dir, days, log=print, sample_rate=1.0):
    """Logged bids with their outcomes, column by column. A laptop-scale
    week has millions of bids: sample_rate keeps a share of them."""
    wins, mtw = set(), {}
    for n in ledger.read(ledger_dir, "notices", days):
        if n.get("duplicate"):
            continue
        if n.get("type") == "win":
            wins.add(n["request_id"])
        if n.get("type") in ("win", "loss") and n.get("min_to_win"):
            mtw[n["request_id"]] = float(n["min_to_win"])
    cols = defaultdict(list)
    for b in ledger.read(ledger_dir, "bids", days):
        d = b.get("decision") or {}
        if d.get("action") != "bid" or not b.get("features"):
            continue
        fv = b["features"]
        rid = b["request_id"]
        if not _sampled(rid, sample_rate):
            continue
        cols["request_id"].append(rid)
        cols["day"].append(int(b.get("day", 0)))
        cols["bid"].append(float(d.get("bid") or 0))
        cols["explore"].append(float(d.get("explore_mult") or 1))
        cols["media"].append(float(d.get("media_value") or 0))
        cols["exchange"].append((b.get("request") or {}).get("exchange") or "")
        cols["fv"].append(json.dumps(fv, separators=(",", ":")))
        cols["win"].append(int(rid in wins))
        cols["mtw"].append(mtw.get(rid, math.nan))
    out = {k: np.array(v, dtype=object if k in ("request_id", "fv", "exchange") else float)
           for k, v in cols.items()}
    log(f"win data: {len(out.get('bid', [])):,} bids on days {days[0]}-{days[-1]}, "
        f"{int(np.nansum(out.get('win', [0])))} wins, "
        f"{int(np.isfinite(out.get('mtw', np.array([]))).sum()):,} minimum-bid-to-win reports")
    return out


def _matrix(fvs, bids, vocab=None):
    raws = [bid_inputs(json.loads(f), b) for f, b in zip(fvs, bids)]
    if vocab is None:
        vocab = {}
        for k in CATEGORICAL:
            counts = defaultdict(int)
            for r in raws:
                counts[sys.intern(r[k])] += 1
            kept = sorted((v for v, c in counts.items() if c >= 20), key=lambda v: (-counts[v], v))
            vocab[k] = {v: i for i, v in enumerate(kept)}
            vocab[k][OTHER] = len(kept)
    X = np.array([[float(vocab[k].get(r[k], vocab[k][OTHER])) for k in CATEGORICAL] +
                  [r[k] for k in NUMERIC] for r in raws], float)
    return X, vocab


# no bid input, so no monotone constraint; LightGBM 4.x crashes with
# monotone_constraints_method set but no constraints and a custom loss
CURVE_PARAMS = {k: v for k, v in PARAMS.items() if k != "monotone_constraints_method"}
CURVE_PARAMS["num_leaves"] = 15
CURVE_ROUNDS = 300
CURVE_SCALES = (0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8)


def _curve_loss(kind, v, mu, s):
    """Negative log likelihood, per row, of a logistic log price to beat
    with location mu and scale s. kind: 1 a win at log bid v (the price
    was at most v), 0 a loss (above v), 2 a reported minimum bid to win
    (the price was v)."""
    z = (v - mu) / s
    F = 1 / (1 + np.exp(-z))
    eps = 1e-12
    return np.where(kind == 1, -np.log(F + eps),
                    np.where(kind == 0, -np.log(1 - F + eps),
                             -np.log(F * (1 - F) + eps) + np.log(s)))


def _curve_objective(kind, v, s):
    """Gradient and Hessian of _curve_loss with respect to mu."""
    def fobj(preds, _ds):
        F = 1 / (1 + np.exp(-(v - preds) / s))
        q = F * (1 - F) / s ** 2
        g = np.where(kind == 1, (1 - F) / s, np.where(kind == 0, -F / s, (1 - 2 * F) / s))
        h = np.where(kind == 2, 2 * q, q)
        return g, np.maximum(h, 1e-6)
    return fobj


def train_curve(fvs, bids, wins, mtws, floors, vocab, log=print):
    """A win curve (Chapter 31): trees that read only the request and
    predict mu(x), the location of the log price to beat, which is
    logistic with scale s. A won bid says the price was at most the bid, a
    lost one that it was above, and a reported minimum bid to win gives
    the price. The scale is fit by trying each on a first pass."""
    import lightgbm as lgb

    kind = np.where(np.isfinite(mtws) & (mtws > 0), 2, wins.astype(int))
    v = np.log(np.maximum(np.where(kind == 2, np.nan_to_num(mtws), bids), 1e-4))
    # the price to beat is at least the floor: a reported price below it
    # is the floor
    v = np.where(kind == 2, np.maximum(v, np.log(np.maximum(floors, 1e-4))), v)
    offset = float(np.median(v))
    X, _ = _matrix(fvs, bids, vocab)
    X[:, [len(CATEGORICAL) + NUMERIC.index(k) for k in BID_FEATURES]] = np.nan
    names = list(CATEGORICAL) + list(NUMERIC)

    def fit(s, rounds):
        ds = lgb.Dataset(X, np.zeros(len(v)), feature_name=names,
                         categorical_feature=list(CATEGORICAL), free_raw_data=False)
        return lgb.train({**CURVE_PARAMS, "objective": _curve_objective(kind, v - offset, s)},
                         ds, num_boost_round=rounds)

    # the scale: a short fit at each, then the best on the training rows
    losses = {}
    for s in CURVE_SCALES:
        b = fit(s, 60)
        losses[s] = float(_curve_loss(kind, v - offset, b.predict(X), s).mean())
    s = min(losses, key=losses.get)
    booster = fit(s, CURVE_ROUNDS)
    log(f"win curve: scale {s} ({', '.join(f'{k}: {l:.3f}' for k, l in losses.items())}), "
        f"{int((kind == 2).sum()):,} reported prices, {int((kind == 1).sum()):,} wins, "
        f"{int((kind == 0).sum()):,} losses")
    return booster, offset, s


def train(ledger_dir, models_dir, days=None, use_mtw=True, last_days=7, sim_out=None, log=print,
          curve=False, curse=False, sample_rate=1.0):
    import lightgbm as lgb

    all_days = ledger.days(ledger_dir, "bids")
    days = days or all_days[-last_days:]
    data = load(ledger_dir, days, log, sample_rate)
    ev_day = max(days)
    tr = data["day"] < ev_day
    ev = ~tr
    # training rows: real bids, plus pseudo-bids around each reported price
    fv_tr, b_tr = list(data["fv"][tr]), list(data["bid"][tr])
    y_tr, w_tr = list(data["win"][tr]), [1.0] * int(tr.sum())
    n_pseudo = 0
    if use_mtw:
        for f, m in zip(data["fv"][tr], data["mtw"][tr]):
            if not np.isfinite(m) or m <= 0:
                continue
            for k in MTW_MULTS:
                fv_tr.append(f)
                b_tr.append(m * k)
                y_tr.append(int(m * k >= m))
                w_tr.append(MTW_WEIGHT / len(MTW_MULTS))
                n_pseudo += 1
    X_tr, vocab = _matrix(fv_tr, b_tr)
    names = list(CATEGORICAL) + list(NUMERIC)
    mono = [1 if n in BID_FEATURES else 0 for n in names]
    ds = lgb.Dataset(X_tr, np.array(y_tr), weight=np.array(w_tr), feature_name=names,
                     categorical_feature=list(CATEGORICAL), free_raw_data=False)
    booster = lgb.train({**PARAMS, "monotone_constraints": mono}, ds, num_boost_round=ROUNDS)
    X_ev, _ = _matrix(data["fv"][ev], data["bid"][ev], vocab)
    p = booster.predict(X_ev)
    y = data["win"][ev]
    rep = {"days": [int(min(days)), int(ev_day)], "eval_day": int(ev_day),
           "train_bids": int(tr.sum()), "pseudo_rows": n_pseudo, "use_mtw": use_mtw,
           "eval_bids": int(ev.sum()), "win_rate": _r(y.mean()),
           "auc": _r(metrics.roc(y, p)[1]), "log_loss": _r(metrics.log_loss(y, p)),
           "calibration": _r(p.sum() / max(y.sum(), 1), 3),
           "by_bid_decile": _buckets(np.log(np.maximum(data["bid"][ev], 1e-4)), y, p, 10),
           "by_explore": _explore(data["explore"][ev], y, p),
           "by_exchange": {ex: {"bids": int(m.sum()), "actual": _r(y[m].mean()), "predicted": _r(p[m].mean())}
                           for ex in sorted(set(data["exchange"][ev]))
                           for m in [data["exchange"][ev] == ex]},
           "importance": _importance(booster, names)}
    if sim_out:
        rep["at_other_bids"] = _truth_curve(sim_out, booster, vocab, data, ev)
    cmodel = None
    if curve:
        floors = np.array([json.loads(f).get("numeric", {}).get("bidfloor", 0) or 0
                           for f in data["fv"]], float)
        cb, off, s = train_curve(data["fv"][tr], data["bid"][tr], data["win"][tr],
                                 data["mtw"][tr] if use_mtw else np.full(int(tr.sum()), np.nan),
                                 floors[tr], vocab, log)
        cmodel = (cb, off, s)
        pc = curve_prob(cmodel, vocab, data["fv"][ev], data["bid"][ev], floors[ev])
        rep["curve"] = {"scale": s, "offset": _r(off, 4),
                        "auc": _r(metrics.roc(y, pc)[1]), "log_loss": _r(metrics.log_loss(y, pc)),
                        "calibration": _r(pc.sum() / max(y.sum(), 1), 3),
                        "by_bid_decile": _buckets(np.log(np.maximum(data["bid"][ev], 1e-4)), y, pc, 10),
                        "importance": _importance(cb, list(CATEGORICAL) + list(NUMERIC))}
        if sim_out:
            rep["curve"]["at_other_bids"] = _truth_curve(
                sim_out, None, vocab, data, ev,
                predict=lambda fv, b: curve_prob(cmodel, vocab, fv, b, floors[ev][_ok_mask(sim_out, data, ev)]))
        log(f"win curve on day {ev_day}: AUC {rep['curve']['auc']} (bid-input model {rep['auc']}), "
            f"log loss {rep['curve']['log_loss']} ({rep['log_loss']})")
    curse_rep = None
    if curse and cmodel:
        # the curse needs final installs: the days at least 2 before the newest
        # (in the simulator, 99% of installs reach us within a day of the bid)
        cdays = [x for x in all_days if x <= max(all_days) - 2][-last_days:]
        curse_rep = curse_table(ledger_dir, cdays, cmodel, vocab, sim_out, log, sample_rate) \
            if cdays else None
        rep["curse"] = curse_rep
    return _publish(models_dir, booster, vocab, rep, data, ev, log, cmodel, curse_rep)


def curve_prob(cmodel, vocab, fvs, bids, floors):
    """The win curve's chance of winning at each bid."""
    cb, off, s = cmodel
    X, _ = _matrix(fvs, bids, vocab)
    X[:, [len(CATEGORICAL) + NUMERIC.index(k) for k in BID_FEATURES]] = np.nan
    mu = off + cb.predict(X)
    p = 1 / (1 + np.exp(-(np.log(np.maximum(bids, 1e-4)) - mu) / s))
    return np.where(bids >= floors, p, 0.0)


def market_price(cmodel, vocab, fvs):
    """The win curve's median price to beat, CPM: the bid that wins half
    the time."""
    cb, off, _ = cmodel
    X, _ = _matrix(fvs, np.ones(len(fvs)), vocab)
    X[:, [len(CATEGORICAL) + NUMERIC.index(k) for k in BID_FEATURES]] = np.nan
    return np.exp(off + cb.predict(X))


CURSE_EDGES = (0.0, 1.0, 1.5, 2.0, 3.0, 0.0)  # value / market price; 0 = no upper end


def curse_table(ledger_dir, days, cmodel, vocab, sim_out=None, log=print, sample_rate=1.0):
    """The winner's curse, measured (Chapter 31): on won impressions of
    install campaigns (CPI, ROAS, Ad ROAS) whose installs are final, installs per predicted install,
    by the margin of the media value over the win curve's market price.
    Wins far above the market return fewer installs than predicted. The
    table goes with the win bundle, and the bidder marks values down by
    it. With the simulator, the same by its true install rates."""
    from . import simout
    wins, cpi = set(), {}
    for n in ledger.read(ledger_dir, "notices", days):
        if n.get("type") == "win" and not n.get("duplicate"):
            wins.add(n["request_id"])
    rows = defaultdict(list)
    for b in ledger.read(ledger_dir, "bids", days):
        d = b.get("decision") or {}
        if d.get("action") != "bid" or b["request_id"] not in wins:
            continue
        if not _sampled(b["request_id"], sample_rate):
            continue
        if ((b.get("ad") or {}).get("goal") or "").upper() not in INSTALL_GOALS or not d.get("p_install"):
            continue
        if d.get("install_rate_from") != "model":
            continue  # the table corrects the deployed conversion model, not the priors
        rows["request_id"].append(b["request_id"])
        rows["fv"].append(json.dumps(b["features"], separators=(",", ":")))
        rows["media"].append(float(d.get("media_value") or 0))
        rows["p_install"].append(float(d["p_install"]))
        rows["kappa"].append(float(d.get("kappa") or 1))
        rows["price"].append(float(d.get("bid") or 0))
    if not rows:
        return None
    installs = set()
    for pb in ledger.read(ledger_dir, "postbacks", ledger.days(ledger_dir, "postbacks")):
        f = pb.get("fields") or pb
        if f.get("event_name") != "install":
            continue
        # the MMP echoes our click ID ("c" + request ID) or impression ID ("i" + ...)
        tid = f.get("click_id") or f.get("impression_id") or ""
        if tid:
            installs.add(tid[1:])
    rid = np.array(rows["request_id"], dtype=object)
    inst = np.array([r in installs for r in rid], float)
    pred = np.array(rows["p_install"]) * np.array(rows["kappa"])
    margin = np.array(rows["media"]) / market_price(cmodel, vocab, rows["fv"])
    true = None
    tt = simout.table(sim_out, "truth", "impressions",
                      ["impression_id", "p_click", "p_install_click", "p_install_view"]) if sim_out else None
    if tt is not None:
        t = tt.set_index("impression_id").reindex("i" + pd_index(rid))
        pc = t.p_click.to_numpy(float)
        true = np.nan_to_num(pc * t.p_install_click.to_numpy(float) + (1 - pc) * t.p_install_view.to_numpy(float))
    out = {"wins": int(len(rid)), "installs": int(inst.sum()), "buckets": []}
    for lo, hi in zip(CURSE_EDGES[:-1], CURSE_EDGES[1:]):
        m = (margin >= lo) & ((margin < hi) if hi > 0 else True)
        if m.sum() == 0:
            continue
        k = inst[m].sum() / max(pred[m].sum(), 1e-9)
        b = {"lo": lo, "hi": hi, "wins": int(m.sum()), "installs": int(inst[m].sum()),
             "predicted": _r(pred[m].sum(), 1), "kappa": _r(k, 3)}
        if true is not None:
            b["true_kappa"] = _r(true[m].sum() / max(pred[m].sum(), 1e-9), 3)
        out["buckets"].append(b)
    # shrink each bucket toward no correction by its count of installs, so a
    # thin bucket can't swing bids: kappa = (installs + K) / (predicted + K)
    for b in out["buckets"]:
        b["kappa_used"] = _r((b["installs"] + CURSE_K) / (b["predicted"] + CURSE_K), 3)
    log("winner's curse on " + f"{out['wins']:,} won install-campaign impressions ({out['installs']} installs): " + ", ".join(
        f"x{b['lo']}-{b['hi'] or 'up'}: {b['kappa']}" + (f" (true {b['true_kappa']})" if 'true_kappa' in b else "")
        for b in out["buckets"]))
    return out


CURSE_K = 20.0
# the goals whose value rests on the install rate: CPI, and the revenue goals
# (value per install x install rate)
INSTALL_GOALS = ("CPI", "ROAS", "AD_ROAS")


def pd_index(rid):
    import pandas as pd
    return pd.Index(rid.astype(str))


def _ok_mask(sim_out, data, ev):
    from . import simout
    t = simout.table(sim_out, "truth", "auctions", ["request_id", "true_highest_other"])
    t = t.drop_duplicates("request_id").set_index("request_id").true_highest_other
    return np.isfinite(t.reindex(data["request_id"][ev]).to_numpy(float))


def _truth_curve(sim_out, booster, vocab, data, ev, predict=None):
    """The test shading depends on: the win rate at bids we didn't make.
    The simulator knows each auction's highest competing price, so it knows
    whether any other bid would have won. A real DSP has only its
    exploration bids and the reported minimum bids to win for this."""
    from . import simout
    t = simout.table(sim_out, "truth", "auctions", ["request_id", "true_highest_other"])
    if t is None:
        return None
    t = t.drop_duplicates("request_id").set_index("request_id").true_highest_other
    rid = data["request_id"][ev]
    h = t.reindex(rid).to_numpy(float)
    fl = np.array([json.loads(f).get("numeric", {}).get("bidfloor", 0) for f in data["fv"][ev]], float)
    to_beat = np.maximum(np.nan_to_num(h), fl)
    ok = np.isfinite(h)
    out = []
    for k in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        b = data["bid"][ev] * k
        if predict is not None:
            p = predict(data["fv"][ev][ok], b[ok])
        else:
            X, _ = _matrix(data["fv"][ev][ok], b[ok], vocab)
            p = booster.predict(X)
        y = (b[ok] >= to_beat[ok]).astype(float)  # the exchange: bid >= max(price, floor)
        out.append({"bid_multiplier": k, "true": _r(y.mean()), "predicted": _r(p.mean()),
                    "log_loss": _r(metrics.log_loss(y, p))})
    return out


def _buckets(x, y, p, n):
    order = np.argsort(x, kind="stable")
    return [{"bids": int(len(ix)), "bid_from": _r(math.exp(x[ix[0]]), 2), "bid_to": _r(math.exp(x[ix[-1]]), 2),
             "actual": _r(y[ix].mean()), "predicted": _r(p[ix].mean())}
            for ix in np.array_split(order, n) if len(ix)]


def _explore(mult, y, p):
    edges = [("shaded only (x1)", (mult > 0.999) & (mult < 1.001)), ("x0.6-0.8", mult < 0.8),
             ("x0.8-1.0", (mult >= 0.8) & (mult < 0.999)), ("x1.0-1.2", (mult > 1.001) & (mult < 1.2)),
             ("x1.2-1.5", mult >= 1.2)]
    return [{"bucket": name, "bids": int(m.sum()), "actual": _r(y[m].mean()), "predicted": _r(p[m].mean())}
            for name, m in edges if m.sum()]


def _importance(booster, names):
    g = booster.feature_importance("gain")
    tot = g.sum() or 1
    return [{"feature": names[i], "gain_share": round(float(g[i] / tot), 3)}
            for i in np.argsort(-g)[:10]]


def _publish(models_dir, booster, vocab, rep, data, ev, log, cmodel=None, curse=None):
    root = Path(models_dir) / "win"
    root.mkdir(parents=True, exist_ok=True)
    n = max([int(p.name[4:]) for p in root.glob("win-[0-9]*") if p.name[4:].isdigit()], default=0) + 1
    version = f"win-{n:04d}"
    d = root / version
    d.mkdir()
    # the Go bundle loader's spec: the bid features are numerics it fills in
    spec = {"model": "win", "version": version, "adtrain_version": __version__,
            "categorical": list(CATEGORICAL), "numeric": list(NUMERIC), "zero_if_missing": [],
            "other": OTHER, "vocab": vocab, "targets": ["win"], "bid_features": list(BID_FEATURES)}
    if cmodel:
        cb, off, s = cmodel
        spec["targets"].append("curve")
        spec["curve_offset"], spec["curve_scale"] = off, s
        dm = cb.dump_model()
        # a booster trained with a custom loss dumps no objective; the Go
        # loader serves its raw score (the log price to beat)
        dm["objective"] = dm.get("objective") or "custom"
        (d / "curve.json").write_text(json.dumps(dm))
        cb.save_model(str(d / "curve.txt"))
    if curse and curse.get("buckets"):
        (d / "curse.json").write_text(json.dumps(
            {"buckets": [{"lo": b["lo"], "hi": b["hi"], "kappa": b["kappa_used"]}
                         for b in curse["buckets"]]}))
    (d / "model.json").write_text(json.dumps(spec))
    (d / "win.json").write_text(json.dumps(booster.dump_model()))
    booster.save_model(str(d / "win.txt"))
    # golden examples: the FeatureVector with the bid features filled in,
    # at the logged bid and at two others, so the check covers the curve
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(ev)
    with open(d / "golden.jsonl", "w") as f:
        for i in sorted(rng.choice(idx, size=min(70, len(idx)), replace=False)):
            fv = json.loads(data["fv"][i])
            for k in (0.5, 1.0, 1.5):
                b = data["bid"][i] * k
                r = bid_inputs(fv, b)
                g = json.loads(json.dumps(fv))
                g.setdefault("numeric", {}).update({k2: r[k2] for k2 in BID_FEATURES if math.isfinite(r[k2])})
                x = np.array([[float(vocab[c].get(r[c], vocab[c][OTHER])) for c in CATEGORICAL] +
                              [r[c] for c in NUMERIC]], float)
                preds = {"win": float(booster.predict(x)[0])}
                if cmodel:
                    xc = x.copy()
                    xc[:, [len(CATEGORICAL) + NUMERIC.index(k2) for k2 in BID_FEATURES]] = np.nan
                    preds["curve"] = float(cmodel[0].predict(xc)[0])
                f.write(json.dumps({"features": g, "ad": {}, "bid": b,
                                    "predictions": preds}) + "\n")
    rep["version"] = version
    (d / "report.json").write_text(json.dumps(rep, indent=1))
    log(f"published {d}: AUC {rep['auc']}, calibration {rep['calibration']}")
    return str(d), rep


def print_report(r: dict) -> None:
    print(f"\n{r['version']}: trained on {r['train_bids']:,} bids (+{r['pseudo_rows']:,} rows from "
          f"minimum-bid-to-win reports), evaluated on day {r['eval_day']}: {r['eval_bids']:,} bids, "
          f"win rate {r['win_rate']}")
    print(f"  AUC {r['auc']}, log loss {r['log_loss']}, predicted wins per actual {r['calibration']}")
    print("  by bid level:")
    for b in r["by_bid_decile"]:
        print(f"    ${b['bid_from']:>6.2f}-${b['bid_to']:<7.2f} {b['bids']:>7,} bids  actual {b['actual']:.3f}  predicted {b['predicted']:.3f}")
    if r.get("at_other_bids"):
        print("  at other bids (the simulator's truth):")
        for b in r["at_other_bids"]:
            print(f"    logged bid x{b['bid_multiplier']:<5} true {b['true']:.3f}  predicted {b['predicted']:.3f}  log loss {b['log_loss']:.3f}")
    c = r.get("curve")
    if c:
        print(f"  win curve (scale {c['scale']}): AUC {c['auc']}, log loss {c['log_loss']}, "
              f"predicted wins per actual {c['calibration']}")
        for b in c.get("at_other_bids") or []:
            print(f"    logged bid x{b['bid_multiplier']:<5} true {b['true']:.3f}  predicted {b['predicted']:.3f}  log loss {b['log_loss']:.3f}")
    cu = r.get("curse")
    if cu:
        print(f"  the winner's curse on {cu['wins']:,} won install-campaign impressions ({cu['installs']} installs), "
              "installs per predicted install by value / market price:")
        for b in cu["buckets"]:
            print(f"    x{b['lo']}-{b['hi'] or 'up':<4} {b['wins']:>7,} wins  kappa {b['kappa']}"
                  + (f"  true {b['true_kappa']}" if b.get("true_kappa") is not None else "")
                  + f"  used {b['kappa_used']}")
    print("  by exploration multiplier:")
    for b in r["by_explore"]:
        print(f"    {b['bucket']:<18}{b['bids']:>7,} bids  actual {b['actual']:.3f}  predicted {b['predicted']:.3f}")


def _r(x, k: int = 4):
    x = float(x)
    return None if math.isnan(x) else round(x, k)
