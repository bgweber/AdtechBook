"""Train, validate, and publish one click model (Chapters 9 and 11).

    examples = dataset.build(ledger)            # impressions + labels
    split out of time: train on earlier days, evaluate on the last
    FTRL, one pass over the shuffled training days, then the bias refit
    publish the artifact, read it back, and predict as the bidder will
    metrics, curves, calibration, readouts, stability, checks -> report

The checks decide whether the model may be deployed without a person
looking at it. The bidder enforces them: a model that fails is deployed
only with force, the manual intervention of Chapter 11.
"""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import mmh3
import numpy as np

from . import artifact, click_rates, dataset, install_rates, metrics
from .ftrl import FTRL
from .tokens import Spec


@dataclass
class Params:
    click_window_s: int = 3600
    downsample_rate: float = 1.0  # share of non-clicks kept; 1 = all
    alpha: float = 0.1
    beta: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    # The crawl DSP retrains from scratch each day, so the training days
    # are shuffled: in time order, FTRL keeps learning online and its
    # final weights fit the last hours best, which can leave the model
    # miscalibrated for the rest of the day. Evaluation stays out of time.
    shuffle: bool = True
    epochs: int = 1  # passes over the training days
    warm_start: bool = True  # start the bias at the base rate's log-odds
    recenter: bool = True  # refit the bias so training clicks match
    seed: int = 0
    # Publish a model that predicts this click rate for everything, as a
    # broken retrain might (the Chapter 12 exercise). None trains normally.
    constant: float | None = None
    install_prior: float = 0.03  # click-to-install rate before any data
    install_k: float = 200.0  # clicks' worth of weight for the parent rate
    train_days: list | None = None
    eval_day: int | None = None
    golden_examples: int = 200
    # validation checks (Chapter 11)
    min_eval_clicks: int = 50
    min_auc: float = 0.55
    calibration_range: tuple = (0.8, 1.25)
    min_cosine: float = 0.8
    spec: Spec = field(default_factory=Spec)


def train(ledger_dir, models_dir, params: Params | None = None,
          sim_out=None, log=print) -> tuple[str, dict]:
    p = params or Params()
    t0 = time.time()
    ds = dataset.build(ledger_dir, p.click_window_s)
    c = ds.counts
    log(f"examples: {c['examples']:,} impressions with final labels, "
        f"{c['positives']:,} clicks (from {c['impressions']:,} impressions, "
        f"{c['clicks']:,} clicks, {c['bids']:,} bids in the ledger)")
    for w in ds.warnings:
        log("warning: " + w)
    if not ds.examples:
        raise ValueError(
            "no click examples yet: run the simulator against the bidder "
            "in bid mode first (docker compose up)")

    tr, ev, split = _split(ds.examples, p, ds.data_end_ms)
    log(f"split: {split}")
    if not tr or not ev:
        raise ValueError(f"not enough data to split ({split})")

    # FTRL over the shuffled training days, with negative downsampling
    spec = p.spec
    model = FTRL(p.alpha, p.beta, p.l1, p.l2)
    names = defaultdict(Counter)  # bucket -> tokens seen in it
    rows = []
    for e in tr:
        if e.label == 0 and p.downsample_rate < 1 and \
                _u(e.request_id) >= p.downsample_rate:
            continue
        toks = spec.tokens(e.features, e.ad)
        for t in toks:
            names[spec.bucket(t)][t] += 1
        rows.append((spec.buckets(toks), e.label))
    kept = len(rows)
    if p.constant is not None:
        # a deliberately broken model: no weights, one rate for everything
        rate = min(max(p.constant, 1e-6), 1 - 1e-6)
        bias, weights = math.log(rate / (1 - rate)), {}
        log(f"constant model: predicts {rate:.1%} for every impression "
            "(a deliberately broken model, for the Chapter 12 exercise)")
    else:
        if p.warm_start and rows:
            model.start_bias(sum(y for _, y in rows) / kept)
        rng = np.random.default_rng(p.seed)
        for _ in range(p.epochs):
            if p.shuffle:
                rng.shuffle(rows)
            for b, y in rows:
                model.update(b, y)
        if p.recenter and rows:
            model.recenter_bias(rows)
        bias, weights = model.weights()
        log(f"trained on {kept:,} examples: {len(weights):,} nonzero weights "
            f"of {len(names):,} buckets seen")

    # publish, then read the artifact back, so every number below is what
    # the bidder will compute
    version = artifact.next_version(models_dir)
    schema = max((e.features.get("schema_version", 1) for e in tr), default=1)
    meta = {
        "model": "click-ftrl", "feature_schema_version": schema,
        "bias": bias, "downsample_rate": p.downsample_rate,
        "spec": spec.to_json(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ftrl": {"alpha": p.alpha, "beta": p.beta, "l1": p.l1, "l2": p.l2},
    }
    tmp = artifact.write(models_dir, version, meta, weights)
    served = artifact.Served(tmp)

    y = np.array([e.label for e in ev])
    pred = np.array([served.predict(e.features, e.ad) for e in ev])
    logged = np.array([e.logged_p for e in ev])
    true_p = _truth(sim_out, ev, log)

    report = {
        "version": version,
        "created_at": meta["trained_at"],
        "data": {**c, "split": split, "train_examples": len(tr),
                 "train_kept": kept, "eval_examples": len(ev),
                 "warnings": ds.warnings},
        "params": {k: v for k, v in asdict(p).items() if k != "spec"},
        "metrics": metrics.summary(y, pred),
        "logged": metrics.summary(y, logged),
        "logged_sources": dict(Counter(e.logged_model or "prior" for e in ev)),
    }
    if true_p is not None:
        m = ~np.isnan(true_p)
        report["truth"] = {
            "matched": int(m.sum()),
            "mean_true": float(true_p[m].mean()) if m.any() else None,
            "mean_pred": float(pred[m].mean()) if m.any() else None,
            "pred_over_true": float(pred[m].sum() / true_p[m].sum())
            if m.any() else None}
    report["curves"] = {"roc": metrics.roc(y, pred)[0],
                        "pr": metrics.pr(y, pred)[0],
                        "roc_logged": metrics.roc(y, logged)[0]}
    report["calibration"] = metrics.calibration(y, pred, true_p=true_p)
    report["calibration_logged"] = metrics.calibration(y, logged, true_p=true_p)
    report["segments"] = {
        k: metrics.segments([_seg(e, k) for e in ev], y, pred, logged)
        for k in ("placement", "exchange", "campaign", "country")}
    report["readout"] = _readout(spec, weights, names)
    report["stability"] = _stability(models_dir, spec, weights, ev, pred, log)
    report["checks"] = _checks(p, report)
    report["passed"] = all(ch["pass"] for ch in report["checks"])
    report["seconds"] = round(time.time() - t0, 1)

    # golden examples: the bidder must reproduce these predictions
    step = max(1, len(ev) // p.golden_examples)
    with open(tmp / "golden.jsonl", "w") as f:
        import json
        for e, pp in list(zip(ev, pred))[::step][: p.golden_examples]:
            f.write(json.dumps({"features": e.features, "ad": e.ad,
                                "p": float(pp)}) + "\n")
    # click-to-install rates for install campaigns, published in the same
    # bundle, so deploying the model deploys them too (Chapter 10)
    rates, report["install_rates"] = install_rates.compute(
        ledger_dir, ds, p.install_prior, p.install_k,
        truth=_truth_map(sim_out, "p_install_click"))
    artifact.write_json(tmp / "install_rates.json", rates)
    # segment click rates, the fallback when the model can't be used
    report["click_rates"] = click_rates.compute(tr)
    artifact.write_json(tmp / "click_rates.json", report["click_rates"])
    ro = report["install_rates"]["overall"]
    log(f"install rates: {ro['installs']} installs from {ro['clicks']:,} "
        f"mature clicks, overall {ro['smoothed']:.2%} after smoothing "
        f"(k={p.install_k:g}, prior {p.install_prior:.0%})")
    artifact.write_json(tmp / "report.json", report)
    artifact.finish(tmp, version)
    m = report["metrics"]
    log(f"{version}: AUC {m['auc']:.3f}, PR AUC {m['pr_auc']:.3f}, "
        f"log loss {m['log_loss']:.4f} (logged {report['logged']['log_loss']:.4f}), "
        f"calibration {m['calibration_ratio']:.2f}")
    for ch in report["checks"]:
        log(f"  check {ch['name']}: {'pass' if ch['pass'] else 'FAIL'} "
            f"({ch['detail']})")
    log(f"{version}: {'passed' if report['passed'] else 'FAILED'} its checks "
        f"in {report['seconds']}s")
    return version, report


def _split(ex, p: Params, data_end_ms: int):
    """Out-of-time split. Only complete days are compared: a day that's
    still running covers only some hours, and hours differ (early in the
    day, devices have seen few of our ads), so it would mislead."""
    days = sorted({e.day for e in ex})
    if p.eval_day is not None:
        tdays = p.train_days or [d for d in days if d < p.eval_day]
        tr = [e for e in ex if e.day in set(tdays)]
        ev = [e for e in ex if e.day == p.eval_day]
        return tr, ev, f"train on days {tdays}, evaluate on day {p.eval_day}"
    # a day is complete once the logs reach its last half hour; its last
    # click window's worth of impressions is already left out as not final
    full = [d for d in days
            if d < days[-1] or data_end_ms >= (d + 1) * dataset.DAY_MS - 1_800_000]
    note = ""
    if days and days[-1] not in full:
        note = (f" Day {days[-1]} is still running and is left out; train "
                "again when it's done.")
    if len(full) >= 2:
        last = full[-1]
        tr = [e for e in ex if e.day < last]
        ev = [e for e in ex if e.day == last]
        return tr, ev, (f"train on days {full[:-1]}, evaluate on day {last} "
                        "(out of time)." + note)
    # one day: still out of time, by hour
    use = [e for e in ex if e.day in set(full)] or ex
    cut = use[int(len(use) * 0.75)].ts_ms
    tr = [e for e in use if e.ts_ms < cut]
    ev = [e for e in use if e.ts_ms >= cut]
    h = (cut % dataset.DAY_MS) / 3_600_000
    return tr, ev, (f"one complete day of data: train before {h:.1f}h, "
                    "evaluate after (out of time). Hours differ, so this "
                    "is a rough check: play two or more full days for a "
                    "day split." + note)


def _u(rid: str) -> float:
    """A stable uniform draw per request, so downsampling is repeatable."""
    return mmh3.hash(rid, 1, signed=False) / 2 ** 32


def _seg(e, k):
    if k == "campaign":
        return e.ad.get("campaign", "")
    return (e.features.get("strings") or {}).get(k, "null")


def _readout(spec: Spec, weights: dict, names: dict, k: int = 15) -> dict:
    rows = []
    for b, w in weights.items():
        cnt = names.get(b)
        if not cnt:
            continue
        tok, n = cnt.most_common(1)[0]
        rows.append({"token": tok, "label": spec.label(tok), "weight": w,
                     "count": sum(cnt.values()), "tokens_in_bucket": len(cnt),
                     "cross": "*" in tok})
    single = [r for r in rows if not r["cross"]]
    cross = [r for r in rows if r["cross"]]
    by = lambda rs, rev: sorted(rs, key=lambda r: r["weight"], reverse=rev)[:k]
    return {"top_tokens": by(single, True), "bottom_tokens": by(single, False),
            "top_crosses": by(cross, True), "bottom_crosses": by(cross, False),
            "nonzero_tokens": len(single), "nonzero_crosses": len(cross)}


def _stability(models_dir, spec, weights, ev, pred, log) -> dict:
    """Compare with the deployed model: the stability check guards what's
    serving. With nothing deployed, there's nothing to compare against."""
    prev = artifact.deployed(models_dir)
    if prev is None or not (artifact.root(models_dir) / prev).exists():
        return {"previous": None, "note": "no deployed model to compare against"}
    old = artifact.Served(artifact.root(models_dir) / prev)
    # weights compare bucket by bucket only in the same hash space
    same_space = (old.spec.num_buckets, old.spec.seed) == \
        (spec.num_buckets, spec.seed)
    cos = metrics.cosine(weights, old.weights) if same_space else float("nan")
    old_pred = np.array([old.predict(e.features, e.ad) for e in ev])
    rel = np.abs(pred / np.maximum(old_pred, 1e-9) - 1)
    out = {"previous": prev, "basis": "deployed", "cosine": cos,
           "median_rel_change": float(np.median(rel)),
           "p90_rel_change": float(np.quantile(rel, 0.9)),
           "previous_nonzero": len(old.weights), "nonzero": len(weights)}
    log(f"stability vs {prev} (deployed): cosine {cos:.3f}, median "
        f"prediction change {out['median_rel_change']:.1%}")
    return out


def _checks(p: Params, r: dict) -> list:
    m, lg, st = r["metrics"], r["logged"], r["stability"]
    lo, hi = p.calibration_range
    cal = m["calibration_ratio"]
    out = [
        {"name": "enough_eval_clicks", "value": m["clicks"],
         "threshold": p.min_eval_clicks, "pass": m["clicks"] >= p.min_eval_clicks,
         "detail": f"{m['clicks']} clicks in evaluation, need {p.min_eval_clicks}"},
        {"name": "auc", "value": m["auc"], "threshold": p.min_auc,
         "pass": bool(m["auc"] >= p.min_auc),
         "detail": f"AUC {m['auc']:.3f}, need {p.min_auc}"},
        {"name": "calibration", "value": cal, "threshold": [lo, hi],
         "pass": bool(lo <= cal <= hi),
         "detail": f"predicted/actual clicks {cal:.2f}, need {lo} to {hi}"
                   + ("" if lo <= cal <= hi else _bought_note(r))},
        {"name": "beats_logged", "value": m["log_loss"],
         "threshold": lg["log_loss"], "pass": bool(m["log_loss"] < lg["log_loss"]),
         "detail": f"log loss {m['log_loss']:.4f} vs {lg['log_loss']:.4f} for "
                   "what the bidder predicted when it bid"},
    ]
    if st.get("previous"):
        cos = st["cosine"]
        out.append({"name": "stability", "value": cos, "threshold": p.min_cosine,
                    "pass": bool(cos >= p.min_cosine),
                    "detail": f"cosine {cos:.3f} with {st['previous']}, "
                              f"need {p.min_cosine}"})
    else:
        out.append({"name": "stability", "value": None,
                    "threshold": p.min_cosine, "pass": True,
                    "detail": "no deployed model to compare against"})
    return out


def _bought_note(r: dict) -> str:
    """When a model bought most of the evaluation day, say so: a model
    wins most where it predicts too high, so the impressions it buys look
    overpredicted to any similar model (the winner's curse, Chapter 31)."""
    src = r.get("logged_sources") or {}
    n = sum(src.values())
    models = {k: v for k, v in src.items() if k != "prior"}
    if not n or sum(models.values()) < 0.5 * n:
        return ""
    top = max(models, key=models.get)
    lc = r["logged"]["calibration_ratio"]
    return (f". Most evaluation impressions were bought by {top}, which "
            f"predicted {lc:.2f}x the actual clicks on them: models win "
            "where they predict too high, the winner's curse (Chapter 31). "
            "Review before forcing a deploy")


def _truth_map(sim_out, column):
    """impression ID -> one of the simulator's true probabilities."""
    if not sim_out:
        return None
    f = Path(sim_out) / "truth" / "impressions.parquet"
    if not f.exists():
        return None
    try:
        import pandas as pd
        t = pd.read_parquet(f, columns=["impression_id", column])
    except Exception:
        return None
    return dict(zip(t.impression_id, t[column]))


def _truth(sim_out, ev, log):
    """The simulator's true click rates, when its output is available."""
    if not sim_out:
        return None
    f = Path(sim_out) / "truth" / "impressions.parquet"
    if not f.exists():
        return None
    try:
        import pandas as pd
        t = pd.read_parquet(f, columns=["impression_id", "p_click"])
    except Exception as err:  # pandas or pyarrow not installed
        log(f"note: true click rates not read ({err})")
        return None
    m = dict(zip(t.impression_id, t.p_click))
    out = np.array([m.get("i" + e.request_id, np.nan) for e in ev], float)
    n = int((~np.isnan(out)).sum())
    log(f"true click rates from the simulator for {n:,} of {len(ev):,} "
        "evaluation impressions")
    return out if n else None
