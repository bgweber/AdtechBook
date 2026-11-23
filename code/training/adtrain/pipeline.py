"""The conversion model pipeline (Chapter 18).

One command runs every step that Chapters 14 and 15 ran by hand, and stops
at the first one that fails:

1. build a dataset from the ledger;
2. validate it (the dataset's own checks, Chapter 14);
3. train the LightGBM bundle on it (Chapter 15);
4. validate the bundle: its predictions vary, its metrics are in range,
   its top predictions aren't far above what happened, and it is no worse
   than the champion on the same rows;
5. publish it as the challenger.

The registry is files, as in Part II. Under models/conversion/:

    gbm-NNNN/       one bundle per training run, never changed afterwards
    DEPLOYED        the champion: the version the bidder serves
    CHALLENGER      the latest bundle that passed the pipeline
    deploys.jsonl   every deploy, promotion, and rollback (the bidder writes it)
    pipeline/run-NNNN.json   one record per pipeline run

The bidder owns DEPLOYED; the pipeline owns CHALLENGER. Promoting the
challenger is a deploy, from the console, so a person decides what serves.

    python -m adtrain pipeline --ledger out/ledger --models out/models
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np

from . import datasets, gbm, metrics

FAMILY = "conversion"

# Model checks. The limits are wide because the simulated market is
# small: a few dozen installs a day at the tiny scale.
CLICK_AUC_MIN = 0.65
CLICK_CALIBRATION = (0.8, 1.25)
INSTALL_CALIBRATION = (0.6, 1.6)
TOP_QUINTILE_MAX = 2.0     # predicted / actual, top fifth of predictions
MIN_PREDICTION_SD = 1e-4   # a model that predicts one number for every row
CHAMPION_LOGLOSS_SLACK = 0.01   # challenger may be up to 1% worse on click log loss
CHAMPION_AUC_SLACK = 0.03       # and up to 0.03 worse on install AUC


def root(models_dir) -> Path:
    return Path(models_dir) / FAMILY


def read_alias(models_dir, name: str) -> str:
    f = root(models_dir) / name
    return f.read_text().strip() if f.exists() else ""


def write_alias(models_dir, name: str, version: str) -> None:
    d = root(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (name + ".tmp")
    tmp.write_text(version + "\n")
    tmp.replace(d / name)


def check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def validate_bundle(bundle_dir, report: dict, df, champion_dir=None, log=print) -> list[dict]:
    """The model checks. df is the dataset the bundle was trained on."""
    out = []
    m = report.get("models", {})
    out.append(check("dataset passed", report.get("dataset_passed") is True,
                     f"trained on {report.get('dataset_version')}, which "
                     + ("passed" if report.get("dataset_passed") else "failed") + " its checks"))
    for t in ("click", "install_click"):
        if t not in m:
            out.append(check(f"{t} trained", False, "no model: too few labels"))
    sds = {t: r.get("prediction_sd") for t, r in m.items()}
    flat = [t for t, v in sds.items() if v is not None and v < MIN_PREDICTION_SD]
    out.append(check("predictions vary", not flat,
                     "standard deviation of evaluation predictions: "
                     + ", ".join(f"{t} {v:.2g}" for t, v in sds.items() if v is not None)
                     + (f"; {', '.join(flat)} predict one number for everything" if flat else "")))
    c = m.get("click")
    if c:
        lo, hi = CLICK_CALIBRATION
        out.append(check("click AUC", (c["auc"] or 0) >= CLICK_AUC_MIN,
                         f"{c['auc']} (limit {CLICK_AUC_MIN})"))
        out.append(check("click calibration", lo <= c["calibration"] <= hi,
                         f"{c['calibration']} predicted per actual click (limit {lo}-{hi})"))
    ic = m.get("install_click")
    if ic:
        lo, hi = INSTALL_CALIBRATION
        out.append(check("install calibration", lo <= ic["calibration"] <= hi,
                         f"{ic['calibration']} predicted per actual install after a click "
                         f"(limit {lo}-{hi})"))
        top = (ic.get("by_prediction") or [None])[-1]
        if top and top["actual"]:
            r = top["predicted"] / top["actual"]
            out.append(check("top predictions", r <= TOP_QUINTILE_MAX,
                             f"the top fifth of install predictions average {top['predicted']:.3f}, "
                             f"against {top['actual']:.3f} actual: {r:.2f}x (limit {TOP_QUINTILE_MAX}x); "
                             "bids are highest there"))
        elif top:
            out.append(check("top predictions", top["predicted"] < 0.05,
                             f"no installs in the top fifth, which averages {top['predicted']:.3f}"))
    out.extend(_vs_champion(bundle_dir, report, df, champion_dir, log))
    return out


def _vs_champion(bundle_dir, report, df, champion_dir, log) -> list[dict]:
    """Score the challenger and the champion on the same evaluation rows."""
    if not champion_dir:
        return [check("vs. champion", True, "no champion deployed yet")]
    champ = Path(champion_dir)
    if not gbm.load_boosters(champ):
        return [check("vs. champion", True, f"{champ.name} has no LightGBM model files "
                      "(published before adtrain 0.6), so it can't be rescored; skipped")]
    out = []
    for t, metric in (("click", "log loss"), ("install_click", "AUC")):
        r = report["models"].get(t)
        if not r:
            continue
        # score only the evaluation rows: rescoring the whole dataset with
        # two bundles takes more memory than a small Docker VM has
        ix = gbm.rows_for(df, t, r["eval_days"])
        sub = df.iloc[ix]
        y = sub[gbm.TARGETS[t][0]].to_numpy()
        if not len(ix) or not y.sum():
            continue
        raws = gbm.dataset_raws(sub)
        new, old = gbm.score(bundle_dir, sub, raws, t), gbm.score(champ, sub, raws, t)
        if t not in old or t not in new:
            continue
        if metric == "log loss":
            a, b = metrics.log_loss(y, new[t]), metrics.log_loss(y, old[t])
            ok = a <= b * (1 + CHAMPION_LOGLOSS_SLACK)
            detail = f"log loss {a:.4f} vs. {b:.4f} for {champ.name}"
        else:
            a, b = metrics.roc(y, new[t])[1], metrics.roc(y, old[t])[1]
            ok = a >= b - CHAMPION_AUC_SLACK
            detail = f"AUC {a:.3f} vs. {b:.3f} for {champ.name}"
        out.append(check(f"{t} vs. champion", ok,
                         f"{detail}, days {r['eval_days'][0]}-{r['eval_days'][1]}, "
                         f"{len(ix):,} rows"))
    return out


def run(ledger_dir, models_dir, sim_out=None, leak=False, inject=None, log=print) -> dict:
    """Run the pipeline once; returns its record, also written to disk."""
    stages, t0 = [], time.time()
    rec = {"started_unix": int(t0), "leak": leak, "inject": inject, "stages": stages,
           "passed": False, "challenger": None}

    def stage(name, status, detail, start, **kw):
        stages.append({"name": name, "status": status, "detail": detail,
                       "seconds": round(time.time() - start, 1), **kw})
        log(f"[{status}] {name}: {detail}")

    def done():
        rec["seconds"] = round(time.time() - t0, 1)
        d = root(models_dir) / "pipeline"
        d.mkdir(parents=True, exist_ok=True)
        n = len(list(d.glob("run-*.json"))) + 1
        rec["run"] = f"run-{n:04d}"
        (d / f"{rec['run']}.json").write_text(json.dumps(rec, indent=1))
        return rec

    s = time.time()
    df, manifest = datasets.build(ledger_dir, datasets.Settings(leak=leak), log=lambda *_: None)
    ddir = datasets.write(df, manifest, Path(models_dir).parent / "datasets")
    n_rows = len(df)
    del df  # training reads it back from disk; holding both doubles the memory
    gc.collect()
    manifest = json.loads((Path(ddir) / "manifest.json").read_text())
    rec["dataset"] = manifest["version"]
    stage("build dataset", "ok", f"{manifest['version']}: {n_rows:,} impressions", s)

    s = time.time()
    failed = [c for c in manifest["checks"] if not c["passed"]]
    if failed:
        stage("validate dataset", "failed",
              "; ".join(f"{c['name']}: {c['detail']}" for c in failed), s, checks=manifest["checks"])
        return done()
    stage("validate dataset", "ok", f"{len(manifest['checks'])} checks pass", s,
          checks=manifest["checks"])

    s = time.time()
    try:
        bdir, report = gbm.train(ddir, models_dir, sim_out, inject=inject, log=lambda *_: None)
    except RuntimeError as err:
        stage("train", "failed", str(err), s)
        return done()
    rec["bundle"] = report["version"]
    stage("train", "ok", f"{report['version']}: " + ", ".join(
        f"{t} AUC {r['auc']}" for t, r in report["models"].items()), s)

    s = time.time()
    gc.collect()
    champion = read_alias(models_dir, "DEPLOYED")
    cdir = root(models_dir) / champion if champion else None
    df = datasets.read(ddir)[0]
    checks = validate_bundle(bdir, report, df, cdir, log)
    del df
    (Path(bdir) / "validation.json").write_text(json.dumps(
        {"champion": champion or None, "checks": checks}, indent=1))
    failed = [c for c in checks if not c["passed"]]
    if failed:
        stage("validate model", "failed",
              "; ".join(f"{c['name']}: {c['detail']}" for c in failed), s, checks=checks)
        return done()
    stage("validate model", "ok", f"{len(checks)} checks pass", s, checks=checks)

    s = time.time()
    write_alias(models_dir, "CHALLENGER", report["version"])
    rec["passed"], rec["challenger"] = True, report["version"]
    stage("publish", "ok", f"{report['version']} is the challenger; promote it from the "
          "console to serve it" + (f" in place of {champion}" if champion else ""), s)
    return done()


def runs(models_dir, limit: int = 20) -> list[dict]:
    d = root(models_dir) / "pipeline"
    files = sorted(d.glob("run-*.json"))[-limit:] if d.exists() else []
    return [json.loads(f.read_text()) for f in reversed(files)]


def print_run(rec: dict) -> None:
    print(f"\n{rec['run']}: {'passed' if rec['passed'] else 'stopped'} "
          f"in {rec['seconds']}s" + (f", challenger {rec['challenger']}" if rec["challenger"] else ""))
    for st in rec["stages"]:
        print(f"  {st['status']:>6}  {st['name']}")
        for c in st.get("checks") or []:
            if st["name"] == "validate model" or not c["passed"]:
                print(f"          {'pass' if c['passed'] else 'FAIL'}  {c['name']}: {c['detail']}")
