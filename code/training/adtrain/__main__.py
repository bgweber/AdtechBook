"""Command line.

    python -m adtrain train --ledger out/ledger --models out/models
    python -m adtrain serve --ledger out/ledger --models out/models --port 8090
    python -m adtrain parity --ledger out/ledger
    python -m adtrain trace <request_id> --ledger out/ledger
    python -m adtrain find --clicked --ledger out/ledger
    python -m adtrain labels --ledger out/ledger --sim-out out/sim
    python -m adtrain dataset --ledger out/ledger
    python -m adtrain gbm --dataset out/datasets/conversions/conversions-0001
    python -m adtrain pipeline --ledger out/ledger --models out/models
    python -m adtrain replay --day 12 --b gbm-0003
    python -m adtrain ope --policy uniform
    python -m adtrain periods
"""
from __future__ import annotations

import argparse

from .train import Params, train


def main() -> None:
    ap = argparse.ArgumentParser(prog="adtrain")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "serve"):
        s = sub.add_parser(name)
        s.add_argument("--ledger", default="out/ledger")
        s.add_argument("--models", default="out/models")
        s.add_argument("--sim-out", default="out/sim",
                       help="simulator output, for true click rates (optional)")
    t = sub.choices["train"]
    t.add_argument("--eval-day", type=int, default=None)
    t.add_argument("--train-days", type=int, nargs="*", default=None)
    t.add_argument("--downsample-rate", type=float, default=1.0)
    t.add_argument("--constant", type=float, default=None,
                   help="publish a model that predicts this click rate for "
                        "everything (the Chapter 12 exercise)")
    t.add_argument("--no-crosses", action="store_true",
                   help="train without crosses, to see what they add")
    s = sub.choices["serve"]
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8090)
    pc = sub.add_parser("parity", help="compare logged FeatureVectors with "
                        "ones recomputed from raw context (Chapter 8)")
    pc.add_argument("--ledger", default="out/ledger")
    tc = sub.add_parser("trace", help="follow one request through the logs")
    tc.add_argument("request_id")
    tc.add_argument("--ledger", default="out/ledger")
    dc = sub.add_parser("dataset", help="build a conversion training "
                        "dataset, with label cutoffs (Chapter 14)")
    dc.add_argument("--ledger", default="out/ledger")
    dc.add_argument("--out", default="out/datasets")
    dc.add_argument("--leak", action="store_true",
                    help="join the device's history as of now, not as of "
                         "each bid (the Chapter 14 exercise)")
    dc.add_argument("--days", type=int, default=28,
                    help="the newest days of impressions to use (0 for every day)")
    dc.add_argument("--sample-rate", type=float, default=1.0,
                    help="keep this share of rows with no click or install")
    gc = sub.add_parser("gbm", help="train LightGBM click and install "
                        "models on a dataset (Chapter 15)")
    gc.add_argument("--dataset", required=True, help="a dataset version directory")
    gc.add_argument("--models", default="out/models")
    gc.add_argument("--sim-out", default="out/sim")
    gc.add_argument("--eval-day", type=int, default=None)
    gc.add_argument("--inject", choices=["no-features"], default=None,
                    help="train as if every feature were missing (Chapter 18)")
    pp = sub.add_parser("pipeline", help="dataset, checks, LightGBM, model "
                        "checks, challenger, stopping at the first failure (Chapter 18)")
    pp.add_argument("--ledger", default="out/ledger")
    pp.add_argument("--models", default="out/models")
    pp.add_argument("--sim-out", default="out/sim")
    pp.add_argument("--leak", action="store_true",
                    help="build the leaky dataset (Chapter 14), to see the pipeline stop")
    pp.add_argument("--inject", choices=["no-features"], default=None,
                    help="train as if every feature were missing")
    rc = sub.add_parser("replay", help="rescore a logged day through two "
                        "bundles and compare bids, wins, and spend (Chapter 16)")
    rc.add_argument("--ledger", default="out/ledger")
    rc.add_argument("--models", default="out/models")
    rc.add_argument("--sim-out", default="out/sim")
    rc.add_argument("--day", type=int, required=True)
    rc.add_argument("--a", default="logged", help="a bundle version, or logged")
    rc.add_argument("--b", required=True, help="a bundle version")
    oc = sub.add_parser("ope", help="estimate a campaign selection policy "
                        "from logged propensities (Chapter 16)")
    oc.add_argument("--ledger", default="out/ledger")
    oc.add_argument("--policy", choices=["uniform", "cpi-first"], default="uniform")
    oc.add_argument("--reward", choices=["click", "install"], default="click")
    oc.add_argument("--dataset", default=None, help="labels from this dataset "
                    "(default: the newest as-of one)")
    ab = sub.add_parser("abtest", help="an A/B test's readout: CPI by arm, "
                        "with confidence intervals (Chapter 19)")
    ab.add_argument("experiment")
    ab.add_argument("--ledger", default="out/ledger")
    ab.add_argument("--sim-out", default="out/sim")
    ab.add_argument("--dataset", default=None)
    ab.add_argument("--goal", default="CPI", help="the campaigns to read: CPI "
                    "(cost per install), AD_ROAS (D3 ad revenue per dollar, "
                    "for a value A/B test, Chapter 30), or INSTALL (every install "
                    "campaign, with the cost per caused install, Chapter 32)")
    ex = sub.add_parser("explain", help="one logged bid, term by term (Chapter 33)")
    ex.add_argument("--ledger", default="out/ledger")
    ex.add_argument("--request", default=None, help="a request ID; default: the newest day's last bid")
    ex.add_argument("--campaign", type=int, default=None)
    ex.add_argument("--day", type=int, default=None)
    fr = sub.add_parser("frontier", help="the consistency check: spend and cost per install "
                        "over pacing multipliers, for one campaign (Chapter 33)")
    fr.add_argument("--ledger", default="out/ledger")
    fr.add_argument("--sim-out", default="out/sim")
    fr.add_argument("--campaign", type=int, required=True)
    fr.add_argument("--budget", type=float, default=None, help="default: the campaign's")
    fr.add_argument("--target-cpi", type=float, default=None)
    fr.add_argument("--target-roas", type=float, default=None)
    fr.add_argument("--day", type=int, default=None)
    rp = sub.add_parser("rlpace", help="pacing policies in a simulator built from logged "
                        "auctions: PI, MPC, and Q-learning (Chapter 34)")
    rp.add_argument("--ledger", default="out/ledger")
    rp.add_argument("--sim-out", default="out/sim")
    rp.add_argument("--campaign", type=int, required=True)
    rp.add_argument("--test-day", type=int, default=None, help="default: the newest")
    rp.add_argument("--train-day", type=int, default=None, help="default: the day before the test day")
    rp.add_argument("--budget", type=float, default=None, help="default: the campaign's")
    rp.add_argument("--episodes", type=int, default=10000, help="Q-learning's simulated days")
    up = sub.add_parser("uplift", help="uplift models from a lift study (Chapter 32)")
    up.add_argument("--ledger", default="out/ledger")
    up.add_argument("--models", default="out/models")
    up.add_argument("--sim-out", default="out/sim")
    wn = sub.add_parser("win", help="train a win rate model for bid "
                        "shading on the newest days (Chapter 20)")
    wn.add_argument("--ledger", default="out/ledger")
    wn.add_argument("--models", default="out/models")
    wn.add_argument("--sim-out", default="out/sim")
    wn.add_argument("--days", type=int, default=7, help="the newest days to use")
    wn.add_argument("--no-mtw", action="store_true",
                    help="ignore the exchanges' minimum-bid-to-win reports")
    wn.add_argument("--sample-rate", type=float, default=1.0,
                    help="keep this share of bids; a laptop-scale market needs about 0.1")
    wn.add_argument("--curve", action="store_true",
                    help="also train a win curve: the whole curve from one pass "
                         "through the trees (Chapter 31)")
    wn.add_argument("--curse", action="store_true",
                    help="with --curve, measure the winner's curse by margin over "
                         "the market price, for the bidder to correct (Chapter 31)")
    kc = sub.add_parser("kappa", help="calibration multipliers by campaign, "
                        "from matured installs against predicted (Chapter 21)")
    kc.add_argument("--ledger", default="out/ledger")
    kc.add_argument("--models", default="out/models")
    lf = sub.add_parser("lift", help="incrementality from the ghost-bid "
                        "holdout, by app, with the simulator's truth (Chapter 23)")
    lf.add_argument("--ledger", default="out/ledger")
    lf.add_argument("--sim-out", default="out/sim")
    lf.add_argument("--days", type=int, default=28)
    lf.add_argument("--app", default=None, help="one app bundle")
    lf.add_argument("--holdout-share", type=float, default=None,
                    help="keep only this share of devices in the holdout, below 0.05, "
                         "to see the interval widen")
    ox = sub.add_parser("onnx", help="export a bundle's model to ONNX and time "
                        "ONNX Runtime on it (Chapter 17)")
    ox.add_argument("--bundle", required=True, help="a conversion bundle directory")
    ox.add_argument("--target", default="click")
    ox.add_argument("--out", default="out/onnx")
    qc = sub.add_parser("quarantine", help="mark bad postbacks so dataset builds "
                        "and device profiles skip them (Chapter 24)")
    qc.add_argument("action", choices=["list", "add", "remove", "inspect"])
    qc.add_argument("entry", nargs="?", help="the entry to remove, such as q-0001")
    qc.add_argument("--ledger", default="out/ledger")
    qc.add_argument("--datasets", default="out/datasets")
    qc.add_argument("--day", type=int, help="a whole day of received postbacks; or --from-ms/--to-ms")
    qc.add_argument("--from-ms", type=int)
    qc.add_argument("--to-ms", type=int)
    qc.add_argument("--where", action="append", default=[],
                    help="field=value: only postbacks with this field value (repeatable)")
    qc.add_argument("--field", default="match_type", help="inspect: break down by this field")
    qc.add_argument("--reason", default="")
    pf = sub.add_parser("pace", help="the pacing forecast: each campaign's "
                        "starting multiplier for tomorrow (Chapter 22)")
    pf.add_argument("--ledger", default="out/ledger")
    pf.add_argument("--models", default="out/models")
    pf.add_argument("--sim-out", default="out/sim", help="for the campaigns' budgets")
    pf.add_argument("--day", type=int, default=None)
    st = sub.add_parser("shadetest", help="a shading A/B test's readout: "
                        "win rate, spend, and surplus by arm (Chapter 20)")
    st.add_argument("experiment")
    st.add_argument("--ledger", default="out/ledger")
    st.add_argument("--models", default="out/models")
    sh = sub.add_parser("shadow", help="compare a shadow model's live "
                        "predictions with offline ones (Chapter 19)")
    sh.add_argument("experiment")
    sh.add_argument("--ledger", default="out/ledger")
    sh.add_argument("--models", default="out/models")
    pe = sub.add_parser("periods", help="what each serving model bought, "
                        "day by day (CPI campaigns)")
    pe.add_argument("--ledger", default="out/ledger")
    pe.add_argument("--sim-out", default="out/sim")
    pe.add_argument("--dataset", default=None)
    lc = sub.add_parser("labels", help="install label delays and what "
                        "they mean (Chapter 13)")
    lc.add_argument("--ledger", default="out/ledger")
    lc.add_argument("--sim-out", default="out/sim")
    fc = sub.add_parser("find", help="list a few request IDs to trace")
    fc.add_argument("--ledger", default="out/ledger")
    fc.add_argument("--day", type=int, default=None)
    fc.add_argument("--goal", choices=["cpc", "cpi"], default=None)
    fc.add_argument("--reason", default=None,
                    help="a no-bid reason, such as below_floor or holdout")
    fc.add_argument("--explored", action="store_true",
                    help="bids with an exploration multiplier")
    fc.add_argument("--clicked", action="store_true",
                    help="bids that led to a click")
    fc.add_argument("--limit", type=int, default=5)
    a = ap.parse_args()
    if a.cmd == "dataset":
        from . import datasets
        df, m = datasets.build(a.ledger, datasets.Settings(
            leak=a.leak, sample_rate=a.sample_rate, max_days=a.days))
        d = datasets.write(df, m, a.out)
        datasets.print_manifest(datasets.read(d)[1])
        print(f"wrote {d}")
        return
    if a.cmd == "gbm":
        from . import gbm
        gbm.train(a.dataset, a.models, a.sim_out, a.eval_day, inject=a.inject)
        return
    if a.cmd == "pipeline":
        from . import pipeline
        pipeline.print_run(pipeline.run(a.ledger, a.models, a.sim_out, leak=a.leak,
                                        inject=a.inject))
        return
    if a.cmd == "abtest":
        from . import evaluate as ev
        ev.print_abtest(ev.abtest(a.ledger, a.experiment, a.sim_out, a.dataset,
                                  goal=a.goal.upper()))
        return
    if a.cmd == "explain":
        from . import pricing
        pricing.print_explain(pricing.explain(a.ledger, a.request, a.campaign, a.day))
        return
    if a.cmd == "frontier":
        from . import pricing
        pricing.print_frontier(pricing.frontier(a.ledger, a.campaign, a.budget, a.target_cpi,
                                                a.target_roas, a.day, a.sim_out))
        return
    if a.cmd == "rlpace":
        from . import rlpace
        rlpace.print_compare(rlpace.compare(a.ledger, a.campaign, a.sim_out, a.test_day, a.train_day,
                                            a.budget, a.episodes, log=lambda *_: None))
        return
    if a.cmd == "uplift":
        from . import uplift
        _, rep = uplift.train(a.ledger, a.models, a.sim_out, log=lambda *_: None)
        uplift.print_report(rep)
        return
    if a.cmd == "win":
        from . import win
        _, rep = win.train(a.ledger, a.models, use_mtw=not a.no_mtw, last_days=a.days,
                           sim_out=a.sim_out, curve=a.curve or a.curse, curse=a.curse,
                           sample_rate=a.sample_rate)
        win.print_report(rep)
        return
    if a.cmd == "kappa":
        from . import kappa
        c = kappa.compute(a.ledger)
        kappa.publish(a.models, c)
        kappa.print_calibration(c)
        return
    if a.cmd == "lift":
        from . import lift
        lift.print_report(lift.measure(a.ledger, a.sim_out, a.days, a.holdout_share, a.app))
        return
    if a.cmd == "quarantine":
        from . import quarantine as qr
        if a.action == "list":
            qr.print_list(qr.load(a.datasets))
        elif a.action == "inspect":
            if a.day is None:
                raise SystemExit("inspect needs --day")
            qr.print_inspect(qr.inspect(a.ledger, a.day, a.field))
        elif a.action == "remove":
            print("removed" if qr.remove(a.datasets, a.entry or "") else f"no entry {a.entry}")
        else:
            lo, hi = a.from_ms, a.to_ms
            if a.day is not None:
                lo, hi = a.day * qr.DAY_MS, (a.day + 1) * qr.DAY_MS
            if lo is None or hi is None:
                raise SystemExit("add needs --day, or --from-ms and --to-ms")
            where = dict(w.split("=", 1) for w in a.where)
            e = qr.add(a.datasets, "postbacks", lo, hi, where, a.reason)
            qr.print_list([e])
            print("rebuild the datasets (or run the pipeline), and recount the bidder's "
                  "device profiles: POST /api/profiles/recount")
        return
    if a.cmd == "onnx":
        from . import onnx
        onnx.print_report(onnx.export(a.bundle, a.out, a.target))
        return
    if a.cmd == "pace":
        from . import pacing, simout
        camp = simout.table(a.sim_out, "logs", "campaigns")
        budgets = {} if camp is None else {str(c): b for c, b in zip(camp.campaign_id, camp.daily_budget)}
        f = pacing.forecast(a.ledger, budgets, a.day)
        pacing.publish(a.models, f)
        pacing.print_forecast(f)
        return
    if a.cmd == "shadetest":
        from . import evaluate as ev
        ev.print_shadetest(ev.shadetest(a.ledger, a.models, a.experiment))
        return
    if a.cmd == "shadow":
        from . import evaluate as ev
        r = ev.shadow(a.ledger, a.models, a.experiment)
        print(f"\nshadow {r['experiment']} ({r['model']}): {r['scored']:,} requests scored live")
        print(f"  click: {r['click_match']:,} of {r['click_rows']:,} match offline to 1e-9 "
              f"(largest difference {r['click_max_diff']:.2g}); {r['click_out_of_bounds']:,} outside "
              "the bidder's bounds, priced by the fallback")
        print(f"  install after a click: {r['install_match']:,} of {r['install_rows']:,} "
              f"(largest difference {r['install_max_diff']:.2g})")
        return
    if a.cmd in ("replay", "ope", "periods"):
        from . import evaluate as ev
        if a.cmd == "replay":
            ev.print_replay(ev.replay(a.ledger, a.models, a.day, a.b, a.a, a.sim_out))
        elif a.cmd == "ope":
            ev.print_ope(ev.ope(a.ledger, a.policy, a.reward, dataset_dir=a.dataset))
        else:
            ev.print_periods(ev.periods(a.ledger, a.sim_out, a.dataset))
        return
    if a.cmd == "labels":
        from . import labels
        labels.print_report(labels.report(a.ledger, a.sim_out))
        return
    if a.cmd == "find":
        from . import parity
        parity.print_found(parity.find(a.ledger, a.day, a.goal, a.reason,
                                       a.explored, a.clicked, a.limit))
        return
    if a.cmd == "parity":
        from . import parity
        r = parity.check(a.ledger)
        print(f"parity: {r['rows']:,} bids compared, {r['mismatched_rows']:,} "
              f"with a mismatch: {'PASS' if r['passed'] else 'FAIL'}")
        for f in r["features"]:
            if f["mismatched"]:
                print(f"  {f['feature']}: {f['mismatched']:,} of {f['compared']:,} "
                      f"({f['rate']:.1%}), e.g. {f['examples'][0]}")
        for d in r["days"]:
            print(f"  day {d['day']}: {d['mismatched']:,} of {d['rows']:,} bids mismatched")
        if r["note"]:
            print(r["note"])
        return
    if a.cmd == "trace":
        from . import parity
        parity.print_trace(parity.trace(a.ledger, a.request_id))
        return
    if a.cmd == "train":
        p = Params(eval_day=a.eval_day, train_days=a.train_days,
                   downsample_rate=a.downsample_rate, constant=a.constant)
        if a.no_crosses:
            p.spec.crosses = []
        train(a.ledger, a.models, p, sim_out=a.sim_out)
    else:
        from .server import serve
        serve(a.ledger, a.models, a.sim_out, a.host, a.port)


if __name__ == "__main__":
    main()
