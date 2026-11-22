"""Command line.

    python -m adtrain train --ledger out/ledger --models out/models
    python -m adtrain serve --ledger out/ledger --models out/models --port 8090
    python -m adtrain parity --ledger out/ledger
    python -m adtrain trace <request_id> --ledger out/ledger
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
    a = ap.parse_args()
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
