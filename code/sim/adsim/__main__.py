"""Command line: python -m adsim --preset crawl --days 1 --out out/crawl"""
from __future__ import annotations

import argparse
import time

from . import openrtb
from .config import preset
from .live import HttpPolicy
from .run import simulate


def main() -> None:
    p = argparse.ArgumentParser(prog="adsim")
    p.add_argument("--preset", default="crawl",
                   choices=["crawl", "walk", "run"])
    p.add_argument("--scale", default="laptop",
                   choices=["tiny", "laptop", "medium", "large"])
    p.add_argument("--days", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--start-day", default="0",
                   help="continue the same market at a later day, with new "
                        "traffic; 'auto' in live mode asks the bidder for "
                        "the day after the latest one in its ledger")
    p.add_argument("--out", default="out/sim")
    p.add_argument("--openrtb-sample", type=int, default=1000,
                   help="also write this many requests as OpenRTB JSON")
    p.add_argument("--bidder", default=None, metavar="URL",
                   help="live mode: send requests to a bidder, such as "
                        "http://localhost:8080, instead of the built-in "
                        "Python policy")
    p.add_argument("--bidder-admin", default=None, metavar="URL",
                   help="the bidder's admin API, for the campaign book "
                        "(default: the bidder's host on port 8081)")
    p.add_argument("--scenario", default="",
                   help="live mode: failure scenarios for Chapter 12, comma "
                        "separated: wrong-target (a CPC target 100x too "
                        "high from 06:00), lost-billing (an exchange stops "
                        "sending billing notices for rewarded video from "
                        "12:00)")
    p.add_argument("--workers", type=int, default=32,
                   help="concurrent connections to the bidder in live mode")
    a = p.parse_args()
    t0 = time.time()
    scen = tuple(x for x in a.scenario.replace(" ", "").split(",") if x)
    if scen and not a.bidder:
        p.error("--scenario needs --bidder")
    policy = (HttpPolicy(a.bidder, workers=a.workers, admin_url=a.bidder_admin,
                         seed=a.seed, scenarios=scen) if a.bidder else None)
    if a.start_day == "auto":
        start = policy.next_day() if policy else 0
        if start:
            print(f"the bidder's ledger has days 0 to {start - 1}: "
                  f"continuing at day {start}")
    else:
        start = int(a.start_day)
    cfg = preset(a.preset, a.scale, seed=a.seed, days=a.days).with_(
        start_day=start)
    res = simulate(cfg, policy=policy, progress=True)
    if policy is not None:
        policy.close()
        print(f"live mode: {policy.sent:,} requests to {a.bidder} "
              f"at {policy.rate:,.0f} requests/s, "
              f"{policy.notices_sent:,} notices, {policy.clicks_sent:,} clicks, "
              f"{policy.postbacks_sent:,} postbacks")
    res.write(a.out)
    if a.openrtb_sample:
        req = res.logs["bid_requests"].head(a.openrtb_sample)
        openrtb.write_jsonl(req, f"{a.out}/logs/bid_requests_sample.jsonl")
    print(f"wrote {a.out} in {time.time() - t0:.0f}s")
    for name, df in res.logs.items():
        print(f"  logs/{name}: {len(df):,} rows")


if __name__ == "__main__":
    main()
