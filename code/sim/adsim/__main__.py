"""Command line: python -m adsim --preset crawl --days 1 --out out/crawl"""
from __future__ import annotations

import argparse
import time

from . import openrtb
from .config import preset
from .history import DAY_MS
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
    p.add_argument("--settle", default=None, choices=["run", "day"],
                   help="live mode: when the MMP settles installs and sends "
                        "postbacks: at the end of the run, or at the end of "
                        "each simulated day, so the bidder hears about "
                        "conversions while it bids (default: day for walk "
                        "and run, run for crawl, as Part II used)")
    p.add_argument("--scenario", default="",
                   help="live mode: failure scenarios for Chapter 12, comma "
                        "separated: wrong-target (a CPC target 100x too "
                        "high from 06:00), lost-billing (an exchange stops "
                        "sending billing notices for rewarded video from "
                        "12:00), creative-test (one campaign gets a much "
                        "better creative, Chapter 21; keep it on for every "
                        "run of the same market), bad-labels (the MMP credits "
                        "false installs on the run's first day, 06:00-18:00, "
                        "Chapter 24), ad-roas (hypercasual apps earn ad "
                        "revenue and the market gets Ad ROAS campaigns, "
                        "Chapter 30; keep it on for every run of the same "
                        "market), uplift (heavy players install games of "
                        "their favorite genre on their own, so many credited "
                        "installs weren't caused, Chapter 32; keep it on too), "
                        "traffic-surge (three times the requests from 13:00 to "
                        "16:00 UTC on the run's first day, Chapter 34), "
                        "full-share (every advertiser shares its install data, "
                        "for the Part IV dataset; keep it on), "
                        "new-game (at 12:00 UTC on the run's first day, a new hypercasual hit "
                        "takes over the genre's audience; Chapter 29)")
    p.add_argument("--workers", type=int, default=32,
                   help="concurrent connections to the bidder in live mode")
    a = p.parse_args()
    t0 = time.time()
    scen = tuple(x for x in a.scenario.replace(" ", "").split(",") if x)
    # creative-test changes the world, not the traffic: it's a config flag
    creative_test = "creative-test" in scen
    # bad-labels changes what the MMP sends on the run's first day
    bad_labels = "bad-labels" in scen
    # ad-roas changes the world too: hypercasual apps earn ad revenue, and
    # the market gets Ad ROAS campaigns (Chapter 30)
    ad_roas = "ad-roas" in scen
    # uplift changes the world: heavy players install their favorite genre
    # on their own (Chapter 32)
    uplift = "uplift" in scen
    # traffic-surge triples the requests from 13:00 to 16:00 UTC on the
    # run's first day (Chapter 34)
    surge = "traffic-surge" in scen
    # full-share: every advertiser shares its install data (the Part IV
    # dataset, Chapters 25-29); keep it on for every run of the market
    full_share = "full-share" in scen
    # new-game: at 12:00 UTC on the run's first day, a new hypercasual hit
    # (not ours) takes over the genre's audience, and the genre's games get
    # 40% of their installs from then on (Chapter 29); run the shifted days
    # in one command
    new_game = "new-game" in scen
    scen = tuple(x for x in scen
                 if x not in ("creative-test", "bad-labels", "ad-roas", "uplift", "traffic-surge",
                              "full-share", "new-game"))
    if scen and not a.bidder:
        p.error("--scenario needs --bidder")
    policy = (HttpPolicy(a.bidder, workers=a.workers, admin_url=a.bidder_admin,
                         seed=a.seed, scenarios=scen,
                         pending_path=f"{a.out}/state/pending_postbacks.parquet")
              if a.bidder else None)
    if a.start_day == "auto":
        start = policy.next_day() if policy else 0
        if start:
            print(f"the bidder's ledger has days 0 to {start - 1}: "
                  f"continuing at day {start}")
    else:
        start = int(a.start_day)
    cfg = preset(a.preset, a.scale, seed=a.seed, days=a.days).with_(
        start_day=start, creative_test=creative_test, ad_revenue=ad_roas, uplift=uplift,
        new_game_ms=(start * DAY_MS + 12 * 3_600_000) if new_game else -1)
    if uplift or full_share:
        # a lift study's advertisers share their full install data, and so
        # do all of them in the Part IV dataset's market
        cfg = cfg.with_(nonattributed_sharing=1.0)
    settle = a.settle or ("run" if a.preset == "crawl" else "day")
    if policy is not None and settle == "day" and a.days > 1:
        # one simulated day at a time: the MMP settles each day's installs
        # at its end, and postbacks due later go out when their time comes
        for d in range(a.days):
            res = simulate(cfg.with_(days=1, start_day=start + d, bad_labels=bad_labels and d == 0,
                                     surge=surge and d == 0),
                           policy=policy, progress=True)
            res.write(f"{a.out}/runs/start={start + d:04d}")
    else:
        res = simulate(cfg.with_(bad_labels=bad_labels, surge=surge), policy=policy, progress=True)
    if policy is not None:
        policy.close()
        print(f"live mode: {policy.sent:,} requests to {a.bidder} "
              f"at {policy.rate:,.0f} requests/s, "
              f"{policy.notices_sent:,} notices, {policy.clicks_sent:,} clicks, "
              f"{policy.postbacks_sent:,} postbacks")
    res.write(a.out)
    if policy is not None and not (settle == "day" and a.days > 1):
        # Live runs continue the market, so each run's logs and truth are
        # also kept by start day, for analysis across runs.
        res.write(f"{a.out}/runs/start={start:04d}")
    if a.openrtb_sample:
        req = res.logs["bid_requests"].head(a.openrtb_sample)
        openrtb.write_jsonl(req, f"{a.out}/logs/bid_requests_sample.jsonl")
    print(f"wrote {a.out} in {time.time() - t0:.0f}s")
    for name, df in res.logs.items():
        print(f"  logs/{name}: {len(df):,} rows")


if __name__ == "__main__":
    main()
