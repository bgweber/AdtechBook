"""Command line: python -m adsim --preset crawl --days 1 --out out/crawl"""
from __future__ import annotations

import argparse
import time

from . import openrtb
from .config import preset
from .run import simulate


def main() -> None:
    p = argparse.ArgumentParser(prog="adsim")
    p.add_argument("--preset", default="crawl",
                   choices=["crawl", "walk", "run"])
    p.add_argument("--scale", default="laptop",
                   choices=["tiny", "laptop", "medium", "large"])
    p.add_argument("--days", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="out/sim")
    p.add_argument("--openrtb-sample", type=int, default=1000,
                   help="also write this many requests as OpenRTB JSON")
    a = p.parse_args()
    cfg = preset(a.preset, a.scale, seed=a.seed, days=a.days)
    t0 = time.time()
    res = simulate(cfg, progress=True)
    res.write(a.out)
    if a.openrtb_sample:
        req = res.logs["bid_requests"].head(a.openrtb_sample)
        openrtb.write_jsonl(req, f"{a.out}/logs/bid_requests_sample.jsonl")
    print(f"wrote {a.out} in {time.time() - t0:.0f}s")
    for name, df in res.logs.items():
        print(f"  logs/{name}: {len(df):,} rows")


if __name__ == "__main__":
    main()
