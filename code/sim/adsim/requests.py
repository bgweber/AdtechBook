"""Bid requests from simulated exchanges.

Each device sends requests from the publisher apps it has installed, on a
daily cycle in its local time. Exchanges add their own quirks: country
codes in two formats, rewarded flags in an extension, missing device
models, and junk appended to app bundles. Bots on fraudulent apps add
invalid traffic.

The table returned has two kinds of columns. Columns that an exchange
would send are named as in OpenRTB (`ifa`, `bundle`, `bidfloor`, ...),
in the exchange's own format. Columns that only the simulator knows are
prefixed with `true_` and must never be used as model inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GEOS
from .history import DAY_MS
from .world import World

# Share of a day's requests in each local hour: low at night, peaks in
# the evening.
DIURNAL = np.array([
    1.0, 0.6, 0.4, 0.3, 0.3, 0.5, 1.0, 1.8, 2.3, 2.4, 2.4, 2.5,
    2.7, 2.6, 2.5, 2.5, 2.7, 3.0, 3.4, 3.8, 4.0, 3.6, 2.8, 1.8,
])
DIURNAL = DIURNAL / DIURNAL.sum()

PLACEMENT_WEIGHT = {"rewarded_video": 0.4, "interstitial": 0.4,
                    "banner": 0.25, "native": 0.15}
FLOOR_BASE = {"rewarded_video": 6.0, "interstitial": 4.0,
              "banner": 0.5, "native": 1.5}  # USD CPM at a cpm_scale of 1
ZERO_IFA = "00000000-0000-0000-0000-000000000000"


SURGE_EXTRA = 2  # extra requests per usual one during a surge (Chapter 34)


def make_requests(world: World, installed: pd.DataFrame,
                  day: int) -> pd.DataFrame:
    cfg = world.cfg
    rng = np.random.default_rng([cfg.seed, 3, day])
    devs, apps = world.devices, world.apps
    # publisher apps each device can send requests from
    pub = installed.merge(apps[["app_id", "is_publisher", "popularity"]],
                          on="app_id")
    pub = pub[pub.is_publisher].reset_index(drop=True)
    n_req = rng.poisson(devs.activity.to_numpy())
    # pick one of the device's publisher apps for each request
    pub = pub.sort_values("device_idx", kind="stable").reset_index(drop=True)
    starts = np.searchsorted(pub.device_idx.to_numpy(),
                             np.arange(len(devs)))
    counts = np.bincount(pub.device_idx, minlength=len(devs))
    has = counts > 0
    n_req = np.where(has, n_req, 0)
    dev_idx = np.repeat(np.arange(len(devs)), n_req)
    w = pub.popularity.to_numpy() ** 0.5
    pick = np.empty(len(dev_idx), dtype=int)
    for d in np.flatnonzero(n_req):
        s, c = starts[d], counts[d]
        ww = w[s:s + c]
        sel = slice(np.searchsorted(dev_idx, d),
                    np.searchsorted(dev_idx, d, side="right"))
        pick[sel] = s + rng.choice(c, size=n_req[d], p=ww / ww.sum())
    app_id = pub.app_id.to_numpy()[pick]
    n = len(dev_idx)
    # time of day, in local time, converted to UTC
    geo = devs.geo.to_numpy()[dev_idx]
    offset = np.array([GEOS[g].utc_offset for g in geo])
    local_hour = rng.choice(24, size=n, p=DIURNAL)
    utc_hour = (local_hour - offset) % 24
    ts = (day * DAY_MS + utc_hour * 3_600_000
          + rng.integers(0, 3_600_000, size=n))
    if cfg.surge and day == cfg.start_day:
        # a traffic surge (Chapter 34), on the run's first day: a large
        # promotion triples the requests from 13:00 to 16:00 UTC, with
        # devices and apps drawn from the day's own traffic
        srng = np.random.default_rng([cfg.seed, 43, day])
        lo, hi = day * DAY_MS + 13 * 3_600_000, day * DAY_MS + 16 * 3_600_000
        extra = srng.choice(n, size=SURGE_EXTRA * int(((ts >= lo) & (ts < hi)).sum()))
        dev_idx = np.concatenate([dev_idx, dev_idx[extra]])
        app_id = np.concatenate([app_id, app_id[extra]])
        ts = np.concatenate([ts, lo + srng.integers(0, hi - lo, size=len(extra))])
        n = len(dev_idx)
        geo = devs.geo.to_numpy()[dev_idx]
    # exchange, placement, floor
    ex = cfg.exchanges
    ex_idx = rng.choice(len(ex), size=n,
                        p=[e.traffic_share for e in ex])
    app_pl = apps.placements.str.split(",").to_numpy()
    placement = np.empty(n, dtype=object)
    for a in np.unique(app_id):
        m = app_id == a
        opts = app_pl[a]
        pw = np.array([PLACEMENT_WEIGHT[p] for p in opts])
        placement[m] = rng.choice(opts, size=m.sum(), p=pw / pw.sum())
    cpm_scale = np.array([GEOS[g].cpm_scale for g in geo])
    comp = np.array([e.competition for e in ex])[ex_idx]
    pop = apps.popularity.to_numpy()[app_id]
    floor = (np.array([FLOOR_BASE[p] for p in placement]) * cpm_scale
             * comp * (0.6 + 0.2 * np.log1p(pop))
             * rng.lognormal(0.0, 0.35, n))
    floor = np.round(floor * (rng.random(n) < 0.9), 2)  # some have none
    # identity
    ifa = devs.device_id.to_numpy()[dev_idx].astype(object)
    lmt = devs.lmt.to_numpy()[dev_idx]
    ifa[lmt] = ZERO_IFA
    no_ifa = rng.random(n) < cfg.no_ifa_rate
    ifa[no_ifa] = ""
    # exchange quirks
    country = np.where(
        np.array([e.country_format for e in ex])[ex_idx] == "alpha2",
        [GEOS[g].code2 for g in geo], geo)
    model = devs.model.to_numpy()[dev_idx].astype(object)
    miss = rng.random(n) < np.array(
        [e.model_missing_rate for e in ex])[ex_idx]
    model[miss] = ""
    bundle = apps.bundle.to_numpy()[app_id].astype(object)
    suffix = rng.random(n) < np.array(
        [e.bundle_suffix_rate for e in ex])[ex_idx]
    bundle[suffix] = bundle[suffix] + "$hl=en"
    rew = placement == "rewarded_video"
    in_ext = np.array([e.rewarded_in_ext for e in ex])[ex_idx]
    conn = np.where(rng.random(n) < 0.55, 2,
                    rng.choice([4, 5, 6], size=n, p=[0.3, 0.5, 0.2]))
    tier = devs.tier.to_numpy()[dev_idx]
    order = np.argsort(ts, kind="stable")
    req = pd.DataFrame({
        "request_id": [f"{ex[e].name[:2]}{day:03d}{i:09d}"
                       for i, e in enumerate(ex_idx)],
        "ts_ms": ts,
        "exchange": np.array([e.name for e in ex])[ex_idx],
        "tmax": np.array([e.tmax_ms for e in ex])[ex_idx],
        "at": 1,
        "ifa": ifa,
        "os": "android",
        "osv": devs.osv.to_numpy()[dev_idx].astype(str),
        "make": devs.make.to_numpy()[dev_idx],
        "model": model,
        "connectiontype": conn,
        "country": country,
        "bundle": bundle,
        "placement": placement,  # normalized; raw flags below
        "instl": np.isin(placement, ["rewarded_video", "interstitial"]
                         ).astype(int),
        "rwdd": (rew & ~in_ext).astype(int),
        "ext_rewarded": (rew & in_ext).astype(int),
        "bidfloor": floor,
        # simulator-only columns
        "true_device_idx": dev_idx,
        "true_app_id": app_id,
        "true_geo": geo,
        "true_tier": tier,
        "true_invalid": devs.is_bot.to_numpy()[dev_idx]
        | apps.is_fraud.to_numpy()[app_id],
    }).iloc[order].reset_index(drop=True)
    return req
