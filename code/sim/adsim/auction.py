"""First-price auctions, notifications, and whether a win is shown.

Competing bidders are summarized by the highest competing price for each
request. It depends on the placement, the geo, the exchange, and, through
a common-value term, on the same device traits that drive our own value.
That correlation is what creates the winner's curse: the impressions we
win cheaply are disproportionately the ones other bidders valued less,
often for good reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GEOS
from .requests import FLOOR_BASE
from .world import World

CLEAR_BY_PLACEMENT = {"rewarded_video": 0.85, "interstitial": 0.9,
                      "banner": 1.0, "native": 0.95}
NO_COMPETITION = 0.08  # share of auctions with no other bidder


def competing_price(world: World, req: pd.DataFrame,
                    rng: np.random.Generator) -> np.ndarray:
    """Highest competing bid (CPM) for each request; 0 if none."""
    d = world.devices
    dev = req.true_device_idx.to_numpy()
    cpm = np.array([GEOS[g].cpm_scale for g in req.true_geo])
    ex = {e.name: e for e in world.cfg.exchanges}
    comp = np.array([ex[e].competition for e in req.exchange])
    base = np.array([FLOOR_BASE[p] for p in req.placement])
    common = (0.6 * np.log(d.click_prop.to_numpy()[dev])
              + 0.4 * np.log(d.install_prop.to_numpy()[dev] + 0.05)
              + 2.0 * d.payer_prob.to_numpy()[dev])
    h = base * cpm * comp * 3.0 * np.exp(common + rng.normal(0, 0.45,
                                                             len(req)))
    h[rng.random(len(req)) < NO_COMPETITION] = 0.0
    return h


def run_auctions(world: World, req: pd.DataFrame, bids: pd.DataFrame,
                 rng: np.random.Generator) -> pd.DataFrame:
    """Resolve auctions for the requests we bid on.

    `bids` has one row per real (non-ghost) bid, with `request_id` and
    `bid` (CPM). Returns one row per bid with the outcome, the price paid,
    the true highest other price, and what the exchange reports.
    """
    r = req.set_index("request_id").loc[bids.request_id]
    h = competing_price(world, r.reset_index(), rng)
    floor = r.bidfloor.to_numpy()
    bid = bids.bid.to_numpy()
    to_beat = np.maximum(h, floor)
    below_floor = bid < floor
    win = (bid >= to_beat) & ~below_floor
    mbtw = np.round(to_beat + 0.01, 2)
    ex = {e.name: e for e in world.cfg.exchanges}
    policy = np.array([ex[e].mbtw for e in r.exchange])
    reported = mbtw.astype(float).copy()
    reported[policy == "rounded"] = np.ceil(mbtw[policy == "rounded"]
                                            * 4) / 4
    reported[(policy == "loss_only") & win] = np.nan
    reported[policy == "none"] = np.nan
    return pd.DataFrame({
        "request_id": bids.request_id.to_numpy(),
        "exchange": r.exchange.to_numpy(),
        "bid": bid,
        "win": win,
        "below_floor": below_floor,
        "price": np.where(win, bid, np.nan),  # first price
        "true_highest_other": h,
        "reported_mbtw": reported,
        "sends_loss": np.array([ex[e].sends_loss_notices
                                for e in r.exchange]),
    })


def clear(world: World, req: pd.DataFrame, results: pd.DataFrame,
          rng: np.random.Generator) -> pd.DataFrame:
    """Decide which wins become shown, billable impressions."""
    won = results[results.win]
    r = req.set_index("request_id").loc[won.request_id]
    ex = {e.name: e for e in world.cfg.exchanges}
    p = (np.array([ex[e].clear_rate for e in r.exchange])
         * np.array([CLEAR_BY_PLACEMENT[p] for p in r.placement]))
    shown = rng.random(len(won)) < p
    # pre-fetched ads are shown some time after the auction
    delay = np.minimum(rng.exponential(60_000, len(won)), 1_800_000)
    return pd.DataFrame({
        "request_id": won.request_id.to_numpy(),
        "shown": shown,
        "imp_ts_ms": (r.ts_ms.to_numpy() + delay).astype(np.int64),
    })
