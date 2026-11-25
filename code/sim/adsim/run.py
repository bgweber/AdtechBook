"""Run the simulated market against a bidding policy.

`simulate` generates the world and its history, then steps through each
day in chunks of `cfg.chunk_minutes`. In each chunk, the policy decides
on the chunk's requests, the market runs the auctions, decides which wins
are shown, and records what the DSP would see. When all days have run,
installs, attribution, postbacks, and purchases are resolved. Clicks are
resolved chunk by chunk, since they follow impressions within seconds.

The result is two sets of tables:

- `logs`: what a DSP would have in its own logs and from its partners.
- `truth`: the hidden traits and true probabilities behind them, for
  checking models. Never use these as model inputs.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import auction, mmp, outcomes, truth
from .config import SimConfig
from .history import DAY_MS, make_history
from .policy import CrawlPolicy, standardize
from .requests import make_requests
from .world import make_world

NEW_GAME_SHARE = 0.4  # installs of the genre's games after a new hit launches (Chapter 29)

PUBLIC_REQUEST_COLS = [
    "request_id", "ts_ms", "exchange", "tmax", "at", "ifa", "os", "osv",
    "make", "model", "connectiontype", "country", "bundle", "placement",
    "instl", "rwdd", "ext_rewarded", "bidfloor"]


@dataclass
class SimResult:
    logs: dict = field(default_factory=dict)
    truth: dict = field(default_factory=dict)

    def write(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        for kind, tables in (("logs", self.logs), ("truth", self.truth)):
            (out / kind).mkdir(parents=True, exist_ok=True)
            for name, df in tables.items():
                df.to_parquet(out / kind / f"{name}.parquet", index=False)


def simulate(cfg: SimConfig, policy=None, keep_requests: bool = True,
             progress: bool = False) -> SimResult:
    world = make_world(cfg)
    history, installed = make_history(world)
    if policy is None:
        policy = CrawlPolicy(world.campaigns, world.creatives,
                             holdout_share=cfg.holdout_share,
                             holdout_salt=cfg.holdout_salt, seed=cfg.seed)
    # A run that continues at a later day gets fresh random streams.
    tail = [cfg.start_day] if cfg.start_day else []
    rng = np.random.default_rng([cfg.seed, 4, *tail])
    # Clicks get their own stream, so resolving them as the run goes
    # doesn't change the auctions.
    rng_click = np.random.default_rng([cfg.seed, 5, *tail])
    camps = world.campaigns.set_index("campaign_id")
    adv_app = camps.app_id
    all_req, all_dec, all_auc, all_imp = [], [], [], []
    # A live bidder gets the campaign book before the first request, as a
    # DSP's campaign management system would push it.
    if hasattr(policy, "setup"):
        policy.setup(world.campaigns,
                     world.creatives.drop(columns=["quality"], errors="ignore"))
    for day in range(cfg.start_day, cfg.start_day + cfg.days):
        req = make_requests(world, installed, day)
        spend = defaultdict(float)
        seen = defaultdict(int)  # (device, campaign) impressions today
        step = cfg.chunk_minutes * 60_000
        edges = np.arange(day * DAY_MS, (day + 1) * DAY_MS + step, step)
        ts = req.ts_ms.to_numpy()
        for lo, hi in zip(edges[:-1], edges[1:]):
            chunk = req[(ts >= lo) & (ts < hi)]
            if chunk.empty:
                continue
            remaining = {c: camps.daily_budget[c] - spend[c]
                         for c in camps.index}
            dec = standardize(policy.decide(
                chunk[PUBLIC_REQUEST_COLS],
                {"remaining": remaining, "now_ms": lo}))
            all_dec.append(dec)
            # Chunks with no bids still run, on empty tables, so a run
            # where the DSP never bids (listen-only) has well-formed logs.
            bids = dec[dec.action == "bid"]
            res = auction.run_auctions(world, chunk, bids, rng)
            res = res.merge(bids[["request_id", "campaign_id",
                                  "creative_id"]], on="request_id")
            all_auc.append(res)
            clr = auction.clear(world, chunk, res, rng)
            won = res[res.win].merge(clr, on="request_id")
            imp = won[won.shown].merge(
                chunk[["request_id", "true_device_idx", "true_app_id",
                       "placement", "true_geo"]], on="request_id")
            dev = imp.true_device_idx.to_numpy()
            cid = imp.campaign_id.to_numpy().astype(int)
            n_seen = np.array([seen[(d, c)] for d, c in zip(dev, cid)])
            for d, c in zip(dev, cid):
                seen[(d, c)] += 1
            imp["adv_app_id"] = adv_app.loc[cid].to_numpy()
            imp["n_seen_today"] = n_seen
            p = truth.response(world, dev, imp.true_app_id.to_numpy(),
                               imp.placement.to_numpy(),
                               imp.adv_app_id.to_numpy(),
                               imp.creative_id.to_numpy().astype(int),
                               n_seen)
            if cfg.new_game_ms >= 0:
                # a new game takes over a genre's audience (Chapter 29): its
                # players install the genre's other games far less often
                hit = ((imp.imp_ts_ms.to_numpy() >= cfg.new_game_ms)
                       & (world.apps.genre.to_numpy()[imp.adv_app_id.to_numpy()] == cfg.new_game_genre))
                for k in ("p_install_click", "p_install_view"):
                    p[k] = np.where(hit, p[k] * NEW_GAME_SHARE, p[k])
            for k, v in p.items():
                imp[k] = v
            for c, cost in zip(cid, imp.price.to_numpy() / 1000):
                spend[c] += cost
            # Clicks happen seconds after the impression, so they're
            # resolved here, while the run goes, and a live bidder hears
            # about them through its click trackers.
            imp["impression_id"] = "i" + imp.request_id.astype(str)
            imp["creative_id"] = imp.creative_id.astype(int)
            imp["campaign_id"] = imp.campaign_id.astype(int)
            imp = outcomes.engagement_and_clicks(world, imp, rng_click)
            all_imp.append(imp)
            # A live bidder hears about its wins, losses, billed
            # impressions, and clicks through the URLs in its bids.
            if hasattr(policy, "notify"):
                policy.notify(res, imp)
        if keep_requests:
            all_req.append(req)
        if progress:
            print(f"day {day}: {len(req):,} requests")
    dec = pd.concat(all_dec, ignore_index=True)
    auc = pd.concat(all_auc, ignore_index=True)
    imps = pd.concat(all_imp, ignore_index=True)
    clicks = imps[imps.clicked].copy()
    clicks["click_id"] = "c" + clicks.request_id
    horizon = cfg.start_day + cfg.days + cfg.click_window_s / 86400 + 1
    organic = mmp.organic_installs(world, installed, horizon)
    installs = mmp.resolve_installs(world, installed, organic, imps,
                                    clicks)
    events = mmp.purchases(world, installs)
    if cfg.ad_revenue:
        events = pd.concat([events, mmp.ad_revenue(world, installs, events)],
                           ignore_index=True).sort_values(
            "ts_ms", kind="stable").reset_index(drop=True)
    pbs = mmp.postbacks(world, installs, events)
    if cfg.bad_labels:
        bad = mmp.misattributed(world, imps, cfg.start_day)
        print(f"scenario bad-labels: the MMP credits {len(bad):,} false installs on day "
              f"{cfg.start_day}, {mmp.BAD_LABELS_FROM_H:02d}:00-{mmp.BAD_LABELS_TO_H:02d}:00 "
              "(probabilistic matching)")
        pbs = pd.concat([pbs, bad], ignore_index=True).sort_values(
            "received_ts_ms", kind="stable").reset_index(drop=True)
    # A live bidder gets the MMP's postbacks at its postback endpoint. The
    # installs are resolved once the run ends, so they're sent then, each
    # with the simulated time the MMP would have sent it.
    if hasattr(policy, "postbacks"):
        policy.postbacks(pbs, until_ms=(cfg.start_day + cfg.days) * 86_400_000)
    na, sharing = mmp.nonattributed(world, installs, events, history)
    notes = _notifications(auc, imps, rng)
    req_all = pd.concat(all_req, ignore_index=True) if all_req else None
    # Each exchange invoices what it billed, by day, exchange, and ad
    # format (Chapter 12 reconciles this with the DSP's own count).
    invoices = _invoices(imps, req_all) if req_all is not None else None
    if invoices is not None and hasattr(policy, "invoices"):
        policy.invoices(invoices)
    res = SimResult()
    if req_all is not None:
        res.logs["bid_requests"] = req_all[PUBLIC_REQUEST_COLS]
    res.logs["bid_log"] = dec
    res.logs["notifications"] = notes
    res.logs["impressions"] = imps[[
        "impression_id", "request_id", "imp_ts_ms", "campaign_id",
        "creative_id", "video_quartile"]]
    res.logs["clicks"] = clicks[["click_id", "impression_id",
                                 "click_ts_ms"]].assign(
        click_ts_ms=lambda d: d.click_ts_ms.astype(np.int64))
    res.logs["postbacks"] = pbs
    if invoices is not None:
        res.logs["invoices"] = invoices
    res.logs["nonattributed"] = na
    res.logs["campaigns"] = world.campaigns
    res.logs["creatives"] = world.creatives
    res.logs["app_metadata"] = world.apps[["app_id", "bundle", "name",
                                           "genre", "monetization"]]
    t = res.truth
    t["devices"] = world.devices
    t["apps"] = world.apps
    t["history"] = history.assign(
        device_id=world.devices.device_id.to_numpy()[history.device_idx])
    t["impressions"] = _truth_impressions(world, imps, organic, cfg)
    t["auctions"] = auc[["request_id", "bid", "win", "true_highest_other"]]
    t["installs"] = installs.assign(
        device_id=world.devices.device_id.to_numpy()[installs.device_idx])
    t["events"] = events
    if req_all is not None:
        t["requests"] = req_all[["request_id", "true_device_idx",
                                 "true_app_id", "true_geo", "true_tier",
                                 "true_invalid"]]
    t["sharing_apps"] = pd.DataFrame({"app_id": sharing})
    return res


def _notifications(auc: pd.DataFrame, imps: pd.DataFrame, rng):
    """Win, loss, and billing notices, with some duplicates from retries."""
    win = auc[auc.win]
    loss = auc[~auc.win & auc.sends_loss]
    reason = np.where(loss.below_floor, 100, 102)  # OpenRTB loss codes
    rows = [
        pd.DataFrame({"request_id": win.request_id, "type": "win",
                      "price": win.price, "loss_reason": np.nan,
                      "mbtw": win.reported_mbtw}),
        pd.DataFrame({"request_id": loss.request_id, "type": "loss",
                      "price": np.nan, "loss_reason": reason,
                      "mbtw": loss.reported_mbtw}),
        pd.DataFrame({"request_id": imps.request_id, "type": "billing",
                      "price": imps.price, "loss_reason": np.nan,
                      "mbtw": np.nan}),
    ]
    n = pd.concat(rows, ignore_index=True)
    n["event_id"] = n.type.astype(str).str[0] + n.request_id.astype(str)
    dup = n[rng.random(len(n)) < 0.005]
    return pd.concat([n, dup], ignore_index=True)


def _invoices(imps: pd.DataFrame, req: pd.DataFrame) -> pd.DataFrame:
    """What each exchange bills: shown impressions at their clearing
    price, by the day of the auction, exchange, and placement."""
    x = imps[["request_id", "placement", "price"]].merge(
        req[["request_id", "ts_ms", "exchange"]], on="request_id")
    x["day"] = x.ts_ms // DAY_MS
    g = x.groupby(["day", "exchange", "placement"], as_index=False).agg(
        impressions=("price", "size"), spend=("price", "sum"))
    g["spend"] = g.spend / 1000
    return g


def _truth_impressions(world, imps, organic, cfg):
    dev = imps.true_device_idx.to_numpy()
    app = imps.adv_app_id.to_numpy()
    rate = truth.organic_rate(world, dev, app)
    p0 = 1 - np.exp(-rate * cfg.click_window_s / 86400)
    p_ad = (imps.p_click * imps.p_install_click
            + (1 - imps.p_click) * imps.p_install_view).to_numpy()
    return pd.DataFrame({
        "impression_id": imps.impression_id,
        "device_idx": dev,
        "adv_app_id": app,
        "p_click": imps.p_click,
        "p_install_click": imps.p_install_click,
        "p_install_view": imps.p_install_view,
        "p_install_ad": p_ad,
        "p_organic_window": p0,
        "lift": (1 - p0) * p_ad,
        "expected_d7_revenue": truth.expected_revenue(world, dev, app, 7),
        "expected_d7_ad_revenue": truth.expected_ad_revenue(world, dev, app, 7),
        "expected_d3_ad_revenue": truth.expected_ad_revenue(world, dev, app, 3),
        "n_seen_today": imps.n_seen_today,
    })
