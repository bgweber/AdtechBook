"""Installs, attribution, postbacks, and purchases.

Every install in the simulated world comes from one of two processes:
organic installs, which happen at a device- and app-specific rate whether
or not anyone shows an ad, and installs caused by our ads. Other ad
networks also touch some devices. A simulated MMP then applies last-touch
attribution with click-through and view-through windows, and sends
postbacks to us for the installs it credits to us, with delays, losses,
and duplicates. Advertisers that share non-attributed data also send the
rest.

The ground truth keeps what really happened: which installs our ads
caused, and which ones we were only credited with.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .history import DAY_MS
from . import truth
from .world import World


def organic_installs(world: World, installed: pd.DataFrame,
                     horizon_days: float) -> pd.DataFrame:
    """Organic installs of advertiser apps over the horizon."""
    cfg = world.cfg
    rng = np.random.default_rng([cfg.seed, 6])
    adv = world.apps.app_id[world.apps.is_advertiser].to_numpy()
    n_dev = len(world.devices)
    have = set(zip(installed.device_idx, installed.app_id))
    rows = []
    for start in range(0, n_dev, 4000):
        dev = np.arange(start, min(start + 4000, n_dev))
        d = np.repeat(dev, len(adv))
        a = np.tile(adv, len(dev))
        rate = truth.organic_rate(world, d, a)
        hit = rng.random(len(d)) < 1 - np.exp(-rate * horizon_days)
        rows.append(pd.DataFrame({
            "device_idx": d[hit], "app_id": a[hit],
            "ts_ms": (rng.uniform(0, horizon_days, hit.sum())
                      * DAY_MS).astype(np.int64)}))
    org = pd.concat(rows, ignore_index=True)
    keep = [(x, y) not in have for x, y in zip(org.device_idx, org.app_id)]
    return org[keep].reset_index(drop=True)


def resolve_installs(world: World, installed: pd.DataFrame,
                     organic: pd.DataFrame, imps: pd.DataFrame,
                     clicks: pd.DataFrame) -> pd.DataFrame:
    """Combine organic and ad-driven installs and attribute them.

    `imps` are shown impressions with ad-path install draws, `clicks` the
    clicks among them. Returns one row per install with its source, whether
    our ads caused it, and who the MMP credited.
    """
    cfg = world.cfg
    rng = np.random.default_rng([cfg.seed, 7])
    ad = imps[imps.ad_install][[
        "true_device_idx", "adv_app_id", "ad_install_ts_ms",
        "ad_install_path", "imp_ts_ms", "impression_id"]].rename(
        columns={"true_device_idx": "device_idx", "adv_app_id": "app_id",
                 "ad_install_ts_ms": "ts_ms"})
    ad = ad.assign(source="ad")
    org = organic.assign(source="organic", ad_install_path="",
                         imp_ts_ms=np.nan, impression_id="")
    allc = pd.concat([org, ad], ignore_index=True)
    allc = allc.sort_values("ts_ms", kind="stable")
    # the first install of each app on each device is the one that counts
    first = allc.drop_duplicates(["device_idx", "app_id"], keep="first")
    have = set(zip(installed.device_idx, installed.app_id))
    first = first[[(x, y) not in have for x, y in
                   zip(first.device_idx, first.app_id)]].copy()
    # caused: the ad installed first, and no organic install would have
    # happened within the click window after the impression
    org_t = organic.set_index(["device_idx", "app_id"]).ts_ms
    key = list(zip(first.device_idx, first.app_id))
    o = org_t.reindex(key).to_numpy(dtype=float)
    window = cfg.click_window_s * 1000
    first["caused"] = (first.source == "ad") & ~(
        (o >= first.imp_ts_ms.to_numpy() - window)
        & (o <= first.imp_ts_ms.to_numpy() + window))
    first = first.reset_index(drop=True)
    return _attribute(world, first, imps, clicks, rng)


def _latest_before(events: pd.DataFrame, inst: pd.DataFrame,
                   tcol: str, window_ms: int) -> pd.DataFrame:
    """For each install, our latest event before it within the window."""
    if events.empty:
        return pd.DataFrame(index=inst.index,
                            columns=["t", "ref", "campaign_id"])
    ev = events.sort_values(tcol)
    ins = inst[["device_idx", "app_id", "ts_ms"]].copy()
    ins["ts_ms"] = ins.ts_ms.astype(float)
    ins["_row"] = np.arange(len(ins))
    ins = ins.sort_values("ts_ms")
    ev = ev.rename(columns={tcol: "t"})
    ev["t"] = ev.t.astype(float)
    m = pd.merge_asof(ins, ev, left_on="ts_ms", right_on="t",
                      by=["device_idx", "app_id"], direction="backward",
                      tolerance=float(window_ms))
    return m.sort_values("_row").set_index(inst.index)


def _attribute(world, inst, imps, clicks, rng):
    cfg = world.cfg
    n = len(inst)
    ours_c = clicks.rename(columns={"true_device_idx": "device_idx",
                                    "adv_app_id": "app_id"})[[
        "device_idx", "app_id", "click_ts_ms", "click_id", "campaign_id"]]
    ours_c = ours_c.rename(columns={"click_id": "ref"})
    ours_v = imps.rename(columns={"true_device_idx": "device_idx",
                                  "adv_app_id": "app_id"})[[
        "device_idx", "app_id", "imp_ts_ms", "impression_id",
        "campaign_id"]].rename(columns={"impression_id": "ref"})
    c = _latest_before(ours_c, inst, "click_ts_ms", cfg.click_window_s * 1000)
    v = _latest_before(ours_v, inst, "imp_ts_ms", cfg.view_window_s * 1000)
    # another network's touch: a click or a view before the install
    other = rng.random(n) < cfg.other_network_touch
    other_click = other & (rng.random(n) < 0.6)
    t_other = inst.ts_ms.to_numpy() - rng.exponential(8 * 3_600_000, n)
    oc = np.where(other_click, t_other, np.nan)
    ov = np.where(other & ~other_click, t_other, np.nan)
    ov[ov < inst.ts_ms.to_numpy() - cfg.view_window_s * 1000] = np.nan
    our_c = c.t.to_numpy(dtype=float)
    our_v = v.t.to_numpy(dtype=float)
    credit = np.full(n, "organic", dtype=object)
    touch = np.full(n, "", dtype=object)
    any_click = ~np.isnan(our_c) | ~np.isnan(oc)
    ours_click_wins = ~np.isnan(our_c) & ~(oc > our_c)
    credit[any_click] = np.where(ours_click_wins[any_click], "us", "other")
    touch[any_click] = "click"
    no_click = ~any_click
    any_view = no_click & (~np.isnan(our_v) | ~np.isnan(ov))
    ours_view_wins = ~np.isnan(our_v) & ~(ov > our_v)
    credit[any_view] = np.where(ours_view_wins[any_view], "us", "other")
    touch[any_view] = "view"
    out = inst.copy()
    out["credited_to"] = credit
    out["touch_type"] = touch
    out["click_id"] = np.where((credit == "us") & (touch == "click"),
                               c.ref.to_numpy(), "")
    out["impression_ref"] = np.where((credit == "us") & (touch == "view"),
                                     v.ref.to_numpy(), "")
    camp = np.where(touch == "click", c.campaign_id.to_numpy(),
                    v.campaign_id.to_numpy())
    out["campaign_id"] = np.where(credit == "us", camp, -1)
    ctit = (out.ts_ms.to_numpy() - np.nan_to_num(our_c)) / 1000
    out["rejected"] = (credit == "us") & (touch == "click") & (ctit < 10)
    out["campaign_id"] = out.campaign_id.fillna(-1).astype(int)
    return out


def purchases(world: World, installs: pd.DataFrame) -> pd.DataFrame:
    """Purchases in the 30 days after each install, and D1 retention."""
    cfg = world.cfg
    rng = np.random.default_rng([cfg.seed, 8])
    dev = installs.device_idx.to_numpy()
    app = installs.app_id.to_numpy()
    payer = rng.random(len(installs)) < truth.payer_prob(world, dev, app)
    rows = np.flatnonzero(payer)
    n_rep = rng.poisson(truth.REPEAT_PURCHASES_30D, len(rows))
    t0 = installs.ts_ms.to_numpy()[rows]
    first = t0 + rng.exponential(truth.FIRST_PURCHASE_MEAN_DAYS,
                                 len(rows)) * DAY_MS
    rep = np.repeat(rows, n_rep)
    t_rep = (installs.ts_ms.to_numpy()[rep]
             + rng.uniform(0, 30, len(rep)) * DAY_MS)
    idx = np.concatenate([rows, rep])
    ts = np.concatenate([first, t_rep])
    scale = (world.devices.spend_scale.to_numpy()[dev[idx]]
             * world.apps.price_level.to_numpy()[app[idx]])
    amount = np.round(rng.lognormal(np.log(scale), truth.AMOUNT_SIGMA), 2)
    pur = pd.DataFrame({
        "install_row": idx, "device_idx": dev[idx], "app_id": app[idx],
        "event": "purchase", "ts_ms": ts.astype(np.int64),
        "revenue": np.maximum(amount, 0.99)})
    ret_p = (world.apps.retention.to_numpy()[app]
             * np.where(world.devices.is_bot.to_numpy()[dev], 0, 1))
    retained = rng.random(len(installs)) < ret_p
    r = np.flatnonzero(retained)
    ret = pd.DataFrame({
        "install_row": r, "device_idx": dev[r], "app_id": app[r],
        "event": "d1_open",
        "ts_ms": (installs.ts_ms.to_numpy()[r] + DAY_MS
                  + rng.uniform(0, DAY_MS, len(r))).astype(np.int64),
        "revenue": 0.0})
    return pd.concat([pur, ret], ignore_index=True).sort_values(
        "ts_ms", kind="stable").reset_index(drop=True)


def postbacks(world: World, installs: pd.DataFrame,
              events: pd.DataFrame) -> pd.DataFrame:
    """Postbacks the MMP sends us, in the MMP's own vocabulary."""
    cfg = world.cfg
    rng = np.random.default_rng([cfg.seed, 9])
    ours = installs[(installs.credited_to == "us") & ~installs.rejected]
    ours = ours.reset_index().rename(columns={"index": "install_row"})
    apps = world.apps
    devs = world.devices
    ifa = np.where(devs.lmt.to_numpy(), "", devs.device_id.to_numpy())
    inst_pb = pd.DataFrame({
        "install_row": ours.install_row,
        "event_name": "install",
        "event_ts_ms": ours.ts_ms.astype(np.int64),
        "revenue_usd": 0.0,
        "touch_type": ours.touch_type,
        "click_id": ours.click_id,
        "impression_id": ours.impression_ref,
        "campaign_id": ours.campaign_id,
    })
    ev = events.merge(ours[["install_row", "touch_type", "click_id",
                            "impression_ref", "campaign_id"]],
                      on="install_row")
    ev_pb = pd.DataFrame({
        "install_row": ev.install_row,
        "event_name": ev.event.map({"purchase": "af_purchase",
                                    "d1_open": "af_d1_open"}),
        "event_ts_ms": ev.ts_ms,
        "revenue_usd": ev.revenue,
        "touch_type": ev.touch_type,
        "click_id": ev.click_id,
        "impression_id": ev.impression_ref,
        "campaign_id": ev.campaign_id,
    })
    pb = pd.concat([inst_pb, ev_pb], ignore_index=True)
    row = pb.install_row.to_numpy()
    pb["app_bundle"] = apps.bundle.to_numpy()[installs.app_id.to_numpy()[row]]
    pb["advertising_id"] = ifa[installs.device_idx.to_numpy()[row]]
    n = len(pb)
    delay = rng.lognormal(np.log(120_000), 1.5, n)
    slow = rng.random(n) < 0.01
    delay[slow] += rng.uniform(1, 48, slow.sum()) * 3_600_000
    pb["received_ts_ms"] = (pb.event_ts_ms + delay).astype(np.int64)
    pb = pb[rng.random(n) >= cfg.postback_loss]
    dup = pb[rng.random(len(pb)) < cfg.postback_duplicate].copy()
    dup["received_ts_ms"] += rng.integers(1_000, 600_000, len(dup))
    pb = pd.concat([pb, dup], ignore_index=True)
    pb["postback_id"] = [f"pb{r:09d}{e}" for r, e in
                         zip(pb.install_row, pb.event_name.str[:6])]
    return pb.drop(columns=["install_row"]).sort_values(
        "received_ts_ms", kind="stable").reset_index(drop=True)


def nonattributed(world: World, installs: pd.DataFrame,
                  events: pd.DataFrame, history: pd.DataFrame):
    """Installs and events not credited to us, for sharing advertisers."""
    cfg = world.cfg
    rng = np.random.default_rng([cfg.seed, 10])
    adv = world.apps.app_id[world.apps.is_advertiser].to_numpy()
    sharing = adv[rng.random(len(adv)) < cfg.nonattributed_sharing]
    devs = world.devices
    ifa = np.where(devs.lmt.to_numpy(), "", devs.device_id.to_numpy())
    others = installs[(installs.credited_to != "us")
                      & installs.app_id.isin(sharing)]
    inst = pd.DataFrame({
        "advertising_id": ifa[others.device_idx],
        "app_bundle": world.apps.bundle.to_numpy()[others.app_id],
        "event_name": "install",
        "event_ts_ms": others.ts_ms.astype(np.int64),
        "revenue_usd": 0.0,
        "media_source": np.where(others.credited_to == "other",
                                 "other_network", "organic")})
    ev = events[events.install_row.isin(others.index)]
    evs = pd.DataFrame({
        "advertising_id": ifa[ev.device_idx],
        "app_bundle": world.apps.bundle.to_numpy()[ev.app_id],
        "event_name": ev.event.map({"purchase": "af_purchase",
                                    "d1_open": "af_d1_open"}),
        "event_ts_ms": ev.ts_ms, "revenue_usd": ev.revenue,
        "media_source": "unknown"})
    hist = history[history.app_id.isin(sharing)]
    hist = pd.DataFrame({
        "advertising_id": ifa[hist.device_idx],
        "app_bundle": world.apps.bundle.to_numpy()[hist.app_id],
        "event_name": hist.event.map({"install": "install",
                                      "uninstall": "uninstall",
                                      "purchase": "af_purchase"}),
        "event_ts_ms": hist.ts_ms, "revenue_usd": hist.revenue,
        "media_source": "unknown"})
    out = pd.concat([hist, inst, evs], ignore_index=True)
    out = out[out.advertising_id != ""]
    return out.sort_values("event_ts_ms", kind="stable").reset_index(
        drop=True), sharing
