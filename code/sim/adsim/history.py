"""Device histories before the simulation starts.

Each device gets a timestamped history of app installs, uninstalls, and
purchases over `history_days`. The history decides which publisher apps
a device sends bid requests from, which advertiser apps it already has,
and it is the raw material for non-attributed MMP data, device profiles,
and pretraining in later chapters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GENRES
from .world import World

DAY_MS = 86_400_000


def _genre_index(apps: pd.DataFrame) -> np.ndarray:
    lookup = {g: i for i, g in enumerate(GENRES)}
    return apps.genre.map(lookup).to_numpy()


def install_weights(world: World, dev_idx: np.ndarray) -> np.ndarray:
    """Unnormalized weights of each device installing each app."""
    apps = world.apps
    g = _genre_index(apps)
    taste = world.affinity[dev_idx][:, g] * len(GENRES) + 0.05
    pop = apps.popularity.to_numpy()[None, :]
    return taste * pop * np.exp(apps.appeal.to_numpy())[None, :]


def make_history(world: World, chunk: int = 5000):
    """Return (events, installed) for the period before the start."""
    cfg = world.cfg
    rng = np.random.default_rng([cfg.seed, 2])
    apps = world.apps
    n_dev, n_apps = len(world.devices), len(apps)
    is_pub = apps.is_publisher.to_numpy()
    is_fraud = apps.is_fraud.to_numpy()
    iap = (apps.monetization == "iap").to_numpy()
    bots = world.devices.is_bot.to_numpy()
    rows_d, rows_a, rows_t = [], [], []
    for start in range(0, n_dev, chunk):
        idx = np.arange(start, min(start + chunk, n_dev))
        w = install_weights(world, idx)
        w[:, is_fraud] *= 0.01
        b = bots[idx]
        w[b] = np.where(is_fraud, 1.0, 0.0)[None, :] + 1e-9
        k = 2 + rng.poisson(6, size=len(idx))
        k = np.minimum(k, n_apps)
        k[b] = np.minimum(3, max(1, is_fraud.sum()))
        # Gumbel top-k: sample k apps per device without replacement
        scores = np.log(w) + rng.gumbel(size=w.shape)
        order = np.argsort(-scores, axis=1)
        for j, d in enumerate(idx):
            chosen = order[j, : k[j]]
            if not is_pub[chosen].any():  # every device plays something
                pubs = np.flatnonzero(is_pub & ~is_fraud)
                chosen = np.append(chosen, pubs[np.argmax(scores[j, pubs])])
            rows_d.append(np.full(len(chosen), d))
            rows_a.append(chosen)
    dev = np.concatenate(rows_d)
    app = np.concatenate(rows_a)
    n = len(dev)
    t_install = -rng.uniform(0, cfg.history_days, n) * DAY_MS
    # some apps were uninstalled again, usually soon after install
    churn = rng.random(n) < 0.3
    t_uninstall = t_install + rng.exponential(3.0, n) * DAY_MS
    churn &= t_uninstall < 0
    keep_pub = is_pub[app]
    churn &= ~(keep_pub & (rng.random(n) < 0.8))  # pub apps mostly kept
    events = [pd.DataFrame({
        "device_idx": dev, "app_id": app, "event": "install",
        "ts_ms": t_install.astype(np.int64), "revenue": 0.0})]
    events.append(pd.DataFrame({
        "device_idx": dev[churn], "app_id": app[churn],
        "event": "uninstall", "ts_ms": t_uninstall[churn].astype(np.int64),
        "revenue": 0.0}))
    # purchases in IAP apps by payers
    devs = world.devices
    payer = rng.random(n) < devs.payer_prob.to_numpy()[dev] * 1.5
    payer &= iap[app]
    p_rows = np.flatnonzero(payer)
    n_buys = 1 + rng.poisson(2.0, size=len(p_rows))
    rep = np.repeat(p_rows, n_buys)
    end = np.where(churn[rep], t_uninstall[rep], 0.0)
    span = np.maximum(end - t_install[rep], 1.0)
    t_buy = t_install[rep] + span * rng.beta(0.6, 2.0, len(rep))
    amount = _price_points(
        rng, devs.spend_scale.to_numpy()[dev[rep]]
        * apps.price_level.to_numpy()[app[rep]])
    events.append(pd.DataFrame({
        "device_idx": dev[rep], "app_id": app[rep], "event": "purchase",
        "ts_ms": t_buy.astype(np.int64), "revenue": amount}))
    events = pd.concat(events, ignore_index=True)
    events = events.sort_values(["device_idx", "ts_ms"], kind="stable")
    installed = pd.DataFrame({"device_idx": dev[~churn],
                              "app_id": app[~churn]})
    return events.reset_index(drop=True), installed


PRICE_POINTS = np.array([0.99, 1.99, 2.99, 4.99, 9.99, 19.99, 49.99, 99.99])


def _price_points(rng: np.random.Generator, scale: np.ndarray):
    raw = rng.lognormal(np.log(np.maximum(scale, 0.1)), 0.8)
    idx = np.searchsorted(PRICE_POINTS, raw).clip(0, len(PRICE_POINTS) - 1)
    return PRICE_POINTS[idx]
