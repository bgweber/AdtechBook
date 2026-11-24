"""The true response model: how devices respond to ads.

These functions are the ground truth of the simulated market. Given a
device, a placement, and an ad, they return the true probabilities of a
click, an install after a click, an install after a view without a click,
and an organic install that would happen without any ad. Models in the
book are trained on outcomes sampled from these probabilities, and their
calibration and lift estimates can be checked against them exactly.

The response model has a few deliberate structures, so that the methods
in the book have something to find:

- placement and creative effects on clicks, including accidental clicks
  on dense ad layouts, which rarely install;
- genre taste, which matters more for installs than for clicks, and which
  sequence and two-tower models can learn from install histories;
- creative style preferences per device, for personalized creatives;
- fatigue, so repeated impressions of a campaign lose effect;
- an organic install rate, so some attributed installs were not caused by
  the ad, and lift differs from attributed conversion.
"""
from __future__ import annotations

import numpy as np

from .config import GENRES, GEOS, STYLES
from .world import World

CLICK_BASE = {"rewarded_video": -3.2, "interstitial": -3.6,
              "banner": -5.0, "native": -4.3}
VIEW_BASE = {"rewarded_video": -5.6, "interstitial": -6.5,
             "banner": -8.5, "native": -7.5}
ORGANIC_DAILY = 5e-4  # scale of the organic install rate per device-app
BOT_CTR = 0.25


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _taste(world: World, dev: np.ndarray, app: np.ndarray) -> np.ndarray:
    gi = {g: i for i, g in enumerate(GENRES)}
    genre = world.apps.genre.map(gi).to_numpy()[app]
    return world.affinity[dev, genre] * len(GENRES)  # mean 1


def response(world: World, dev, pub_app, placement, adv_app, creative,
             n_seen):
    """True probabilities for impressions.

    All arguments are arrays of equal length: device index, publisher app
    id, placement name, advertiser app id, creative id, and the number of
    impressions of this campaign the device has seen today. Returns a dict
    of arrays: p_click, p_install_click, p_install_view.
    """
    d, apps, cr = world.devices, world.apps, world.creatives
    taste = np.log(_taste(world, dev, adv_app) + 0.05)
    si = {s: i for i, s in enumerate(STYLES)}
    style = cr["style"].map(si).to_numpy()[creative]
    style_fit = world.style_pref[dev, style] * len(STYLES) - 1.0
    quality = cr.quality.to_numpy()[creative]
    density = apps.ad_density.to_numpy()[pub_app] - 1.0
    fatigue = np.log1p(n_seen)
    appeal = apps.appeal.to_numpy()[adv_app]
    store = apps.store_quality.to_numpy()[adv_app]
    cbase = np.array([CLICK_BASE[p] for p in placement])
    vbase = np.array([VIEW_BASE[p] for p in placement])
    click_prop = np.log(d.click_prop.to_numpy()[dev])
    inst_prop = d.install_prop.to_numpy()[dev]
    p_click = sigmoid(cbase + click_prop + 0.35 * taste + 0.3 * quality
                      + 0.3 * style_fit + 0.5 * density - 0.25 * fatigue
                      + 0.2 * appeal)
    p_ic = sigmoid(-2.2 + 0.9 * taste + np.log(inst_prop + 1e-9)
                   + 0.8 * appeal + store - 0.9 * density
                   + 0.2 * quality - 0.15 * fatigue)
    p_iv = sigmoid(vbase + 0.9 * taste + np.log(inst_prop + 1e-9)
                   + 0.8 * appeal + 0.4 * style_fit + 0.3 * quality
                   - 0.3 * fatigue)
    if world.cfg.ad_revenue:
        # hypercasual games convert far better than the rest: simple,
        # instantly playable, and free (Chapter 30)
        hc = apps.monetization.to_numpy()[adv_app] == "ads"
        p_ic = np.where(hc, sigmoid(logit(p_ic) + HC_INSTALL_LIFT), p_ic)
        p_iv = np.where(hc, sigmoid(logit(p_iv) + HC_INSTALL_LIFT), p_iv)
    bot = d.is_bot.to_numpy()[dev]
    p_click = np.where(bot, BOT_CTR, p_click)
    p_ic = np.where(bot, 0.0, p_ic)
    p_iv = np.where(bot, 0.0, p_iv)
    return {"p_click": p_click, "p_install_click": p_ic,
            "p_install_view": p_iv}


def organic_rate(world: World, dev, adv_app) -> np.ndarray:
    """True rate of organic installs per day, for device-app pairs."""
    apps, d = world.apps, world.devices
    pop = apps.popularity.to_numpy()[adv_app]
    pop = pop / np.median(apps.popularity)
    taste = _taste(world, dev, adv_app)
    rate = (ORGANIC_DAILY * pop * taste * d.install_prop.to_numpy()[dev]
            * np.exp(apps.appeal.to_numpy()[adv_app]))
    return rate


def payer_prob(world: World, dev, adv_app) -> np.ndarray:
    """True chance that an install of this app by this device pays."""
    apps, d = world.apps, world.devices
    iap = (apps.monetization.to_numpy()[adv_app] == "iap")
    taste = _taste(world, dev, adv_app)
    p = d.payer_prob.to_numpy()[dev] * (0.6 + 0.4 * np.minimum(taste, 3))
    return np.where(iap, np.clip(p, 0, 0.95), 0.0)


# The purchase process for a payer, used by the MMP module: a first
# purchase after an exponential delay, then a Poisson number of repeat
# purchases spread uniformly over 30 days. Amounts are lognormal.
FIRST_PURCHASE_MEAN_DAYS = 1.5
REPEAT_PURCHASES_30D = 2.0
AMOUNT_SIGMA = 0.8


# Ad revenue for ad-monetized (hypercasual) apps (Chapter 30). A player who
# installs sees ads on the install day and on every later day they're
# still playing. Day 1 follows the app's D1 retention; after that a player
# keeps playing each day with a chance that rises with their taste for the
# genre. A day's revenue is the device's ad impressions (its activity, as
# the DSP sees it in requests) times the eCPM of its country, with noise.
AD_IMPS_PER_DAY = 40  # interstitials and rewarded videos, for a device of average activity
AD_ECPM_USD = 15.0  # in a country with cpm_scale 1.0 (the US)
AD_DAY_SIGMA = 0.5  # lognormal noise on each day's revenue
AD_DAY0_SHARE = 0.5  # the install day is a partial day
HC_INSTALL_LIFT = 2.6  # log-odds lift of a hypercasual install, with ad revenue on: an IPM of tens, not single digits


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def ad_stickiness(world: World, dev, adv_app) -> np.ndarray:
    """Chance a player still playing on day d plays on day d + 1 (d >= 1)."""
    return np.clip(0.55 + 0.1 * np.minimum(_taste(world, dev, adv_app), 3),
                   0, 0.9)


def ad_revenue_per_day(world: World, dev) -> np.ndarray:
    """Mean ad revenue on a full day of play, before the day's noise."""
    d = world.devices
    cpm = np.array([GEOS[g].cpm_scale for g in d.geo.to_numpy()[dev]])
    act = d.activity.to_numpy()[dev] / world.cfg.requests_per_device
    return act * AD_IMPS_PER_DAY * AD_ECPM_USD * cpm / 1000


def expected_ad_revenue(world: World, dev, adv_app,
                        days: int = 7) -> np.ndarray:
    """True expected ad revenue per install within `days` of the install."""
    apps, d = world.apps, world.devices
    ads = apps.monetization.to_numpy()[adv_app] == "ads"
    r1 = (apps.retention.to_numpy()[adv_app]
          * np.where(d.is_bot.to_numpy()[dev], 0, 1))
    q = ad_stickiness(world, dev, adv_app)
    later = sum(q ** (k - 1) for k in range(1, days))  # days 1 .. days-1
    mean = ad_revenue_per_day(world, dev) * np.exp(AD_DAY_SIGMA ** 2 / 2)
    ev = mean * (AD_DAY0_SHARE + r1 * later)
    return np.where(ads & ~d.is_bot.to_numpy()[dev], ev, 0.0)


def expected_revenue(world: World, dev, adv_app,
                     days: float = 7.0) -> np.ndarray:
    """True expected revenue per install within `days` of the install."""
    apps, d = world.apps, world.devices
    scale = (d.spend_scale.to_numpy()[dev]
             * apps.price_level.to_numpy()[adv_app])
    mean_amount = scale * np.exp(AMOUNT_SIGMA ** 2 / 2)
    p_first = 1 - np.exp(-days / FIRST_PURCHASE_MEAN_DAYS)
    n = p_first + REPEAT_PURCHASES_30D * min(days, 30.0) / 30.0
    return payer_prob(world, dev, adv_app) * n * mean_amount
