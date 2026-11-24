"""The synthetic world: devices, apps, campaigns, and creatives.

Everything here is generated once per run from the seed. Devices carry
hidden traits (genre taste, click, install and spend propensities) that
drive every outcome in the simulation. The traits are ground truth: they
are written to their own tables and never appear in the logs a DSP sees.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FORMATS, GENRES, GEOS, PLACEMENTS, STYLES, SimConfig

# (make, model as the exchange sends it, marketing name, tier, newest OS)
DEVICE_MODELS = [
    ("samsung", "SM-S918B", "Galaxy S23 Ultra", 3, 15),
    ("samsung", "SM-A546E", "Galaxy A54", 2, 15),
    ("samsung", "SM-A155F", "Galaxy A15", 1, 14),
    ("samsung", "SM-A057F", "Galaxy A05s", 1, 14),
    ("xiaomi", "23129RAA4G", "Redmi Note 13", 1, 14),
    ("xiaomi", "2201116SG", "Redmi Note 11 Pro", 1, 13),
    ("xiaomi", "23049PCD8G", "Xiaomi 13T", 2, 14),
    ("oppo", "CPH2591", "Reno11", 2, 14),
    ("vivo", "V2312", "Y28", 1, 14),
    ("realme", "RMX3834", "C67", 1, 14),
    ("motorola", "motorola edge 40", "Edge 40", 2, 14),
    ("motorola", "moto g54 5G", "Moto G54", 1, 14),
    ("google", "Pixel 8", "Pixel 8", 3, 15),
    ("google", "Pixel 7a", "Pixel 7a", 2, 15),
    ("oneplus", "CPH2581", "OnePlus 12", 3, 15),
    ("tecno", "TECNO KJ5", "Spark 20", 1, 13),
    ("infinix", "Infinix X6833B", "Note 30", 1, 13),
]

_SYLLABLES = ("gem", "farm", "cozy", "tile", "jelly", "royal", "block",
              "word", "merge", "cash", "slot", "hero", "dragon", "kingdom",
              "garden", "bubble", "pop", "candy", "race", "goal", "tycoon",
              "idle", "dash", "star", "quest", "mine", "craft", "zen")


@dataclass
class World:
    cfg: SimConfig
    devices: pd.DataFrame  # one row per device, hidden traits included
    apps: pd.DataFrame  # one row per app in the catalog
    campaigns: pd.DataFrame
    creatives: pd.DataFrame
    affinity: np.ndarray  # devices x genres, rows sum to 1
    style_pref: np.ndarray  # devices x creative styles, rows sum to 1


def _uuids(rng: np.random.Generator, n: int) -> np.ndarray:
    hi = rng.integers(0, 2**63, size=n, dtype=np.int64)
    lo = rng.integers(0, 2**63, size=n, dtype=np.int64)
    out = []
    for a, b in zip(hi, lo):
        h = f"{a:016x}{b:016x}"
        out.append(f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}")
    return np.array(out)


def _bundles(rng: np.random.Generator, n: int) -> list[str]:
    names, seen = [], set()
    while len(names) < n:
        a, b = rng.choice(_SYLLABLES, size=2, replace=False)
        dev = rng.choice(_SYLLABLES)
        name = f"com.{dev}games.{a}{b}"
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def make_devices(cfg: SimConfig, rng: np.random.Generator):
    n = cfg.n_devices
    geo_codes = list(cfg.geos)
    geo_w = np.array([cfg.geos[g] for g in geo_codes], dtype=float)
    geo = rng.choice(geo_codes, size=n, p=geo_w / geo_w.sum())
    spend = np.array([GEOS[g].spend_scale for g in geo])
    payer = np.array([GEOS[g].payer_scale for g in geo])
    # richer geos skew toward higher device tiers
    cpm = np.array([GEOS[g].cpm_scale for g in geo])
    tiers = np.array([m[3] for m in DEVICE_MODELS])
    model_idx = np.empty(n, dtype=int)
    for i, c in enumerate(cpm):
        w = np.exp((tiers - 1.5) * (c - 0.4) * 2.0)
        model_idx[i] = rng.choice(len(DEVICE_MODELS), p=w / w.sum())
    newest = np.array([DEVICE_MODELS[i][4] for i in model_idx])
    osv = newest - rng.choice([0, 0, 1, 1, 2, 3], size=n)
    affinity = rng.dirichlet(np.full(len(GENRES), 0.4), size=n)
    style_pref = rng.dirichlet(np.full(len(STYLES), 0.8), size=n)
    is_bot = rng.random(n) < cfg.bot_share
    devices = pd.DataFrame({
        "device_id": _uuids(rng, n),
        "geo": geo,
        "os": "android",
        "osv": osv.astype(int),
        "make": [DEVICE_MODELS[i][0] for i in model_idx],
        "model": [DEVICE_MODELS[i][1] for i in model_idx],
        "model_name": [DEVICE_MODELS[i][2] for i in model_idx],
        "tier": [DEVICE_MODELS[i][3] for i in model_idx],
        "lmt": rng.random(n) < cfg.lmt_rate,
        "is_bot": is_bot,
        # hidden traits
        "click_prop": rng.lognormal(0.0, 0.6, n),
        "install_prop": rng.lognormal(0.0, 0.5, n),
        "payer_prob": np.clip(rng.beta(0.6, 12.0, n) * payer, 0, 0.9),
        "spend_scale": rng.lognormal(np.log(spend), 0.8),
        "activity": rng.lognormal(np.log(cfg.requests_per_device), 0.7, n)
        * np.where(is_bot, 5.0, 1.0),
        "favorite_genre": np.array(GENRES)[affinity.argmax(1)],
    })
    devices.loc[devices.is_bot, ["install_prop", "payer_prob"]] = 0.0
    return devices, affinity, style_pref


def make_apps(cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_apps
    bundles = _bundles(rng, n)
    genre = rng.choice(GENRES, size=n)
    popularity = rng.pareto(1.2, n) + 0.05
    role = rng.random(n)
    is_pub = role < cfg.share_publisher
    is_adv = (role >= 1 - cfg.share_advertiser) | (rng.random(n) < 0.05)
    # The book's two running examples.
    bundles[0], genre[0], is_pub[0], is_adv[0] = (
        "com.example.gemgarden", "puzzle", True, False)
    popularity[0] = popularity.max() * 1.2
    bundles[1], genre[1], is_pub[1], is_adv[1] = (
        "com.example.cozyfarm", "farming", False, True)
    names = [b.split(".")[-1].title() for b in bundles]
    names[0], names[1] = "Gem Garden", "Cozy Farm Tycoon"
    fraud = is_pub & (rng.random(n) < cfg.fraud_app_share)
    fraud[0] = False
    placements = []
    for g in genre:
        if g == "hypercasual":
            placements.append("interstitial,banner,rewarded_video")
        elif g in ("puzzle", "casual", "farming"):
            placements.append("rewarded_video,interstitial,banner")
        else:
            placements.append("rewarded_video,interstitial,native")
    monet = np.where(genre == "hypercasual", "ads", "iap")
    return pd.DataFrame({
        "app_id": np.arange(n),
        "bundle": bundles,
        "name": names,
        "genre": genre,
        "is_publisher": is_pub,
        "is_advertiser": is_adv,
        "is_fraud": fraud,
        "placements": placements,
        "popularity": popularity,
        "appeal": rng.normal(0.0, 0.5, n),  # how good the game looks
        "store_quality": rng.normal(0.0, 0.3, n),  # store page conversion
        "ad_density": rng.uniform(0.5, 1.5, n),  # accidental-click factor
        "monetization": monet,
        "price_level": rng.lognormal(0.0, 0.4, n),
        "retention": rng.beta(4, 6, n),
    })


def make_campaigns(cfg: SimConfig, apps: pd.DataFrame,
                   rng: np.random.Generator):
    adv = apps[apps.is_advertiser].app_id.to_numpy()
    adv = np.concatenate([[1], adv[adv != 1]])  # Cozy Farm Tycoon first
    rows = []
    geo_codes = list(cfg.geos)
    for c in range(cfg.n_campaigns):
        app_id = int(adv[c % len(adv)])
        goal = cfg.goals[c % len(cfg.goals)]
        k = rng.integers(1, len(geo_codes) + 1)
        geos = sorted(rng.choice(geo_codes, size=k, replace=False))
        if c == 0:
            geos = sorted(geo_codes)
        mean_cpm = float(np.mean([GEOS[g].cpm_scale for g in geos]))
        target = {
            "CPC": round(0.4 * mean_cpm + 0.02, 3),
            "CPI": round(8.0 * mean_cpm + 0.5, 2),
            "CPA": round(25.0 * mean_cpm + 2.0, 2),
            "ROAS": round(float(rng.uniform(0.15, 0.4)), 2),
        }[goal]
        rows.append({
            "campaign_id": c,
            "app_id": app_id,
            "goal": goal,
            "target": target,
            "daily_budget": round(float(rng.uniform(10, 60) * mean_cpm
                                        * 5 * cfg.n_devices / 20_000), 2),
            "geos": ",".join(geos),
            "start_day": 0,
        })
    campaigns = pd.DataFrame(rows)
    if cfg.late_launch_share > 0:
        # some campaigns launch later, for cold-start checks (Chapter 15);
        # a separate stream, so presets without them don't change
        r2 = np.random.default_rng([cfg.seed, 31])
        late = r2.random(len(campaigns)) < cfg.late_launch_share
        lo, hi = cfg.late_launch_days
        campaigns.loc[late, "start_day"] = r2.integers(lo, hi + 1, late.sum())
    if cfg.ad_revenue:
        campaigns = pd.concat([campaigns, _ad_roas_campaigns(cfg, apps,
                                                             len(campaigns))],
                              ignore_index=True)
    crows = []
    cid = 0
    for c in campaigns.itertuples():
        for _ in range(cfg.creatives_per_campaign):
            fmt = rng.choice(FORMATS, p=[0.6, 0.25, 0.15])
            crows.append({
                "creative_id": cid,
                "campaign_id": c.campaign_id,
                "format": fmt,
                "length_s": int(rng.choice([15, 30])) if fmt == "video"
                else 0,
                "orientation": rng.choice(["portrait", "landscape"],
                                          p=[0.8, 0.2]),
                "style": rng.choice(STYLES),
                "quality": float(rng.normal(0.0, 0.3)),
            })
            cid += 1
    creatives = pd.DataFrame(crows)
    if cfg.creative_test:
        # the creative test of Chapter 21: the CPI campaign with the largest
        # budget among those running from day 0 gets one creative far better
        # than its other two
        cpi = campaigns[(campaigns.goal == "CPI") & (campaigns.start_day == 0)]
        c = int(cpi.sort_values("daily_budget", ascending=False).campaign_id.iloc[0])
        ids = creatives.index[creatives.campaign_id == c]
        creatives.loc[ids, "quality"] = -0.5
        creatives.loc[ids[0], "quality"] = 3.0
    return campaigns, creatives


def _ad_roas_campaigns(cfg: SimConfig, apps: pd.DataFrame,
                       first_id: int) -> pd.DataFrame:
    """Ad ROAS campaigns for hypercasual apps (Chapter 30).

    Added after the market's other campaigns, from their own random stream,
    so the other campaigns are the same with or without them. If the
    catalog has too few hypercasual advertisers, some hypercasual apps
    become advertisers.
    """
    rng = np.random.default_rng([cfg.seed, 41])
    hc = apps.genre.to_numpy() == "hypercasual"
    adv = np.flatnonzero(hc & apps.is_advertiser.to_numpy())
    need = cfg.ad_roas_campaigns - len(adv)
    if need > 0:
        more = np.flatnonzero(hc & ~apps.is_advertiser.to_numpy()
                              & (apps.app_id.to_numpy() > 1))[:need]
        apps.loc[more, "is_advertiser"] = True
        adv = np.concatenate([adv, more])
    geo_codes = list(cfg.geos)
    rows = []
    for i in range(cfg.ad_roas_campaigns):
        k = rng.integers(2, len(geo_codes) + 1)
        geos = sorted(rng.choice(geo_codes, size=k, replace=False))
        mean_cpm = float(np.mean([GEOS[g].cpm_scale for g in geos]))
        rows.append({
            "campaign_id": first_id + i,
            "app_id": int(adv[i % len(adv)]),
            "goal": "AD_ROAS",  # the target is D3 ad revenue / spend
            "target": round(float(rng.uniform(0.1, 0.2)), 2),
            "daily_budget": round(float(rng.uniform(40, 160) * mean_cpm
                                        * 5 * cfg.n_devices / 20_000), 2),
            "geos": ",".join(geos),
            "start_day": 0,
        })
    return pd.DataFrame(rows)


def make_world(cfg: SimConfig) -> World:
    rng = np.random.default_rng([cfg.seed, 1])
    devices, affinity, style_pref = make_devices(cfg, rng)
    apps = make_apps(cfg, rng)
    campaigns, creatives = make_campaigns(cfg, apps, rng)
    return World(cfg, devices, apps, campaigns, creatives, affinity,
                 style_pref)


def placement_list(apps: pd.DataFrame) -> list[list[str]]:
    return [p.split(",") for p in apps.placements]


