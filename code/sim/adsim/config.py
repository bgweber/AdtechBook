"""Configuration and presets for the simulated market.

A `SimConfig` fully determines a simulation run, together with its seed.
Presets match the stages of the book:

- `crawl`: low-CPM Android geos, two exchanges, CPC and a first CPI
  campaign (Part II).
- `walk`: adds tier-1 geos, a third exchange, CPI and ROAS campaigns, and
  postbacks shared as non-attributed data by some advertisers (Part III).
- `run`: more of everything, for the deep models in Part IV.

`scale` multiplies the number of devices and apps, from a laptop-sized
`tiny` world to a Spark-sized `large` one.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

GENRES = (
    "puzzle", "casual", "farming", "strategy",
    "rpg", "casino", "sports", "hypercasual",
)
PLACEMENTS = ("rewarded_video", "interstitial", "banner", "native")
FORMATS = ("video", "playable", "static")
STYLES = ("gameplay", "story", "ugc", "minimal")


@dataclass(frozen=True)
class Geo:
    """A country, with its market parameters."""

    code3: str
    code2: str
    utc_offset: int  # hours
    cpm_scale: float  # multiplies competing bids and floors
    payer_scale: float  # multiplies the chance an installer pays
    spend_scale: float  # typical purchase size, in USD


GEOS = {
    "IND": Geo("IND", "IN", 5, 0.12, 0.25, 1.5),
    "BRA": Geo("BRA", "BR", -3, 0.20, 0.45, 2.5),
    "IDN": Geo("IDN", "ID", 7, 0.15, 0.35, 1.8),
    "MEX": Geo("MEX", "MX", -6, 0.25, 0.50, 3.0),
    "USA": Geo("USA", "US", -5, 1.00, 1.00, 9.0),
    "JPN": Geo("JPN", "JP", 9, 0.90, 1.20, 12.0),
    "DEU": Geo("DEU", "DE", 1, 0.70, 0.80, 7.0),
}


@dataclass(frozen=True)
class Exchange:
    """A simulated exchange and its quirks.

    `country_format` is how it sends `device.geo.country` ("alpha3" per
    OpenRTB, or "alpha2"). `mbtw` is how it reports the minimum bid to win:
    "exact", "rounded", "loss_only", or "none".
    """

    name: str
    traffic_share: float
    country_format: str = "alpha3"
    mbtw: str = "exact"
    sends_loss_notices: bool = True
    rewarded_in_ext: bool = False  # older OpenRTB: rewarded flag in imp.ext
    model_missing_rate: float = 0.02
    bundle_suffix_rate: float = 0.0  # appends junk to app.bundle
    clear_rate: float = 0.85
    competition: float = 1.0  # multiplies competing bids
    tmax_ms: int = 250


@dataclass(frozen=True)
class SimConfig:
    """Everything that defines a simulation run, apart from the policy."""

    seed: int = 7
    days: int = 1
    start: str = "2026-01-05T00:00:00Z"  # a Monday, UTC
    n_devices: int = 20_000
    n_apps: int = 300  # publisher and advertiser apps share one catalog
    share_publisher: float = 0.6
    share_advertiser: float = 0.25
    n_campaigns: int = 8
    creatives_per_campaign: int = 3
    geos: dict = field(default_factory=lambda: {
        "IND": 0.5, "BRA": 0.3, "IDN": 0.2})
    exchanges: tuple = (
        Exchange("alpha_exchange", 0.6),
        Exchange("beta_exchange", 0.4, country_format="alpha2",
                 mbtw="rounded", rewarded_in_ext=True,
                 model_missing_rate=0.15, bundle_suffix_rate=0.05,
                 clear_rate=0.7),
    )
    goals: tuple = ("CPC", "CPC", "CPI")  # drawn in turn for campaigns
    requests_per_device: float = 25.0  # mean requests per device per day
    no_ifa_rate: float = 0.03  # requests with no device ID
    lmt_rate: float = 0.02  # devices that send the all-zeros IFA
    bot_share: float = 0.01  # devices that are bots (invalid traffic)
    fraud_app_share: float = 0.03  # publisher apps that host the bots
    history_days: int = 180
    nonattributed_sharing: float = 0.0  # advertisers sharing other installs
    other_network_touch: float = 0.35  # installs with another network's touch
    click_window_s: int = 7 * 86400  # MMP click-through window
    view_window_s: int = 86400  # MMP view-through window
    postback_loss: float = 0.005
    postback_duplicate: float = 0.005
    holdout_share: float = 0.05
    holdout_salt: str = "holdout-v1"
    chunk_minutes: int = 15  # the run loop's step

    def with_(self, **kw) -> "SimConfig":
        return replace(self, **kw)


SCALES = {"tiny": 0.1, "laptop": 1.0, "medium": 10.0, "large": 100.0}


def preset(name: str = "crawl", scale: str = "laptop",
           seed: int = 7, days: int = 1) -> SimConfig:
    """Return the config for a part of the book at a given scale."""
    s = SCALES[scale]
    base = SimConfig(seed=seed, days=days)
    if name == "crawl":
        cfg = base
    elif name == "walk":
        cfg = base.with_(
            geos={"IND": 0.3, "BRA": 0.2, "IDN": 0.1, "MEX": 0.1,
                  "USA": 0.2, "JPN": 0.05, "DEU": 0.05},
            exchanges=base.exchanges + (
                Exchange("gamma_exchange", 0.3, mbtw="loss_only",
                         sends_loss_notices=True, clear_rate=0.9,
                         competition=1.2, tmax_ms=200),),
            goals=("CPI", "CPI", "ROAS", "CPC"),
            n_campaigns=16,
            nonattributed_sharing=0.5,
        )
    elif name == "run":
        cfg = preset("walk", "laptop", seed, days).with_(
            n_campaigns=40, n_apps=800, creatives_per_campaign=6,
            goals=("CPI", "ROAS", "ROAS", "CPI", "CPA"),
            nonattributed_sharing=0.7,
        )
    else:
        raise ValueError(f"unknown preset {name!r}")
    shares = [e.traffic_share for e in cfg.exchanges]
    total = sum(shares)
    exchanges = tuple(replace(e, traffic_share=e.traffic_share / total)
                      for e in cfg.exchanges)
    return cfg.with_(
        exchanges=exchanges,
        n_devices=max(500, int(cfg.n_devices * s)),
        n_apps=max(60, int(cfg.n_apps * min(s, 10) ** 0.5)),
    )
