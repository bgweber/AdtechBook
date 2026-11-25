"""Bidding policies that the simulated market can run against.

A policy sees only what a DSP would see: the bid requests as exchanges
send them, campaign configuration, public app metadata, and its own state
(such as budget spent so far). It returns one decision per request: a
bid, a ghost bid for the holdout, or a no-bid with a reason.

`CrawlPolicy` is the Part II bidder before it has a model: weighted
random campaign selection, prior click and install rates by placement, a
fixed shade, randomized bid multipliers on a share of bids, and a device
holdout with ghost bids. The Go reference bidder replaces it; this policy
exists so the simulator can produce logs on its own.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .config import GEOS

ALPHA2_TO_3 = {g.code2: g.code3 for g in GEOS.values()}
ZERO_IFA = "00000000-0000-0000-0000-000000000000"
CTR_PRIOR = {"rewarded_video": 0.035, "interstitial": 0.025,
             "banner": 0.006, "native": 0.012}
COMPATIBLE = {"rewarded_video": {"video", "playable"},
              "interstitial": {"video", "playable", "static"},
              "banner": {"static"}, "native": {"static", "video"}}


def holdout_bucket(ifa: np.ndarray, salt: str, buckets: int = 1000):
    """Stable bucket for each device ID, from a salted hash."""
    out = np.empty(len(ifa), dtype=int)
    for i, x in enumerate(ifa):
        h = hashlib.sha1(f"{salt}:{x}".encode()).digest()
        out[i] = int.from_bytes(h[:8], "big") % buckets
    return out


def normalize(req: pd.DataFrame) -> pd.DataFrame:
    """Normalize the fields the policy uses, as the bidder would at
    ingest: three-letter countries, the rewarded flag, clean bundles."""
    out = req.copy()
    out["country3"] = out.country.map(lambda c: ALPHA2_TO_3.get(c, c))
    out["bundle_norm"] = (out.bundle.str.split("$").str[0]
                          .str.strip().str.lower())
    out["valid_ifa"] = (out.ifa != "") & (out.ifa != ZERO_IFA)
    return out


# Every policy's decisions are logged with these columns. A policy may
# return fewer; `standardize` fills in the rest as missing.
DECISION_COLUMNS = [
    "request_id", "ts_ms", "action", "reason", "campaign_id", "creative_id",
    "n_eligible", "selection_prob", "p_click_pred", "advertiser_value",
    "media_value", "explore_mult", "bid", "holdout", "explored",
    "policy_version"]


def standardize(dec: pd.DataFrame) -> pd.DataFrame:
    """Decisions with every column in DECISION_COLUMNS, in order."""
    out = dec.copy()
    for c in DECISION_COLUMNS:
        if c not in out:
            out[c] = np.nan
    for c in ("action", "reason", "policy_version"):
        out[c] = out[c].astype(object).where(out[c].notna(), "")
    for c in ("holdout", "explored"):
        out[c] = out[c].fillna(False).astype(bool)
    for c in ("campaign_id", "creative_id", "n_eligible"):
        out[c] = out[c].astype("Int64")
    for c in ("selection_prob", "p_click_pred", "advertiser_value",
              "media_value", "explore_mult", "bid"):
        out[c] = out[c].astype(float)
    return out[DECISION_COLUMNS + [c for c in out if c not in DECISION_COLUMNS]]


class CrawlPolicy:
    version = "crawl-prior-v1"

    def __init__(self, campaigns: pd.DataFrame, creatives: pd.DataFrame,
                 margin: float = 0.2, shade: float = 0.7,
                 explore_share: float = 0.05,
                 explore_range: tuple = (0.6, 1.5),
                 holdout_share: float = 0.05,
                 holdout_salt: str = "holdout-v1",
                 cvr_prior: float = 0.03, value_prior: float = 1.0,
                 action_prior: float = 0.2, ad_value_prior: float = 0.35,
                 cap_cpm: float = 50.0,
                 seed: int = 0):
        self.campaigns = campaigns.reset_index(drop=True)
        self.creatives = creatives
        self.margin, self.shade = margin, shade
        self.explore_share, self.explore_range = explore_share, explore_range
        self.holdout_buckets = int(round(holdout_share * 1000))
        self.salt = holdout_salt
        self.cvr, self.value, self.action = (cvr_prior, value_prior,
                                             action_prior)
        self.ad_value = ad_value_prior
        self.cap = cap_cpm
        self.rng = np.random.default_rng([seed, 100])
        self.geo_sets = [set(g.split(",")) for g in self.campaigns.geos]
        self.fmt_by_campaign = {
            c: set(g.format) for c, g in creatives.groupby("campaign_id")}
        self.creatives_for = {}  # (campaign_id, placement) -> creative ids
        for c in self.campaigns.campaign_id:
            for p, fmts in COMPATIBLE.items():
                ids = creatives[(creatives.campaign_id == c)
                                & creatives.format.isin(fmts)].creative_id
                self.creatives_for[(c, p)] = ids.to_numpy()

    def _value(self, camp, p_click):
        goal, target = camp.goal, camp.target
        if goal == "CPC":
            return target * p_click * 1000
        if goal == "CPI":
            return target * p_click * self.cvr * 1000
        if goal == "CPA":
            return target * p_click * self.cvr * self.action * 1000
        if goal == "ROAS":
            return p_click * self.cvr * self.value / target * 1000
        if goal == "AD_ROAS":  # a hypercasual install earns far less
            return p_click * self.cvr * self.ad_value / target * 1000
        raise ValueError(goal)

    def decide(self, req: pd.DataFrame, state: dict) -> pd.DataFrame:
        """Return one decision per request.

        `state["remaining"]` maps campaign_id to remaining daily budget.
        """
        r = normalize(req)
        n = len(r)
        rng = self.rng
        bucket = holdout_bucket(r.ifa.to_numpy(), self.salt)
        out = {k: np.full(n, v, dtype=object) for k, v in [
            ("action", "nobid"), ("reason", "")]}
        cols = {k: np.full(n, np.nan) for k in [
            "campaign_id", "creative_id", "n_eligible", "selection_prob",
            "p_click_pred", "advertiser_value", "media_value",
            "explore_mult", "bid"]}
        remaining = state.get("remaining", {})
        camps = self.campaigns
        ids = camps.campaign_id.to_numpy()
        budget = camps.daily_budget.to_numpy()
        live = np.array([remaining.get(c, 1.0) > 0 for c in ids])
        weight = np.array([max(remaining.get(c, b) / b, 0.0) + 0.1
                           for c, b in zip(ids, budget)])
        elig_cache = {}
        for i, row in enumerate(r.itertuples()):
            if not row.valid_ifa:
                out["reason"][i] = "no_device_id"
                continue
            key = (row.country3, row.placement)
            if key not in elig_cache:
                elig_cache[key] = np.array([
                    j for j in range(len(camps))
                    if live[j] and row.country3 in self.geo_sets[j]
                    and len(self.creatives_for[(ids[j], row.placement)])],
                    dtype=int)
            elig = elig_cache[key]
            if len(elig) == 0:
                out["reason"][i] = "no_eligible_campaign"
                continue
            w = weight[elig]
            k = rng.choice(len(elig), p=w / w.sum())
            camp = camps.iloc[elig[k]]
            cr_ids = self.creatives_for[(camp.campaign_id, row.placement)]
            c = rng.integers(len(cr_ids))
            p_click = CTR_PRIOR[row.placement]
            a = self._value(camp, p_click)
            vm = a * (1 - self.margin)
            mult = 1.0
            if rng.random() < self.explore_share:
                lo, hi = np.log(self.explore_range)
                mult = float(np.exp(rng.uniform(lo, hi)))
            bid = min(vm * self.shade * mult, self.cap)
            cols["campaign_id"][i] = camp.campaign_id
            cols["creative_id"][i] = cr_ids[c]
            cols["n_eligible"][i] = len(elig)
            cols["selection_prob"][i] = w[k] / w.sum() / len(cr_ids)
            cols["p_click_pred"][i] = p_click
            cols["advertiser_value"][i] = a
            cols["media_value"][i] = vm
            cols["explore_mult"][i] = mult
            if vm < row.bidfloor:
                out["reason"][i] = "below_floor"
                continue
            bid = max(bid, row.bidfloor) if row.bidfloor > 0 else bid
            cols["bid"][i] = round(bid, 4)
            if bucket[i] < self.holdout_buckets:
                out["action"][i] = "ghost"
                out["reason"][i] = "holdout"
            else:
                out["action"][i] = "bid"
        dec = pd.DataFrame({"request_id": r.request_id.to_numpy(),
                            "ts_ms": r.ts_ms.to_numpy(), **out, **cols})
        dec["holdout"] = bucket < self.holdout_buckets
        dec["explored"] = dec.explore_mult.fillna(1.0) != 1.0
        dec["policy_version"] = self.version
        for c in ("campaign_id", "creative_id", "n_eligible"):
            dec[c] = dec[c].astype("Int64")
        return dec
