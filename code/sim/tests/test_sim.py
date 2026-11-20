"""Tests for the simulated market. Run with: pytest code/sim"""
import numpy as np
import pandas as pd
import pytest

from adsim import preset, simulate
from adsim.policy import holdout_bucket, normalize
from adsim.openrtb import to_openrtb


@pytest.fixture(scope="module")
def res():
    return simulate(preset("crawl", "tiny", seed=11))


def test_deterministic():
    a = simulate(preset("crawl", "tiny", seed=3))
    b = simulate(preset("crawl", "tiny", seed=3))
    for name in ("bid_log", "impressions", "postbacks"):
        pd.testing.assert_frame_equal(a.logs[name], b.logs[name])


def test_funnel_is_nested(res):
    bl, imp, cl = (res.logs[k] for k in ("bid_log", "impressions",
                                         "clicks"))
    auc = res.truth["auctions"]
    bids = set(bl.request_id[bl.action == "bid"])
    assert set(auc.request_id) <= bids
    wins = set(auc.request_id[auc.win])
    assert set(imp.request_id) <= wins
    assert set(cl.impression_id) <= set(imp.impression_id)


def test_ghost_bids_never_reach_auctions(res):
    bl = res.logs["bid_log"]
    ghosts = set(bl.request_id[bl.action == "ghost"])
    assert ghosts.isdisjoint(set(res.truth["auctions"].request_id))
    assert bl.loc[bl.action == "ghost", "holdout"].all()


def test_first_price(res):
    auc = res.truth["auctions"]
    n = res.logs["notifications"]
    wins = n[n.type == "win"].drop_duplicates("event_id")
    m = wins.merge(auc, on="request_id")
    assert np.allclose(m.price_x if "price_x" in m else m.price, m.bid)
    # a win means our bid beat the highest other bid
    w = auc[auc.win]
    assert (w.bid >= w.true_highest_other).all()


def test_holdout_share_and_stability():
    ifa = np.array([f"dev-{i}" for i in range(20000)])
    b1 = holdout_bucket(ifa, "s")
    b2 = holdout_bucket(ifa, "s")
    assert (b1 == b2).all()
    assert abs((b1 < 50).mean() - 0.05) < 0.01
    assert (holdout_bucket(ifa, "other") != b1).mean() > 0.9


def test_click_rate_matches_truth(res):
    imp = res.logs["impressions"]
    ti = res.truth["impressions"]
    clicked = imp.impression_id.isin(res.logs["clicks"].impression_id)
    p = ti.set_index("impression_id").loc[imp.impression_id].p_click
    n = len(p)
    se = np.sqrt((p * (1 - p)).sum()) / n
    assert abs(clicked.mean() - p.mean()) < 4 * se


def test_attribution_windows(res):
    ins = res.truth["installs"]
    ours = ins[ins.credited_to == "us"]
    assert (ours.touch_type.isin(["click", "view"])).all()
    pb = res.logs["postbacks"]
    assert set(pb.event_name) <= {"install", "af_purchase", "af_d1_open"}
    # postbacks arrive after the event
    assert (pb.received_ts_ms >= pb.event_ts_ms).all()


def test_truth_not_in_logs(res):
    for name, df in res.logs.items():
        assert not [c for c in df.columns if c.startswith("true_")], name


def test_normalize_country_and_bundle():
    req = pd.DataFrame({"country": ["IN", "IND"],
                        "bundle": ["com.a.b$hl=en", "Com.A.B "],
                        "ifa": ["", "00000000-0000-0000-0000-000000000000"]})
    out = normalize(req)
    assert list(out.country3) == ["IND", "IND"]
    assert list(out.bundle_norm) == ["com.a.b", "com.a.b"]
    assert not out.valid_ifa.any()


def test_openrtb_roundtrip(res):
    row = next(res.logs["bid_requests"].itertuples())
    r = to_openrtb(row)
    assert r["id"] == row.request_id and r["imp"][0]["id"] == "1"
    assert r["device"]["os"] == "android"
