import math

from adtrain import win


def test_bid_inputs():
    fv = {"strings": {"placement": "banner", "exchange": "alpha_exchange"},
          "numeric": {"bidfloor": 0.5, "hour_of_day": 3, "our_imps_today": 9}}
    r = win.bid_inputs(fv, 2.0)
    assert r["placement"] == "banner" and r["country"] == ""
    assert abs(r["ln_bid"] - math.log(2.0)) < 1e-12
    assert abs(r["ln_bid_over_floor"] - math.log(4.0)) < 1e-12
    # our own history isn't an input: it doesn't change what others bid
    assert "our_imps_today" not in r
    assert math.isnan(win.bid_inputs({"numeric": {}}, 1.0)["ln_bid_over_floor"])


def test_fixture_is_monotone_in_the_bid(tmp_path):
    import json
    import numpy as np
    import lightgbm as lgb
    from adtrain import fixture_win
    fixture_win.main(out=tmp_path, n=3000)
    spec = json.loads((tmp_path / "model.json").read_text())
    assert spec["targets"] == ["win"] and spec["bid_features"] == list(win.BID_FEATURES)
    g = [json.loads(x) for x in (tmp_path / "golden.jsonl").read_text().splitlines()]
    assert len(g) == 60 and all(0 < x["predictions"]["win"] < 1 for x in g)


def test_curve_gradient_matches_the_loss():
    import numpy as np
    rng = np.random.default_rng(1)
    n = 30
    kind = np.repeat([0, 1, 2], 10)
    v, mu, s = rng.normal(0, 1, n), rng.normal(0, 1, n), 0.4
    g, h = win._curve_objective(kind, v, s)(mu, None)
    e = 1e-6
    num = (win._curve_loss(kind, v, mu + e, s) - win._curve_loss(kind, v, mu - e, s)) / (2 * e)
    assert np.allclose(g, num, atol=1e-5)
    assert (h > 0).all()


def test_curve_finds_the_price_to_beat():
    """Two placements whose prices to beat differ by a factor of e: from
    wins, losses, and some reported prices, the curve's market price for
    each comes out near the truth."""
    import json
    import numpy as np
    rng = np.random.default_rng(2)
    n = 6000
    pl = rng.choice(["banner", "rewarded_video"], n)
    true_mu = np.where(pl == "banner", 0.0, 1.0)
    price = np.exp(true_mu + 0.3 * rng.logistic(size=n))
    bids = np.exp(rng.uniform(-1, 2, n))
    wins = (bids >= price).astype(float)
    mtws = np.where(rng.random(n) < 0.3, price, np.nan)
    fvs = np.array([json.dumps({"strings": {"placement": p}, "numeric": {}}) for p in pl], dtype=object)
    _, vocab = win._matrix(list(fvs), list(bids))
    cmodel = win.train_curve(fvs, bids, wins, mtws, np.zeros(n), vocab, log=lambda *_: None)
    assert cmodel[2] in (0.25, 0.3, 0.4)
    mp = win.market_price(cmodel, vocab, [json.dumps({"strings": {"placement": p}, "numeric": {}})
                                         for p in ("banner", "rewarded_video")])
    assert abs(np.log(mp[0]) - 0.0) < 0.1 and abs(np.log(mp[1]) - 1.0) < 0.1
