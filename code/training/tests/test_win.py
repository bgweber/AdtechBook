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
