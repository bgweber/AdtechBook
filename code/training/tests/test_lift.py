from adtrain import lift


def test_estimate_and_interval():
    # 1,000 treated devices install at 5%, 200 holdout devices at 1%; 80%
    # of treated devices saw an ad
    r = {"treatment": [1000, 50], "control": [200, 2], "exposed": 800}
    e = lift._estimate("app", r, spend=100.0, attributed=40, caused=35)
    assert abs(e["itt"] - 0.04) < 1e-9
    assert abs(e["lift_per_exposed"] - 0.05) < 1e-9
    assert abs(e["incremental_installs"] - 40) < 1e-6
    assert e["incrementality"] == 1.0 and e["icpi"] == 2.5
    lo, hi = e["incremental_ci"]
    assert lo < 40 < hi
    assert e["true_incrementality"] == 0.875


def test_empty_holdout_is_not_certain():
    # no holdout installs among 12 devices: the interval must be wide
    small = lift._estimate("a", {"treatment": [1000, 50], "control": [12, 0], "exposed": 1000}, 10, 40, None)
    big = lift._estimate("a", {"treatment": [1000, 50], "control": [400, 0], "exposed": 1000}, 10, 40, None)
    w = lambda e: e["incremental_ci"][1] - e["incremental_ci"][0]
    assert w(small) > 3 * w(big)
    assert small["incremental_ci"][0] < 0
