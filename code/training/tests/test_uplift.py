import numpy as np

from adtrain import uplift


def test_qini_perfect_and_random():
    # treated rows install only in the first half; control never
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0] * 50)
    t = np.array([1, 1, 1, 1, 0, 0, 0, 0] * 50)
    curve, coef = uplift.qini(y.astype(float) * t, y, t)
    assert curve[-1] == 1.0 and curve[2] >= 0.99 and coef > 0.3
    _, c0 = uplift.qini(np.random.default_rng(0).random(len(y)), y, t)
    assert abs(c0) < 0.15


def test_ratio_is_caused_per_credited_and_capped():
    r = uplift.ratio(np.array([0.10, 0.10, 0.10]), np.array([0.05, 0.12, 0.0]), np.array([0.05, 0.05, 0.02]))
    assert np.allclose(r, [1.0, 0.0, uplift.R_MAX])
