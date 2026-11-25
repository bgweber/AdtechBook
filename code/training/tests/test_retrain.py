import numpy as np
import pandas as pd

from adtrain import retrain as R

H = R.HOUR


def test_delay_cdf_uses_installs_reported_before_the_time():
    imp = pd.DataFrame({"install": [1, 1, 1, 0, 1, 1],
                        "imp_ts_ms": [0, 0, 0, 0, 10 * H, 80 * H],
                        "install_received_ts_ms": [1 * H, 2 * H, 4 * H, -1, 20 * H, 81 * H]})
    # the impressions at 10 h and 80 h are too recent for their delays to be complete
    F, d = R.delay_cdf(imp, before_ms=81 * H)
    assert len(d) == 3
    assert F(0) == 0 and F(1 * H) == 1 / 3 and F(3 * H) == 2 / 3 and F(10 * H) == 1


def test_delay_weights_remove_the_bias_of_unreported_installs():
    # one feature, a constant install rate, and delays: the naive model
    # underpredicts recent impressions, the weighted one doesn't
    rng = np.random.default_rng(0)
    n, p = 400_000, 0.03
    t = rng.uniform(0, 10 * 24 * H, n)
    inst = rng.random(n) < p
    recv = t + rng.exponential(6 * H, n)
    T = 10 * 24 * H
    imp = pd.DataFrame({"install": inst.astype(int), "imp_ts_ms": t, "install_received_ts_ms": np.where(inst, recv, -1)})
    F, _ = R.delay_cdf(imp, T)
    y = (inst & (recv <= T)).astype(float)
    recent = t > T - 6 * H
    naive = y[recent].mean()
    w = np.where(y > 0, np.minimum(1 / np.maximum(F(T - t), 1e-9), R.MAX_WEIGHT), 1.0)
    weighted = (w * y)[recent].sum() / w[recent].sum()
    assert naive < 0.7 * p
    assert abs(weighted / p - 1) < 0.15
