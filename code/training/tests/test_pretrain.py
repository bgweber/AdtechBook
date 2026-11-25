import numpy as np
import pandas as pd
import pytest

from adtrain import pretrain as P

DAY = P.DAY_MS


def _hist(n_dev=300, seed=0):
    """Fake histories: half the devices like apps a0-a2, half a3-a5. Each
    installs three apps from its side, one a day."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dev):
        side = d % 2
        mine = [f"com.a{i}" for i in range(3 * side, 3 * side + 3)]
        for k, app in enumerate(rng.permutation(mine)):
            rows.append((f"d{d}", -(5 - k) * DAY, "shared_install", app, 1))
            rows.append((f"d{d}", -(5 - k) * DAY + 3600_000, "seen", "com.pub", 1))
    hist = pd.DataFrame(rows, columns=["device", "ts_ms", "kind", "app", "n"])
    apps = pd.DataFrame({"bundle": [f"com.a{i}" for i in range(6)] + ["com.pub"],
                         "genre": ["x"] * 3 + ["y"] * 3 + ["z"]})
    return hist, apps


def test_window_is_before_the_time_and_masks_the_target():
    hist, apps = _hist()
    ev = P.Events(hist, apps)
    ts, k, a, g = ev.window("d0", -3 * DAY)
    assert (ts < -3 * DAY).all() and len(ts) == 4
    w = ev.window("d0", 0, drop_app=a[0])
    assert a[0] not in w[2]
    assert ev.window("d0", -10 * DAY) is None


def test_popularity_rows_are_distributions():
    hist, apps = _hist()
    ev = P.Events(hist, apps)
    inst = hist[hist.kind == "shared_install"]
    pop = P.Popularity(inst, ev.app_id, len(ev.apps) + 2)
    q = np.exp(pop.logq)
    assert np.allclose(q.sum(1), 1) and (q[:, :2] == 0).all()
    assert pop.index(np.array([10 * DAY]))[0] == len(pop.periods) - 1  # after the last period: the latest


def test_pretraining_learns_a_planted_taste():
    pytest.importorskip("torch")
    hist, apps = _hist()
    ev = P.Events(hist, apps)
    inst = hist[hist.kind == "shared_install"]
    pop = P.Popularity(inst, ev.app_id, len(ev.apps) + 2)
    P.EPOCHS, epochs = 30, P.EPOCHS
    try:
        model, info = P.pretrain(ev, hist, 0, pop, warm=False, log=lambda *a: None)
    finally:
        P.EPOCHS = epochs
    assert info["loss"] < info["loss_popularity_only"]
    # a device that installed a0 should have more lift for a1 than for a4
    H, newest = P.states(model, ev, [("d0", -3 * DAY, None), ("d1", -3 * DAY, None)])
    assert (newest < -3 * DAY).all()
    lift = P._lift(model, H)
    a1, a4 = ev.app_id["com.a1"], ev.app_id["com.a4"]
    assert lift[0, a1] > lift[0, a4] and lift[1, a4] > lift[1, a1]
