import numpy as np
import pandas as pd

from adtrain import creatives as C


def test_fit_logistic_recovers_a_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-(-1 + 1.5 * x)))).astype(float)
    w, H = C.fit_logistic(np.column_stack([np.ones_like(x), x]), y)
    assert abs(w[0] + 1) < 0.1 and abs(w[1] - 1.5) < 0.1
    assert np.all(np.linalg.eigvalsh(H) > 0)


def test_features_and_style_lift():
    cr = pd.DataFrame({"creative_id": [0, 1, 2], "style": ["a", "b", "a"], "format": ["video", "static", "video"]})
    F = C.Features(cr)
    h = C.History(2, F.n_s)
    for _ in range(50):  # device 0 clicks style a, never style b
        h.add(0, 0, 1, 0)
        h.add(0, 1, 0, 0)
    lift = h.lift(np.array([0, 0]))
    assert lift[0, 0] > 0 > lift[0, 1]
    X = F.x(np.array([0, 1]), lift)
    assert X.shape == (2, F.dim) and X[0, 1] == 1 and X[1, 2] == 1
    o = 1 + F.n_c + F.n_s + F.n_f
    assert X[0, o + 0] == lift[0, 0] and X[1, o + 1] == lift[1, 1]
