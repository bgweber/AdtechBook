import numpy as np
import pandas as pd
import pytest

from adtrain import deep, snapshot


def test_split_days():
    s = snapshot.split_days(list(range(14)))
    assert s[13] == "tail" and s[12] == s[11] == "test" and s[10] == s[9] == "valid" and s[0] == "train"


def test_platt_fixes_a_scaled_prediction():
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0.01, 0.2, 20000)
    y = (rng.random(20000) < p_true).astype(float)
    cal = deep._platt(p_true * 2, y)
    assert abs(cal(p_true * 2).mean() / y.mean() - 1) < 0.05


def _fake(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    apps = [f"com.a{i}" for i in range(12)]
    imp = pd.DataFrame({f: rng.choice(["x", "y", "z"], n) for f in deep.FIELDS})
    imp["publisher_app"] = rng.choice(apps, n)
    imp["adv_bundle"] = rng.choice(apps, n)
    imp["recent_apps"] = [list(rng.choice(apps, 3)) for _ in range(n)]
    for k in deep.NUMERIC + ("hour_of_day",):
        imp[k] = rng.integers(0, 5, n).astype(float)
    imp["click"] = (rng.random(n) < 0.1).astype(int)
    imp["install_click"] = imp.click * (rng.random(n) < 0.2)
    imp["install_view"] = (1 - imp.click) * (rng.random(n) < 0.02)
    imp["click_final"] = True
    imp["install_final"] = True
    imp["split"] = np.where(np.arange(n) < 2000, "train", np.where(np.arange(n) < 2500, "valid", "test"))
    return imp


def test_dcn_trains_and_finds_untouched_rows():
    pytest.importorskip("torch")
    imp = _fake()
    imp.loc[imp.split == "test", "campaign"] = "only_in_test"  # an ID training never sees
    vocabs = deep.build_vocabs(imp)
    enc = deep.encode(imp, vocabs)
    model, enc, untouched, _ = deep.train_dcn(imp, enc, vocabs, log=lambda *a: None)
    p = deep.predict(model, enc, np.flatnonzero(imp.split == "test"))
    assert set(p) == set(deep.HEADS) and all(((v > 0) & (v < 1)).all() for v in p.values())
    assert untouched["emb.campaign.weight"][vocabs["campaign"]["only_in_test"]]
    assert not untouched["emb.campaign.weight"][vocabs["campaign"]["x"]]
