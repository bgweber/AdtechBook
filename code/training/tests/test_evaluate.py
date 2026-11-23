import numpy as np

from adtrain import datasets, evaluate, gbm

from test_gbm import synthetic


def test_pav_keeps_the_mean_and_order():
    y, w = [0.1, 0.3, 0.2, 0.5, 0.4], [10, 10, 30, 10, 10]
    f = evaluate._pav(y, w)
    assert np.all(np.diff(f) >= 0)
    assert abs(np.dot(f, w) - np.dot(y, w)) < 1e-12


def test_target_probabilities():
    d = {"campaign_id": 5, "eligible": [3, 5, 7], "eligible_goals": ["CPI", "CPC", "CPI"],
         "n_creatives": 2}
    assert evaluate._target_prob("uniform", d) == 1 / 3 / 2
    assert evaluate._target_prob("cpi-first", d) == 1 / 7 / 2
    assert evaluate._target_prob("uniform", {"campaign_id": 1}) is None


def test_price_matches_the_bidder():
    row = {"goal": "CPI", "p_click": 0.02, "p_install": 0.1, "value": 2 * 0.02 * 0.1 * 1000,
           "floor": 0.1, "explore": 1.0}
    # doubling the install rate doubles the value; bid = value x 0.8 x 0.7
    bid = evaluate._price([row], np.array([0.02]), np.array([0.2]))[0]
    assert abs(bid - round(8.0 * 0.8 * 0.7, 4)) < 1e-9
    # below the floor: no bid
    assert evaluate._price([{**row, "floor": 100.0}], np.array([0.02]), np.array([0.2]))[0] == 0


def test_dump_evaluator_matches_lightgbm(tmp_path):
    df, m = synthetic(n_days=4, per_day=800)
    d = datasets.write(df, m, tmp_path / "datasets")
    out, _ = gbm.train(d, tmp_path / "models", log=lambda *_: None)
    raws = gbm.dataset_raws(df.iloc[:500])
    want = gbm.score(out, df.iloc[:500], raws, "click")["click"]
    import json
    from pathlib import Path
    spec = json.loads((Path(out) / "model.json").read_text())
    X = np.array([gbm.encode(r, spec["vocab"]) for r in raws], float)
    got = evaluate.DumpModel(Path(out) / "click.json").predict(X)
    assert np.max(np.abs(got - want)) < 1e-12
