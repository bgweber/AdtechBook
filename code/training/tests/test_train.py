import json
from pathlib import Path

import mmh3
import numpy as np

from adtrain import artifact, metrics
from adtrain.fixture import write_ledger
from adtrain.ftrl import FTRL
from adtrain.tokens import Spec
from adtrain.train import Params, train

TESTDATA = Path(__file__).parents[2] / "dsp" / "internal" / "model" / "testdata"


def test_hash_matches_go():
    # the same values are in code/dsp/internal/model/model_test.go
    assert mmh3.hash("placement_banner", 0, signed=False) == 3073373577
    assert mmh3.hash("héllo_wörld", 0, signed=False) == 3367439918
    assert mmh3.hash("abc", 42, signed=False) == 1313807976


def test_tokens_rules():
    s = Spec(strings=["placement", "campaign", "country"],
             numeric_buckets={"our_imps_today": [1, 3]},
             sets={"apps_seen": 2}, crosses=[["placement", "campaign"],
                                             ["country", "campaign"]])
    fv = {"strings": {"placement": "banner"}, "numeric": {"our_imps_today": 3.0},
          "sets": {"apps_seen": ["c", "a", "b"]}}
    assert s.tokens(fv, {"campaign": "7"}) == [
        "placement_banner", "campaign_7", "country_null",
        "our_imps_today_b2", "apps_seen_a", "apps_seen_b",
        "placement_banner*campaign_7"]  # no cross with a missing country
    assert s.tokens({"strings": {}}, {})[3] == "our_imps_today_missing"
    assert s.label("our_imps_today_b1") == "our_imps_today 1 to 3"


def test_fixture_tokens_still_match():
    # testdata is what the Go tests check against; if this fails, the
    # Python tokens changed: regenerate with python -m adtrain.fixture
    m = artifact.Served(TESTDATA / "click-fixture")
    for r in json.loads((TESTDATA / "tokens.json").read_text()):
        toks = m.spec.tokens(r["features"], r["ad"])
        assert toks == r["tokens"]
        assert m.spec.buckets(toks) == r["buckets"]


def test_ftrl_learns():
    rng = np.random.default_rng(0)
    m = FTRL(alpha=0.1, l1=0.1, l2=0.1)
    for _ in range(20000):
        a = int(rng.integers(2))
        y = int(rng.random() < (0.2 if a else 0.02))
        m.update([a], y)
    assert abs(m.predict([1]) - 0.2) < 0.03
    assert abs(m.predict([0]) - 0.02) < 0.01


def test_metrics():
    _, auc = metrics.roc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert abs(auc - 0.75) < 1e-9
    _, auc = metrics.roc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    assert abs(auc - 0.5) < 1e-9
    assert abs(metrics.cosine({1: 1.0, 2: 1.0}, {1: 1.0}) - 2 ** -0.5) < 1e-9


def test_pipeline(tmp_path):
    write_ledger(tmp_path / "ledger")
    models = tmp_path / "models"
    p = Params(downsample_rate=0.5, min_eval_clicks=10)
    v1, r1 = train(tmp_path / "ledger", models, p, log=lambda m: None)
    assert v1 == "click-0001"
    assert r1["data"]["split"].startswith("train on days [0]")
    assert r1["metrics"]["auc"] > 0.6
    assert r1["stability"]["previous"] is None
    d = artifact.root(models) / v1
    for f in ("model.json", "weights.bin", "golden.jsonl", "report.json",
              "install_rates.json"):
        assert (d / f).exists()
    # install rates: smoothed toward their parents, campaign 2 highest
    rates = json.loads((d / "install_rates.json").read_text())
    ir = r1["install_rates"]
    assert ir["overall"]["installs"] > 0
    c = rates["campaigns"]
    assert c["2"] > c["0"]
    for key, v in rates["segments"].items():
        assert min(rates["prior"], c[key.split("|")[0]], v) > 0
    # the golden predictions are the served model's
    served = artifact.Served(d)
    g = json.loads((d / "golden.jsonl").read_text().splitlines()[0])
    assert served.predict(g["features"], g["ad"]) == g["p"]
    # report.json is strict JSON, readable by Go and browsers
    json.loads((d / "report.json").read_text(), parse_constant=lambda c: 1 / 0)
    # with nothing deployed, there's nothing to compare against
    v2, r2 = train(tmp_path / "ledger", models, p, log=lambda m: None)
    assert r2["stability"]["previous"] is None
    # once v1 is deployed, a run on the same data is stable against it
    (artifact.root(models) / "DEPLOYED").write_text(v1 + "\n")
    v2, r2 = train(tmp_path / "ledger", models, p, log=lambda m: None)
    assert v2 == "click-0003"
    assert r2["stability"]["previous"] == v1
    assert r2["stability"]["cosine"] > 0.999


def test_click_rates_and_constant_model(tmp_path):
    write_ledger(tmp_path / "ledger")
    models = tmp_path / "models"
    v, r = train(tmp_path / "ledger", models, Params(min_eval_clicks=10),
                 log=lambda m: None)
    cr = json.loads((artifact.root(models) / v / "click_rates.json").read_text())
    # rewarded video clicks more than banners in the synthetic market
    assert cr["placement"]["rewarded_video"] > cr["placement"]["banner"]
    assert all(0 < x < 1 for x in cr["segments"].values())
    # a constant model fails its checks, and predicts the same everywhere
    v2, r2 = train(tmp_path / "ledger", models, Params(constant=0.25),
                   log=lambda m: None)
    assert not r2["passed"]
    served = artifact.Served(artifact.root(models) / v2)
    assert served.weights == {}
    g = [json.loads(x) for x in (artifact.root(models) / v2 / "golden.jsonl")
         .read_text().splitlines()]
    assert all(abs(x["p"] - 0.25) < 1e-9 for x in g)
