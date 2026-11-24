import json

import numpy as np
import pandas as pd

from adtrain import datasets, gbm


def synthetic(n_days=8, per_day=1500, seed=0):
    """A dataset where rewarded placements click and install more."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(n_days):
        for i in range(per_day):
            rewarded = rng.random() < 0.5
            installs = int(rng.random() < 0.3)
            fv = {"schema_version": 1,
                  "numeric": {"bidfloor": float(rng.uniform(0.5, 5)), "rewarded": int(rewarded)},
                  "strings": {"placement": "rewarded_video" if rewarded else "banner",
                              "country": rng.choice(["usa", "ind"])}}
            ad = {"campaign": str(rng.integers(3)), "goal": "cpi"}
            click = int(rng.random() < (0.08 if rewarded else 0.02))
            ic = int(click and rng.random() < (0.1 + 0.7 * installs))
            iv = int(not click and rng.random() < 0.004)
            rows.append(dict(request_id=f"r{day}-{i}", day=day, imp_ts_ms=day * 86_400_000,
                             exchange="x", placement=fv["strings"]["placement"],
                             country=fv["strings"]["country"], campaign_id=int(ad["campaign"]),
                             goal="cpi", creative_id=0, price_cpm=1.0,
                             logged_p_click=0.03, logged_p_install_click=0.3,
                             logged_model="", features=json.dumps(fv), ad=json.dumps(ad),
                             device_prior_installs=installs, device_prior_purchases=0,
                             logged_device_prior_installs=installs,
                             click=click, install=ic or iv, install_click=ic, install_view=iv,
                             d7_revenue=0.0, click_final=True, install_final=True,
                             d7_final=True, sample_weight=1.0))
    df = pd.DataFrame(rows)
    manifest = {"settings": {"leak": False}, "checks": [], "passed": True}
    return df, manifest


def test_train_publishes_a_bundle(tmp_path):
    df, m = synthetic()
    d = datasets.write(df, m, tmp_path / "datasets")
    out, rep = gbm.train(d, tmp_path / "models", log=lambda *_: None)
    files = {p.name for p in (tmp_path / "models" / "conversion" / rep["version"]).iterdir()}
    assert {"model.json", "click.json", "install_click.json", "install_view.json",
            "golden.jsonl", "report.json"} <= files
    click = rep["models"]["click"]
    assert click["auc"] > 0.6 and 0.7 < click["calibration"] < 1.4
    # five buckets by predicted value, rising with it
    b = click["by_prediction"]
    assert len(b) == 5 and b[0]["predicted"] < b[-1]["predicted"]
    # the device's install history is the strongest signal after a click
    top = rep["models"]["install_click"]["importance"][0]["feature"]
    assert top == "device_prior_installs"
    golden = [json.loads(x) for x in open(f"{out}/golden.jsonl")]
    assert golden and set(golden[0]["predictions"]) == {"click", "install_click", "install_view"}


def test_values_per_install_shrink_toward_overall():
    import pandas as pd
    rows = []
    for i in range(40):  # app 1: 40 installs at $2.00
        rows.append({"install": 1, "d7_final": True, "d7_revenue": 2.0, "ad": '{"adv_app": "1"}'})
    for i in range(2):   # app 2: 2 installs at $20.00
        rows.append({"install": 1, "d7_final": True, "d7_revenue": 20.0, "ad": '{"adv_app": "2"}'})
    v = gbm._values(pd.DataFrame(rows))
    overall = (40 * 2 + 2 * 20) / 42
    assert abs(v["overall"] - overall) < 1e-3
    # two installs barely move app 2 off the overall; forty move app 1 most of the way
    assert v["by_adv_app"]["2"] < overall + 0.2 * (20 - overall)
    assert abs(v["by_adv_app"]["1"] - (80 + 20 * overall) / 60) < 1e-3


def test_value_model_for_ad_roas(tmp_path):
    # Ad ROAS (Chapter 30): installs of app "9" earn D3 ad revenue that
    # depends on the country, which the per-app average can't see
    df, m = synthetic()
    rng = np.random.default_rng(1)
    hc = rng.random(len(df)) < 0.5
    df.loc[hc, "ad"] = '{"campaign": "5", "goal": "AD_ROAS", "adv_app": "9"}'
    df.loc[~hc, "ad"] = '{"campaign": "1", "goal": "cpi", "adv_app": "3"}'
    df.loc[hc, "install"] = (rng.random(hc.sum()) < 0.05).astype(int)
    usa = (df.country == "usa").to_numpy()
    rev = np.where(usa, 0.30, 0.03) * rng.lognormal(0, 0.5, len(df))
    df["d3_revenue"] = np.where(hc & (df.install == 1).to_numpy(), rev, 0.0)
    df["d3_final"] = df.install == 1
    d = datasets.write(df, m, tmp_path / "datasets")
    out, rep = gbm.train(d, tmp_path / "models", log=lambda *_: None)
    v = rep["models"]["value"]
    assert "skipped" not in v, v
    t, base = v["models"]["tweedie"], v["models"]["per_app_average"]
    assert t["corr"] > base["corr"] + 0.3
    assert 0.7 < t["calibration"] < 1.4
    golden = [json.loads(x) for x in open(f"{out}/golden.jsonl")]
    assert "value" in golden[0]["predictions"]
    assert all(g["predictions"]["value"] > 0 for g in golden)  # a log link
    # the per-app table keeps the ad revenue app apart from the purchase app
    vals = json.loads(open(f"{out}/values.json").read())
    assert vals["by_adv_app"]["9"] < 1.0
