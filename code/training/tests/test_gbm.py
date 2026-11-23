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
