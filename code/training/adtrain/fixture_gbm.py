"""A small LightGBM bundle for the Go bidder's tests (Chapter 15).

    python -m adtrain.fixture_gbm code/dsp/internal/gbm/testdata

Trains click and install models on synthetic FeatureVectors that exercise
every path the Go tree walker has to match: category splits, the "other"
code for unseen strings, missing numbers (NaN), and the features that are
zero when missing. The golden examples include strings the vocabulary has
never seen.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

from . import gbm


def synthetic(n: int, rng):
    fvs, ads, y_click, y_inst = [], [], [], []
    for i in range(n):
        pl = rng.choice(["banner", "interstitial", "rewarded_video", "native"])
        cc = rng.choice(["ind", "bra", "usa", "jpn", "deu", "mex"])
        model = f"m{int(rng.integers(0, 40))}"
        num = {"bidfloor": float(rng.gamma(2, 0.3)), "hour_of_day": float(rng.integers(0, 24)),
               "device_known": 1.0}
        if rng.random() < 0.7:
            num["device_requests"] = float(rng.integers(1, 200))
        if rng.random() < 0.5:
            num["our_imps_today"] = float(rng.integers(0, 12))
        if rng.random() < 0.2:
            num["device_prior_installs"] = float(rng.integers(1, 4))
        fv = {"schema_version": 1, "numeric": num,
              "strings": {"exchange": rng.choice(["alpha_exchange", "beta_exchange"]),
                          "placement": pl, "country": cc, "os": "android",
                          "os_major": str(rng.integers(10, 15)), "make": "x",
                          "model": model, "connection_type": "wifi",
                          "publisher_app": f"com.example.app{int(rng.integers(0, 30))}"}}
        camp = int(rng.integers(0, 6))
        ad = {"campaign": str(camp), "creative": str(camp * 3 + int(rng.integers(0, 3))),
              "creative_format": "video" if pl != "banner" else "static",
              "adv_app": str(camp + 10), "goal": "cpi"}
        z = (-3.5 + {"banner": -1.0, "interstitial": 0.3, "rewarded_video": 0.6, "native": 0.0}[pl]
             + (0.8 if cc in ("usa", "jpn") else 0.0)
             + 0.1 * num.get("our_imps_today", 0.0) + 0.5 * num.get("device_prior_installs", 0.0))
        c = int(rng.random() < 1 / (1 + np.exp(-z)))
        inst = int(c and rng.random() < 0.15 + 0.1 * (camp % 3))
        fvs.append(fv); ads.append(ad); y_click.append(c); y_inst.append(inst)
    return fvs, ads, np.array(y_click), np.array(y_inst)


def main(out_dir: str) -> None:
    import lightgbm as lgb
    rng = np.random.default_rng(3)
    fvs, ads, yc, yi = synthetic(6000, rng)
    raws = [gbm.raw_values(f, a) for f, a in zip(fvs, ads)]
    vocab = gbm.build_vocab(raws, min_count=20)
    X = np.array([gbm.encode(r, vocab) for r in raws], float)
    names = list(gbm.CATEGORICAL) + list(gbm.NUMERIC)
    params = {**gbm.PARAMS, "min_data_in_leaf": 20}
    models = {}
    for name, y, rows in (("click", yc, np.arange(len(yc))),
                          ("install_click", yi, np.flatnonzero(yc == 1))):
        ds = lgb.Dataset(X[rows], y[rows], feature_name=names,
                         categorical_feature=list(gbm.CATEGORICAL))
        models[name] = lgb.train(params, ds, num_boost_round=60)
    d = Path(out_dir) / "gbm-fixture"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    spec = {"model": "gbm", "version": "gbm-fixture", "categorical": list(gbm.CATEGORICAL),
            "numeric": list(gbm.NUMERIC), "zero_if_missing": list(gbm.ZERO_IF_MISSING),
            "other": gbm.OTHER, "vocab": vocab, "targets": sorted(models)}
    (d / "model.json").write_text(json.dumps(spec))
    for name, b in models.items():
        (d / f"{name}.json").write_text(json.dumps(b.dump_model()))
    # golden examples, including strings the vocabulary has never seen
    gf, ga, _, _ = synthetic(150, np.random.default_rng(4))
    for i in range(0, 150, 10):
        gf[i]["strings"]["model"] = "never-seen"
        gf[i]["strings"]["country"] = "zzz"
    with open(d / "golden.jsonl", "w") as f:
        for fv, ad in zip(gf, ga):
            x = np.array([gbm.encode(gbm.raw_values(fv, ad), vocab)], float)
            preds = {n: float(b.predict(x)[0]) for n, b in models.items()}
            f.write(json.dumps({"features": fv, "ad": ad, "predictions": preds}) + "\n")
    print(f"wrote {d}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "code/dsp/internal/gbm/testdata")
