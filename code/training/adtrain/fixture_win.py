"""A small win model for the Go tests (code/dsp/internal/gbm/testdata).

    python -m adtrain.fixture_win
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import win

OUT = Path(__file__).resolve().parents[2] / "dsp" / "internal" / "gbm" / "testdata" / "win-fixture"


def main(out=OUT, n=6000, seed=0):
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    places = ["banner", "interstitial", "rewarded_video"]
    med = {"banner": 0.8, "interstitial": 3.0, "rewarded_video": 6.0}
    fvs, bids, y = [], [], []
    for _ in range(n):
        p = places[rng.integers(3)]
        floor = round(float(rng.uniform(0, 1)), 2)
        h = med[p] * math.exp(rng.normal(0, 0.5))
        b = med[p] * math.exp(rng.normal(0, 0.7))
        fvs.append(json.dumps({"schema_version": 1, "strings": {"placement": p, "exchange": "alpha_exchange"},
                               "numeric": {"bidfloor": floor, "hour_of_day": int(rng.integers(24))}}))
        bids.append(b)
        y.append(int(b >= max(h, floor)))
    X, vocab = win._matrix(fvs, bids)
    names = list(win.CATEGORICAL) + list(win.NUMERIC)
    mono = [1 if k in win.BID_FEATURES else 0 for k in names]
    ds = lgb.Dataset(X, np.array(y), feature_name=names, categorical_feature=list(win.CATEGORICAL))
    booster = lgb.train({**win.PARAMS, "num_leaves": 7, "monotone_constraints": mono}, ds,
                        num_boost_round=30)
    out.mkdir(parents=True, exist_ok=True)
    spec = {"model": "win", "version": "win-fixture", "categorical": list(win.CATEGORICAL),
            "numeric": list(win.NUMERIC), "zero_if_missing": [], "other": win.OTHER,
            "vocab": vocab, "targets": ["win"], "bid_features": list(win.BID_FEATURES)}
    (out / "model.json").write_text(json.dumps(spec))
    (out / "win.json").write_text(json.dumps(booster.dump_model()))
    with open(out / "golden.jsonl", "w") as f:
        for i in range(60):
            fv = json.loads(fvs[i])
            r = win.bid_inputs(fv, bids[i])
            fv["numeric"].update({k: r[k] for k in win.BID_FEATURES if math.isfinite(r[k])})
            x = np.array([[float(vocab[c].get(r[c], vocab[c][win.OTHER])) for c in win.CATEGORICAL] +
                          [r[c] for c in win.NUMERIC]], float)
            f.write(json.dumps({"features": fv, "ad": {}, "predictions": {"win": float(booster.predict(x)[0])}}) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
