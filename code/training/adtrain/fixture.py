"""A small synthetic ledger and model, for tests in both languages.

    python -m adtrain.fixture ../dsp/internal/model/testdata

writes a model artifact trained on synthetic data (with negative
downsampling, so the bidder's correction is tested too), plus
tokens.json: the tokens and buckets for a few examples. The Go tests
load both and must reproduce them exactly.
"""
from __future__ import annotations

import gzip
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import artifact
from .train import Params, train

DAY_MS = 86_400_000


def write_ledger(root: Path, days: int = 2, per_day: int = 3000, seed: int = 1):
    """Bids, billing notices, and clicks in the bidder's ledger format."""
    rng = np.random.default_rng(seed)
    apps = [f"com.example.app{i}" for i in range(12)]
    places = ["rewarded_video", "interstitial", "banner", "native"]
    place_eff = {"rewarded_video": 0.8, "interstitial": 0.5, "banner": -1.0,
                 "native": 0.0}
    app_eff = rng.normal(0, 0.5, len(apps))
    for day in range(days):
        bids, notes, clicks, posts = [], [], [], []
        for i in range(per_day):
            rid = f"al{day:03d}{i:09d}"
            ts = day * DAY_MS + int(rng.integers(0, DAY_MS - 4_000_000))
            a = int(rng.integers(len(apps)))
            pl = places[int(rng.integers(len(places)))]
            camp = int(rng.integers(3))
            imps_today = int(rng.poisson(2))
            fv = {"schema_version": 1,
                  "numeric": {"bidfloor": round(float(rng.uniform(0, 5)), 2),
                              "hour_of_day": (ts % DAY_MS) // 3_600_000,
                              "device_known": 1.0,
                              "our_imps_today": float(imps_today)},
                  "strings": {"exchange": "alpha_exchange",
                              "publisher_app": apps[a], "placement": pl,
                              "country": ["ind", "bra"][i % 2],
                              "os": "android"},
                  "sets": {"apps_seen": sorted({apps[int(j)] for j in
                                                rng.integers(0, len(apps), 3)})}}
            if i % 7 == 0:  # some missing values
                del fv["strings"]["country"]
                del fv["numeric"]["our_imps_today"]
            ad = {"campaign": str(camp), "adv_app": str(100 + camp),
                  "goal": "cpc", "creative": str(10 * camp + i % 2),
                  "creative_format": "video", "creative_length": "15",
                  "creative_orientation": "portrait", "creative_style": "x"}
            z = -3.3 + place_eff[pl] + app_eff[a] + 0.3 * camp - 0.1 * imps_today
            y = rng.random() < 1 / (1 + math.exp(-z))
            bids.append({"ts_ms": ts, "day": day, "request_id": rid,
                         "decision": {"action": "bid", "campaign_id": camp,
                                      "p_click": 0.03},
                         "features": fv, "ad": ad})
            notes.append({"event_id": "b" + rid, "type": "billing",
                          "ts_ms": ts + 500, "day": day, "request_id": rid,
                          "campaign_id": camp, "price": 1.0})
            if y:
                clicks.append({"event_id": "c" + rid, "ts_ms": ts + 4000,
                               "day": day, "request_id": rid,
                               "campaign_id": camp})
                if rng.random() < 0.05 * (1 + camp):  # an install, via the MMP
                    posts.append({"postback_id": f"pb{rid}", "received_ts_ms": ts + 900_000,
                                  "day": day, "fields": {
                                      "postback_id": f"pb{rid}", "event_name": "install",
                                      "touch_type": "click", "click_id": "c" + rid,
                                      "campaign_id": str(camp),
                                      "event_ts_ms": str(ts + 600_000)}})
        # a later notice, so the last day's click windows have all closed
        if day == days - 1:
            notes.append({"event_id": "wend", "type": "win",
                          "ts_ms": (day + 1) * DAY_MS + 3_600_000, "day": day,
                          "request_id": "end", "campaign_id": 0})
        for stream, rows in (("bids", bids), ("notices", notes),
                             ("clicks", clicks), ("postbacks", posts)):
            d = root / stream / f"day={day:04d}"
            d.mkdir(parents=True, exist_ok=True)
            with gzip.open(d / "part-0000.jsonl.gz", "wt") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")


def build(out_dir: Path) -> Path:
    """Train the fixture model and write it, with tokens.json, to out_dir."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_ledger(tmp / "ledger")
        version, _ = train(tmp / "ledger", tmp / "models",
                           Params(downsample_rate=0.5, golden_examples=50,
                                  min_eval_clicks=1),
                           log=lambda m: None)
        src = artifact.root(tmp / "models") / version
        dst = Path(out_dir) / "click-fixture"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        (dst / "report.json").unlink()  # large, and not needed by the tests
        served = artifact.Served(dst)
        rows = []
        for line in (dst / "golden.jsonl").read_text().splitlines()[:5]:
            g = json.loads(line)
            toks = served.spec.tokens(g["features"], g["ad"])
            rows.append({"features": g["features"], "ad": g["ad"],
                         "tokens": toks, "buckets": served.spec.buckets(toks)})
        (Path(out_dir) / "tokens.json").write_text(json.dumps(rows, indent=1) + "\n")
        return dst


if __name__ == "__main__":
    print(build(Path(sys.argv[1] if len(sys.argv) > 1 else "testdata")))
