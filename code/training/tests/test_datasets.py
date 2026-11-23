import gzip
import json
from pathlib import Path

from adtrain import datasets

H = 3_600_000
D = 86_400_000


def write(root: Path, stream: str, day: int, rows):
    d = root / stream / f"day={day:04d}"
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "part-0000.jsonl.gz", "at") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def bid(rid, day, ifa, logged_installs=0):
    num = {"bidfloor": 0.5}
    if logged_installs:
        num["device_prior_installs"] = logged_installs
    return {"request_id": rid, "day": day,
            "decision": {"action": "bid", "campaign_id": 1, "p_click": 0.02},
            "request": {"placement": "interstitial", "exchange": "alpha_exchange",
                        "country": "ind", "ifa": ifa},
            "features": {"schema_version": 1, "numeric": num, "strings": {}},
            "ad": {"campaign": "1", "creative": "3", "goal": "cpi"}}


def test_labels_cutoffs_and_as_of_history(tmp_path):
    # device d1: an impression on day 0 that leads to a click and an
    # install, then a second impression on day 4, after the install
    write(tmp_path, "bids", 0, [bid("r1", 0, "d1"), bid("r2", 0, "d2")])
    write(tmp_path, "bids", 4, [bid("r3", 4, "d1", logged_installs=1)])
    write(tmp_path, "notices", 0, [
        {"type": "billing", "request_id": "r1", "ts_ms": 1 * H, "price": 2.0},
        {"type": "billing", "request_id": "r2", "ts_ms": 2 * H, "price": 1.0},
        {"type": "billing", "request_id": "r2", "ts_ms": 2 * H, "duplicate": True}])
    write(tmp_path, "notices", 4, [
        {"type": "billing", "request_id": "r3", "ts_ms": 4 * D + H, "price": 2.0},
        {"type": "win", "request_id": "x", "ts_ms": 4 * D + 20 * H}])  # "now"
    write(tmp_path, "clicks", 0, [{"request_id": "r1", "ts_ms": 1 * H + 60_000,
                                   "event_id": "cr1"}])
    pb = {"postback_id": "p1", "received_ts_ms": 3 * H, "seen_ts_ms": 3 * H, "day": 0,
          "fields": {"postback_id": "p1", "event_name": "install", "touch_type": "click",
                     "click_id": "cr1", "advertising_id": "d1",
                     "event_ts_ms": str(2 * H)}}
    write(tmp_path, "postbacks", 0, [pb, {**pb, "duplicate": True}])

    df, m = datasets.build(tmp_path, log=lambda *_: None)
    r = df.set_index("request_id")
    assert len(df) == 3 and m["counts"]["duplicate_notices"] == 1
    assert r.loc["r1", "click"] == 1 and r.loc["r1", "install_click"] == 1
    assert r.loc["r2", "install"] == 0
    # day 0 rows are final for installs (cutoff 3 days); day 4's isn't
    assert r.loc["r1", "install_final"] and not r.loc["r3", "install_final"]
    # as of each bid: r1 came before the install, r3 after it
    assert r.loc["r1", "device_prior_installs"] == 0
    assert r.loc["r3", "device_prior_installs"] == 1
    checks = {c["name"]: c for c in m["checks"]}
    assert checks["backfill matches logged"]["passed"]

    # joining the device history as of the build leaks the label: r1's own
    # install shows up in its features, and the check against what the
    # bidder logged catches it
    leak, lm = datasets.build(tmp_path, datasets.Settings(leak=True), log=lambda *_: None)
    assert leak.set_index("request_id").loc["r1", "device_prior_installs"] == 1
    v = datasets.write(df, m, tmp_path / "datasets")
    back, bm = datasets.read(v)
    assert len(back) == 3 and bm["version"] == "conversions-0001"
