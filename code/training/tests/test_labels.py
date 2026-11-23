import gzip
import json
from pathlib import Path

from adtrain import labels

H = 3_600_000
D = 86_400_000


def write(root: Path, stream: str, day: int, rows):
    d = root / stream / f"day={day:04d}"
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "part-0000.jsonl.gz", "at") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def bid(rid, day, placement):
    return {"request_id": rid, "day": day, "decision": {"action": "bid", "campaign_id": 1},
            "request": {"placement": placement, "exchange": "alpha_exchange"}}


def billing(rid, ts):
    return {"type": "billing", "request_id": rid, "ts_ms": ts}


def postback(pid, recv, **f):
    return {"postback_id": pid, "received_ts_ms": recv, "day": recv // D,
            "fields": {"postback_id": pid, **f}}


def test_delays_and_maturity(tmp_path):
    write(tmp_path, "bids", 0, [bid("r1", 0, "banner"), bid("r2", 0, "interstitial"),
                                bid("r3", 0, "interstitial")])
    write(tmp_path, "notices", 0, [billing("r1", 1 * H), billing("r2", 2 * H),
                                   billing("r3", 3 * H)])
    write(tmp_path, "postbacks", 0, [
        # a click-through install an hour after its impression
        postback("p1", 2 * H, event_name="install", touch_type="click", click_id="cr1"),
        # the MMP sends some postbacks twice
        postback("p1", 2 * H + 5000, event_name="install", touch_type="click", click_id="cr1"),
    ])
    write(tmp_path, "postbacks", 1, [
        # a view-through install that arrives the next day
        postback("p2", D + 4 * H, event_name="install", touch_type="view", impression_id="ir2"),
        postback("p3", D + 5 * H, event_name="af_purchase", touch_type="view",
                 impression_id="ir2", revenue_usd="4.99"),
        # not one of our impressions
        postback("p4", D + 6 * H, event_name="install", touch_type="click", click_id="cxx"),
    ])
    r = labels.report(tmp_path)
    assert r["installs"] == 2 and r["purchases"] == 1
    assert r["purchase_revenue"] == 4.99
    d = r["delays"]
    assert d["click"]["n"] == 1 and d["click"]["share_by_hour"][0] == 1.0  # by 1h
    assert d["view"]["n"] == 1 and d["view"]["share_by_hour"][4] == 0.0  # not by 12h
    m = r["maturity"][0]
    assert m["day"] == 0 and m["installs_so_far"] == 2
    assert m["share_by_end_of_day"][:2] == [0.5, 1.0]
    assert m["share_by_end_of_day"][2] is None  # day 2 hasn't happened
    by = {p["placement"]: p for p in r["by_placement"]}
    assert by["interstitial"]["attributed_view"] == 1
    assert by["banner"]["attributed_click"] == 1
    assert "caused" not in by["banner"]  # no simulator truth given
