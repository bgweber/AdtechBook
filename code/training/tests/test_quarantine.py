from adtrain import datasets, quarantine as qr

from test_datasets import D, H, bid, write


def pb(pid, rid, ifa, recv, match="id_matching"):
    return {"postback_id": pid, "received_ts_ms": recv, "seen_ts_ms": recv, "day": recv // D,
            "fields": {"postback_id": pid, "event_name": "install", "touch_type": "view",
                       "impression_id": "i" + rid, "advertising_id": ifa,
                       "event_ts_ms": str(recv - 60_000), "match_type": match}}


def test_quarantine_skips_marked_postbacks(tmp_path):
    led, dsets = tmp_path / "ledger", tmp_path / "datasets"
    write(led, "bids", 0, [bid("r1", 0, "d1"), bid("r2", 0, "d2")])
    write(led, "notices", 0, [
        {"type": "billing", "request_id": "r1", "ts_ms": 1 * H, "price": 2.0},
        {"type": "billing", "request_id": "r2", "ts_ms": 7 * H, "price": 2.0},
        {"type": "win", "request_id": "x", "ts_ms": 4 * D}])
    write(led, "postbacks", 0, [pb("p1", "r1", "d1", 2 * H),
                                pb("p2", "r2", "d2", 8 * H, "probabilistic")])
    df, m = datasets.build(led, log=lambda *_: None, quarantine=[])
    assert df.set_index("request_id").loc["r2", "install"] == 1
    e = qr.add(dsets, "postbacks", 0, D, {"match_type": "probabilistic"}, "test")
    assert e["id"] == "q-0001" and qr.load(dsets)[0]["where"] == {"match_type": "probabilistic"}
    df, m = datasets.build(led, log=lambda *_: None, quarantine=qr.load(dsets))
    r = df.set_index("request_id")
    assert r.loc["r1", "install"] == 1 and r.loc["r2", "install"] == 0
    assert m["counts"]["quarantined_postbacks"] == 1 and m["quarantine"][0]["skipped"] == 1
    # the device history the bidder recounts skips it too
    h = qr.device_history(led, dsets)
    assert h["devices"] == {"d1": [1, 0]} and h["skipped"] == {"q-0001": 1}
    # a quarantine outside the time range marks nothing
    qr.remove(dsets, "q-0001")
    qr.add(dsets, "postbacks", D, 2 * D, {"match_type": "probabilistic"})
    assert qr.device_history(led, dsets)["devices"]["d2"] == [1, 0]
    ins = qr.inspect(led, 0)
    assert ins["by_day"] == {0: {"id_matching": 1, "probabilistic": 1}}


def test_install_label_rate_check(tmp_path):
    led = tmp_path / "ledger"
    n = 0
    for day in range(5):
        bids, notices, pbs = [], [], []
        for i in range(100):
            n += 1
            rid, ifa = f"r{n}", f"d{n}"
            bids.append(bid(rid, day, ifa))
            notices.append({"type": "billing", "request_id": rid, "ts_ms": day * D + H + i, "price": 1.0})
            if i < (30 if day == 2 else 2):
                pbs.append(pb(f"p{n}", rid, ifa, day * D + 2 * H + i))
        write(led, "bids", day, bids)
        write(led, "notices", day, notices)
        write(led, "postbacks", day, pbs)
    write(led, "notices", 9, [{"type": "win", "request_id": "x", "ts_ms": 9 * D}])
    df, m = datasets.build(led, log=lambda *_: None, quarantine=[])
    c = {x["name"]: x for x in m["checks"]}
    assert not c["install label rate"]["passed"] and "day 2" in c["install label rate"]["detail"]
