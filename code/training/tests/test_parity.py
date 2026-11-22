import gzip
import json
from pathlib import Path

from adtrain import features_ref, parity

CASES = (Path(__file__).parents[2] / "dsp" / "internal" / "features" /
         "testdata" / "parity_cases.jsonl")


def cases():
    return [json.loads(x) for x in CASES.read_text().splitlines()]


def test_reference_matches_the_bidder():
    # the same file is checked by the Go test in internal/features
    for c in cases():
        n = features_ref.normalize(c["request"], c["exchange"])
        ref = features_ref.build(n, c["profile"], c["ts_ms"])
        assert features_ref.flatten(ref) == features_ref.flatten(c["features"]), \
            c["request"]["id"]


def test_normalize_quirks():
    req = {"id": "r", "imp": [{"video": {}, "ext": {"is_rewarded_inventory": 1},
                               "bidfloor": 1.5}],
           "app": {"bundle": " Com.Example.Game$hl=en "},
           "device": {"geo": {"country": "br"}, "ifa": "0" * 8 + "-0000-0000-0000-" + "0" * 12}}
    n = features_ref.normalize(req, "beta_exchange")
    assert n["placement"] == "rewarded_video" and n["rewarded"]
    assert n["bundle"] == "com.example.game"
    assert n["country"] == "BRA"
    assert not n["valid_ifa"]


def write_ledger(root: Path, rows):
    for stream, recs in (("raw", [r for r, _ in rows]), ("bids", [b for _, b in rows])):
        d = root / stream / "day=0000"
        d.mkdir(parents=True, exist_ok=True)
        with gzip.open(d / "part-0000.jsonl.gz", "wt") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")


def test_check_flags_a_skew(tmp_path):
    rows = []
    for i, c in enumerate(cases()):
        rid = c["request"]["id"]
        fv = json.loads(json.dumps(c["features"]))
        if i == 0:  # the bidder changed how it normalizes one feature
            fv["strings"]["os_major"] = fv["strings"].get("os_major", "") + ".2"
        rows.append(({"request_id": rid, "ts_ms": c["ts_ms"], "day": 0,
                      "exchange": c["exchange"], "request": c["request"],
                      "profile": c["profile"]},
                     {"request_id": rid, "decision": {"action": "bid"}, "features": fv}))
    write_ledger(tmp_path, rows)
    r = parity.check(tmp_path)
    assert r["rows"] == len(rows) and r["mismatched_rows"] == 1 and not r["passed"]
    top = r["features"][0]
    assert top["feature"] == "os_major" and top["mismatched"] == 1
    assert top["examples"][0]["logged"].endswith(".2")
