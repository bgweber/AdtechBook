import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from adtrain import deep, export  # noqa: E402

GO_TESTDATA = Path(__file__).resolve().parents[2] / "dsp" / "internal" / "deep" / "testdata" / "dcn_tiny.json"


def tiny():
    """A tiny DCNv2 with random weights, a few rows of raw inputs, and their
    PyTorch logits: the Go loader's test bundle."""
    dims = deep.DIM, deep.HIDDEN
    deep.DIM, deep.HIDDEN = 2, (4, 3)
    try:
        return _tiny()
    finally:
        deep.DIM, deep.HIDDEN = dims


def _tiny():
    rng = np.random.default_rng(3)
    n = 12
    imp = pd.DataFrame({f: rng.choice(["a", "b", "c"], n) for f in deep.FIELDS})
    imp["publisher_app"] = rng.choice(["com.p1", "com.p2"], n)
    imp["adv_bundle"] = rng.choice(["com.a1", "com.a2", "com.a3"], n)
    imp["recent_apps"] = [list(rng.choice(["com.p1", "com.a1", "com.a2"], k)) for k in rng.integers(0, 4, n)]
    for k in deep.NUMERIC:
        imp[k] = rng.uniform(0, 50, n)
    imp.loc[0, "bidfloor"] = np.nan
    imp["hour_of_day"] = rng.integers(0, 24, n).astype(float)
    vocabs = deep.build_vocabs(imp)
    enc = deep.encode(imp, vocabs)
    mu, sd = enc["numeric"].mean(0), enc["numeric"].std(0) + 1e-6
    torch.manual_seed(0)
    model = deep._model(vocabs, enc["numeric"].shape[1])
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.3)
    max_recent = enc["recent_apps"].shape[1]
    raws = [export._raw(imp, i) for i in range(n)]
    raws[1]["ids"] = {f: "unseen" for f in deep.FIELDS}
    raws[2]["recent_apps"] = []

    def enc_fn(df):
        e = deep.encode(df, vocabs)
        r = np.zeros((len(df), max_recent), np.int64)
        w = min(max_recent, e["recent_apps"].shape[1])
        r[:, :w] = e["recent_apps"][:, :w]
        e["recent_apps"] = r
        e["numeric"] = ((e["numeric"] - mu) / sd).astype(np.float32)
        return e

    gold = export._logits(model, enc_fn, raws)
    cands = export._candidates(imp)
    return model, vocabs, mu, sd, max_recent, raws, gold, cands


def test_save_dcn_writes_a_bundle_the_go_bidder_can_check(tmp_path):
    model, vocabs, mu, sd, L, raws, gold, cands = tiny()
    v = export.save_dcn(tmp_path / "dcn.json", model, vocabs, mu, sd, L, raws, gold,
                        {"requests": raws, "candidates": cands})
    spec = json.loads((tmp_path / "dcn.json").read_text())
    assert v.startswith("dcn-") and spec["version"] == v
    assert len(spec["golden"]) == len(raws) and len(spec["golden"][0]["logits"]) == 3
    d = 2 * (len(deep.FIELDS) + len(deep.APP_FIELDS) + 1) + len(deep.NUMERIC) + 2
    assert spec["cross"][0]["w"]["rows"] == d and spec["cross"][0]["w"]["cols"] == d
    assert len(spec["numeric_mean"]) == len(deep.NUMERIC) + 2


def test_go_testdata_is_current():
    """The Go tests load testdata/dcn_tiny.json; regenerate it with
    python -m tests.test_export when the export format changes."""
    spec = json.loads(GO_TESTDATA.read_text())
    assert spec["format"] == "adtrain-dcnv2-1"
    assert len(spec["golden"]) == 12


if __name__ == "__main__":  # regenerate the Go test bundle
    model, vocabs, mu, sd, L, raws, gold, cands = tiny()
    GO_TESTDATA.parent.mkdir(parents=True, exist_ok=True)
    export.save_dcn(GO_TESTDATA, model, vocabs, mu, sd, L, raws, gold, {"requests": raws, "candidates": cands})
    print("wrote", GO_TESTDATA)
