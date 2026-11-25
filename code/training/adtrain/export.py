"""Exporting the deep conversion model for the bidder, with parity checks
(Chapter 28).

Trains Chapter 26's DCNv2 on the Part IV dataset and exports it twice:

- `dcn.onnx`, for a bidder that serves with ONNX Runtime, checked against
  PyTorch with onnxruntime, when it's installed;
- `dcn.json`, for the reference bidder, which has no dependencies and no
  cgo, so it runs the network in pure Go, as it runs LightGBM. The file has
  the vocabularies, the numeric transforms, the weights, and golden vectors:
  raw inputs, edge cases included, with PyTorch's outputs, which the Go
  loader checks before the model can load.

It also copies Chapter 27's app vectors and one day of device states, with
their version, for the bidder's dot products and its version check, and a
sample of requests and candidates for the Go benchmark:

    python -m adtrain export --snapshot out/snapshot --out out/export

The Go side: `docker compose run --rm --no-deps --entrypoint deepcheck bidder -dir /out/export`.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, deep

GOLDEN = 40  # test rows in the golden vectors, plus the edge cases
TOL = 1e-4  # the largest difference in a logit the bidder accepts


def _raw(imp: pd.DataFrame, i: int) -> dict:
    """One impression as the bidder sees it: strings and raw numbers."""
    r = imp.iloc[i]
    return {"ids": {f: str(r[f]) for f in deep.FIELDS},
            "apps": {f: str(r[f]) for f in deep.APP_FIELDS},
            "recent_apps": [str(a) for a in r.recent_apps],
            "numeric": {k: (None if pd.isna(r[k]) else float(r[k])) for k in deep.NUMERIC},
            "hour_of_day": None if pd.isna(r.hour_of_day) else float(r.hour_of_day)}


def _frame(raws: list[dict]) -> pd.DataFrame:
    rows = []
    for x in raws:
        d = {**x["ids"], **x["apps"], **x["numeric"], "recent_apps": x["recent_apps"],
             "hour_of_day": x["hour_of_day"]}
        rows.append(d)
    return pd.DataFrame(rows)


def edge_cases(imp, rng) -> list[dict]:
    """Inputs the bidder must handle exactly as training does: unseen IDs,
    no recent apps, the most recent apps, and missing numbers."""
    te = np.flatnonzero(imp.split.to_numpy() == "test")
    out = []
    a = _raw(imp, int(rng.choice(te)))
    a["ids"] = {f: "never-seen-" + f for f in deep.FIELDS}
    a["apps"] = {f: "com.never.seen" for f in deep.APP_FIELDS}
    out.append(a)
    b = _raw(imp, int(rng.choice(te)))
    b["recent_apps"] = []
    out.append(b)
    n = imp.recent_apps.map(len).to_numpy()
    c = _raw(imp, int(np.argmax(n)))
    out.append(c)
    d = _raw(imp, int(rng.choice(te)))
    d["numeric"] = {k: None for k in deep.NUMERIC}
    d["hour_of_day"] = None
    out.append(d)
    e = _raw(imp, int(rng.choice(te)))
    e["recent_apps"] = e["recent_apps"] + ["com.never.seen"]
    out.append(e)
    return out


def _logits(model, enc_fn, raws):
    """PyTorch's logits for raw inputs, through the same encoding as
    training."""
    enc = enc_fn(_frame(raws))
    out = deep.predict(model, enc, np.arange(len(raws)), logits=True)
    return np.column_stack([out[k] for k in deep.HEADS])


def save_dcn(path, model, vocabs, mu, sd, max_recent, golden_raws, golden_out, extra=None) -> str:
    """The bundle the Go bidder loads: vocabularies, transforms, weights,
    and golden vectors. Returns the version, a hash of the weights."""
    sdict = {k: v.detach().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    h = hashlib.sha256()
    for k in sorted(sdict):
        h.update(k.encode())
        h.update(sdict[k].tobytes())
    version = "dcn-" + h.hexdigest()[:10]

    def mat(k):
        a = sdict[k]
        return {"rows": int(a.shape[0]), "cols": int(a.shape[1]) if a.ndim > 1 else 1,
                "data": [float(x) for x in a.ravel()]}

    spec = {
        "format": "adtrain-dcnv2-1",
        "version": version,
        "adtrain_version": __version__,
        "dim": int(sdict["app.weight"].shape[1]),
        "fields": list(deep.FIELDS),
        "app_fields": list(deep.APP_FIELDS),
        "numeric": list(deep.NUMERIC),
        "numeric_transform": "log1p(max(x, 0)), missing 0; then sin and cos of 2 pi hour/24; then (x - mean) / sd",
        "numeric_mean": [float(x) for x in mu],
        "numeric_sd": [float(x) for x in sd],
        "max_recent_apps": int(max_recent),
        "pad": deep.PAD, "oov": deep.OOV,
        "vocabs": {f: vocabs[f] for f in list(deep.FIELDS) + ["app"]},
        "heads": list(deep.HEADS),
        "embeddings": {f: mat(f"emb.{f}.weight") for f in deep.FIELDS},
        "app_embedding": mat("app.weight"),
        "cross": [{"w": mat(f"cross.{i}.weight"), "b": mat(f"cross.{i}.bias")}
                  for i in range(sum(1 for k in sdict if k.startswith("cross.") and k.endswith(".weight")))],
        "mlp": [{"w": mat(f"mlp.{2 * i}.weight"), "b": mat(f"mlp.{2 * i}.bias")}
                for i in range(sum(1 for k in sdict if k.startswith("mlp.") and k.endswith(".weight")))],
        "head_weights": {k: {"w": mat(f"heads.{k}.weight"), "b": mat(f"heads.{k}.bias")} for k in deep.HEADS},
        "golden": [{"input": r, "logits": [float(x) for x in o]} for r, o in zip(golden_raws, golden_out)],
        "tolerance": TOL,
        **(extra or {}),
    }
    Path(path).write_text(json.dumps(spec, separators=(",", ":")))
    return version


def _onnx(model, path, enc, idx, log):
    """Export to ONNX, and compare onnxruntime's logits with PyTorch's."""
    import torch

    class Flat(torch.nn.Module):  # one tensor per kind of input, for the runtime
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, ids, apps, recent, numeric):
            x = {f: ids[:, i] for i, f in enumerate(deep.FIELDS)}
            x.update({f: apps[:, i] for i, f in enumerate(deep.APP_FIELDS)})
            x["recent_apps"], x["numeric"] = recent, numeric
            out = self.m(x)
            return torch.stack([out[k] for k in deep.HEADS], dim=1)

    flat = Flat(model).eval()
    ids = np.column_stack([enc[f][idx] for f in deep.FIELDS])
    apps = np.column_stack([enc[f][idx] for f in deep.APP_FIELDS])
    args = tuple(torch.from_numpy(a) for a in (ids, apps, enc["recent_apps"][idx], enc["numeric"][idx]))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(flat, args, str(path), dynamo=False, opset_version=17,
                          input_names=["ids", "apps", "recent_apps", "numeric"], output_names=["logits"],
                          dynamic_axes={"ids": {0: "batch"}, "apps": {0: "batch"}, "recent_apps": {0: "batch"},
                                        "numeric": {0: "batch"}, "logits": {0: "batch"}})
    with torch.no_grad():
        ref = flat(*args).numpy()
    rep = {"file": Path(path).name, "bytes": Path(path).stat().st_size, "opset": 17, "rows": int(len(idx))}
    try:
        import onnxruntime as ort
    except ImportError:
        rep["parity"] = "onnxruntime not installed"
        return rep
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"ids": ids, "apps": apps, "recent_apps": enc["recent_apps"][idx],
                          "numeric": enc["numeric"][idx]})[0]
    rep["max_abs_diff"] = float(np.abs(got - ref).max())
    rep["parity"] = "ok" if rep["max_abs_diff"] < TOL else "FAILED"
    # one request at a time, as a bidder calls it
    one = {"ids": ids[:1], "apps": apps[:1], "recent_apps": enc["recent_apps"][idx][:1], "numeric": enc["numeric"][idx][:1]}
    for _ in range(50):
        sess.run(None, one)
    t = []
    for _ in range(500):
        t0 = time.perf_counter()
        sess.run(None, one)
        t.append(time.perf_counter() - t0)
    rep["onnxruntime_us_one_row_p50"] = round(float(np.percentile(t, 50)) * 1e6, 1)
    return rep


def _candidates(imp) -> list[dict]:
    """The distinct campaign and creative combinations in the dataset: the
    candidates a request is scored against in the Go benchmark."""
    cols = ["campaign", "creative", "creative_format", "goal", "adv_bundle"]
    c = imp[cols].astype(str).drop_duplicates().sort_values(cols)
    return c.to_dict("records")


def _vectors(pretrain_dir: Path, imp, out: Path) -> dict:
    """Chapter 27's app vectors and the device states of the newest test
    day, with their version, for the bidder's dot products."""
    av, ds = pretrain_dir / "app_vectors.parquet", pretrain_dir / "device_states.parquet"
    if not (av.exists() and ds.exists()):
        return {"skipped": f"no {av.name} or {ds.name} in {pretrain_dir}: run adtrain pretrain first"}
    a = pd.read_parquet(av)
    s = pd.read_parquet(ds)
    day = int(imp[imp.split == "test"].day.max())
    s = s[s.day == day].dropna()
    dims = [c for c in a.columns if c.startswith("e")]
    hd = [c for c in s.columns if c.startswith("h")]
    (out / "app_vectors.json").write_text(json.dumps({
        "version": str(a.version.iloc[0]), "dim": len(dims), "apps": list(a.app),
        "data": [float(x) for x in a[dims].to_numpy(np.float32).ravel()]}, separators=(",", ":")))
    (out / "device_states.json").write_text(json.dumps({
        "version": str(s.version.iloc[0]), "day": day, "dim": len(hd), "devices": list(s.device),
        "data": [float(x) for x in s[hd].to_numpy(np.float32).ravel()]}, separators=(",", ":")))
    return {"version": str(a.version.iloc[0]), "apps": len(a), "device_states": len(s), "day": day}


def run(snapshot_dir="out/snapshot", out_dir="out/export", pretrain_dir="out/pretrain", log=print) -> dict:
    from . import snapshot
    s = snapshot.load(snapshot_dir)
    imp = s["impressions"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vocabs = deep.build_vocabs(imp)
    enc = deep.encode(imp, vocabs)
    max_recent = enc["recent_apps"].shape[1]
    tr = imp.split.to_numpy() == "train"
    mu, sd = enc["numeric"][tr].mean(0), enc["numeric"][tr].std(0) + 1e-6  # as train_dcn standardizes
    t0 = time.time()
    model, enc_std, _, _ = deep.train_dcn(imp, enc, vocabs, log=lambda *_: None)
    secs = time.time() - t0

    def enc_fn(df):
        e = deep.encode(df, vocabs)
        r = np.zeros((len(df), max_recent), np.int64)  # the training width; longer lists are cut
        w = min(max_recent, e["recent_apps"].shape[1])
        r[:, :w] = e["recent_apps"][:, :w]
        e["recent_apps"] = r
        e["numeric"] = ((e["numeric"] - mu) / sd).astype(np.float32)
        return e

    rng = np.random.default_rng(deep.SEED)
    te = np.flatnonzero(imp.split.to_numpy() == "test")
    raws = [_raw(imp, int(i)) for i in rng.choice(te, GOLDEN, replace=False)] + edge_cases(imp, rng)
    gold = _logits(model, enc_fn, raws)
    # the golden logits must match training's own encoding of the same rows
    check = deep.predict(model, enc_std, te[:2000], logits=True)
    again = _logits(model, enc_fn, [_raw(imp, int(i)) for i in te[:200]])
    enc_diff = float(np.abs(again - np.column_stack([check[k][:200] for k in deep.HEADS])).max())
    # requests for the Go benchmark: the test day's impressions, as raw inputs
    sample = [_raw(imp, int(i)) for i in rng.choice(te, 500, replace=False)]
    version = save_dcn(out / "dcn.json", model, vocabs, mu, sd, max_recent, raws, gold,
                       {"requests": sample, "candidates": _candidates(imp)})
    rep = {"model": "export", "adtrain_version": __version__, "version": version,
           "train_seconds": round(secs, 1), "params": int(sum(p.numel() for p in model.parameters())),
           "golden": len(raws), "edge_cases": len(raws) - GOLDEN, "encoding_check_max_diff": enc_diff,
           "dcn_json_bytes": (out / "dcn.json").stat().st_size}
    rep["vectors"] = _vectors(Path(pretrain_dir), imp, out)
    try:
        rep["onnx"] = _onnx(model, out / "dcn.onnx", enc_std, te[:2000], log)
    except Exception as e:  # the Go bundle doesn't need it: report and go on
        rep["onnx"] = {"parity": f"the ONNX export failed: {type(e).__name__}: {e}"}
    (out / "export.json").write_text(json.dumps(rep, indent=2))
    rep["out_dir"] = str(out)
    return rep


def print_report(r: dict) -> None:
    o = r["onnx"]
    print(f"Exported the Chapter 26 DCNv2 ({r['params']:,} parameters, trained in {r['train_seconds']} s), "
          f"version {r['version']}, to {r['out_dir']}:")
    print(f"  dcn.json for the Go bidder: {r['dcn_json_bytes'] / 1e6:.1f} MB, with {r['golden']} golden vectors "
          f"({r['edge_cases']} edge cases: unseen IDs, no recent apps, the most recent apps, missing numbers, "
          f"an unseen recent app)")
    print(f"  re-encoding raw inputs against training's encoding: largest logit difference {r['encoding_check_max_diff']:.1e}")
    if "max_abs_diff" in o:
        print(f"  dcn.onnx ({o['bytes'] / 1e3:.0f} kB, opset {o['opset']}): onnxruntime against PyTorch on "
              f"{o['rows']:,} test rows, largest logit difference {o['max_abs_diff']:.1e}: {o['parity']}; "
              f"{o['onnxruntime_us_one_row_p50']} us for one row (p50, one thread)")
    elif "bytes" in o:
        print(f"  dcn.onnx ({o['bytes'] / 1e3:.0f} kB): {o['parity']}")
    else:
        print(f"  dcn.onnx: {o['parity']}")
    v = r["vectors"]
    if "skipped" in v:
        print(f"  vectors: {v['skipped']}")
    else:
        print(f"  app vectors and {v['device_states']:,} device states of day {v['day']}, version {v['version']}")
    print("  next, in Go: docker compose run --rm --no-deps --entrypoint deepcheck bidder -dir /out/export")
