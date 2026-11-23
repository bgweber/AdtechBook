"""Export a bundle's model to ONNX, and time ONNX Runtime on it (Chapter 17).

The bidder evaluates LightGBM's trees in Go, with no runtime to install.
The usual alternative is ONNX: convert the model once, and serve it with
ONNX Runtime, which runs any model the converters handle, trees or neural
networks. This writes the converted model and the bundle's golden examples
as arrays, checks that ONNX Runtime agrees with LightGBM on them, and times
it one row at a time and in batches. code/onnxbench times the same file
from Go, through ONNX Runtime's C API without cgo.

    pip install onnx onnxruntime onnxmltools
    python -m adtrain onnx --bundle out/models/conversion/gbm-0012 --out out/onnx
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from . import gbm


def export(bundle_dir: str, out_dir: str, target: str = "click") -> dict:
    try:
        import lightgbm as lgb
        import onnx
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError as e:
        raise SystemExit(f"{e}: pip install onnx onnxruntime onnxmltools")
    spec = json.load(open(os.path.join(bundle_dir, "model.json")))
    booster = lgb.Booster(model_file=os.path.join(bundle_dir, f"{target}.txt"))
    nf = booster.num_feature()
    m = onnxmltools.convert_lightgbm(booster, initial_types=[("x", FloatTensorType([None, nf]))],
                                     zipmap=False, target_opset=15)
    # the converter fixes the label output's batch size at 1; make it any
    d = m.graph.output[0].type.tensor_type.shape.dim[0]
    d.ClearField("dim_value")
    d.dim_param = "N"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{target}.onnx")
    onnx.save(m, path)

    xs, want = [], []
    for line in open(os.path.join(bundle_dir, "golden.jsonl")):
        g = json.loads(line)
        if target not in g["predictions"]:
            continue
        xs.append(gbm.encode(gbm.raw_values(g["features"], g.get("ad") or {}), spec["vocab"]))
        want.append(g["predictions"][target])
    X = np.asarray(xs, dtype=np.float32)
    want = np.asarray(want, dtype=np.float64)
    X.tofile(os.path.join(out_dir, "golden_X.f32"))
    want.tofile(os.path.join(out_dir, "golden_want.f64"))
    out = {"model": path, "rows": len(want), "features": nf,
           "trees": booster.num_trees()}
    try:
        import onnxruntime as ort
    except ImportError:
        return out
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1  # like one bid request
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    p = sess.run(None, {"x": X})[1][:, 1]
    out["max_abs_diff"] = float(np.abs(p - want).max())
    out["onnxruntime"] = ort.__version__
    for batch in (1, 8, 64):
        rows = [X[i:i + batch] for i in range(0, len(X) - batch + 1, batch)]
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < 1.0:
            for r in rows:
                sess.run(None, {"x": r})
                n += len(r)
        out[f"us_per_row_batch_{batch}"] = round(1e6 * (time.perf_counter() - t0) / n, 2)
    return out


def print_report(r: dict) -> None:
    print(f"wrote {r['model']}: {r['trees']} trees, {r['features']} inputs; "
          f"{r['rows']} golden rows as golden_X.f32 and golden_want.f64")
    if "max_abs_diff" not in r:
        print("onnxruntime isn't installed: no parity check or timings")
        return
    print(f"ONNX Runtime {r['onnxruntime']} vs LightGBM on the golden rows: "
          f"largest difference {r['max_abs_diff']:.2g} (float32 inputs)")
    for b in (1, 8, 64):
        print(f"  batch {b:>2}: {r[f'us_per_row_batch_{b}']:.2f} µs per row, one thread, from Python")
