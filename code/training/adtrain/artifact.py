"""The model artifact the Go bidder loads (Chapter 9).

    <models>/click/<version>/
        model.json     how to build tokens, the bias, the downsampling rate
        weights.bin    nonzero weights: sorted (uint32 bucket, float32) pairs
        golden.jsonl   examples with this code's predictions, for parity
        report.json    metrics, curves, readouts, and checks, for the console

A version is written to a temporary directory and renamed into place, so
the bidder never sees half an artifact.
"""
from __future__ import annotations

import json
import math
import shutil
import struct
from pathlib import Path

import numpy as np

from .tokens import Spec

FAMILY = "click"


class Served:
    """A model read back from its artifact, predicting as the bidder does:
    float32 weights, summed with the bias, then the downsampling fix."""

    def __init__(self, path: str | Path):
        path = Path(path)
        self.meta = json.loads((path / "model.json").read_text())
        self.spec = Spec.from_json(self.meta["spec"])
        self.bias = float(self.meta["bias"])
        self.rate = float(self.meta.get("downsample_rate", 1.0))
        raw = (path / "weights.bin").read_bytes()
        self.weights = {b: w for b, w in struct.iter_unpack("<If", raw)}
        self.version = self.meta["version"]

    def predict(self, fv: dict, ad: dict) -> float:
        z = self.bias
        for b in self.spec.buckets(self.spec.tokens(fv, ad)):
            z += self.weights.get(b, 0.0)
        p = 1.0 / (1.0 + math.exp(-z))
        w = self.rate
        if 0 < w < 1:
            p = p / (p + (1 - p) / w)
        return p


def root(models_dir) -> Path:
    return Path(models_dir) / FAMILY


def versions(models_dir) -> list[str]:
    r = root(models_dir)
    if not r.exists():
        return []
    return sorted(d.name for d in r.iterdir()
                  if d.is_dir() and (d / "model.json").exists()
                  and not d.name.startswith("."))


def next_version(models_dir) -> str:
    nums = [int(v.split("-")[1]) for v in versions(models_dir)
            if v.startswith(f"{FAMILY}-") and v.split("-")[1].isdigit()]
    return f"{FAMILY}-{max(nums, default=0) + 1:04d}"


def deployed(models_dir) -> str | None:
    f = root(models_dir) / "DEPLOYED"
    if f.exists():
        v = f.read_text().strip()
        return v or None
    return None


def write(models_dir, version: str, meta: dict, weights: dict) -> Path:
    """Write model.json and weights.bin to a temporary directory."""
    tmp = root(models_dir) / f".{version}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    items = sorted(weights.items())
    meta = dict(meta, version=version, nonzero_weights=len(items))
    (tmp / "model.json").write_text(json.dumps(meta, indent=2) + "\n")
    with open(tmp / "weights.bin", "wb") as f:
        for b, w in items:
            f.write(struct.pack("<If", b, np.float32(w)))
    return tmp


def finish(tmp: Path, version: str) -> Path:
    final = tmp.parent / version
    tmp.rename(final)
    return final


def write_json(path: Path, obj) -> None:
    """Write JSON that Go and browsers can read: NaN and infinities,
    which JSON has no words for, become null."""
    path.write_text(json.dumps(clean(obj), indent=1, allow_nan=False) + "\n")


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        o = float(o)
        return o if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o
