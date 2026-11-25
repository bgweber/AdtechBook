"""The console's Deep Models tab (Part IV): what's in the output folder,
each chapter's latest report, jobs that run the chapters' commands, and the
embedding viewer's data.

The training service serves these to the bidder's console:

    POST /deep/status       the datasets, the reports, and the running job
    POST /deep/run          {"job": "deep" | "pretrain" | "export" | "retrain" | "select" | "creatives"}
    POST /deep/embeddings   {"source": "dcn" | "transformer" | "coinstall" | "devices"}

The commands write the same files as on the command line, so a chapter run
from a terminal shows up here too.
"""
from __future__ import annotations

import io
import json
import threading
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

JOBS = {
    "deep": ("Chapters 25 and 26: LightGBM against a DCNv2", "snapshot"),
    "pretrain": ("Chapter 27: pretraining on install histories", "snapshot"),
    "export": ("Chapter 28: export for the bidder", "snapshot"),
    "retrain": ("Chapter 29: retraining through a market shift", "snapshot-shift"),
    "select": ("Chapter 35: learned campaign selection", "snapshot"),
    "creatives": ("Chapter 36: personalized creative selection", "snapshot"),
}
SOURCES = {
    "dcn": "DCNv2 app embeddings (Chapter 26), learned from attributed impressions",
    "transformer": "The transformer's app table (Chapter 27), pretrained on install histories",
    "coinstall": "App vectors from which devices install apps together (PPMI and SVD)",
    "devices": "Device states (Chapter 27), the transformer's summary of each device's history",
}
K = 5  # nearest neighbors listed


class Jobs:
    """One job at a time, in a thread, with its printed report kept."""

    def __init__(self, root: Path):
        self.root = root
        self.mu = threading.Lock()
        self.job: dict = {"state": "idle"}

    def status(self) -> dict:
        with self.mu:
            return dict(self.job)

    def start(self, name: str) -> dict:
        if name not in JOBS:
            raise ValueError(f"unknown job {name!r}")
        snap = self.root / JOBS[name][1]
        if not (snap / "manifest.json").exists():
            raise ValueError(f"no dataset in {snap}: download it from the releases and unzip it into "
                             f"the out folder, or build it (WALKTHROUGH.md)")
        with self.mu:
            if self.job.get("state") == "running":
                raise ValueError(f"{self.job['name']} is already running")
            self.job = {"state": "running", "name": name, "title": JOBS[name][0],
                        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        threading.Thread(target=self._run, args=(name, snap), daemon=True).start()
        return self.status()

    def _run(self, name: str, snap: Path) -> None:
        t0, buf = time.time(), io.StringIO()
        try:
            with redirect_stdout(buf):
                run(name, snap, self.root)
            state, err = "done", ""
        except Exception as e:
            traceback.print_exc()
            state, err = "failed", f"{type(e).__name__}: {e}"
        with self.mu:
            self.job.update(state=state, error=err, seconds=round(time.time() - t0, 1),
                            output=buf.getvalue()[-20000:])


def run(name: str, snap: Path, root: Path) -> None:
    """One chapter's command, as on the command line, printing its report."""
    quiet = lambda *_: None
    if name == "deep":
        from . import deep
        deep.print_report(deep.run(str(snap), log=quiet, out_dir=root / "deep"))
    elif name == "pretrain":
        from . import pretrain
        pretrain.print_report(pretrain.run(str(snap), str(root / "pretrain"), log=quiet))
    elif name == "export":
        from . import export
        export.print_report(export.run(str(snap), str(root / "export"), str(root / "pretrain"), log=quiet))
    elif name == "retrain":
        from . import retrain
        retrain.print_report(retrain.run(str(snap), str(root / "retrain"), log=quiet))
    elif name == "select":
        from . import selection
        selection.print_report(selection.run(str(snap), str(root / "select"), log=quiet))
    elif name == "creatives":
        from . import creatives
        creatives.print_report(creatives.run(str(snap), str(root / "creatives"), log=quiet))


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _mtime(p: Path):
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
    except OSError:
        return None


def status(root: Path, jobs: Jobs | None = None) -> dict:
    """The datasets in the output folder, and each chapter's latest report."""
    out = {"root": str(root), "datasets": {}, "reports": {}, "job": jobs.status() if jobs else None,
           "jobs": {k: v[0] for k, v in JOBS.items()}}
    for name in ("snapshot", "snapshot-shift"):
        m = _read(root / name / "manifest.json")
        if m:
            out["datasets"][name] = {k: m.get(k) for k in ("adtrain_version", "adsim_version", "days", "rows",
                                                          "by_split", "devices", "apps", "creative_truth")}
    files = {"deep": root / "deep" / "report.json", "pretrain": root / "pretrain" / "report.json",
             "export": root / "export" / "export.json", "deepcheck": root / "export" / "deepcheck.json",
             "retrain": root / "retrain" / "report.json", "select": root / "select" / "report.json",
             "creatives": root / "creatives" / "report.json"}
    for k, p in files.items():
        r = _read(p)
        if r is not None:
            out["reports"][k] = {"written": _mtime(p), "report": r}
    hourly = root / "retrain" / "calibration_by_hour.csv"
    if hourly.exists() and "retrain" in out["reports"]:
        h = pd.read_csv(hourly)
        den = "true_credited" if "true_credited" in h else "true"
        arms = [c for c in h.columns if c not in ("hour", "group", "impressions", den)]
        out["reports"]["retrain"]["hourly"] = {"arms": arms, "truth": den,
                                               "rows": h.round(6).to_dict("records")}
    return out


def _pca2(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    P = U[:, :2] * S[:2]
    if P.shape[1] < 2:
        P = np.column_stack([P, np.zeros(len(P))])
    return P / (np.abs(P).max() + 1e-12)


def tsne(X: np.ndarray, perplexity=None, iters=600, seed=7) -> np.ndarray:
    """A small exact t-SNE (van der Maaten and Hinton, 2008) in numpy, from
    a PCA start: enough for a hundred apps or a few thousand devices, and
    the same map every time for the same vectors."""
    n = len(X)
    if n < 5:
        return _pca2(X)
    perplexity = perplexity or min(30.0, max(5.0, (n - 1) / 4))
    E = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)  # cosine geometry, as the neighbors
    D = np.maximum(2 - 2 * E @ E.T, 0)
    P = np.zeros((n, n))
    target = np.log(perplexity)
    for i in range(n):  # a binary search for each point's bandwidth
        lo, hi, beta = 0.0, np.inf, 1.0
        d = np.delete(D[i], i)
        for _ in range(60):
            w = np.exp(-(d - d.min()) * beta)
            sw = w.sum()
            h = np.log(sw) + beta * ((d - d.min()) * w).sum() / sw
            if abs(h - target) < 1e-4:
                break
            if h > target:
                lo, beta = beta, beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi, beta = beta, (beta + lo) / 2
        P[i, np.arange(n) != i] = w / sw
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)
    Y = _pca2(X) * 1e-2
    rng = np.random.default_rng(seed)
    Y = Y + rng.normal(0, 1e-4, Y.shape)
    vel, gains = np.zeros_like(Y), np.ones_like(Y)
    lr = max(n / 12 / 4, 50.0)
    for it in range(iters):
        ex = 12.0 if it < 100 else 1.0
        diff = Y[:, None, :] - Y[None, :, :]
        num = 1 / (1 + (diff ** 2).sum(-1))
        np.fill_diagonal(num, 0)
        Q = np.maximum(num / num.sum(), 1e-12)
        G = 4 * (((ex * P - Q) * num)[:, :, None] * diff).sum(1)
        mom = 0.5 if it < 250 else 0.8
        gains = np.where(np.sign(G) != np.sign(vel), gains + 0.2, gains * 0.8).clip(0.01)
        vel = mom * vel - lr * gains * G
        Y = Y + vel
        Y = Y - Y.mean(0)
    return Y / (np.abs(Y).max(0) + 1e-12)


_cache: dict = {}


def _neighbors(X: np.ndarray, k=K):
    E = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    idx = np.argsort(-S, axis=1)[:, :k]
    return idx, np.take_along_axis(S, idx, axis=1)


def embeddings(root: Path, source: str, layout: str = "tsne") -> dict:
    """2-D positions (PCA), genres, and nearest neighbors (cosine, in the
    full space) for one source of vectors."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}")
    snap = root / "snapshot"
    apps = pd.read_parquet(snap / "apps.parquet") if (snap / "apps.parquet").exists() else None
    if apps is None:
        raise ValueError(f"no Part IV dataset in {snap}")
    genre = dict(zip(apps.bundle, apps.genre))
    note, version = "", None
    if source == "dcn":
        p = root / "deep" / "app_embeddings.parquet"
        if not p.exists():
            raise ValueError("run Chapters 25 and 26 first (adtrain deep)")
        df = pd.read_parquet(p)
        keys = list(df.app)
        X = df[[c for c in df.columns if c.startswith("e")]].to_numpy(float)
        keep = (df.impressions.to_numpy() >= 200) & np.array([k in genre for k in keys])
        note = "apps in at least 200 impressions, as in Chapter 26"
        keys, X = [k for k, m in zip(keys, keep) if m], X[keep]
    elif source == "transformer":
        p = root / "pretrain" / "app_vectors.parquet"
        if not p.exists():
            raise ValueError("run Chapter 27 first (adtrain pretrain)")
        df = pd.read_parquet(p)
        version = str(df.version.iloc[0])
        hist = pd.read_parquet(snap / "histories.parquet", columns=["app", "kind"])
        installed = set(hist[hist.kind.isin(["install", "shared_install"])].app)
        df = df[df.app.isin(installed) & df.app.isin(genre)]
        note = "apps with installs in the histories, the ones pretraining had targets for"
        keys = list(df.app)
        X = df[[c for c in df.columns if c.startswith("e")]].to_numpy(float)
    elif source == "coinstall":
        from .pretrain import coinstall_vectors
        hist = pd.read_parquet(snap / "histories.parquet")
        app_id = {a: i + 2 for i, a in enumerate(sorted(set(apps.bundle) | set(hist.app)))}
        W, known = coinstall_vectors(hist, app_id, int(hist.ts_ms.max()) + 1, 8)
        inv = {v: k for k, v in app_id.items()}
        keys = [inv[j] for j in sorted(known) if inv[j] in genre]
        X = W[[app_id[k] for k in keys]]
        note = "apps with installs in the histories"
    else:  # devices
        p = root / "pretrain" / "device_states.parquet"
        if not p.exists():
            raise ValueError("run Chapter 27 first (adtrain pretrain)")
        st = pd.read_parquet(p)
        version = str(st.version.iloc[0])
        st = st[st.day == st.day.max()].dropna()
        hist = pd.read_parquet(snap / "histories.parquet")
        inst = hist[hist.kind.isin(["install", "shared_install"])].drop_duplicates(["device", "app"])
        inst = inst.assign(genre=inst.app.map(genre)).dropna(subset=["genre"])
        top = inst.groupby("device").genre.agg(lambda s: s.value_counts().index[0])
        st = st[st.device.isin(top.index)]
        if len(st) > 800:  # a seeded sample keeps the map readable, and its t-SNE to seconds
            st = st.sample(800, random_state=7).sort_values("device")
        keys = list(st.device)
        X = st[[c for c in st.columns if c.startswith("h")]].to_numpy(float)
        genre = top.to_dict()
        note = (f"a sample of {len(st)} device states of day {int(st.day.max())}, by the genre each device "
                f"installed most")
    g = np.array([genre.get(k, "") for k in keys])
    if layout == "tsne":
        key = (source, X.shape, float(np.round(X.sum(), 6)))
        if key not in _cache:  # t-SNE takes seconds for the devices: once per set of vectors
            _cache[key] = tsne(X, iters=600 if len(X) < 500 else 350)
        P = _cache[key]
    else:
        P = _pca2(X)
    idx, sim = _neighbors(X)
    same = float(np.mean([(g[idx[i]] == g[i]).mean() for i in range(len(keys))])) if len(keys) > K else None
    base = float(np.mean([(g[np.arange(len(g)) != i] == g[i]).mean() for i in range(len(g))])) if len(g) > 1 else None
    points = [{"key": k, "genre": str(g[i]), "x": round(float(P[i, 0]), 4), "y": round(float(P[i, 1]), 4),
               "neighbors": [[keys[j], str(g[j]), round(float(s), 3)] for j, s in zip(idx[i], sim[i])]}
              for i, k in enumerate(keys)]
    if source == "devices":  # the neighbor lists of 1,800 devices aren't useful on screen
        for p_ in points:
            p_["neighbors"] = p_["neighbors"][:3]
    return {"source": source, "layout": layout, "title": SOURCES[source], "note": note, "version": version,
            "dim": int(X.shape[1]), "n": len(keys), "points": points,
            "same_genre_neighbors": None if same is None else round(same, 3),
            "same_genre_random": None if base is None else round(base, 3),
            "explained": _explained(X)}


def _explained(X):
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False) ** 2
    return round(float(s[:2].sum() / max(s.sum(), 1e-12)), 3)
