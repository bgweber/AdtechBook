"""Deep conversion models with embeddings, on the Part IV dataset (Chapters
25 and 26).

A small multi-task DCNv2 in PyTorch, with three heads: click, install
after a click, and install after a view, the same three targets as the
LightGBM models of Part III. Each ID field gets an embedding; apps share one
table, whether an app is the publisher, the advertiser, or one of the
device's recent apps, and the recent apps are mean-pooled into one vector.
The cross network learns feature crosses explicitly, the MLP the rest.

The comparison is with LightGBM trained on the same rows, and everything is
checked against the simulator's true rates:

    python -m adtrain deep --snapshot out/snapshot

It trains on a laptop's CPU in a few minutes. On a GPU, with a bigger
snapshot, wider embeddings, and more epochs, the same code scales: the book
provides the code, not the compute.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, gbm, metrics

# ID fields with their own embedding table; apps share one
FIELDS = ("exchange", "placement", "country", "os", "os_major", "make", "model", "connection_type",
          "campaign", "creative", "creative_format", "goal")
APP_FIELDS = ("publisher_app", "adv_bundle")
NUMERIC = ("bidfloor", "rewarded", "interstitial", "device_known", "our_imps_today",
           "seconds_since_our_last_imp", "device_prior_installs", "device_prior_purchases",
           "device_requests", "seconds_since_first_seen", "seconds_since_last_seen")
HEADS = ("click", "install_click", "install_view")
DIM, CROSS, HIDDEN = 8, 2, (64, 32)
EPOCHS, BATCH, LR = 4, 1024, 2e-3
SEED = 7
PAD, OOV = 0, 1


def _vocab(values, min_count=1):
    s = pd.Series(values).value_counts()
    return {v: i + 2 for i, v in enumerate(sorted(s[s >= min_count].index))}


def encode(imp: pd.DataFrame, vocabs: dict) -> dict:
    """The model's inputs: an index per ID field, padded app lists, and
    standardized numerics."""
    out = {}
    for f in FIELDS:
        v = vocabs[f]
        out[f] = np.array([v.get(str(x), OOV) for x in imp[f].astype(str)], dtype=np.int64)
    va = vocabs["app"]
    for f in APP_FIELDS:
        out[f] = np.array([va.get(x, OOV) for x in imp[f]], dtype=np.int64)
    L = max(1, int(imp.recent_apps.map(len).max()))
    rec = np.zeros((len(imp), L), dtype=np.int64)
    for i, apps in enumerate(imp.recent_apps):
        for j, a in enumerate(apps[:L]):
            rec[i, j] = va.get(a, OOV)
    out["recent_apps"] = rec
    num = np.column_stack([np.log1p(np.clip(imp[k].fillna(0).to_numpy(float), 0, None)) for k in NUMERIC])
    hour = imp.hour_of_day.fillna(0).to_numpy(float)
    num = np.column_stack([num, np.sin(2 * math.pi * hour / 24), np.cos(2 * math.pi * hour / 24)])
    out["numeric"] = num.astype(np.float32)
    return out


def build_vocabs(imp: pd.DataFrame) -> dict:
    """Vocabularies over the whole snapshot, as a pipeline that rebuilds
    them every day would: IDs that only appear after the training days get
    rows that training never touches."""
    v = {f: _vocab(imp[f].astype(str)) for f in FIELDS}
    apps = list(imp.publisher_app) + list(imp.adv_bundle) + [a for xs in imp.recent_apps for a in xs]
    v["app"] = _vocab(apps)
    return v


def _model(vocabs, n_num):
    import torch
    import torch.nn as nn

    class DCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.ModuleDict({f: nn.Embedding(len(vocabs[f]) + 2, DIM, padding_idx=PAD) for f in FIELDS})
            self.app = nn.Embedding(len(vocabs["app"]) + 2, DIM, padding_idx=PAD)
            d = DIM * (len(FIELDS) + len(APP_FIELDS) + 1) + n_num
            self.cross = nn.ModuleList([nn.Linear(d, d) for _ in range(CROSS)])
            layers, h = [], d
            for w in HIDDEN:
                layers += [nn.Linear(h, w), nn.ReLU()]
                h = w
            self.mlp = nn.Sequential(*layers)
            self.heads = nn.ModuleDict({k: nn.Linear(d + h, 1) for k in HEADS})

        def forward(self, x):
            parts = [self.emb[f](x[f]) for f in FIELDS] + [self.app(x[f]) for f in APP_FIELDS]
            rec = x["recent_apps"]
            mask = (rec != PAD).float().unsqueeze(-1)
            pooled = (self.app(rec) * mask).sum(1) / mask.sum(1).clamp(min=1)  # mean pooling
            x0 = torch.cat(parts + [pooled, x["numeric"]], dim=1)
            xl = x0
            for lin in self.cross:  # DCNv2: x_{l+1} = x0 * (W x_l + b) + x_l
                xl = x0 * lin(xl) + xl
            z = torch.cat([xl, self.mlp(x0)], dim=1)
            return {k: self.heads[k](z).squeeze(-1) for k in HEADS}

    return DCN()


def _tensors(enc, idx):
    import torch
    return {k: torch.from_numpy(v[idx]) for k, v in enc.items()}


def _masks(imp):
    c = imp.click.to_numpy()
    fin = imp.install_final.to_numpy(bool)
    return {"click": imp.click_final.to_numpy(bool),
            "install_click": fin & (c == 1), "install_view": fin & (c == 0)}


def _labels(imp):
    return {"click": imp.click.to_numpy(float), "install_click": imp.install_click.to_numpy(float),
            "install_view": imp.install_view.to_numpy(float)}


def train_dcn(imp, enc, vocabs, log=print):
    import torch
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    model = _model(vocabs, enc["numeric"].shape[1])
    init = {k: v.detach().clone() for k, v in model.state_dict().items() if "emb" in k or k.startswith("app.")}
    tr = np.flatnonzero(imp.split.to_numpy() == "train")
    va = np.flatnonzero(imp.split.to_numpy() == "valid")
    # standardize the numerics on the training rows
    mu, sd = enc["numeric"][tr].mean(0), enc["numeric"][tr].std(0) + 1e-6
    enc = {**enc, "numeric": ((enc["numeric"] - mu) / sd).astype(np.float32)}
    y, m = _labels(imp), _masks(imp)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.default_rng(SEED)
    best, best_state, t0 = math.inf, None, time.time()
    for ep in range(EPOCHS):
        model.train()
        order = rng.permutation(tr)
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            out = model(_tensors(enc, b))
            loss = 0
            for k in HEADS:
                mk = torch.from_numpy(m[k][b])
                if mk.any():
                    loss = loss + (bce(out[k], torch.from_numpy(y[k][b]).float()) * mk).sum() / mk.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        vl = _loss(model, enc, va, y, m)
        log(f"  epoch {ep + 1}: validation loss {vl:.4f}")
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    secs = time.time() - t0
    # embedding rows training never touched: unchanged since initialization
    untouched = {}
    for k, v in model.state_dict().items():
        if k in init:
            same = (v == init[k]).all(1).numpy()
            same[PAD] = same[OOV] = False  # padding, and the out-of-vocabulary bucket
            untouched[k] = same
    return model, enc, untouched, secs


def _loss(model, enc, idx, y, m):
    import torch
    model.eval()
    with torch.no_grad():
        out = predict(model, enc, idx, logits=True)
    bce = lambda z, t: np.logaddexp(0, z) - t * z
    tot = 0.0
    for k in HEADS:
        mk = m[k][idx]
        if mk.any():
            tot += float(bce(out[k][mk], y[k][idx][mk]).mean())
    return tot


def predict(model, enc, idx, logits=False):
    import torch
    model.eval()
    res = {k: [] for k in HEADS}
    with torch.no_grad():
        for s in range(0, len(idx), 8192):
            out = model(_tensors(enc, idx[s:s + 8192]))
            for k in HEADS:
                res[k].append(out[k].numpy())
    res = {k: np.concatenate(v) for k, v in res.items()}
    return res if logits else {k: 1 / (1 + np.exp(-v)) for k, v in res.items()}


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _platt(p, y, iters=50):
    """Platt scaling: a logistic fit, a z + b, on the prediction's log odds
    z, by Newton's method."""
    z, a, b = _logit(p), 1.0, 0.0
    for _ in range(iters):
        q = 1 / (1 + np.exp(-(a * z + b)))
        g = np.array([((q - y) * z).sum(), (q - y).sum()])
        w = q * (1 - q)
        H = np.array([[(w * z * z).sum(), (w * z).sum()], [(w * z).sum(), w.sum()]]) + 1e-9 * np.eye(2)
        a, b = np.array([a, b]) - np.linalg.solve(H, g)
    return lambda x: 1 / (1 + np.exp(-(a * _logit(x) + b)))


def train_trees(imp, log=print):
    """LightGBM on the same rows, inputs, and targets, for comparison."""
    import lightgbm as lgb
    raws = []
    for r in imp.itertuples(index=False):
        d = {k: getattr(r, k) for k in gbm.CATEGORICAL if k != "adv_app"}
        d["adv_app"] = r.adv_bundle
        d.update({k: getattr(r, k) for k in gbm.NUMERIC})
        raws.append(d)
    tr = imp.split.to_numpy() == "train"
    vocab = gbm.build_vocab([r for r, t in zip(raws, tr) if t], min_count=10)
    X = np.array([gbm.encode(r, vocab) for r in raws], float)
    y, m = _labels(imp), _masks(imp)
    models, t0 = {}, time.time()
    for k in HEADS:
        sel = tr & m[k]
        ds = lgb.Dataset(X[sel], y[k][sel], feature_name=list(gbm.CATEGORICAL) + list(gbm.NUMERIC),
                         categorical_feature=list(gbm.CATEGORICAL))
        models[k] = lgb.train({**gbm.PARAMS, "objective": "binary", "seed": SEED, "deterministic": True,
                               "num_threads": 1, "verbose": -1}, ds, num_boost_round=gbm.ROUNDS)
    return models, X, time.time() - t0


def _metrics(p, y, true, mask):
    if mask.sum() == 0:
        return {}
    p, y, t = p[mask], y[mask], true[mask]
    return {"rows": int(mask.sum()), "auc": _r(_auc(y, p)) if 0 < y.sum() < len(y) else None,
            "log_loss": _r(metrics.log_loss(y, p)),
            "calibration_vs_truth": _r(p.mean() / t.mean()),
            "error_vs_truth": _r(np.abs(p - t).mean() / t.mean())}


def _auc(y, p):
    r = pd.Series(p).rank().to_numpy()
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def _r(x, k=4):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


def genre_neighbors(model, vocabs, apps: pd.DataFrame, imp: pd.DataFrame, k=5, min_rows=200) -> dict:
    """Do apps with similar embeddings share a genre? For each app seen in
    enough impressions, the share of its k nearest neighbors (by cosine)
    with its genre, against the share for random pairs."""
    genre = dict(zip(apps.bundle, apps.genre))
    counts = pd.concat([imp.publisher_app, imp.adv_bundle]).value_counts()
    keep = [a for a in counts[counts >= min_rows].index if a in vocabs["app"] and a in genre]
    W = model.app.weight.detach().numpy()[[vocabs["app"][a] for a in keep]]
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    S = W @ W.T
    np.fill_diagonal(S, -np.inf)
    g = np.array([genre[a] for a in keep])
    nn_same = np.mean([(g[np.argsort(-S[i])[:k]] == g[i]).mean() for i in range(len(keep))])
    base = np.mean([(g[np.arange(len(g)) != i] == g[i]).mean() for i in range(len(keep))])
    examples = []
    for i in np.argsort(-counts.reindex(keep).to_numpy())[:3]:
        nb = np.argsort(-S[i])[:3]
        examples.append({"app": keep[i], "genre": g[i], "neighbors": [[keep[j], g[j]] for j in nb]})
    return {"apps": len(keep), "k": k, "same_genre_neighbors": _r(nn_same, 3),
            "same_genre_random": _r(base, 3), "examples": examples}


def coinstall_neighbors(hist: pd.DataFrame, apps: pd.DataFrame, k=5, dim=8) -> dict:
    """The same check for app vectors learned from behavior instead: which
    devices install apps together (credited and shared installs), as a
    PPMI matrix factored to dim dimensions."""
    genre = dict(zip(apps.bundle, apps.genre))
    inst = hist[hist.kind.isin(["install", "shared_install"])]
    M = pd.crosstab(inst.device, inst.app).clip(0, 1)
    names = [a for a in M.columns if a in genre]
    M = M[names]
    C = (M.T @ M).to_numpy(float).copy()
    np.fill_diagonal(C, 0)
    tot, r, c = C.sum(), C.sum(1), C.sum(0)
    P = np.maximum(np.log(np.maximum(C * tot / np.maximum(np.outer(r, c), 1e-12), 1e-12)), 0)
    U, S, _ = np.linalg.svd(P)
    W = U[:, :dim] * np.sqrt(S[:dim])
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    Sm = W @ W.T
    np.fill_diagonal(Sm, -np.inf)
    g = np.array([genre[a] for a in names])
    same = np.mean([(g[np.argsort(-Sm[i])[:k]] == g[i]).mean() for i in range(len(names))])
    base = np.mean([(g[np.arange(len(g)) != i] == g[i]).mean() for i in range(len(names))])
    return {"apps": len(names), "k": k, "same_genre_neighbors": _r(same, 3), "same_genre_random": _r(base, 3)}


def run(snapshot_dir="out/snapshot", log=print, out_dir=None) -> dict:
    from . import snapshot
    s = snapshot.load(snapshot_dir)
    imp, truth = s["impressions"], s["truth"]
    true = {"click": truth.p_click.to_numpy(float), "install_click": truth.p_install_click.to_numpy(float),
            "install_view": truth.p_install_view.to_numpy(float)}
    vocabs = build_vocabs(imp)
    enc = encode(imp, vocabs)
    log(f"deep: {len(imp):,} impressions, {len(vocabs['app']):,} apps, "
        f"{sum(len(v) for k, v in vocabs.items() if k != 'app'):,} other IDs")
    model, enc, untouched, dcn_secs = train_dcn(imp, enc, vocabs, log)
    log("deep: training LightGBM on the same rows")
    trees, X, gbm_secs = train_trees(imp)
    y, m = _labels(imp), _masks(imp)
    sp = imp.split.to_numpy()
    va, te = np.flatnonzero(sp == "valid"), np.flatnonzero(sp == "test")
    p_dcn = predict(model, enc, te)
    p_dcn_va = predict(model, enc, va)
    p_gbm = {k: trees[k].predict(X[te]) for k in HEADS}
    rep = {"model": "dcnv2", "adtrain_version": __version__, "snapshot": s["manifest"]["days"],
           "rows": {sp_: int((sp == sp_).sum()) for sp_ in ("train", "valid", "test")},
           "params": int(sum(p.numel() for p in model.parameters())),
           "train_seconds": {"dcnv2": round(dcn_secs, 1), "lightgbm": round(gbm_secs, 1)},
           "heads": {}}
    # zeroing the embedding rows training never touched
    import torch
    zeroed = {k: v.clone() for k, v in model.state_dict().items()}
    n_untouched = {}
    for k, same in untouched.items():
        zeroed[k][torch.from_numpy(same)] = 0
        n_untouched[k.replace(".weight", "").replace("emb.", "")] = int(same.sum())
    model_z = _model(vocabs, enc["numeric"].shape[1])
    model_z.load_state_dict(zeroed)
    p_z = predict(model_z, enc, te)
    for k in HEADS:
        mk = m[k][te]
        cal = _platt(p_dcn_va[k][m[k][va]], y[k][va][m[k][va]])
        rep["heads"][k] = {
            "lightgbm": _metrics(p_gbm[k], y[k][te], true[k][te], mk),
            "dcnv2": _metrics(p_dcn[k], y[k][te], true[k][te], mk),
            "dcnv2_zeroed": _metrics(p_z[k], y[k][te], true[k][te], mk),
            "dcnv2_platt": _metrics(cal(p_dcn[k]), y[k][te], true[k][te], mk),
        }
    # the rows whose IDs training never saw
    new = np.zeros(len(te), bool)
    for f, key in (("campaign", "emb.campaign.weight"), ("creative", "emb.creative.weight")):
        idx = enc[f][te]
        new |= untouched[key][idx]
    rep["untouched_rows"] = n_untouched
    rep["test_rows_with_untrained_ids"] = int(new.sum())
    if new.any():
        k = "click"
        rep["new_ids_click"] = {"dcnv2": _metrics(p_dcn[k], y[k][te], true[k][te], new & m[k][te]),
                                "dcnv2_zeroed": _metrics(p_z[k], y[k][te], true[k][te], new & m[k][te]),
                                "lightgbm": _metrics(p_gbm[k], y[k][te], true[k][te], new & m[k][te])}
    rep["genre_neighbors"] = genre_neighbors(model, vocabs, s["apps"], imp)
    rep["genre_neighbors_coinstall"] = coinstall_neighbors(s["histories"], s["apps"])
    rep["slices"] = slices(imp, te, p_gbm, p_dcn, true, m)
    rep["cost"] = cost(model, enc, trees, X, te)
    counts = pd.concat([imp.publisher_app, imp.adv_bundle]).value_counts()
    _save(out_dir or Path(snapshot_dir).parent / "deep", rep, model, vocabs, untouched, counts)
    return rep


def _save(out_dir, rep, model, vocabs, untouched, counts) -> None:
    """The report, and the learned app embeddings for the console's viewer."""
    import json
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    E = model.app.weight.detach().numpy()
    apps = sorted(vocabs["app"], key=vocabs["app"].get)
    idx = [vocabs["app"][a] for a in apps]
    df = pd.DataFrame({"app": apps, "untouched": untouched["app.weight"][idx],
                       "impressions": counts.reindex(apps).fillna(0).astype(int).to_numpy(),
                       **{f"e{i}": E[idx, i] for i in range(E.shape[1])}})
    df.to_parquet(out / "app_embeddings.parquet", index=False)
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    rep["out_dir"] = str(out)


def slices(imp, te, p_gbm, p_dcn, true, m) -> list:
    """Where trees should struggle (Chapter 25): rare advertiser apps, device
    and app pairs never seen in training, and devices with installs."""
    tr = imp.split.to_numpy() == "train"
    n_app = imp[tr].adv_bundle.value_counts()
    rare = imp.adv_bundle.map(n_app).fillna(0).to_numpy()[te] < np.quantile(n_app, 0.25)
    pairs = set(zip(imp.device[tr], imp.adv_bundle[tr]))
    unseen = np.array([(d, a) not in pairs for d, a in zip(imp.device.to_numpy()[te], imp.adv_bundle.to_numpy()[te])])
    hist = imp.device_prior_installs.fillna(0).to_numpy()[te] >= 1
    out = []
    for name, s_ in (("all", np.ones(len(te), bool)), ("rare advertiser apps", rare),
                     ("device-app pairs not in training", unseen), ("devices with installs", hist)):
        row = {"slice": name, "rows": int(s_.sum())}
        for k in ("click", "install_view"):
            mk = s_ & m[k][te]
            if mk.sum() < 200:
                continue
            t = true[k][te][mk]
            for model, p in (("lightgbm", p_gbm[k]), ("dcnv2", p_dcn[k])):
                row[f"{k}_{model}"] = _r(np.abs(p[mk] - t).mean() / t.mean(), 3)
        out.append(row)
    return out


def cost(model, enc, trees, X, te, n=2000) -> dict:
    """Time per prediction on this machine's CPU, one at a time and in
    batches of 1,000, for the three heads."""
    import torch
    idx = te[:n]
    t0 = time.time()
    for i in idx[:200]:
        predict(model, enc, np.array([i]))
    dcn_one = (time.time() - t0) / 200
    t0 = time.time()
    predict(model, enc, idx)
    dcn_batch = (time.time() - t0) / len(idx)
    t0 = time.time()
    for i in idx[:200]:
        for k in HEADS:
            trees[k].predict(X[i:i + 1])
    gbm_one = (time.time() - t0) / 200
    t0 = time.time()
    for k in HEADS:
        trees[k].predict(X[idx])
    gbm_batch = (time.time() - t0) / len(idx)
    us = lambda s: round(s * 1e6, 1)
    return {"dcnv2_one_us": us(dcn_one), "dcnv2_batch_us": us(dcn_batch),
            "lightgbm_one_us": us(gbm_one), "lightgbm_batch_us": us(gbm_batch),
            "threads": torch.get_num_threads(), "gpu": torch.cuda.is_available()}


def print_report(r: dict) -> None:
    print(f"DCNv2, {r['params']:,} parameters: trained in {r['train_seconds']['dcnv2']} s on the CPU "
          f"(LightGBM {r['train_seconds']['lightgbm']} s); train/valid/test "
          f"{r['rows']['train']:,} / {r['rows']['valid']:,} / {r['rows']['test']:,} impressions")
    print(f"  {'head':<14}{'model':<14}{'rows':>8}{'AUC':>8}{'log loss':>10}{'calib. vs truth':>17}{'error vs truth':>16}")
    for k, h in r["heads"].items():
        for name, x in h.items():
            if x:
                print(f"  {k:<14}{name:<14}{x['rows']:>8,}{x['auc'] or 0:>8.3f}{x['log_loss']:>10.4f}"
                      f"{x['calibration_vs_truth']:>17.3f}{x['error_vs_truth']:>16.3f}")
    print(f"  embedding rows training never touched: " + ", ".join(f"{k} {v}" for k, v in r["untouched_rows"].items() if v))
    print(f"  test impressions with an untrained campaign or creative: {r['test_rows_with_untrained_ids']:,}")
    if "new_ids_click" in r:
        for name, x in r["new_ids_click"].items():
            if x:
                print(f"    click on them, {name:<13} calibration vs truth {x['calibration_vs_truth']}, "
                      f"error vs truth {x['error_vs_truth']}")
    print("  error against the true rates, by slice (Chapter 25):")
    for s_ in r["slices"]:
        print(f"    {s_['slice']:<34}{s_['rows']:>8,} rows  click: trees {s_.get('click_lightgbm', '-')}, "
              f"DCNv2 {s_.get('click_dcnv2', '-')}; install after view: trees {s_.get('install_view_lightgbm', '-')}, "
              f"DCNv2 {s_.get('install_view_dcnv2', '-')}")
    c = r["cost"]
    print(f"  time per prediction, three heads, CPU ({c['threads']} threads): DCNv2 {c['dcnv2_one_us']} us alone, "
          f"{c['dcnv2_batch_us']} us in batches; LightGBM in Python {c['lightgbm_one_us']} us alone, "
          f"{c['lightgbm_batch_us']} us in batches")
    g = r["genre_neighbors"]
    print(f"  app embeddings: {g['same_genre_neighbors']:.0%} of each app's {g['k']} nearest neighbors share its "
          f"genre, against {g['same_genre_random']:.0%} for random pairs ({g['apps']} apps)")
    for e in g["examples"]:
        print(f"    {e['app']} ({e['genre']}): " + ", ".join(f"{a} ({gg})" for a, gg in e["neighbors"]))
    c = r["genre_neighbors_coinstall"]
    print(f"  app vectors from which devices install apps together: {c['same_genre_neighbors']:.0%} of the "
          f"{c['k']} nearest neighbors share the genre, against {c['same_genre_random']:.0%} ({c['apps']} apps)")
    if r.get("out_dir"):
        print(f"  wrote the report and the app embeddings to {r['out_dir']}")
