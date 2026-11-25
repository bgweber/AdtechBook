"""Pretraining on non-attributed installs, and Design A: deep signals into
trees, on the Part IV dataset (Chapter 27).

A small causal transformer reads each device's events in time order, the
apps it was seen in on the bid stream, its clicks, and its installs,
credited to us or shared by advertisers, and learns to predict the next
install. No attributed label is needed: every install in every history is
a target. Its app table starts from co-install vectors (the warm start).

The device state, the transformer's last hidden vector, is computed once
a day for every device from the events before that day, and scored against
the advertised app. The scores go into the LightGBM install models as
features, and each variant is checked against the simulator's true rates:

    python -m adtrain pretrain --snapshot out/snapshot

It runs on a laptop's CPU in a minute or two and writes the app vectors,
the daily device states, and the report to out/pretrain. The model is tiny
and the histories are short; the same code takes longer histories, more
devices, and wider models on a GPU: the book provides the code, not the
compute.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, deep, gbm

DAY_MS = 86_400_000
KINDS = {"seen": 1, "click": 2, "install": 3, "shared_install": 4}
OUTCOMES = (3, 4)  # installs, credited or shared, are the only targets
CTX = 128  # the last 128 events
DIM, LAYERS, HEADS_N, FF = 32, 2, 2, 64
EPOCHS, BATCH, LR = 6, 128, 3e-3
GAPS = 16
SEED = 7
PAD, OOV = 0, 1
HEADS = ("click", "install_click", "install_view")
# the device history inputs of Part III, dropped for the plain variant
HISTORY = ("device_known", "device_requests", "seconds_since_first_seen", "seconds_since_last_seen",
           "device_prior_installs", "device_prior_purchases")


def _gap(dt_ms):
    """A bucket for the time since the previous event: log2 of hours."""
    h = np.maximum(dt_ms, 0) / 3.6e6
    return np.clip(np.floor(np.log2(1 + h)).astype(np.int64) + 1, 1, GAPS - 1)


class Events:
    """Each device's events in time order, as token arrays."""

    def __init__(self, hist: pd.DataFrame, apps: pd.DataFrame):
        names = sorted(set(apps.bundle) | set(hist.app))
        self.app_id = {a: i + 2 for i, a in enumerate(names)}
        self.apps = names
        h = hist.sort_values(["device", "ts_ms", "kind"], kind="stable")
        self.by_dev = {}
        for d, g in h.groupby("device", sort=False):
            self.by_dev[d] = (g.ts_ms.to_numpy(np.int64), g.kind.map(KINDS).to_numpy(np.int64),
                              g.app.map(self.app_id).to_numpy(np.int64))

    def window(self, device, before_ms, drop_app=None):
        """The device's last CTX events before a time, optionally without
        one app's events (to mask the target)."""
        if device not in self.by_dev:
            return None
        ts, k, a = self.by_dev[device]
        n = np.searchsorted(ts, before_ms, side="left")
        ts, k, a = ts[:n], k[:n], a[:n]
        if drop_app is not None:
            keep = a != drop_app
            ts, k, a = ts[keep], k[keep], a[keep]
        if len(ts) == 0:
            return None
        ts, k, a = ts[-CTX:], k[-CTX:], a[-CTX:]
        g = np.concatenate([[GAPS - 1], _gap(np.diff(ts))]) if len(ts) > 1 else np.array([GAPS - 1])
        return ts, k, a, g


def _pad(wins):
    n = len(wins)
    out = {x: np.zeros((n, CTX), np.int64) for x in ("kind", "app", "gap", "ts")}
    length = np.zeros(n, np.int64)
    for i, (ts, k, a, g) in enumerate(wins):
        L = len(k)
        out["kind"][i, :L], out["app"][i, :L], out["gap"][i, :L], out["ts"][i, :L] = k, a, g, ts
        length[i] = L
    return out, length


def _period(ts):
    """The popularity period of an install: its day in the run, or its month
    before the run."""
    ts = np.asarray(ts, np.int64)
    return np.where(ts >= 0, ts // DAY_MS, -1 - (-ts) // (30 * DAY_MS))


class Popularity:
    """Each app's share of installs in each period, smoothed toward its
    share overall. The model is trained with log q added to its logits,
    so its dot product learns only the device's lift over what everyone is
    installing at the time, the popularity correction of Chapter 27."""

    def __init__(self, inst: pd.DataFrame, app_id: dict, n_apps: int, prior=20.0):
        a = inst.app.map(app_id).to_numpy()
        per = _period(inst.ts_ms.to_numpy())
        tot = np.bincount(a, minlength=n_apps).astype(float) + 0.5
        tot[:2] = 0
        self.q_all = tot / tot.sum()
        self.periods = {p: i for i, p in enumerate(sorted(set(per)))}
        Q = np.zeros((len(self.periods), n_apps))
        np.add.at(Q, ([self.periods[x] for x in per], a), 1.0)
        Q = (Q + prior * self.q_all) / (Q.sum(1, keepdims=True) + prior)
        with np.errstate(divide="ignore"):
            self.logq = np.log(Q)
        self.logq[:, :2] = -np.inf

    def index(self, ts):
        per = _period(ts)
        keys = np.array(sorted(self.periods))
        j = np.clip(np.searchsorted(keys, per, side="right") - 1, 0, len(keys) - 1)  # the latest period seen
        return j


def _model(n_apps):
    import torch
    from torch import nn

    class Pretrain(nn.Module):
        def __init__(self):
            super().__init__()
            self.app = nn.Embedding(n_apps, DIM, padding_idx=PAD)
            self.kind = nn.Embedding(len(KINDS) + 1, DIM, padding_idx=PAD)
            self.gap = nn.Embedding(GAPS, DIM, padding_idx=PAD)
            self.pos = nn.Embedding(CTX, DIM)
            layer = nn.TransformerEncoderLayer(DIM, HEADS_N, FF, dropout=0.1, batch_first=True, norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(DIM)
            nn.init.normal_(self.app.weight, std=0.1)

        def forward(self, kind, app, gap, length):
            L = kind.shape[1]
            pos = torch.arange(L).expand_as(kind)
            x = self.app(app) + self.kind(kind) + self.gap(gap) + self.pos(pos)
            causal = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
            pad = torch.arange(L)[None, :] >= length[:, None]
            return self.norm(self.blocks(x, mask=causal, src_key_padding_mask=pad))

        def logits(self, h):
            # the device's lift for each app: the output layer shares the input app table
            out = h @ self.app.weight.T
            out[..., :2] = float("-inf")  # PAD and OOV are never predicted
            return out

    return Pretrain()


def coinstall_vectors(hist, app_id, before_ms, dim):
    """App vectors from which devices install apps together, before a time:
    a PPMI matrix factored with SVD (the warm start, and Chapter 26's
    check)."""
    inst = hist[hist.kind.isin(["install", "shared_install"]) & (hist.ts_ms < before_ms)]
    M = pd.crosstab(inst.device, inst.app).clip(0, 1)
    C = (M.T @ M).to_numpy(float).copy()
    np.fill_diagonal(C, 0)
    tot, r, c = C.sum(), C.sum(1), C.sum(0)
    P = np.maximum(np.log(np.maximum(C * tot / np.maximum(np.outer(r, c), 1e-12), 1e-12)), 0)
    U, S, _ = np.linalg.svd(P)
    d = min(dim, len(S))
    W = U[:, :d] * np.sqrt(S[:d])
    out = np.zeros((len(app_id) + 2, dim))
    for j, a in enumerate(M.columns):
        out[app_id[a], :d] = W[j]
    return out, {app_id[a] for a in M.columns}


def _targets(X, length):
    """The target at each position: the next event's app, if that event is
    an install, and its time (for the popularity of its period)."""
    kind, app, ts = X["kind"], X["app"], X["ts"]
    n, L = kind.shape
    out = np.isin(kind, OUTCOMES) & (np.arange(L)[None, :] < length[:, None])
    y = np.full((n, L), -100)
    t = np.zeros((n, L), np.int64)
    y[:, :-1] = np.where(out[:, 1:], app[:, 1:], -100)
    t[:, :-1] = ts[:, 1:]
    return y, t


def _baseline_loss(y, t, pop):
    """The loss of the popularity of each install's period alone, what the
    model must beat to have learned anything about devices."""
    m = y >= 0
    lq = np.nan_to_num(pop.logq[pop.index(t[m])], neginf=-1e9)
    lq = lq - np.logaddexp.reduce(lq, axis=1, keepdims=True)
    return round(float(-lq[np.arange(m.sum()), y[m]].mean()), 3)


def pretrain(ev: Events, hist, cutoff_ms, pop: Popularity, warm=True, log=print):
    """Pretraining on every device's events before the cutoff: predict the
    next install, with the popularity of the install's period added to the
    logits, so the model learns each device's lift, not what's popular."""
    import torch
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    wins = []
    for d, (ts, k, a) in ev.by_dev.items():
        n = np.searchsorted(ts, cutoff_ms, side="left")
        for end in range(n, 1, -CTX):  # windows of CTX events, newest first
            w = ev.window(d, ts[end - 1] + 1)
            if w is not None and len(w[0]) >= 2:
                wins.append(w)
    X, length = _pad(wins)
    y, t = _targets(X, length)
    n_targets = int((y >= 0).sum())
    model = _model(len(ev.apps) + 2)
    if warm:
        W, _ = coinstall_vectors(hist, ev.app_id, cutoff_ms, DIM)
        with torch.no_grad():
            model.app.weight[:] = torch.tensor(0.3 * W / (np.abs(W).max() + 1e-9), dtype=torch.float32) \
                + 0.02 * torch.randn(model.app.weight.shape)
            model.app.weight[PAD] = 0
    T = {k: torch.from_numpy(v) for k, v in X.items()}
    Tl = torch.from_numpy(length)
    logq = torch.tensor(np.nan_to_num(pop.logq, neginf=-1e9), dtype=torch.float32)
    Ty, Tp = torch.from_numpy(y), torch.from_numpy(pop.index(t.ravel()).reshape(t.shape))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    t0 = time.time()
    model.train()
    for ep in range(EPOCHS):
        order = rng.permutation(len(length))
        tot = 0.0
        for s in range(0, len(order), BATCH):
            b = torch.from_numpy(order[s:s + BATCH])
            Lb = int(Tl[b].max())
            h = model(T["kind"][b, :Lb], T["app"][b, :Lb], T["gap"][b, :Lb], Tl[b])
            z = model.logits(h) + logq[Tp[b, :Lb]]
            loss = torch.nn.functional.cross_entropy(z.reshape(-1, z.shape[-1]), Ty[b, :Lb].reshape(-1),
                                                     ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        log(f"pretrain: epoch {ep + 1}, next-install loss {tot / len(order):.3f}")
    model.eval()
    return model, {"windows": len(wins), "targets": n_targets, "loss": round(tot / len(order), 3),
                   "loss_popularity_only": _baseline_loss(y, t, pop), "seconds": round(time.time() - t0, 1),
                   "params": int(sum(p.numel() for p in model.parameters()))}


def states(model, ev: Events, keys):
    """The device state, the last hidden vector, for each (device, before
    time, masked app) key; None where the device has no events yet."""
    import torch
    wins, idx = [], []
    for i, (d, t, drop) in enumerate(keys):
        w = ev.window(d, t, drop)
        if w is not None:
            wins.append(w)
            idx.append(i)
    H = np.full((len(keys), DIM), np.nan, np.float32)
    newest = np.full(len(keys), -1, np.int64)
    if not wins:
        return H, newest
    X, length = _pad(wins)
    order = np.argsort(length, kind="stable")
    out = np.zeros((len(wins), DIM), np.float32)
    with torch.no_grad():
        for s in range(0, len(wins), 1024):
            sl = order[s:s + 1024]
            Lb = int(length[sl].max())
            h = model(*(torch.from_numpy(X[k][sl, :Lb]) for k in ("kind", "app", "gap")), torch.from_numpy(length[sl]))
            out[sl] = h[torch.arange(h.shape[0]), torch.from_numpy(length[sl] - 1)].numpy()
    H[idx] = out
    newest[idx] = [w[0][-1] for w in wins]
    return H, newest


def _lift(model, H):
    """Each device's lift for each app: the state's dot product with the
    app's vector, 0 for a device with no history."""
    import torch
    with torch.no_grad():
        return model.logits(torch.from_numpy(np.nan_to_num(H))).numpy()


def affinity(model, ev, imp, log=print):
    """The affinity of each impression's device for the advertised app,
    from the device state at the start of the impression's day, with the
    advertised app's own events removed from the inputs."""
    day0 = imp.day.to_numpy() * DAY_MS
    dev, adv = imp.device.to_numpy(), imp.adv_bundle.map(ev.app_id).fillna(OOV).astype(int).to_numpy()
    keys = pd.DataFrame({"d": dev, "t": day0, "a": adv})
    uniq = keys.drop_duplicates().reset_index(drop=True)
    drop = []
    for d, t, a in zip(uniq.d, uniq.t, uniq.a):
        x = ev.by_dev.get(d)
        drop.append(a if x is not None and (x[2][:np.searchsorted(x[0], t)] == a).any() else -1)
    uniq["m"] = drop
    st = uniq[["d", "t", "m"]].drop_duplicates().reset_index(drop=True)
    Hs, ns = states(model, ev, [(d, t, None if m < 0 else m) for d, t, m in zip(st.d, st.t, st.m)])
    j = uniq.merge(st.reset_index(), on=["d", "t", "m"], how="left")["index"].to_numpy()
    H, newest = Hs[j], ns[j]
    a = uniq.a.to_numpy()
    s_lift = _lift(model, H)[np.arange(len(uniq)), a]
    s_lift[np.isnan(H[:, 0]) | (a == OOV)] = np.nan
    # leakage check: every input event is before the impression's day
    leak = int((newest >= uniq.t.to_numpy()).sum())
    pos = keys.merge(uniq.reset_index(), on=["d", "t", "a"], how="left")["index"].to_numpy()
    return s_lift[pos], {"device_days": len(st), "masked": int((st.m >= 0).sum()),
                                      "with_history": int((~np.isnan(Hs[:, 0])).sum()), "leaks": leak}


def pooled_affinity(hist, ev, imp, W, known):
    """Variant 2: a pooled profile, the mean co-install vector of the
    device's installs before the day, without the advertised app, against
    the advertised app's vector (cosine)."""
    inst = hist[hist.kind.isin(["install", "shared_install"])].sort_values("ts_ms")
    by = {d: (g.ts_ms.to_numpy(), g.app.map(ev.app_id).to_numpy()) for d, g in inst.groupby("device")}
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    out = np.full(len(imp), np.nan)
    cache = {}
    for i, (d, day, a) in enumerate(zip(imp.device.to_numpy(), imp.day.to_numpy(), imp.adv_bundle.to_numpy())):
        j = ev.app_id.get(a, OOV)
        key = (d, day, j)
        if key not in cache:
            v = np.nan
            if d in by and j in known:
                ts, ap = by[d]
                ap = ap[:np.searchsorted(ts, day * DAY_MS)]
                ap = [x for x in ap if x != j and x in known]
                if ap:
                    m = Wn[ap].mean(0)
                    v = float(m @ Wn[j] / (np.linalg.norm(m) + 1e-9))
            cache[key] = v
        out[i] = cache[key]
    return out


def next_install(models, ev, hist, pop: Popularity, known, start_ms, end_ms, k=5):
    """Pretraining's own test: for each first install after the cutoff, rank
    the apps the device doesn't have yet, from its state at the start of
    that day. The share of installs in each scorer's top k. Popularity
    over the last 7 days is the baseline to beat: the market shifts as
    campaigns start and stop, and that alone predicts most installs."""
    allinst = hist[hist.kind.isin(["install", "shared_install"])]
    inst = allinst[(allinst.ts_ms >= start_ms) & (allinst.ts_ms < end_ms)]
    inst = inst[inst.app.map(ev.app_id).isin(known)]
    cand = np.array(sorted(known))
    day0 = (inst.ts_ms.to_numpy() // DAY_MS) * DAY_MS
    target = inst.app.map(ev.app_id).to_numpy()
    by = {d: (g.ts_ms.to_numpy(), g.app.map(ev.app_id).to_numpy()) for d, g in allinst.groupby("device")}
    owned = [set(by[d][1][by[d][0] < t]) for d, t in zip(inst.device, day0)]
    n_apps = len(pop.q_all)
    recent = {}
    for t in set(day0):  # installs by everyone in the 7 days before the day
        w = allinst[(allinst.ts_ms >= t - 7 * DAY_MS) & (allinst.ts_ms < t)]
        c = np.bincount(w.app.map(ev.app_id), minlength=n_apps) + 20 * pop.q_all
        with np.errstate(divide="ignore"):
            recent[t] = np.log(c / c.sum())
    rec = np.array([recent[t] for t in day0])

    def hits(score):
        n = 0
        for i in range(len(target)):
            sc = score[i, cand].copy()
            sc[[j in owned[i] for j in cand]] = -np.inf
            n += target[i] in cand[np.argsort(-sc, kind="stable")[:k]]
        return round(n / max(len(target), 1), 3)

    out = {"installs": len(target), "k": k, "candidates": len(cand),
           "popularity": hits(np.tile(np.log(pop.q_all + 1e-12), (len(target), 1))),
           "popularity, last 7 days": hits(rec)}
    for name, model in models.items():
        H, _ = states(model, ev, list(zip(inst.device, day0, [None] * len(inst))))
        out[name] = hits(rec + _lift(model, H))
    return out


def _tree_X(imp, extra: dict, drop=()):
    raws = []
    for r in imp.itertuples(index=False):
        d = {k: getattr(r, k) for k in gbm.CATEGORICAL if k != "adv_app"}
        d["adv_app"] = r.adv_bundle
        d.update({k: getattr(r, k) for k in gbm.NUMERIC})
        raws.append(d)
    tr = imp.split.to_numpy() == "train"
    vocab = gbm.build_vocab([r for r, t in zip(raws, tr) if t], min_count=10)
    X = np.array([gbm.encode(r, vocab) for r in raws], float)
    names = list(gbm.CATEGORICAL) + list(gbm.NUMERIC)
    keep = [i for i, n in enumerate(names) if n not in drop]
    X, names = X[:, keep], [names[i] for i in keep]
    for n, v in extra.items():
        X = np.column_stack([X, v])
        names.append(n)
    return X, names


def design_a(imp, variants, truth, log=print):
    """LightGBM on the same rows for each variant, scored on the test days."""
    import lightgbm as lgb
    y, m = deep._labels(imp), deep._masks(imp)
    sp = imp.split.to_numpy()
    tr, te = sp == "train", np.flatnonzero(sp == "test")
    true = {"click": truth.p_click.to_numpy(float), "install_click": truth.p_install_click.to_numpy(float),
            "install_view": truth.p_install_view.to_numpy(float)}
    out, preds = {}, {}
    for name, (extra, drop) in variants.items():
        X, names = _tree_X(imp, extra, drop)
        cats = [c for c in gbm.CATEGORICAL if c in names]
        out[name], preds[name] = {}, {}
        for k in HEADS:
            sel = tr & m[k]
            ds = lgb.Dataset(X[sel], y[k][sel], feature_name=names, categorical_feature=cats)
            b = lgb.train({**gbm.PARAMS, "objective": "binary", "seed": SEED, "deterministic": True,
                           "num_threads": 1, "verbose": -1}, ds, num_boost_round=gbm.ROUNDS)
            p = b.predict(X[te])
            preds[name][k] = p
            out[name][k] = deep._metrics(p, y[k][te], true[k][te], m[k][te])
            if extra and k == "install_view":
                gain = dict(zip(names, b.feature_importance("gain")))
                tot = sum(gain.values()) or 1
                out[name]["install_view_gain_share"] = {n: round(gain[n] / tot, 3) for n in extra}
        log(f"pretrain: trees, {name}")
    return out, preds, te, true, m


def segments(imp, te, preds, true, m, names):
    """Error against the true install-after-view rate, by history length."""
    n_inst = imp.device_prior_installs.fillna(0).to_numpy()[te]
    known = imp.device_known.fillna(0).to_numpy()[te] > 0
    rows = []
    for seg, s in (("all", np.ones(len(te), bool)), ("no prior installs", n_inst == 0),
                   ("1-2 prior installs", (n_inst >= 1) & (n_inst <= 2)), ("3+ prior installs", n_inst >= 3),
                   ("new devices", ~known)):
        k = "install_view"
        mk = s & m[k][te]
        if mk.sum() < 500:
            continue
        t = true[k][te][mk]
        row = {"segment": seg, "rows": int(mk.sum())}
        for n in names:
            row[n] = deep._r(np.abs(preds[n][k][mk] - t).mean() / t.mean(), 3)
        rows.append(row)
    return rows


def run(snapshot_dir="out/snapshot", out_dir="out/pretrain", log=print) -> dict:
    import torch
    torch.use_deterministic_algorithms(True)
    from . import snapshot
    s = snapshot.load(snapshot_dir)
    imp, hist, apps, truth = s["impressions"], s["histories"], s["apps"], s["truth"]
    # pretraining stops where validation starts
    cutoff = int(imp[imp.split != "train"].day.min()) * DAY_MS
    hist = hist[hist.ts_ms < (int(imp.day.max()) + 1) * DAY_MS]
    # an install of an app the device already has is a reinstall: keep the first
    inst = hist.kind.isin(["install", "shared_install"])
    first = ~hist[inst].sort_values("ts_ms", kind="stable").duplicated(["device", "app"])
    hist = pd.concat([hist[~inst], hist[inst].loc[first[first].index]]).sort_values(
        ["device", "ts_ms", "kind"], kind="stable").reset_index(drop=True)
    rep_reinstalls = int((~first).sum())
    ev = Events(hist, apps)
    log(f"pretrain: {len(ev.by_dev):,} devices, {len(hist):,} events, {len(ev.apps)} apps; cutoff day {cutoff // DAY_MS}")
    W, known = coinstall_vectors(hist, ev.app_id, cutoff, 8)
    pre = hist[(hist.ts_ms < cutoff) & hist.kind.isin(["install", "shared_install"])]
    pop = Popularity(pre, ev.app_id, len(ev.apps) + 2)
    rep = {"model": "pretrain", "adtrain_version": __version__, "snapshot": s["manifest"]["days"],
           "cutoff_day": cutoff // DAY_MS, "events": len(hist), "devices": len(ev.by_dev),
           "reinstalls_dropped": rep_reinstalls,
           "installs_per_device": round(len(pre) / max(pre.device.nunique(), 1), 1)}
    models = {}
    for name, warm in (("transformer, warm start", True), ("transformer, cold start", False)):
        models[name], rep[name] = pretrain(ev, hist, cutoff, pop, warm, log=log)
    warm = models["transformer, warm start"]
    end = (int(imp.day.max()) + 1) * DAY_MS
    rep["next_install"] = next_install(models, ev, hist, pop, known, cutoff, end)
    log("pretrain: device states and affinity scores")
    t0 = time.time()
    feats = {}
    for name, model in models.items():
        feats[name], chk = affinity(model, ev, imp, log)
        rep[name]["affinity"] = chk
    rep["state_seconds"] = round(time.time() - t0, 1)
    pooled = pooled_affinity(hist, ev, imp, W, known)
    variants = {
        "1 plain": ({}, HISTORY),
        "2 Part III": ({}, ()),
        "3 + pooled co-install": ({"pooled_affinity": pooled}, ()),
        "4 + pretrained, cold start": ({"affinity": feats["transformer, cold start"]}, ()),
        "5 + pretrained, warm start": ({"affinity": feats["transformer, warm start"]}, ()),
    }
    rep["design_a"], preds, te, true, m = design_a(imp, variants, truth, log)
    rep["segments"] = segments(imp, te, preds, true, m, list(variants))
    rep["within_app"] = within_app(imp, truth, {"pooled co-install": pooled, **feats})
    # the pretrained app table: does it know genres?
    genre = dict(zip(apps.bundle, apps.genre))
    names = [a for a in ev.apps if ev.app_id[a] in known and a in genre]
    rep["genre_neighbors"] = {}
    for name, model in models.items():
        E = model.app.weight.detach().numpy()[[ev.app_id[a] for a in names]]
        rep["genre_neighbors"][name] = _genre(E, [genre[a] for a in names])
    Wk = W[[ev.app_id[a] for a in names]]
    rep["genre_neighbors"]["co-install vectors"] = _genre(Wk, [genre[a] for a in names])
    _save(out_dir, warm, ev, imp, rep, s)
    return rep


def within_app(imp, truth, feats: dict):
    """Rank correlation of each signal with the true install-after-view
    rate within each advertised app, averaged over apps: the trees already
    know the app, so only what varies across devices can help them."""
    te = imp.split.to_numpy() == "test"
    y = truth.p_install_view.to_numpy(float)
    out = {}
    for name, x in feats.items():
        d = pd.DataFrame({"app": imp.adv_bundle.to_numpy(), "x": x, "y": y})[te].dropna()
        cs = [g.x.corr(g.y, method="spearman") for _, g in d.groupby("app") if len(g) >= 300]
        out[name] = deep._r(float(np.nanmean(cs)), 3)
    return out


def _genre(E, g, k=5):
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    g = np.array(g)
    same = np.mean([(g[np.argsort(-S[i])[:k]] == g[i]).mean() for i in range(len(g))])
    base = np.mean([(g[np.arange(len(g)) != i] == g[i]).mean() for i in range(len(g))])
    return {"apps": len(g), "k": k, "same_genre_neighbors": deep._r(same, 3), "same_genre_random": deep._r(base, 3)}


def _save(out_dir, model, ev, imp, rep, s):
    """The app vectors and daily device states, stamped with the model's
    version: a dot product between vectors of two versions means nothing."""
    import torch
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.numpy().tobytes())
    version = "pre-" + h.hexdigest()[:10]
    rep["version"] = version
    E = model.app.weight.detach().numpy()
    pd.DataFrame({"version": version, "app": ev.apps, **{f"e{i}": E[2:, i] for i in range(DIM)}}) \
        .to_parquet(out / "app_vectors.parquet", index=False)
    dd = imp[["device", "day"]].drop_duplicates().sort_values(["day", "device"])
    H, _ = states(model, ev, list(zip(dd.device, dd.day * DAY_MS, [None] * len(dd))))
    st = pd.DataFrame({"version": version, "device": dd.device.to_numpy(), "day": dd.day.to_numpy(),
                       **{f"h{i}": H[:, i] for i in range(DIM)}})
    st.to_parquet(out / "device_states.parquet", index=False)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    rep["out_dir"] = str(out)


def print_report(r: dict) -> None:
    w = r["transformer, warm start"]
    print(f"Pretraining, {w['params']:,} parameters: {r['devices']:,} devices, {w['targets']:,} next-install targets "
          f"before day {r['cutoff_day']}, {r['installs_per_device']} installs per device ({r['reinstalls_dropped']:,} "
          f"reinstalls dropped); trained in {w['seconds']} s "
          f"on the CPU")
    print(f"  next-install loss: warm start {w['loss']}, cold start {r['transformer, cold start']['loss']}, "
          f"popularity alone {w['loss_popularity_only']}")
    n = r["next_install"]
    print(f"  next install after the cutoff, {n['installs']:,} installs: share in each scorer's top {n['k']} of "
          f"{n['candidates']} apps")
    for k_ in ("popularity", "popularity, last 7 days", "transformer, cold start", "transformer, warm start"):
        print(f"    {k_:<28}{n[k_]:.0%}")
    print("  app vectors: share of each app's 5 nearest neighbors with its genre")
    for k_, g in r["genre_neighbors"].items():
        print(f"    {k_:<28}{g['same_genre_neighbors']:.0%} (random {g['same_genre_random']:.0%}, {g['apps']} apps)")
    print("  within each advertised app, rank correlation with the true install-after-view rate:")
    for k_, v in r["within_app"].items():
        print(f"    {k_:<28}{v:+.2f}")
    a = w["affinity"]
    print(f"  device states: {a['device_days']:,} device-days ({a['masked']:,} with the advertised app masked), {a['with_history']:,} with history, computed in "
          f"{r['state_seconds']} s; events on or after the impression's day in any state: {a['leaks']}")
    print("  Design A, LightGBM on the test days (error and calibration against the true rates):")
    print(f"    {'variant':<30}{'install after view':>24}{'install after click':>24}{'click':>16}")
    print(f"    {'':<30}{'AUC   calib.  error':>24}{'AUC   calib.  error':>24}{'AUC   error':>16}")
    for name, h in r["design_a"].items():
        iv, ic, c = h["install_view"], h["install_click"], h["click"]
        print(f"    {name:<30}{iv['auc']:>8.3f}{iv['calibration_vs_truth']:>8.3f}{iv['error_vs_truth']:>8.3f}"
              f"{ic['auc']:>8.3f}{ic['calibration_vs_truth']:>8.3f}{ic['error_vs_truth']:>8.3f}"
              f"{c['auc']:>8.3f}{c['error_vs_truth']:>8.3f}")
    for name, h in r["design_a"].items():
        if "install_view_gain_share" in h:
            print(f"    {name}: share of the install-after-view trees' gain: " +
                  ", ".join(f"{k_} {v:.0%}" for k_, v in h["install_view_gain_share"].items()))
    print("  error against the true install-after-view rate, by device history:")
    names = list(r["design_a"])
    print(f"    {'segment':<22}{'rows':>8}" + "".join(f"{n_[:2]:>7}" for n_ in names))
    for s_ in r["segments"]:
        print(f"    {s_['segment']:<22}{s_['rows']:>8,}" + "".join(f"{s_[n_]:>7.3f}" for n_ in names))
    print(f"  wrote app vectors, device states, and the report to {r.get('out_dir')} (version {r['version']})")
