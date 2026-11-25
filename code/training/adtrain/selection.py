"""Learned campaign selection on the Part IV dataset (Chapter 35).

The teacher is the full pipeline: LightGBM click and install models, trained
on the training days, score every eligible campaign on a sample of logged
requests (eligible: the campaign has started and targets the request's
country), and each campaign's value per impression follows from its goal:

    CPI      p(install) x target CPI
    CPC      p(click) x target CPC
    ROAS     p(install) x the app's revenue per install / target ROAS

The student is a two-tower bid predictor: a user tower over the request and
the device's history, and a campaign tower over the campaign's ID (randomly
dropped in training), advertised app, goal, format, and target, whose dot
product plus a bias predicts the log of the teacher's value. It's trained
with a listwise loss: a softmax over each request's eligible campaigns,
against the teacher's values.

On held-out days, retrieval is judged by recall at k, the share of requests
whose teacher winner is in the top k, and value recall, against:

- the walk candidate cap: each campaign's average value in the request's
  segment (country and placement), the same for every device;
- install-affinity retrieval: the device's installs, as co-install vectors,
  against the campaign's app;
- the predictor and install affinity merged, 80% and 20% of the slots;
- random.

Campaign 9 starts on day 9, after the training days, so neither model has
seen it: it's the cold-start check.

    python -m adtrain select --snapshot out/snapshot

(The module is selection.py, so it doesn't shadow Python's select module.)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, gbm

SEED = 7
TRAIN_REQUESTS, TEST_REQUESTS = 20_000, 5_000
DIM, EPOCHS, BATCH, LR = 16, 12, 256, 3e-3
ID_DROP = 0.3
AFFINITY_SHARE = 0.2
KS = (1, 2, 3, 5, 8)
SEG_K = 2  # the segments compare recall at 2, where the methods still differ
CAMPAIGN_FIELDS = ("campaign", "creative", "creative_format", "goal", "adv_app")
USER_CATS = ("placement", "country", "os", "exchange", "connection_type")
USER_NUMS = ("device_known", "device_requests", "seconds_since_last_seen", "device_prior_installs",
             "our_imps_today", "bidfloor")
UNK = 0


# ------------------------------------------------------------------ teacher

def _campaign_book(imp, camps):
    """Each campaign's fields as the models see them, from its impressions."""
    rows = []
    for cid, g in imp.groupby("campaign_id"):
        top = g.groupby(["creative", "creative_format"]).size().idxmax()
        r = g.iloc[0]
        c = camps.set_index("campaign_id").loc[cid]
        rows.append({"campaign_id": int(cid), "campaign": str(r.campaign), "creative": str(top[0]),
                     "creative_format": str(top[1]), "goal": str(r.goal), "adv_bundle": r.adv_bundle,
                     "target": float(c.target), "geos": set(str(c.geos).lower().split(",")),
                     "start_day": int(c.start_day), "goal_type": str(c.goal)})
    return pd.DataFrame(rows).sort_values("campaign_id").reset_index(drop=True)


def _raw(r, c=None):
    d = {k: getattr(r, k) for k in gbm.CATEGORICAL if k not in ("adv_app",)}
    d["adv_app"] = r.adv_bundle
    d.update({k: getattr(r, k) for k in gbm.NUMERIC})
    if c is not None:  # the same request, for another campaign
        d.update(campaign=c.campaign, creative=c.creative, creative_format=c.creative_format,
                 goal=c.goal, adv_app=c.adv_bundle)
    return d


def teacher(imp, book, log=print):
    """LightGBM click and install models trained on the training days."""
    import lightgbm as lgb
    tr = imp[imp.split == "train"]
    raws = [_raw(r) for r in tr.itertuples(index=False)]
    vocab = gbm.build_vocab(raws, min_count=10)
    X = np.array([gbm.encode(r, vocab) for r in raws], float)
    names = list(gbm.CATEGORICAL) + list(gbm.NUMERIC)
    c, fin = tr.click.to_numpy(), tr.install_final.to_numpy(bool)
    heads = {"click": (np.ones(len(tr), bool), c), "install_click": (fin & (c == 1), tr.install_click.to_numpy()),
             "install_view": (fin & (c == 0), tr.install_view.to_numpy())}
    models = {}
    for k, (m, y) in heads.items():
        ds = lgb.Dataset(X[m], y[m], feature_name=names, categorical_feature=list(gbm.CATEGORICAL))
        models[k] = lgb.train({**gbm.PARAMS, "objective": "binary", "seed": SEED, "deterministic": True,
                               "num_threads": 1, "verbose": -1}, ds, num_boost_round=gbm.ROUNDS)
    # revenue per install for ROAS goals: D3 revenue of the app's final installs, shrunk to the mean
    ins = tr[(tr.install == 1) & tr.d3_final]
    g = ins.groupby("adv_bundle").d3_revenue.agg(["sum", "count"])
    prior = float(ins.d3_revenue.mean()) if len(ins) else 0.1
    rev = ((g["sum"] + 5 * prior) / (g["count"] + 5)).to_dict()
    book = book.assign(rev_per_install=[rev.get(a, prior) for a in book.adv_bundle])
    log(f"select: teacher trained on {len(tr):,} impressions")
    return models, vocab, book


def eligible(req, book):
    """Campaigns that have started and target the request's country."""
    day, country = req.day.to_numpy(), req.country.astype(str).str.lower().to_numpy()
    E = np.zeros((len(req), len(book)), bool)
    for j, c in enumerate(book.itertuples(index=False)):
        E[:, j] = (day >= c.start_day) & np.isin(country, list(c.geos))
    return E


def score(req, book, E, models, vocab):
    """The teacher's value per impression for every eligible campaign: a
    matrix of requests by campaigns, NaN where not eligible."""
    rows, idx = [], []
    reqs = list(req.itertuples(index=False))
    camps = list(book.itertuples(index=False))
    for i, j in zip(*np.nonzero(E)):
        rows.append(gbm.encode(_raw(reqs[i], camps[j]), vocab))
        idx.append((i, j))
    X = np.array(rows, float)
    pc = models["click"].predict(X)
    pi = pc * models["install_click"].predict(X) + (1 - pc) * models["install_view"].predict(X)
    V = np.full(E.shape, np.nan)
    for (i, j), c, ins in zip(idx, pc, pi):
        b = camps[j]
        if b.goal_type == "CPC":
            V[i, j] = c * b.target
        elif b.goal_type == "CPI":
            V[i, j] = ins * b.target
        else:  # ROAS and Ad ROAS: revenue per impression over the target return
            V[i, j] = ins * b.rev_per_install / max(b.target, 1e-6)
    return V


# ------------------------------------------------------------------ student

class Enc:
    """Vocabularies for the towers: one shared app table, and one per field."""

    def __init__(self, req, book):
        apps = set(req.publisher_app.astype(str)) | set(book.adv_bundle) | {a for xs in req.recent_apps for a in xs}
        self.app = {a: i + 1 for i, a in enumerate(sorted(apps))}
        self.cat = {f: {v: i + 1 for i, v in enumerate(sorted(set(req[f].astype(str))))} for f in USER_CATS}
        self.goal = {g: i + 1 for i, g in enumerate(sorted(set(book.goal_type)))}
        self.fmt = {g: i + 1 for i, g in enumerate(sorted(set(book.creative_format)))}

    def user(self, req, L=20):
        cats = np.column_stack([[self.cat[f].get(str(v), UNK) for v in req[f]] for f in USER_CATS]).astype(np.int64)
        pub = np.array([self.app.get(str(a), UNK) for a in req.publisher_app], np.int64)
        rec = np.zeros((len(req), L), np.int64)
        for i, xs in enumerate(req.recent_apps):
            ids = [self.app.get(a, UNK) for a in list(xs)[:L]]
            rec[i, :len(ids)] = ids
        num = np.column_stack([np.log1p(np.clip(req[k].fillna(0).to_numpy(float), 0, None)) for k in USER_NUMS])
        h = req.hour_of_day.fillna(0).to_numpy(float)
        num = np.column_stack([num, np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)]).astype(np.float32)
        return cats, pub, rec, num

    def campaigns(self, book, known_ids):
        ids = np.array([(cid + 1) if cid in known_ids else UNK for cid in book.campaign_id], np.int64)
        app = np.array([self.app.get(a, UNK) for a in book.adv_bundle], np.int64)
        goal = np.array([self.goal.get(g, UNK) for g in book.goal_type], np.int64)
        fmt = np.array([self.fmt.get(f, UNK) for f in book.creative_format], np.int64)
        tgt = np.log(book.target.to_numpy(float)).astype(np.float32)
        return ids, app, goal, fmt, tgt


def _model(enc, n_campaigns, n_num):
    import torch
    from torch import nn

    class TwoTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.app = nn.Embedding(len(enc.app) + 1, DIM, padding_idx=0)
            self.cats = nn.ModuleList([nn.Embedding(len(enc.cat[f]) + 1, 8) for f in USER_CATS])
            self.user = nn.Sequential(nn.Linear(8 * len(USER_CATS) + 2 * DIM + n_num, 64), nn.ReLU(),
                                      nn.Linear(64, DIM))
            self.cid = nn.Embedding(n_campaigns + 1, DIM)  # row 0: unknown, or dropped
            self.goal = nn.Embedding(len(enc.goal) + 1, 4)
            self.fmt = nn.Embedding(len(enc.fmt) + 1, 4)
            self.camp = nn.Sequential(nn.Linear(2 * DIM + 4 + 4 + 1, 64), nn.ReLU(), nn.Linear(64, DIM))
            self.bias = nn.Embedding(n_campaigns + 1, 1)
            self.goal_bias = nn.Embedding(len(enc.goal) + 1, 1)

        def u(self, cats, pub, rec, num):
            mask = (rec > 0).float().unsqueeze(-1)
            pooled = (self.app(rec) * mask).sum(1) / mask.sum(1).clamp(min=1)
            parts = [e(cats[:, i]) for i, e in enumerate(self.cats)] + [self.app(pub), pooled, num]
            return self.user(torch.cat(parts, 1))

        def v(self, ids, app, goal, fmt, tgt):
            x = torch.cat([self.cid(ids), self.app(app), self.goal(goal), self.fmt(fmt), tgt[:, None]], 1)
            return self.camp(x), (self.bias(ids) + self.goal_bias(goal)).squeeze(-1)

    return TwoTower()


def train_student(enc, U, C, V, E, log=print):
    """Listwise: a softmax over each request's eligible campaigns, against the
    teacher's values (a softmax of log value is proportional to value)."""
    import torch
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    cats, pub, rec, num = (torch.from_numpy(x) for x in U)
    ids, app, goal, fmt, tgt = (torch.from_numpy(x) for x in C)
    logv = np.where(E, np.log(np.maximum(np.nan_to_num(V, nan=1e-12), 1e-12)), -np.inf)
    target = torch.softmax(torch.from_numpy(logv).float(), 1)
    mask = torch.from_numpy(E)
    model = _model(enc, len(ids), num.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t0 = time.time()
    for ep in range(EPOCHS):
        order = rng.permutation(len(num))
        tot = 0.0
        for s in range(0, len(order), BATCH):
            b = torch.from_numpy(order[s:s + BATCH])
            drop = torch.from_numpy(rng.random(len(ids)) < ID_DROP)
            vv, bias = model.v(torch.where(drop, 0, ids), app, goal, fmt, tgt)
            z = model.u(cats[b], pub[b], rec[b], num[b]) @ vv.T + bias
            z = z.masked_fill(~mask[b], -1e9)
            loss = -(target[b] * torch.log_softmax(z, 1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        log(f"select: epoch {ep + 1}, listwise loss {tot / len(order):.4f}")
    return model, round(time.time() - t0, 1)


def student_scores(model, U, C):
    import torch
    with torch.no_grad():
        u = model.u(*(torch.from_numpy(x) for x in U))
        v, b = model.v(*(torch.from_numpy(x) for x in C))
        return (u @ v.T + b).numpy(), u.numpy(), v.numpy()


# ------------------------------------------------------------------ baselines

def segment_scores(train_req, train_V, train_E, test_req, book):
    """The walk cap: each campaign's average log value in the request's
    segment, country and placement, over the training requests."""
    logv = np.where(train_E, np.log(np.maximum(np.nan_to_num(train_V, nan=1e-12), 1e-12)), np.nan)
    seg = (train_req.country.astype(str) + "|" + train_req.placement.astype(str)).to_numpy()
    df = pd.DataFrame(logv, columns=book.campaign_id).assign(seg=seg)
    by_seg = df.groupby("seg").mean()
    overall = df.drop(columns="seg").mean()
    goal_mean = pd.Series(overall.to_numpy(), index=book.goal_type).groupby(level=0).mean()
    test_seg = (test_req.country.astype(str) + "|" + test_req.placement.astype(str)).to_numpy()
    out = np.zeros((len(test_req), len(book)))
    for j, c in enumerate(book.itertuples(index=False)):
        col = by_seg[c.campaign_id] if c.campaign_id in by_seg else pd.Series(dtype=float)
        fb = overall.get(c.campaign_id, np.nan)
        if np.isnan(fb):  # a campaign the cap has never seen: its goal's average
            fb = goal_mean.get(c.goal_type, overall.mean())
        out[:, j] = pd.Series(test_seg).map(col).fillna(fb).to_numpy()
    return out


def affinity_scores(hist, test_req, book, cutoff_ms):
    """Install affinity: the mean co-install vector of the device's installs
    before the request's day, against the campaign's app (cosine). Devices
    without installs get the apps' install counts."""
    from .pretrain import coinstall_vectors
    apps = sorted(set(hist.app) | set(book.adv_bundle))
    app_id = {a: i + 2 for i, a in enumerate(apps)}
    W, known = coinstall_vectors(hist, app_id, cutoff_ms, 8)
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    inst = hist[hist.kind.isin(["install", "shared_install"])].sort_values("ts_ms")
    pop = inst[inst.ts_ms < cutoff_ms].app.value_counts()
    by = {d: (g.ts_ms.to_numpy(), g.app.map(app_id).to_numpy()) for d, g in inst.groupby("device")}
    capp = np.array([app_id[a] for a in book.adv_bundle])
    popv = np.log1p(np.array([pop.get(a, 0) for a in book.adv_bundle], float)) * 1e-3
    out = np.zeros((len(test_req), len(book)))
    has = np.zeros(len(test_req), bool)
    for i, (d, day) in enumerate(zip(test_req.device.to_numpy(), test_req.day.to_numpy())):
        out[i] = popv
        if d in by:
            ts, ap = by[d]
            ap = [x for x in ap[ts < day * 86_400_000] if x in known]
            if ap:
                m = Wn[ap].mean(0)
                out[i] += Wn[capp] @ m / (np.linalg.norm(m) + 1e-9)
                has[i] = True
    return out, has


# ------------------------------------------------------------------ evaluation

def _topk(S, E, k):
    S = np.where(E, S, -np.inf)
    return np.argsort(-S, axis=1, kind="stable")[:, :k]


def merged(S_pred, S_aff, E, k):
    """The top slots from the predictor, the rest from install affinity,
    without duplicates."""
    n_aff = 0 if k < 2 else max(1, round(k * AFFINITY_SHARE))
    a, b = _topk(S_pred, E, len(E[0])), _topk(S_aff, E, len(E[0]))
    out = np.zeros((len(E), k), int)
    for i in range(len(E)):
        top = list(a[i, :k - n_aff])
        for j in b[i]:
            if len(top) >= k:
                break
            if j not in top and E[i, j]:
                top.append(j)
        for j in a[i]:
            if len(top) >= k:
                break
            if j not in top:
                top.append(j)
        out[i] = top[:k]
    return out


def recall(top_fn, V, E, rows=None):
    """Recall at k against the teacher's winner, and value recall: the best
    retrieved value over the best value."""
    rows = np.arange(len(V)) if rows is None else rows
    Vn = np.where(E, np.nan_to_num(V, nan=0), 0)
    win = Vn.argmax(1)
    best = Vn.max(1)
    out = {}
    for k in KS:
        T = top_fn(k)[rows]
        hit = (T == win[rows, None]).any(1)
        got = np.take_along_axis(Vn[rows], T, 1).max(1)
        out[k] = {"recall": round(float(hit.mean()), 3),
                  "value_recall": round(float((got / np.maximum(best[rows], 1e-12)).mean()), 3)}
    return out


def run(snapshot_dir="out/snapshot", out_dir=None, log=print) -> dict:
    import torch
    torch.use_deterministic_algorithms(True)
    from . import snapshot
    s = snapshot.load(snapshot_dir)
    imp, camps, hist = s["impressions"], s["campaigns"], s["histories"]
    rng = np.random.default_rng(SEED)
    book = _campaign_book(imp, camps)
    t0 = time.time()
    models, vocab, book = teacher(imp, book, log)
    tr_days = imp.split.to_numpy() == "train"
    te_days = imp.split.to_numpy() == "test"
    train_req = imp.iloc[np.sort(rng.choice(np.flatnonzero(tr_days), TRAIN_REQUESTS, replace=False))].reset_index(drop=True)
    test_req = imp.iloc[np.sort(rng.choice(np.flatnonzero(te_days), TEST_REQUESTS, replace=False))].reset_index(drop=True)
    E_tr, E_te = eligible(train_req, book), eligible(test_req, book)
    V_tr, V_te = score(train_req, book, E_tr, models, vocab), score(test_req, book, E_te, models, vocab)
    teacher_s = round(time.time() - t0, 1)
    log(f"select: teacher scored {int(E_tr.sum() + E_te.sum()):,} request-campaign pairs in {teacher_s} s")
    enc = Enc(train_req, book)
    known = set(book.campaign_id[E_tr.any(0)])
    C = enc.campaigns(book, known)
    model, student_s = train_student(enc, enc.user(train_req), C, V_tr, E_tr, log)
    S_pred, U_te, Vc = student_scores(model, enc.user(test_req), C)
    S_seg = segment_scores(train_req, V_tr, E_tr, test_req, book)
    cutoff = int(imp[imp.split != "train"].day.min()) * 86_400_000
    S_aff, has_hist = affinity_scores(hist, test_req, book, cutoff)
    S_rand = rng.random(S_pred.shape)
    methods = {"bid predictor": lambda k: _topk(S_pred, E_te, k),
               "predictor + install affinity": lambda k: merged(S_pred, S_aff, E_te, k),
               "walk cap (segment value)": lambda k: _topk(S_seg, E_te, k),
               "install affinity": lambda k: _topk(S_aff, E_te, k),
               "random": lambda k: _topk(S_rand, E_te, k)}
    Vn = np.where(E_te, np.nan_to_num(V_te, nan=0), 0)
    win = Vn.argmax(1)
    new_ids = [j for j, c in enumerate(book.campaign_id) if c not in known]
    segments = {"all requests": np.arange(len(test_req)),
                "the winner is the new campaign": np.flatnonzero(np.isin(win, new_ids)),
                "devices with no installs": np.flatnonzero(~has_hist)}
    for g in sorted(set(book.goal_type)):
        segments[f"the winner's goal is {g}"] = np.flatnonzero(book.goal_type.to_numpy()[win] == g)
    rep = {"model": "select", "adtrain_version": __version__, "snapshot": s["manifest"]["days"],
           "campaigns": len(book), "new_campaigns": [int(book.campaign_id[j]) for j in new_ids],
           "train_requests": len(train_req), "test_requests": len(test_req),
           "eligible_per_request": round(float(E_te.sum(1).mean()), 1), "teacher_seconds": teacher_s,
           "student_seconds": student_s, "params": int(sum(p.numel() for p in model.parameters())),
           "recall": {}, "segments": {}}
    for name, f in methods.items():
        rep["recall"][name] = recall(f, V_te, E_te)
    for seg, rows in segments.items():
        if len(rows) < 30:
            continue
        rep["segments"][seg] = {"requests": int(len(rows)),
                                **{m: recall(methods[m], V_te, E_te, rows)[SEG_K]["recall"]
                                   for m in ("bid predictor", "predictor + install affinity",
                                             "walk cap (segment value)", "install affinity")}}
    rep["winner_goal_share"] = {g: round(float((book.goal_type.to_numpy()[win] == g).mean()), 3)
                                for g in sorted(set(book.goal_type))}
    # cold start: where each method ranks the new campaigns, against the teacher's rank
    rep["new_campaign_rank"] = {}
    for j in new_ids:
        e = E_te[:, j]
        if e.sum() < 30:
            continue
        rk = lambda S_: (np.argsort(np.argsort(-np.where(E_te, S_, -np.inf), 1), 1)[e, j] + 1)
        row = {"requests": int(e.sum()), "eligible": round(float(E_te[e].sum(1).mean()), 1),
               "teacher": round(float(rk(Vn).mean()), 2)}
        for name, S_ in (("bid predictor", S_pred), ("walk cap (segment value)", S_seg),
                         ("install affinity", S_aff)):
            row[name] = round(float(rk(S_).mean()), 2)
            row[name + " within 1"] = round(float((np.abs(rk(S_) - rk(Vn)) <= 1).mean()), 3)
        rep["new_campaign_rank"][str(int(book.campaign_id[j]))] = row
    # how well the predictor's order matches the teacher's on each request
    rho = []
    for i in range(len(test_req)):
        e = E_te[i]
        if e.sum() >= 3:
            rho.append(pd.Series(S_pred[i, e]).corr(pd.Series(np.log(Vn[i, e] + 1e-12)), method="spearman"))
    rep["rank_correlation"] = round(float(np.nanmean(rho)), 3)
    # install recall: the campaign a device installed from, among its top k
    inst = imp[te_days & (imp.install == 1)].reset_index(drop=True)
    if len(inst):
        E_i = eligible(inst, book)
        col = {c: j for j, c in enumerate(book.campaign_id)}
        got = np.array([col[c] for c in inst.campaign_id])
        S_pi, _, _ = student_scores(model, enc.user(inst), C)
        S_ai, _ = affinity_scores(hist, inst, book, cutoff)
        rep["install_recall"] = {"installs": int(len(inst))}
        for name, S in (("bid predictor", S_pi), ("install affinity", S_ai)):
            T = _topk(S, E_i, 3)
            rep["install_recall"][name] = round(float((T == got[:, None]).any(1).mean()), 3)
    out = Path(out_dir or Path(snapshot_dir).parent / "select")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"campaign_id": book.campaign_id, "goal": book.goal_type, "adv_bundle": book.adv_bundle,
                  "bias": None, **{f"v{i}": Vc[:, i] for i in range(Vc.shape[1])}}).drop(columns="bias") \
        .to_parquet(out / "campaign_vectors.parquet", index=False)
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    rep["out_dir"] = str(out)
    return rep


def print_report(r: dict) -> None:
    print(f"Learned campaign selection: {r['campaigns']} campaigns, {r['eligible_per_request']} eligible per request; "
          f"the teacher scored {r['train_requests']:,} training and {r['test_requests']:,} test requests in "
          f"{r['teacher_seconds']} s; the two-tower predictor ({r['params']:,} parameters) trained in "
          f"{r['student_seconds']} s")
    print(f"  the predictor's order against the teacher's, rank correlation per request: {r['rank_correlation']:.2f}")
    print("  recall at k: the share of test requests whose teacher winner is in the top k (value recall in brackets)")
    print(f"    {'':<32}" + "".join(f"{'k=' + str(k):>15}" for k in KS))
    at = lambda rec, k: rec[k] if k in rec else rec[str(k)]
    for name, rec in r["recall"].items():
        print(f"    {name:<32}" + "".join(f"{at(rec, k)['recall']:>8.0%} ({at(rec, k)['value_recall']:.2f})" for k in KS))
    print("  the teacher's winner, by goal: " + ", ".join(f"{g} {v:.0%}" for g, v in r["winner_goal_share"].items()))
    print(f"  recall at {SEG_K}, by segment:")
    segs = r["segments"]
    names = ("bid predictor", "predictor + install affinity", "walk cap (segment value)", "install affinity")
    print(f"    {'segment':<36}{'requests':>9}" + "".join(f"{n[:16]:>18}" for n in names))
    for seg, row in segs.items():
        print(f"    {seg:<36}{row['requests']:>9,}" + "".join(f"{row[n]:>18.0%}" for n in names))
    for cid, row in r.get("new_campaign_rank", {}).items():
        print(f"  new campaign {cid}, unseen in training, on {row['requests']:,} requests ({row['eligible']} eligible): "
              f"average rank by the teacher {row['teacher']}, by the bid predictor {row['bid predictor']} "
              f"({row['bid predictor within 1']:.0%} within one place), by the walk cap {row['walk cap (segment value)']}, "
              f"by install affinity {row['install affinity']}")
    ir = r.get("install_recall")
    if ir:
        print(f"  install recall at 3, {ir['installs']} installs on the test days: the installed campaign in the device's "
              f"top 3: bid predictor {ir['bid predictor']:.0%}, install affinity {ir['install affinity']:.0%}")
    print(f"  wrote the campaign vectors and the report to {r.get('out_dir')}")
