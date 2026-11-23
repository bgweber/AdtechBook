"""Gradient-boosted conversion models with LightGBM (Chapter 15).

Trains three binary models on a conversion dataset (`adtrain dataset`):

- click: did the impression get a click (every impression);
- install_click: an install credited to the click (impressions with a click);
- install_view: a view-through install (impressions without a click).

A purchase model needs far more installs than a small simulated market
makes, so it's trained only when there are enough purchases.

Features come from the FeatureVector and ad the bidder logged, plus the
device's earlier conversions. Strings become categories through a
vocabulary stored with the model: values seen fewer than `min_count` times
in training, and values never seen, share one "other" code, so the Go
bidder maps them the same way. The bundle holds each model's trees (from
LightGBM's `dump_model`), the vocabulary, golden examples for the parity
check, and a report.

    python -m adtrain gbm --dataset out/datasets/conversions/conversions-0001
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import __version__, datasets, metrics, simout

CATEGORICAL = ("exchange", "placement", "country", "os", "os_major", "make",
               "model", "connection_type", "publisher_app", "campaign",
               "creative", "creative_format", "adv_app", "goal")
NUMERIC = ("bidfloor", "hour_of_day", "rewarded", "interstitial",
           "device_known", "device_requests", "seconds_since_first_seen",
           "seconds_since_last_seen", "our_imps_today",
           "seconds_since_our_last_imp", "device_prior_installs",
           "device_prior_purchases")
# logged only when nonzero, so missing means zero
ZERO_IF_MISSING = ("device_prior_installs", "device_prior_purchases")
OTHER = "__other__"

TARGETS = {
    "click": ("click", "click_final", None),
    "install_click": ("install_click", "install_final", "click"),
    "install_view": ("install_view", "install_final", "no_click"),
}
# installs are rare, so their models are evaluated on more days
EVAL_DAYS = {"click": 1, "install_click": 3, "install_view": 3}
# Few positives: smaller trees, and no features that measure how long the
# DSP has known a device. Those drift while the market is young: every
# device is new in the first days, so a model trained then learns that
# new devices install, and underpredicts once they aren't (Chapter 15).
SMALL = {"num_leaves": 7, "min_data_in_leaf": 100}
AGE_FEATURES = ("seconds_since_first_seen", "seconds_since_last_seen",
                "device_requests")
TARGET_PARAMS = {"install_click": SMALL, "install_view": SMALL}
TARGET_EXCLUDE = {"install_click": AGE_FEATURES, "install_view": AGE_FEATURES}

PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15,
          "min_data_in_leaf": 50, "feature_fraction": 0.8,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "verbose": -1, "seed": 0, "deterministic": True,
          "num_threads": 1, "max_cat_to_onehot": 4, "cat_smooth": 10}
ROUNDS = 200


def raw_values(fv: dict, ad: dict) -> dict:
    """The model's inputs from one FeatureVector and ad, before encoding."""
    s = fv.get("strings") or {}
    n = fv.get("numeric") or {}
    out = {}
    for k in CATEGORICAL:
        v = ad.get(k) if k in ("campaign", "creative", "creative_format",
                               "adv_app", "goal") else s.get(k)
        out[k] = "" if v is None else str(v)
    for k in NUMERIC:
        if k in n:
            out[k] = float(n[k])
        else:
            out[k] = 0.0 if k in ZERO_IF_MISSING else math.nan
    return out


def encode(raw: dict, vocab: dict) -> list[float]:
    """One row of model inputs: category codes, then numbers."""
    row = []
    for k in CATEGORICAL:
        codes = vocab[k]
        row.append(float(codes.get(raw[k], codes[OTHER])))
    for k in NUMERIC:
        row.append(raw[k])
    return row


def columns(df):
    """The model inputs for every dataset row, as one array per categorical
    feature (interned strings) and one matrix of numbers, with the
    dataset's device history (as of each bid, or leaky as of the build)
    in place of what was logged, so a backfill reaches every row."""
    import sys
    n = len(df)
    cats = {k: np.empty(n, dtype=object) for k in CATEGORICAL}
    nums = np.empty((n, len(NUMERIC)), float)
    for i, (f, a) in enumerate(zip(df.features, df.ad)):
        r = raw_values(json.loads(f), json.loads(a))
        for k in CATEGORICAL:
            cats[k][i] = sys.intern(r[k])
        nums[i] = [r[k] for k in NUMERIC]
    nums[:, NUMERIC.index("device_prior_installs")] = df.device_prior_installs.to_numpy(float)
    nums[:, NUMERIC.index("device_prior_purchases")] = df.device_prior_purchases.to_numpy(float)
    return cats, nums


def vocab_from_columns(cats, mask=None, min_count: int = 10) -> dict:
    """build_vocab, on columns: the same codes, in the same order."""
    import pandas as pd
    vocab = {}
    for k in CATEGORICAL:
        col = cats[k] if mask is None else cats[k][mask]
        counts = pd.Series(col).value_counts()
        kept = sorted((v for v, c in counts.items() if c >= min_count),
                      key=lambda v: (-counts[v], v))
        codes = {v: i for i, v in enumerate(kept)}
        codes[OTHER] = len(kept)
        vocab[k] = codes
    return vocab


def encode_columns(cats, nums, vocab) -> np.ndarray:
    import pandas as pd
    cols = [pd.Series(cats[k]).map(vocab[k]).fillna(vocab[k][OTHER]).to_numpy(float)
            for k in CATEGORICAL]
    return np.column_stack(cols + [nums])


def build_vocab(raws: list[dict], min_count: int = 10) -> dict:
    vocab = {}
    for k in CATEGORICAL:
        counts = {}
        for r in raws:
            counts[r[k]] = counts.get(r[k], 0) + 1
        kept = sorted((v for v, c in counts.items() if c >= min_count),
                      key=lambda v: (-counts[v], v))
        codes = {v: i for i, v in enumerate(kept)}
        codes[OTHER] = len(kept)
        vocab[k] = codes
    return vocab


def train(dataset_dir, models_dir, sim_out=None, eval_day=None,
          min_purchases: int = 30, inject: str | None = None,
          log=print) -> tuple[str, dict]:
    """Train the conversion models on a dataset and publish a bundle.

    inject="no-features" trains as if the feature pipeline had broken and
    every input were missing: the models learn only the base rate. The
    pipeline's checks should stop it (Chapter 18)."""
    import lightgbm as lgb

    df, manifest = datasets.read(dataset_dir)
    if not manifest.get("passed", False):
        log("note: this dataset failed validation; training anyway, but the "
            "model's report says so")
    # inputs column by column: a parsed dict per row would need several
    # GB for a few hundred thousand impressions
    cats, nums = columns(df)
    tt = simout.table(sim_out, "truth", "impressions",
                      ["impression_id", "p_click", "p_install_click", "p_install_view"])
    truth = tt.set_index("impression_id") if tt is not None else None
    report = {"dataset": str(dataset_dir), "dataset_version": manifest.get("version"),
              "dataset_passed": manifest.get("passed"),
              "leaky_device_history": manifest["settings"].get("leak", False),
              "lineage": _lineage(dataset_dir, manifest, inject),
              "models": {}}
    trained = {}
    # one vocabulary for all models, from every day but the newest
    last = int(df.day.max())
    early = df.day.to_numpy() < last
    vocab = vocab_from_columns(cats, early if early.any() else None)
    X = encode_columns(cats, nums, vocab)
    del cats, nums
    if inject == "no-features":
        log("injected: every feature missing (Chapter 18 exercise)")
        X[:] = np.nan
    elif inject:
        raise ValueError(f"unknown injection {inject!r}")
    for name, (label, final, subset) in TARGETS.items():
        mask = df[final].to_numpy()
        if subset == "click":
            mask = mask & (df.click.to_numpy() == 1)
        elif subset == "no_click":
            mask = mask & (df.click.to_numpy() == 0)
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            log(f"{name}: no rows with final labels yet")
            continue
        dd = df.day.to_numpy()
        if eval_day is not None:
            last_ev = eval_day
        else:
            # the newest day whose labels are (nearly) all final
            share = df[final].groupby(df.day).mean()
            done = [int(d) for d, v in share.items() if v >= 0.99]
            if not done:
                log(f"{name}: no day has final labels yet")
                continue
            last_ev = max(done)
        ev_day = last_ev - EVAL_DAYS[name] + 1
        tr = idx[dd[idx] < ev_day]
        ev = idx[(dd[idx] >= ev_day) & (dd[idx] <= last_ev)]
        y = df[label].to_numpy()
        if len(tr) == 0 or y[tr].sum() < 10:
            log(f"{name}: too few positives to train ({int(y[tr].sum())})")
            continue
        Xt = X[tr].copy()
        names = list(CATEGORICAL) + list(NUMERIC)
        for f in TARGET_EXCLUDE.get(name, ()):
            Xt[:, names.index(f)] = np.nan  # constant: the model never splits on it
        ds = lgb.Dataset(Xt, y[tr], weight=df.sample_weight.to_numpy()[tr],
                         feature_name=names, categorical_feature=list(CATEGORICAL),
                         free_raw_data=False)
        booster = lgb.train({**PARAMS, **TARGET_PARAMS.get(name, {})}, ds,
                            num_boost_round=ROUNDS)
        p = booster.predict(X[ev])
        rep = _evaluate(name, df.iloc[ev], y[ev], p, truth, [ev_day, last_ev],
                        int(len(tr)), int(y[tr].sum()))
        rep["importance"] = _importance(booster)
        rep["prediction_sd"] = _r(float(np.std(p)), 6)
        report["models"][name] = rep
        trained[name] = (booster, X)
        log(f"{name}: trained on {len(tr):,} rows ({int(y[tr].sum())} positives), "
            f"days {ev_day}-{last_ev}: {len(ev):,} rows, {int(y[ev].sum())} positives, "
            f"AUC {rep['auc']}, calibration {rep['calibration']}, true {rep.get('true_calibration')}")
    n_purch = int((df.d7_revenue > 0).sum())
    report["purchase_model"] = (f"skipped: {n_purch} installs with a purchase, "
                                f"fewer than {min_purchases}")
    if not trained:
        raise RuntimeError("no model could be trained: too few labels")
    return _publish(models_dir, trained, vocab, report, df, log)


def _evaluate(name, rows, y, p, truth, ev_day, n_train, pos_train) -> dict:
    base = {"click": "logged_p_click", "install_click": "logged_p_install_click"}.get(name)
    out = {"eval_days": ev_day, "train_rows": n_train, "train_positives": pos_train,
           "eval_rows": int(len(y)), "eval_positives": int(y.sum()),
           "auc": _r(metrics.roc(y, p)[1]), "log_loss": _r(metrics.log_loss(y, p)),
           "calibration": _r(p.sum() / max(y.sum(), 1e-9), 3)}
    if base is not None:
        lp = rows[base].astype(float).fillna(0).to_numpy()
        lp = np.clip(lp, 1e-6, 1 - 1e-6)
        out["logged"] = {"auc": _r(metrics.roc(y, lp)[1]),
                         "log_loss": _r(metrics.log_loss(y, lp)),
                         "calibration": _r(lp.sum() / max(y.sum(), 1e-9), 3)}
    tk = {"click": "p_click", "install_click": "p_install_click",
          "install_view": "p_install_view"}[name]
    if truth is not None:
        tv = truth[tk].reindex("i" + rows.request_id).to_numpy(float)
        ok = ~np.isnan(tv)
        if ok.any():
            out["true_calibration"] = _r(p[ok].sum() / max(tv[ok].sum(), 1e-9), 3)
    seg = {}
    for pl in sorted(rows.placement.unique()):
        m = (rows.placement == pl).to_numpy()
        if m.sum() and y[m].sum():
            seg[pl] = {"rows": int(m.sum()), "positives": int(y[m].sum()),
                       "calibration": _r(p[m].sum() / y[m].sum(), 3)}
    out["by_placement"] = seg
    # Calibration by predicted value: a model can be calibrated on average
    # and still overconfident at the top, which is where bidding on it
    # spends the most (Chapter 15).
    order = np.argsort(p, kind="stable")
    out["by_prediction"] = [
        {"rows": int(len(ix)), "predicted": _r(p[ix].mean(), 4),
         "actual": _r(y[ix].mean(), 4)}
        for ix in np.array_split(order, 5) if len(ix)]
    return out


def _importance(booster) -> list:
    g = booster.feature_importance("gain")
    names = booster.feature_name()
    tot = g.sum() or 1.0
    order = np.argsort(-g)[:10]
    return [{"feature": names[i], "gain_share": round(float(g[i] / tot), 3)} for i in order]


def _publish(models_dir, trained, vocab, report, df, log):
    root = Path(models_dir) / "conversion"
    root.mkdir(parents=True, exist_ok=True)
    n = max([int(p.name[4:]) for p in root.glob("gbm-[0-9]*") if p.name[4:].isdigit()],
            default=0) + 1
    version = f"gbm-{n:04d}"
    d = root / version
    d.mkdir()
    spec = {"model": "gbm", "version": version, "adtrain_version": __version__,
            "categorical": list(CATEGORICAL), "numeric": list(NUMERIC),
            "zero_if_missing": list(ZERO_IF_MISSING), "other": OTHER,
            "vocab": vocab, "targets": sorted(trained),
            "dataset_version": report["dataset_version"]}
    (d / "model.json").write_text(json.dumps(spec))
    for name, (booster, _) in trained.items():
        (d / f"{name}.json").write_text(json.dumps(booster.dump_model()))
        # LightGBM's own format, so later code can score with this exact
        # model in Python (the pipeline compares it with the champion)
        booster.save_model(str(d / f"{name}.txt"))
    # Golden examples: the logged FeatureVector and ad, the device history
    # the model used, and each model's prediction, for the Go parity check.
    rng = np.random.default_rng(0)
    pick = rng.choice(len(df), size=min(200, len(df)), replace=False)
    with open(d / "golden.jsonl", "w") as f:
        for i in sorted(pick):
            row = df.iloc[[i]]
            fv, ad = json.loads(row.features.iloc[0]), json.loads(row.ad.iloc[0])
            raw = dataset_raws(row)[0]
            num = fv.setdefault("numeric", {})
            # the bidder serves the history it has at bid time
            for k in ZERO_IF_MISSING:
                v = raw[k]
                if v:
                    num[k] = v
                else:
                    num.pop(k, None)
            x = np.array([encode(raw, vocab)], float)
            preds = {name: float(b.predict(x)[0]) for name, (b, _) in trained.items()}
            f.write(json.dumps({"features": fv, "ad": ad, "predictions": preds}) + "\n")
    values = _values(df)
    if values:
        (d / "values.json").write_text(json.dumps(values))
        report["values"] = {"overall": values["overall"], "apps": len(values["by_adv_app"]),
                            "installs": values["installs"]}
    report["version"] = version
    (d / "report.json").write_text(json.dumps(report, indent=1))
    log(f"published {d}")
    return str(d), report


VALUE_K = 20  # installs of evidence the overall value per install is worth


def _values(df) -> dict | None:
    """D7 revenue per install, by advertiser app, shrunk toward the overall
    value (Chapter 21). ROAS pricing uses it: a purchase model needs far
    more purchases than a small market makes, but an average per app
    needs only installs."""
    m = (df.install == 1) & df.d7_final
    if m.sum() == 0:
        return None
    apps = [json.loads(a).get("adv_app") or "" for a in df.ad[m]]
    rev = df.d7_revenue[m].to_numpy(float)
    overall = float(rev.mean())
    by = {}
    for app in sorted(set(apps)):
        r = rev[[a == app for a in apps]]
        by[app] = round((r.sum() + VALUE_K * overall) / (len(r) + VALUE_K), 4)
    return {"overall": round(overall, 4), "by_adv_app": by, "installs": int(m.sum()), "k": VALUE_K}


def _lineage(dataset_dir, manifest, inject) -> dict:
    """Where the bundle came from: enough to rebuild it (Chapter 18)."""
    import hashlib
    import platform

    import lightgbm
    mpath = Path(dataset_dir) / "manifest.json"
    sha = hashlib.sha256(mpath.read_bytes()).hexdigest()[:16] if mpath.exists() else None
    return {"dataset_version": manifest.get("version"), "dataset_manifest_sha256": sha,
            "dataset_settings": manifest.get("settings"),
            "dataset_built_at_sim_ms": manifest.get("built_at_sim_ms"),
            "adtrain_version": __version__, "lightgbm_version": lightgbm.__version__,
            "python_version": platform.python_version(),
            "params": PARAMS, "target_params": TARGET_PARAMS, "rounds": ROUNDS,
            "excluded_features": {k: list(v) for k, v in TARGET_EXCLUDE.items()},
            "eval_days": EVAL_DAYS, "inject": inject}


def load_boosters(bundle_dir) -> dict:
    """The bundle's models as LightGBM boosters, for scoring in Python.
    Bundles published before adtrain 0.6 have no .txt files: empty."""
    import lightgbm as lgb
    out = {}
    for t in TARGETS:
        f = Path(bundle_dir) / f"{t}.txt"
        if f.exists():
            out[t] = lgb.Booster(model_file=str(f))
    return out


def score(bundle_dir, df, raws, target=None) -> dict:
    """Each model's predictions for dataset rows, with the bundle's
    vocabulary; one model if target is given."""
    spec = json.loads((Path(bundle_dir) / "model.json").read_text())
    X = np.array([encode(r, spec["vocab"]) for r in raws], float)
    return {t: b.predict(X) for t, b in load_boosters(bundle_dir).items()
            if target in (None, t)}


def rows_for(df, name, days) -> np.ndarray:
    """Indices of the rows a target is evaluated on, for days [first, last]."""
    label, final, subset = TARGETS[name]
    m = df[final].to_numpy() & (df.day.to_numpy() >= days[0]) & (df.day.to_numpy() <= days[1])
    if subset == "click":
        m &= df.click.to_numpy() == 1
    elif subset == "no_click":
        m &= df.click.to_numpy() == 0
    return np.flatnonzero(m)


def dataset_raws(df) -> list[dict]:
    """Model inputs for each row, with the dataset's device history."""
    raws = [raw_values(json.loads(f), json.loads(a)) for f, a in zip(df.features, df.ad)]
    for r, v, p in zip(raws, df.device_prior_installs, df.device_prior_purchases):
        r["device_prior_installs"] = float(v)
        r["device_prior_purchases"] = float(p)
    return raws


def _r(x, k: int = 4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), k)
