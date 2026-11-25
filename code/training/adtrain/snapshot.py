"""The Part IV dataset: one frozen snapshot of a simulated market for the
deep model chapters (25-29).

Built once from a run's bidder ledger and simulator output, it holds what
those chapters need and nothing else, so each chapter's models train in
minutes, on the same rows and the same split, without running the market
again. Everything is in out/snapshot/:

    impressions.parquet  one row per billed impression: the features the
                         bidder logged at bid time, flattened (the device's
                         recent apps as a list), the campaign and creative,
                         the labels (click, install after click and after
                         view, credited install, D3 ad revenue), whether
                         each label is final, and the split by day
    histories.parquet    each device's history, in time order: the
                         publisher apps it was seen in (counted by day),
                         its clicks, its credited installs, and its installs
                         from the advertisers' shared data
    truth.parquet        the simulator's true rates for each impression,
                         by request ID: for checking models only, never as
                         a feature, which is why it's a separate file
    apps.parquet         every app, with its genre (from the truth: for
                         checks, such as whether embeddings group genres)
    campaigns.parquet    the campaigns, with goal, target, and budget
    manifest.json        the run, the versions, the split, row counts,
                         and a checksum for every file

    python -m adtrain snapshot --ledger out/ledger --sim-out out/sim --out out/snapshot

The book's snapshot is also a release download; build your own, with more
days or a larger market, to scale up.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import __version__, datasets, gbm, ledger, simout

DAY_MS = 86_400_000
FILES = ("impressions", "histories", "truth", "apps", "campaigns")
# In the simulator, 99% of installs reach us within a day of the bid, so the
# snapshot treats install labels as final after a day, not the three days
# the Part III pipeline waits; only the newest day's aren't.
INSTALL_CUTOFF_S = 86_400


def split_days(days: list[int]) -> dict[int, str]:
    """The fixed split by day: the last day is the tail (its installs
    aren't final), the two before it test, the two before those validate,
    and the rest train."""
    days = sorted(days)
    out = {}
    for i, d in enumerate(reversed(days)):
        out[d] = "tail" if i == 0 else "test" if i <= 2 else "valid" if i <= 4 else "train"
    return out


def build(ledger_dir, sim_out, out_dir, log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = simout.table(sim_out, "logs", "app_metadata")
    camp = simout.table(sim_out, "logs", "campaigns")
    if meta is None or camp is None:
        raise RuntimeError("needs the simulator's logs (--sim-out)")
    bundle = dict(zip(meta.app_id.astype(str), meta.bundle))

    # impressions: the conversion dataset over every day, flattened
    log("snapshot: building the impressions from the ledger")
    df, _ = datasets.build(ledger_dir, datasets.Settings(install_cutoff_s=INSTALL_CUTOFF_S, max_days=0),
                           log=lambda *a: None)
    days = sorted(int(d) for d in df.day.unique())
    split = split_days(days)
    rows = defaultdict(list)
    for fv_s, ad_s in zip(df.features, df.ad):
        fv, ad = json.loads(fv_s), json.loads(ad_s)
        raw = gbm.raw_values(fv, ad)
        for k in gbm.CATEGORICAL:
            rows[k].append(raw[k])
        for k in gbm.NUMERIC:
            rows[k].append(raw[k])
        rec = (fv.get("lists") or {}).get("recent_apps") or {}
        rows["recent_apps"].append(list(rec.get("values") or []))
        rows["recent_ages_s"].append([int(x) for x in (rec.get("age_seconds") or [])])
        rows["apps_seen"].append(sorted((fv.get("sets") or {}).get("apps_seen") or []))
        rows["adv_bundle"].append(bundle.get(str(ad.get("adv_app")), ""))
    imp = pd.DataFrame({
        "request_id": df.request_id, "day": df.day.astype(int), "split": df.day.map(split),
        "imp_ts_ms": df.imp_ts_ms.astype("int64"), "campaign_id": df.campaign_id, "creative_id": df.creative_id,
        "price_cpm": df.price_cpm,
        "logged_p_click": df.logged_p_click, "logged_p_install_click": df.logged_p_install_click,
        "logged_model": df.logged_model,
        **{k: v for k, v in rows.items()},
        "click": df.click.astype(int), "install_click": df.install_click.astype(int),
        "install_view": df.install_view.astype(int), "install": df.install.astype(int),
        "d3_revenue": df.d3_revenue.astype(float),
        "click_final": df.click_final, "install_final": df.install_final, "d3_final": df.d3_final,
        "install_received_ts_ms": (df.install_received_ts_ms.astype("int64") if "install_received_ts_ms" in df
                                   else -1),
    })

    # the device of each impression, and the histories, from the bid stream
    log("snapshot: reading the bid stream for the device histories")
    rid_dev, seen, first = {}, defaultdict(int), {}
    want = set(imp.request_id)
    adv_of = {}
    for b in ledger.read(ledger_dir, "bids", days):
        ifa = ((b.get("request") or {}).get("ifa") or "").lower()
        if not ifa:
            continue
        rid = b["request_id"]
        if rid in want:
            rid_dev[rid] = ifa
        d = b.get("decision") or {}
        if d.get("action") == "bid":
            adv_of[rid] = (ifa, bundle.get(str((b.get("ad") or {}).get("adv_app")), ""))
        pub = ((b.get("features") or {}).get("strings") or {}).get("publisher_app")
        if pub:
            ts = int(b["ts_ms"])
            k = (ifa, pub, ts // DAY_MS)
            seen[k] += 1
            first[k] = min(first.get(k, ts), ts)
    imp.insert(1, "device", imp.request_id.map(rid_dev).fillna(""))
    ev = defaultdict(list)

    def add(dev, ts, kind, app, n=1):
        ev["device"].append(dev)
        ev["ts_ms"].append(int(ts))
        ev["kind"].append(kind)
        ev["app"].append(app)
        ev["n"].append(n)
    for (dev, app, day), n in seen.items():
        add(dev, first[(dev, app, day)], "seen", app, n)
    for c in ledger.read(ledger_dir, "clicks", days):
        x = adv_of.get(c.get("request_id"))
        if x:
            add(x[0], c.get("ts_ms") or c.get("received_ts_ms") or 0, "click", x[1])
    seen_pb = set()
    for p in ledger.read(ledger_dir, "postbacks"):
        f = p.get("fields") or {}
        if f.get("event_name") != "install" or f.get("postback_id") in seen_pb:
            continue
        seen_pb.add(f.get("postback_id"))
        add((f.get("advertising_id") or "").lower(), float(f.get("event_ts_ms") or 0), "install", f.get("app_bundle"))
    na = simout.table(sim_out, "logs", "nonattributed")
    if na is not None:
        na = na[na.event_name == "install"]
        for dev, app, ts in zip(na.advertising_id, na.app_bundle, na.event_ts_ms):
            add(str(dev).lower(), ts, "shared_install", app)
    hist = pd.DataFrame(ev).sort_values(["device", "ts_ms", "kind"], kind="stable").reset_index(drop=True)

    # the truth, by request ID, kept apart
    tt = simout.table(sim_out, "truth", "impressions",
                      ["impression_id", "p_click", "p_install_click", "p_install_view", "lift",
                       "expected_d3_ad_revenue"])
    if tt is None:
        raise RuntimeError("needs the simulator's truth (--sim-out)")
    tt = tt.assign(request_id=tt.impression_id.str[1:]).drop(columns="impression_id")
    truth = imp[["request_id"]].merge(tt, on="request_id", how="left")

    apps = simout.table(sim_out, "truth", "apps")
    apps = apps.merge(meta[["app_id", "bundle"]], on="app_id", how="left") if "bundle" not in apps else apps
    cols = [c for c in ("app_id", "bundle", "genre", "style", "monetization", "is_advertiser") if c in apps]
    apps = apps[cols]
    campaigns = camp.drop_duplicates("campaign_id", keep="last")

    files = {"impressions": imp, "histories": hist, "truth": truth, "apps": apps, "campaigns": campaigns}
    sums = {}
    for name, frame in files.items():
        path = out / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        sums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    by_split = imp.groupby("split").agg(rows=("request_id", "size"), clicks=("click", "sum"),
                                        installs=("install", "sum"), days=("day", "nunique"))
    manifest = {
        "adtrain_version": __version__, "adsim_version": _adsim_version(),
        "days": [days[0], days[-1]], "split": {str(d): s for d, s in split.items()},
        "rows": {k: int(len(v)) for k, v in files.items()},
        "by_split": {s: {k: int(v) for k, v in r.items()} for s, r in by_split.iterrows()},
        "devices": int(imp.device.nunique()), "apps": int(len(apps)),
        "install_cutoff_s": INSTALL_CUTOFF_S,
        "sha256": sums,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"snapshot: {len(imp):,} impressions over days {days[0]}-{days[-1]}, {len(hist):,} history events, "
        f"in {out}")
    return manifest


def _adsim_version():
    try:
        from adsim import __version__ as v
        return v
    except ImportError:
        return None


def load(path="out/snapshot", check=True) -> dict:
    """The snapshot's tables and manifest. With check, a warning when it was
    built by another adtrain version, or a file doesn't match its checksum."""
    p = Path(path)
    if not (p / "manifest.json").exists():
        raise RuntimeError(f"no snapshot in {p}: build one with `python -m adtrain snapshot`, "
                           "or download the book's from the releases")
    m = json.loads((p / "manifest.json").read_text())
    out = {"manifest": m}
    for name in FILES:
        f = p / f"{name}.parquet"
        if check and hashlib.sha256(f.read_bytes()).hexdigest() != m["sha256"][name]:
            print(f"warning: {f} doesn't match the manifest's checksum")
        out[name] = pd.read_parquet(f)
    for name, sha in (m.get("extra") or {}).items():  # tables added later, such as the creative truth (Chapter 36)
        f = p / f"{name}.parquet"
        if f.exists():
            if check and hashlib.sha256(f.read_bytes()).hexdigest() != sha:
                print(f"warning: {f} doesn't match the manifest's checksum")
            out[name] = pd.read_parquet(f)
    if check and m.get("adtrain_version", "").split(".")[:2] != __version__.split(".")[:2]:
        print(f"note: the snapshot was built with adtrain {m.get('adtrain_version')}, this is {__version__}")
    return out


def print_manifest(m: dict) -> None:
    print(f"snapshot: days {m['days'][0]}-{m['days'][1]}, {m['rows']['impressions']:,} impressions, "
          f"{m['devices']:,} devices, {m['rows']['histories']:,} history events "
          f"(adtrain {m['adtrain_version']}, adsim {m['adsim_version']})")
    for s in ("train", "valid", "test", "tail"):
        r = m["by_split"].get(s)
        if r:
            print(f"  {s:<6}{r['days']:>3} days {r['rows']:>9,} impressions {r['clicks']:>7,} clicks "
                  f"{r['installs']:>6,} installs")
