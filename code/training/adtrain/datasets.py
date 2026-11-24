"""Conversion training datasets (Chapter 14).

One row per shown impression, built from the bidder's ledger:

    billing notices (impressions, deduplicated on the event ID)
      join bids, on the request ID: the FeatureVector, ad, and decision
      left join clicks (the first click within the click window)
      left join install postbacks: click-through on the click ID,
        view-through on the impression ID (deduplicated on postback ID)
      left join purchase postbacks, within 7 days of the install

Every label has a cutoff: a row's label is final only once the label's
window has passed by the time the dataset is built, and rows whose label
isn't final are marked, not silently counted as negatives.

Two device-history features show point-in-time correctness. They count the
device's earlier installs and purchases credited to us, using only the
postbacks the bidder had received before the bid (an as-of join on the
time the bidder received them). With `leak=True`, they're computed from everything received
by the time the dataset is built instead, which is what joining today's
device profile does: the features then include the row's own install.

Postbacks marked in the quarantine list (`out/datasets/quarantine.json`,
Chapter 24) are skipped, for labels and device history alike, and the
manifest records which entries were applied.

Each build is a version in `out/datasets/conversions/<version>/`, with the
rows as Parquet, partitioned by day of impression, and a manifest of its
inputs, settings, statistics, and validation checks.

    python -m adtrain dataset --ledger out/ledger --out out/datasets
"""
from __future__ import annotations

import bisect
import json
import time
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from . import __version__, ledger

DAY_MS = 86_400_000
HOUR_MS = 3_600_000
LabelRateRise = 2.0  # a day's install rate above this many times the median fails
SCHEMA_VERSIONS = {1}  # FeatureVector schema versions this code understands


@dataclass
class Settings:
    click_window_s: int = 3600
    install_cutoff_s: int = 3 * 86400  # install labels are final after this
    revenue_window_s: int = 7 * 86400  # purchases counted after the install
    revenue_cutoff_s: int = 10 * 86400  # D7 revenue labels are final after this
    # Ad ROAS (Chapter 30) uses D3 revenue: final 3.5 days after the install
    short_revenue_window_s: int = 3 * 86400
    short_revenue_cutoff_s: int = int(3.5 * 86400)
    leak: bool = False  # device history as of the build, not the bid
    sample_rate: float = 1.0  # share of rows kept; weights are 1 / rate
    # the newest days of impressions to use; device history still comes
    # from every postback. Bounds memory on long runs. 0 is every day.
    max_days: int = 28


def _device_hash(ifa: str) -> int:
    import zlib
    return zlib.crc32(ifa.encode()) if ifa else 0


def build(ledger_dir, settings: Settings | None = None, log=print, quarantine=None):
    """Returns (rows as a pandas DataFrame, a manifest dict). quarantine is
    a list of entries; None reads the list next to the ledger, in
    out/datasets/quarantine.json."""
    import pandas as pd
    from . import quarantine as qr

    if quarantine is None:
        quarantine = qr.load(Path(ledger_dir).parent / "datasets")
    in_quarantine = qr.matcher(quarantine)
    quarantined = defaultdict(int)
    touched = {}  # device -> when its first quarantined postback arrived

    s = settings or Settings()
    t0 = time.time()
    # Impressions: the first billing notice for each request. The newest
    # notice time is "now" for the build: everything the ledger had by
    # then is known, and nothing after it.
    imps, now = {}, 0
    dup_notices = 0
    all_days = ledger.days(ledger_dir, "bids")
    use = all_days[-s.max_days:] if s.max_days and all_days else all_days
    first = use[0] if use else 0
    nd = [d for d in ledger.days(ledger_dir, "notices") if d >= first]
    for r in ledger.read(ledger_dir, "notices", nd):
        now = max(now, r.get("ts_ms", 0))
        if r.get("type") != "billing":
            continue
        if r.get("duplicate") or r["request_id"] in imps:
            dup_notices += 1
            continue
        imps[r["request_id"]] = (int(r["ts_ms"]), float(r.get("price") or 0))
    clicks = {}
    for r in ledger.read(ledger_dir, "clicks", [d for d in ledger.days(ledger_dir, "clicks") if d >= first]):
        if r.get("duplicate"):
            continue
        rid = r["request_id"]
        if rid not in clicks or r["ts_ms"] < clicks[rid]:
            clicks[rid] = int(r["ts_ms"])
    # Postbacks, deduplicated on their ID, in the MMP's vocabulary.
    seen = set()
    installs = {}  # request ID -> (touch, install ts, received ts)
    purchases = defaultdict(list)  # request ID -> [(event ts, received, revenue)]
    device_installs = defaultdict(list)  # advertising ID -> [received ts]
    device_purchases = defaultdict(list)
    unmatched = dup_pb = n_install_pb = 0
    for p in ledger.read(ledger_dir, "postbacks"):
        f = p.get("fields") or {}
        pid = f.get("postback_id") or p.get("postback_id")
        if pid in seen:
            dup_pb += 1
            continue
        seen.add(pid)
        q = in_quarantine(p)
        if q:
            quarantined[q] += 1
            ifa_q = ((p.get("fields") or {}).get("advertising_id") or "").lower()
            touched[ifa_q] = min(touched.get(ifa_q, 1 << 62), int(p.get("received_ts_ms") or 0))
            continue
        recv = int(p["received_ts_ms"])
        if recv > now:
            continue  # not received yet when the dataset is built
        # when the bidder could first use it, for the as-of join
        avail = max(recv, int(p.get("seen_ts_ms") or 0))
        name = f.get("event_name")
        rid = _request_id(f)
        ifa = (f.get("advertising_id") or "").lower()
        if name == "install":
            if ifa:
                device_installs[ifa].append(avail)
            # an install received in the window's first week may be for an
            # impression before the window: don't count it as unmatched
            counts_toward_joins = not first or recv >= (first + 7) * DAY_MS
            n_install_pb += counts_toward_joins
            if rid not in imps:
                unmatched += counts_toward_joins
                continue
            installs.setdefault(rid, (f.get("touch_type") or "?",
                                      _int(f.get("event_ts_ms")), recv))
        elif name in ("af_purchase", "af_ad_revenue"):
            # D7 revenue is purchases plus, for hypercasual apps, the ad
            # revenue their players earn (Chapter 30)
            if ifa and name == "af_purchase":
                device_purchases[ifa].append(avail)
            if rid in imps:
                purchases[rid].append((_int(f.get("event_ts_ms")), recv,
                                       _float(f.get("revenue_usd"))))
    for d in (device_installs, device_purchases):
        for v in d.values():
            v.sort()

    # one list per column: a dict per row costs several times the memory
    cols, schema_bad, no_features = defaultdict(list), 0, 0
    for b in ledger.read(ledger_dir, "bids", use):
        rid = b.get("request_id")
        if rid not in imps or (b.get("decision") or {}).get("action") != "bid":
            continue
        fv = b.get("features")
        if not fv or not b.get("ad"):
            no_features += 1
            continue
        if fv.get("schema_version") not in SCHEMA_VERSIONS:
            schema_bad += 1
        t, price = imps.pop(rid)  # pop: a request is one row, even if logged twice
        req, d, ad = b.get("request") or {}, b.get("decision") or {}, b["ad"]
        ifa = (req.get("ifa") or "").lower()
        ct = clicks.get(rid)
        click = int(ct is not None and t - 1000 <= ct <= t + s.click_window_s * 1000)
        inst = installs.get(rid)
        weight = 1.0
        if s.sample_rate < 1.0 and not click and inst is None:
            # keep every positive; downsample the rest as they're read, by a
            # hash of the request ID, and weight what's kept. Sampling here,
            # not after, bounds the build's memory on big markets.
            if zlib.crc32(rid.encode()) / 2**32 >= s.sample_rate:
                continue
            weight = 1.0 / s.sample_rate
        rev = rev3 = 0.0
        d3_final = False
        if inst is not None and inst[1] is not None:
            end = inst[1] + s.revenue_window_s * 1000
            rev = sum(v for (et, _, v) in purchases.get(rid, ())
                      if et is not None and et <= end)
            end3 = inst[1] + s.short_revenue_window_s * 1000
            rev3 = sum(v for (et, _, v) in purchases.get(rid, ())
                       if et is not None and et <= end3)
            d3_final = inst[1] + s.short_revenue_cutoff_s * 1000 <= now
        as_of = now if s.leak else t
        row = {
            "request_id": rid, "day": int(b.get("day", t // DAY_MS)),
            "imp_ts_ms": t, "exchange": req.get("exchange"),
            "placement": req.get("placement"), "country": req.get("country"),
            "campaign_id": int(ad.get("campaign", -1)),
            "goal": ad.get("goal"), "creative_id": int(ad.get("creative", -1)),
            "price_cpm": price,
            "logged_p_click": float(d.get("p_click") or 0.0),
            "logged_p_install_click": d.get("p_install_click"),
            "logged_model": d.get("model") or "",
            "features": json.dumps(fv, separators=(",", ":")),
            "ad": json.dumps(ad, separators=(",", ":")),
            "device_prior_installs": _count_before(device_installs.get(ifa), as_of),
            "device_prior_purchases": _count_before(device_purchases.get(ifa), as_of),
            # what the bidder logged for the same feature, if it had it
            "logged_device_prior_installs": int((fv.get("numeric") or {}).get(
                "device_prior_installs", 0)),
            "click": click,
            "install": int(inst is not None),
            "install_click": int(inst is not None and inst[0] == "click"),
            "install_view": int(inst is not None and inst[0] == "view"),
            "d7_revenue": round(rev, 2),
            "d3_revenue": round(rev3, 4),
            "click_final": t + s.click_window_s * 1000 <= now,
            "install_final": t + s.install_cutoff_s * 1000 <= now,
            "d7_final": t + s.revenue_cutoff_s * 1000 <= now,
            "d3_final": d3_final,
            "sample_weight": weight,
            # experiments (Chapter 19): the test, the arm, and the device,
            # for confidence intervals that resample devices
            "experiment": d.get("experiment") or "",
            "arm": d.get("arm") or "",
            "device_hash": _device_hash(ifa),
            # the bidder may have logged history that counted quarantined
            # postbacks for this device: the backfill check leaves it out
            "quarantine_device": t > touched.get(ifa, 1 << 62),
        }
        for k, v in row.items():
            cols[k].append(v)
    # sorted by impression time, a column at a time, freeing each list as
    # it goes: the features column alone is hundreds of MB
    import numpy as np
    order = np.argsort(np.asarray(cols.get("imp_ts_ms", []), dtype=np.int64), kind="stable")
    data = {}
    for k in list(cols):
        v = cols.pop(k)
        data[k] = pd.Series(v).iloc[order].reset_index(drop=True) if len(v) else pd.Series(v)
        del v
    df = pd.DataFrame(data, copy=False)
    del data
    counts = {
        "impressions": len(df), "duplicate_notices": dup_notices,
        "shown_bids_without_features": no_features,
        "unknown_schema_versions": schema_bad,
        "install_postbacks": n_install_pb, "unmatched_installs": unmatched,
        "duplicate_postbacks": dup_pb, "impressions_without_bids": len(imps),
        "quarantined_postbacks": sum(quarantined.values()),
    }
    manifest = {
        "built_at_sim_ms": now, "adtrain_version": __version__,
        "settings": asdict(s),
        "inputs": {st: [d for d in ledger.days(ledger_dir, st) if st == "postbacks" or d >= first]
                   for st in ("bids", "notices", "clicks", "postbacks")},
        "counts": counts, "stats": _stats(df), "seconds": round(time.time() - t0, 1),
        "quarantine": [dict(e, skipped=quarantined.get(e["id"], 0)) for e in quarantine],
    }
    manifest["checks"] = validate(df, manifest)
    manifest["passed"] = all(c["passed"] for c in manifest["checks"])
    log(f"dataset: {len(df):,} impressions, {int(df.click.sum()) if len(df) else 0:,} "
        f"clicks, {int(df.install.sum()) if len(df) else 0:,} installs, "
        f"{'leaky' if s.leak else 'as-of'} device history")
    return df, manifest


def validate(df, manifest) -> list[dict]:
    """The checks from Chapter 14, on the newest complete day against the
    days before it. A failed check means: don't train on this data."""
    c = manifest["counts"]
    out = []

    def check(name, passed, detail):
        out.append({"name": name, "passed": bool(passed), "detail": detail})

    check("schema", c["unknown_schema_versions"] == 0,
          f"{c['unknown_schema_versions']:,} rows with a FeatureVector schema "
          f"version this code doesn't know (it knows {sorted(SCHEMA_VERSIONS)})")
    missing = c["shown_bids_without_features"] / max(1, c["impressions"])
    check("features logged", missing <= 0.01,
          f"{missing:.2%} of shown bids have no FeatureVector (limit 1%)")
    um = c["unmatched_installs"] / max(1, c["install_postbacks"])
    check("label joins", um <= 0.05,
          f"{um:.1%} of install postbacks match none of our impressions (limit 5%)")
    if len(df) == 0:
        check("volume", False, "no impressions")
        return out
    # The device history is rebuilt from postbacks for every row, which is
    # how a new feature gets history (a backfill). Where the bidder logged
    # the same feature, the two must agree (Chapter 14).
    # The overlap window: days on which the bidder logged the feature.
    live = set(df.day[df.logged_device_prior_installs > 0])
    ov = df[df.day.isin(live)]
    qd = ov.quarantine_device if "quarantine_device" in ov else None
    n_q = int(qd.sum()) if qd is not None else 0
    if qd is not None:
        ov = ov[~qd]
    if len(ov):
        mm = float((ov.device_prior_installs != ov.logged_device_prior_installs).mean())
        check("backfill matches logged", mm <= 0.001,
              f"{mm:.2%} of rows on the {len(live)} days the bidder logged it have a "
              "device install count that differs from the logged one (limit 0.1%)" +
              (f"; {n_q:,} rows after a device's quarantined postbacks left out" if n_q else ""))
    else:
        check("backfill matches logged", True,
              "the bidder didn't log device installs on any day: nothing to compare")
    per_day = df.groupby("day").agg(n=("request_id", "size"),
                                    clicks=("click", "sum"))
    days = sorted(per_day.index)
    complete = [d for d in days if (d + 1) * DAY_MS <= manifest["built_at_sim_ms"]]
    if len(complete) >= 2:
        last, prev = complete[-1], complete[:-1][-7:]
        med = float(per_day.loc[prev, "n"].median())
        n = int(per_day.loc[last, "n"])
        check("volume", n >= 0.5 * med,
              f"day {last}: {n:,} impressions, against a median of {med:,.0f} "
              "on the days before (limit: half)")
        ctr = per_day.clicks / per_day.n
        r, m = float(ctr.loc[last]), float(ctr.loc[prev].median())
        check("label rate", m == 0 or 1 / 3 <= r / m <= 3,
              f"day {last}: click rate {r:.2%}, against {m:.2%} on the days "
              "before (limit: a third to three times)")
        # install labels by day of impression: a day far above the others
        # is a batch of bad labels (Chapter 24). Recent days are lower, not
        # higher, while their installs arrive, so only a rise is checked.
        inst = df.groupby("day").install.agg(["sum", "size"])
        rate = inst["sum"] / inst["size"]
        med = float(rate.median())
        high = [(int(d), float(r)) for d, r in rate.items()
                if med > 0 and r > LabelRateRise * med and inst.loc[d, "sum"] >= 20]
        check("install label rate", not high,
              ("; ".join(f"day {d}: {r:.3%} installs per impression" for d, r in high) +
               f", against a median of {med:.3%} over {len(rate)} days (limit {LabelRateRise:g} times)")
              if high else f"no day above {LabelRateRise:g} times the median of {med:.3%} "
              f"installs per impression over {len(rate)} days")
        gaps = [d for d in range(days[0], days[-1] + 1) if d not in per_day.index]
        check("freshness", not gaps, f"days with no impressions: {gaps or 'none'}")
    else:
        check("volume", True, "fewer than two complete days: nothing to compare")
    return out


def _stats(df) -> dict:
    if len(df) == 0:
        return {}
    fin = df[df.install_final]
    return {
        "days": [int(df.day.min()), int(df.day.max())],
        "click_rate": round(float(df.click.mean()), 5),
        "install_final_rows": int(len(fin)),
        "install_rate_final": round(float(fin.install.mean()), 6) if len(fin) else None,
        "installs_per_click_final": round(float(fin.install_click.sum() /
                                               max(1, fin.click.sum())), 4),
        "d7_revenue_final": round(float(df[df.d7_final].d7_revenue.sum()), 2),
        "device_prior_installs_mean": round(float(df.device_prior_installs.mean()), 4),
        "by_placement": {k: int(v) for k, v in df.placement.value_counts().items()},
    }


def write(df, manifest, out_dir, name: str = "conversions") -> str:
    """Write a new version, `<name>-NNNN`, and return its directory."""
    root = Path(out_dir) / name
    root.mkdir(parents=True, exist_ok=True)
    n = len([p for p in root.iterdir() if p.is_dir()]) + 1
    version = f"{name}-{n:04d}"
    d = root / version
    d.mkdir()
    for day, part in df.groupby("day"):
        (d / f"day={int(day):04d}").mkdir()
        part.to_parquet(d / f"day={int(day):04d}" / "part-0000.parquet", index=False)
    manifest = {**manifest, "version": version}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return str(d)


def read(path):
    """A dataset version's rows and manifest."""
    import pandas as pd
    p = Path(path)
    parts = sorted(p.glob("day=*/part-*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True) if parts else pd.DataFrame()
    return df, json.loads((p / "manifest.json").read_text())


def _count_before(times, t) -> int:
    """How many of the sorted times are strictly before t."""
    return bisect.bisect_left(times, t) if times else 0


def _request_id(f: dict):
    for k, prefix in (("click_id", "c"), ("impression_id", "i")):
        v = f.get(k) or ""
        if v.startswith(prefix) and len(v) > 1:
            return v[1:]
    return None


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def print_manifest(m: dict) -> None:
    st = m.get("stats", {})
    print(f"version {m.get('version', '(not written)')}: days {st.get('days')}, "
          f"{m['counts']['impressions']:,} impressions; "
          f"device history {'LEAKY (as of the build)' if m['settings']['leak'] else 'as of each bid'}")
    print(f"  click rate {st.get('click_rate', 0):.2%}; installs per click "
          f"{st.get('installs_per_click_final', 0):.2%} on {st.get('install_final_rows', 0):,} "
          f"rows with final install labels")
    for e in m.get("quarantine") or []:
        print(f"  quarantine {e['id']}: skipped {e.get('skipped', 0):,} postbacks ({e.get('reason', '')})")
    for c in m["checks"]:
        print(f"  {'pass' if c['passed'] else 'FAIL'}  {c['name']}: {c['detail']}")
