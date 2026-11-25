"""The true rates of every creative a campaign could have shown, for the
impressions of a Part IV snapshot (Chapter 36).

The snapshot has the true rates of the creative each impression showed. A
creative policy needs the rates of the ones it didn't, which only the
simulated market knows: each device has a hidden preference over creative
styles, and each creative a hidden quality. This rebuilds the market's world
from its preset and seed, checks that it reproduces the snapshot's own true
rates, and writes two tables next to the snapshot:

- creatives.parquet: each creative's public content (campaign, format,
  style, orientation, length), what a content embedding would see;
- creative_truth.parquet: for each impression from the first validation day
  on, and each creative of its campaign, the true click and install rates.

    python -m adsim.creative_truth --snapshot out/snapshot
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, truth
from .config import preset
from .world import make_world


def build(snapshot: str, preset_name="walk", scale="tiny", seed=7, scenario="ad-roas,full-share"):
    snap = Path(snapshot)
    m = json.loads((snap / "manifest.json").read_text())
    scen = {x for x in scenario.replace(" ", "").split(",") if x}
    cfg = preset(preset_name, scale, seed=seed).with_(
        ad_revenue="ad-roas" in scen, uplift="uplift" in scen, creative_test="creative-test" in scen)
    if "full-share" in scen or "uplift" in scen:
        cfg = cfg.with_(nonattributed_sharing=1.0)
    w = make_world(cfg)
    imp = pd.read_parquet(snap / "impressions.parquet").sort_values("imp_ts_ms", kind="stable")
    # the device's impressions of the campaign earlier that day: its fatigue
    imp["n_seen"] = imp.groupby(["device", "campaign_id", "day"]).cumcount()
    first = min(int(d) for d, s in m["split"].items() if s != "train")
    imp = imp[imp.day >= first].reset_index(drop=True)
    dev = {d.lower(): i for i, d in enumerate(w.devices.device_id)}
    app = {b: i for i, b in enumerate(w.apps.bundle)}
    d = imp.device.str.lower().map(dev)
    if d.isna().any():
        raise SystemExit("the snapshot's devices aren't this market's: check --preset, --scale, --seed")
    d, pub, adv = d.to_numpy(int), imp.publisher_app.map(app).to_numpy(), imp.adv_bundle.map(app).to_numpy()
    cr = w.creatives
    by_campaign = cr.groupby("campaign_id").creative_id.apply(sorted)
    rows = []
    for j in range(int(by_campaign.map(len).max())):
        cid = imp.campaign_id.map(lambda c: by_campaign[c][j] if j < len(by_campaign[c]) else -1).to_numpy()
        ok = cid >= 0
        p = truth.response(w, d[ok], pub[ok], imp.placement.to_numpy()[ok], adv[ok], cid[ok],
                           imp.n_seen.to_numpy()[ok])
        p_inst = p["p_click"] * p["p_install_click"] + (1 - p["p_click"]) * p["p_install_view"]
        rows.append(pd.DataFrame({"request_id": imp.request_id.to_numpy()[ok], "creative_id": cid[ok],
                                  "p_click": p["p_click"], "p_install": p_inst}))
    ct = pd.concat(rows).sort_values(["request_id", "creative_id"]).reset_index(drop=True)
    # the check: the shown creatives' rates match the snapshot's truth
    t = pd.read_parquet(snap / "truth.parquet").set_index("request_id")
    shown = ct.merge(imp[["request_id", "creative_id"]], on=["request_id", "creative_id"])
    tt = t.reindex(shown.request_id)
    corr = float(np.corrcoef(shown.p_click, tt.p_click)[0, 1])
    if corr < 0.999:
        raise SystemExit(f"the rebuilt market doesn't match the snapshot (click rate correlation {corr:.4f}): "
                         "check --preset, --scale, --seed, --scenario")
    pub_cols = ["creative_id", "campaign_id", "format", "style", "orientation", "length_s"]
    creatives = cr[pub_cols].copy()
    creatives.to_parquet(snap / "creatives.parquet", index=False)
    ct.to_parquet(snap / "creative_truth.parquet", index=False)
    extra = m.get("extra", {})
    for name in ("creatives", "creative_truth"):
        extra[name] = hashlib.sha256((snap / f"{name}.parquet").read_bytes()).hexdigest()
    m["extra"] = extra
    m["creative_truth"] = {"adsim_version": __version__, "from_day": first, "impressions": int(len(imp)),
                           "rows": int(len(ct)), "click_rate_correlation": round(corr, 6)}
    (snap / "manifest.json").write_text(json.dumps(m, indent=2))
    return m["creative_truth"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--snapshot", default="out/snapshot")
    p.add_argument("--preset", default="walk")
    p.add_argument("--scale", default="tiny")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--scenario", default="ad-roas,full-share", help="the market's scenarios, as for the run")
    a = p.parse_args()
    r = build(a.snapshot, a.preset, a.scale, a.seed, a.scenario)
    print(f"creative truth: {r['impressions']:,} impressions from day {r['from_day']}, {r['rows']:,} creative rows; "
          f"the rebuilt market's click rates match the snapshot's (correlation {r['click_rate_correlation']})")


if __name__ == "__main__":
    main()
