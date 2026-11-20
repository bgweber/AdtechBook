"""What happens after an impression is shown: engagement and clicks, and
whether the ad leads to an install.

Install draws here are only the ad's own path to an install. Whether an
install actually happens, whether the ad caused it, and who gets credit
are decided in `mmp.py`, together with organic installs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import STYLES
from .truth import sigmoid
from .world import World

VIDEO_PLACEMENTS = ("rewarded_video", "interstitial")


def engagement_and_clicks(world: World, imps: pd.DataFrame,
                          rng: np.random.Generator) -> pd.DataFrame:
    """Add video quartiles, clicks, and ad-path install draws.

    `imps` has one row per shown impression with true probabilities
    (`p_click`, `p_install_click`, `p_install_view`), device index,
    creative id, placement, and `imp_ts_ms`.
    """
    n = len(imps)
    cr = world.creatives
    dev = imps.true_device_idx.to_numpy()
    creative = imps.creative_id.to_numpy()
    fmt = cr.format.to_numpy()[creative]
    si = {s: i for i, s in enumerate(STYLES)}
    style = cr["style"].map(si).to_numpy()[creative]
    fit = world.style_pref[dev, style] * len(STYLES) - 1.0
    quality = cr.quality.to_numpy()[creative]
    is_video = (np.isin(imps.placement, VIDEO_PLACEMENTS)
                & np.isin(fmt, ["video", "playable"]))
    rewarded = (imps.placement == "rewarded_video").to_numpy()
    p_complete = sigmoid(0.6 + 1.5 * rewarded + 1.2 * fit + quality)
    complete = rng.random(n) < p_complete
    quartile = np.where(complete, 4, rng.integers(0, 4, n))
    quartile = np.where(is_video, quartile, -1)  # -1: not a video
    click = rng.random(n) < imps.p_click.to_numpy()
    t_imp = imps.imp_ts_ms.to_numpy()
    late = rng.random(n) < 0.015
    click_delay = np.where(
        late, rng.uniform(60_000, 1_800_000, n),
        np.minimum(rng.lognormal(np.log(4_000), 1.0, n), 60_000))
    t_click = np.where(click, t_imp + click_delay, np.nan)
    # the ad's own path to an install
    inst_c = click & (rng.random(n) < imps.p_install_click.to_numpy())
    inst_v = ~click & (rng.random(n) < imps.p_install_view.to_numpy())
    cfg = world.cfg
    d_c = np.minimum(rng.lognormal(np.log(15 * 60_000), 1.8, n),
                     cfg.click_window_s * 1000 * 0.99)
    d_v = np.minimum(rng.lognormal(np.log(4 * 3_600_000), 1.2, n),
                     cfg.view_window_s * 1000 * 0.99)
    t_ad = np.where(inst_c, t_click + d_c,
                    np.where(inst_v, t_imp + d_v, np.nan))
    out = imps.copy()
    out["video_quartile"] = quartile
    out["clicked"] = click
    out["click_ts_ms"] = t_click
    out["ad_install"] = inst_c | inst_v
    out["ad_install_path"] = np.where(inst_c, "click",
                                      np.where(inst_v, "view", ""))
    out["ad_install_ts_ms"] = t_ad
    return out
