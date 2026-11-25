"""Ad revenue for hypercasual apps (Chapter 30): the realized D7 ad revenue
of installs matches the truth table's expected value, and markets without
the ad-roas scenario don't change."""
import numpy as np
import pandas as pd

from adsim import preset, simulate
from adsim import mmp, truth
from adsim.world import make_world


def test_walk_seed_7_is_unchanged():
    # Part III's results come from the walk market with seed 7; the ad
    # revenue code must leave it alone when ad_revenue is off.
    r = simulate(preset("walk", "tiny", seed=7, days=1))
    n = {k: len(v) for k, v in r.logs.items()}
    assert n["impressions"] == 14_507 and n["clicks"] == 440
    assert n["postbacks"] == 57 and n["campaigns"] == 16
    assert n["nonattributed"] == 3_539
    assert len(r.truth["installs"]) == 637
    assert not (r.logs["postbacks"].event_name == "af_ad_revenue").any()


def test_ad_revenue_campaigns():
    cfg = preset("walk", "tiny", seed=11, days=1).with_(ad_revenue=True)
    r = simulate(cfg)
    camp = r.logs["campaigns"]
    assert (camp.goal == "AD_ROAS").sum() == cfg.ad_roas_campaigns
    # the other campaigns are the same as without ad revenue
    base = make_world(cfg.with_(ad_revenue=False)).campaigns
    assert camp[camp.goal != "AD_ROAS"].reset_index(drop=True).equals(base)
    assert (r.logs["postbacks"].event_name == "af_ad_revenue").any()


def test_ad_revenue_matches_truth():
    # 20,000 installs of hypercasual apps by random devices: the realized
    # D7 ad revenue matches the truth table's expected value.
    cfg = preset("walk", "tiny", seed=11, days=1).with_(ad_revenue=True)
    world = make_world(cfg)
    rng = np.random.default_rng(0)
    hc = world.apps.app_id[world.apps.monetization == "ads"].to_numpy()
    n = 20_000
    installs = pd.DataFrame({
        "device_idx": rng.integers(0, len(world.devices), n),
        "app_id": rng.choice(hc, n),
        "ts_ms": rng.integers(0, 86_400_000, n)})
    events = mmp.purchases(world, installs)
    ad = mmp.ad_revenue(world, installs, events)
    age = ad.ts_ms.to_numpy() - installs.ts_ms.to_numpy()[ad.install_row]
    bots = world.devices.is_bot.to_numpy()[installs.device_idx.to_numpy()]
    for days in (3, 7):  # Ad ROAS campaigns use D3 (Chapter 30)
        dn = ad[age < days * 86_400_000].groupby("install_row").revenue.sum()
        real = dn.reindex(range(n), fill_value=0.0).to_numpy()
        exp = truth.expected_ad_revenue(
            world, installs.device_idx.to_numpy(), installs.app_id.to_numpy(),
            days)
        assert (real[~bots] > 0).all() and (real[bots] == 0).all()
        assert abs(real.sum() / exp.sum() - 1) < 0.05, days
        # and it's learnable: the truth explains much of the variation
        assert np.corrcoef(real, exp)[0, 1] > 0.5, days


def test_uplift_raises_heavy_players_organic_rate():
    """The uplift scenario (Chapter 32): heavy players install on their own
    far more often, light players less."""
    base = make_world(preset("walk", "tiny", seed=7))
    up = make_world(preset("walk", "tiny", seed=7).with_(uplift=True))
    adv = np.flatnonzero(base.apps.is_advertiser.to_numpy())[:5]
    act = base.devices.activity.to_numpy()
    heavy, light = np.argsort(act)[-50:], np.argsort(act)[:50]
    for devs, more in ((heavy, True), (light, False)):
        d = np.repeat(devs, len(adv))
        a = np.tile(adv, len(devs))
        r = truth.organic_rate(up, d, a).sum() / truth.organic_rate(base, d, a).sum()
        assert (r > 5) if more else (r < 0.5)


def test_traffic_surge_triples_the_afternoon():
    from adsim.history import make_history
    from adsim.requests import make_requests
    counts = []
    for s in (False, True):
        cfg = preset("walk", "tiny", seed=7, days=1).with_(start_day=5, surge=s)
        w = make_world(cfg)
        _, inst = make_history(w)
        r = make_requests(w, inst, 5)
        h = ((r.ts_ms - 5 * 86_400_000) // 3_600_000).to_numpy()
        counts.append(np.bincount(h, minlength=24))
        assert r.request_id.is_unique
    assert (counts[1][13:16] > 2.5 * counts[0][13:16]).all()
    assert (counts[1][:13] == counts[0][:13]).all()


def test_new_game_cuts_the_genres_installs_after_noon():
    # Chapter 29: from the shift on, the genre's games get 40% of their
    # installs; everything else, and everything before, is unchanged
    from adsim.run import NEW_GAME_SHARE
    base = preset("walk", "tiny", seed=7, days=1).with_(ad_revenue=True)
    noon = 12 * 3_600_000
    ra, rb = simulate(base), simulate(base.with_(new_game_ms=noon))
    a = ra.truth["impressions"].merge(ra.logs["impressions"][["impression_id", "imp_ts_ms"]], on="impression_id")
    b = rb.truth["impressions"]
    m = a.merge(b, on="impression_id", suffixes=("", "_shift"))
    apps = make_world(base).apps
    hc = apps.genre.to_numpy()[m.adv_app_id.to_numpy()] == "hypercasual"
    late = m.imp_ts_ms.to_numpy() >= noon
    pos = m.p_install_view.to_numpy() > 0  # bots install nothing
    r = np.where(pos, m.p_install_view_shift / np.where(pos, m.p_install_view, 1), 1.0)
    assert np.allclose(r[hc & late & pos], NEW_GAME_SHARE)
    assert np.allclose(r[~(hc & late)], 1.0)
    assert np.allclose(m.p_click, m.p_click_shift)
