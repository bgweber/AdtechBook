"""Part II's results come from the crawl preset with seed 7. Part III adds
to the simulator, and this test makes sure none of that changes the crawl
market: if it fails, Part II's numbers would no longer reproduce."""
from adsim import preset, simulate


def test_crawl_seed_7_is_unchanged():
    r = simulate(preset("crawl", "tiny", seed=7, days=1))
    n = {k: len(v) for k, v in r.logs.items()}
    assert n["bid_requests"] == 66_851
    assert n["impressions"] == 12_262
    assert n["clicks"] == 368
    assert n["postbacks"] == 64
    assert n["campaigns"] == 8 and n["creatives"] == 24
    assert len(r.truth["installs"]) == 638
    assert round(float(r.truth["impressions"].p_click.sum()), 4) == 386.4713
