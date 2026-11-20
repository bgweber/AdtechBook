# adsim: the simulated mobile ad market

The market half of the book's companion bid simulator. It generates a
synthetic world of devices, apps, campaigns, and exchanges, runs
first-price auctions against a bidding policy, and resolves what happens
next: impressions, clicks, installs, MMP attribution, postbacks, and
purchases. Every run is deterministic for a given config and seed.

Its most important property is that it knows the truth. Next to the logs a
DSP would see, it writes the hidden traits and true probabilities behind
them, so models can be checked against true click rates, true lift, and
true value, not just against noisy outcomes.

## Run it

```bash
pip install -e code/sim[test]
python -m adsim --preset crawl --scale laptop --days 1 --seed 7 --out out/crawl
pytest code/sim
```

Presets match the parts of the book: `crawl` (Part II: low-CPM Android
geos, two exchanges, CPC and a first CPI campaign), `walk` (Part III: adds
tier-1 geos, a third exchange, ROAS goals, and non-attributed data), and
`run` (Part IV: more apps, campaigns, and creatives). Scales: `tiny`
(about 10 seconds), `laptop` (about a minute per day), `medium`, `large`.

From Python:

```python
from adsim import preset, simulate
res = simulate(preset("crawl", seed=7))
res.logs["bid_log"].head()
res.truth["impressions"].head()  # true p_click, lift, value, ...
```

## What it writes

`logs/` is what a DSP would have. Column names follow OpenRTB and the
MMP's own vocabulary, in each source's own format.

| Table | One row per | Notes |
|---|---|---|
| `bid_requests` | Request | Exchange quirks: alpha-2 or alpha-3 countries, rewarded flag in `ext`, missing models, junk on bundles, empty and all-zeros IFAs |
| `bid_requests_sample.jsonl` | Request | The first N requests as OpenRTB 2.6 JSON, for the Go bidder |
| `bid_log` | Request | The policy's decision: bid, ghost (holdout), or no-bid with a reason; campaign, creative, selection probability, values, exploration multiplier |
| `notifications` | Notice | Win, loss (with reason and reported minimum bid to win, by exchange policy), and billing; about 0.5% duplicates |
| `impressions` | Shown impression | With video quartile reached |
| `clicks` | Click | With the click ID passed to the MMP |
| `postbacks` | Postback | Installs and in-app events credited to us, with delays, losses, and duplicates |
| `nonattributed` | Event | Installs and events credited elsewhere, plus history, for advertisers that share them |
| `campaigns`, `creatives`, `app_metadata` | Entity | Configuration and public store metadata |

`truth/` is for checking models. Never use it as model input.

| Table | Contents |
|---|---|
| `devices` | Hidden traits: genre taste, click, install, and spend propensities, bot flag |
| `apps` | Appeal, store quality, ad density, fraud flag, monetization |
| `history` | Installs, uninstalls, and purchases before the simulation |
| `requests` | The device, app, and geo behind each request, and whether it's invalid |
| `auctions` | The highest competing bid for every auction we entered |
| `impressions` | True p_click, p_install_click, p_install_view, organic probability, lift, and expected D7 revenue |
| `installs` | Every install: organic or ad, caused or not, and who the MMP credited |
| `events` | Every purchase and D1 open |

## How the market works

Each generator matches a row of the "Generators to Build" table in the
book plan.

- **World** (`world.py`): devices with hidden traits, an app catalog that
  includes Gem Garden and Cozy Farm Tycoon, campaigns, and creatives.
- **Device history** (`history.py`): installs, uninstalls, and purchases
  over the previous 180 days, sampled from genre taste and app popularity.
  A device sends bid requests from the publisher apps it has installed.
- **Bid requests** (`requests.py`): a daily cycle in local time, floors by
  placement and geo, and exchange quirks.
- **Invalid traffic**: bots on fraudulent apps, with high click rates and
  no installs. Competing bidders avoid them, so a naive bidder buys more
  than its share.
- **Auctions** (`auction.py`): first price, with a highest competing bid
  that shares a common-value term with our own value, which creates the
  winner's curse. Minimum bid to win is reported exactly, rounded, only on
  losses, or not at all, by exchange.
- **Clicks and engagement** (`outcomes.py`) and the true response model
  (`truth.py`): placement, creative, style fit, taste, accidental clicks,
  and fatigue.
- **Installs and lift** (`mmp.py`): organic installs happen at their own
  rate. An install is caused by our ad only if it wouldn't have happened
  organically around the same time.
- **MMP attribution and postbacks** (`mmp.py`): last touch, with
  click-through and view-through windows and other networks' touches.
- **Purchases and revenue**: payers, purchase timing, and amounts; D1
  retention events.
- **Holdouts** (`policy.py`): a salted hash of the device ID; held-out
  devices get ghost bids.

## Not yet built

- An interface for the Go bidder to bid live, rather than a Python policy.
- Device state that updates during the run (new installs don't yet change
  which apps a device plays, and profiles aren't simulated).
- Click injection and click spamming fraud; ad revenue for hypercasual
  games; iOS and ATT; protocol buffer schemas for the logs.
- Performance work: the crawl policy loops in Python, so `medium` and
  `large` scales are slow.
