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


## Live mode: bidding against a real bidder

By default, the simulator runs a built-in Python policy. In live mode, the
simulated exchanges send every bid request to a bidder over HTTP instead,
the way real exchanges do:

```bash
python -m adsim --preset crawl --scale tiny --days 1 --bidder http://localhost:8080
```

- Each request is rendered as OpenRTB 2.6 JSON in its exchange's dialect and
  sent to `POST /openrtb/{exchange}`.
- The simulated time goes in the `X-Adsim-Time` header, in milliseconds, so
  the bidder's budgets and day boundaries follow the simulated clock.
- A response slower than the request's `tmax` counts as a no-bid
  (`reason = timeout`), so a slow bidder loses auctions.
- When the run starts, the simulator pushes the campaign book to the
  bidder's admin API (`PUT /api/campaigns`, on port 8081 unless you pass
  `--bidder-admin`), without the creatives' true quality.
- After each auction, it calls the notice URLs in the bid: `nurl` on a win,
  `lurl` on a loss (on exchanges that send them), and `burl` when the ad is
  shown and billed, with `${AUCTION_PRICE}`, `${AUCTION_LOSS}`, and
  `${AUCTION_MIN_TO_WIN}` filled in. About 0.5% of notices are retried, so
  the bidder has to deduplicate them.
- When the run ends and installs are resolved, the simulated MMP sends its
  postbacks to the bidder's `GET /postback`, with a shared token in
  `X-MMP-Token` (`adsim-mmp`, the bidder's default) and the MMP's own
  field names, each at the simulated time the MMP would have sent it.
- When a shown ad is clicked, it calls the click trackers in the bid's ad
  markup (`adm`, a native-style JSON object:
  `{"native": {"link": {"url": ..., "clicktrackers": [...]}}}`), at the
  simulated time of the click, as the ad SDK on the device would. Clicks
  are retried at the same rate as notices.
- `bid_log` records the exchange's view: each request's answer, the
  bidder's `latency_ms`, and the reason for every no-bid. The bidder keeps
  its own, fuller ledger; a ghost bid looks like any other no-bid to the
  exchange.

A simulated day at the `tiny` scale takes about 20 to 40 seconds in live
mode, and at the `laptop` scale a few minutes, depending on your machine.

To keep a market going, for example after deploying a model, continue it
at a later day with `--start-day`. The world (devices, apps, campaigns)
is the same, since it comes from the seed, and the traffic is new:

```bash
python -m adsim --preset crawl --scale tiny --days 1 --start-day 2 --bidder http://localhost:8080
```

Failure scenarios for the operations chapter (Chapter 12) start on the
first simulated day of a live run:

- `--scenario wrong-target`: at 06:00, the campaign system pushes the
  first CPC campaign with its target 100 times too high.
- `--scenario lost-billing`: from 12:00, beta_exchange stops sending
  billing notices for rewarded video, while still billing for it.
- `--scenario bad-labels` (Chapter 24): from 06:00 to 18:00 on the run's
  first day, the MMP turns on probabilistic matching for view-throughs and
  credits us with installs, for 2% of that day's impressions, on devices
  that never installed. Every postback carries a `match_type` field:
  `id_matching`, or `probabilistic` for these.

At the end of every live run, each simulated exchange also sends its
invoice (`PUT /api/invoices`): what it billed, by day, exchange, and
placement, also written to `logs/invoices`.

With `--start-day auto`, as Docker Compose runs it, the simulator asks the
bidder (`GET /api/ledger` on the admin port) for the day after the latest
one in its ledger, so each run continues the market.

The MMP sends each postback at the simulated time it would really arrive,
which for some installs and purchases is days after the impression. A live
run sends the postbacks due before it ends; the rest wait in
`state/pending_postbacks.parquet` and go out during the next run, when their
time comes, so the bidder never hears about an install before it happens.
Each live run also keeps a copy of its logs and truth in
`runs/start=NNNN/`, so analysis can read every run, not just the latest.

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

- Device state that updates during the run (new installs don't yet change
  which apps a device plays, and profiles aren't simulated).
- Click injection and click spamming fraud; ad revenue for hypercasual
  games; iOS and ATT; protocol buffer schemas for the logs.
- Performance work: the crawl policy loops in Python, so `medium` and
  `large` scales are slow.
