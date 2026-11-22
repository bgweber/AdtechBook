# The reference DSP

The bidder the book builds, in Go. It grows chapter by chapter, from a
listening bidder in Part II to the full DSP in Part IV. Stages are
configuration (`-stage crawl|walk|run`), and a git tag marks the end of each
part (`part-2`, ...), so you can always see the code for the chapter you're
reading.

## Run it against the simulated market

With Docker, from the repository root:

```bash
docker compose up --build
```

This needs Docker with Compose v2 (`docker compose version` should work),
which current versions of Docker Desktop include.

Without Docker, from the repository root, in three terminals:

```bash
cd code/dsp
go run ./cmd/bidder -ledger ../../out/ledger -models ../../out/models -trainer-url http://localhost:8090
```

```bash
pip install -e code/training[truth]
python -m adtrain serve --ledger out/ledger --models out/models --sim-out out/sim
```

```bash
pip install -e code/sim
python -m adsim --preset crawl --scale tiny --days 2 --bidder http://localhost:8080 --out out/sim
```

Then open the console at <http://localhost:8081>, train a click model on
its Models tab, and play the next day with it:

```bash
python -m adsim --preset crawl --scale tiny --days 1 --start-day auto --bidder http://localhost:8080 --out out/sim
```

## Ports

| Port | Who uses it | What's there |
|---|---|---|
| 8080, the bid port | Exchanges, here the simulator | `POST /openrtb/{exchange}`: OpenRTB 2.6 bid requests. A no-bid is an empty `204`. `GET /notice/{win,loss,billing}`: notices, through the URLs in each bid. `GET /click`: the click tracker in each bid's ad markup. `GET /postback`: install and event postbacks from the advertiser's MMP, with the shared token (`-mmp-token`) in `X-MMP-Token`. |
| 8081, the admin port | You, and the campaign system | The console, `GET /api/stats`, `PUT /api/campaigns` (the campaign book; adsim pushes it when a live run starts), `GET /api/models`, `GET /api/models/report?version=`, `POST /api/models/deploy` (`{"version": ..., "force": false}`; an empty version goes back to the priors), `POST /api/train` (`{"deploy": true}`) and `GET /api/train`, `GET /api/ledger` (the next simulated day to play), `GET /api/ops` (switches, alerts, throttle rates, reconciliation, audit log), `POST /api/ops/switch` (`{"kind": "campaign", "key": "3", "on": true}`; kinds: global, model, exchange, campaign, feature), `PUT /api/invoices`, `POST /api/parity` (run the feature parity check) and `GET /api/parity`, and `GET /healthz` |

Admin traffic never touches the bid port, so looking at the console can't
slow down bidding.

## What's built

| Chapter | Piece | Status |
|---|---|---|
| 6 | Parsing and normalization of each exchange's quirks | Works |
| 6 | Device profiles from the bidder's own traffic | Works |
| 6 | Eligibility (country, budget, fitting creative) and weighted random selection, with the campaigns each filter removed logged on every bid (`decision.filtered`) and counted on the Live tab | Works |
| 7 | Listen mode, and bid mode on a share of requests (`-bid-share`) | Works |
| 7 | The console: traffic, floors, decisions, latency, and campaigns | Works |
| 8 | The FeatureVector in the bid ledger, and win, loss, and billing notices | Works |
| 10 | Crawl pricing from a prior click rate, with margin, shading, and exploration | Works |
| 23 | A 5% device holdout, logged as ghost bids | Works |
| 8, 9 | Click trackers in the ad markup, and the click log | Works |
| 9 | Serving the click model: hashed tokens, the downsampling correction | Works |
| 11 | Deploys: a golden-example parity check, the validation checks (forced deploys override them, parity never), warmup with a latency check, hot swap, rollback to priors, and bids by model version for each simulated hour | Works |
| 9, 11 | The Models page: ROC and PR curves, calibration, top crosses and tokens, stability, training and deploying from the console | Works |
| 8 | The MMP postback endpoint (minimal version): a shared token, postbacks logged in the MMP's own fields | Works |
| 10 | CPI pricing: target CPI × predicted click rate × the bundle's smoothed install rate after a click for the segment | Works |
| 8 | Raw context (the request as received and the profile snapshot) for 5% of bids, and the parity check: FeatureVectors recomputed by a second implementation in the training code, from the console or `python -m adtrain parity` | Works |
| 8 | Injected skews for the exercise: `-skew bundle` (keep junk appended to bundles) or `-skew conn` (finer cellular connection types), in the bidder only | Works |
| 12 | Kill switches (global, exchange, campaign, model, feature) with an audit log and a switches version on every bid | Works |
| 12 | The fallback ladder: model, then the bundle's segment click rates (placement, country, exchange), then placement priors; also when a prediction is outside 0.01% to 50% | Works |
| 12 | Even-spend throttling, shaped like yesterday's traffic, with the throttle rate logged on every bid | Works |
| 12 | Alerts: spend far above plan, no bids, fallback use, prediction collapse, latency, reconciliation gap | Works |
| 12 | Spend reconciliation against the exchanges' invoices, by day, exchange, and ad format | Works |

## Modes

- `-mode listen`: run the whole bid path and log every request, but never
  bid (Chapter 7).
- `-mode bid`: bid on `-bid-share` of requests (default 1, all of them).
  Start with `-bid-share 0.01`, as Chapter 7 suggests.

## Layout

```
cmd/bidder/          main: flags, the bid and admin servers
internal/openrtb/    the OpenRTB 2.6 fields the bidder reads
internal/normalize/  one internal request, whatever the exchange's quirks
internal/bidder/     the bid path and notice handler: parse, normalize, decide, respond, log
internal/campaigns/  the campaign book, eligibility, and spend from billing notices
internal/profile/    device profiles from the bidder's own traffic
internal/features/   the FeatureVector and its canonicalization rules
internal/policy/     crawl selection and pricing, and the holdout
internal/model/      the click model: tokens, MurmurHash3, predict, the registry and deploys
internal/ops/        kill switches, audit log, throttling, alerts, reconciliation
internal/ledger/     the bid ledger, one JSON line per request, per day
internal/stats/      live numbers for the console
internal/clock/      simulated time from the X-Adsim-Time header
internal/console/    the admin console (static/index.html), and starting training runs
```

## The bid ledger and notice log

The bidder writes gzip-compressed JSON lines, by simulated day:

```
out/ledger/bids/day=0000/part-0000.jsonl.gz     one record per request
out/ledger/notices/day=0000/part-0000.jsonl.gz  one record per notice
out/ledger/clicks/day=0000/part-0000.jsonl.gz   one record per click
out/ledger/postbacks/day=0000/part-0000.jsonl.gz one record per MMP postback
out/ledger/raw/day=0000/part-0000.jsonl.gz       raw context for 5% of bids
```

A bid record has the normalized request, the decision with every term of
the price, and the FeatureVector, which every bid and ghost bid carries,
along with a 5% sample of no-bids. Bids and ghost bids also log the ad
that was scored (campaign and creative attributes), and the click model
version, if one priced the bid. A notice record is the exchange's notice
as sent, with an event ID; retries are logged and marked as duplicates, but
never counted twice. Parts are closed after two seconds without traffic,
so you can read them while the bidder runs:

```python
import glob, pandas as pd
bids = pd.concat(pd.read_json(f, lines=True, compression="gzip")
                 for f in glob.glob("out/ledger/bids/day=*/*.jsonl.gz"))
```

Writes are queued off the bid path; if the queue fills, records are dropped
and counted, never allowed to slow down bidding.

## Models

Published models live in `out/models/click/<version>/` (see
[`code/training/README.md`](../training/README.md)). Deploying one, from
the console or at startup, loads it, predicts its golden examples, and
refuses it if any prediction differs from the training code's by more than
1e-9. It also refuses a model that failed its validation checks unless the
deploy is forced, which is the manual intervention Chapter 11 describes.
The deployed version is written to `out/models/click/DEPLOYED`, so a
restarted bidder comes back with it, and every deploy is appended to
`deploys.jsonl`.

## Operations (Chapter 12)

The console's Controls tab has the alerts, the kill switches, each
campaign's spend against plan with its throttle rate, the spend
reconciliation, and the audit log. Switches and throttle rates are saved
in `out/ops`, so a restarted bidder keeps them. Every bid record logs the
campaign book and switches versions it was priced with, where its click
rate came from (`p_click_from`: model, fallback_segment, fallback_bounds,
or prior), and the campaign's throttle rate. Alert thresholds are at the
top of `internal/ops/alerts.go`.

To try the Chapter 12 exercise, play a day with both failure scenarios:

```bash
ADSIM_SCENARIO=wrong-target,lost-billing docker compose run --rm sim
```

In PowerShell: `$env:ADSIM_SCENARIO="wrong-target,lost-billing"; docker
compose run --rm sim`, then `Remove-Item Env:ADSIM_SCENARIO`. At 06:00 a
CPC target becomes 100 times too high: watch the spend alert, pause the
campaign, and see how much it overspent. From 12:00, beta_exchange stops
sending billing notices for rewarded video: the reconciliation finds it
when the invoices arrive at the end of the run. Then publish the broken
model from the Models tab, force-deploy it, play another day, and watch
the prediction alert; switch the model off and watch the fallback.

## Tests

```bash
go vet ./... && go test ./...
```
