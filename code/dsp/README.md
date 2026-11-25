# The reference DSP

The bidder the book builds, in Go. It grows chapter by chapter, from a
listening bidder in Part II to the full DSP in Part IV, as one program:
stages are configuration (`-stage crawl` for Part II, `-stage walk` from
Part III on), and each print edition's git tag has the code that matches it.

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
| 8081, the admin port | You, and the campaign system | The console, `GET /api/stats`, `PUT /api/campaigns` (the campaign book; adsim pushes it when a live run starts), `GET /api/models`, `GET /api/models/report?version=`, `POST /api/models/deploy` (`{"version": ..., "force": false}`; an empty version goes back to the priors), `POST /api/train` (`{"deploy": true}`) and `GET /api/train`, `GET /api/ledger` (the next simulated day to play), `GET /api/ops` (switches, alerts, throttle rates, reconciliation, audit log), `POST /api/ops/switch` (`{"kind": "campaign", "key": "3", "on": true}`; kinds: global, model, exchange, campaign, feature), `PUT /api/invoices`, `POST /api/parity` (run the feature parity check) and `GET /api/parity`, `POST /api/labels` (read the install labels, Chapter 13) and `GET /api/labels`, `GET /api/datasets` and `POST /api/datasets` (`{"leak": false}`, build a conversion dataset, Chapter 14), `GET /api/conversion`, `POST /api/conversion/train` (`{"dataset": "conversions-0001"}`), `POST /api/conversion/deploy` (`{"version": ..., "force": false}`) and `POST /api/conversion/undeploy` (LightGBM conversion models, Chapter 15), `POST /api/conversion/pipeline` (`{"leak": false, "inject": "no-features"}`, run the pipeline) and `GET /api/conversion/pipeline` (its runs), `POST /api/conversion/promote`, `POST /api/conversion/rollback`, and `GET /api/conversion/history` (Chapter 18), `GET` and `POST /api/policy/selection` (`{"name": "uniform"}`: budget, uniform, or cpi-first), `POST /api/eval/replay` (`{"day": 19, "a": "logged", "b": "gbm-0003"}`), `POST /api/eval/ope` (`{"policy": "uniform", "reward": "click"}`), and `POST /api/eval/periods` (Chapter 16), `GET /api/experiment`, `POST /api/experiment/start` (`{"kind": "shadow" | "canary" | "ab", "challenger": "gbm-0003", "share": 0.5, "budget_split": true, "stages": [0.05, 0.25, 0.5], "stage_hours": 4}`), `POST /api/experiment/stop`, `POST /api/experiment/inject` (`{"fault": "scale"}`), and `POST /api/eval/abtest` and `/api/eval/shadow` (Chapter 19), `GET /api/win`, `POST /api/win/train` (`{"use_mtw": true}`), `POST /api/win/deploy` and `/api/win/undeploy`, and `POST /api/eval/shadetest` (Chapter 20), `GET` and `POST /api/calibration` (`{}` to update, `{"off": true}`), `GET` and `POST /api/creatives` (`{"thompson": true, "reset": false}`), and the `auction` selection policy (Chapter 21), `GET` and `POST /api/pacing` (`{"on": true}`), `POST /api/pacing/overbid` (`{"campaign": 9, "overbid": 1.5}`), and `POST /api/pacing/forecast` (Chapter 22), `POST /api/eval/lift` (`{"holdout_share": 0.01, "app": "com.racegames.wordcraft"}`, Chapter 23), `GET /api/latency` (stage timings, load test runs, the cap and its audit), `POST /api/latency/reset` (`{"which": "load"}` or `"live"`), `POST /api/policy/auction-cap` (`{"cap": 3}`), and `/debug/pprof/` (Chapter 17), `GET /api/drills`, `POST /api/drills/start` (`{"kind": "shift", "factor": 1.1}` or `{"kind": "parity"}`) and `/api/drills/stop`, `POST /api/parity/day` (`{"day": 41}`), `GET /api/quarantine`, `POST /api/quarantine/add` (`{"from_ms": ..., "to_ms": ..., "where": {"match_type": "probabilistic"}, "reason": ...}`), `/remove`, and `/inspect` (`{"day": 41}`), `POST /api/profiles/recount`, and `/playbook.html?alert=` (Chapter 24), and `GET /healthz` |

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
| 13 | The walk stage (`-stage walk`, with the simulator's `walk` preset): tier-1 and low-CPM countries, three exchanges, CPI, ROAS, and CPC campaigns | Works |
| 13 | The Labels tab: install label delays for click-through and view-through installs, label maturity by day of impression, and attributed vs. caused installs by placement (`python -m adtrain labels`) | Works |
| 14 | Device history from postbacks: installs and purchases credited to us, counted when the postback arrives and logged on the FeatureVector; in walk, device profiles are saved across restarts (`out/state/profiles.gob.gz`) | Works |
| 14 | Conversion datasets with as-of device history, label finality per cutoff, validation checks, and a leaky version for the exercise (Conversions tab) | Works |
| 15 | LightGBM click, install-after-click, and install-after-view models, served by a native Go tree evaluator (`internal/gbm`); deploys check 200 golden examples and refuse a bundle trained on a failed dataset unless forced; CPI bids use the bundle's install rates | Works |
| 16 | Switchable campaign selection (budget-weighted, uniform, CPI-first), with the eligible campaigns, their goals, and the creative count logged for off-policy evaluation; the Evaluate tab: replay, IPS estimates against live runs, and serving periods | Works |
| 17 | Serving performance: stage timings with p50 and p99 for live and load-test requests (`internal/latency`, the Serving tab); the load generator (`cmd/loadgen`, `docker compose run --rm loadgen`), which replays the raw request log open loop at rising rates with an `X-Load-Test` header the bidder answers without writing anything; Go's profiler on the admin port; bitsets for categorical splits; cached creative chances for Thompson sampling; the win model's inputs built once per request and its grid batched; an auction candidate cap with an audit of what it gives up; ONNX Runtime from Go without cgo for comparison (`code/onnxbench`) | Works |
| 24 | Operating: a prediction monitor (each model's hourly mean against the same hour on up to four days before, robust z and PSI, a shift alert after three quarter hours, annotated with the last deploy; `internal/ops/monitor.go`); a daily parity check when the simulated day turns, with an alert; playbooks for every alert (`internal/console/static/playbooks`), linked from the alerts; incident drills (a bad deploy that shifts click predictions, a serving bug in connection_type; bad labels come from the simulator's `bad-labels` scenario); quarantine of postbacks by time range and field, read by every dataset build, with an install label rate check and a recount of device profiles; pacing that starts each day at a gamma of at most 1 | Works |
| 19 | Experiments (`internal/experiment`): shadow scoring on a sample of requests; a staged canary by device hash, with guardrails on the same requests (prediction ratios by placement, fallback share, spend rate), automatic rollback, and promotion; budget-split A/B tests; every bid record carries the experiment and arm | Works |
| 20 | Win rate models (`gbm.WinModel`, the `win` family of the registry): shading by expected surplus over a grid of shares of the media value; exploration multiplies the shaded bid; shading A/B tests against the fixed 0.7; bid records log `shade_from`, `shade`, and `win_prob` | Works |
| 21 | Pricing for goals: installs from both paths, D7 revenue per install by app for ROAS (`values.json` in the bundle), calibration multipliers by campaign, the internal auction by expected surplus with 10% exploration and logged candidates, Thompson sampling over creatives with matured posteriors; selection, creative stats, and Thompson's on/off survive restarts | Works |
| 22 | Pacing by bid modification (`internal/ops/pacing.go`): a PI controller on each campaign's gamma in log space every simulated quarter hour (Kp 0.5, Ki 0.05, bounds 0.3-2, at most 0.2 in ln gamma a step), against the traffic-shaped plan; days start from the replay forecast or yesterday's gamma; manual overbids logged apart; throttling as the backstop; state survives restarts | Works |
| 23 | Incrementality from the ghost-bid holdout (the Evaluate tab, `adtrain lift`), against the simulator's true caused installs | Works |
| 28 | Serving deep models (`internal/deep`, `cmd/deepcheck`): the Chapter 26 DCNv2 run in pure Go from the bundle `adtrain export` writes, with golden vectors checked at load and no allocations per request; the request's share of the first layers computed once for all candidates; versioned app vectors and device states scored with dot products, refused when versions don't match, with both versions' tables kept during a rollover; top-k over a contiguous matrix; the console's Deep models tab (Chapters 25 to 29): datasets, run buttons, each chapter's latest report, the Go check run in the bidder, and an embedding viewer | Works |
| 30 | Value prediction for Ad ROAS (with the simulator's `--scenario ad-roas`): `AD_ROAS` campaigns priced on D3 ad revenue per install, from the bundle's `value` model (LightGBM, Tweedie, served natively with an exp link) when it has one, else the app's average; value A/B tests (`value` experiments) price Ad ROAS bids with a challenger's value model and everything else with the champion | Works |
| 31 | Win curves and the winner's curse: a win bundle with a `curve` target (LightGBM on the request alone for the location of the log price to beat, logistic with the spec's `curve_scale`, served with a raw link) shades with one tree pass per request and gives the market price; its `curse.json` table marks values down by the margin of the media value over that price; curse A/B tests (`curse` experiments) apply a challenger's table to half the devices, with each campaign's budget split | Works |
| 32 | Uplift: lift studies (`study` experiments: the control arm's devices get ghost bids only) and incrementality tilts (`tilt` experiments: install campaigns' values times an uplift bundle's ratio, (mu1 - mu0) / mu_attr, over its average across the bids made, `1 - alpha + alpha r / mean`, capped at 2); uplift bundles under `models/uplift`, tested, never deployed | Works |
| 33 | The unified pricing equation: campaign levers saved in `models/pricing/levers.json` (a second CPI goal with a weight, each goal with the campaign's calibration; a tilt alpha with a pinned uplift bundle); the bid ledger logs each goal's value, weight, and calibration, the tilt and ratio, and the margin; the Pricing tab explains a logged bid term by term and runs the consistency check (through the training service) | Works |
| 34 | Pacing policies in a simulator built from logged auctions: the Pricing tab runs the training service's comparison of the PI controller, MPC, and Q-learning for one campaign on its newest day (the bidder's pacing is unchanged) | Works |
| 18 | Champion and challenger: `DEPLOYED` is the champion, `CHALLENGER` the latest bundle that passed the pipeline; promote, roll back, and a deploy log with the action and the version before; a bundle that failed the pipeline's checks needs force | Works |

## Modes

- `-mode listen`: run the whole bid path and log every request, but never
  bid (Chapter 7).
- `-mode bid`: bid on `-bid-share` of requests (default 1, all of them).
  Start with `-bid-share 0.01`, as Chapter 7 suggests.

## Layout

```
cmd/bidder/          main: flags, the bid and admin servers
cmd/loadgen/         the load test: replays the raw request log at rising rates (Chapter 17)
cmd/deepcheck/       the deep model check: parity, per-request benchmarks, versions (Chapter 28)
internal/openrtb/    the OpenRTB 2.6 fields the bidder reads
internal/normalize/  one internal request, whatever the exchange's quirks
internal/bidder/     the bid path and notice handler: parse, normalize, decide, respond, log
internal/campaigns/  the campaign book, eligibility, and spend from billing notices
internal/profile/    device profiles from the bidder's own traffic
internal/features/   the FeatureVector and its canonicalization rules
internal/policy/     crawl selection and pricing, and the holdout
internal/model/      the click model: tokens, MurmurHash3, predict, the registry and deploys
internal/gbm/        LightGBM trees in Go: conversion bundles, golden checks, their registry
internal/experiment/ shadow, canary, and A/B tests next to the champion
internal/ops/        kill switches, audit log, throttling, alerts, reconciliation
internal/ledger/     the bid ledger, one JSON line per request, per day
internal/stats/      live numbers for the console
internal/latency/    stage timings and their percentiles (Chapter 17)
internal/deep/       the DCNv2 in Go, and versioned vectors and dot products (Chapter 28)
internal/clock/      simulated time from the X-Adsim-Time header
internal/console/    the admin console (static/index.html), playbooks (static/playbooks), and starting training runs
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

(PowerShell versions of every walkthrough command are in
[WALKTHROUGH-windows.md](../../WALKTHROUGH-windows.md).) At 06:00 a
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
