# adtrain: training the click model

The Python side of the click model in Chapters 9 and 11. It reads the
bidder's ledger, builds click examples, trains logistic regression with
FTRL on hashed tokens, validates the model out of time, and publishes an
artifact the Go bidder loads.

```bash
pip install -e code/training[truth,test]
python -m adtrain train --ledger out/ledger --models out/models --sim-out out/sim
python -m adtrain serve --port 8090    # what the console's Train button calls
pytest code/training
```

Two more commands for Chapter 8:

```bash
python -m adtrain parity --ledger out/ledger           # feature parity check
python -m adtrain trace <request_id> --ledger out/ledger  # one request, bid to label
python -m adtrain find --clicked --ledger out/ledger      # request IDs to trace
python -m adtrain labels --ledger out/ledger --sim-out out/sim  # label delays, attributed vs caused (Ch 13)
python -m adtrain dataset --ledger out/ledger --out out/datasets  # conversion dataset (Ch 14), the newest 28 days; --leak for the exercise
python -m adtrain gbm --dataset out/datasets/conversions/conversions-0001 --models out/models --sim-out out/sim  # LightGBM (Ch 15)
python -m adtrain replay --day 19 --a gbm-0006 --b gbm-0008  # rescore a logged day (Ch 16)
python -m adtrain ope --policy uniform  # a selection policy, from logged propensities (Ch 16)
python -m adtrain periods  # what each serving conversion model bought (Ch 16)
python -m adtrain shadow exp-0001  # shadow scores against offline ones (Ch 19)
python -m adtrain abtest exp-0005 --sim-out out/sim  # an A/B test's readout (Ch 19)
python -m adtrain win --sim-out out/sim  # a win rate model on the last 7 days; --no-mtw to ignore reports (Ch 20)
python -m adtrain shadetest exp-0006  # a shading test: win rate, spend, surplus by arm (Ch 20)
python -m adtrain kappa  # calibration multipliers by campaign, from matured installs (Ch 21)
python -m adtrain pace --sim-out out/sim  # the pacing forecast: tomorrow's starting gamma by campaign (Ch 22)
python -m adtrain lift --sim-out out/sim  # incrementality from the ghost-bid holdout, by app (Ch 23)
python -m adtrain quarantine inspect --day 74  # install postbacks by match_type, against the days before (Ch 24)
python -m adtrain quarantine add --day 74 --where match_type=probabilistic --reason "..."  # skipped by every dataset build (Ch 24)
python -m adtrain quarantine list  # and: quarantine remove q-0001
python -m adtrain onnx --bundle out/models/conversion/gbm-0012  # export to ONNX, time ONNX Runtime; needs onnx, onnxruntime, onnxmltools (Ch 17)
python -m adtrain pipeline --ledger out/ledger --models out/models  # dataset to challenger, with checks (Ch 18); --leak or --inject no-features to see it stop
python -m adtrain abtest exp-0002 --goal AD_ROAS --sim-out out/sim  # a value A/B test: D3 ad ROAS by arm (Ch 30)
python -m adtrain win --curve --curse --sim-out out/sim  # also a win curve and a winner's curse table (Ch 31)
python -m adtrain abtest exp-0002 --sim-out out/sim  # a curse A/B test: CPI by arm (Ch 31)
python -m adtrain uplift --sim-out out/sim  # uplift models from the newest lift study (Ch 32)
python -m adtrain abtest exp-0002 --goal INSTALL --sim-out out/sim  # a tilt A/B test: cost per caused install by arm (Ch 32)
python -m adtrain explain --campaign 16  # one logged bid, term by term (Ch 33)
python -m adtrain frontier --campaign 16 --day 15 --budget 88 --target-roas 0.5  # the consistency check, and the next day against it (Ch 33)
python -m adtrain rlpace --campaign 21  # PI, MPC, and Q-learning pacing in a logged-auction simulator (Ch 34)
```

**Value models (Chapter 30).** On a market run with `--scenario ad-roas`,
datasets carry each install's D3 revenue (`d3_revenue`, final 3.5 days
after the install) next to D7, and `adtrain gbm` adds a `value` model to
the bundle: LightGBM with a Tweedie loss on the D3 ad revenue of installs
of hypercasual apps (every install of an app with an Ad ROAS campaign,
whichever campaign brought it), from the same inputs as the install
models. The report compares it on the newest two days with final labels
against the per-app average the bidder uses without it, and against a
least-squares model on log revenue, by correlation with the outcomes and
with the simulator's true value, calibration, the share of revenue in the
top fifth of predictions, and by country. The bidder serves it natively,
like the other trees, with an exp link. Value models for purchase revenue
(two-part, Tweedie, ZILN on sparse D7 purchases) are left to the reader:
they need far more days of data.

The parity check recomputes the FeatureVector of every bid whose raw
context the bidder logged (5% of bids), with a second implementation of
the bidder's normalization and features (`features_ref.py`), and reports
mismatches by feature, exchange, and day. Both implementations are tested
on the same cases, from `code/dsp/internal/features/testdata`.

## The pipeline

| Step | Module | Chapter |
|---|---|---|
| Click examples: billing notices joined to the bid ledger on the request ID, labels from the first click within one hour, only impressions whose window has closed | `dataset.py`, `ledger.py` | 9 |
| Tokens: one per string, bucketed numerics, many-hot sets, crosses, missing values as their own tokens; MurmurHash3 into 2^24 buckets | `tokens.py` | 9 |
| FTRL-Proximal: the bias started at the base rate, one pass over the shuffled training days, then the bias refit so training clicks match; optional negative downsampling | `ftrl.py`, `train.py` | 9 |
| Out-of-time split: train on every day but the last, evaluate on the last (with one day, by hour) | `train.py` | 9 |
| Metrics: log loss, normalized entropy, ROC and PR curves with their areas, calibration by decile, by-segment tables, and the simulator's true click rates when its output is there | `metrics.py` | 9 |
| Readouts: top and bottom tokens and crosses by weight | `train.py` | 9 |
| Install rates after a click, from MMP postbacks joined to our clicks, by campaign, country, and placement, each shrunk toward its parent with k = 200 (segment, campaign, all campaigns, a 3% prior); published in the bundle for CPI pricing | `install_rates.py` | 8, 10 |
| Stability: cosine similarity of the weights with the deployed model, and how much predictions moved (no deployed model, no comparison) | `train.py` | 11 |
| Checks: enough evaluation clicks, AUC, calibration, beating what the bidder logged, and stability | `train.py` | 11 |
| The artifact: `model.json`, `weights.bin`, `golden.jsonl`, `report.json` | `artifact.py` | 9, 11 |
| Conversion datasets: one row per impression with click, install (click and view), and 7-day revenue labels, each with its cutoff and a flag for whether it is final; device install history as of the bid, from the time each postback reached the bidder; checks for schema, logged features, label joins, backfill against what the bidder logged, volume, and label rate | `datasets.py` | 14 |
| LightGBM conversion models: click, install after a click, install after a view (no click), categorical features coded from one vocabulary with an `__other__` code, evaluated on the newest days whose labels are final, with calibration by placement and by predicted value; a bundle of `dump_model` trees, `golden.jsonl`, and `report.json` | `gbm.py` | 15 |
| Replay: a logged day's priced bids rescored with two bundles and repriced as the bidder does; wins from an isotonic win curve by exchange, checked against the simulator's highest other bid; bids that change hands; a numpy evaluator of LightGBM's JSON dump for older bundles | `evaluate.py` | 16 |
| Off-policy evaluation: IPS and self-normalized IPS with 95% intervals, effective sample size, for the uniform and CPI-first selection policies from any logged policy | `evaluate.py` | 16 |
| A/B readout: cost per install by arm from final installs, the treatment/control ratio with a 95% interval from resampling devices, and the same ratio from the simulator's true install probabilities; shadow parity | `evaluate.py` | 19 |
| Win rate model: LightGBM on the win notice, the bid as ln(bid) and ln(bid/floor) with a monotone constraint, request features only (no campaign or ad), the minimum-bid-to-win reports as weighted rows above and below each reported price; calibration by bid level, exploration multiplier, and exchange, and against the simulator's truth at other bids | `win.py` | 20 |
| Win curve: LightGBM on the request alone (no bid) for the location of the log price to beat, logistic with a scale chosen by grid, trained with a custom censored loss (won bids: the price was at most the bid; lost: above; reported minimum bids to win: the price); the winner's curse table, installs per predicted install on won install-campaign impressions priced with the conversion model, by the media value over the curve's market price, shrunk toward 1 | `win.py` | 31 |
| Uplift models from a lift study: one row per device, app, and day, labeled by an install from any source within a day; a T-learner (LightGBM on devices bid on and on held-out devices), a transformed-outcome model, and a model of installs credited to us, on a few inputs (requests per day, our impressions today, app, placement, exchange, country); Qini curves on held-out devices; the share of installs caused, measured, predicted, and true, by fifth of requests per day; the incrementality ratio averaged over the bids made, for the tilt | `uplift.py` | 32 |
| Unified pricing: a logged bid explained term by term (each goal's value, calibration, and weight, the curse correction, the tilt, pacing, margin, shading, exploration, floor and cap), checked against the logged media value; the consistency check, the day's bids for one campaign replayed at pacing multipliers from 0.3 to 2 for spend and predicted installs and revenue, against a budget and a CPI or ROAS target, and the next day's actual spend and installs against the forecast | `pricing.py` | 33 |
| Pacing policies in a simulator built from one campaign's logged auctions (base bids, the day's win curves by exchange, billed share, advertiser value; skipped requests filled in by hour; the budget a hard stop), checked by replaying the bidder's multipliers; the PI controller ported from the bidder, MPC re-planning from the day before, and tabular Q-learning on randomized days (traffic and competition, surges); value won, delivery, tracking error, volatility | `rlpace.py` | 34 |
| Calibration multipliers: realized over predicted installs on billed impressions at least 3 days old, the last 10 days, each day weighted 0.8 per day of age, shrunk toward the overall with k = 20 predicted installs, bounded to 0.5-2; D7 revenue per install by app in each bundle, shrunk with k = 20 installs | `kappa.py`, `gbm.py` | 21 |
| Pacing forecast: the newest day's bids per campaign, without that day's gamma, repriced over a grid of gammas with the observed win rates by exchange and bid level and the billed share of wins; requests skipped for budget counted at the average; the gamma whose forecast spend meets the budget | `pacing.py` | 22 |
| Incrementality: per advertiser app, devices with a real bid against holdout devices with a ghost bid; installs from any source (our postbacks plus the advertiser's shared non-attributed installs) after the first bid; ITT, exposure share, lift per exposed device, incremental installs, incrementality, iCPI, with Agresti-Caffo intervals; the simulator's caused installs on the same devices; a smaller holdout by a second hash | `lift.py` | 23 |
| The pipeline: build a dataset, validate it, train, validate the models (predictions vary; click AUC and calibration; install calibration; top fifth of install predictions against what happened; no worse than the champion when both score the same rows), publish as the challenger; one record per run, and lineage in each bundle's report | `pipeline.py` | 18 |

Every number in the report is computed by reading the published artifact
back and predicting the way the bidder does (float32 weights, the
downsampling correction), so the report describes the model that will be
served.

## The artifact

```
out/models/click/click-0003/
  model.json     the token spec, hash, bias, and downsampling rate
  weights.bin    nonzero weights, sorted (uint32 bucket, float32 weight) pairs
  golden.jsonl   200 evaluation examples with this code's predictions
  install_rates.json  smoothed click-to-install rates, for CPI campaigns
  report.json    metrics, curves, calibration, readouts, stability, checks
```

The bidder never computes any of the spec from its own configuration, and
it refuses to deploy a model whose golden predictions it can't reproduce.

## Parity tests

`python -m adtrain.fixture code/dsp/internal/model/testdata` trains a
small model on synthetic data and writes it, with the tokens and buckets
for a few examples, where the Go tests read them. `pytest` checks that
Python still produces the same tokens; `go test` checks that Go does, and
that it reproduces every golden prediction. Change the token rules on one
side only and a test fails. After a deliberate change, regenerate the
fixture.
