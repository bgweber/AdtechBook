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
```

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
