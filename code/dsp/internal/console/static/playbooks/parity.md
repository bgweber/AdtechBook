# Parity failure

**Owner:** ML team. **Severity:** ticket.

## What it means

The daily parity check recomputed the FeatureVector of yesterday's sampled bids from their raw requests, with the training code, and more than 0.1% of rows have a different value for some feature. Training and serving disagree, so the models are scoring inputs they weren't trained on. Usual causes: a bidder deploy that changed how a feature is computed, a canonicalization change, or an upstream field that changed format.

## Confirm it

- The parity report, with examples of logged and recomputed values: the Models tab, or

  ```bash
  curl -s -X POST localhost:8081/api/parity/day -d '{"day": 41}'
  docker compose exec trainer python -m adtrain parity --ledger /out/ledger
  ```

- Which exchanges and which days: a feature that broke on one day after a deploy is a code change; one that drifts is upstream.
- The audit log for a bidder deploy on the day it started.

## Safe first actions

- If a bidder deploy lines up, roll the bidder back.
- If the feature matters to the serving model, switch it off with the feature kill switch while it's fixed, so the model sees it as missing rather than wrong:

  ```bash
  curl -s -X POST localhost:8081/api/ops/switch -d '{"kind": "feature", "key": "connection_type", "on": true}'
  ```

## Escalate

To bidder engineering for a serving change; to the data owner for an upstream one.

## Verify recovery

The next day's parity check passes, and the alert resolves. The days with mismatches stay in the ledger: build datasets so that training doesn't learn from the mismatched values, or check that it doesn't matter.
