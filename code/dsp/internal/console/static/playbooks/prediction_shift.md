# Prediction shift

**Owner:** ML team. **Severity:** page.

## What it means

A model's mean prediction over the last simulated hour is far from the same hour on the days before: more than 4 robust standard deviations, and more than about 5%, for three quarter hours in a row. Almost every model problem changes the distribution of predictions within minutes. The usual causes, most likely first:

1. A deploy: a new conversion bundle, vocabulary, or bidder build. The alert names the last change before the shift began.
2. An input problem: a feature missing, or computed differently in serving (see the parity playbook).
3. A real change in traffic: a new exchange, placement, or campaign mix. These are rarer, and usually slower.

## Confirm it

- On the Controls tab, the **Prediction monitor** chart: the model's hourly mean against the band of the baseline days. A deploy shows as a step at the time of the change.
- The deploy history: `curl -s localhost:8081/api/conversion/history`.
- Calibration on clicks, which arrive within the hour: on the Live tab, the campaigns table compares predicted and actual click rates. A model that predicts 10% too high has a ratio near 0.9.
- Whether it's everywhere or one segment: the bid ledger, by exchange and placement (`python -m adtrain find`, then `trace`).

## Safe first actions

- If the shift lines up with a deploy, roll it back. Don't debug first:

  ```bash
  curl -s -X POST localhost:8081/api/conversion/rollback
  ```

- If it lines up with no deploy and predictions are far too high, switch the model to its fallback while you look:

  ```bash
  curl -s -X POST localhost:8081/api/ops/switch -d '{"kind": "model", "on": true}'
  ```

## Escalate

- To bidder engineering, if no model deploy lines up and the shift is limited to some hosts or exchanges.
- To the account team, if spend or CPI moved enough for advertisers to notice.

## Verify recovery

The alert resolves after two calm quarter hours, once the hourly mean is back inside the baseline band. Then find out what the bad build did differently before deploying again, and add a check that would have stopped it.
