# Prediction collapse

**Owner:** ML team. **Severity:** page.

## What it means

The click predictions have lost their spread (p99 within 1.5 times p50), or the median moved threefold from the hour before. A model that predicts nearly the same value for every request isn't ranking anything. The classic cause is an untrained or badly trained model that reached production; a missing input that every row now shares is the other.

## Confirm it

- The alert's p50 and p99, and the deploy history: `curl -s localhost:8081/api/conversion/history`.
- The model's report on the Conversions tab: a validation AUC near 0.5 means it learned nothing.

## Safe first actions

Roll back to the previous model:

```bash
curl -s -X POST localhost:8081/api/conversion/rollback
```

If there's no previous model, switch the model off; the fallback ladder prices from segment rates:

```bash
curl -s -X POST localhost:8081/api/ops/switch -d '{"kind": "model", "on": true}'
```

## Escalate

To whoever owns the training pipeline, if its checks passed a model like this: the checks need a new threshold.

## Verify recovery

p50 and p99 separate again within the hour, and the alert resolves.
