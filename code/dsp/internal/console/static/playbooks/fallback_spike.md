# Fallback spike

**Owner:** ML team. **Severity:** page.

## What it means

More than 20% of bids priced while a model is deployed were priced by the fallback ladder instead: the model is switched off, failed to load, or predicts outside its sane bounds (0.01% to 50%).

## Confirm it

- The kill switches on the Controls tab: is the model switch on?
- The bidder's log for a failed load.
- The decision reasons on the Live tab: `fallback_bounds` means predictions out of bounds, which usually means a broken input.

## Safe first actions

- If someone switched the model off, find out why in the audit log before switching it back on.
- If predictions are out of bounds after a deploy, roll back: `curl -s -X POST localhost:8081/api/conversion/rollback`.

## Escalate

To bidder engineering if the model fails to load and no deploy changed it.

## Verify recovery

The fallback share drops below 20% within the hour and the alert resolves.
