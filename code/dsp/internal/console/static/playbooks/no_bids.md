# No bids

**Owner:** bidder on-call. **Severity:** page.

## What it means

Requests keep arriving but the bidder bid on fewer than 0.5% of them in the last simulated hour.

## Confirm it

- The kill switches on the Controls tab.
- The decision reasons on the Live tab: which filter is removing everything (budgets spent, no eligible campaign, bad requests).
- The bidder's log for errors.

## Safe first actions

Undo the last switch or deploy that lines up with it.

## Escalate

To bidder engineering if no switch or deploy explains it.

## Verify recovery

Bids per request return to normal within the hour.
