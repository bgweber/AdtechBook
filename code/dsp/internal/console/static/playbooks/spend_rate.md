# Spend rate

**Owner:** bidder on-call. **Severity:** page.

## What it means

A campaign spent more than four times its plan in the last simulated hour, is well ahead of its plan for the day, and the hour took at least a tenth of its budget. Usual causes: a target or bid multiplier changed in the campaign system, a pacing state that didn't load, or a model that overpredicts for that campaign's supply.

## Confirm it

- The audit log on the Controls tab for a target or budget change.
- The campaign's pacing chart on the Pricing tab.

## Safe first actions

Pause the campaign if the cause isn't clear within minutes; a paused campaign loses some delivery, a runaway one loses the budget:

```bash
curl -s -X POST localhost:8081/api/ops/switch -d '{"kind": "campaign", "key": "3", "on": true}'
```

## Escalate

To the account team, who can confirm whether the change was intended.

## Verify recovery

Spend per hour returns to plan, and the alert resolves.
