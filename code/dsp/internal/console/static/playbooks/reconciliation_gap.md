# Reconciliation gap

**Owner:** bidder on-call. **Severity:** ticket.

## What it means

Our count of billed spend differs from an exchange's invoice by more than 2% for a day: one side is missing events. The usual cause is an exchange that stopped sending billing notices for one ad format.

## Confirm it

The reconciliation table on the Controls tab, by exchange and placement: a gap in one format points at the exchange.

## Safe first actions

None in the bidder: pacing and budgets rely on billing notices, so a campaign on that format may overspend; watch its spend.

## Escalate

To the exchange's integration contact, with the day, the format, and the counts.

## Verify recovery

The next day's invoice matches within 2%.
