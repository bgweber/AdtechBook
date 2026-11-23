# Latency

**Owner:** bidder on-call. **Severity:** page.

## What it means

The 99th percentile of request time inside the bidder is above 100 ms, so exchanges are timing out some of our bids.

## Confirm it

- The Serving tab: which stage grew. A slow stage at low load is code; every stage slow at high load is a CPU that can't keep up, and requests are queueing.
- Go's profiler, while it's slow:

  ```bash
  go tool pprof -top http://localhost:8081/debug/pprof/profile?seconds=10
  ```

## Safe first actions

- Roll back a bidder or model deploy that lines up with it.
- Cap the internal auction to shed work: `curl -s -X POST localhost:8081/api/policy/auction-cap -d '{"cap": 3}'`.

## Escalate

To bidder engineering, with the profile.

## Verify recovery

p99 back under the deadline on the Serving tab, and the alert resolves.
