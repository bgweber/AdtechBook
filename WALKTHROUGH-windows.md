# The Walkthrough on Windows: PowerShell Versions

The [walkthrough](WALKTHROUGH.md) gives its commands for a Linux shell, bash. On Windows you have two choices:

- **WSL 2 (recommended).** Docker Desktop runs on WSL 2 by default. Open a WSL terminal (for example, Ubuntu from the Microsoft Store), clone the repository there, and run the walkthrough's commands as written.
- **PowerShell.** Every `docker compose` command works the same in PowerShell. The differences are in how a few things are written, below, and this page lists the walkthrough's commands that need them, chapter by chapter.

## What changes in PowerShell

| bash | PowerShell | Note |
|---|---|---|
| `ADSIM_DAYS=7 docker compose up` | `$env:ADSIM_DAYS=7; docker compose up` | In bash, a variable before a command applies to that command only. In PowerShell it stays set for the rest of the window, so clear it when you're done. |
| (nothing to clear) | `Remove-Item Env:ADSIM_DAYS` | Clears one variable. Forgetting to clear `ADSIM_SCENARIO` reruns the scenario on the next run. |
| `rm -rf out` | `Remove-Item -Recurse -Force out` | Deletes the output folder. |
| `\` at the end of a line | `` ` `` (backtick) at the end of a line | Or put the whole command on one line. |
| `curl -s -X POST localhost:8081/api/... -d '{"cap": 3}'` | `Invoke-RestMethod -Method Post http://localhost:8081/api/... -Body '{"cap": 3}'` | `curl.exe` also works, but quotes inside JSON need escaping. |

To see what's set: `Get-ChildItem Env: | Where-Object Name -Match '^(ADSIM|BIDDER|LOADGEN)_'`. To clear all of them at once:

```powershell
Get-ChildItem Env: | Where-Object Name -Match '^(ADSIM|BIDDER|LOADGEN)_' | ForEach-Object { Remove-Item "Env:$($_.Name)" }
```

## Before you start

```powershell
$env:ADSIM_DAYS=7; docker compose up --build
```

To start over:

```powershell
docker compose down
Remove-Item -Recurse -Force out
```

## Chapter 7: Listening Before Bidding

```powershell
docker compose down; Remove-Item -Recurse -Force out
$env:BIDDER_MODE="listen"; $env:ADSIM_DAYS=7; docker compose up --build
Remove-Item Env:BIDDER_MODE
```

Bidding on 1% of requests:

```powershell
docker compose down
$env:BIDDER_BID_SHARE=0.01; docker compose up --build
Remove-Item Env:BIDDER_BID_SHARE
```

## Chapter 8: Logging at Bid Time

```powershell
docker compose down
$env:BIDDER_SKEW="conn"; $env:ADSIM_DAYS=1; docker compose up --build
Remove-Item Env:BIDDER_SKEW
```

## Chapter 9: A Click Model

```powershell
docker compose down; Remove-Item -Recurse -Force out
$env:ADSIM_DAYS=7; docker compose up --build
```

## Chapter 12: Operating the First DSP

```powershell
$env:ADSIM_SCENARIO="wrong-target,lost-billing"; docker compose run --rm sim
Remove-Item Env:ADSIM_SCENARIO
```

## Part III: A Competitive DSP

```powershell
docker compose down; Remove-Item -Recurse -Force out
$env:ADSIM_PRESET="walk"; $env:BIDDER_STAGE="walk"; $env:ADSIM_DAYS=7; docker compose up --build
```

Keep `ADSIM_PRESET` and `BIDDER_STAGE` set to `walk` for the rest of Part III. If you open a new PowerShell window, set them again before running anything, or the bidder restarts in the crawl stage.

## Chapter 17: Serving Performance

```powershell
$env:LOADGEN_LABEL="before"; docker compose run --rm loadgen
Invoke-RestMethod -Method Post http://localhost:8081/api/policy/auction-cap -Body '{"cap": 3}'
$env:LOADGEN_LABEL="cap 3"; docker compose run --rm loadgen
```

## Chapter 21: Pricing for Goals

```powershell
$env:ADSIM_SCENARIO="creative-test"; $env:ADSIM_DAYS=1; docker compose run --rm sim
```

Keep `ADSIM_SCENARIO=creative-test` set for every later run of the same market.

## Chapter 24: Operating a Competitive DSP

```powershell
$env:ADSIM_SCENARIO="bad-labels"; $env:ADSIM_DAYS=1; docker compose run --rm sim
Remove-Item Env:ADSIM_SCENARIO
$env:ADSIM_DAYS=2; docker compose run --rm sim
```

Quarantining from the command line instead of the console:

```powershell
docker compose exec trainer python -m adtrain quarantine add --datasets /out/datasets `
  --day 74 --where match_type=probabilistic --reason "probabilistic matching turned on by mistake"
```

Recounting the device profiles:

```powershell
Invoke-RestMethod -Method Post http://localhost:8081/api/profiles/recount
```
