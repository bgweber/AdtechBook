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

## The Part IV Dataset

To download it, unzip [`part4-dataset.zip`](https://github.com/bgweber/AdtechBook/releases/download/part4-data/part4-dataset.zip), from the `part4-data` release, into `out\`. To build it instead:

```powershell
docker compose down; Remove-Item -Recurse -Force out
$env:ADSIM_PRESET="walk"; $env:BIDDER_STAGE="walk"; $env:ADSIM_SCENARIO="ad-roas,full-share"
docker compose build
docker compose up -d bidder trainer
$env:ADSIM_DAYS=7; docker compose run --rm sim
```

After deploying the conversion bundle on the Conversions tab, the second week:

```powershell
docker compose run --rm sim
```

The `docker compose exec trainer ...` and `docker compose run --rm --no-deps trainer ...` commands run as written.

## Chapter 29: Retraining and Real-Time Learning

To download the shift dataset, unzip [`part4-shift.zip`](https://github.com/bgweber/AdtechBook/releases/download/part4-data/part4-shift.zip), from the `part4-data` release, into `out\`. To build it instead, on the Part IV market with the bidder and trainer up:

```powershell
$env:ADSIM_SCENARIO="ad-roas,full-share,new-game"; $env:ADSIM_DAYS=4; docker compose run --rm sim
docker compose exec trainer python -m adtrain snapshot --ledger /out/ledger --sim-out /out/sim --out /out/snapshot-shift
```

## Chapter 30: Value Prediction

Starting over on the market with ad revenue, and keeping the variables set for the rest of the chapter:

```powershell
docker compose down; Remove-Item -Recurse -Force out
$env:ADSIM_PRESET="walk"; $env:BIDDER_STAGE="walk"; $env:ADSIM_SCENARIO="ad-roas"
$env:ADSIM_DAYS=9; docker compose up --build
```

The value A/B test's three days, from a second window with the same variables set:

```powershell
$env:ADSIM_DAYS=3; docker compose run --rm sim
```

For the laptop scale, add `$env:ADSIM_SCALE="laptop"`. The `docker compose exec trainer ...` commands run as written.

## Chapter 31: Win Rate Curves and the Winner's Curse

With the Chapter 30 variables still set and its bundle deployed, the five days before training, from a second window:

```powershell
$env:ADSIM_DAYS=5; docker compose run --rm sim
```

The curse A/B test's three days, twice (once before **Stop**, once after):

```powershell
$env:ADSIM_DAYS=3; docker compose run --rm sim
```

The benchmark, with Go installed:

```powershell
cd code\dsp; $env:GBM_WIN_BUNDLE="..\..\out\models\win\win-0001"; go test ./internal/gbm -run x -bench WinShadeFromEnv; cd ..\..
```

The `docker compose exec trainer ...` commands run as written.

## Chapter 32: Uplift Modeling

Starting over on the market with sure things, with the bidder and trainer first:

```powershell
docker compose down; Remove-Item -Recurse -Force out
$env:ADSIM_PRESET="walk"; $env:BIDDER_STAGE="walk"; $env:ADSIM_SCENARIO="ad-roas,uplift"
docker compose build
docker compose up -d bidder trainer
```

After **Start a lift study 50/50**, the nine days:

```powershell
$env:ADSIM_DAYS=9; docker compose run --rm sim
```

The tilt test's three days, twice (once before **Stop**, once after):

```powershell
$env:ADSIM_DAYS=3; docker compose run --rm sim
```

The `docker compose exec trainer ...` commands run as written.

## Chapter 33: The Unified Pricing Equation

Continuing Chapter 32's market, with its variables still set:

```powershell
docker compose build
docker compose up -d bidder trainer
```

After saving campaign 16's levers on the Pricing tab, the three days:

```powershell
$env:ADSIM_DAYS=3; docker compose run --rm sim
```

The `docker compose exec trainer ...` commands run as written.

## Chapter 34: Pacing with Reinforcement Learning

Continuing Chapter 33's market, with its variables still set:

```powershell
docker compose build
docker compose up -d bidder trainer
```

After clicking **PI controller** on the Pricing tab, a normal day, then a day with a surge:

```powershell
$env:ADSIM_DAYS=1; docker compose run --rm sim
$env:ADSIM_SCENARIO="ad-roas,uplift,traffic-surge"; docker compose run --rm sim
$env:ADSIM_SCENARIO="ad-roas,uplift"
```

The last line sets the scenario back, so later runs don't surge. The `docker compose exec trainer ...` commands run as written.
