# AGENTS.md: Running the Book's Exercises with an AI Agent

This file is for AI coding agents, such as Claude Code, Codex, Cursor, or
Gemini CLI, that help a reader run the companion code of *Machine Learning in
Mobile AdTech*, chapter by chapter. It complements
[WALKTHROUGH.md](WALKTHROUGH.md), which is written for people: the walkthrough
has the commands and what you should see, and this file tells an agent how
to guide a reader through them, check their results, and get them unstuck.

## For readers: using this file

Most agents read `AGENTS.md` on their own when you open the repo. For Claude
Code, create a `CLAUDE.md` in the repo root with one line, `@AGENTS.md`
(the repo ignores `CLAUDE.md`, so yours stays yours). Then ask the way you'd
ask a colleague who knows the code:

- "I'm starting Chapter 15. Give me the commands to run, including the cd."
- "Here's my output. Did it work, and does it match the book?"
- "Is it still running? Nothing has printed for two minutes."
- "Do I need to stop Docker and delete `out` before this chapter?"
- "Explain step 2 before I run it."
- "Is there anything to look at on the console for this chapter?"
- "I got this error." (paste it)

Paste outputs and screenshots as they are; the agent's job is to read them
against the book.

## For agents: how to help

**The reader runs the commands.** Give complete blocks they can paste,
starting with the `cd` to the repo root, in order, with nothing to fill in
except values you can't know (and say which). Don't run commands on their
machine unless they ask you to, and never run `git` commands, delete
folders, or push without being asked.

**Match their shell.** The walkthrough's commands are bash (Linux, macOS, WSL
2). On Windows without WSL, give PowerShell: `$env:VAR="value"` instead of
`VAR=value` before a command (and note that it stays set for the rest of the
session), `Remove-Item -Recurse -Force out` instead of `rm -rf out`, `dir`
instead of `ls`, and backticks instead of backslashes to continue lines.
[WALKTHROUGH-windows.md](WALKTHROUGH-windows.md) has the PowerShell versions
of the commands that differ. Ask once if you can't tell.

**Read before answering.** For each chapter, read its section of
WALKTHROUGH.md and the chapter's "Try it on the simulator" box in
`chapters/NN-*.qmd`. The walkthrough is the source of truth for commands
and for the numbers to compare with ("In ours, ..."). When you're unsure what
a command does or prints, read the code: `code/sim` (adsim, the simulated
market), `code/dsp` (the Go bidder and its console), `code/training`
(adtrain, the training service and every `python -m adtrain` command).

**Establish the state first.** Most problems come from running a chapter on
the wrong state. Before giving commands, know:

- which market `out/` holds (`crawl` for Part II, `walk` for Part III, and
  which scenarios, such as `ad-roas`, `uplift`, `full-share`), and how many
  days it has played;
- what's running (`docker compose ps`);
- whether the code changed since the images were built (after the reader
  pulls new code, `docker compose build` before anything else; `up --build`
  builds only the services it starts, so build all of them explicitly when a
  chapter changes the simulator).

**Answer first, one step at a time.** Say what each command does in a
sentence, how long it takes, and what it prints when it works. Say when a
command prints nothing until it's done. When the reader pastes output, say
first whether it worked, then what to compare.

**Judge results honestly.** Tell the reader what should match the book and
what shouldn't (below). If a result differs, find out why before calling it
fine or broken. Never invent numbers or results; if the walkthrough doesn't
say what to expect, say so.

## The setup

- `docker compose up --build` starts the **bidder** (bids on port 8080, the
  console at <http://localhost:8081>), the **trainer** (the training service,
  used by the console's buttons), and the **sim** (plays simulated days
  against the bidder, then exits; the bidder and trainer keep running).
- `docker compose run --rm sim` plays more days on the same market,
  continuing from the last day in the ledger ("continuing at day 9" is
  normal). Days to play, the market, and scenarios come from environment
  variables (`ADSIM_DAYS`, `ADSIM_PRESET`, `BIDDER_STAGE`, `ADSIM_SCENARIO`;
  the table is in WALKTHROUGH.md). Keep a chapter's variables set for every
  command in it.
- `docker compose exec trainer python -m adtrain ...` runs a training
  command in the running trainer. `docker compose run --rm --no-deps trainer
  ...` runs one in a fresh container, without the bidder: Part IV's commands
  work this way; their reports also show on the console's Deep models tab, which can run them too.
- Everything lands in `./out`, mounted as `/out` in every container. Inside a
  container, always use `/out/...` paths: a relative path writes inside the
  container, and the file is gone when it exits.
- The console's tabs: Live, Models, Labels, Conversions, Evaluate,
  Experiments, Pricing, Serving, Deep models, Controls. Refresh the page
  after the bidder restarts.

**Starting over.** `docker compose down`, then delete `out` (bash `rm -rf
out`; PowerShell `Remove-Item -Recurse -Force out`). Stop the bidder before
deleting: a bidder that's shutting down writes its device profiles to
`out/state` and can recreate files. Start over only when the walkthrough
says to (Chapter 7, Chapter 9, Part III, Chapters 30 and 32, and building
the Part IV dataset). Otherwise continue the market. Never delete `out`
when the chapter reads something in it, such as `out/snapshot` in Part IV.

## Checking a run

**Should match the book closely:** anything from a frozen dataset (Part
IV's AUCs, calibrations, and counts, from `part4-dataset.zip` and
`part4-shift.zip`), and the shapes and stories of the simulator runs: which
campaign overspends, which model wins, the direction of each effect.

**Won't match exactly:**

- Simulator runs: the market is seeded, but how far the run has got, which
  day the reader trained on, and what they clicked on the console change the
  numbers. Compare with the walkthrough's numbers by size and direction.
- Timings: they depend on the machine and on what else is running.
- Small differences between machines in PyTorch and LightGBM runs: the
  third decimal of a deep model's error, or a few tree splits.

A result in the wrong direction, a count of zero, or a check that fails is
worth investigating; a number a few percent off usually isn't.

## Common situations

- **"Is it still running?"** The simulator prints a line per day. Most
  `adtrain` commands print their report only at the end: Part IV's take a
  few minutes and look frozen until then. `docker compose ps` shows what's up.
- **"Debug this Compose error with Gordon" in the output.** A Docker Desktop
  tip, not an error. Read the lines after it.
- **Network or DNS errors during a build,** often right after the computer
  wakes from sleep. Wait for the network, then run the build again.
- **The console shows old behavior, or a new button is missing.** The images
  are stale: `docker compose build`, then restart.
- **"The simulator started from day 9."** It continues the ledger. That's
  right unless the chapter says to start over.
- **A file the chapter needs is missing.** Check where the command wrote it
  (`/out/...` or a relative path inside the container), and whether an
  earlier step was skipped.
- **Port 8080 or 8081 is busy.** Another run is up: `docker compose down`.
- **Out of memory at the laptop scale.** Build datasets with
  `--sample-rate 0.2`, as Chapter 30's section says.

## Chapter by chapter

Each entry says what the exercise is for and what to watch for. The
commands are in WALKTHROUGH.md under the same heading.

### Part II: a first DSP (the `crawl` market)

- **5, Starting from scratch.** `up --build`, then the Live tab, and `find`
  and `trace` on one request. Success: a trace with the decision,
  FeatureVector, win, billing, and click.
- **6, Eligibility.** Same run: the eligibility filters on the Live tab, and
  `find --reason` for no-bids.
- **7, Listening.** Starts over in listen mode for 7 days, then bids on 1%.
  The traffic tables fill; wins appear only after the 1% restart.
- **8, Logging.** A skewed bidder (`BIDDER_SKEW=conn`) for one day, then the
  parity check flags connection type. Remind the reader to restart without
  the skew afterward.
- **9, A click model.** Starts over for 7 days, then **Train** on the Models
  tab and the `--no-crosses` ablation. The top-decile calibration story
  involves bots (`is_bot` in `out/sim/truth/devices.parquet`).
- **10, Pricing a click.** `trace` on a CPI bid and an explored one; the
  `pricing:` line has every term.
- **11, Shipping.** Deploy, play a day, train and deploy again, roll back.
- **12, Operating.** The `wrong-target,lost-billing` scenario for a day; the
  spend alert, the kill switch, reconciliation; then the constant model and
  its prediction alert.

### Part III: a competitive DSP (the `walk` market)

Starts over with `ADSIM_PRESET=walk BIDDER_STAGE=walk` for 7 days; keep both
set for every command in Part III.

- **13, Labels.** The Labels tab: delays, maturity, attributed against
  caused.
- **14, Datasets.** An as-of dataset passes its checks, a leaky one fails
  the device-history check. That failure is the point.
- **15, LightGBM.** The leaky bundle scores better offline and worse live.
  Deploying it needs **Force deploy**. Live comparisons are small numbers at
  the tiny scale.
- **16, Offline evaluation.** Replay, off-policy evaluation, and serving
  periods on the Evaluate tab. Replay's two checks must pass first.
- **17, Serving performance.** Play a day before the load test (it needs the
  campaign book). Latency depends on the machine; the knee and the cap's
  trade-off are what to compare.
- **18, Pipeline and registry.** Run it, promote, roll back, then break it
  twice on purpose: both breaks should stop the pipeline.
- **19, Experiments.** Shadow, canary (break it at 25% and watch the
  rollback), and an A/B test read out after the installs arrive.
- **20, Bid shading.** Two win models, with and without the minimum-bid-to-win
  reports, then a shading A/B test.
- **21, Pricing for goals.** Run the pipeline and promote first. Internal
  auction, calibration multipliers, and Thompson sampling with the
  `creative-test` scenario, kept on for later runs of the same market.
- **22, Pacing.** The forecast, the PI controller, and the overbid drill:
  campaigns pinned at 2 or near 0.3 are part of the story, not bugs.
- **23, Incrementality.** The holdout's lift by app; wide intervals at the
  tiny scale are expected.
- **24, Operating.** The `bad-labels` incident: the install label rate check
  fails, quarantine, rebuild, retrain, recount. The failing check is the
  point.

### Part IV: a state-of-the-art DSP

Part IV's exercises run on the smallest market, in minutes, and leave
scaling up to the reader: the code scales, the compute is theirs. Expect
honest results at this scale: trees often beat deep models, and the
chapters say so.

- **The Deep models tab** on the console runs Chapters 25 to 29, 35, and 36 with a button each and shows their latest
  reports, from the tab or a terminal, and has an embedding viewer for the vectors Chapters 26 and 27 learn.
  It needs the bidder and trainer up (`docker compose up -d bidder trainer`) and the datasets in `out/`.
  A job that fails shows its error on the tab; the trainer's log (`docker compose logs trainer`) has the
  traceback.
- **The Part IV dataset.** Download `part4-dataset.zip` from the releases and
  unzip it into `out/` (files in `out/snapshot/`), or build it (about twenty
  minutes). Chapters 25 to 28, 35, and 36 read it; Chapter 29 has its own,
  `part4-shift.zip`, in `out/snapshot-shift/`. Numbers from the downloads
  should match the book.
- **25 and 26, Trees and deep models.** `trainer deep`: LightGBM against a
  DCNv2 on the same rows, against the truth. LightGBM should win slightly;
  app embeddings shouldn't group genres (about 12%), co-install vectors
  should (about 57%).
- **27, Pretraining.** `trainer pretrain`, about three minutes, writes
  `out/pretrain`. A pooled co-install profile should help the trees, and the
  transformer should barely move them: devices have about four installs
  each, too few for it. That's the chapter's lesson, not a failure.
- **28, Serving.** `trainer pretrain` first (for the vectors), then `trainer
  export` and `deepcheck` in the bidder's image. Success: golden vectors
  match PyTorch within about 1e-5, the ONNX check says ok, and the rollover
  refuses about half the records with one version loaded and none with
  both. Timings vary by machine.
- **29, Retraining.** `trainer retrain` on the shift dataset, about two
  minutes. The daily model overpredicts the hypercasual campaigns after the
  shift until its retrain; kappa fixes most of it sooner; the naive 6-hourly
  model sits at about two-thirds of the truth; the delay-weighted one is
  close.
- **30 to 34** run on the live market again: start over for 30 (`ad-roas`)
  and for 32 (`ad-roas,uplift`, with `docker compose build` first, since the
  simulator changed), and continue the market for 31, 33, and 34. Their A/B
  tests need the test stopped, more days played so the labels arrive, a
  dataset built, and then **Read out**. On the tiny market, many readouts
  can't tell the arms apart; the walkthrough says when the laptop scale is
  needed.
- **35, Learned campaign selection.** `trainer select` on the Part IV dataset, about half a minute, or the
  Chapter 35 card on the Deep models tab. The predictor and the walk cap should be close overall (the Ad
  ROAS campaigns win almost every auction); the difference is for CPC winners and devices without installs.
- **36, Personalized creative selection.** `trainer creatives` on the Part IV dataset, about a minute, or the
  Chapter 36 card. It needs `creative_truth.parquet` in the snapshot: an older download lacks it, so
  re-download, or run `docker compose run --rm --no-deps --entrypoint python sim -m adsim.creative_truth
  --snapshot /out/snapshot`, which refuses if the rebuilt world doesn't match the logged rates (a snapshot
  built with other settings needs `--preset`, `--scale`, `--seed`, and `--scenario` to match). Personalized
  sampling should be about twice per-campaign sampling's gain and close to the best fixed creative, with the
  gap on devices with more history; it captures under a third of the oracle's gain at this scale.
- **37 to 41** have no exercises, by design.

## When the code is wrong

If a step fails because of a bug in the companion code, not the reader's
setup, say so plainly, show the evidence (the command, the output, the
lines of code), and suggest the smallest fix. If the reader wants to report
it, help them write an issue with the chapter, the commands, and the output.
