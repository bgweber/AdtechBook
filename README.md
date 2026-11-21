# Machine Learning in Mobile AdTech

<p align="center">
  <img src="images/cover-front.jpg" alt="Cover of Machine Learning in Mobile AdTech: From Foundations to the Frontier of Demand Side Platforms, by Ben G Weber" width="320">
</p>

Source for the book *Machine Learning in Mobile AdTech*. One set of
[Quarto](https://quarto.org) files builds three editions:

| Edition | Output | When it builds |
|---|---|---|
| Free web edition | HTML on GitHub Pages | Every push to `main` |
| Paperback | KDP 7" x 10" black-and-white interior PDF | Version tags (`v*`) or a manual run |
| Kindle | Reflowable EPUB 3 | Version tags (`v*`) or a manual run |

Read online: <https://bgweber.github.io/AdtechBook/>

## The companion code

The book comes with a simulated mobile ad market, **adsim**, and the code
that bids into it and learns from it. The simulator knows the truth: next to
the logs a DSP would see, it writes the true click, install, and lift
probabilities behind every impression, so you can check a model against
what really happened, not just against noisy outcomes.

What exists today and what's planned:

| Component | Status |
|---|---|
| `code/sim/`: adsim 0.1, the simulated market | **Works.** Crawl, walk, and run markets; OpenRTB requests with exchange quirks; first-price auctions with a winner's curse; clicks, installs, organic installs, MMP attribution, delayed postbacks, purchases; holdouts; truth tables |
| `code/dsp/`: the Go reference bidder | Planned |
| `code/training/`: training, validation, publishing | Planned |
| `docker/`: Docker Compose for the whole stack | Planned |

Each chapter's "Try it on the simulator" box says whether its exercise runs
on adsim 0.1 today or what it still needs.

Quick start, no Docker needed:

```bash
pip install -e code/sim[test]
python -m adsim --preset crawl --scale laptop --days 1 --seed 7 --out out/crawl
pytest code/sim
```

See [`code/sim/README.md`](code/sim/README.md) for the tables it writes and
how the market works.

## Toward a full simulator

adsim 0.1 simulates the market with a Python bidding policy, one run at a
time. The full simulator turns it into a market that a real bidder can
trade in, with the problems a real DSP has to survive. Planned, roughly in
this order:

**A live market for a real bidder (adsim 0.2)**

- The simulator sends OpenRTB bid requests over HTTP to a bidder in
  simulated time and enforces the exchange's timeout, so a slow bidder
  loses auctions just as it would in production.
- A documented policy interface for Python bidders, and a listen-only mode
  for the "watch before you bid" chapters.
- Devices that change during the run: installs change what a device plays,
  tastes drift, sessions have a rhythm, and ads cause fatigue. This is what
  gives sequence models and transformers real signal to learn, and the
  truth tables record how much lift that signal is worth.

**A market that fights back (adsim 0.3)**

- Competing bidders that learn and adapt their shading.
- Market shifts: seasonality, holiday CPM spikes, a big buyer entering or
  leaving, traffic surges.
- Fault injection for incident labs: exchange outages, postback delays,
  duplicate floods, schema changes, clock skew, a bad model push.

**The rest of the mobile ecosystem (adsim 0.4)**

- iOS and ATT: devices without an IFA, SKAdNetwork-style postbacks with
  coarse values and delays, contextual-only signals.
- Creative content features, for creative selection models.
- Click injection and click spamming fraud.

**Throughout**

- A scoreboard: fixed seeds and markets, with policies scored on ROAS,
  incremental installs, and pacing error against the simulator's truth.
  Write a `decide()` function and see whether you can beat the baseline.
- Speed, so the `medium` and `large` scales run in reasonable time.

## Docker

Everything beyond the simulator will run with Docker Compose, so the bidder,
the simulator, the pipelines, and their infrastructure start with one
command on any machine. Planned services, grouped into Compose profiles so
you start only what a chapter needs:

| Profile | Services |
|---|---|
| `core` | The simulator, the Go bidder |
| `ml` | Training and validation pipelines, experiment tracking, a model registry |
| `streaming` | An event stream and online feature store, for real-time learning |
| `gpu` | Deep model training and serving on an NVIDIA GPU |

The simulator alone will keep working without Docker (the quick start
above), and so will rendering the book.

### GPU training

Part IV trains deep models: multi-task networks, embeddings, and
transformers over device histories. The training code uses PyTorch, fits in
4 GB of GPU memory, and falls back to CPU at smaller scales. The `gpu`
profile needs an NVIDIA GPU and the NVIDIA Container Toolkit, on Linux or on
Windows with WSL 2. Docker on macOS can't use the GPU, so Macs run the CPU
fallback.

## Contributing

The book and the code are open source, and contributions are welcome. Good
places to start are the planned exercises and simulator features above: each
one says what it needs. Open an issue to discuss a change before a large
pull request.

## Repository layout

```
_quarto.yml                 Book config: chapters, HTML/PDF/EPUB formats
index.qmd, preface.qmd      Welcome page and preface
parts/, chapters/           Part pages and one .qmd file per chapter
appendices/                 Glossary, OpenRTB fields, templates, reading, the simulated world
code/sim/                   Python: adsim, the simulated market
code/dsp/                   Go: the reference bidder (planned)
code/training/              Python: training, validation, publishing (planned)
images/                     Figures, the Kindle cover (cover.jpg), and the front cover
                            shown in this README (cover-front.jpg)
styles/preamble.tex         Print styling: headers, code boxes, copyright page
styles/epub.css             Kindle styling
_freeze/                    Saved code output (committed; see below)
requirements.txt            Python packages CI needs to render
.github/workflows/          Book build/publish and code test workflows
```

## Tests

The `Tests` workflow runs on every push and pull request: `pytest code` for
the Python code, and `go vet` and `go test ./...` in `code/dsp/` once the
bidder exists. To run them locally:

```bash
pytest code
```

## Setup

1. Install [Quarto](https://quarto.org/docs/get-started/).
2. Install Python 3.12 and the packages: `pip install -r requirements.txt`.
3. For the PDF, install a TeX distribution: `quarto install tinytex`.

## Local preview and builds

Live preview of the web edition (reloads on save):

```bash
quarto preview
```

Build every edition into `_book/`:

```bash
quarto render
```

Build one edition:

```bash
quarto render --to pdf
```

## Code execution and GPU chapters

Code cells run when a chapter renders. `execute: freeze: auto` saves the
results in `_freeze/`, and CI reuses them instead of running the code again.
CI has no GPU, so for any chapter with GPU code:

1. Render it locally on a machine with an NVIDIA GPU:
   `quarto render chapters/NN-name.qmd`
2. Commit the updated `_freeze/` folder along with the chapter.

## Publishing

**Web edition.** Enable GitHub Pages once: *Settings > Pages > Source:
GitHub Actions*. After that, every push to `main` deploys the site.

**Paperback and Kindle.** Tag a release:

```bash
git tag v1.0.0
```

```bash
git push origin v1.0.0
```

The workflow attaches the PDF and EPUB to the GitHub release for that tag.
Upload the PDF as the KDP paperback interior and the EPUB as the Kindle
manuscript. You can also run the workflow manually from the *Actions* tab and
download the files from the run's artifacts.

## Releases

- `main` is the online book. It changes whenever a chapter or the code changes.
- A git tag (`v1.0.0`, `v1.1.0`, ...) marks each print and Kindle edition.
  Readers of a printed copy can check out that tag to get the text and code
  that match their book.
- Each tag's release notes record the simulator version it was built and
  tested with (for example, `adsim 0.1`), and which "Try it on the
  simulator" exercises are runnable at that tag.
- Before tagging, open the EPUB in Kindle Previewer and check tables,
  equations, figures, and long code lines.

## Before the first release

- Replace the Amazon URL placeholders (`https://amazon.com`).
