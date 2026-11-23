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
| Free ebook | Reflowable EPUB 3, a free download from each GitHub release | Version tags (`v*`) or a manual run |

Read online: <https://bgweber.github.io/AdtechBook/>

> **Run the simulator:** start a simulated ad market, a Go bidder, and its console with one command,
> then follow the book chapter by chapter. The **[walkthrough](WALKTHROUGH.md)** has the commands for
> the exercises in Parts II and III and screenshots of what you'll see. The commands are for Linux and
> macOS; on Windows, use WSL 2 or the [PowerShell versions](WALKTHROUGH-windows.md).
>
> ```bash
> docker compose up --build    # then open http://localhost:8081
> ```

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
| `code/dsp/`: the Go reference bidder | **Part II complete: bidding, crawl stage, with a click model and operations.** Parsing and normalization of each exchange's quirks, device profiles, eligibility and weighted selection, crawl pricing, a 5% holdout, the bid ledger with FeatureVectors, win, loss, billing, and click notices, serving the click model with a parity check before every deploy, CPI pricing from MMP postbacks, kill switches, a fallback ladder, throttling, alerts, and reconciliation, and a live console with Models and Controls tabs. See [`code/dsp/README.md`](code/dsp/README.md) |
| `code/training/`: training, validation, publishing | **Click model works.** Click examples from the ledger, hashed tokens with crosses, FTRL, out-of-time validation (ROC and PR curves, calibration, readouts, cosine stability), and model artifacts with golden examples. See [`code/training/README.md`](code/training/README.md) |
| `compose.yaml`: Docker Compose | **Started.** The bidder, the training service, and the simulator; infrastructure services are planned |

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

Watch the Go bidder trade in the simulated market, with Docker:

```bash
docker compose up --build
```

Then open the console at <http://localhost:8081>. The simulator plays two
days and stops. Open the **Models** tab and click **Train**: the training
service builds click examples from the bidder's logs, trains a click
model, validates it, and the bidder deploys it if it passes its checks.
Then play more days with the model pricing the bids:

```bash
docker compose run --rm sim
```

Each run continues the market where the bidder's ledger left off, so the
model sees new days, never a replay. To start over, stop everything and
delete the `out` folder.

This needs a current
Docker with Compose v2, the `docker compose` command (Docker Desktop
includes it). If you get `unknown flag: --build` or `'compose' is not a
docker command`, your Docker is too old: update Docker Desktop. Without
Docker, see [`code/dsp/README.md`](code/dsp/README.md).

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
pull request. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

- **Text and figures:** [CC BY-NC-SA 4.0](LICENSE). You can share and adapt
  them for non-commercial purposes, with credit, under the same license. The
  cover and the author photo are not covered by this license.
- **Code:** [MIT](LICENSE-CODE). This covers `code/`, `scripts/`, `filters/`,
  the build configuration, and the code samples in the book, so you can use
  them in your own projects, including commercial ones.

## Repository layout

```
_quarto.yml                 Book config: chapters, HTML/PDF/EPUB formats
index.qmd, preface.qmd      Welcome page and preface
parts/, chapters/           Part pages and one .qmd file per chapter
appendices/                 Glossary, OpenRTB fields, templates, reading, the simulated world
code/sim/                   Python: adsim, the simulated market
code/dsp/                   Go: the reference bidder and its console
code/training/              Python: training, validation, publishing (adtrain)
compose.yaml                Docker Compose: the bidder, the trainer, and the simulator
images/                     Figures, the ebook cover (cover.jpg, added to the EPUB by
                            scripts/polish_epub.py), and the README cover (cover-front.jpg)
styles/preamble.tex         Print styling: headers, code boxes, copyright page
styles/epub.css             Ebook styling
scripts/polish_epub.py      Makes the EPUB a native ebook (runs after each render)
_freeze/                    Saved code output (committed; see below)
requirements.txt            Python packages CI needs to render
LICENSE, LICENSE-CODE       CC BY-NC-SA 4.0 for text and figures; MIT for code
CONTRIBUTING.md             How to contribute, and how contributions are licensed
.github/workflows/          Book build/publish and code test workflows
```

## Tests

The `Tests` workflow runs on every push and pull request: `pytest code` for
the Python code, and `go vet` and `go test ./...` in `code/dsp/`. To run
them locally:

```bash
pip install -e code/sim[test] -e code/training[test]
pytest code
cd code/dsp && go test ./...
```

The Go tests include a parity test against a model the Python code
trained (`code/dsp/internal/model/testdata`, made with
`python -m adtrain.fixture`), so training and serving can't drift apart
without a test failing.

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

**Paperback and free ebook.** Tag a release:

```bash
git tag v1.0.0
```

```bash
git push origin v1.0.0
```

The workflow attaches the PDF and EPUB to the GitHub release for that tag.
Upload the PDF as the KDP paperback interior. The EPUB is not sold on
KDP: it stays on the GitHub release as a free download, which the web
edition links to, and readers can send it to Kindle, Kobo, or Apple Books.
The EPUB's cover is added by `scripts/polish_epub.py`, not Quarto's
`epub-cover-image`, whose SVG cover page made Kindle Previewer's conversion
fail. You can also run the workflow manually from the
*Actions* tab and download the files from the run's artifacts.

## Releases

- `main` is the online book. It changes whenever a chapter or the code changes.
- A git tag (`v1.0.0`, `v1.1.0`, ...) marks each print edition and its free EPUB.
  Readers of a printed copy can check out that tag to get the text and code
  that match their book.
- Each tag's release notes record the simulator version it was built and
  tested with (for example, `adsim 0.1`), and which "Try it on the
  simulator" exercises are runnable at that tag.
- Print comes first. Lock the PDF with a KDP proof copy before building the
  EPUB: during print work, build only the PDF and web editions
  (`quarto render --to pdf`, `quarto preview`). Then build the EPUB once
  (`quarto render --to epub`), open it in Kindle Previewer, and check tables,
  equations, figures, and long code lines. Kindle Previewer's conversion is
  CPU-heavy for this book, so don't rebuild the EPUB for every change.

## Before the first release

- Once the paperback is live, optionally swap the Amazon ISBN search links
  (`https://www.amazon.com/s?k=9798144378116`) for the product page URL.
