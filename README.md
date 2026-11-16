# Machine Learning in Mobile AdTech

Source for the book *Machine Learning in Mobile AdTech*. One set of
[Quarto](https://quarto.org) files builds three editions:

| Edition | Output | When it builds |
|---|---|---|
| Free web edition | HTML on GitHub Pages | Every push to `main` |
| Paperback | KDP 7" x 10" black-and-white interior PDF | Version tags (`v*`) or a manual run |
| Kindle | Reflowable EPUB 3 | Version tags (`v*`) or a manual run |

Read online: <https://bgweber.github.io/AdtechBook/>

## Repository layout

```
_quarto.yml                 Book config: chapters, HTML/PDF/EPUB formats
index.qmd                   Welcome page
chapters/                   One .qmd file per chapter
code/dsp/                   Go: reference DSP and serving code
code/training/              Python: training, validation, publishing
code/data/                  Python: synthetic data generator
images/                     Figures and the Kindle cover (cover.jpg)
styles/preamble.tex         Print styling: headers, code boxes, copyright page
styles/epub.css             Kindle styling
_freeze/                    Saved code output (committed; see below)
requirements.txt            Python packages CI needs to render
.github/workflows/          Book build/publish and code test workflows
```

## Tests

The `Tests` workflow runs on every push and pull request: `go vet` and
`go test ./...` in `code/dsp/`, and `pytest code` for the Python code. To run
them locally:

```bash
cd code/dsp && go test ./...
```

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

## Before the first release

- Replace the Amazon URL placeholders (`https://amazon.com`).
- Add the ISBN to the copyright page in `styles/preamble.tex`.
- Replace `images/cover.jpg` with the real cover (1600 x 2560 px or larger).
