# Code

The runnable code behind the book. Chapters show excerpts from these files,
so the code in the book is code that builds and passes tests.

```
code/
  dsp/        Go: the reference DSP (bid path, features, in-process inference)
  training/   Python: training datasets, training, validation, publishing
  data/       Python: synthetic data generator used by the examples
```

## Conventions

- **Go** (`code/dsp/`): one Go module, `go.mod` in `code/dsp/`. Tests run
  with `go test ./...`.
- **Python** (`code/training/`, `code/data/`): Python 3.12. Tests are
  `test_*.py` files and run with `pytest code`.
- Examples use a fixed random seed and synthetic data from `code/data/`
  unless a chapter names a public dataset. Nothing downloads data at render
  time.
- GPU training code uses PyTorch, falls back to CPU, and fits in 4 GB of GPU
  memory. Heavy steps (GPU training, PySpark at scale, load tests) run
  outside the book render; chapters show their captured output.
- Lines stay within about 80 characters so listings fit the print page.

The `Tests` workflow runs `go test` and `pytest` on every push and pull
request, and skips each language until it has code.
