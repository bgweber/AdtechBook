# Code

The runnable code behind the book: the simulated market (`sim/`), the Go
bidder (`dsp/`), and the training code (`training/`). They grow part by
part. Chapters will show excerpts from these files, so the code in the book
is code that builds and passes tests.

```
code/
  dsp/        Go: the reference DSP (bid path, features, model serving, console)
  training/   Python: training datasets, training, validation, publishing
  sim/        Python: the simulated market (adsim) that every example runs on
```

## Conventions

- **Go** (`code/dsp/`): one Go module, `go.mod` in `code/dsp/`. Tests run
  with `go test ./...`.
- **Python** (`code/training/`, `code/sim/`): Python 3.12. Tests are
  `test_*.py` files and run with `pytest code`.
- Examples use a fixed random seed and synthetic data from the simulator in `code/sim/`
  unless a chapter names a public dataset. Nothing downloads data at render
  time.
- GPU training code uses PyTorch, falls back to CPU, and fits in 4 GB of GPU
  memory. Heavy steps (GPU training, PySpark at scale, load tests) run
  outside the book render; chapters show their captured output.
- Lines stay within about 80 characters so listings fit the print page.

The `Tests` workflow runs `go test` and `pytest` on every push and pull
request.
