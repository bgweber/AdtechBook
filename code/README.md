# Code

The runnable code behind the book: the simulated market (`sim/`), the Go
bidder (`dsp/`), the training code (`training/`), and Chapter 17's ONNX
Runtime comparison (`onnxbench/`). The excerpts in the book come from these
files, so the code in the book builds and passes tests.

```
code/
  dsp/        Go: the reference DSP (bid path, features, model serving, console)
  training/   Python: datasets, training, validation, publishing (adtrain)
  sim/        Python: the simulated market (adsim) that every example runs on
  onnxbench/  Go: ONNX Runtime from Go against the bidder's tree evaluator (Ch 17)
```

## Conventions

- **Go** (`code/dsp/`): one Go module, `go.mod` in `code/dsp/`. Tests run
  with `go test ./...`. `onnxbench/` is its own module and isn't part of the
  bidder.
- **Python** (`code/training/`, `code/sim/`): Python 3.12. Tests are
  `test_*.py` files and run with `pytest code`.
- Examples use a fixed random seed and synthetic data from the simulator in
  `code/sim/` unless a chapter names a public dataset. Nothing downloads data
  at render time.
- Part IV's deep models use PyTorch on the CPU and run in minutes on a
  laptop. Scaling them up to a GPU is left to the reader.
- Lines stay within about 80 characters so listings fit the print page.

The `Tests` workflow runs `go test` and `pytest` on every push and pull
request.
