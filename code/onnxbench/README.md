# onnxbench: ONNX Runtime from Go, without cgo (Chapter 17)

The bidder evaluates its LightGBM models with its own tree walker
(`code/dsp/internal/gbm`): no runtime to install, and parity with LightGBM
to about 1e-16 on every bundle's golden examples. Chapter 17 compares it
with the usual alternative, ONNX Runtime, on the same model. This directory
is that comparison. It is not part of the bidder or of `docker compose`.

It uses [amikos-tech/pure-onnx](https://github.com/amikos-tech/pure-onnx)
(MIT license), which loads ONNX Runtime's C library with
[purego](https://github.com/ebitengine/purego) instead of cgo, so the
binary still builds with `CGO_ENABLED=0`. You need Go 1.25 or later (the
library's `go.mod` asks for it) and ONNX Runtime's shared library, which the
`onnxruntime` Python package includes.

```bash
# export the click model of a bundle, with its golden examples as arrays,
# and time ONNX Runtime from Python
pip install onnx onnxruntime onnxmltools
cd code/training
python -m adtrain onnx --bundle ../../out/models/conversion/gbm-0012 --out ../../out/onnx

# the same model from Go
cd ../onnxbench
go mod tidy
LIB=$(python -c "import onnxruntime, os; d = os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'); print(os.path.join(d, [f for f in os.listdir(d) if f.startswith('libonnxruntime.so')][0]))")
go run . "$LIB" ../../out/onnx/click.onnx ../../out/onnx/golden_X.f32 ../../out/onnx/golden_want.f64

# the bidder's own evaluator on the same bundle
cd ../dsp
GBM_BUNDLE=../../out/models/conversion/gbm-0012 go test ./internal/gbm -run x -bench BundleFromEnv
```

On Windows and macOS, the library is `onnxruntime.dll` or
`libonnxruntime.dylib` in the same folder.

## What we measured

One 200-tree click model with 15 leaves per tree and 26 inputs, on two
cores of a shared cloud machine, ONNX Runtime 1.29.0:

| Evaluator | One row | Batch of 8, per row | Larger batches, per row |
|---|---|---|---|
| Bidder's tree walker, category lookups in maps | 17.1 µs | | |
| Bidder's tree walker, category bitsets | 10.2 µs | 8.6 to 10.1 µs | 7.1 to 8.6 µs (32 rows) |
| ONNX Runtime from Go (pure-onnx) | 20.4 µs | 14.7 µs | 11.5 µs (64 rows) |
| ONNX Runtime from Python, one thread | 25.7 µs | 16.6 µs | 10.6 µs (64 rows) |

The tree walker's batches go tree by tree, every row through one tree
before the next; the ranges are three runs, since timings on a shared
machine move by 10% or more.

ONNX Runtime agreed with LightGBM to 1.6e-7: it takes float32 inputs, so
it rounds the inputs the trees compare against their thresholds. The tree
walker takes float64 and matches to 1e-16.

Build notes: the library's `go.mod` needs Go 1.25, and `go get` needs the
Go module proxy. We built it with Go 1.24 by copying its `ort` package into
a module of our own, with purego v0.10.2 and `golang.org/x/sys` v0.30.0.
The converter (onnxmltools) fixes the label output's batch size at 1;
`adtrain onnx` makes it variable, or every batch larger than one fails.
