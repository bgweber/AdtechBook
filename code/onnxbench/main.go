// Command onnxbench times ONNX Runtime from Go on a model exported by
// "python -m adtrain onnx" (Chapter 17), through amikos-tech/pure-onnx,
// which calls ONNX Runtime's C library with purego rather than cgo. It
// checks the model against the bundle's golden examples, then times one
// row per call and batches of 8 and 64.
//
//	go run . /path/to/libonnxruntime.so out/onnx/click.onnx \
//	    out/onnx/golden_X.f32 out/onnx/golden_want.f64
//
// The bidder's own evaluator, on the same bundle, for comparison:
//
//	cd ../dsp && GBM_BUNDLE=../../out/models/conversion/gbm-0012 \
//	    go test ./internal/gbm -run x -bench BundleFromEnv
package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
	"time"

	"github.com/amikos-tech/pure-onnx/ort"
)

func readF32(p string) []float32 {
	b, _ := os.ReadFile(p)
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}
func readF64(p string) []float64 {
	b, _ := os.ReadFile(p)
	out := make([]float64, len(b)/8)
	for i := range out {
		out[i] = math.Float64frombits(binary.LittleEndian.Uint64(b[8*i:]))
	}
	return out
}

func session(nf, batch int) (*ort.AdvancedSession, *ort.Tensor[float32], *ort.Tensor[float32]) {
	in, err := ort.NewTensor(ort.Shape{int64(batch), int64(nf)}, make([]float32, batch*nf))
	must(err)
	lab, err := ort.NewEmptyTensor[int64](ort.Shape{int64(batch)})
	must(err)
	prob, err := ort.NewEmptyTensor[float32](ort.Shape{int64(batch), 2})
	must(err)
	s, err := ort.NewAdvancedSession(os.Args[2], []string{"x"}, []string{"label", "probabilities"},
		[]ort.Value{in}, []ort.Value{lab, prob}, nil)
	must(err)
	return s, in, prob
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	if len(os.Args) != 5 {
		fmt.Fprintln(os.Stderr, "usage: onnxbench libonnxruntime model.onnx golden_X.f32 golden_want.f64")
		os.Exit(2)
	}
	must(ort.SetSharedLibraryPath(os.Args[1]))
	must(ort.InitializeEnvironment())
	X := readF32(os.Args[3])
	want := readF64(os.Args[4])
	nf := len(X) / len(want)
	n := len(want)
	// parity, one row at a time
	s1, in1, p1 := session(nf, 1)
	maxd := 0.0
	for i := 0; i < n; i++ {
		copy(in1.GetData(), X[i*nf:(i+1)*nf])
		must(s1.Run())
		d := math.Abs(float64(p1.GetData()[1]) - want[i])
		if d > maxd {
			maxd = d
		}
	}
	fmt.Printf("parity rows=%d max_abs_diff=%.2g\n", n, maxd)
	for _, batch := range []int{1, 8, 64} {
		s, in, _ := session(nf, batch)
		lat := []float64{}
		start := time.Now()
		rows := 0
		for it := 0; time.Since(start) < 2*time.Second; it++ {
			for b := 0; b < batch; b++ {
				r := (it*batch + b) % n
				copy(in.GetData()[b*nf:(b+1)*nf], X[r*nf:(r+1)*nf])
			}
			t := time.Now()
			must(s.Run())
			lat = append(lat, float64(time.Since(t).Nanoseconds())/1e3)
			rows += batch
		}
		sort.Float64s(lat)
		fmt.Printf("onnx batch=%d calls=%d p50=%.1fus p99=%.1fus per_row=%.2fus\n", batch, len(lat),
			lat[len(lat)/2], lat[len(lat)*99/100], lat[len(lat)/2]/float64(batch))
	}
}
