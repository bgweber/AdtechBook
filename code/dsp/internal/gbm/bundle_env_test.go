package gbm

import (
	"fmt"
	"os"
	"testing"
)

// GBM_BUNDLE=out/models/conversion/gbm-0001 go test ./internal/gbm -run TestBundleFromEnv
// checks a real published bundle against its golden examples.
func TestBundleFromEnv(t *testing.T) {
	dir := os.Getenv("GBM_BUNDLE")
	if dir == "" {
		t.Skip("set GBM_BUNDLE to a bundle directory")
	}
	b, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	n, worst, err := b.CheckGolden(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%d golden examples, largest difference %g", n, worst)
	if worst > 1e-9 {
		t.Fatalf("predictions differ by up to %g", worst)
	}
}

// GBM_BUNDLE=out/models/conversion/gbm-0001 go test ./internal/gbm -run x -bench BundleFromEnv
// times the native evaluator on a real bundle's golden rows: building the
// inputs, then one target, then all three targets of the conversion model.
func BenchmarkBundleFromEnv(b *testing.B) {
	dir := os.Getenv("GBM_BUNDLE")
	if dir == "" {
		b.Skip("set GBM_BUNDLE to a bundle directory")
	}
	bun, err := Load(dir)
	if err != nil {
		b.Fatal(err)
	}
	rows, err := goldenRows(dir)
	if err != nil {
		b.Fatal(err)
	}
	xs := make([][]float64, len(rows))
	for i, r := range rows {
		xs[i] = bun.Inputs(&r.Features, r.Ad)
	}
	b.Run("inputs", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			r := &rows[i%len(rows)]
			bun.Inputs(&r.Features, r.Ad)
		}
	})
	b.Run("click", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			bun.Predict("click", xs[i%len(xs)])
		}
	})
	b.Run("all-targets", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			x := xs[i%len(xs)]
			for _, t := range []string{"click", "install_click", "install_view"} {
				bun.Predict(t, x)
			}
		}
	})
	for _, n := range []int{1, 8, 32} {
		batch := make([][]float64, n)
		out := make([]float64, n)
		b.Run(fmt.Sprintf("click-rows-%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				for r := range batch {
					batch[r] = xs[(i*n+r)%len(xs)]
				}
				for r, x := range batch {
					out[r], _ = bun.Predict("click", x)
				}
			}
		})
		b.Run(fmt.Sprintf("click-batch-%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				for r := range batch {
					batch[r] = xs[(i*n+r)%len(xs)]
				}
				bun.PredictBatch("click", batch, out)
			}
		})
	}
}
