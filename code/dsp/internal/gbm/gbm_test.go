package gbm

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/bgweber/adtechbook/code/dsp/internal/features"
)

// The fixture comes from `python -m adtrain.fixture_gbm`: LightGBM models
// and their own predictions on golden examples, including strings the
// vocabulary has never seen and missing numbers.
func TestGoldenParity(t *testing.T) {
	dir := filepath.Join("testdata", "gbm-fixture")
	b, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	n, worst, err := b.CheckGolden(dir)
	if err != nil {
		t.Fatal(err)
	}
	if n < 100 {
		t.Fatalf("only %d golden examples", n)
	}
	if worst > 1e-9 {
		t.Fatalf("predictions differ from LightGBM's by up to %g", worst)
	}
}

func TestUnknownStringsAndMissingNumbers(t *testing.T) {
	b, err := Load(filepath.Join("testdata", "gbm-fixture"))
	if err != nil {
		t.Fatal(err)
	}
	fv := &features.FeatureVector{Numeric: map[string]float64{}, Strings: map[string]string{"placement": "a-new-format"}}
	x := b.Inputs(fv, map[string]string{})
	for i, k := range b.Spec.Categorical {
		if x[i] != float64(b.Spec.Vocab[k][b.Spec.Other]) {
			t.Errorf("%s: unknown value should map to the other code", k)
		}
	}
	for j, k := range b.Spec.Numeric {
		v := x[len(b.Spec.Categorical)+j]
		zero := k == "device_prior_installs" || k == "device_prior_purchases"
		if zero && v != 0 || !zero && !math.IsNaN(v) {
			t.Errorf("%s: missing value is %v", k, v)
		}
	}
	p, ok := b.Predict("click", x)
	if !ok || p <= 0 || p >= 1 {
		t.Fatalf("prediction %v, %v", p, ok)
	}
}

func TestPredictBatchMatchesPredict(t *testing.T) {
	b, err := Load("testdata/gbm-fixture")
	if err != nil {
		t.Fatal(err)
	}
	rows, err := goldenRows("testdata/gbm-fixture")
	if err != nil {
		t.Fatal(err)
	}
	var xs [][]float64
	for _, r := range rows {
		xs = append(xs, b.Inputs(&r.Features, r.Ad))
	}
	for target := range b.Models {
		out := make([]float64, len(xs))
		b.PredictBatch(target, xs, out)
		for i, x := range xs {
			want, _ := b.Predict(target, x)
			if math.Abs(out[i]-want) > 1e-12 {
				t.Fatalf("%s row %d: batch %g, single %g", target, i, out[i], want)
			}
		}
	}
}
