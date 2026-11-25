package gbm

import (
	"math"
	"testing"
)

// The tilt is 1 - alpha + alpha r / mean r, capped; without a mean it's 1.
func TestTilt(t *testing.T) {
	b := &Bundle{Spec: Spec{UpliftMean: 0.8}}
	for _, c := range []struct{ r, alpha, want float64 }{
		{0.8, 1, 1}, {0.4, 1, 0.5}, {0.4, 0.5, 0.75}, {3, 1, MaxTilt}, {0, 1, 0},
	} {
		if got := b.Tilt(c.r, c.alpha); math.Abs(got-c.want) > 1e-12 {
			t.Fatalf("Tilt(%v, %v) = %v, want %v", c.r, c.alpha, got, c.want)
		}
	}
	if (&Bundle{}).Tilt(0.4, 1) != 1 {
		t.Fatal("a bundle without a mean tilted")
	}
	if _, ok := (&Bundle{Models: map[string]*Ensemble{}}).Uplift(nil, nil); ok {
		t.Fatal("a bundle without uplift models gave a ratio")
	}
}
