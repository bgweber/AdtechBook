package gbm

import (
	"path/filepath"
	"testing"

	"github.com/bgweber/adtechbook/code/dsp/internal/features"
)

func loadWin(t *testing.T) *WinModel {
	t.Helper()
	dir := filepath.Join("testdata", "win-fixture")
	b, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if n, worst, err := b.CheckGolden(dir); err != nil || n == 0 || worst > Tolerance {
		t.Fatalf("golden: %d examples, worst %g, %v", n, worst, err)
	}
	w, err := NewWinModel(b)
	if err != nil {
		t.Fatal(err)
	}
	return w
}

func TestWinRateRisesWithTheBid(t *testing.T) {
	w := loadWin(t)
	fv := &features.FeatureVector{Strings: map[string]string{"placement": "interstitial",
		"exchange": "alpha_exchange"}, Numeric: map[string]float64{"bidfloor": 0.5}}
	x := w.Inputs(fv, nil)
	prev := -1.0
	for _, b := range []float64{0.5, 1, 2, 3, 5, 8, 13} {
		p := w.Prob(x, b, 0.5)
		if p < prev {
			t.Fatalf("win rate fell from %.3f to %.3f as the bid rose to %g", prev, p, b)
		}
		prev = p
	}
}

func TestShadeMaximizesSurplus(t *testing.T) {
	w := loadWin(t)
	fv := &features.FeatureVector{Strings: map[string]string{"placement": "interstitial",
		"exchange": "alpha_exchange"}, Numeric: map[string]float64{"bidfloor": 0.5}}
	media := 6.0
	bid, p := w.Shade(fv, media, 0.5)
	if bid <= 0.5 || bid >= media {
		t.Fatalf("shaded bid %.2f for a $%.2f value", bid, media)
	}
	x := w.Inputs(fv, nil)
	best := (media - bid) * p
	for _, k := range ShadeGrid {
		if s := (media - k*media) * w.Prob(x, k*media, 0.5); s > best+1e-12 {
			t.Fatalf("bid %.2f has more expected surplus (%.4f) than the chosen %.2f (%.4f)",
				k*media, s, bid, best)
		}
	}
	// a cheap placement clears lower, so the same value shades further
	fv.Strings["placement"] = "banner"
	if b2, _ := w.Shade(fv, media, 0.5); b2 >= bid {
		t.Fatalf("banner bid %.2f, not below the interstitial's %.2f", b2, bid)
	}
}

// The batched grid gives the bid and win rate that one bid at a time does
// (Chapter 17).
func TestShadeInputsMatchesOneBidAtATime(t *testing.T) {
	w := loadWin(t)
	for _, pl := range []string{"interstitial", "banner", "rewarded_video"} {
		fv := &features.FeatureVector{Strings: map[string]string{"placement": pl,
			"exchange": "alpha_exchange"}, Numeric: map[string]float64{"bidfloor": 0.5}}
		x := w.Inputs(fv, nil)
		for _, media := range []float64{0.3, 0.5, 0.9, 2, 6, 15} {
			for _, floor := range []float64{0, 0.5, 1} {
				gotB, gotP := w.ShadeInputs(x, media, floor)
				best, wantB, wantP := -1.0, 0.0, 0.0
				for _, b := range append(scaled(media), floor) {
					if b <= 0 || b > media || b < floor {
						continue
					}
					p := w.Prob(append([]float64(nil), x...), b, floor)
					if s := (media - b) * p; s > best {
						best, wantB, wantP = s, b, p
					}
				}
				if gotB != wantB || gotP != wantP {
					t.Fatalf("%s media %g floor %g: batch (%g, %g), one at a time (%g, %g)",
						pl, media, floor, gotB, gotP, wantB, wantP)
				}
			}
		}
	}
}

func scaled(media float64) []float64 {
	var out []float64
	for _, k := range ShadeGrid {
		out = append(out, k*media)
	}
	return out
}
