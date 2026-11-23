package gbm

import (
	"fmt"
	"math"

	"github.com/bgweber/adtechbook/code/dsp/internal/features"
)

// WinModel is a win rate model (Chapter 20): a bundle with one target,
// "win", whose inputs include the bid, as ln(bid) and ln(bid / floor).
type WinModel struct {
	*Bundle
	iBid, iRatio int
}

// NewWinModel wraps a loaded win bundle.
func NewWinModel(b *Bundle) (*WinModel, error) {
	if b == nil {
		return nil, fmt.Errorf("no bundle")
	}
	if _, ok := b.Models["win"]; !ok {
		return nil, fmt.Errorf("%s has no win model", b.Spec.Version)
	}
	w := &WinModel{Bundle: b, iBid: -1, iRatio: -1}
	for i, k := range b.Spec.Numeric {
		switch k {
		case "ln_bid":
			w.iBid = len(b.Spec.Categorical) + i
		case "ln_bid_over_floor":
			w.iRatio = len(b.Spec.Categorical) + i
		}
	}
	if w.iBid < 0 {
		return nil, fmt.Errorf("%s has no ln_bid input", b.Spec.Version)
	}
	return w, nil
}

// Prob is the predicted chance that a bid wins, with x from Inputs.
func (w *WinModel) Prob(x []float64, bid, floor float64) float64 {
	w.setBid(x, bid, floor)
	return w.Models["win"].Predict(x)
}

func (w *WinModel) setBid(x []float64, bid, floor float64) {
	x[w.iBid] = math.Log(math.Max(bid, 1e-4))
	if w.iRatio >= 0 {
		if floor > 0 {
			x[w.iRatio] = math.Log(math.Max(bid, 1e-4) / floor)
		} else {
			x[w.iRatio] = math.NaN()
		}
	}
}

// ShadeGrid are the shares of the media value tried as bids.
var ShadeGrid = []float64{0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
	0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0}

// Shade picks the bid that maximizes expected surplus, (value - bid) times
// the chance of winning, among the grid of shares of the media value and
// the floor. It returns the bid and its predicted win rate.
func (w *WinModel) Shade(fv *features.FeatureVector, media, floor float64) (float64, float64) {
	return w.ShadeInputs(w.Inputs(fv, nil), media, floor)
}

// ShadeInputs is Shade with the request's inputs already built: they
// don't depend on the campaign, so the bidder builds them once per request
// rather than once per campaign (Chapter 17). The grid's bids go through
// the trees together, tree by tree, which keeps each tree in cache; the
// answer is the same as one bid at a time.
func (w *WinModel) ShadeInputs(x []float64, media, floor float64) (float64, float64) {
	var bids []float64
	for _, k := range ShadeGrid {
		if b := k * media; b > 0 && b <= media && b >= floor {
			bids = append(bids, b)
		}
	}
	if floor > 0 && floor <= media {
		bids = append(bids, floor)
	}
	if len(bids) == 0 {
		return 0, 0
	}
	rows := make([][]float64, len(bids))
	flat := make([]float64, len(bids)*len(x))
	for i, b := range bids {
		rows[i] = flat[i*len(x) : (i+1)*len(x)]
		copy(rows[i], x)
		w.setBid(rows[i], b, floor)
	}
	p := make([]float64, len(bids))
	w.Models["win"].PredictBatch(rows, p)
	best, bestB, bestP := -1.0, 0.0, 0.0
	for i, b := range bids {
		if s := (media - b) * p[i]; s > best {
			best, bestB, bestP = s, b, p[i]
		}
	}
	return bestB, bestP
}
