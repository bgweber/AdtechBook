package gbm

import (
	"fmt"
	"math"

	"github.com/bgweber/adtechbook/code/dsp/internal/features"
)

// Uplift is the incrementality ratio for one device and app (Chapter 32):
// installs our bids cause per install credited to us, from an uplift
// bundle's three models: the install rate, from any source, of devices we
// bid on (mu1) and of held-out devices (mu0), and the rate of installs
// credited to us (mu_attr). r = (mu1 - mu0) / mu_attr, between 0 and
// RMax. ok is false for a bundle without the three models.
func (b *Bundle) Uplift(fv *features.FeatureVector, ad map[string]string) (r float64, ok bool) {
	if b == nil {
		return 0, false
	}
	m1, ok1 := b.Models["mu1"]
	m0, ok0 := b.Models["mu0"]
	ma, oka := b.Models["mu_attr"]
	if !ok1 || !ok0 || !oka {
		return 0, false
	}
	x := b.Inputs(fv, ad)
	p1, p0, pa := m1.Predict(x), m0.Predict(x), ma.Predict(x)
	return math.Min(RMax, math.Max(0, (p1-p0)/math.Max(pa, 1e-9))), true
}

// RMax caps the ratio: our ads can cause installs another network gets
// credit for, so it can be above 1, but not by much.
const RMax = 1.5

// MaxTilt caps the tilt: a device the model thinks twice as incremental as
// the average gets at most twice the value.
const MaxTilt = 2.0

// Tilt is the multiplier on an install campaign's value (Chapter 32):
// 1 - alpha + alpha * r / mean r. Dividing by the bundle's average share
// keeps the average bid about where it was, so the tilt moves money toward
// devices where the ads make a difference rather than spending less.
func (b *Bundle) Tilt(r, alpha float64) float64 {
	mean := b.Spec.UpliftMean
	if !(mean > 0) {
		return 1
	}
	return math.Min(MaxTilt, 1-alpha+alpha*r/mean)
}

// Pin loads an uplift bundle for the incrementality tilts that campaigns
// opt into (Chapter 33), after the same checks as a deploy; "" unpins.
// Uplift bundles are never deployed as a champion: a tilt test or a
// campaign's levers use them.
func (r *Registry) Pin(version string) error {
	if version == "" {
		r.pinned.Store(nil)
		return nil
	}
	b, res := r.Prepare(version, false)
	if b == nil {
		return fmt.Errorf("%s", res.Message)
	}
	if _, ok := b.Models["mu_attr"]; !ok || !(b.Spec.UpliftMean > 0) {
		return fmt.Errorf("%s isn't an uplift bundle", version)
	}
	r.pinned.Store(b)
	return nil
}

// Pinned is the pinned uplift bundle, or nil.
func (r *Registry) Pinned() *Bundle {
	if r == nil {
		return nil
	}
	return r.pinned.Load()
}
