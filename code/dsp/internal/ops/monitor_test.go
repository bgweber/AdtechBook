package ops

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/bgweber/adtechbook/code/dsp/internal/clock"
)

// Five days of click predictions that cycle through the day, then a 30%
// shift at noon on the sixth: the shift alert fires within two hours of
// it, names the deploy, and resolves after a rollback (Chapter 24).
func TestPredictionShiftAlert(t *testing.T) {
	dir := t.TempDir()
	o := New(dir)
	b := book()
	rng := rand.New(rand.NewPCG(1, 2))
	scale := 1.0
	firstFiring := int64(-1)
	for day := int64(0); day < 6; day++ {
		for ts := day * clock.DayMS; ts < (day+1)*clock.DayMS; ts += 10_000 {
			if day == 5 && ts%clock.DayMS == 12*hourMS {
				scale = 1.30
				o.Note("conversion bundle gbm-0009 deployed (drill)", "test", ts)
			}
			if day == 5 && ts%clock.DayMS == 18*hourMS {
				scale = 1.0
				o.Note("conversion models rolled back to gbm-0008", "test", ts)
			}
			hour := float64(ts%clock.DayMS) / hourMS
			base := 0.01 * (1 + 0.4*math.Sin(2*math.Pi*hour/24)) // the daily cycle
			p := base * math.Exp(0.6*rng.NormFloat64()) * scale
			o.Observe(BidObs{TsMS: ts, Exchange: "alpha", Action: "bid", PClick: p, PInstall: 0.03,
				PFrom: "model", ModelOn: true, LatencyUS: 500})
			o.Step(ts, b)
			if firstFiring < 0 && firing(o, "prediction_shift") {
				firstFiring = ts
			}
		}
	}
	if firstFiring < 5*clock.DayMS+12*hourMS {
		t.Fatalf("fired before the shift, at %s", simClock(firstFiring))
	}
	if firstFiring > 5*clock.DayMS+14*hourMS {
		t.Fatalf("fired late, at %s", simClock(firstFiring))
	}
	if firing(o, "prediction_shift") {
		t.Fatal("still firing after the rollback")
	}
	var shift *Alert
	for _, a := range o.Alerts() {
		if a.Name == "prediction_shift" {
			shift = &a
			break
		}
	}
	if shift == nil || shift.Scope != "click model" || !strings.Contains(shift.Detail, "gbm-0009") ||
		shift.Playbook == "" {
		t.Fatalf("alert %+v", shift)
	}
	// the install model didn't move
	for _, a := range o.Alerts() {
		if a.Name == "prediction_shift" && a.Scope == "install model" {
			t.Fatalf("install alert %+v", a)
		}
	}
	// the baselines survive a restart
	if len(New(dir).ShiftView()) == 0 {
		t.Fatal("monitor not saved")
	}
}

func firing(o *Ops, name string) bool {
	for _, a := range o.Alerts() {
		if a.Name == name && a.State == "firing" {
			return true
		}
	}
	return false
}
