package ops

import (
	"math"
	"testing"

	"github.com/bgweber/adtechbook/code/dsp/internal/campaigns"
)

func pacedBook() *campaigns.Book {
	b := campaigns.NewBook()
	b.Load([]campaigns.Campaign{{ID: 1, Goal: "CPI", Target: 2, DailyBudget: 96, Geos: "USA"},
		{ID: 2, Goal: "CPI", Target: 2, DailyBudget: 96, Geos: "USA"}}, nil)
	return b
}

// Campaign 1 spends twice its plan and campaign 2 half: the controller
// lowers the first's gamma and raises the second's, within the bounds.
func TestPacingController(t *testing.T) {
	o := New(t.TempDir())
	o.SetPacing(true)
	book := pacedBook()
	for k := int64(1); k <= 48; k++ { // half a day of quarters; flat plan
		book.AddSpend(0, 1, 2*1) // $1 a quarter is the plan: 96 / 96
		book.AddSpend(0, 2, 0.5)
		o.Step(k*QuarterMS, book)
	}
	g1, _ := o.Pace(1)
	g2, _ := o.Pace(2)
	if g1 >= 0.6 || g1 < PaceMin {
		t.Fatalf("ahead of plan: gamma %.2f", g1)
	}
	if g2 <= 1.4 || g2 > PaceMax {
		t.Fatalf("behind plan: gamma %.2f", g2)
	}
	// it moves at most PaceMaxStep (in ln) a step, so it never jumps
	_, st := o.PaceView()
	pts := st[1].Today
	for i := 1; i < len(pts); i++ {
		if r := pts[i].Gamma / pts[i-1].Gamma; r < 0.81 || r > 1.23 {
			t.Fatalf("gamma jumped from %.3f to %.3f", pts[i-1].Gamma, pts[i].Gamma)
		}
	}
	// with the controller off, gamma is 1; a manual overbid is separate
	o.SetOverbid(2, 1.5)
	o.SetPacing(false)
	if g, ob := o.Pace(2); g != 1 || ob != 1.5 {
		t.Fatalf("off: gamma %.2f, overbid %.2f", g, ob)
	}
}

// A new day starts from the forecast when one was loaded for it, else
// from where yesterday ended.
func TestPacingDayStart(t *testing.T) {
	o := New(t.TempDir())
	o.SetPacing(true)
	book := pacedBook()
	for k := int64(1); k < 96; k++ { // day 0, up to its last quarter
		book.AddSpend(0, 1, 2)
		o.Step(k*QuarterMS, book)
	}
	end, _ := o.Pace(1)
	o.SetForecast(1, map[int]float64{2: 1.7})
	o.Step(96*QuarterMS+QuarterMS, book)
	_, st := o.PaceView()
	if st[1].Gamma0 != end || st[1].From != "yesterday" {
		t.Fatalf("campaign 1 started day 1 at %.2f (%s), ended day 0 at %.2f", st[1].Gamma0, st[1].From, end)
	}
	if st[2].Gamma0 != 1.7 || st[2].From != "forecast" {
		t.Fatalf("campaign 2 started at %.2f (%s)", st[2].Gamma0, st[2].From)
	}
	// the ramp: campaign 2 starts the day at 1, not its forecast of 1.7,
	// and the step limit takes it there within the first hours of control
	if g, _ := o.Pace(2); g > PaceRampStart*math.Exp(PaceMaxStep)+1e-9 {
		t.Fatalf("campaign 2 opened the day at gamma %.2f", g)
	}
	for k := int64(98); k < 96+5*4; k++ { // on plan for five hours
		book.AddSpend(1, 2, 1)
		o.Step(k*QuarterMS, book)
	}
	if g, _ := o.Pace(2); g < 1.6 {
		t.Fatalf("campaign 2 at gamma %.2f after five hours on plan, still short of its forecast", g)
	}
}
