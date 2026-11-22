package ops

import (
	"math"
	"strings"
	"testing"

	"github.com/bgweber/adtechbook/code/dsp/internal/campaigns"
	"github.com/bgweber/adtechbook/code/dsp/internal/clock"
)

func book() *campaigns.Book {
	b := campaigns.NewBook()
	b.Load([]campaigns.Campaign{
		{ID: 1, Goal: "CPC", Target: 0.1, DailyBudget: 24, Geos: "IND"},
		{ID: 2, Goal: "CPC", Target: 0.1, DailyBudget: 24, Geos: "IND"},
	}, nil)
	return b
}

func TestSwitchesAuditAndPersistence(t *testing.T) {
	dir := t.TempDir()
	o := New(dir)
	if _, err := o.Set("campaign", "3", true, "test", 1000); err != nil {
		t.Fatal(err)
	}
	if _, err := o.Set("model", "", true, "test", 2000); err != nil {
		t.Fatal(err)
	}
	if _, err := o.Set("bogus", "", true, "test", 0); err == nil {
		t.Fatal("unknown switch accepted")
	}
	sw := o.Switches()
	if !sw.Campaigns[3] || !sw.Model || sw.Global || sw.Version != 2 {
		t.Fatalf("switches %+v", sw)
	}
	if a := o.Audit(); len(a) != 2 || !strings.Contains(a[0].What, "click model") {
		t.Fatalf("audit %+v", a)
	}
	// a restarted bidder comes back with the same switches and log
	o2 := New(dir)
	if s := o2.Switches(); !s.Campaigns[3] || !s.Model || len(o2.Audit()) != 2 {
		t.Fatalf("after restart: %+v", s)
	}
	o2.Set("campaign", "3", false, "test", 3000)
	if o2.Switches().Campaigns[3] {
		t.Fatal("campaign 3 still paused")
	}
}

// A campaign spending far ahead of plan gets throttled down, and one
// behind plan stays open.
func TestThrottle(t *testing.T) {
	o := New(t.TempDir())
	b := book()
	day := int64(1)
	start := day * clock.DayMS
	o.Step(start, b) // a new day
	// campaign 1 spends half its budget in the first six hours (ahead of
	// the 25% plan); campaign 2 spends nothing
	b.AddSpend(int(day), 1, 12)
	ts := start + 6*hourMS
	o.Step(ts, b)
	if r := o.Rate(1); r >= 1 {
		t.Fatalf("campaign 1 ahead of plan, rate %v", r)
	}
	if r := o.Rate(2); r != 1 {
		t.Fatalf("campaign 2 behind plan, rate %v", r)
	}
	o.Throttle = false
	if o.Rate(1) != 1 {
		t.Fatal("throttling off should give rate 1")
	}
}

func TestSpendRateAlert(t *testing.T) {
	o := New(t.TempDir())
	b := book()
	day := int64(1)
	start := day * clock.DayMS
	o.Step(start, b)
	// campaign 1: $10 billed in hour 5, its plan for an hour is $1
	for i := 0; i < 100; i++ {
		ts := start + 5*hourMS + int64(i)*30_000
		o.ObserveBilling(ts, day, "alpha", "banner", 1, 0.1)
		b.AddSpend(int(day), 1, 0.1)
	}
	o.Step(start+6*hourMS, b)
	var fired []string
	for _, a := range o.Alerts() {
		if a.State == "firing" {
			fired = append(fired, a.Name+"|"+a.Scope)
		}
	}
	if len(fired) == 0 || fired[0] != "spend_rate|campaign 1" && fired[0] != "spend_rate|overall" {
		t.Fatalf("fired %v", fired)
	}
	for _, f := range fired {
		if strings.Contains(f, "campaign 2") {
			t.Fatalf("campaign 2 shouldn't alert: %v", fired)
		}
	}
}

func TestPredictionCollapse(t *testing.T) {
	o := New(t.TempDir())
	b := book()
	start := int64(clock.DayMS)
	for i := 0; i < 400; i++ {
		o.Observe(BidObs{TsMS: start + int64(i)*5000, Exchange: "alpha", Action: "bid",
			PClick: 0.25, PFrom: "model", ModelOn: true})
	}
	o.Step(start+hourMS, b)
	for _, a := range o.Alerts() {
		if a.Name == "prediction_collapse" && a.State == "firing" {
			return
		}
	}
	t.Fatalf("no prediction_collapse alert: %+v", o.Alerts())
}

func TestReconciliation(t *testing.T) {
	o := New(t.TempDir())
	o.ObserveBilling(1000, 0, "beta", "banner", 1, 1.0)
	o.ObserveBilling(1000, 0, "beta", "native", 1, 0.6)
	o.LoadInvoices([]InvoiceLine{
		{Day: 0, Exchange: "beta", Placement: "banner", Imps: 1, Spend: 1.0},
		{Day: 0, Exchange: "beta", Placement: "native", Imps: 2, Spend: 1.0},
	})
	r := o.Reconciliation()
	if len(r) != 1 || math.Abs(r[0].Gap-(-0.2)) > 1e-9 {
		t.Fatalf("recon %+v", r)
	}
	for _, p := range r[0].Placements {
		if p.Placement == "native" && math.Abs(p.Gap+0.4) > 1e-9 {
			t.Fatalf("native gap %v", p.Gap)
		}
	}
	found := false
	for _, a := range o.Alerts() {
		found = found || (a.Name == "reconciliation_gap" && a.State == "firing")
	}
	if !found {
		t.Fatal("no reconciliation alert for a 20% gap")
	}
}
