package policy

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/bgweber/adtechbook/code/dsp/internal/campaigns"
)

func TestGoalValue(t *testing.T) {
	p := DefaultCrawl()
	cpi := &campaigns.Campaign{Goal: "CPI", Target: 3}
	roas := &campaigns.Campaign{Goal: "ROAS", Target: 0.4}
	// 2% clicks, 5% installs after a click, 0.02% after a view
	pInst := 0.02*0.05 + 0.98*0.0002
	if v := p.GoalValue(cpi, 0.02, 0.05, 0.0002, 0); math.Abs(v-3*pInst*1000) > 1e-9 {
		t.Fatalf("CPI value %g", v)
	}
	// the chapter's example: IPM 1.0 at $2.00 D7 revenue per install and a
	// 40% ROAS target is a $5.00 CPM
	if v := p.GoalValue(roas, 0.02, 0.05, 0, 2.0); math.Abs(v-5) > 1e-9 {
		t.Fatalf("ROAS value %g, want 5", v)
	}
}

func TestKappasLoadBoundsAndDefaults(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "kappa-0001.json"),
		[]byte(`{"version":"kappa-0001","overall":0.8,"by_campaign":{"1":0.7,"2":5}}`), 0o644)
	k := &Kappas{}
	if k.Get(1) != 1 {
		t.Fatal("no calibration should be 1")
	}
	if _, err := k.Load(dir); err != nil {
		t.Fatal(err)
	}
	if k.Get(1) != 0.7 || k.Get(2) != 2 || k.Get(3) != 0.8 {
		t.Fatalf("got %g %g %g", k.Get(1), k.Get(2), k.Get(3))
	}
}

func TestThompsonFavorsTheBetterCreative(t *testing.T) {
	c := NewCreatives()
	c.SetThompson(true)
	// three creatives, day 0: one installs at 0.6%, two at 0.15%
	for i := 0; i < 5000; i++ {
		for _, cr := range []int{1, 2, 3} {
			id := string(rune('a'+cr)) + string(rune(i))
			c.Impression(id, cr, 9, 0)
			rate := 0.0015
			if cr == 1 {
				rate = 0.006
			}
			if float64(i%1000)/1000 < rate {
				c.Install(id)
			}
		}
	}
	rng := rand.New(rand.NewPCG(1, 2))
	// day 0's impressions don't count until a day has passed
	if _, p, _ := c.Pick([]int{1, 2, 3}, rng); p > 0.6 {
		t.Fatalf("fresh impressions counted: probability %.2f", p)
	}
	c.Impression("tomorrow", 2, 9, 1)
	n := 0
	for i := 0; i < 1000; i++ {
		if i, _, from := c.Pick([]int{1, 2, 3}, rng); i == 0 && from == "thompson" {
			n++
		}
	}
	if n < 900 {
		t.Fatalf("the best creative won %d of 1000 draws", n)
	}
	r := c.Report()[9]
	if len(r) != 3 || r[0].Share < 0.9 {
		t.Fatalf("report %+v", r)
	}
}

func TestCreativesSurviveARestart(t *testing.T) {
	a := NewCreatives()
	a.SetThompson(true)
	a.Impression("r1", 1, 9, 3)
	a.Impression("r2", 2, 9, 4)
	a.Install("r1")
	path := filepath.Join(t.TempDir(), "creatives.json")
	if err := a.Save(path); err != nil {
		t.Fatal(err)
	}
	b := NewCreatives()
	if err := b.Load(path); err != nil {
		t.Fatal(err)
	}
	if !b.Thompson() {
		t.Fatal("Thompson sampling switched off by a restart")
	}
	r := b.Report()[9]
	if len(r) != 2 || r[0].Imps != 1 || r[0].Installs != 1 {
		t.Fatalf("after a restart: %+v", r)
	}
}

// The logged chance of each creative matches how often it's picked, across
// refreshes of the cached posteriors (Chapter 17).
func TestThompsonLoggedChancesMatchPicks(t *testing.T) {
	c := NewCreatives()
	c.SetThompson(true)
	for i := 0; i < 3000; i++ {
		for cr := 1; cr <= 3; cr++ {
			id := fmt.Sprintf("%d-%d", cr, i)
			c.Impression(id, cr, 9, 0)
			if i%1000 < 2+cr { // 0.3%, 0.4%, 0.5%: close, so all get picked
				c.Install(id)
			}
		}
	}
	c.Impression("tomorrow", 1, 9, 1)
	rng := rand.New(rand.NewPCG(3, 4))
	const n = 3 * PickRefresh
	count, logged := [3]float64{}, [3]float64{}
	for k := 0; k < n; k++ {
		i, p, _ := c.Pick([]int{1, 2, 3}, rng)
		count[i]++
		logged[i] = p
	}
	for i := range count {
		if math.Abs(count[i]/n-logged[i]) > 0.03 {
			t.Fatalf("creative %d: picked %.3f, logged %.3f", i, count[i]/n, logged[i])
		}
	}
}
