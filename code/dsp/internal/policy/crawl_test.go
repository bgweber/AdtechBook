package policy

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/bgweber/adtechbook/code/dsp/internal/campaigns"
)

func book() *campaigns.Book {
	b := campaigns.NewBook()
	b.Load([]campaigns.Campaign{
		{ID: 0, Goal: "CPC", Target: 0.10, DailyBudget: 10, Geos: "BRA,IND"},
		{ID: 1, Goal: "CPI", Target: 2.00, DailyBudget: 10, Geos: "IND"},
	}, []campaigns.Creative{
		{ID: 10, CampaignID: 0, Format: "video"}, {ID: 11, CampaignID: 0, Format: "static"},
		{ID: 20, CampaignID: 1, Format: "playable"},
	})
	return b
}

func TestEligibility(t *testing.T) {
	b := book()
	if n := len(b.Eligible("IND", "rewarded_video", 0)); n != 2 {
		t.Errorf("IND rewarded video: want 2 campaigns, got %d", n)
	}
	if n := len(b.Eligible("BRA", "banner", 0)); n != 1 {
		t.Errorf("BRA banner: want 1 campaign, got %d", n)
	}
	if n := len(b.Eligible("USA", "interstitial", 0)); n != 0 {
		t.Errorf("USA: want no campaigns, got %d", n)
	}
	b.AddSpend(0, 0, 10) // campaign 0 spends its whole budget
	if n := len(b.Eligible("BRA", "banner", 0)); n != 0 {
		t.Errorf("after budget is spent: want 0, got %d", n)
	}
	if n := len(b.Eligible("BRA", "banner", 1)); n != 1 {
		t.Errorf("next day: want 1, got %d", n)
	}
}

func TestPriceTerms(t *testing.T) {
	p := DefaultCrawl()
	p.ExploreShare = 0
	b := book()
	rng := rand.New(rand.NewPCG(1, 2))
	cands := b.Eligible("BRA", "rewarded_video", 0)
	d := p.Decide(cands, "rewarded_video", "device-1", 0, rng, nil)
	// CPC $0.10 x 3.5% click rate = $3.50 CPM, minus 20% margin = $2.80,
	// shaded by 0.7 = $1.96.
	if math.Abs(d.AdvertiserValue-3.5) > 1e-9 || math.Abs(d.MediaValue-2.8) > 1e-9 ||
		math.Abs(d.Bid-1.96) > 1e-9 {
		t.Errorf("unexpected price terms: %+v", d)
	}
	if d.SelectionProb != 1 {
		t.Errorf("one campaign with one fitting creative: want 1, got %v", d.SelectionProb)
	}
	if d := p.Decide(cands, "rewarded_video", "device-1", 5, rng, nil); d.Reason != "below_floor" {
		t.Errorf("floor above value: want below_floor, got %+v", d)
	}
}

func TestHoldoutIsStableAndSized(t *testing.T) {
	p := DefaultCrawl()
	n := 0
	for i := 0; i < 100000; i++ {
		id := string(rune('a'+i%26)) + string(rune(i))
		if p.InHoldout(id) {
			n++
		}
		if p.InHoldout(id) != p.InHoldout(id) {
			t.Fatal("holdout not stable")
		}
	}
	if share := float64(n) / 100000; share < 0.04 || share > 0.06 {
		t.Errorf("holdout share %.3f, want about 0.05", share)
	}
}

// A CPI campaign values a click at the target CPI times the chance it
// leads to an install: the prior before there's a bundle, then the
// bundle's rate (Chapter 10).
func TestCPIValue(t *testing.T) {
	p := DefaultCrawl()
	c := &campaigns.Campaign{Goal: "CPI", Target: 1.20}
	// the chapter's example: $1.20 x 3% x 1.5% x 1000 = $0.54 CPM
	if v := p.Value(c, 0.015, 0.03); math.Abs(v-0.54) > 1e-12 {
		t.Fatalf("CPI value %v, want 0.54", v)
	}
	cands := []campaigns.Candidate{{Campaign: c, Creatives: []campaigns.Creative{{ID: 1}}, Remaining: 1}}
	p.ExploreShare, p.HoldoutPerMil = 0, 0
	rng := rand.New(rand.NewPCG(1, 2))
	d := p.Decide(cands, "banner", "d", 0, rng, nil)
	if d.PInstallClick != p.CVRPrior || d.InstallRateFrom != "prior" {
		t.Fatalf("no bundle: want the prior, got %v from %s", d.PInstallClick, d.InstallRateFrom)
	}
	d = p.Decide(cands, "banner", "d", 0, rng, func(*campaigns.Campaign, campaigns.Creative) (Prediction, bool) {
		return Prediction{PClick: 0.015, Version: "v", PInstallClick: 0.05, InstallRateFrom: "segment"}, true
	})
	if d.PInstallClick != 0.05 || math.Abs(d.AdvertiserValue-1.20*0.015*0.05*1000) > 1e-12 {
		t.Fatalf("with a bundle: %+v", d)
	}
}

// With a cap, the auction prices only the campaigns with the best recent
// scores (Chapter 17); a campaign never priced goes first.
func TestAuctionCap(t *testing.T) {
	p := DefaultCrawl()
	p.ExploreShare, p.HoldoutPerMil, p.AuctionExplore = 0, 0, 0
	p.SetSelection("auction")
	var cands []campaigns.Candidate
	for i, target := range []float64{0.02, 0.50, 0.10, 0.05} {
		cands = append(cands, campaigns.Candidate{Campaign: &campaigns.Campaign{ID: i, Goal: "CPC", Target: target},
			Creatives: []campaigns.Creative{{ID: 10 * i}}, Remaining: 1})
	}
	priced := map[int]int{}
	predict := func(c *campaigns.Campaign, _ campaigns.Creative) (Prediction, bool) {
		priced[c.ID]++
		return Prediction{PClick: 0.01, Version: "v"}, true
	}
	rng := rand.New(rand.NewPCG(1, 2))
	p.Decide(cands, "banner", "d", 0, rng, predict) // uncapped: every campaign gets a score
	if len(priced) != 4 {
		t.Fatalf("priced %v", priced)
	}
	p.AuctionCap.Store(2)
	clear(priced)
	d := p.Decide(cands, "banner", "d", 0, rng, predict)
	if len(priced) != 2 || priced[1] != 1 || priced[2] != 1 || d.Capped != 2 || d.CampaignID != 1 {
		t.Fatalf("capped at 2: priced %v, capped %d, picked %d", priced, d.Capped, d.CampaignID)
	}
	// a new campaign has no estimate yet, so it's priced first
	cands = append(cands, campaigns.Candidate{Campaign: &campaigns.Campaign{ID: 9, Goal: "CPC", Target: 0.01},
		Creatives: []campaigns.Creative{{ID: 90}}, Remaining: 1})
	clear(priced)
	p.Decide(cands, "banner", "d", 0, rng, predict)
	if priced[9] != 1 || priced[1] != 1 || len(priced) != 2 {
		t.Fatalf("new campaign: priced %v", priced)
	}
}

// The incrementality tilt (Chapter 32) multiplies an install campaign's
// value and logs the ratio; CPC campaigns are left alone.
func TestTilt(t *testing.T) {
	p := DefaultCrawl()
	p.ExploreShare, p.HoldoutPerMil = 0, 0
	c := &campaigns.Campaign{Goal: "CPI", Target: 1.20}
	cands := []campaigns.Candidate{{Campaign: c, Creatives: []campaigns.Creative{{ID: 1}}, Remaining: 1}}
	pred := func(*campaigns.Campaign, campaigns.Creative) (Prediction, bool) {
		return Prediction{PClick: 0.015, Version: "v", PInstallClick: 0.05, InstallRateFrom: "model"}, true
	}
	rng := rand.New(rand.NewPCG(1, 2))
	base := p.DecideWith(cands, "banner", "d", 0, rng, pred, nil, nil)
	tilt := func(*campaigns.Campaign, campaigns.Creative) (float64, float64, bool) { return 0.5, 0.4, true }
	d := p.DecideWith(cands, "banner", "d", 0, rng, pred, nil, nil, tilt)
	if math.Abs(d.AdvertiserValue-0.5*base.AdvertiserValue) > 1e-12 || d.Tilt != 0.5 || d.Uplift != 0.4 {
		t.Fatalf("tilted %+v, base value %v", d, base.AdvertiserValue)
	}
	c.Goal = "CPC"
	if d := p.DecideWith(cands, "banner", "d", 0, rng, pred, nil, nil, tilt); d.Tilt != 0 {
		t.Fatalf("a CPC campaign was tilted: %+v", d)
	}
}

// Weighted goals (Chapter 33): a second CPI goal mixes into the value by
// its weight, and the ledger gets both goals' terms.
func TestWeightedGoals(t *testing.T) {
	p := DefaultCrawl()
	p.ExploreShare, p.HoldoutPerMil = 0, 0
	c := &campaigns.Campaign{ID: 7, Goal: "ROAS", Target: 0.2}
	cands := []campaigns.Candidate{{Campaign: c, Creatives: []campaigns.Creative{{ID: 1}}, Remaining: 1}}
	pred := func(*campaigns.Campaign, campaigns.Creative) (Prediction, bool) {
		return Prediction{PClick: 0.01, Version: "v", PInstallClick: 0.1, PInstallView: 0.001,
			InstallRateFrom: "model", ValuePerInstall: 0.6, ValueFrom: "model"}, true
	}
	rng := rand.New(rand.NewPCG(1, 2))
	one := p.DecideWith(cands, "banner", "d", 0, rng, pred, nil, nil)
	if err := p.SetLevers(&LeverSet{Campaigns: map[int]Levers{7: {SecondGoal: "CPI", SecondTarget: 3, Weight: 0.5}}}, ""); err != nil {
		t.Fatal(err)
	}
	two := p.DecideWith(cands, "banner", "d", 0, rng, pred, nil, nil)
	pInst := 0.01*0.1 + 0.99*0.001
	want := 0.5*one.AdvertiserValue + 0.5*3*pInst*1000
	if math.Abs(two.AdvertiserValue-want) > 1e-9 || len(two.Goals) != 2 || two.Goals[1].Goal != "CPI" {
		t.Fatalf("weighted value %v, want %v: %+v", two.AdvertiserValue, want, two.Goals)
	}
	if err := p.SetLevers(&LeverSet{Campaigns: map[int]Levers{7: {SecondGoal: "ROAS", SecondTarget: 1, Weight: 0.5}}}, ""); err == nil {
		t.Fatal("a second ROAS goal was accepted")
	}
}
