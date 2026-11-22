// Package policy decides whether to bid, for which campaign and creative,
// and at what price.
//
// Crawl is the policy of Part II, before the DSP has a trained model:
//
//  1. Eligibility: campaigns that target the country, have budget left,
//     and have a creative that fits the placement (Chapter 6).
//  2. Selection: one campaign at random, weighted toward campaigns with
//     more of today's budget left, and one of its creatives at random.
//     The selection probability is logged, so later models can correct
//     for how candidates were chosen (Chapter 8).
//  3. Prediction: the deployed click model of Chapter 9, or before there
//     is one, a prior click rate for the placement. For install campaigns,
//     the bundle's smoothed click-to-install rate for the segment, or
//     before there is one, a prior rate.
//  4. Pricing: the click's value to the advertiser, minus our margin,
//     shaded down for the first-price auction, with a little random
//     exploration around that price (Chapter 10).
//  5. Holdout: a fixed 5% of devices get a ghost bid instead of a real
//     one, so incrementality can be measured later (Chapter 23).
package policy

import (
	"hash/fnv"
	"math"
	"math/rand/v2"

	"github.com/bgweber/adtechbook/code/dsp/internal/campaigns"
)

// CTRPrior is the click rate assumed for each placement before there's a
// click model.
var CTRPrior = map[string]float64{
	"rewarded_video": 0.035, "interstitial": 0.025, "banner": 0.006, "native": 0.012,
}

// Crawl holds the policy's settings.
type Crawl struct {
	Margin        float64 // our share of the advertiser's value
	Shade         float64 // first-price shading factor
	ExploreShare  float64 // share of bids with a random price multiplier
	ExploreLo     float64
	ExploreHi     float64
	HoldoutPerMil int    // devices per thousand in the holdout
	HoldoutSalt   string // change it to draw a new holdout
	CVRPrior      float64
	ValuePrior    float64
	ActionPrior   float64
	CapCPM        float64
}

// DefaultCrawl matches the simulator's built-in crawl policy.
func DefaultCrawl() *Crawl {
	return &Crawl{Margin: 0.2, Shade: 0.7, ExploreShare: 0.05, ExploreLo: 0.6, ExploreHi: 1.5,
		HoldoutPerMil: 50, HoldoutSalt: "holdout-v1", CVRPrior: 0.03, ValuePrior: 1.0,
		ActionPrior: 0.2, CapCPM: 50}
}

// Decision is everything the policy decided, for the response and the
// ledger. Every term of the price is kept.
type Decision struct {
	Action          string  `json:"action"` // bid, ghost, nobid
	Reason          string  `json:"reason,omitempty"`
	CampaignID      int     `json:"campaign_id"`
	CreativeID      int     `json:"creative_id"`
	NEligible       int     `json:"n_eligible"`
	SelectionProb   float64 `json:"selection_prob"`
	PClick          float64 `json:"p_click"`
	PClickFrom      string  `json:"p_click_from"`              // model, fallback_*, or prior
	Throttle        float64 `json:"throttle,omitempty"`        // the campaign's throttle rate
	PInstallClick   float64 `json:"p_install_click,omitempty"` // install campaigns
	InstallRateFrom string  `json:"install_rate_from,omitempty"`
	AdvertiserValue float64 `json:"advertiser_value"` // CPM
	MediaValue      float64 `json:"media_value"`      // CPM, after margin
	ExploreMult     float64 `json:"explore_mult"`
	Bid             float64 `json:"bid"` // CPM
	Holdout         bool    `json:"holdout"`
	Model           string  `json:"model,omitempty"` // click model version; empty = prior
	// Filtered says which campaigns were removed before selection, and
	// by which filter: geo, not_started, no_fitting_creative,
	// budget_spent, paused, or throttled (Chapters 6 and 12).
	Filtered map[int]string `json:"filtered,omitempty"`
}

// Prediction is what the deployed bundle says about one candidate.
type Prediction struct {
	PClick          float64
	PClickFrom      string // model, fallback_segment, fallback_bounds, fallback_prior
	Version         string
	PInstallClick   float64 // 0 when the bundle has no install rates
	InstallRateFrom string  // segment, campaign, overall, or prior
}

// Predictor scores the chosen campaign and creative. ok is false when
// no model is deployed.
type Predictor func(c *campaigns.Campaign, cr campaigns.Creative) (Prediction, bool)

// Value is what an impression is worth to the advertiser, as a CPM. For
// install goals, a click is worth the target times the chance a click
// leads to an install (Chapter 10).
func (p *Crawl) Value(c *campaigns.Campaign, pClick, pInstall float64) float64 {
	switch c.Goal {
	case "CPC":
		return c.Target * pClick * 1000
	case "CPI":
		return c.Target * pClick * pInstall * 1000
	case "CPA":
		return c.Target * pClick * pInstall * p.ActionPrior * 1000
	case "ROAS":
		return pClick * pInstall * p.ValuePrior / c.Target * 1000
	}
	return 0
}

// InHoldout reports whether a device is in the holdout. The bucket comes
// from a salted hash of the device ID, so it's stable across requests,
// days, and restarts.
func (p *Crawl) InHoldout(ifa string) bool {
	h := fnv.New64a()
	h.Write([]byte(p.HoldoutSalt + ":" + ifa))
	return int(h.Sum64()%1000) < p.HoldoutPerMil
}

// Decide runs selection, prediction, and pricing for one request.
func (p *Crawl) Decide(cands []campaigns.Candidate, placement, ifa string,
	floor float64, rng *rand.Rand, predict Predictor) Decision {
	if len(cands) == 0 {
		return Decision{Action: "nobid", Reason: "no_eligible_campaign", CampaignID: -1, CreativeID: -1}
	}
	total := 0.0
	for _, c := range cands {
		total += c.Remaining + 0.1
	}
	pick, acc, r := len(cands)-1, 0.0, rng.Float64()*total
	for i, c := range cands {
		acc += c.Remaining + 0.1
		if r < acc {
			pick = i
			break
		}
	}
	c := cands[pick]
	cr := c.Creatives[rng.IntN(len(c.Creatives))]
	d := Decision{
		CampaignID: c.Campaign.ID, CreativeID: cr.ID, NEligible: len(cands),
		SelectionProb: (c.Remaining + 0.1) / total / float64(len(c.Creatives)),
		PClick:        CTRPrior[placement], PClickFrom: "prior", ExploreMult: 1,
	}
	pInstall, from := p.CVRPrior, "prior"
	if predict != nil {
		if pr, ok := predict(c.Campaign, cr); ok {
			d.PClick, d.Model, d.PClickFrom = pr.PClick, pr.Version, pr.PClickFrom
			if d.PClickFrom == "" {
				d.PClickFrom = "model"
			}
			if pr.PInstallClick > 0 {
				pInstall, from = pr.PInstallClick, pr.InstallRateFrom
			}
		}
	}
	if c.Campaign.Goal != "CPC" {
		d.PInstallClick, d.InstallRateFrom = pInstall, from
	}
	d.AdvertiserValue = p.Value(c.Campaign, d.PClick, pInstall)
	d.MediaValue = d.AdvertiserValue * (1 - p.Margin)
	if rng.Float64() < p.ExploreShare {
		lo, hi := math.Log(p.ExploreLo), math.Log(p.ExploreHi)
		d.ExploreMult = math.Exp(lo + rng.Float64()*(hi-lo))
	}
	if d.MediaValue < floor {
		d.Action, d.Reason = "nobid", "below_floor"
		return d
	}
	bid := math.Min(d.MediaValue*p.Shade*d.ExploreMult, p.CapCPM)
	if floor > 0 {
		bid = math.Max(bid, floor)
	}
	d.Bid = math.Round(bid*1e4) / 1e4
	d.Holdout = p.InHoldout(ifa)
	if d.Holdout {
		d.Action, d.Reason = "ghost", "holdout"
	} else {
		d.Action = "bid"
	}
	return d
}
