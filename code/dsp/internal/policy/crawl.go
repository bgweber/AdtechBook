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
	"sort"
	"sync"
	"sync/atomic"

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

	selection atomic.Int32 // a Selection, switchable at run time

	// Walk pricing (Chapter 21): calibration multipliers by campaign,
	// Thompson sampling over creatives, and the share of internal
	// auctions that pick a campaign at random instead, to keep learning
	// and keep the selection probabilities useful offline.
	Kappa          *Kappas
	Creatives      *Creatives
	AuctionExplore float64
	// Pace is a campaign's pacing multiplier and manual overbid (Chapter
	// 22); nil is 1 and 1.
	Pace func(campaignID int) (gamma, overbid float64)

	// AuctionCap prices only the most promising campaigns in the internal
	// auction (Chapter 17), by a cheap estimate: each campaign's recent
	// mean score. 0 prices every eligible campaign.
	AuctionCap atomic.Int32
	est        scoreEstimates
	audit      capAudit
}

// scoreEstimates are each campaign's recent mean auction score, for the
// candidate cap.
type scoreEstimates struct {
	mu sync.Mutex
	m  map[int]float64
}

// estAlpha is the weight of the newest score in the moving average.
const estAlpha = 0.02

func (e *scoreEstimates) get(id int) (float64, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	v, ok := e.m[id]
	return v, ok
}

func (e *scoreEstimates) update(id int, score float64) {
	if math.IsInf(score, -1) {
		score = 0 // below the floor: it couldn't have bid
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.m == nil {
		e.m = map[int]float64{}
	}
	if v, ok := e.m[id]; ok {
		e.m[id] = v + estAlpha*(score-v)
	} else {
		e.m[id] = score
	}
}

// Selection is how the crawl policy picks among eligible campaigns.
type Selection int32

const (
	// SelectBudget weights each campaign by the share of today's budget
	// it has left, plus 0.1 (Chapter 6). The default.
	SelectBudget Selection = iota
	// SelectUniform gives every eligible campaign the same chance.
	SelectUniform
	// SelectCPIFirst gives CPI campaigns three times the weight of others.
	SelectCPIFirst
	// SelectAuction prices every eligible campaign and picks the one with
	// the most expected surplus, the internal auction of Chapter 21.
	SelectAuction
)

// Selections are the names the console and the ledger use.
var Selections = []string{"budget", "uniform", "cpi-first", "auction"}

// SetSelection switches the selection policy; false for an unknown name.
// Off-policy evaluation (Chapter 16) estimates one from another's logs.
func (p *Crawl) SetSelection(name string) bool {
	for i, n := range Selections {
		if n == name {
			p.selection.Store(int32(i))
			return true
		}
	}
	return false
}

// SelectionName is the current selection policy.
func (p *Crawl) SelectionName() string { return Selections[p.selection.Load()] }

func (p *Crawl) weight(c campaigns.Candidate, sel Selection) float64 {
	switch sel {
	case SelectUniform:
		return 1
	case SelectCPIFirst:
		if c.Campaign.Goal == "CPI" {
			return 3
		}
		return 1
	}
	return c.Remaining + 0.1 // budget, and the auction's exploration
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
	Action        string  `json:"action"` // bid, ghost, nobid
	Reason        string  `json:"reason,omitempty"`
	CampaignID    int     `json:"campaign_id"`
	CreativeID    int     `json:"creative_id"`
	NEligible     int     `json:"n_eligible"`
	SelectionProb float64 `json:"selection_prob"`
	// What off-policy evaluation needs to compute another policy's
	// probability for the same choice (Chapter 16): the policy, the
	// eligible campaigns and their goals, and the chosen one's creatives.
	Selection       string   `json:"selection,omitempty"`
	Eligible        []int    `json:"eligible,omitempty"`
	EligibleGoals   []string `json:"eligible_goals,omitempty"`
	NCreatives      int      `json:"n_creatives,omitempty"`
	PClick          float64  `json:"p_click"`
	PClickFrom      string   `json:"p_click_from"`              // model, fallback_*, or prior
	Throttle        float64  `json:"throttle,omitempty"`        // the campaign's throttle rate
	PInstallClick   float64  `json:"p_install_click,omitempty"` // install campaigns
	InstallRateFrom string   `json:"install_rate_from,omitempty"`
	PInstallView    float64  `json:"p_install_view,omitempty"`   // view-through, from the conversion models
	Conversion      string   `json:"conversion_model,omitempty"` // conversion bundle version (Part III)
	// Walk pricing terms (Chapter 21): installs per impression from both
	// paths, D7 revenue per install for ROAS goals, the campaign's
	// target, and its calibration multiplier.
	PInstall        float64 `json:"p_install,omitempty"`
	ValuePerInstall float64 `json:"value_per_install,omitempty"`
	ValueFrom       string  `json:"value_from,omitempty"`
	Target          float64 `json:"target,omitempty"`
	Kappa           float64 `json:"kappa,omitempty"`
	Pacing          float64 `json:"pacing,omitempty"`        // gamma, from the pacing controller (Chapter 22)
	Overbid         float64 `json:"overbid,omitempty"`       // a manual overbid, logged apart from gamma
	Score           float64 `json:"score,omitempty"`         // the internal auction's expected surplus
	CreativeFrom    string  `json:"creative_from,omitempty"` // random or thompson
	AuctionExplored bool    `json:"auction_explored,omitempty"`
	Capped          int     `json:"capped,omitempty"`     // campaigns the auction cap left unpriced (Chapter 17)
	Candidates      []Cand  `json:"candidates,omitempty"` // the internal auction, best first
	AdvertiserValue float64 `json:"advertiser_value"`     // CPM
	MediaValue      float64 `json:"media_value"`          // CPM, after margin
	ExploreMult     float64 `json:"explore_mult"`
	Bid             float64 `json:"bid"` // CPM
	Holdout         bool    `json:"holdout"`
	Model           string  `json:"model,omitempty"` // click model version; empty = prior
	// The running experiment and this request's arm (Chapter 19), and in
	// shadow mode, the challenger's scores for the same choice.
	// How the bid was shaded (Chapter 20): fixed, or model, with the win
	// model's predicted win rate for the shaded bid.
	ShadeFrom   string       `json:"shade_from,omitempty"`
	ShadeFactor float64      `json:"shade,omitempty"` // shaded bid / media value
	WinProb     float64      `json:"win_prob,omitempty"`
	Experiment  string       `json:"experiment,omitempty"`
	Arm         string       `json:"arm,omitempty"` // control or treatment
	Shadow      *ShadowScore `json:"shadow,omitempty"`
	// Filtered says which campaigns were removed before selection, and
	// by which filter: geo, not_started, no_fitting_creative,
	// budget_spent, paused, or throttled (Chapters 6 and 12).
	Filtered map[int]string `json:"filtered,omitempty"`
}

// Cand is one campaign in the internal auction, for the bid ledger.
type Cand struct {
	CampaignID int     `json:"campaign_id"`
	Goal       string  `json:"goal"`
	Value      float64 `json:"value"` // advertiser value x kappa, CPM
	Bid        float64 `json:"bid"`   // shaded, before exploration and floors
	WinProb    float64 `json:"win_prob,omitempty"`
	Score      float64 `json:"score"`
	BelowFloor bool    `json:"below_floor,omitempty"` // its media value is below the floor: it can't bid
}

// ShadowScore is a challenger's prediction for a request it doesn't bid on.
type ShadowScore struct {
	Model         string  `json:"model"`
	PClick        float64 `json:"p_click"`
	PInstallClick float64 `json:"p_install_click,omitempty"`
	PInstallView  float64 `json:"p_install_view,omitempty"`
	LatencyUS     float64 `json:"latency_us"`
}

// Prediction is what the deployed bundle says about one candidate.
type Prediction struct {
	PClick          float64
	PClickFrom      string // model, fallback_segment, fallback_bounds, fallback_prior
	Version         string
	PInstallClick   float64 // 0 when the bundle has no install rates
	InstallRateFrom string  // model, segment, campaign, overall, or prior
	PInstallView    float64 // view-through installs per impression (Part III)
	ValuePerInstall float64 // expected D7 revenue per install, for ROAS goals (Chapter 21)
	ValueFrom       string
	Conversion      string // the conversion bundle's version, if one is deployed
}

// Shader prices a bid from the media value and the floor with a win rate
// model (Chapter 20): the bid with the most expected surplus, and its
// predicted win rate. ok is false to keep the fixed shade.
type Shader func(media, floor float64) (bid, winProb float64, ok bool)

// Predictor scores the chosen campaign and creative. ok is false when
// no model is deployed.
type Predictor func(c *campaigns.Campaign, cr campaigns.Creative) (Prediction, bool)

// Value is what an impression is worth to the advertiser, as a CPM. For
// install goals, a click is worth the target times the chance a click
// leads to an install (Chapter 10).
func (p *Crawl) Value(c *campaigns.Campaign, pClick, pInstall float64) float64 {
	return p.GoalValue(c, pClick, pInstall, 0, p.ValuePrior)
}

// GoalValue is the advertiser value for any goal (Chapter 21), from
// installs per impression by both paths, a click and a view:
// P(install) = p_click x p_install|click + (1 - p_click) x p_install|view.
// Without a view-through prediction, it's the crawl formula.
func (p *Crawl) GoalValue(c *campaigns.Campaign, pClick, pInstallClick, pInstallView,
	valuePerInstall float64) float64 {
	pInst := pClick*pInstallClick + (1-pClick)*pInstallView
	switch c.Goal {
	case "CPC":
		return c.Target * pClick * 1000
	case "CPI":
		return c.Target * pInst * 1000
	case "CPA":
		return c.Target * pInst * p.ActionPrior * 1000
	case "ROAS":
		return pInst * valuePerInstall / c.Target * 1000
	}
	return 0
}

func (p *Crawl) legacyValue(c *campaigns.Campaign, pClick, pInstall float64) float64 {
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

// priced is one campaign and creative, priced term by term.
type priced struct {
	c              campaigns.Candidate
	cr             campaigns.Creative
	crProb         float64
	crFrom         string
	pr             Prediction
	hasPred        bool
	pInst, vpi, k  float64
	gamma, overbid float64
	vfrom, ifrom   string
	value, media   float64 // advertiser value x kappa; after margin
	shaded, win    float64
	shadeFrom      string
	score          float64
	pInstallClick  float64
}

// price runs prediction, value, calibration, margin, and shading for one
// campaign, in the order of Chapter 21: predict, value, multipliers,
// margin, shade. Floors and caps come later, in Decide.
func (p *Crawl) price(c campaigns.Candidate, placement string, floor float64, rng *rand.Rand,
	predict Predictor, shade Shader) priced {
	x := priced{c: c}
	ids := make([]int, len(c.Creatives))
	for i, cr := range c.Creatives {
		ids[i] = cr.ID
	}
	i, prob, from := p.Creatives.Pick(ids, rng) // uniform when off or nil, as in crawl
	x.cr, x.crProb, x.crFrom = c.Creatives[i], prob, from
	x.pInstallClick, x.ifrom = p.CVRPrior, "prior"
	if predict != nil {
		if pr, ok := predict(c.Campaign, x.cr); ok {
			x.pr, x.hasPred = pr, true
			if x.pr.PClickFrom == "" {
				x.pr.PClickFrom = "model"
			}
			if pr.PInstallClick > 0 {
				x.pInstallClick, x.ifrom = pr.PInstallClick, pr.InstallRateFrom
			}
		}
	}
	if !x.hasPred {
		x.pr = Prediction{PClick: CTRPrior[placement], PClickFrom: "prior"}
	}
	x.vpi, x.vfrom = p.ValuePrior, "prior"
	if x.pr.ValuePerInstall > 0 {
		x.vpi, x.vfrom = x.pr.ValuePerInstall, x.pr.ValueFrom
	}
	pc := x.pr.PClick
	x.pInst = pc*x.pInstallClick + (1-pc)*x.pr.PInstallView
	x.k = 1
	if c.Campaign.Goal != "CPC" {
		x.k = p.Kappa.Get(c.Campaign.ID)
	}
	if x.pr.PInstallView > 0 || x.pr.ValuePerInstall > 0 {
		x.value = p.GoalValue(c.Campaign, pc, x.pInstallClick, x.pr.PInstallView, x.vpi) * x.k
	} else {
		x.value = p.legacyValue(c.Campaign, pc, x.pInstallClick) * x.k // crawl, unchanged
	}
	// pacing scales the value before the margin and shading (Chapter 22);
	// the auction's score still uses the unscaled value
	x.gamma, x.overbid = 1, 1
	if p.Pace != nil {
		x.gamma, x.overbid = p.Pace(c.Campaign.ID)
	}
	x.media = x.value * x.gamma * x.overbid * (1 - p.Margin)
	x.shaded, x.shadeFrom = x.media*p.Shade, "fixed"
	if shade != nil && x.media >= floor && x.media > 0 {
		if b, w, ok := shade(x.media, floor); ok && b > 0 {
			x.shaded, x.shadeFrom, x.win = b, "model", w
		}
	}
	// expected surplus: the chance of winning times what the impression is
	// worth to the advertiser minus what we'd pay; with the fixed shade
	// there's no win model, so the surplus if it wins
	w := x.win
	if x.shadeFrom != "model" {
		w = 1
	}
	b := math.Max(x.shaded, floor)
	x.score = w * (x.value - b)
	if x.media < floor {
		x.score = math.Inf(-1)
	}
	return x
}

// toPrice is which candidates the auction prices: all of them, or with a
// cap, the k with the best recent scores. One never priced yet goes first,
// so each gets a score.
func (p *Crawl) toPrice(cands []campaigns.Candidate, k int) []int {
	idx := make([]int, len(cands))
	for i := range idx {
		idx[i] = i
	}
	if k <= 0 || len(cands) <= k {
		return idx
	}
	est := make([]float64, len(cands))
	for i, c := range cands {
		if v, ok := p.est.get(c.Campaign.ID); ok {
			est[i] = v
		} else {
			est[i] = math.Inf(1)
		}
	}
	sort.SliceStable(idx, func(a, b int) bool { return est[idx[a]] > est[idx[b]] })
	return idx[:k]
}

// CapAudit is what the auction cap costs (Chapter 17), from requests
// priced both ways: how often the capped auction's best is the full
// auction's best, and the share of the best score given up when it isn't.
type CapAudit struct {
	Audits int64   `json:"audits"`
	Same   float64 `json:"same_share"`
	Lost   float64 `json:"score_lost_share"`
}

type capAudit struct {
	mu               sync.Mutex
	n, same          int64
	bestSum, lostSum float64
}

// AuditCap prices every candidate and compares the full auction's best
// with the capped auction's. The bidder calls it on a sample of load test
// requests, after their timings are taken.
func (p *Crawl) AuditCap(cands []campaigns.Candidate, placement string, floor float64, rng *rand.Rand,
	predict Predictor, shade Shader) {
	k := int(p.AuctionCap.Load())
	if k <= 0 || len(cands) <= k {
		return
	}
	inCap := map[int]bool{}
	for _, i := range p.toPrice(cands, k) {
		inCap[i] = true
	}
	best, bestCap := math.Inf(-1), math.Inf(-1)
	for i, c := range cands {
		sc := p.price(c, placement, floor, rng, predict, shade).score
		best = math.Max(best, sc)
		if inCap[i] {
			bestCap = math.Max(bestCap, sc)
		}
	}
	if math.IsInf(best, -1) || best <= 0 {
		return // nothing could bid profitably either way
	}
	p.audit.mu.Lock()
	defer p.audit.mu.Unlock()
	p.audit.n++
	if bestCap == best {
		p.audit.same++
	}
	p.audit.bestSum += best
	p.audit.lostSum += best - math.Max(bestCap, 0)
}

// CapAuditResult reports the audits so far; ResetCapAudit clears them.
func (p *Crawl) CapAuditResult() CapAudit {
	p.audit.mu.Lock()
	defer p.audit.mu.Unlock()
	a := CapAudit{Audits: p.audit.n}
	if a.Audits > 0 {
		a.Same = float64(p.audit.same) / float64(a.Audits)
		a.Lost = p.audit.lostSum / p.audit.bestSum
	}
	return a
}

// ResetCapAudit forgets the audits, as when the cap changes.
func (p *Crawl) ResetCapAudit() {
	p.audit.mu.Lock()
	p.audit.n, p.audit.same, p.audit.bestSum, p.audit.lostSum = 0, 0, 0, 0
	p.audit.mu.Unlock()
}

// Decide runs selection, prediction, and pricing for one request.
func (p *Crawl) Decide(cands []campaigns.Candidate, placement, ifa string,
	floor float64, rng *rand.Rand, predict Predictor, shade ...Shader) Decision {
	if len(cands) == 0 {
		return Decision{Action: "nobid", Reason: "no_eligible_campaign", CampaignID: -1, CreativeID: -1}
	}
	var sh Shader
	if len(shade) > 0 {
		sh = shade[0]
	}
	sel := Selection(p.selection.Load())
	weighted := func(s Selection) (int, float64) {
		total := 0.0
		for _, c := range cands {
			total += p.weight(c, s)
		}
		pick, acc, r := len(cands)-1, 0.0, rng.Float64()*total
		for i, c := range cands {
			acc += p.weight(c, s)
			if r < acc {
				pick = i
				break
			}
		}
		return pick, total
	}
	var x priced
	var prob float64
	var cl []Cand
	explored := false
	capped := 0
	if sel == SelectAuction {
		// the internal auction (Chapter 21): price every campaign, pick the
		// most expected surplus; a share of requests explore instead
		priceIdx := p.toPrice(cands, int(p.AuctionCap.Load()))
		all := make([]priced, len(cands))
		done := make([]bool, len(cands))
		best := priceIdx[0]
		for _, i := range priceIdx {
			all[i], done[i] = p.price(cands[i], placement, floor, rng, predict, sh), true
			p.est.update(cands[i].Campaign.ID, all[i].score)
			if all[i].score > all[best].score {
				best = i
			}
		}
		pick := best
		eps := p.AuctionExplore
		total := 0.0
		for _, c := range cands {
			total += p.weight(c, SelectBudget)
		}
		if rng.Float64() < eps {
			pick, _ = weighted(SelectBudget)
			explored = true
			if !done[pick] { // exploring past the cap: price it now
				all[pick], done[pick] = p.price(cands[pick], placement, floor, rng, predict, sh), true
			}
		}
		x = all[pick]
		prob = eps * p.weight(cands[pick], SelectBudget) / total
		if pick == best {
			prob += 1 - eps
		}
		var order []int
		for i := range all {
			if done[i] {
				order = append(order, i)
			}
		}
		capped = len(cands) - len(order)
		sort.Slice(order, func(a, b int) bool { return all[order[a]].score > all[order[b]].score })
		for _, i := range order[:min(5, len(order))] {
			y := all[i]
			sc, below := y.score, math.IsInf(y.score, -1)
			if below {
				sc = 0
			}
			cl = append(cl, Cand{CampaignID: y.c.Campaign.ID, Goal: y.c.Campaign.Goal,
				Value: round4(y.value), Bid: round4(y.shaded), WinProb: round4(y.win), Score: round4(sc),
				BelowFloor: below})
		}
	} else {
		pick, total := weighted(sel)
		prob = p.weight(cands[pick], sel) / total
		x = p.price(cands[pick], placement, floor, rng, predict, sh)
	}
	c, cr := x.c, x.cr
	d := Decision{
		CampaignID: c.Campaign.ID, CreativeID: cr.ID, NEligible: len(cands),
		SelectionProb: prob * x.crProb, Selection: Selections[sel], NCreatives: len(c.Creatives),
		CreativeFrom: x.crFrom, AuctionExplored: explored, Candidates: cl, Capped: capped,
		PClick: x.pr.PClick, PClickFrom: x.pr.PClickFrom, Model: x.pr.Version, ExploreMult: 1,
		PInstallView: x.pr.PInstallView, Conversion: x.pr.Conversion,
	}
	if !x.hasPred {
		d.Model = ""
	}
	for _, e := range cands {
		d.Eligible = append(d.Eligible, e.Campaign.ID)
		d.EligibleGoals = append(d.EligibleGoals, e.Campaign.Goal)
	}
	if c.Campaign.Goal != "CPC" {
		d.PInstallClick, d.InstallRateFrom = x.pInstallClick, x.ifrom
		if x.pr.PInstallView > 0 || x.pr.ValuePerInstall > 0 {
			d.PInstall, d.Target, d.Kappa = round6(x.pInst), c.Campaign.Target, x.k
			if c.Campaign.Goal == "ROAS" {
				d.ValuePerInstall, d.ValueFrom = x.vpi, x.vfrom
			}
		}
	}
	if x.gamma != 1 {
		d.Pacing = round4(x.gamma)
	}
	if x.overbid != 1 {
		d.Overbid = x.overbid
	}
	if sel == SelectAuction && !math.IsInf(x.score, -1) {
		d.Score = round4(x.score)
	}
	d.AdvertiserValue = x.value
	d.MediaValue = x.media
	if rng.Float64() < p.ExploreShare {
		lo, hi := math.Log(p.ExploreLo), math.Log(p.ExploreHi)
		d.ExploreMult = math.Exp(lo + rng.Float64()*(hi-lo))
	}
	if d.MediaValue < floor {
		d.Action, d.Reason = "nobid", "below_floor"
		return d
	}
	d.ShadeFrom, d.WinProb = x.shadeFrom, x.win
	if d.MediaValue > 0 {
		d.ShadeFactor = x.shaded / d.MediaValue
	}
	// exploration multiplies the shaded bid, so the win model keeps seeing
	// bids above and below its own choice
	bid := math.Min(x.shaded*d.ExploreMult, p.CapCPM)
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

func round4(x float64) float64 { return math.Round(x*1e4) / 1e4 }
func round6(x float64) float64 { return math.Round(x*1e6) / 1e6 }
