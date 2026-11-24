// Package bidder is the bid path: parse, normalize, look up the device,
// build features, decide, respond, and log.
//
// In Part II the bidder grows chapter by chapter. In listen mode
// (Chapter 7) it runs the whole path, logs every request, and never bids.
// In bid mode it adds eligibility and selection (Chapter 6), the
// FeatureVector in the bid ledger (Chapter 8), and crawl pricing with a
// prior click rate (Chapter 10), and it receives win, loss, and billing
// notices to track spend and impressions. Each bid's ad markup carries a
// click tracker, so clicks come back to the bidder too, and once a click
// model is deployed (Chapter 9), it replaces the prior.
package bidder

import (
	"encoding/json"
	"fmt"
	"github.com/bgweber/adtechbook/code/dsp/internal/experiment"
	"io"
	"math/rand/v2"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/bgweber/adtechbook/code/dsp/internal/campaigns"
	"github.com/bgweber/adtechbook/code/dsp/internal/clock"
	"github.com/bgweber/adtechbook/code/dsp/internal/features"
	"github.com/bgweber/adtechbook/code/dsp/internal/gbm"
	"github.com/bgweber/adtechbook/code/dsp/internal/latency"
	"github.com/bgweber/adtechbook/code/dsp/internal/ledger"
	"github.com/bgweber/adtechbook/code/dsp/internal/model"
	"github.com/bgweber/adtechbook/code/dsp/internal/normalize"
	"github.com/bgweber/adtechbook/code/dsp/internal/openrtb"
	"github.com/bgweber/adtechbook/code/dsp/internal/ops"
	"github.com/bgweber/adtechbook/code/dsp/internal/policy"
	"github.com/bgweber/adtechbook/code/dsp/internal/profile"
	"github.com/bgweber/adtechbook/code/dsp/internal/stats"
)

// Modes the bidder can run in.
const (
	ModeListen = "listen" // run the full path, log, never bid
	ModeBid    = "bid"    // bid on a share of requests
)

// RawSample is the share of bids and ghost bids whose raw context, the
// request as the exchange sent it and the profile snapshot the features
// were built from, is logged too: for parity checks and backfills
// (Chapter 8).
const RawSample = 0.05

// RawRecord is one row of the raw context log.
type RawRecord struct {
	RequestID string           `json:"request_id"`
	TsMS      int64            `json:"ts_ms"`
	Day       int64            `json:"day"`
	Exchange  string           `json:"exchange"`
	Request   json.RawMessage  `json:"request"` // as received
	Profile   profile.Snapshot `json:"profile"` // before this request
}

// LoadTestHeader marks a request from the load generator (Chapter 17):
// the bidder runs the whole bid path and answers, but writes nothing, so
// a load test leaves no trace in the ledger, the profiles, the budgets,
// or the alerts. Its timings go to their own recorder.
const LoadTestHeader = "X-Load-Test"

// CapAuditSample is the share of capped load test auctions priced again
// in full, to measure what the cap gives up (Chapter 17).
const CapAuditSample = 0.05

// NoBidSample is the share of no-bids whose FeatureVector is logged.
// Every bid and ghost bid carries one; no-bids are sampled (Chapter 8).
const NoBidSample = 0.05

// Bidder handles bid requests and notices on the bid port.
type Bidder struct {
	Mode      string
	Version   string
	BidShare  float64 // share of requests to bid on in bid mode
	NoticeURL string  // base URL exchanges call with notices
	Policy    *policy.Crawl
	Book      *campaigns.Book
	Profiles  *profile.Store
	Bids      *ledger.Writer
	Notices   *ledger.Writer
	Clicks    *ledger.Writer
	Postbacks *ledger.Writer
	Raw       *ledger.Writer // raw context for a sample of bids (Chapter 8)
	MMPToken  string         // the shared secret the MMP sends; empty = accept all
	Stats     *stats.Stats
	Models    *model.Registry // the deployed click model, if any
	// Conversion is the deployed LightGBM bundle (Part III), if any: its
	// click model replaces the FTRL one, and its install models replace
	// the segment install rates.
	Conversion *gbm.Registry
	Exp        *experiment.Manager // shadow, canary, and A/B tests (Chapter 19)
	Win        *gbm.Registry       // win rate models for shading (Chapter 20)

	lastAdForTest map[string]string
	Ops           *ops.Ops // kill switches, throttling, alerts (Chapter 12)

	// Lat times live requests' stages; LoadLat the load test's (Chapter 17).
	Lat, LoadLat *latency.Recorder

	mu      sync.Mutex
	pending map[string]pending // request ID -> our bid, until notices arrive
	seen    map[string]bool    // notice event IDs, for deduplication
	clockMS atomic.Int64       // the newest request time seen
}

type pending struct {
	IFA        string
	CampaignID int
	CreativeID int
	Day        int64
	TsMS       int64
	PClick     float64
	Exchange   string
	Placement  string
	Experiment string // the experiment and arm the bid was in (Chapter 19)
	Arm        int
}

// BidRecord is one row of the bid ledger.
type BidRecord struct {
	TsMS      int64                   `json:"ts_ms"`
	Day       int64                   `json:"day"`
	RequestID string                  `json:"request_id"`
	Mode      string                  `json:"mode"`
	LatencyUS int64                   `json:"latency_us"`
	Policy    string                  `json:"policy_version"`
	Campaigns int                     `json:"campaigns_version"`
	Switches  int                     `json:"switches_version"` // kill switches (Chapter 12)
	Decision  policy.Decision         `json:"decision"`
	Request   normalize.Request       `json:"request"`
	Features  *features.FeatureVector `json:"features,omitempty"`
	Ad        map[string]string       `json:"ad,omitempty"` // the campaign and creative scored
}

// PostbackRecord is one MMP postback, logged as the MMP sent it, in its
// own field names (Chapter 8); training translates it in the batch join.
type PostbackRecord struct {
	PostbackID string            `json:"postback_id"`
	ReceivedMS int64             `json:"received_ts_ms"`
	Day        int64             `json:"day"`
	Fields     map[string]string `json:"fields"`
	Duplicate  bool              `json:"duplicate,omitempty"`
	// SeenMS is the bidder's own clock when the postback arrived: the
	// newest request time it had seen. Device history built from
	// postbacks is as of this time, since that's when bids could use it.
	SeenMS int64 `json:"seen_ts_ms,omitempty"`
}

// ClickRecord is one row of the click log.
type ClickRecord struct {
	EventID    string `json:"event_id"`
	TsMS       int64  `json:"ts_ms"`
	Day        int64  `json:"day"`
	RequestID  string `json:"request_id"`
	CampaignID int    `json:"campaign_id"`
	CreativeID int    `json:"creative_id"`
	Duplicate  bool   `json:"duplicate,omitempty"`
}

// NoticeRecord is one row of the notice log, as the exchange sent it.
type NoticeRecord struct {
	EventID    string  `json:"event_id"`
	Type       string  `json:"type"` // win, loss, billing
	TsMS       int64   `json:"ts_ms"`
	Day        int64   `json:"day"`
	RequestID  string  `json:"request_id"`
	CampaignID int     `json:"campaign_id"`
	CreativeID int     `json:"creative_id"`
	Price      float64 `json:"price,omitempty"`       // CPM paid
	LossReason int     `json:"loss_reason,omitempty"` // OpenRTB loss code
	MinToWin   float64 `json:"min_to_win,omitempty"`  // as reported, if at all
	Duplicate  bool    `json:"duplicate,omitempty"`
}

func (b *Bidder) init() {
	b.mu.Lock()
	if b.pending == nil {
		b.pending = map[string]pending{}
		b.seen = map[string]bool{}
	}
	if b.Lat == nil {
		b.Lat, b.LoadLat = latency.NewRecorder(), latency.NewRecorder()
	}
	b.mu.Unlock()
}

// ServeHTTP routes the bid port: POST /openrtb/{exchange} for bid
// requests, GET /notice/{win,loss,billing} for notices, and GET /click
// for click trackers.
func (b *Bidder) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	b.init()
	if strings.HasPrefix(r.URL.Path, "/notice/") {
		b.notice(w, r)
		return
	}
	if r.URL.Path == "/click" {
		b.click(w, r)
		return
	}
	if r.URL.Path == "/postback" {
		b.postback(w, r)
		return
	}
	b.bid(w, r)
}

func (b *Bidder) bid(w http.ResponseWriter, r *http.Request) {
	tm := latency.Start()
	loadTest := r.Header.Get(LoadTestHeader) == "1"
	ts, _ := clock.Now(r)
	for !loadTest { // a replayed request doesn't move the clock
		if c := b.clockMS.Load(); ts <= c || b.clockMS.CompareAndSwap(c, ts) {
			break
		}
	}
	day := ts / clock.DayMS
	exchange := strings.TrimPrefix(r.URL.Path, "/openrtb/")
	w.Header().Set("X-Bidder-Version", b.Version)
	rng := rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))

	var req openrtb.BidRequest
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err == nil {
		err = json.Unmarshal(body, &req)
	}
	var n normalize.Request
	if err == nil {
		n, err = normalize.Normalize(exchange, &req)
	}
	tm.Lap(latency.Parse)
	d := policy.Decision{Action: "nobid", CampaignID: -1, CreativeID: -1}
	live := b.Exp.Live()
	if loadTest {
		live = nil // no experiment counts from a load test
	}
	expArm := campaigns.ArmNone
	var fv *features.FeatureVector
	var ad map[string]string
	var snap profile.Snapshot
	sw := b.Ops.Switches()
	if !loadTest {
		b.Ops.Step(ts, b.Book) // throttle rates and alerts, once per simulated quarter hour
	}
	var modelsD, shadeD time.Duration // inside the policy, then charged to their stages
	var audit func()                  // the auction cap's audit, after the timings
	switch {
	case err != nil:
		d.Reason = "bad_request"
	case !n.ValidIFA:
		d.Reason = "no_device_id"
	default:
		if loadTest {
			snap = b.Profiles.Peek(n.IFA, day)
		} else {
			snap = b.Profiles.Observe(n.IFA, n.Bundle, ts, day)
		}
		v := features.Build(&n, snap, ts)
		fv = &v
		tm.Lap(latency.Profile)
		switch {
		case b.Mode != ModeBid:
			d.Reason = "listen"
		case sw.Global:
			d.Reason = "kill_global"
		case sw.Exchanges[exchange]:
			d.Reason = "kill_exchange"
		case !loadTest && rng.Float64() >= b.BidShare:
			d.Reason = "not_sampled"
		default:
			// the experiment's arm for this device: the challenger's bundle
			// prices treatment devices, from the arm's part of each budget
			// when the test splits budgets
			arm := live.Arm(n.IFA)
			elig, why := b.Book.EligibleArm(n.Country, n.Placement, int(day), arm)
			cands, reason := b.filter(elig, why, sw, rng)
			tm.Lap(latency.Select)
			if reason != "" {
				d.Reason, d.Filtered = reason, why
				break
			}
			// the internal auction scores every campaign: keep each one's ad,
			// so the ledger logs the ad that bid, not the last one scored
			ads := map[[2]int]map[string]string{}
			predict := func(c *campaigns.Campaign, cr campaigns.Creative) (policy.Prediction, bool) {
				t0 := time.Now()
				defer func() { modelsD += time.Since(t0) }()
				tm.Candidates++
				ad = features.Ad(c, cr)
				ads[[2]int{c.ID, cr.ID}] = ad
				if arm == campaigns.ArmControl && live.Kind == experiment.Value {
					// a value test's control: the champion, with each app's
					// average value for Ad ROAS, even if it has a value model
					pr, ok := b.predict(fv, ad, exchange, sw)
					if strings.ToUpper(ad["goal"]) == "AD_ROAS" {
						pr.ValuePerInstall, pr.ValueFrom = 0, ""
						if cb := b.Conversion.Current(); cb != nil {
							if v, vok := cb.ValuePerInstall(ad["adv_app"]); vok {
								pr.ValuePerInstall, pr.ValueFrom = v, "bundle"
							}
						}
					}
					return pr, ok
				}
				if arm == campaigns.ArmTreatment && live.Bundle != nil {
					if live.Kind == experiment.Value {
						// the champion's predictions, the challenger's value
						pr, ok := b.predict(fv, ad, exchange, sw)
						if v, vok := valueFromModel(live.Bundle, fv, ad); vok {
							pr.ValuePerInstall, pr.ValueFrom = v, "model"
						}
						return pr, ok
					}
					return b.predictWith(fv, ad, exchange, sw, live.Bundle, live.Fault)
				}
				return b.predict(fv, ad, exchange, sw)
			}
			// shading: the win model in a shading test's treatment arm, else
			// the deployed one, else the fixed factor (Chapter 20)
			var wm *gbm.WinModel
			if cur := b.Win.Current(); cur != nil {
				wm, _ = gbm.NewWinModel(cur)
			}
			if live != nil && live.Kind == experiment.Shading && arm == campaigns.ArmTreatment {
				wm = live.Win
			}
			var shade policy.Shader
			if wm != nil {
				var wx []float64 // the win model's inputs: the same for every campaign
				shade = func(media, floor float64) (float64, float64, bool) {
					t0 := time.Now()
					defer func() { shadeD += time.Since(t0) }()
					if wx == nil {
						wx = wm.Inputs(fv, nil)
					}
					bid, p := wm.ShadeInputs(wx, media, floor)
					return bid, p, bid > 0
				}
			}
			// the winner's curse correction (Chapter 31): the deployed win
			// bundle's table, or in a curse test the challenger's in the
			// treatment arm and none in the control
			cm := wm
			if live != nil && live.Kind == experiment.Curse {
				cm = nil
				if arm == campaigns.ArmTreatment {
					cm = live.Win
				}
			}
			var curse policy.Curse
			if cm != nil && cm.Curve() && cm.Curse != nil {
				var cx []float64
				price := -1.0
				curse = func(media float64) float64 {
					if price < 0 {
						if cx == nil {
							cx = cm.Inputs(fv, nil)
						}
						price = cm.MarketPrice(cx)
					}
					if price <= 0 {
						return 1
					}
					return cm.Curse.Kappa(media / price)
				}
			}
			d = b.Policy.DecideWith(cands, n.Placement, n.IFA, n.Floor, rng, predict, shade, curse)
			if loadTest && d.Capped > 0 && rng.Float64() < CapAuditSample {
				audit = func() { b.Policy.AuditCap(cands, n.Placement, n.Floor, rng, predict, shade, curse) }
			}
			if a, ok := ads[[2]int{d.CampaignID, d.CreativeID}]; ok {
				ad = a
			}
			b.lastAdForTest = ad
			d.Filtered = why
			if live != nil {
				d.Experiment = live.ID
				if arm != campaigns.ArmNone {
					d.Arm = [2]string{"control", "treatment"}[arm]
				}
				expArm = arm
				if live.Sampled(n.ID) && d.CampaignID >= 0 && ad != nil {
					t0 := time.Now()
					if pr, ok := b.predictWith(fv, ad, exchange, sw, live.Bundle, ""); ok {
						took := time.Since(t0)
						modelsD += took
						d.Shadow = &policy.ShadowScore{Model: live.Bundle.Spec.Version, PClick: pr.PClick,
							PInstallClick: pr.PInstallClick, PInstallView: pr.PInstallView,
							LatencyUS: float64(took.Nanoseconds()) / 1000}
						b.Exp.ObserveShadow(live, n.Placement, &d, d.Shadow, ts)
					}
				}
				var champ *policy.Prediction
				if arm == campaigns.ArmTreatment && d.CampaignID >= 0 && ad != nil {
					// the champion's prediction for the same request: the
					// canary's guardrails compare the two on the same traffic
					t0 := time.Now()
					if pr, ok := b.predict(fv, ad, exchange, sw); ok {
						champ = &pr
					}
					modelsD += time.Since(t0)
				}
				b.Exp.Observe(live, arm, n.Placement, &d, champ, ts)
			}
			if d.CampaignID >= 0 {
				d.Throttle = b.Ops.Rate(d.CampaignID)
			}
			tm.Lap(latency.Price)
			tm.Add(latency.Models, modelsD, latency.Price)
			tm.Add(latency.Shade, shadeD, latency.Price)
		}
	}
	tm.Lap(latency.Select) // no-bids before selection: bad requests, kill switches

	if d.Action == "bid" {
		if !loadTest {
			b.mu.Lock()
			b.pending[n.ID] = pending{n.IFA, d.CampaignID, d.CreativeID, day, ts, d.PClick,
				exchange, n.Placement, d.Experiment, expArm}
			b.mu.Unlock()
		}
		b.respond(w, &req, &d, ad)
	} else {
		// A no-bid, including a ghost bid, is an empty 204.
		w.WriteHeader(http.StatusNoContent)
	}

	tm.Lap(latency.Respond)
	if loadTest {
		tm.Done()
		b.LoadLat.Record(tm)
		if audit != nil {
			audit()
		}
		return
	}
	lat := time.Since(tm.Start()).Microseconds()
	pol := "crawl-prior-v1"
	switch {
	case d.PClickFrom == "model":
		pol = "crawl-model-v1"
	case strings.HasPrefix(d.PClickFrom, "fallback"):
		pol = "crawl-fallback-v1"
	}
	rec := BidRecord{TsMS: ts, Day: day, RequestID: n.ID, Mode: b.Mode, LatencyUS: lat,
		Policy: pol, Campaigns: b.Book.Version(), Switches: sw.Version, Decision: d, Request: n}
	if fv != nil && (d.Action != "nobid" || rng.Float64() < NoBidSample) {
		rec.Features = fv
	}
	if d.Action != "nobid" {
		rec.Ad = ad
		if b.Raw != nil && fv != nil && rng.Float64() < RawSample && json.Valid(body) {
			b.Raw.Log(day, RawRecord{RequestID: n.ID, TsMS: ts, Day: day, Exchange: exchange,
				Request: body, Profile: snap})
		}
	}
	if rec.RequestID == "" {
		rec.RequestID = req.ID
	}
	b.Bids.Log(day, rec)
	pInst := 0.0
	if d.InstallRateFrom == "model" {
		pInst = d.PInstallClick
	}
	b.Ops.Observe(ops.BidObs{TsMS: ts, Exchange: exchange, Action: d.Action, PClick: d.PClick,
		PInstall: pInst, PFrom: d.PClickFrom, LatencyUS: lat, ModelOn: b.Models.Current() != nil || b.Conversion.Current() != nil, Listen: b.Mode != ModeBid})
	version := "priors"
	switch {
	case d.PClickFrom == "model":
		version = d.Model
	case strings.HasPrefix(d.PClickFrom, "fallback"):
		version = "fallback (" + d.Model + ")"
	}
	b.Stats.Observe(stats.Observation{TsMS: ts, Exchange: exchange, Placement: n.Placement,
		Country: n.Country, Floor: n.Floor, ValidIFA: n.ValidIFA, Action: d.Action,
		Reason: d.Reason, LatencyUS: lat, CampaignID: d.CampaignID, Bid: d.Bid,
		Filtered: d.Filtered, PricedBy: version})
	tm.Lap(latency.Log)
	tm.Done()
	b.Lat.Record(tm)
}

// filter applies the campaign kill switches and even-spend throttling to
// the eligible campaigns (Chapter 12). The reason says why none are left.
// The reasons go into why, next to the book's own filters.
func (b *Bidder) filter(cands []campaigns.Candidate, why map[int]string, sw *ops.Switches,
	rng *rand.Rand) ([]campaigns.Candidate, string) {
	if len(cands) == 0 {
		return nil, "no_eligible_campaign"
	}
	out := cands[:0:0]
	paused := 0
	for _, c := range cands {
		if sw.Campaigns[c.Campaign.ID] {
			why[c.Campaign.ID] = "paused"
			paused++
			continue
		}
		if rng.Float64() >= b.Ops.Rate(c.Campaign.ID) {
			why[c.Campaign.ID] = "throttled"
			continue
		}
		out = append(out, c)
	}
	switch {
	case len(out) > 0:
		return out, ""
	case paused == len(cands):
		return nil, "campaign_paused"
	default:
		return nil, "throttled"
	}
}

// predict walks the fallback ladder (Chapter 12): the deployed model; or,
// when its kill switch is on or its prediction is outside sane bounds,
// the bundle's segment click rates; or, with no bundle at all, nothing,
// and the policy uses its placement priors. Features whose kill switch is
// on are treated as missing.
func (b *Bidder) predict(fv *features.FeatureVector, ad map[string]string, exchange string,
	sw *ops.Switches) (policy.Prediction, bool) {
	return b.predictWith(fv, ad, exchange, sw, b.Conversion.Current(), "")
}

// MaxValuePerInstall bounds a value model's prediction: past it, the
// bidder falls back to the app's average.
const MaxValuePerInstall = 100.0

// valueFromModel is the bundle's value model's D3 ad revenue per install
// for an Ad ROAS ad (Chapter 30), if it has one and the prediction is sane.
func valueFromModel(cb *gbm.Bundle, fv *features.FeatureVector, ad map[string]string) (float64, bool) {
	if cb == nil || strings.ToUpper(ad["goal"]) != "AD_ROAS" {
		return 0, false
	}
	if _, ok := cb.Models["value"]; !ok {
		return 0, false
	}
	v, _ := cb.Predict("value", cb.Inputs(fv, ad))
	if v <= 0 || v > MaxValuePerInstall || v != v {
		return 0, false
	}
	return v, true
}

// predictWith predicts with a given conversion bundle: the champion's, or
// an experiment's challenger. fault "scale" triples the bundle's click and
// install predictions, the broken model of the Chapter 19 exercise.
func (b *Bidder) predictWith(fv *features.FeatureVector, ad map[string]string, exchange string,
	sw *ops.Switches, cb *gbm.Bundle, fault string) (policy.Prediction, bool) {
	m := b.Models.Current()
	if m == nil && cb == nil {
		return policy.Prediction{}, false
	}
	place, country := fv.Strings["placement"], fv.Strings["country"]
	pr := policy.Prediction{}
	if m != nil {
		pr.Version = m.Version
		if m.Rates != nil {
			pr.PInstallClick, pr.InstallRateFrom = m.Rates.Lookup(ad["campaign"], country, place)
		}
	}
	fallback := func(why string) (policy.Prediction, bool) {
		if m == nil || m.ClickRates == nil {
			pr.PClick, pr.PClickFrom = policy.CTRPrior[place], "fallback_prior"
		} else {
			pr.PClick, pr.PClickFrom = m.ClickRates.Lookup(place, country, exchange), "fallback_"+why
		}
		return pr, true
	}
	if sw.Model {
		return fallback("segment")
	}
	in, adIn := fv, ad
	if len(sw.Features) > 0 {
		in = withoutFeatures(fv, sw.Features)
		adIn = map[string]string{}
		for k, v := range ad {
			if !sw.Features[k] {
				adIn[k] = v
			}
		}
	}
	var p float64
	clickDone := false
	if cb != nil {
		pr.Conversion = cb.Spec.Version
		x := cb.Inputs(in, adIn)
		k := 1.0
		if fault == "scale" {
			k = 3
		}
		if v, ok := cb.Predict("install_click", x); ok && v > 0 && v*k < 1 {
			v *= k
			pr.PInstallClick, pr.InstallRateFrom = v, "model"
		}
		if v, ok := cb.Predict("install_view", x); ok && v > 0 && v < 1 {
			pr.PInstallView = v
		}
		if g := strings.ToUpper(ad["goal"]); g == "ROAS" || g == "AD_ROAS" {
			if v, ok := cb.ValuePerInstall(ad["adv_app"]); ok {
				pr.ValuePerInstall, pr.ValueFrom = v, "bundle"
			}
			if v, ok := valueFromModel(cb, in, adIn); ok {
				pr.ValuePerInstall, pr.ValueFrom = v, "model"
			}
		}
		if v, ok := cb.Predict("click", x); ok {
			// the shift drill (Chapter 24): 1 unless it's running on this bundle
			p, clickDone, pr.Version = v*k*b.Ops.Shift(cb.Spec.Version), true, cb.Spec.Version
		}
	}
	if !clickDone {
		if m == nil {
			pr.PClick, pr.PClickFrom = policy.CTRPrior[place], "prior"
			return pr, true
		}
		p = m.Predict(in, adIn)
	}
	if p < MinPClick || p > MaxPClick || p != p {
		return fallback("bounds")
	}
	pr.PClick, pr.PClickFrom = p, "model"
	return pr, true
}

// Sane bounds on a click prediction; outside them, the fallback prices.
const (
	MinPClick = 1e-4
	MaxPClick = 0.5
)

// withoutFeatures copies a FeatureVector without the named features.
func withoutFeatures(fv *features.FeatureVector, off map[string]bool) *features.FeatureVector {
	c := *fv
	c.Strings, c.Numeric, c.Sets = map[string]string{}, map[string]float64{}, map[string][]string{}
	for k, v := range fv.Strings {
		if !off[k] {
			c.Strings[k] = v
		}
	}
	for k, v := range fv.Numeric {
		if !off[k] {
			c.Numeric[k] = v
		}
	}
	for k, v := range fv.Sets {
		if !off[k] {
			c.Sets[k] = v
		}
	}
	return &c
}

// respond writes a bid. The notice URLs carry the IDs the notice handler
// needs, and the OpenRTB macros the exchange fills in. The ad markup is a
// small native-style object whose click tracker brings clicks back here.
func (b *Bidder) respond(w http.ResponseWriter, req *openrtb.BidRequest, d *policy.Decision,
	ad map[string]string) {
	q := fmt.Sprintf("rid=%s&cid=%d&crid=%d", url.QueryEscape(req.ID), d.CampaignID, d.CreativeID)
	adm, _ := json.Marshal(map[string]any{"native": map[string]any{"link": map[string]any{
		"url":           "https://apps.example/app/" + ad["adv_app"],
		"clicktrackers": []string{b.NoticeURL + "/click?" + q},
	}}})
	resp := openrtb.BidResponse{ID: req.ID, Cur: "USD", SeatBid: []openrtb.SeatBid{{Bid: []openrtb.Bid{{
		ID: req.ID + "-1", ImpID: req.Imp[0].ID, Price: d.Bid,
		CID: strconv.Itoa(d.CampaignID), CrID: strconv.Itoa(d.CreativeID), AdM: string(adm),
		NURL: b.NoticeURL + "/notice/win?" + q + "&price=${AUCTION_PRICE}&mtw=${AUCTION_MIN_TO_WIN}",
		BURL: b.NoticeURL + "/notice/billing?" + q + "&price=${AUCTION_PRICE}",
		LURL: b.NoticeURL + "/notice/loss?" + q + "&reason=${AUCTION_LOSS}&mtw=${AUCTION_MIN_TO_WIN}",
	}}}}}
	body, _ := json.Marshal(resp)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(body)
}

// notice handles a win, loss, or billing notice from an exchange.
func (b *Bidder) notice(w http.ResponseWriter, r *http.Request) {
	ts, _ := clock.Now(r)
	kind := strings.TrimPrefix(r.URL.Path, "/notice/")
	q := r.URL.Query()
	rid := q.Get("rid")
	if rid == "" || (kind != "win" && kind != "loss" && kind != "billing") {
		http.Error(w, "bad notice", http.StatusBadRequest)
		return
	}
	cid, _ := strconv.Atoi(q.Get("cid"))
	crid, _ := strconv.Atoi(q.Get("crid"))
	num := func(k string) float64 {
		v, err := strconv.ParseFloat(q.Get(k), 64)
		if err != nil {
			return 0
		}
		return v
	}
	rec := NoticeRecord{EventID: kind[:1] + rid, Type: kind, TsMS: ts, Day: ts / clock.DayMS,
		RequestID: rid, CampaignID: cid, CreativeID: crid, Price: num("price"),
		LossReason: int(num("reason")), MinToWin: num("mtw")}

	b.mu.Lock()
	rec.Duplicate = b.seen[rec.EventID]
	b.seen[rec.EventID] = true
	p, ok := b.pending[rid]
	if kind != "win" && ok {
		delete(b.pending, rid) // billing or loss is the last notice
	}
	b.pruneLocked(ts)
	b.mu.Unlock()

	if !rec.Duplicate && kind == "billing" {
		day := rec.Day
		if ok {
			day = p.Day // spend counts against the day we bid
			b.Profiles.RecordImpression(p.IFA, ts, ts/clock.DayMS)
			b.Stats.ObservePrediction(cid, p.PClick)
		}
		arm := campaigns.ArmNone
		if ok {
			arm = p.Arm
			b.Exp.ObserveSpend(p.Experiment, p.Arm, rec.Price/1000)
		}
		b.Book.AddSpendArm(int(day), cid, arm, rec.Price/1000)
		b.Policy.Creatives.Impression(rid, crid, cid, day) // Thompson sampling (Chapter 21)
		ex, pl := "unknown", "unknown"
		if ok {
			ex, pl = p.Exchange, p.Placement
		}
		b.Ops.ObserveBilling(ts, day, ex, pl, cid, rec.Price/1000)
	}
	b.Notices.Log(rec.Day, rec)
	b.Stats.ObserveNotice(kind, cid, rec.Price, rec.Duplicate)
	w.WriteHeader(http.StatusNoContent)
}

// click handles a click tracker call. The first click per impression
// counts; repeats, from SDK retries or double taps, are logged as
// duplicates (Chapter 9 keeps the first click per impression).
func (b *Bidder) click(w http.ResponseWriter, r *http.Request) {
	ts, _ := clock.Now(r)
	q := r.URL.Query()
	rid := q.Get("rid")
	if rid == "" {
		http.Error(w, "bad click", http.StatusBadRequest)
		return
	}
	cid, _ := strconv.Atoi(q.Get("cid"))
	crid, _ := strconv.Atoi(q.Get("crid"))
	rec := ClickRecord{EventID: "c" + rid, TsMS: ts, Day: ts / clock.DayMS, RequestID: rid,
		CampaignID: cid, CreativeID: crid}
	b.mu.Lock()
	rec.Duplicate = b.seen[rec.EventID]
	b.seen[rec.EventID] = true
	b.mu.Unlock()
	if b.Clicks != nil {
		b.Clicks.Log(rec.Day, rec)
	}
	b.Stats.ObserveClick(cid, rec.Duplicate)
	w.WriteHeader(http.StatusNoContent)
}

// postback receives an install or in-app event postback from the
// advertiser's MMP: the minimal endpoint of Chapter 8. It checks the
// shared token, logs the postback with the fields it carries, and counts
// installs. Joining postbacks to clicks happens later, in batch.
func (b *Bidder) postback(w http.ResponseWriter, r *http.Request) {
	if b.MMPToken != "" && r.Header.Get("X-MMP-Token") != b.MMPToken {
		http.Error(w, "unknown sender", http.StatusForbidden)
		return
	}
	ts, _ := clock.Now(r)
	fields := map[string]string{}
	for k, v := range r.URL.Query() {
		if len(v) > 0 {
			fields[k] = v[0]
		}
	}
	id := fields["postback_id"]
	if id == "" {
		http.Error(w, "no postback_id", http.StatusBadRequest)
		return
	}
	rec := PostbackRecord{PostbackID: id, ReceivedMS: ts, Day: ts / clock.DayMS, Fields: fields,
		SeenMS: max(ts, b.clockMS.Load())}
	b.mu.Lock()
	rec.Duplicate = b.seen["p"+id]
	b.seen["p"+id] = true
	b.mu.Unlock()
	if b.Postbacks != nil {
		b.Postbacks.Log(rec.Day, rec)
	}
	cid, err := strconv.Atoi(fields["campaign_id"])
	if err != nil {
		cid = -1
	}
	b.Stats.ObservePostback(fields["event_name"], fields["touch_type"], cid, rec.Duplicate)
	if !rec.Duplicate && b.Profiles != nil {
		b.Profiles.RecordConversion(strings.ToLower(fields["advertising_id"]), fields["event_name"])
	}
	if !rec.Duplicate && fields["event_name"] == "install" {
		// credit the creative that showed: the MMP echoes our click or
		// impression ID, which is "c" or "i" and the request ID
		for _, k := range []string{"click_id", "impression_id"} {
			if v := fields[k]; len(v) > 1 {
				b.Policy.Creatives.Install(v[1:])
				break
			}
		}
	}
	w.WriteHeader(http.StatusNoContent)
}

// pruneLocked forgets bids and notice IDs older than a simulated day, so
// memory stays bounded. Callers hold b.mu.
func (b *Bidder) pruneLocked(now int64) {
	if len(b.pending) < 200_000 && len(b.seen) < 400_000 {
		return
	}
	for rid, p := range b.pending {
		if now-p.TsMS > clock.DayMS {
			delete(b.pending, rid)
		}
	}
	if len(b.seen) >= 400_000 {
		b.seen = map[string]bool{}
	}
}
