package ops

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"

	"github.com/bgweber/adtechbook/code/dsp/internal/campaigns"
	"github.com/bgweber/adtechbook/code/dsp/internal/clock"
)

// QuarterMS is the step of the ops loop: throttle rates and alerts are
// updated every 15 simulated minutes, and counters kept in 15-minute
// buckets.
const QuarterMS = 15 * 60_000

const (
	hourMS     = 3_600_000
	keepQuarts = 3 * 96 // three simulated days
	predBins   = 64     // log-scale prediction histogram, 1e-5 to 1
)

// Ops holds the switches, the sim-time counters, throttle rates,
// alerts, and reconciliation.
type Ops struct {
	sw switchboard

	// Throttle turns even-spend throttling on (Chapter 12).
	Throttle bool

	mu       sync.Mutex
	quarters map[int64]*quarter
	lastStep int64 // last quarter index the loop ran for
	nowMS    int64
	rates    map[int]float64 // campaign -> throttle rate, today
	rateDay  int64
	alerts   map[string]*Alert
	history  []Alert                 // earlier episodes of alerts that fired again
	billed   map[reconKey]*reconCell // our billed spend
	invoiced map[reconKey]*reconCell // the exchanges' invoices
	modelOn  bool                    // a model is deployed, for the fallback alert
	listen   bool
}

// New returns an Ops that keeps its switches and audit log in dir.
func New(dir string) *Ops {
	o := &Ops{Throttle: true, quarters: map[int64]*quarter{}, rates: map[int]float64{},
		alerts: map[string]*Alert{}, billed: map[reconKey]*reconCell{}, invoiced: map[reconKey]*reconCell{},
		lastStep: -1, rateDay: -1}
	o.sw.dir = dir
	o.sw.load()
	o.loadRates()
	return o
}

// Throttle rates are saved, so a restarted bidder doesn't start every
// campaign at full speed.
func (o *Ops) loadRates() {
	raw, err := os.ReadFile(filepath.Join(o.sw.dir, "throttle.json"))
	if err != nil {
		return
	}
	var st struct {
		Day   int64           `json:"day"`
		Rates map[int]float64 `json:"rates"`
	}
	if json.Unmarshal(raw, &st) == nil && st.Rates != nil {
		o.rates, o.rateDay = st.Rates, st.Day
	}
}

func (o *Ops) saveRatesLocked() {
	if o.sw.dir == "" {
		return
	}
	os.MkdirAll(o.sw.dir, 0o755)
	raw, _ := json.Marshal(map[string]any{"day": o.rateDay, "rates": o.rates})
	tmp := filepath.Join(o.sw.dir, "throttle.json.tmp")
	if os.WriteFile(tmp, raw, 0o644) == nil {
		os.Rename(tmp, filepath.Join(o.sw.dir, "throttle.json"))
	}
}

type quarter struct {
	requests map[string]int64 // by exchange
	bids     map[string]int64
	priced   int64 // bids priced while a model was deployed
	fallback int64 // of those, priced by the fallback
	pred     [predBins]int64
	spend    map[int]float64 // by campaign, from billing notices
	latency  [32]int64       // log2 microseconds
	latencyN int64
}

func (o *Ops) q(ts int64) *quarter {
	k := ts / QuarterMS
	q := o.quarters[k]
	if q == nil {
		q = &quarter{requests: map[string]int64{}, bids: map[string]int64{}, spend: map[int]float64{}}
		o.quarters[k] = q
		for old := range o.quarters {
			if old < k-keepQuarts {
				delete(o.quarters, old)
			}
		}
	}
	return q
}

// BidObs is what the bid path reports to ops for each request.
type BidObs struct {
	TsMS      int64
	Exchange  string
	Action    string // bid, ghost, nobid
	PClick    float64
	PFrom     string // model, fallback_segment, fallback_bounds, prior, ...
	LatencyUS int64
	ModelOn   bool // a model is deployed
	Listen    bool
}

// Observe counts one request.
func (o *Ops) Observe(b BidObs) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if b.TsMS > o.nowMS {
		o.nowMS = b.TsMS
	}
	o.modelOn, o.listen = b.ModelOn, b.Listen
	q := o.q(b.TsMS)
	q.requests[b.Exchange]++
	q.latency[min(31, int(math.Log2(float64(max(b.LatencyUS, 1)))))]++
	q.latencyN++
	if b.Action != "bid" && b.Action != "ghost" {
		return
	}
	q.bids[b.Exchange]++
	if b.ModelOn {
		q.priced++
		if len(b.PFrom) > 8 && b.PFrom[:8] == "fallback" {
			q.fallback++
		} else if b.PFrom == "model" {
			q.pred[predBin(b.PClick)]++
		}
	}
}

// ObserveBilling counts a billed impression, for spend-rate alerts and
// reconciliation.
func (o *Ops) ObserveBilling(tsMS, bidDay int64, exchange, placement string, campaignID int, dollars float64) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.q(tsMS).spend[campaignID] += dollars
	k := reconKey{Day: bidDay, Exchange: exchange, Placement: placement}
	c := o.billed[k]
	if c == nil {
		c = &reconCell{}
		o.billed[k] = c
	}
	c.Imps++
	c.Spend += dollars
}

func predBin(p float64) int {
	if p <= 1e-5 {
		return 0
	}
	b := int((math.Log10(p) + 5) / 5 * predBins)
	return min(max(b, 0), predBins-1)
}

func binMid(b int) float64 { return math.Pow(10, (float64(b)+0.5)/predBins*5-5) }

// Step runs the ops loop once per simulated quarter hour: new throttle
// rates and fresh alerts. The bid path calls it on every request; all
// but the first call in a quarter return at once.
func (o *Ops) Step(tsMS int64, book *campaigns.Book) {
	k := tsMS / QuarterMS
	o.mu.Lock()
	if k <= o.lastStep {
		o.mu.Unlock()
		return
	}
	o.lastStep = k
	o.updateThrottleLocked(tsMS, book)
	o.evaluateLocked(tsMS, book)
	o.mu.Unlock()
}

// Rate is the campaign's throttle rate: the chance of bidding on a
// request it's eligible for (1 when throttling is off).
func (o *Ops) Rate(campaignID int) float64 {
	if !o.Throttle {
		return 1
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if r, ok := o.rates[campaignID]; ok {
		return r
	}
	return 1
}

// plannedShare is the share of a day's spend planned by time ts: an even
// share of the day, shaped like the requests we saw in each hour of the
// day before, or flat on the first day (Chapter 12).
func (o *Ops) plannedShareLocked(ts int64) float64 {
	day := ts / clock.DayMS
	var w [24]float64
	total := 0.0
	for k, q := range o.quarters {
		if k*QuarterMS/clock.DayMS != day-1 {
			continue
		}
		h := (k * QuarterMS % clock.DayMS) / hourMS
		for _, n := range q.requests {
			w[h] += float64(n)
		}
	}
	for _, v := range w {
		total += v
	}
	if total == 0 {
		for i := range w {
			w[i] = 1
		}
		total = 24
	}
	into := float64(ts%clock.DayMS) / hourMS
	share := 0.0
	for h := 0; h < 24; h++ {
		f := math.Min(math.Max(into-float64(h), 0), 1)
		share += w[h] / total * f
	}
	return share
}

// updateThrottleLocked compares each campaign's spend today with its plan
// and moves its rate: down when ahead of plan, up when behind, never
// below 2% and never above 100%.
func (o *Ops) updateThrottleLocked(ts int64, book *campaigns.Book) {
	day := ts / clock.DayMS
	if day != o.rateDay {
		// Rates carry over from the day before. Resetting them all to 100%
		// at midnight would let every campaign start the day at full
		// speed at once: the budget-reset spike of Chapter 12.
		o.rateDay = day
		return
	}
	plan := o.plannedShareLocked(ts)
	if plan < 0.02 {
		return
	}
	for _, v := range book.Views(int(day)) {
		if v.DailyBudget <= 0 {
			continue
		}
		r, ok := o.rates[v.ID]
		if !ok {
			r = 1
		}
		ratio := v.SpendToday / (v.DailyBudget * plan)
		switch {
		case ratio > 1.1:
			r *= 0.7
		case ratio < 0.9:
			r /= 0.7
		}
		o.rates[v.ID] = math.Min(1, math.Max(0.02, r))
	}
	o.saveRatesLocked()
}

// RatesView is the throttle rates for the console.
func (o *Ops) RatesView() (map[int]float64, float64) {
	o.mu.Lock()
	defer o.mu.Unlock()
	out := map[int]float64{}
	for k, v := range o.rates {
		out[k] = v
	}
	return out, o.plannedShareLocked(o.nowMS)
}

// window sums the last n quarters before ts (the current quarter is
// still filling, so it's left out).
func (o *Ops) windowLocked(ts int64, n int) []*quarter {
	var out []*quarter
	k := ts / QuarterMS
	for i := k - int64(n); i < k; i++ {
		if q := o.quarters[i]; q != nil {
			out = append(out, q)
		}
	}
	return out
}

func quantile(h [predBins]int64, q float64) (float64, int64) {
	var n int64
	for _, c := range h {
		n += c
	}
	if n == 0 {
		return 0, 0
	}
	target, seen := int64(math.Ceil(q*float64(n))), int64(0)
	for i, c := range h {
		seen += c
		if seen >= target {
			return binMid(i), n
		}
	}
	return binMid(predBins - 1), n
}

func fmtPct(x float64) string { return fmt.Sprintf("%.2f%%", 100*x) }
