// Package latency times the stages of the bid path (Chapter 17).
//
// An exchange gives a bidder a fixed time to answer, often 100 to 300
// milliseconds including the network, and a bid that arrives late is a
// no-bid. The total is what the exchange sees; the stages say where the
// time goes, so an optimization starts from a measurement rather than a
// guess. A Timer follows one request; a Recorder keeps the most recent
// requests' timings and reports percentiles, since the tail, not the mean,
// is what times out.
package latency

import (
	"math"
	"sort"
	"sync"
	"time"
)

// Stage is one step of the bid path.
type Stage int

// The stages, in the order a request passes through them. Models and
// Shade are timed inside the policy's callbacks, and Price is the
// policy's own time without them.
const (
	Parse   Stage = iota // read and decode the body, normalize
	Profile              // look up the device, build the FeatureVector
	Select               // eligibility, kill switches, throttling
	Models               // conversion model predictions, every candidate
	Shade                // the win model's bid search, every candidate
	Price                // value, multipliers, margin, the auction
	Respond              // write the bid response
	Log                  // ledger, stats, and alerts
	Total                // the whole request
	NStages
)

// Names are the stages' names in the console and the load test's report.
var Names = [NStages]string{"parse", "profile", "select", "models", "shade",
	"price", "respond", "log", "total"}

// Timer times one request.
type Timer struct {
	start, last time.Time
	D           [NStages]time.Duration
	Candidates  int // campaigns scored: what the models and shading scale with
}

// Start starts timing a request.
func Start() *Timer {
	now := time.Now()
	return &Timer{start: now, last: now}
}

// Start is when the request started.
func (t *Timer) Start() time.Time { return t.start }

// Lap charges the time since the last lap to a stage.
func (t *Timer) Lap(s Stage) {
	now := time.Now()
	t.D[s] += now.Sub(t.last)
	t.last = now
}

// Add charges a duration measured elsewhere to a stage, and takes it out
// of another (the stage whose lap included it), so the stages still add
// up to the total.
func (t *Timer) Add(s Stage, d time.Duration, from Stage) {
	t.D[s] += d
	t.D[from] -= d
}

// Done sets the total.
func (t *Timer) Done() { t.D[Total] = time.Since(t.start) }

// Size is how many requests a Recorder keeps: enough for a stable 99th
// percentile, few enough to sort on every console refresh.
const Size = 8192

// Recorder keeps the latest requests' stage times.
type Recorder struct {
	mu    sync.Mutex
	us    [NStages][]float64 // a ring of microseconds per stage
	cands []float64
	next  int
	n     int64
	since time.Time
}

// NewRecorder returns an empty Recorder.
func NewRecorder() *Recorder {
	r := &Recorder{}
	r.Reset()
	return r
}

// Reset forgets every request; the load test resets before each rate.
func (r *Recorder) Reset() {
	r.mu.Lock()
	defer r.mu.Unlock()
	for s := range r.us {
		r.us[s] = make([]float64, 0, Size)
	}
	r.cands = make([]float64, 0, Size)
	r.next, r.n, r.since = 0, 0, time.Now()
}

// Record adds a finished request.
func (r *Recorder) Record(t *Timer) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	put := func(xs []float64, v float64) []float64 {
		if len(xs) < Size {
			return append(xs, v)
		}
		xs[r.next] = v
		return xs
	}
	for s := range r.us {
		r.us[s] = put(r.us[s], float64(t.D[s].Nanoseconds())/1000)
	}
	r.cands = put(r.cands, float64(t.Candidates))
	r.next = (r.next + 1) % Size
	r.n++
}

// StageSummary is one stage's times over the kept requests, in
// microseconds. Share is the stage's part of the mean total.
type StageSummary struct {
	Stage string  `json:"stage"`
	Mean  float64 `json:"mean_us"`
	P50   float64 `json:"p50_us"`
	P90   float64 `json:"p90_us"`
	P99   float64 `json:"p99_us"`
	Max   float64 `json:"max_us"`
	Share float64 `json:"share"`
}

// Summary is what the console and the load test read.
type Summary struct {
	Requests   int64          `json:"requests"` // since the last reset
	Kept       int            `json:"kept"`     // in the percentiles
	Seconds    float64        `json:"seconds"`  // since the last reset
	Candidates float64        `json:"mean_candidates"`
	Stages     []StageSummary `json:"stages"`
}

// Summary computes percentiles over the kept requests.
func (r *Recorder) Summary() Summary {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := Summary{Requests: r.n, Kept: len(r.us[Total]), Seconds: time.Since(r.since).Seconds()}
	if len(r.cands) > 0 {
		out.Candidates = mean(r.cands)
	}
	total := mean(r.us[Total])
	for s := Stage(0); s < NStages; s++ {
		xs := append([]float64(nil), r.us[s]...)
		sort.Float64s(xs)
		ss := StageSummary{Stage: Names[s], Mean: round1(mean(xs)), P50: round1(Quantile(xs, 0.5)),
			P90: round1(Quantile(xs, 0.9)), P99: round1(Quantile(xs, 0.99)), Max: round1(Quantile(xs, 1))}
		if total > 0 {
			ss.Share = math.Round(mean(xs)/total*1000) / 1000
		}
		out.Stages = append(out.Stages, ss)
	}
	return out
}

// Quantile is the q-th quantile of sorted values, by the nearest rank.
func Quantile(sorted []float64, q float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	i := int(math.Ceil(q*float64(len(sorted)))) - 1
	return sorted[max(0, min(i, len(sorted)-1))]
}

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func round1(x float64) float64 { return math.Round(x*10) / 10 }
