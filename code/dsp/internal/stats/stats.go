// Package stats keeps the live numbers the console shows: traffic by
// exchange, placement, and country, floors, decisions, and latency.
//
// These are the views Chapter 7 asks for while the DSP is only
// listening: how much traffic there is, where it comes from, what the
// floors look like, and whether the bidder answers in time.
package stats

import (
	"math"
	"sort"
	"sync"
	"time"

	"github.com/bgweber/adtechbook/code/dsp/internal/clock"
)

// FloorEdges are the upper edges of the floor histogram buckets, in CPM.
var FloorEdges = []float64{0, 0.5, 1, 2, 4, 8, 16, math.Inf(1)}

// Observation is what the bid path reports for each request.
type Observation struct {
	TsMS       int64
	Exchange   string
	Placement  string
	Country    string
	Floor      float64
	ValidIFA   bool
	Action     string
	Reason     string
	LatencyUS  int64
	CampaignID int            // -1 when no campaign was chosen
	Bid        float64        // CPM, for bids and ghost bids
	Filtered   map[int]string // campaign -> the filter that removed it
	PricedBy   string         // model version, "fallback (version)", or "priors"
}

// CampaignStats are one campaign's counts since the bidder started.
type CampaignStats struct {
	Bids   int64 `json:"bids"`
	Ghosts int64 `json:"ghosts"`
	Wins   int64 `json:"wins"`
	Losses int64 `json:"losses"`
	Imps   int64 `json:"imps"`
	// Filtered counts the requests where a filter removed this campaign,
	// by filter (Chapter 6).
	Filtered map[string]int64 `json:"filtered,omitempty"`
	Spend    float64          `json:"spend"`
	Clicks   int64            `json:"clicks"`
	// PredClicks is the sum of the predicted click rates of the shown
	// impressions: what the prior or the model expected. Compared with
	// Clicks, it's the live calibration check.
	PredClicks float64 `json:"pred_clicks"`
	// Installs credited to us by the MMP, after a click or a view.
	Installs     int64   `json:"installs"`
	ViewInstalls int64   `json:"view_installs"`
	BidSum       float64 `json:"-"`
}

type group struct {
	Requests  int64   `json:"requests"`
	FloorSum  float64 `json:"-"`
	ValidIFAs int64   `json:"-"`
	Floors    []int64 `json:"floor_hist"`
}

type Stats struct {
	mu           sync.Mutex
	start        time.Time
	total        int64
	actions      map[string]int64
	reasons      map[string]int64
	exchange     map[string]*group
	placement    map[string]*group
	country      map[string]int64
	day          int64
	hourly       [24]int64 // requests per simulated hour of the latest day
	latency      []int64   // log-scale histogram, see latencyBucket
	camp         map[int]*CampaignStats
	notices      map[string]int64
	dups         int64
	dupClicks    int64
	dupPostbacks int64
	simNow       int64
	wallSec      map[int64]int64            // requests per wall-clock second, last minute
	versions     map[int64]map[string]int64 // simulated hour -> bids by model version
}

func New() *Stats {
	return &Stats{
		start: time.Now(), actions: map[string]int64{}, reasons: map[string]int64{},
		exchange: map[string]*group{}, placement: map[string]*group{},
		country: map[string]int64{}, latency: make([]int64, latencyBuckets),
		camp: map[int]*CampaignStats{}, notices: map[string]int64{},
		wallSec: map[int64]int64{}, day: -1, versions: map[int64]map[string]int64{},
	}
}

func floorBucket(f float64) int {
	for i, e := range FloorEdges {
		if f <= e {
			return i
		}
	}
	return len(FloorEdges) - 1
}

func (s *Stats) Observe(o Observation) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.total++
	s.actions[o.Action]++
	if o.Reason != "" {
		s.reasons[o.Reason]++
	}
	for _, g := range []*group{s.groupFor(s.exchange, o.Exchange), s.groupFor(s.placement, o.Placement)} {
		g.Requests++
		g.FloorSum += o.Floor
		if o.ValidIFA {
			g.ValidIFAs++
		}
		g.Floors[floorBucket(o.Floor)]++
	}
	s.country[o.Country]++
	if d := o.TsMS / clock.DayMS; d > s.day {
		s.day, s.hourly = d, [24]int64{}
	}
	if o.TsMS/clock.DayMS == s.day {
		s.hourly[(o.TsMS%clock.DayMS)/(clock.DayMS/24)]++
	}
	if o.TsMS > s.simNow {
		s.simNow = o.TsMS
	}
	s.latency[latencyBucket(o.LatencyUS)]++
	for id, why := range o.Filtered {
		c := s.campaign(id)
		if c.Filtered == nil {
			c.Filtered = map[string]int64{}
		}
		c.Filtered[why]++
	}
	if o.Action == "bid" && o.PricedBy != "" {
		h := o.TsMS / (clock.DayMS / 24)
		m := s.versions[h]
		if m == nil {
			m = map[string]int64{}
			s.versions[h] = m
			for k := range s.versions {
				if k < h-48 {
					delete(s.versions, k)
				}
			}
		}
		m[o.PricedBy]++
	}
	if o.CampaignID >= 0 && (o.Action == "bid" || o.Action == "ghost") {
		c := s.campaign(o.CampaignID)
		if o.Action == "bid" {
			c.Bids++
			c.BidSum += o.Bid
		} else {
			c.Ghosts++
		}
	}
	now := time.Now().Unix()
	s.wallSec[now]++
	if len(s.wallSec) > 120 {
		for k := range s.wallSec {
			if k < now-60 {
				delete(s.wallSec, k)
			}
		}
	}
}

// Latency buckets are log-scale, with 8 buckets per doubling, so a
// quantile is accurate to about 5%.
const (
	perDoubling    = 8
	latencyBuckets = 30 * perDoubling // up to about 2^30 us, 18 minutes
)

func latencyBucket(us int64) int {
	if us < 1 {
		return 0
	}
	b := int(math.Log2(float64(us)) * perDoubling)
	return min(max(b, 0), latencyBuckets-1)
}

// bucketMid is the geometric middle of a bucket, in microseconds.
func bucketMid(b int) int64 {
	return int64(math.Round(math.Pow(2, (float64(b)+0.5)/perDoubling)))
}

func (s *Stats) campaign(id int) *CampaignStats {
	c, ok := s.camp[id]
	if !ok {
		c = &CampaignStats{}
		s.camp[id] = c
	}
	return c
}

// ObserveNotice counts a win, loss, or billing notice. Duplicates, from
// exchange retries, are counted but don't change wins or spend.
func (s *Stats) ObserveNotice(kind string, campaignID int, price float64, dup bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if dup {
		s.dups++
		return
	}
	s.notices[kind]++
	if campaignID < 0 {
		return
	}
	c := s.campaign(campaignID)
	switch kind {
	case "win":
		c.Wins++
	case "loss":
		c.Losses++
	case "billing":
		c.Imps++
		c.Spend += price / 1000
	}
}

// ObservePrediction adds the predicted click rate of one shown impression.
func (s *Stats) ObservePrediction(campaignID int, p float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if campaignID >= 0 {
		s.campaign(campaignID).PredClicks += p
	}
}

// ObserveClick counts a click. Duplicates are counted apart.
func (s *Stats) ObserveClick(campaignID int, dup bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if dup {
		s.dupClicks++
		return
	}
	s.notices["click"]++
	if campaignID >= 0 {
		s.campaign(campaignID).Clicks++
	}
}

// ObservePostback counts installs from MMP postbacks. Other events, such
// as purchases, are logged but not counted here yet.
func (s *Stats) ObservePostback(event, touch string, campaignID int, dup bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if dup {
		s.dupPostbacks++
		return
	}
	s.notices["postback"]++
	if event != "install" || campaignID < 0 {
		return
	}
	c := s.campaign(campaignID)
	if touch == "view" {
		c.ViewInstalls++
	} else {
		c.Installs++
	}
}

func (s *Stats) groupFor(m map[string]*group, k string) *group {
	g, ok := m[k]
	if !ok {
		g = &group{Floors: make([]int64, len(FloorEdges))}
		m[k] = g
	}
	return g
}

// GroupView is one exchange or placement in a snapshot.
type GroupView struct {
	Name          string  `json:"name"`
	Requests      int64   `json:"requests"`
	Share         float64 `json:"share"`
	MeanFloor     float64 `json:"mean_floor"`
	ValidIFAShare float64 `json:"valid_ifa_share"`
	FloorHist     []int64 `json:"floor_hist"`
}

// Snapshot is the JSON the console polls.
type Snapshot struct {
	Total        int64                 `json:"total"`
	WallRPS      float64               `json:"wall_rps"`
	UptimeS      float64               `json:"uptime_s"`
	SimTimeMS    int64                 `json:"sim_time_ms"`
	SimDay       int64                 `json:"sim_day"`
	Hourly       []int64               `json:"hourly"`
	Actions      map[string]int64      `json:"actions"`
	Reasons      map[string]int64      `json:"reasons"`
	Exchanges    []GroupView           `json:"exchanges"`
	Placements   []GroupView           `json:"placements"`
	Countries    map[string]int64      `json:"countries"`
	FloorEdges   []float64             `json:"floor_edges"`
	LatencyP50US int64                 `json:"latency_p50_us"`
	LatencyP99US int64                 `json:"latency_p99_us"`
	Campaigns    map[int]CampaignStats `json:"campaigns"`
	Notices      map[string]int64      `json:"notices"`
	DupNotices   int64                 `json:"duplicate_notices"`
	DupClicks    int64                 `json:"duplicate_clicks"`
	DupPostbacks int64                 `json:"duplicate_postbacks"`
	// BidsByVersion is bids by the model version that priced them, for
	// each of the last 48 simulated hours (Chapter 11).
	BidsByVersion []VersionHour `json:"bids_by_version"`
}

func (s *Stats) Snapshot() Snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now().Unix()
	var recent int64
	for k, v := range s.wallSec {
		if k >= now-5 && k < now {
			recent += v
		}
	}
	edges := make([]float64, len(FloorEdges)-1) // JSON has no +Inf
	copy(edges, FloorEdges)
	snap := Snapshot{
		Total: s.total, WallRPS: float64(recent) / 5, UptimeS: time.Since(s.start).Seconds(),
		SimTimeMS: s.simNow, SimDay: s.day, Hourly: append([]int64(nil), s.hourly[:]...),
		Actions: copyMap(s.actions), Reasons: copyMap(s.reasons), Countries: copyMap(s.country),
		Exchanges: s.views(s.exchange), Placements: s.views(s.placement), FloorEdges: edges,
		LatencyP50US: s.quantile(0.5), LatencyP99US: s.quantile(0.99),
		Campaigns: map[int]CampaignStats{}, Notices: copyMap(s.notices), DupNotices: s.dups,
		DupClicks: s.dupClicks, DupPostbacks: s.dupPostbacks,
	}
	for id, c := range s.camp {
		cc := *c
		if c.Filtered != nil {
			cc.Filtered = make(map[string]int64, len(c.Filtered))
			for k, v := range c.Filtered {
				cc.Filtered[k] = v
			}
		}
		snap.Campaigns[id] = cc
	}
	for h, m := range s.versions {
		snap.BidsByVersion = append(snap.BidsByVersion, VersionHour{Hour: h, Bids: copyMap(m)})
	}
	sort.Slice(snap.BidsByVersion, func(i, j int) bool { return snap.BidsByVersion[i].Hour < snap.BidsByVersion[j].Hour })
	return snap
}

// VersionHour is one simulated hour of bids by model version.
type VersionHour struct {
	Hour int64            `json:"hour"` // hours since simulated time 0
	Bids map[string]int64 `json:"bids"`
}

func (s *Stats) views(m map[string]*group) []GroupView {
	out := []GroupView{}
	for k, g := range m {
		v := GroupView{Name: k, Requests: g.Requests, FloorHist: append([]int64(nil), g.Floors...)}
		if s.total > 0 {
			v.Share = float64(g.Requests) / float64(s.total)
		}
		if g.Requests > 0 {
			v.MeanFloor = g.FloorSum / float64(g.Requests)
			v.ValidIFAShare = float64(g.ValidIFAs) / float64(g.Requests)
		}
		out = append(out, v)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Requests > out[j].Requests })
	return out
}

// quantile returns the upper edge of the latency bucket holding quantile q.
func (s *Stats) quantile(q float64) int64 {
	var n int64
	for _, c := range s.latency {
		n += c
	}
	if n == 0 {
		return 0
	}
	target, seen := int64(math.Ceil(q*float64(n))), int64(0)
	for i, c := range s.latency {
		seen += c
		if seen >= target {
			return bucketMid(i)
		}
	}
	return 0
}

func copyMap(m map[string]int64) map[string]int64 {
	out := make(map[string]int64, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}
